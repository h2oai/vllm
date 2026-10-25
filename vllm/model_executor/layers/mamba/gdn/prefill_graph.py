# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the GDN core for prefill-only steps.

The GDN core (conv1d + gated delta rule + gated norm) is a splitting op of the
piecewise CUDA graphs, so every prefill step runs it eagerly: per layer about
18 kernel launches plus the Python around them. For short prompts the GPU work
is tens of microseconds per layer, so the step is bound by that host work
(H2O-Lightning-4B on an H100 at ~120 tokens: ~7 ms of kernels in a ~28 ms
step).

For a step whose GDN batch is N prefills and nothing else (no decodes, no
speculative tokens; N <= VLLM_GDN_PREFILL_GRAPH_MAX_SEQS) this module replays a
CUDA graph of the layer's core instead. The graph is captured per (layer, N,
bucket) for a token bucket B >= the step's T tokens, on static buffers:

* the inputs are padded from T to B rows, but the conv1d and the chunk kernels
  read the sequence boundaries from the device-side cu_seqlens: they compute
  the same values as the eager call (bit-exact; measured), and leave the pad
  rows alone. The grids are sized for the worst case of N sequences in B rows:
  the conv1d program map is padded with PAD_SLOT_ID programs (which return at
  once), and the chunk index list with chunks past the last sequence's end
  (fully masked). As a second guard the pad rows also get zero conv output and
  a = b = -inf (g = 0, beta = 0: identity steps of the recurrence).
* the sequence count is rounded up to 1, 2, 4, 8 or 16 (fewer graphs to
  capture under load): the extra sequences are empty (cu_seqlens repeats the
  end), start from no state and point at the null state block, which no
  request owns (the Triton/FLA kernels write a zero final state there).
* FlashInfer's SM90 kernel picks a chunk-parallel split on the host from the
  row count. A graph is used only when the split for B equals the split the
  eager call would pick for T (then it is bit-exact too); else the 64-row
  bucket is tried, else the step runs eagerly.
* the per-step values (cu_seqlens, state slots, has_initial_state, the chunk
  and conv program maps) are copied into the static metadata once per step;
  per layer the eager work is two input copies, the replay and one output copy.
* FlashInfer keeps its scratch buffers in a module-level cache and replaces a
  buffer when a larger one is needed, which would free memory a captured graph
  still points at. Each graph therefore holds references to the library
  buffers that existed when it was captured.

With VLLM_GDN_PREFILL_GRAPH_PRECAPTURE=1 every (sequence count, bucket) shape
of a layer is captured at that layer's first eligible prefill (a few seconds
in all, which a server's warm-up absorbs) instead of when a step first meets
it, so no request under load pays a capture.

Everything else (decodes, MTP, ROCm/XPU/CPU, steps above the largest bucket or
with more prefills) takes the normal eager path.
VLLM_GDN_PREFILL_GRAPH_MAX_TOKENS sets the largest bucket (0 disables).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

# 64 * 1.5^k-ish buckets: at most ~1.5x padding of the GDN rows (the GDN
# kernels are a small part of a short prefill, so padding is cheap there).
_BUCKETS = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


# graph sequence counts: a step's prefill count is rounded up to one of these
_SEQ_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


# the conv1d kernel's BLOCK_M (tokens per program), as in
# compute_causal_conv1d_metadata
_CONV_BLOCK_M = 8


@dataclass
class _Bucket:
    num_seqs: int
    size: int
    metadata: object  # GDNAttentionMetadata with static tensors
    query_start_loc: torch.Tensor  # [N + 1] int32 cu_seqlens of the real rows
    state_slot: torch.Tensor  # [N] int32
    has_initial_state: torch.Tensor  # [N] bool
    chunk_indices: torch.Tensor  # [max chunks, 2] int32
    chunk_offsets: torch.Tensor  # [N + 1] int32
    conv_batch_ptr: torch.Tensor  # [max conv programs] int32
    conv_chunk_offset_ptr: torch.Tensor  # [max conv programs] int32
    row_index: torch.Tensor  # arange(size) int32
    pad_rows: torch.Tensor  # [size] bool, recomputed inside each graph
    inputs: dict[tuple, torch.Tensor]  # (index, shape tail, dtype) -> buffer
    outputs: dict[tuple, torch.Tensor]


class GDNPrefillGraphs:
    def __init__(
        self,
        max_tokens: int | None = None,
        max_seqs: int | None = None,
        precapture: bool | None = None,
    ) -> None:
        self.max_tokens = (
            envs.VLLM_GDN_PREFILL_GRAPH_MAX_TOKENS if max_tokens is None else max_tokens
        )
        self.max_seqs = (
            envs.VLLM_GDN_PREFILL_GRAPH_MAX_SEQS if max_seqs is None else max_seqs
        )
        self.buckets: dict[tuple[int, int], _Bucket] = {}
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        # library scratch buffers each graph was captured against (kept alive)
        self._graph_buffers: dict[tuple, list[torch.Tensor]] = {}
        self.pool = None
        self._stream: torch.cuda.Stream | None = None
        self._fi_sizes: dict[tuple[int, int, int], int | None] = {}
        self._step_metadata = None  # the step's metadata object, compared by `is`
        self._step_bucket: _Bucket | None = None
        self.disabled_reason: str | None = None
        self._shapes_seen: set[tuple[int, int]] = set()
        self.precapture = (
            envs.VLLM_GDN_PREFILL_GRAPH_PRECAPTURE if precapture is None else precapture
        )
        self._precaptured: set[tuple[str, str]] = set()
        # static input/output rows per token bucket, shared by its sequence
        # counts (one bucket runs at a time)
        self._io: dict[int, tuple[dict, dict]] = {}

    # ------------------------------------------------------------------
    def bucket_size(self, num_tokens: int) -> int | None:
        for b in _BUCKETS:
            if b > self.max_tokens:
                return None
            if num_tokens <= b:
                return b
        return None

    def seq_bucket(self, num_seqs: int) -> int | None:
        for n in _SEQ_BUCKETS:
            if n > self.max_seqs:
                return None
            if num_seqs <= n:
                return n
        return None

    def graph_size(self, layer, num_tokens: int, num_seqs: int = 1) -> int | None:
        """The token bucket to replay for N sequences of T tokens in all, or
        None (run eagerly)."""
        size = self.bucket_size(num_tokens)
        graph_seqs = self.seq_bucket(num_seqs)
        if size is None or graph_seqs is None:
            return None
        if layer.gdn_prefill_backend != "flashinfer":
            return size
        heads = layer.num_v_heads // layer.tp_size
        key = (heads, num_tokens, num_seqs)
        if key not in self._fi_sizes:
            want = _flashinfer_plan(num_tokens, heads, num_seqs)
            choice = None
            if want is not None:
                for b in (size, -(-num_tokens // 64) * 64):
                    if (
                        b <= self.max_tokens
                        and _flashinfer_plan(b, heads, graph_seqs) == want
                    ):
                        choice = b
                        break
            self._fi_sizes[key] = choice
        return self._fi_sizes[key]

    def eligible(self, layer, md) -> bool:
        return (
            md is not None
            and md.num_prefills >= 1
            and self.seq_bucket(md.num_prefills) is not None
            and md.num_decodes == 0
            and md.spec_sequence_masks is None
            and md.num_spec_decodes == 0
            and md.graph_pad_rows is None
            and md.prefill_state_indices is not None
            and md.prefill_has_initial_state is not None
            and md.prefill_query_start_loc_cpu is not None
            and md.chunk_offsets is not None
            and md.batch_ptr is not None
            and md.token_chunk_offset_ptr is not None
            and layer.gdn_prefill_backend in ("triton", "flashinfer")
        )

    # ------------------------------------------------------------------
    def run(
        self,
        layer,
        body: Callable[..., None],
        inputs: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> bool:
        """Run ``body(*inputs, out, attn_metadata=...)`` for this layer from a
        CUDA graph. False (nothing done) when the step is not eligible."""
        if self.max_tokens <= 0 or self.disabled_reason is not None:
            return False
        md = layer._layer_attn_metadata()
        if not self.eligible(layer, md):
            return False
        if not out.is_cuda or torch.cuda.is_current_stream_capturing():
            return False
        body_key = (layer.prefix, getattr(body, "__name__", ""))
        if self.precapture and body_key not in self._precaptured:
            self._precaptured.add(body_key)
            if not self._precapture_layer(layer, body, inputs, out):
                return False
            # the synthetic loads overwrote the static metadata
            self._step_metadata = None
        num_tokens = md.num_actual_tokens
        size = self.graph_size(layer, num_tokens, md.num_prefills)
        if size is None:
            return False
        num_seqs = self.seq_bucket(md.num_prefills)
        assert num_seqs is not None

        if md is not self._step_metadata:
            bucket = self._bucket(num_seqs, size, out.device)
            if not self._load_step(bucket, md):
                return False
            self._step_metadata = md
            self._step_bucket = bucket
        bucket = self._step_bucket
        assert bucket is not None
        assert bucket.size == size and bucket.num_seqs == num_seqs

        static_in = []
        for i, x in enumerate(inputs):
            buf = self._buffer(bucket.inputs, (i,), size, x)
            buf[:num_tokens].copy_(x[:num_tokens])
            static_in.append(buf)
        static_out = self._buffer(bucket.outputs, (0,), size, out)

        key = body_key + (num_seqs, size)
        graph = self.graphs.get(key)
        if graph is None:
            graph = self._capture_and_store(
                key, layer, body, bucket, static_in, static_out
            )
            if graph is None:
                return False
        graph.replay()
        out[:num_tokens].copy_(static_out[:num_tokens])
        return True

    def _capture_and_store(self, key, layer, body, bucket, static_in, static_out):
        num_seqs, size = bucket.num_seqs, bucket.size
        try:
            graph = self._capture(layer, body, bucket, static_in, static_out)
        except Exception:
            # never fail the step: fall back to the eager path for good
            self.disabled_reason = "capture failed"
            logger.warning(
                "GDN prefill CUDA graph capture failed for %s (%d sequences, "
                "bucket %d); GDN prefills run eagerly from now on.",
                layer.prefix,
                num_seqs,
                size,
                exc_info=True,
            )
            return None
        self.graphs[key] = graph
        self._graph_buffers[key] = _library_scratch_buffers()
        if (num_seqs, size) not in self._shapes_seen:
            self._shapes_seen.add((num_seqs, size))
            if not self.precapture:  # pre-capture logs once per layer instead
                logger.info(
                    "GDN prefill CUDA graphs: first capture for %d sequences, "
                    "bucket %d (%d shapes, %d graphs so far)",
                    num_seqs,
                    size,
                    len(self._shapes_seen),
                    len(self.graphs),
                )
        return graph

    def _precapture_layer(self, layer, body, inputs, out) -> bool:
        """Capture every (sequence count, bucket) shape of this layer now, on
        synthetic metadata (one sequence filling the bucket, the null state
        block as every slot); a real step only reloads the static metadata."""
        t0 = time.perf_counter()
        body_key = (layer.prefix, getattr(body, "__name__", ""))
        count = 0
        for n in _SEQ_BUCKETS:
            if n > self.max_seqs:
                break
            for size in _BUCKETS:
                if size > self.max_tokens:
                    break
                key = body_key + (n, size)
                if key in self.graphs:
                    continue
                bucket = self._bucket(n, size, out.device)
                self._load_synthetic(bucket)
                static_in = [
                    self._buffer(bucket.inputs, (i,), size, x)
                    for i, x in enumerate(inputs)
                ]
                static_out = self._buffer(bucket.outputs, (0,), size, out)
                if (
                    self._capture_and_store(
                        key, layer, body, bucket, static_in, static_out
                    )
                    is None
                ):
                    return False
                count += 1
        if count:
            logger.info(
                "GDN prefill CUDA graphs: pre-captured %d shapes for %s in %.1f s "
                "(%d graphs in all)",
                count,
                layer.prefix,
                time.perf_counter() - t0,
                len(self.graphs),
            )
        return True

    def _load_synthetic(self, bucket: _Bucket) -> None:
        """Metadata of one sequence filling the bucket (the rest empty), on the
        null state block, for a capture outside a real step."""
        n, size = bucket.num_seqs, bucket.size
        max_chunks, max_programs = self._bounds(n, size)
        bucket.query_start_loc.fill_(size)
        bucket.query_start_loc[0] = 0
        bucket.state_slot.fill_(NULL_BLOCK_ID)
        bucket.has_initial_state.fill_(False)
        bucket.chunk_indices[:, 0] = 0
        bucket.chunk_indices[:, 1] = torch.arange(
            max_chunks, device=bucket.row_index.device
        )
        bucket.chunk_offsets.fill_(max_chunks)
        bucket.chunk_offsets[0] = 0
        bucket.conv_batch_ptr.fill_(0)
        bucket.conv_chunk_offset_ptr.copy_(
            torch.arange(max_programs, device=bucket.row_index.device)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _buffer(store: dict, key: tuple, size: int, like: torch.Tensor):
        k = key + (tuple(like.shape[1:]), like.dtype)
        buf = store.get(k)
        if buf is None:
            # zeros: the pad rows must hold finite values from the start
            buf = torch.zeros(
                (size, *like.shape[1:]), dtype=like.dtype, device=like.device
            )
            store[k] = buf
        return buf

    @staticmethod
    def _bounds(num_seqs: int, size: int) -> tuple[int, int]:
        """Worst-case FLA chunk count and conv1d program count for N sequences
        of B rows in all (each sequence adds at most one partial chunk)."""
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        return -(-size // FLA_CHUNK_SIZE) + num_seqs, -(
            -size // _CONV_BLOCK_M
        ) + num_seqs

    def _load_step(self, bucket: _Bucket, md) -> bool:
        """Copy this step's metadata into the bucket's static tensors."""
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
        )
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        n = bucket.num_seqs  # graph sequences; the first m are real
        m = md.num_prefills
        cu = md.non_spec_query_start_loc
        bucket.query_start_loc[: m + 1].copy_(cu[: m + 1])
        bucket.state_slot[:m].copy_(md.prefill_state_indices[:m])
        bucket.has_initial_state[:m].copy_(md.prefill_has_initial_state[:m])
        if m < n:
            # empty sequences: no rows, no state in, the null block's slot out
            bucket.query_start_loc[m + 1 :] = cu[m]
            bucket.state_slot[m:] = NULL_BLOCK_ID
            bucket.has_initial_state[m:] = False
        if n == 1:
            # one sequence: the static chunk list and conv program map of the
            # whole bucket serve every length (see _bucket)
            return True
        max_chunks, max_programs = self._bounds(n, bucket.size)
        if md.batch_ptr.numel() < max_programs:
            return False
        cu_cpu = md.prefill_query_start_loc_cpu
        chunks = prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE)
        num_chunks = chunks.shape[0]
        # pad the chunk list with chunks past the end of the last real
        # sequence: their loads and stores are fully masked, and their
        # per-sequence chunk numbers stay inside the (max_chunks) workspace
        last_chunks = -(-int(cu_cpu[m] - cu_cpu[m - 1]) // FLA_CHUNK_SIZE)
        tail = max_chunks - num_chunks
        padded = torch.empty((max_chunks, 2), dtype=torch.int32, pin_memory=True)
        padded[:num_chunks] = chunks
        padded[num_chunks:, 0] = m - 1
        padded[num_chunks:, 1] = torch.arange(last_chunks, last_chunks + tail)
        bucket.chunk_indices.copy_(padded, non_blocking=True)
        bucket.chunk_offsets[: m + 1].copy_(md.chunk_offsets[: m + 1])
        if m < n:
            bucket.chunk_offsets[m + 1 :] = md.chunk_offsets[m]
        # the builder's program map is PAD_SLOT_ID past its real programs
        bucket.conv_batch_ptr.copy_(md.batch_ptr[:max_programs])
        bucket.conv_chunk_offset_ptr.copy_(md.token_chunk_offset_ptr[:max_programs])
        return True

    def _bucket(self, num_seqs: int, size: int, device: torch.device) -> _Bucket:
        bucket = self.buckets.get((num_seqs, size))
        if bucket is not None:
            return bucket
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        n = num_seqs
        max_chunks, max_programs = self._bounds(n, size)
        query_start_loc = torch.zeros(n + 1, dtype=torch.int32, device=device)
        state_slot = torch.zeros(n, dtype=torch.int32, device=device)
        has_initial_state = torch.zeros(n, dtype=torch.bool, device=device)
        chunk_indices = torch.zeros((max_chunks, 2), dtype=torch.int32, device=device)
        chunk_offsets = torch.zeros(n + 1, dtype=torch.int32, device=device)
        conv_batch_ptr = torch.full(
            (max_programs,), PAD_SLOT_ID, dtype=torch.int32, device=device
        )
        conv_chunk_offset_ptr = torch.full(
            (max_programs,), PAD_SLOT_ID, dtype=torch.int32, device=device
        )
        if n == 1:
            # every chunk and conv program of the bucket belongs to sequence 0;
            # those past its end are masked, so these never change
            chunk_indices[:, 1] = torch.arange(max_chunks, device=device)
            chunk_offsets[1] = max_chunks
            conv_batch_ptr.fill_(0)
            conv_chunk_offset_ptr.copy_(torch.arange(max_programs, device=device))
        row_index = torch.arange(size, dtype=torch.int32, device=device)
        pad_rows = torch.zeros(size, dtype=torch.bool, device=device)
        # causal_conv1d_fn reads the grid size ("tot") and, when set, the
        # program map from here
        nums_dict = {
            _CONV_BLOCK_M: {
                "tot": max_programs,
                "mlist": None,
                "mlist_len": max_programs,
                "offsetlist": None,
                "batch_ptr": conv_batch_ptr,
                "token_chunk_offset_ptr": conv_chunk_offset_ptr,
            }
        }
        metadata = GDNAttentionMetadata(
            num_prefills=n,
            num_prefill_tokens=size,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=size,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_state_indices_tensor=state_slot,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            # the chunk kernels read the sequence boundaries from here too
            prefill_query_start_loc=query_start_loc,
            prefill_state_indices=state_slot,
            prefill_has_initial_state=has_initial_state,
            nums_dict=nums_dict,
            batch_ptr=conv_batch_ptr,
            token_chunk_offset_ptr=conv_chunk_offset_ptr,
            graph_pad_rows=pad_rows,
        )
        bucket = _Bucket(
            num_seqs=n,
            size=size,
            metadata=metadata,
            query_start_loc=query_start_loc,
            state_slot=state_slot,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            conv_batch_ptr=conv_batch_ptr,
            conv_chunk_offset_ptr=conv_chunk_offset_ptr,
            row_index=row_index,
            pad_rows=pad_rows,
            inputs=self._io.setdefault(size, ({}, {}))[0],
            outputs=self._io[size][1],
        )
        self.buckets[(n, size)] = bucket
        return bucket

    def _capture(self, layer, body, bucket: _Bucket, static_in, static_out):
        n = bucket.num_seqs

        def graph_body():
            torch.ge(bucket.row_index, bucket.query_start_loc[n], out=bucket.pad_rows)
            body(*static_in, static_out, attn_metadata=bucket.metadata)

        t0 = time.perf_counter()
        if self._stream is None:
            self._stream = torch.cuda.Stream()
            self.pool = torch.cuda.graph_pool_handle()
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # One eager run first (Triton JIT/autotune must not run under
            # capture). It writes this step's conv/ssm state slots, so save and
            # restore them: the replay then starts from the same state.
            conv_state, ssm_state = layer.kv_cache[0], layer.kv_cache[1]
            slots = bucket.state_slot.long()
            saved = (conv_state[slots].clone(), ssm_state[slots].clone())
            graph_body()
            conv_state[slots] = saved[0]
            ssm_state[slots] = saved[1]
            # capture_begin/end rather than torch.cuda.graph(): no gc.collect()
            # or empty_cache() per graph at serving time, and thread-local
            # capture, so other threads' CUDA calls are not refused meanwhile.
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                graph_body()
            finally:
                graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)
        logger.debug(
            "Captured a GDN prefill CUDA graph for %s, %d sequences, bucket %d, "
            "in %.1f ms",
            layer.prefix,
            n,
            bucket.size,
            1000 * (time.perf_counter() - t0),
        )
        return graph


def _flashinfer_plan(num_rows: int, num_heads: int, num_seqs: int = 1):
    """The (use CP, CP chunk length) FlashInfer's GDN prefill picks for
    num_seqs sequences in num_rows rows on this device, as in its own dispatch
    (which passes the row count as max_seqlen); None if the helpers are not
    importable (then FlashInfer prefills run eagerly)."""
    try:
        from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
            choose_cp_chunk_len_host,
            should_use_cp_host,
        )
        from flashinfer.utils import (
            get_compute_capability,
            get_device_name,
            get_device_sm_count,
        )
    except Exception:
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    capability = get_compute_capability(device)
    sms = get_device_sm_count(device)
    name = get_device_name(device)
    if not (
        capability[0] in (9, 10, 12)
        and should_use_cp_host(
            num_seqs * num_heads, sms, name, device_capability=capability
        )
    ):
        return ("no-cp",)
    return (
        "cp",
        choose_cp_chunk_len_host(
            num_rows,
            num_heads,
            sms,
            device_capability=capability,
            total_seqlen=num_rows,
            num_seqs=num_seqs,
            device_name=name,
        ),
    )


def _library_scratch_buffers() -> list[torch.Tensor]:
    try:
        from flashinfer.utils import _cache_buf
    except Exception:
        return []
    return list(_cache_buf.values())


gdn_prefill_graphs = GDNPrefillGraphs()
