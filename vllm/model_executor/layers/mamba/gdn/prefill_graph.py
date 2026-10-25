# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the GDN core for single-sequence prefills.

The GDN core (conv1d + gated delta rule + gated norm) is a splitting op of the
piecewise CUDA graphs, so every prefill step runs it eagerly: per layer about
18 kernel launches plus the Python around them. For short prompts the GPU work
is tens of microseconds per layer, so the step is bound by that host work
(H2O-Lightning-4B on an H100 at ~120 tokens: ~7 ms of kernels in a ~28 ms
step).

For a step whose GDN batch is exactly one prefill (no decodes, no speculative
tokens) this module replays a CUDA graph of the layer's core instead. The graph
is captured per (layer, bucket) for a token bucket B >= the prompt's T tokens,
on static buffers:

* the inputs are padded from T to B rows, but the conv1d and the chunk kernels
  read the sequence end T from the device-side cu_seqlens [0, T]: they compute
  the same values as the eager call for T tokens (bit-exact for the Triton/FLA
  kernels; measured), and leave the pad rows alone. The grid and the chunk
  metadata are sized for B. As a second guard the pad rows also get zero conv
  output and a = b = -inf (g = 0, beta = 0: identity steps of the recurrence).
* FlashInfer's SM90 kernel picks a chunk-parallel split on the host from the
  row count. A graph is used only when the split for B equals the split the
  eager call would pick for T (then it is bit-exact too); else the 64-row
  bucket is tried, else the step runs eagerly.
* the per-step values (T, the state slot, has_initial_state) are copied into
  the static metadata once per step; per layer the eager work is two input
  copies, the replay and one output copy.
* FlashInfer keeps its scratch buffers in a module-level cache and replaces a
  buffer when a larger one is needed, which would free memory a captured graph
  still points at. Each graph therefore holds references to the library
  buffers that existed when it was captured.

Everything else (multi-sequence batches, decodes, MTP, ROCm/XPU/CPU, prompts
above the largest bucket) takes the normal eager path.
VLLM_GDN_PREFILL_GRAPH_MAX_TOKENS sets the largest bucket (0 disables).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# 64 * 1.5^k-ish buckets: at most ~1.5x padding of the GDN rows (the GDN
# kernels are a small part of a short prefill, so padding is cheap there).
_BUCKETS = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def _max_tokens_from_env() -> int:
    return envs.VLLM_GDN_PREFILL_GRAPH_MAX_TOKENS


@dataclass
class _Bucket:
    size: int
    metadata: object  # GDNAttentionMetadata with static tensors
    query_start_loc: torch.Tensor  # [0, T] int32, read by the conv1d
    state_slot: torch.Tensor  # [1] int32
    has_initial_state: torch.Tensor  # [1] bool
    row_index: torch.Tensor  # arange(size) int32
    pad_rows: torch.Tensor  # [size] bool, recomputed inside each graph
    inputs: dict[tuple, torch.Tensor]  # (index, shape tail, dtype) -> buffer
    outputs: dict[tuple, torch.Tensor]


class GDNPrefillGraphs:
    def __init__(self, max_tokens: int | None = None) -> None:
        self.max_tokens = _max_tokens_from_env() if max_tokens is None else max_tokens
        self.buckets: dict[int, _Bucket] = {}
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        # library scratch buffers each graph was captured against (kept alive)
        self._graph_buffers: dict[tuple, list[torch.Tensor]] = {}
        self.pool = None
        self._stream: torch.cuda.Stream | None = None
        self._fi_sizes: dict[tuple[int, int], int | None] = {}
        self._step_metadata = None  # the step's metadata object, compared by `is`
        self._step_bucket: _Bucket | None = None
        self.disabled_reason: str | None = None

    # ------------------------------------------------------------------
    def bucket_size(self, num_tokens: int) -> int | None:
        for b in _BUCKETS:
            if b > self.max_tokens:
                return None
            if num_tokens <= b:
                return b
        return None

    def graph_size(self, layer, num_tokens: int) -> int | None:
        """The bucket to replay for T tokens, or None (run eagerly)."""
        size = self.bucket_size(num_tokens)
        if size is None or layer.gdn_prefill_backend != "flashinfer":
            return size
        heads = layer.num_v_heads // layer.tp_size
        key = (heads, num_tokens)
        if key not in self._fi_sizes:
            want = _flashinfer_plan(num_tokens, heads)
            choice = None
            if want is not None:
                for b in (size, -(-num_tokens // 64) * 64):
                    if b <= self.max_tokens and _flashinfer_plan(b, heads) == want:
                        choice = b
                        break
            self._fi_sizes[key] = choice
        return self._fi_sizes[key]

    @staticmethod
    def eligible(layer, md) -> bool:
        return (
            md is not None
            and md.num_prefills == 1
            and md.num_decodes == 0
            and md.spec_sequence_masks is None
            and md.num_spec_decodes == 0
            and md.graph_pad_rows is None
            and md.prefill_state_indices is not None
            and md.prefill_has_initial_state is not None
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
        num_tokens = md.num_actual_tokens
        size = self.graph_size(layer, num_tokens)
        if size is None:
            return False

        if md is not self._step_metadata:
            bucket = self._bucket(size, md, out.device)
            # one prefill, no decodes: non_spec_query_start_loc is [0, T]
            bucket.query_start_loc.copy_(md.non_spec_query_start_loc[:2])
            bucket.state_slot.copy_(md.prefill_state_indices[:1])
            bucket.has_initial_state.copy_(md.prefill_has_initial_state[:1])
            self._step_metadata = md
            self._step_bucket = bucket
        bucket = self._step_bucket
        assert bucket is not None and bucket.size == size

        static_in = []
        for i, x in enumerate(inputs):
            buf = self._buffer(bucket.inputs, (i,), size, x)
            buf[:num_tokens].copy_(x[:num_tokens])
            static_in.append(buf)
        static_out = self._buffer(bucket.outputs, (0,), size, out)

        key = (layer.prefix, getattr(body, "__name__", ""), size)
        graph = self.graphs.get(key)
        if graph is None:
            try:
                graph = self._capture(layer, body, bucket, static_in, static_out)
            except Exception:
                # never fail the step: fall back to the eager path for good
                self.disabled_reason = "capture failed"
                logger.warning(
                    "GDN prefill CUDA graph capture failed for %s (bucket %d); "
                    "GDN prefills run eagerly from now on.",
                    layer.prefix,
                    size,
                    exc_info=True,
                )
                return False
            self.graphs[key] = graph
            self._graph_buffers[key] = _library_scratch_buffers()
        graph.replay()
        out[:num_tokens].copy_(static_out[:num_tokens])
        return True

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

    def _bucket(self, size: int, md, device: torch.device) -> _Bucket:
        bucket = self.buckets.get(size)
        if bucket is not None:
            return bucket
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

        cu_cpu = torch.tensor([0, size], dtype=torch.int32)
        query_start_loc = torch.tensor([0, size], dtype=torch.int32, device=device)
        state_slot = torch.zeros(1, dtype=torch.int32, device=device)
        has_initial_state = torch.zeros(1, dtype=torch.bool, device=device)
        row_index = torch.arange(size, dtype=torch.int32, device=device)
        pad_rows = torch.zeros(size, dtype=torch.bool, device=device)
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            cu_cpu, device=device
        )
        metadata = GDNAttentionMetadata(
            num_prefills=1,
            num_prefill_tokens=size,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=size,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_state_indices_tensor=state_slot,
            chunk_indices=prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE).to(device),
            chunk_offsets=prepare_chunk_offsets(cu_cpu, FLA_CHUNK_SIZE).to(device),
            # the chunk kernels read the sequence end T from here too
            prefill_query_start_loc=query_start_loc,
            prefill_state_indices=state_slot,
            prefill_has_initial_state=has_initial_state,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            graph_pad_rows=pad_rows,
        )
        bucket = _Bucket(
            size=size,
            metadata=metadata,
            query_start_loc=query_start_loc,
            state_slot=state_slot,
            has_initial_state=has_initial_state,
            row_index=row_index,
            pad_rows=pad_rows,
            inputs={},
            outputs={},
        )
        self.buckets[size] = bucket
        return bucket

    def _capture(self, layer, body, bucket: _Bucket, static_in, static_out):
        def graph_body():
            torch.ge(bucket.row_index, bucket.query_start_loc[1], out=bucket.pad_rows)
            body(*static_in, static_out, attn_metadata=bucket.metadata)

        t0 = time.perf_counter()
        if self._stream is None:
            self._stream = torch.cuda.Stream()
            self.pool = torch.cuda.graph_pool_handle()
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # One eager run first (Triton JIT/autotune must not run under
            # capture). It writes this step's conv/ssm state slot, so save and
            # restore it: the replay then starts from the same state.
            conv_state, ssm_state = layer.kv_cache[0], layer.kv_cache[1]
            slot = bucket.state_slot.long()
            saved = (conv_state[slot].clone(), ssm_state[slot].clone())
            graph_body()
            conv_state[slot] = saved[0]
            ssm_state[slot] = saved[1]
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
            "Captured a GDN prefill CUDA graph for %s, bucket %d, in %.1f ms",
            layer.prefix,
            bucket.size,
            1000 * (time.perf_counter() - t0),
        )
        return graph


def _flashinfer_plan(num_rows: int, num_heads: int):
    """The (use CP, CP chunk length) FlashInfer's GDN prefill picks for one
    sequence of num_rows rows on this device, as in its own dispatch; None if
    the helpers are not importable (then FlashInfer prefills run eagerly)."""
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
        and should_use_cp_host(num_heads, sms, name, device_capability=capability)
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
            num_seqs=1,
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
