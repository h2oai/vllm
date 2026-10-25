# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.metrics.stats import PrefillStats, RequestSpecDecodeMetrics
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)

BIDIRECTIONAL_NEEDS_ONE_TOKEN = (
    "attention_mode 'bidirectional' needs max_tokens=1: the readout is the "
    "prefill's next-token distribution; no decoding on a bidirectional prefix"
)


def parse_attention_mode(sampling_params: SamplingParams | None) -> str:
    """H2O per-request attention: the mode a request asks for through
    ``extra_args`` (``vllm_xargs``) ``attention_mode``: "causal" (the default,
    alias one_way) or "bidirectional" (alias two_way). Anything else is a
    ValueError."""
    if sampling_params is None or not sampling_params.extra_args:
        return "causal"
    mode = sampling_params.extra_args.get("attention_mode")
    if mode is None:
        return "causal"
    mode = str(mode).strip().lower().replace("-", "_")
    if mode in ("one_way", "oneway"):
        return "causal"
    if mode in ("two_way", "twoway"):
        return "bidirectional"
    if mode not in ("causal", "bidirectional"):
        raise ValueError(
            f"attention_mode must be 'causal' or 'bidirectional', got {mode!r}"
        )
    return mode


def parse_prefix_cache_write(sampling_params: SamplingParams | None) -> bool:
    """H2O per-request prefix-cache writes: ``extra_args`` (``vllm_xargs``)
    ``prefix_cache_write`` false keeps a causal request out of the prefix cache
    as a writer. It still reads the blocks other requests cached, but commits
    none of its own, so with Mamba "align" caching its prefill is not split at
    the last state-block boundary (the split exists only to land a cacheable
    state there) and runs in one step. Default true (stock behaviour). A bool,
    or "true"/"false"/"1"/"0"; anything else is a ValueError."""
    if sampling_params is None or not sampling_params.extra_args:
        return True
    value = sampling_params.extra_args.get("prefix_cache_write")
    if value is None or isinstance(value, bool):
        return value is not False
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"prefix_cache_write must be true or false, got {value!r}")


def _declared_head_sizes(model_config: Any) -> list[int]:
    """Every attention head size the model declares, from its architecture
    config: the whole-model head size, or each layer's own where layers differ
    (Gemma-4: 256 on the sliding layers, 512 on the full-attention layers);
    sorted, without duplicates."""
    arch = getattr(model_config, "model_arch_config", None)
    if arch is None:
        return [int(model_config.get_head_size())]
    if getattr(arch, "per_layer_overrides", None):
        return sorted(
            {int(arch[i].head_size) for i in range(arch.total_num_hidden_layers)}
        )
    return [int(arch.head_size)]


def _fa4_version_reason(model_config: Any, resolve: Any) -> str | None:
    """Why a layer of this model would not run FlashAttention 4: ``resolve``
    (get_flash_attn_version, evaluated under the engine's config) gives the
    version the backend picks for a head size, with ALiBi and softcapping
    taken into account; the first head size that does not resolve to 4 is the
    reason, since such a layer would raise on the per-sequence flag."""
    text_config = getattr(model_config, "hf_text_config", None)
    requires_alibi = bool(getattr(model_config, "uses_alibi", False))
    requires_softcap = bool(getattr(text_config, "attn_logit_softcapping", None))
    for head_size in _declared_head_sizes(model_config):
        version = resolve(
            requires_alibi=requires_alibi,
            head_size=head_size,
            requires_softcap=requires_softcap,
            supports_fa4_hd256=True,
        )
        if version != 4:
            return (
                f"FlashAttention {version or 'no version'} resolves for head size "
                f"{head_size}"
            )
    return None


def _fa4_device_reason(platform: Any, batch_invariant: bool) -> str | None:
    """Why this device cannot run FlashAttention 4's per-sequence causal flag:
    CUDA only, not under VLLM_BATCH_INVARIANT (which selects version 2), and
    Hopper only: the vendored kernel applies dynamic_causal on compute
    capability 9.x alone (the Blackwell kernels accept the argument and ignore
    it); None when it can."""
    if not platform.is_cuda():
        return "FlashAttention 4 runs on CUDA only"
    if batch_invariant:
        return "VLLM_BATCH_INVARIANT selects FlashAttention 2"
    if not platform.is_device_capability_family(90):
        return (
            "FlashAttention 4 applies a per-sequence causal flag on Hopper "
            "(compute capability 9.x) only"
        )
    return None


def _fa4_platform_reason(vllm_config: Any) -> str | None:
    """Why FlashAttention 4 cannot serve the per-sequence flag on this engine:
    the device (_fa4_device_reason) and the version the backend resolves for
    every head size the model declares; None when it can. Tests replace this
    function."""
    import vllm.envs as envs
    from vllm.platforms import current_platform

    reason = _fa4_device_reason(current_platform, envs.VLLM_BATCH_INVARIANT)
    if reason is not None:
        return reason
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    with set_current_vllm_config(vllm_config):
        return _fa4_version_reason(vllm_config.model_config, get_flash_attn_version)


def flash_attn_per_sequence_support(vllm_config: Any) -> str | None:
    """Why the FLASH_ATTN backend of this engine cannot apply a per-sequence
    causal flag, or None when it can. FlashAttention 4 applies the flag through
    dynamic_causal and FA2/FA3 raise on it, so the version must be set to 4 and
    resolve to 4 for this model. A FA4 call that carries the flag drops the
    mm_prefix and R-SWA masks for every sequence in the batch, so models that
    use them are refused (Gemma-4's "vision" mode clears the mm_prefix ranges
    on its full-attention layers and is fine). Decode context parallelism has
    no per-sequence path."""
    if getattr(vllm_config.attention_config, "flash_attn_version", None) != 4:
        return "attention_config.flash_attn_version must be set to 4"
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if getattr(parallel_config, "decode_context_parallel_size", 1) > 1:
        return "decode context parallelism has no per-sequence path"
    model_config = vllm_config.model_config
    if getattr(model_config, "rswa_window", None) is not None:
        return "R-SWA layers: their mask is dropped beside a per-sequence flag"
    if getattr(model_config, "is_mm_prefix_lm", False):
        text_config = getattr(model_config, "hf_text_config", None)
        if getattr(text_config, "use_bidirectional_attention", None) != "vision":
            return (
                "mm_prefix ranges on full-attention layers: their mask is dropped "
                "beside a per-sequence flag"
            )
    return _fa4_platform_reason(vllm_config)


def check_bidirectional_request(
    sampling_params: SamplingParams, vllm_config: Any
) -> None:
    """Refuse, with a clear error, a bidirectional request this engine cannot
    serve as asked: it needs max_tokens=1; a backend that applies a per-sequence
    causal flag, set explicitly (TRITON_ATTN, or FLASH_ATTN where
    flash_attn_per_sequence_support allows it; another backend would read the
    prompt causally and say nothing); and no KV connector, since its KV is not
    causal and must never be stored or loaded under token hashes."""
    if sampling_params.max_tokens != 1:
        raise ValueError(BIDIRECTIONAL_NEEDS_ONE_TOKEN)
    model_config = vllm_config.model_config
    text_config = getattr(model_config, "hf_text_config", None)
    head_size = int(model_config.get_head_size())
    value_sizes = [
        getattr(text_config, name, None) for name in ("v_head_dim", "v_channels")
    ]
    if (
        any(v is not None and int(v) != head_size for v in value_sizes)
        or (getattr(text_config, "param_sink_number", 0) or 0) > 0
    ):
        # such models (a value head size of their own, or parametric attention
        # sinks) choose their own DiffKV attention backend, which has no
        # per-sequence path
        raise ValueError(
            "attention_mode 'bidirectional' is not served for a model whose value "
            "head size differs from its key head size or that uses parametric "
            "attention sinks: its attention backend applies no per-sequence "
            "causal flag"
        )
    backend = vllm_config.attention_config.backend
    name = getattr(backend, "name", None)
    per_kind = getattr(vllm_config.attention_config, "backend_per_kind", None) or {}
    kinds = {kind: getattr(b, "name", str(b)) for kind, b in per_kind.items()}
    fa_reason = (
        flash_attn_per_sequence_support(vllm_config)
        if name == "FLASH_ATTN" or "FLASH_ATTN" in kinds.values()
        else None
    )
    accepted = {"TRITON_ATTN"} | ({"FLASH_ATTN"} if fa_reason is None else set())
    if name not in accepted:
        raise ValueError(
            "attention_mode 'bidirectional' needs --attention-backend TRITON_ATTN, "
            "or FLASH_ATTN with attention_config.flash_attn_version=4 on a Hopper "
            "GPU (compute capability 9.x) where FlashAttention 4 resolves for every "
            f"head size of this model (got {name or 'auto'}"
            f"{'; ' + fa_reason if name == 'FLASH_ATTN' and fa_reason else ''}): "
            "only those apply a per-sequence causal flag"
        )
    elsewhere = {kind: b for kind, b in kinds.items() if b not in accepted}
    if elsewhere:
        raise ValueError(
            "attention_mode 'bidirectional' needs a backend that applies the "
            "per-sequence causal flag for every attention kind; backend_per_kind "
            f"routes {elsewhere} to another backend"
            f"{'; ' + fa_reason if fa_reason else ''}"
        )
    if vllm_config.kv_transfer_config is not None:
        raise ValueError(
            "attention_mode 'bidirectional' cannot be served with a KV connector: "
            "its KV is not causal and must never be stored or loaded through one"
        )


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        session_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        abort_immediately: bool = False,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
            self.structured_output_request.reasoning_parser_kwargs = (
                reasoning_parser_kwargs
            )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None
        # E/P/D: Connector-specific encoder-cache transfer parameters.
        self.ec_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                self.ec_transfer_params = sampling_params.extra_args.get(
                    "ec_transfer_params"
                )
                self.kv_cache_report_mode = sampling_params.extra_args.get(
                    "kv_cache_report_mode", "incremental"
                )
            else:
                self.kv_cache_report_mode = "incremental"
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Per-position mask used in mixed-mode (chat completion with
        # prompt_embeds). `None` except when both `prompt_token_ids` and
        # `prompt_embeds` are set and their positions are interleaved.
        self.prompt_is_token_ids = prompt_is_token_ids
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        if self.prompt_token_ids is None:
            self._all_token_ids: list[int] = [0] * self.num_prompt_tokens
        elif self.prompt_is_token_ids is None:
            self._all_token_ids = self.prompt_token_ids.copy()
        else:
            # Mixed-mode prompt: positions covered by prompt_embeds hold a sentinel
            # special token id that may lie outside the embedding. Zero them, matching
            # the no-token-ids case above, so embedding gathers over these placeholder
            # ids stay in bounds; the actual inputs come from prompt_embeds.
            self._all_token_ids = [
                t if is_tok else 0
                for t, is_tok in zip(self.prompt_token_ids, self.prompt_is_token_ids)
            ]

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Tokens of output in flight when the request was preempted: delivered
        # on return, but must not mutate the reset counters.
        self.num_stale_output_tokens = 0
        # Drop the stale output instead, for same-step preempt + resume
        # (reset_prefix_cache).
        self.drop_stale_output = False

        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        self.num_in_flight_tokens = 0

        # V2+PP+async: Enforces `pp_size` cadence between same-request decode steps
        # so the worker's broadcast slot ring stays consistent.
        self.next_decode_eligible_step = 0

        # Seq of the most recent step this request was scheduled in; fences
        # deferred block freeing (see Scheduler._free_request_blocks).
        self.last_sched_seq = 0

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        self.session_id = session_id

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # Block-aligned token position of a proven shared prefix worth pinning
        # in the (sparse) prefix cache; 0 means none. Set at admission for
        # hybrid/Mamba models when a shared prefix is detected (Marconi-style).
        self.shared_prefix_boundary = 0

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        self.prefill_stats: PrefillStats | None = PrefillStats()

        # Per-request speculative-decoding acceptance accumulator. Populated by
        # the scheduler when --per-request-spec-decode-metrics is set (eagerly on
        # add_request, then observed each verify step); stays None otherwise.
        self.spec_decode_metrics: RequestSpecDecodeMetrics | None = None

        # Per-request attention mode (H2O): "causal" (default) or "bidirectional".
        # A bidirectional request has its full-attention layers read the whole
        # prompt in both directions (per-sequence causal flag in the attention
        # metadata); it is prefill-only (max_tokens == 1: the next-token logits
        # of the prefill are the readout), is never chunked, and never shares KV
        # blocks with the prefix cache in either direction (its KV is not causal).
        self.attention_mode = parse_attention_mode(sampling_params)
        if self.attention_mode == "bidirectional":
            logger.debug(
                "H2O per-request attention: request %s bidirectional "
                "(prefill-only, no prefix cache)",
                request_id,
            )
            if self.max_tokens != 1:
                raise ValueError(BIDIRECTIONAL_NEEDS_ONE_TOKEN)
            block_hasher = None  # no prefix-cache reads or writes for this request
        # H2O per-request prefix-cache writes: False = read cached blocks, commit
        # none (and no Mamba block-aligned prefill split for this request)
        self.prefix_cache_write = parse_prefix_cache_write(sampling_params)
        # H2O early exit (vllm/h2o_exits.py): a request with an exit policy is
        # prefill-only and never shares KV blocks through the prefix cache (an
        # exited pass leaves the layers above the exit unwritten).
        from vllm.h2o_exits import parse_exit_policy

        self.exit_policy = parse_exit_policy(sampling_params)
        if self.exit_policy is not None:
            if self.max_tokens != 1:
                raise ValueError("exit_policy needs max_tokens=1 (a prefill readout)")
            block_hasher = None
        # steps this request was deferred for lack of room (bounded aging in the
        # scheduler)
        self.bidirectional_deferrals = 0

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

        # If True, request should be aborted immediately after being added to
        # the scheduler so the connector's request_finished hook runs.
        self.abort_immediately = abort_immediately

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            session_id=request.session_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def take_prefill_stats(self) -> PrefillStats | None:
        if self.prefill_stats is None:
            return None
        prefill_stats = self.prefill_stats
        self.prefill_stats = None
        return prefill_stats

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
