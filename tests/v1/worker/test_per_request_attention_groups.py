# SPDX-License-Identifier: Apache-2.0
"""H2O per-request attention: the per-sequence causal flag reaches the FULL-attention layers only.

A request with ``vllm_xargs attention_mode=bidirectional`` reads its full-attention layers in both directions; its
sliding-window (or chunked-local) layers stay causal, the way the two-way reading is trained (Gemma-4: 50 sliding
layers with a causal window, 10 full layers). Both GPU runners build the attention metadata per KV-cache / attention
group, and used to hand the same per-sequence tensor to every group, so a flagged sequence got a two-sided window on
the sliding layers. Now a windowed group gets plain ``True``, a full-attention group the tensor, and an unflagged
batch is unchanged (``True`` everywhere). CPU only: fake metadata builders record what they were given.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    is_windowed_attention_spec,
)
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

_ATTN = dict(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16)
FULL = FullAttentionSpec(**_ATTN)
SLIDING = SlidingWindowSpec(**_ATTN, sliding_window=128)
PROMOTED_SLIDING = FullAttentionSpec(**_ATTN, sliding_window=128)      # hybrid KV cache manager off
CHUNKED = ChunkedLocalAttentionSpec(**_ATTN, attention_chunk_size=256)
PROMOTED_CHUNKED = FullAttentionSpec(**_ATTN, attention_chunk_size=256)
ENCODER_ONLY = EncoderOnlyAttentionSpec(**_ATTN)
MAMBA = MambaSpec(shapes=((2, 4),), dtypes=(torch.float32,), block_size=16)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("spec", "windowed"),
    [
        (FULL, False),
        (SLIDING, True),
        (PROMOTED_SLIDING, True),
        (CHUNKED, True),
        (PROMOTED_CHUNKED, True),
        (ENCODER_ONLY, False),
        (MAMBA, False),
    ],
)
def test_is_windowed_attention_spec(spec, windowed):
    assert is_windowed_attention_spec(spec) is windowed


@pytest.mark.cpu_test
def test_is_windowed_attention_spec_needs_the_layer_spec():
    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"full.0": FULL, "swa.0": PROMOTED_SLIDING})
    with pytest.raises(ValueError, match="layer's own spec"):
        is_windowed_attention_spec(uniform)


class RecordingBuilder:
    """A metadata builder that returns the causal flag it was handed."""

    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        return SimpleNamespace(causal=common_attn_metadata.causal)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)


def _group(spec, layer_names, gid):
    return AttentionGroup(backend=object, layer_names=list(layer_names), kv_cache_spec=spec, kv_cache_group_id=gid,
                          metadata_builders=[RecordingBuilder()])


def _is_the_tensor(value, tensor):
    return isinstance(value, torch.Tensor) and value is tensor


# Gemma-4-like layouts: full layers and sliding layers in two KV-cache groups (the hybrid manager), or in ONE group
# of uniform type with the sliding spec promoted (the hybrid manager off), where the attention groups still split
# by the layers' own specs.
HYBRID = ([KVCacheGroupSpec(["full.0", "full.1"], FULL), KVCacheGroupSpec(["swa.0", "swa.1"], SLIDING)],
          [[("full", FULL, ["full.0", "full.1"])], [("swa", SLIDING, ["swa.0", "swa.1"])]])
UNIFIED = ([KVCacheGroupSpec(["full.0", "swa.0"], UniformTypeKVCacheSpecs(
               block_size=16, kv_cache_specs={"full.0": FULL, "swa.0": PROMOTED_SLIDING}))],
           [[("full", FULL, ["full.0"]), ("swa", PROMOTED_SLIDING, ["swa.0"])]])


def _attn_groups(layout):
    return [[_group(spec, names, gid) for _kind, spec, names in groups] for gid, groups in enumerate(layout)]


# ---------------------------------------------------------------------------------- the V2 runner (vllm/v1/worker/gpu)
def _v2_build(kv_cache_groups, attn_groups, causal):
    num_reqs, num_tokens = 2, 8
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    return build_attn_metadata(
        attn_groups=attn_groups, num_reqs=num_reqs, num_tokens=num_tokens, query_start_loc_gpu=qsl,
        query_start_loc_cpu=qsl, max_query_len=4, seq_lens=torch.tensor([4, 4], dtype=torch.int32), max_seq_len=4,
        block_tables=[torch.zeros(num_reqs, 1, dtype=torch.int32) for _ in kv_cache_groups],
        slot_mappings=torch.zeros(len(kv_cache_groups), num_tokens, dtype=torch.int64),
        kv_cache_config=KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups),
        causal=causal,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", [HYBRID, UNIFIED], ids=["hybrid-manager", "one-unified-group"])
def test_v2_flagged_batch_full_groups_get_the_tensor_sliding_groups_stay_causal(layout):
    kv_cache_groups, groups = layout
    flags = torch.tensor([0, 1], dtype=torch.int32)        # request 0 bidirectional, request 1 causal
    md = _v2_build(kv_cache_groups, _attn_groups(groups), flags)
    for name in ("full.0", "full.1"):
        if name in md:
            assert _is_the_tensor(md[name].causal, flags), name
    for name in ("swa.0", "swa.1"):
        if name in md:
            assert md[name].causal is True, name


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", [HYBRID, UNIFIED], ids=["hybrid-manager", "one-unified-group"])
def test_v2_unflagged_batch_is_unchanged(layout):
    kv_cache_groups, groups = layout
    md = _v2_build(kv_cache_groups, _attn_groups(groups), True)
    assert all(md[name].causal is True for name in md)


@pytest.mark.cpu_test
def test_v2_per_group_bool_mapping_of_the_hybrid_drafters_is_untouched():
    kv_cache_groups, groups = HYBRID
    md = _v2_build(kv_cache_groups, _attn_groups(groups), {0: False})
    assert md["full.0"].causal is False and md["swa.0"].causal is True


# ------------------------------------------------------------------- the V1 runner (vllm/v1/worker/gpu_model_runner)
def _v1_runner(kv_cache_groups, attn_groups, modes):
    """A bare GPUModelRunner on the CPU with just what _build_attention_metadata reads; `modes` = the attention
    mode of requests "a" and "b", both prompts of 4 tokens, prefilled in this step."""
    max_num_reqs = 4
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.kv_cache_config = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups)
    runner.attn_groups = attn_groups
    runner.optimistic_seq_lens_cpu = torch.tensor([4, 4, 0, 0], dtype=torch.int32)
    block_table = SimpleNamespace(get_device_tensor=lambda n: torch.zeros(n, 2, dtype=torch.int32))
    runner.input_batch = SimpleNamespace(
        block_table=[block_table] * len(kv_cache_groups),
        num_computed_tokens_cpu_tensor=torch.zeros(max_num_reqs, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.full((max_num_reqs,), 4, dtype=torch.int32),
        req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1})
    runner.routed_experts_initialized = False
    runner.is_mm_prefix_lm = False
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.cache_config = SimpleNamespace(use_replayssm=False, kv_sharing_fast_prefill=False)
    runner.requests = {rid: SimpleNamespace(attention_mode=mode) for rid, mode in zip(("a", "b"), modes)}
    runner.per_seq_causal_cpu = torch.ones(max_num_reqs + 1, dtype=torch.int32)
    runner.per_seq_causal_gpu = torch.ones(max_num_reqs + 1, dtype=torch.int32)
    runner.batch_has_bidirectional = False
    qsl = torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32)
    runner.query_start_loc = SimpleNamespace(gpu=qsl, cpu=qsl.clone())
    runner.seq_lens = torch.tensor([4, 4, 0, 0], dtype=torch.int32)
    runner.positions = torch.arange(16, dtype=torch.int64)
    runner.dcp_world_size = 1
    runner.speculative_config = None
    return runner


def _v1_build(runner, num_groups):
    md, _spec = runner._build_attention_metadata(
        num_tokens=8, num_reqs=2, max_query_len=4,
        slot_mappings={gid: torch.zeros(8, dtype=torch.int64) for gid in range(num_groups)})
    return md


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", [HYBRID, UNIFIED], ids=["hybrid-manager", "one-unified-group"])
def test_v1_flagged_batch_full_groups_get_the_tensor_sliding_groups_stay_causal(layout):
    kv_cache_groups, groups = layout
    runner = _v1_runner(kv_cache_groups, _attn_groups(groups), ("bidirectional", "causal"))
    md = _v1_build(runner, len(kv_cache_groups))
    assert runner.batch_has_bidirectional is True
    flags = runner.per_seq_causal_gpu[:2]
    assert flags.tolist() == [0, 1]
    for name in ("full.0", "full.1"):
        if name in md:
            assert _is_the_tensor(md[name].causal, flags) or torch.equal(md[name].causal, flags), name
    for name in ("swa.0", "swa.1"):
        if name in md:
            assert md[name].causal is True, name


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", [HYBRID, UNIFIED], ids=["hybrid-manager", "one-unified-group"])
def test_v1_unflagged_batch_is_unchanged(layout):
    kv_cache_groups, groups = layout
    runner = _v1_runner(kv_cache_groups, _attn_groups(groups), ("causal", "causal"))
    md = _v1_build(runner, len(kv_cache_groups))
    assert runner.batch_has_bidirectional is False
    assert all(md[name].causal is True for name in md)
