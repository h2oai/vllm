# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H2O per-request attention: the per-sequence causal flag reaches the full-attention
layers only.

A request with ``vllm_xargs attention_mode=bidirectional`` reads its full-attention
layers in both directions; its sliding-window (or chunked-local) layers stay causal,
the way the two-way reading is trained (Gemma-4: 50 sliding layers with a causal
window, 10 full layers). Both GPU runners build the attention metadata per KV-cache /
attention group and used to hand the same per-sequence tensor to every group, so a
flagged sequence got a two-sided window on the sliding layers. Now a windowed layer
gets plain ``True``, a full-attention layer the tensor, and an unflagged batch is
unchanged (``True`` everywhere). CPU only: fake metadata builders record what they
were given.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
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
from vllm.v1.worker.utils import AttentionGroup, windowed_attention_layer_names

_ATTN = dict(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16)
FULL = FullAttentionSpec(**_ATTN)
SLIDING = SlidingWindowSpec(**_ATTN, sliding_window=128)
PROMOTED_SLIDING = FullAttentionSpec(
    **_ATTN, sliding_window=128
)  # hybrid KV cache manager off
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
    uniform = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"full.0": FULL, "swa.0": PROMOTED_SLIDING}
    )
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
    return AttentionGroup(
        backend=object,
        layer_names=list(layer_names),
        kv_cache_spec=spec,
        kv_cache_group_id=gid,
        metadata_builders=[RecordingBuilder()],
    )


def _is_the_tensor(value, tensor):
    return isinstance(value, torch.Tensor) and value is tensor


# Gemma-4-like layouts: full layers and sliding layers in two KV-cache groups (the
# hybrid manager), or in ONE group
# of uniform type with the sliding spec promoted (the hybrid manager off), where the
# attention groups still split
# by the layers' own specs.
HYBRID = (
    [
        KVCacheGroupSpec(["full.0", "full.1"], FULL),
        KVCacheGroupSpec(["swa.0", "swa.1"], SLIDING),
    ],
    [[("full", FULL, ["full.0", "full.1"])], [("swa", SLIDING, ["swa.0", "swa.1"])]],
)
UNIFIED = (
    [
        KVCacheGroupSpec(
            ["full.0", "swa.0"],
            UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={"full.0": FULL, "swa.0": PROMOTED_SLIDING},
            ),
        )
    ],
    [[("full", FULL, ["full.0"]), ("swa", PROMOTED_SLIDING, ["swa.0"])]],
)
# the hybrid manager off with EQUAL page sizes: every layer in ONE plain merged
# FullAttentionSpec that carries the
# sliding window, so the group's spec says "windowed" for the full-attention layers too;
# only the layers know
MERGED = (
    [KVCacheGroupSpec(["full.0", "swa.0", "full.1"], PROMOTED_SLIDING)],
    [[("mixed", PROMOTED_SLIDING, ["full.0", "swa.0", "full.1"])]],
)
WINDOWED_LAYERS = {
    "swa.0",
    "swa.1",
}  # what windowed_attention_layer_names returns for these layouts
LAYOUTS = [HYBRID, UNIFIED, MERGED]
LAYOUT_IDS = ["hybrid-manager", "one-unified-group", "one-merged-group"]


def _attn_groups(layout):
    return [
        [_group(spec, names, gid) for _kind, spec, names in groups]
        for gid, groups in enumerate(layout)
    ]


# ---------------------------------------------------------------------------------- the
# V2 runner (vllm/v1/worker/gpu)
def _v2_build(
    kv_cache_groups,
    attn_groups,
    causal,
    windowed_layers=WINDOWED_LAYERS,
    for_cudagraph_capture=False,
):
    num_reqs, num_tokens = 2, 8
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    return build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=qsl,
        query_start_loc_cpu=qsl,
        max_query_len=4,
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        max_seq_len=4,
        block_tables=[
            torch.zeros(num_reqs, 1, dtype=torch.int32) for _ in kv_cache_groups
        ],
        slot_mappings=torch.zeros(len(kv_cache_groups), num_tokens, dtype=torch.int64),
        kv_cache_config=KVCacheConfig(
            num_blocks=4, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
        ),
        causal=causal,
        windowed_layers=windowed_layers,
        for_cudagraph_capture=for_cudagraph_capture,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_v2_flagged_batch_full_groups_get_the_tensor_sliding_groups_stay_causal(layout):
    kv_cache_groups, groups = layout
    flags = torch.tensor(
        [0, 1], dtype=torch.int32
    )  # request 0 bidirectional, request 1 causal
    md = _v2_build(kv_cache_groups, _attn_groups(groups), flags)
    for name in ("full.0", "full.1"):
        if name in md:
            assert _is_the_tensor(md[name].causal, flags), name
    for name in ("swa.0", "swa.1"):
        if name in md:
            assert md[name].causal is True, name


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_v2_unflagged_batch_is_unchanged(layout):
    kv_cache_groups, groups = layout
    md = _v2_build(kv_cache_groups, _attn_groups(groups), True)
    assert all(md[name].causal is True for name in md)


@pytest.mark.cpu_test
def test_v2_without_a_layer_set_the_group_spec_decides():
    """Without windowed_layers (a caller without the model) the group's own spec
    decides: right for the hybrid-manager and unified layouts, and wrong but safe (all
    causal) for a merged group."""
    kv_cache_groups, groups = HYBRID
    flags = torch.tensor([0, 1], dtype=torch.int32)
    md = _v2_build(kv_cache_groups, _attn_groups(groups), flags, windowed_layers=None)
    assert _is_the_tensor(md["full.0"].causal, flags) and md["swa.0"].causal is True
    kv_cache_groups, groups = MERGED
    md = _v2_build(kv_cache_groups, _attn_groups(groups), flags, windowed_layers=None)
    assert md["full.0"].causal is True and md["swa.0"].causal is True


@pytest.mark.cpu_test
def test_v2_cudagraph_capture_build_keeps_the_sliding_groups_causal():
    kv_cache_groups, groups = MERGED
    flags = torch.tensor([0, 1], dtype=torch.int32)
    md = _v2_build(
        kv_cache_groups, _attn_groups(groups), flags, for_cudagraph_capture=True
    )
    assert _is_the_tensor(md["full.0"].causal, flags) and md["swa.0"].causal is True


@pytest.mark.cpu_test
def test_v2_per_group_bool_mapping_of_the_hybrid_drafters_is_untouched():
    kv_cache_groups, groups = HYBRID
    md = _v2_build(kv_cache_groups, _attn_groups(groups), {0: False})
    assert md["full.0"].causal is False and md["swa.0"].causal is True


# ------------------------------------------------------------------- the V1 runner
# (vllm/v1/worker/gpu_model_runner)
def _v1_runner(kv_cache_groups, attn_groups, modes):
    """A bare GPUModelRunner on the CPU with just what _build_attention_metadata
    reads; `modes` are the attention modes of requests "a" and "b", both prompts of 4
    tokens, prefilled in this step."""
    max_num_reqs = 4
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=8, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
    )
    runner.attn_groups = attn_groups
    runner.optimistic_seq_lens_cpu = torch.tensor([4, 4, 0, 0], dtype=torch.int32)
    block_table = SimpleNamespace(
        get_device_tensor=lambda n: torch.zeros(n, 2, dtype=torch.int32)
    )
    runner.input_batch = SimpleNamespace(
        block_table=[block_table] * len(kv_cache_groups),
        num_computed_tokens_cpu_tensor=torch.zeros(max_num_reqs, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.full((max_num_reqs,), 4, dtype=torch.int32),
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
    )
    runner.routed_experts_initialized = False
    runner.is_mm_prefix_lm = False
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.cache_config = SimpleNamespace(
        use_replayssm=False, kv_sharing_fast_prefill=False
    )
    runner.requests = {
        rid: SimpleNamespace(attention_mode=mode)
        for rid, mode in zip(("a", "b"), modes)
    }
    runner.per_seq_causal_cpu = torch.ones(max_num_reqs + 1, dtype=torch.int32)
    runner.per_seq_causal_gpu = torch.ones(max_num_reqs + 1, dtype=torch.int32)
    runner.batch_has_bidirectional = False
    qsl = torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32)
    runner.query_start_loc = SimpleNamespace(gpu=qsl, cpu=qsl.clone())
    runner.seq_lens = torch.tensor([4, 4, 0, 0], dtype=torch.int32)
    runner.positions = torch.arange(16, dtype=torch.int64)
    runner.dcp_world_size = 1
    runner.speculative_config = None
    runner._windowed_attention_layers = set(
        WINDOWED_LAYERS
    )  # what the model's layers would say
    return runner


def _v1_build(runner, num_groups):
    md, _spec = runner._build_attention_metadata(
        num_tokens=8,
        num_reqs=2,
        max_query_len=4,
        slot_mappings={
            gid: torch.zeros(8, dtype=torch.int64) for gid in range(num_groups)
        },
    )
    return md


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_v1_flagged_batch_full_groups_get_the_tensor_sliding_groups_stay_causal(layout):
    kv_cache_groups, groups = layout
    runner = _v1_runner(
        kv_cache_groups, _attn_groups(groups), ("bidirectional", "causal")
    )
    md = _v1_build(runner, len(kv_cache_groups))
    assert runner.batch_has_bidirectional is True
    flags = runner.per_seq_causal_gpu[:2]
    assert flags.tolist() == [0, 1]
    for name in ("full.0", "full.1"):
        if name in md:
            assert _is_the_tensor(md[name].causal, flags) or torch.equal(
                md[name].causal, flags
            ), name
    for name in ("swa.0", "swa.1"):
        if name in md:
            assert md[name].causal is True, name


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_v1_unflagged_batch_is_unchanged(layout):
    kv_cache_groups, groups = layout
    runner = _v1_runner(kv_cache_groups, _attn_groups(groups), ("causal", "causal"))
    md = _v1_build(runner, len(kv_cache_groups))
    assert runner.batch_has_bidirectional is False
    assert all(md[name].causal is True for name in md)


# ----------------------------------------------- windowed_attention_layer_names
class _FakeLayer(AttentionLayerBase):
    """An attention-like layer with the two things the helper reads."""

    def __init__(self, spec, sliding_window=None):
        self._spec = spec
        self.sliding_window = sliding_window

    def get_attn_backend(self):
        return object

    def get_kv_cache_spec(self, vllm_config):
        return self._spec


@pytest.mark.cpu_test
def test_windowed_attention_layer_names_from_the_layers():
    """Decided per layer from its own spec, never from a group: sliding, chunked-local
    and promoted specs are windowed; a layer without a spec of its own (KV-shared,
    encoder-only) is windowed when its module carries a sliding window; full, plain
    encoder-only and Mamba layers are not."""
    layers = {
        "full.0": _FakeLayer(FULL),
        "swa.0": _FakeLayer(SLIDING),
        "chunk.0": _FakeLayer(CHUNKED),
        "promoted.0": _FakeLayer(PROMOTED_SLIDING),
        "shared_swa.0": _FakeLayer(None, sliding_window=128),
        "enc.0": _FakeLayer(None),
        "mamba.0": _FakeLayer(MAMBA),
    }
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers)
    )
    assert windowed_attention_layer_names(vllm_config) == {
        "swa.0",
        "chunk.0",
        "promoted.0",
        "shared_swa.0",
    }


class CachingBuilder(RecordingBuilder):
    """A builder whose builds are reused through update_block_table (as
    FlashAttention's are)."""

    supports_update_block_table = True

    def update_block_table(self, cached, block_table, slot_mapping):
        return cached


@pytest.mark.cpu_test
@pytest.mark.parametrize("windowed_first", [False, True])
def test_v1_metadata_cache_never_crosses_windowed_and_full_groups(windowed_first):
    """Two attention groups of ONE merged spec and one builder type (layers split by
    head count or backend): the V1 build cache, reused through update_block_table,
    must not hand the full group's tensor build to the windowed group, nor the
    reverse, whichever is built first."""
    kv_cache_groups = [KVCacheGroupSpec(["full.0", "swa.0"], PROMOTED_SLIDING)]
    full = AttentionGroup(
        backend=object,
        layer_names=["full.0"],
        kv_cache_spec=PROMOTED_SLIDING,
        kv_cache_group_id=0,
        metadata_builders=[CachingBuilder()],
    )
    swa = AttentionGroup(
        backend=object,
        layer_names=["swa.0"],
        kv_cache_spec=PROMOTED_SLIDING,
        kv_cache_group_id=0,
        metadata_builders=[CachingBuilder()],
    )
    attn_groups = [[swa, full] if windowed_first else [full, swa]]
    runner = _v1_runner(kv_cache_groups, attn_groups, ("bidirectional", "causal"))
    md = _v1_build(runner, 1)
    assert torch.equal(md["full.0"].causal, runner.per_seq_causal_gpu[:2])
    assert md["swa.0"].causal is True
