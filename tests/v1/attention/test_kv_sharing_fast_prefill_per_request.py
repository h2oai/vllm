# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H2O per-request attention with ``--kv-sharing-fast-prefill`` (YOCO, Gemma-4
E-series: the KV-shared upper layers run on the logits rows only).

A request flagged ``vllm_xargs attention_mode=bidirectional`` on the generate
runner is prefilled whole in one step; its KV-shared layers read the KV that the
lower layers (two-way on the full-attention layers) wrote in that step, so running
them on the logits rows alone gives the same rows as the full forward. The rebuilt
metadata of those rows must keep the per-sequence causal flag: at the last prompt
token (the one-engine readout) causal and bidirectional see the same keys, but an
earlier logits row of a two-way request (several readout positions per request)
would otherwise be read one-way. The sliding-window layers keep plain ``True``.
The guard against fast prefill on a checkpoint DECLARED non-causal (the pooling
runner, no cache to share) stays. CPU only.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tests.v1.worker.test_per_request_attention_groups import (
    FULL,
    SLIDING,
    _v1_runner,
)
from vllm.model_executor.models.config import Gemma4Config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    KVSharingFastPrefillMetadata,
    create_fast_prefill_custom_backend,
    make_kv_sharing_fast_prefill_common_attn_metadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.gpu.attn_utils import FastPrefillBatchMetadata, build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup


def _common(causal, query_lens, seq_lens, logits_indices, max_logits_per_req=1):
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0)
    num_tokens = int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros(len(query_lens), 4, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
        causal=causal,
        logits_indices_padded=torch.tensor(logits_indices, dtype=torch.int64),
        num_logits_indices=len(logits_indices),
        max_logits_per_req=max_logits_per_req,
    )


@pytest.mark.cpu_test
def test_logits_rows_keep_the_per_sequence_flag():
    # request 0 two-way (5 tokens), request 1 one-way (3 tokens): one logits row
    # each, the last prompt token
    flags = torch.tensor([0, 1], dtype=torch.int32)
    md = make_kv_sharing_fast_prefill_common_attn_metadata(
        _common(flags, [5, 3], [5, 3], [4, 7])
    )
    assert md.causal is flags
    assert md.num_actual_tokens == 2
    assert md.query_start_loc.tolist() == [0, 1, 2]
    assert md.max_query_len == 1
    assert md.seq_lens.tolist() == [5, 3]


@pytest.mark.cpu_test
def test_several_readout_rows_per_request():
    # request 0 has three logits rows, request 1 one: the rebuilt query ranges
    # follow, and the two-way flag of request 0 covers all three rows
    flags = torch.tensor([0, 1], dtype=torch.int32)
    md = make_kv_sharing_fast_prefill_common_attn_metadata(
        _common(flags, [6, 4], [6, 4], [2, 4, 5, 9], max_logits_per_req=3)
    )
    assert md.causal is flags
    assert md.query_start_loc.tolist() == [0, 3, 4]
    assert md.num_actual_tokens == 4
    assert md.max_query_len == 3


@pytest.mark.cpu_test
def test_unflagged_batch_stays_causal():
    md = make_kv_sharing_fast_prefill_common_attn_metadata(
        _common(True, [5, 3], [5, 3], [4, 7])
    )
    assert md.causal is True


@dataclass
class _Recorded:
    causal: object
    num_actual_tokens: int


class _RecordingBuilder:
    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return _Recorded(
            causal=common_attn_metadata.causal,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)


class _FakeBackend:
    @staticmethod
    def get_builder_cls():
        return _RecordingBuilder


def _fast_prefill_builder():
    backend = create_fast_prefill_custom_backend("FastPrefill", _FakeBackend)
    return backend.get_builder_cls()()


@pytest.mark.cpu_test
def test_fast_prefill_builder_hands_the_flag_to_the_backend():
    flags = torch.tensor([0, 1], dtype=torch.int32)
    md = _fast_prefill_builder().build(0, _common(flags, [5, 3], [5, 3], [4, 7]))
    assert isinstance(md, KVSharingFastPrefillMetadata)
    assert md.causal is flags
    assert md.num_actual_tokens == 2
    assert md.num_logits_indices == 2


def _fp_group(spec, layer_names, gid):
    return AttentionGroup(
        backend=object,
        layer_names=list(layer_names),
        kv_cache_spec=spec,
        kv_cache_group_id=gid,
        metadata_builders=[_fast_prefill_builder()],
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("modes", [("bidirectional", "causal"), ("causal", "causal")])
def test_v1_runner_kv_shared_layers_get_the_flag_on_the_logits_rows(modes):
    """The V1 runner's metadata for the KV-shared (fast-prefill) layers of a
    Gemma-4-like model: the full-attention ones get the per-sequence tensor over
    the logits rows, the sliding ones plain True, an unflagged batch True."""
    kv_cache_groups = [
        KVCacheGroupSpec(["full.0", "full.1"], FULL),
        KVCacheGroupSpec(["swa.0", "swa.1"], SLIDING),
    ]
    attn_groups = [
        [_fp_group(FULL, ["full.0", "full.1"], 0)],
        [_fp_group(SLIDING, ["swa.0", "swa.1"], 1)],
    ]
    runner = _v1_runner(kv_cache_groups, attn_groups, modes)
    runner.cache_config = SimpleNamespace(
        use_replayssm=False, kv_sharing_fast_prefill=True
    )
    runner.kv_sharing_fast_prefill_logits_indices = torch.zeros(16, dtype=torch.int64)
    runner.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda n, invalid_modes=None: (None, SimpleNamespace(num_tokens=n))
    )
    md, _ = runner._build_attention_metadata(
        num_tokens=8,
        num_reqs=2,
        max_query_len=4,
        logits_indices=torch.tensor([3, 7], dtype=torch.int64),
        max_num_sampled_tokens=1,
        slot_mappings={gid: torch.zeros(8, dtype=torch.int64) for gid in range(2)},
    )
    for name in ("full.0", "full.1", "swa.0", "swa.1"):
        assert md[name].num_actual_tokens == 2, name
        assert md[name].logits_indices_padded.tolist() == [3, 7], name
    if modes[0] == "bidirectional":
        full = md["full.0"].causal
        assert isinstance(full, torch.Tensor)
        assert full.tolist()[:2] == [0, 1]
    else:
        assert md["full.0"].causal is True
    assert md["swa.0"].causal is True
    assert md["swa.1"].causal is True


@pytest.mark.cpu_test
@pytest.mark.parametrize("flagged", [True, False])
def test_v2_runner_kv_shared_layers_get_the_flag_on_the_logits_rows(flagged):
    """The same for the V2 runner (vllm/v1/worker/gpu), which serves Gemma-4 by
    default: build_attn_metadata with the step's fast-prefill rows."""
    kv_cache_groups = [
        KVCacheGroupSpec(["full.0", "full.1"], FULL),
        KVCacheGroupSpec(["swa.0", "swa.1"], SLIDING),
    ]
    attn_groups = [
        [_fp_group(FULL, ["full.0", "full.1"], 0)],
        [_fp_group(SLIDING, ["swa.0", "swa.1"], 1)],
    ]
    flags = torch.tensor([0, 1], dtype=torch.int32) if flagged else True
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    md = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=2,
        num_tokens=8,
        query_start_loc_gpu=qsl,
        query_start_loc_cpu=qsl,
        max_query_len=4,
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        max_seq_len=4,
        block_tables=[torch.zeros(2, 1, dtype=torch.int32) for _ in range(2)],
        slot_mappings=torch.zeros(2, 8, dtype=torch.int64),
        kv_cache_config=KVCacheConfig(
            num_blocks=4, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
        ),
        causal=flags,
        windowed_layers={"swa.0", "swa.1"},
        fast_prefill=FastPrefillBatchMetadata(
            logits_indices_padded=torch.tensor([3, 7], dtype=torch.int32),
            num_logits_indices=2,
            max_logits_per_req=1,
        ),
    )
    for name in ("full.0", "full.1", "swa.0", "swa.1"):
        assert md[name].num_actual_tokens == 2, name
    if flagged:
        assert md["full.0"].causal is flags
    else:
        assert md["full.0"].causal is True
    assert md["swa.0"].causal is True


class _ArchConfig(list):
    @property
    def total_num_hidden_layers(self):
        return len(self)


def _gemma4_vllm_config(is_causal, fast_prefill):
    # one head size everywhere: the backend selection after the guard returns
    arch = [SimpleNamespace(head_size=256) for _ in range(4)]
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            is_causal=is_causal,
            layer_types=["sliding_attention", "full_attention"] * 2,
        ),
        model_arch_config=_ArchConfig(arch),
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=fast_prefill),
    )


@pytest.mark.cpu_test
def test_guard_still_refuses_a_declared_non_causal_checkpoint():
    with pytest.raises(ValueError, match="attention_mode=bidirectional"):
        Gemma4Config.verify_and_update_config(_gemma4_vllm_config(False, True))


@pytest.mark.cpu_test
@pytest.mark.parametrize(("is_causal", "fast_prefill"), [(True, True), (False, False)])
def test_guard_passes_the_causal_checkpoint_with_fast_prefill(is_causal, fast_prefill):
    # the per-request two-way path serves the causal checkpoint: no refusal
    Gemma4Config.verify_and_update_config(_gemma4_vllm_config(is_causal, fast_prefill))
