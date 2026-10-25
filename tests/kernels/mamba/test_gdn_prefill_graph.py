# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN prefill CUDA graph (gdn/prefill_graph.py) against the eager path.

A prefill-only step (one or several sequences, T tokens in all) runs once
through the real eager ``_forward_core`` and once through
``gdn_prefill_graphs.run``, which replays a graph captured for N sequences and
the bucket B >= T. The core-attention output of the T real
rows, the conv-state pool and the ssm-state pool must be bit-identical (the
graph's kernels read T from the device). Both a fresh prefill and one that
continues from a cached state are checked, the second replay of a captured
graph is checked against a fresh eager run (new inputs, new state slot), and so
is the eager fallback for a batch that is not eligible.

The CPU-only tests cover bucket selection and the eligibility rule.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.prefill_graph import GDNPrefillGraphs


def test_bucket_size() -> None:
    g = GDNPrefillGraphs(max_tokens=2048)
    assert g.bucket_size(1) == 64
    assert g.bucket_size(64) == 64
    assert g.bucket_size(65) == 96
    assert g.bucket_size(117) == 128
    assert g.bucket_size(2048) == 2048
    assert g.bucket_size(2049) is None
    assert GDNPrefillGraphs(max_tokens=100).bucket_size(97) is None
    assert GDNPrefillGraphs(max_tokens=0).bucket_size(10) is None


def _md(**kw):
    base = dict(
        num_prefills=1,
        num_decodes=0,
        spec_sequence_masks=None,
        num_spec_decodes=0,
        graph_pad_rows=None,
        prefill_state_indices=object(),
        prefill_has_initial_state=object(),
        prefill_query_start_loc_cpu=object(),
        chunk_offsets=object(),
        batch_ptr=object(),
        token_chunk_offset_ptr=object(),
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize(
    "kw,ok",
    [
        ({}, True),
        ({"num_prefills": 2}, True),
        ({"num_prefills": 4}, True),
        ({"num_prefills": 5}, False),
        ({"prefill_query_start_loc_cpu": None}, False),
        ({"batch_ptr": None}, False),
        ({"num_decodes": 1}, False),
        ({"num_spec_decodes": 1}, False),
        ({"spec_sequence_masks": object()}, False),
        ({"graph_pad_rows": object()}, False),
        ({"prefill_state_indices": None}, False),
    ],
)
def test_eligible(kw, ok) -> None:
    g = GDNPrefillGraphs(max_tokens=2048, max_seqs=4)
    layer = types.SimpleNamespace(gdn_prefill_backend="triton")
    assert g.eligible(layer, _md(**kw)) is ok
    assert g.eligible(layer, None) is False
    layer.gdn_prefill_backend = "cutedsl"
    assert g.eligible(layer, _md()) is False


def test_flashinfer_size_follows_plan(monkeypatch) -> None:
    from vllm.model_executor.layers.mamba.gdn import prefill_graph

    # a CP split that changes at 600 rows: T=514 must not use the 768 bucket
    monkeypatch.setattr(
        prefill_graph,
        "_flashinfer_plan",
        lambda rows, heads, seqs=1: ("cp", 192 if rows < 600 else 256),
    )
    layer = types.SimpleNamespace(
        gdn_prefill_backend="flashinfer", num_v_heads=32, tp_size=1
    )
    g = GDNPrefillGraphs(max_tokens=2048)
    assert g.bucket_size(514) == 768
    assert g.graph_size(layer, 514) == 576  # the 64-row bucket keeps the plan
    assert g.graph_size(layer, 600) == 768
    monkeypatch.setattr(
        prefill_graph, "_flashinfer_plan", lambda rows, heads, seqs=1: None
    )
    assert GDNPrefillGraphs(max_tokens=2048).graph_size(layer, 514) is None
    layer.gdn_prefill_backend = "triton"
    assert g.graph_size(layer, 514) == 768


def test_disabled_returns_false() -> None:
    g = GDNPrefillGraphs(max_tokens=0)
    assert g.run(None, None, (), None) is False


# ---------------------------------------------------------------------------
# GPU: graph vs eager on the real _forward_core
# ---------------------------------------------------------------------------
from vllm.platforms import current_platform  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and torch.cuda.is_available()
        and current_platform.has_device_capability(80)
        and not current_platform.is_device_capability_family(100)
    ),
    reason="GDN prefill graph GPU test runs the Triton/FLA or FlashInfer chunk "
    "kernels (CUDA SM80-SM90/SM120)",
)

H, HV, K, V = 4, 8, 128, 128
CONV_KERNEL = 4
KEY_DIM, VALUE_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"


def _setup(backend: str):
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.additional_config = {"gdn_prefill_backend": backend}
    device = torch.device("cuda")
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(  # type: ignore[arg-type]
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[PREFIX],
        vllm_config=cfg,
        device=device,
    )

    def make_layer(conv_state, ssm_state, params):
        layer = types.SimpleNamespace()
        layer.prefix = PREFIX
        layer.enable_packed_recurrent_decode = False
        layer.tp_size = 1
        layer.num_k_heads, layer.num_v_heads = H, HV
        layer.head_k_dim, layer.head_v_dim = K, V
        layer.key_dim, layer.value_dim = KEY_DIM, VALUE_DIM
        layer.activation = "silu"
        layer.A_log, layer.dt_bias, conv_w, conv_b = params
        layer.conv1d = types.SimpleNamespace(weight=conv_w, bias=conv_b)
        layer.kv_cache = (conv_state, ssm_state)
        with set_current_vllm_config(cfg):
            layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        layer.gdn_prefill_backend = layer.chunk_gated_delta_rule.gdn_prefill_backend
        for name in ("rearrange_mixed_qkv", "_forward_core", "_layer_attn_metadata"):
            setattr(
                layer,
                name,
                types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
            )
        return layer

    return cfg, device, builder, make_layer


def _meta(cfg, builder, device, seq_lens, query_lens):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config import set_current_vllm_config

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BLOCK_SIZE,
        device,
        arange_block_indices=True,
    )
    with set_current_vllm_config(cfg):
        return builder.build(common_prefix_len=0, common_attn_metadata=common)


def _run(layer, meta, fn, n_rows, inputs):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    out = torch.zeros(n_rows, HV, V, dtype=torch.bfloat16, device="cuda")
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        used = fn(layer, out, *[x.clone() for x in inputs])
    torch.cuda.synchronize()
    return out, used


CASES = [
    # (query lengths, cached context per sequence)
    ([117], [0]),
    ([64], [0]),
    ([300], [0]),
    ([90], [48]),
    ([40, 50], [0, 0]),
    ([117, 5, 64], [0, 32, 0]),
    ([30] * 6, [0] * 6),
    ([200, 100, 2, 7], [0, 16, 0, 64]),
]


@needs_gpu
@pytest.mark.parametrize("backend", ["triton", "flashinfer"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("query_lens,cached", CASES)
def test_prefill_graph_matches_eager(backend, state_dtype, query_lens, cached):
    if backend == "flashinfer" and not current_platform.is_device_capability(90):
        pytest.skip("FlashInfer GDN prefill runs on SM90 here")
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
    )

    torch.manual_seed(0)
    cfg, device, builder, make_layer = _setup(backend)
    graphs = GDNPrefillGraphs(max_tokens=2048, max_seqs=16)
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=0
    )
    pool = 128
    conv0 = torch.randn(pool, *conv_shape, dtype=torch.bfloat16, device=device) * 0.05
    ssm0 = torch.randn(pool, *ssm_shape, dtype=state_dtype, device=device) * 0.05
    params = (
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(HV, dtype=torch.float32, device=device) * 0.1,
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device)
        * 0.1,
        torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1,
    )

    def eager(layer, out, mixed_qkv, b, a):
        layer._forward_core(mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=out)
        return False

    def graph(layer, out, mixed_qkv, b, a):
        return graphs.run(layer, layer._forward_core, (mixed_qkv, b, a), out)

    num_tokens = sum(query_lens)
    # three steps: capture, replay with new inputs, replay with longer contexts
    for step, extra in enumerate((0, 0, 16)):
        seq_lens = [c + extra + q for c, q in zip(cached, query_lens)]
        meta = _meta(cfg, builder, device, seq_lens, query_lens)
        assert meta.num_prefills == len(query_lens) and meta.num_decodes == 0
        # vLLM pads the step's token count; the GDN core sees the padded rows
        n_rows = num_tokens + 5
        inputs = (
            torch.randn(n_rows, CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1,
            torch.randn(n_rows, HV, dtype=torch.bfloat16, device=device) * 0.1,
            torch.randn(n_rows, HV, dtype=torch.bfloat16, device=device) * 0.1,
        )
        conv_e, ssm_e = conv0.clone(), ssm0.clone()
        conv_g, ssm_g = conv0.clone(), ssm0.clone()
        out_e, _ = _run(make_layer(conv_e, ssm_e, params), meta, eager, n_rows, inputs)
        layer_g = make_layer(conv_g, ssm_g, params)
        if step > 0:
            # a replay reads the state pools the graph was captured against
            layer_g.kv_cache = prev_layer.kv_cache  # noqa: F821
            prev_layer.kv_cache[0].copy_(conv0)  # noqa: F821
            prev_layer.kv_cache[1].copy_(ssm0)  # noqa: F821
            conv_g, ssm_g = prev_layer.kv_cache  # noqa: F821
        out_g, used = _run(layer_g, meta, graph, n_rows, inputs)
        assert used, "the graph path was not taken"
        prev_layer = layer_g  # noqa: F841
        n = num_tokens
        torch.testing.assert_close(conv_g, conv_e, atol=0, rtol=0)
        torch.testing.assert_close(out_g[:n], out_e[:n], atol=0, rtol=0)
        torch.testing.assert_close(ssm_g, ssm_e, atol=0, rtol=0)
    assert len(graphs.graphs) == 1


@needs_gpu
def test_decodes_or_too_many_prefills_take_eager_path():
    torch.manual_seed(0)
    cfg, device, builder, make_layer = _setup("triton")
    for max_seqs, seq_lens, query_lens in (
        (16, [40, 50], [1, 50]),  # a decode in the step
        (2, [40, 50, 60], [40, 50, 60]),  # more prefills than max_seqs
    ):
        graphs = GDNPrefillGraphs(max_tokens=2048, max_seqs=max_seqs)
        meta = _meta(cfg, builder, device, seq_lens, query_lens)
        layer = types.SimpleNamespace(
            gdn_prefill_backend="triton",
            _layer_attn_metadata=lambda meta=meta: meta,
        )
        assert not graphs.run(layer, None, (), torch.zeros(1, device=device))
        assert not graphs.graphs
