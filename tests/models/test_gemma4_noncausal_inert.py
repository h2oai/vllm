# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-regression (CPU): with no non-causal declaration the patched Gemma4 code
builds exactly the stock layer objects.

Every Gemma-4 load goes through the patched ``gemma4.py`` (the multimodal
wrapper resolves its language model through the registry by name), so the
inert path must be the stock path. This builds ``Gemma4Attention`` for every
layer of an E4B-shaped config (42 layers, every sixth full, the last 18
KV-shared) and a 31B-shaped one with the attention classes replaced by
recorders, and asserts: the plain ``Attention`` class everywhere, attention type
DECODER everywhere, the stock KV-sharing target names, the stock sliding window
and mm_prefix clamp per layer, no non-causal state, and no extra forward kwargs.
With the declaration the same build switches exactly the full-attention layers.
"""

from contextlib import contextmanager

import pytest
import torch
from torch import nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models import gemma4
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.attention.backend import AttentionType

E4B = dict(num_hidden_layers=42, num_kv_shared_layers=18, sliding_window=512)
G31 = dict(num_hidden_layers=60, num_kv_shared_layers=0, sliding_window=1024)


def _config(shape, **extra) -> Gemma4TextConfig:
    n = shape["num_hidden_layers"]
    return Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(n)],
        max_position_embeddings=256,
        **shape,
        **extra,
    )


class _Recorder(nn.Module):
    """Stands in for Attention / EncoderOnlyAttention and records the call."""

    cls_name = "Attention"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args, self.kwargs = args, dict(kwargs)
        self.layer_name = kwargs.get("prefix", "")


class _EncoderRecorder(_Recorder):
    cls_name = "EncoderOnlyAttention"


class _PassThrough(nn.Module):
    def forward(self, x):
        return x, None


@contextmanager
def _cpu_world():
    update_environment_variables(
        {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1",
         "MASTER_PORT": str(get_open_port())}
    )
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1)
        try:
            yield
        finally:
            cleanup_dist_env_and_memory()


def _build(config, monkeypatch):
    monkeypatch.setattr(gemma4, "Attention", _Recorder)
    monkeypatch.setattr(gemma4, "EncoderOnlyAttention", _EncoderRecorder)
    layers = []
    with _cpu_world():
        for i in range(config.num_hidden_layers):
            layers.append(
                gemma4.Gemma4Attention(
                    config,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.per_layer_config[i].head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    prefix=f"model.layers.{i}.self_attn",
                )
            )
    return layers


def _stock_kv_target(config, i):
    num_shared = config.num_kv_shared_layers
    first = config.num_hidden_layers - num_shared
    if not (num_shared > 0 and i >= first):
        return None
    prev = config.layer_types[:first]
    j = len(prev) - 1 - prev[::-1].index(config.layer_types[i])
    return f"model.layers.{j}.self_attn.attn"


@pytest.mark.cpu_test
@pytest.mark.parametrize("shape", [E4B, G31], ids=["e4b", "31b"])
def test_no_declaration_builds_the_stock_layers(shape, monkeypatch):
    config = _config(shape)
    assert not hasattr(config, "noncausal_full_attention")
    layers = _build(config, monkeypatch)
    for i, layer in enumerate(layers):
        is_sliding = config.layer_types[i] == "sliding_attention"
        assert type(layer.attn) is _Recorder, i                       # plain Attention everywhere
        kw = layer.attn.kwargs
        assert kw["attn_type"] == AttentionType.DECODER, i
        assert kw["per_layer_sliding_window"] == (config.sliding_window if is_sliding else None), i
        assert kw["mm_prefix_clamp_sliding_window"] == is_sliding, i
        assert kw["kv_sharing_target_layer_name"] == _stock_kv_target(config, i), i
        assert layer.noncausal is False and layer.noncausal_kv_source is None, i
        assert layer.is_kv_shared_layer == (_stock_kv_target(config, i) is not None), i


@pytest.mark.cpu_test
def test_declaration_switches_exactly_the_full_layers(monkeypatch):
    config = _config(E4B, is_causal=False)
    layers = _build(config, monkeypatch)
    for i, layer in enumerate(layers):
        full = config.layer_types[i] == "full_attention"
        assert type(layer.attn) is (_EncoderRecorder if full else _Recorder), i
        assert layer.attn.kwargs["attn_type"] == (
            AttentionType.ENCODER_ONLY if full else AttentionType.DECODER
        ), i
        stock_target = _stock_kv_target(config, i)
        if full and stock_target is not None:
            # a non-causal KV-shared layer: no cache sharing, the source kept for the forward
            assert layer.attn.kwargs["kv_sharing_target_layer_name"] is None
            assert layer.noncausal_kv_source == stock_target
        else:
            assert layer.attn.kwargs["kv_sharing_target_layer_name"] == stock_target
        assert layer.attn.kwargs["mm_prefix_clamp_sliding_window"] == (not full), i


@pytest.mark.cpu_test
def test_inert_forward_takes_the_stock_calls(monkeypatch):
    """Without the declaration the forward never looks at a shared-KV dict and
    calls the attention exactly as the stock code does (q, k, v) / (q, None, None)."""
    config = _config(E4B)
    layers = _build(config, monkeypatch)
    calls = []

    def fake_attn(self, q, k, v):
        calls.append((k is None, v is None))
        return torch.zeros(q.shape[0], self.kwargs.get("num_kv_heads", 1) * 0 + q.shape[1])

    monkeypatch.setattr(_Recorder, "forward", fake_attn, raising=False)
    for i in (0, 23, 29):                                  # sliding, full unshared, full shared
        layer = layers[i]
        x = torch.zeros(3, config.hidden_size)
        positions = torch.arange(3)
        layer.o_proj = _PassThrough()           # RowParallelLinear returns (output, bias)
        layer.forward(positions, x)
    assert calls == [(False, False), (False, False), (True, True)]
