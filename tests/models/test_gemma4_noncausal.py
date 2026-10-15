# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Gemma4 non-causal full attention.

A Gemma4 checkpoint (or ``--hf-overrides``) that declares
``noncausal_full_attention: true`` (or ``is_causal: false``) on its text config
reads the prompt in both directions on the full-attention layers only; the
sliding-window layers stay causal. ``Gemma4Config`` applies the declaration
and refuses the generate runner; ``noncausal_layer_plan`` says, per layer,
whether it is non-causal and where a KV-shared layer takes its K/V from.
Without the declaration nothing changes.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PretrainedConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from vllm.config import ModelConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Gemma4Config
from vllm.model_executor.models.gemma4 import (
    assert_noncausal_layers,
    noncausal_full_attention,
    noncausal_layer_plan,
)
from vllm.v1.attention.backend import AttentionType

# google/gemma-4-E4B-it: 42 layers, every sixth full, the last 18 KV-shared.
E4B_LAYER_TYPES = [
    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(42)
]
# google/gemma-4-31B-it: 60 layers, every sixth full, no KV sharing.
G31_LAYER_TYPES = [
    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(60)
]


def _text_config(layer_types, num_kv_shared_layers=0, **extra) -> Gemma4TextConfig:
    return Gemma4TextConfig(
        num_hidden_layers=len(layer_types),
        layer_types=list(layer_types),
        num_kv_shared_layers=num_kv_shared_layers,
        sliding_window=512,
        **extra,
    )


def _model_config(hf_config, text_config=None, runner_type="pooling") -> ModelConfig:
    """Minimal stand-in for ModelConfig; only these three fields are read."""
    return cast(
        ModelConfig,
        SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=text_config if text_config is not None else hf_config,
            runner_type=runner_type,
        ),
    )


@pytest.mark.cpu_test
def test_registered_for_every_gemma4_architecture():
    for arch in (
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
    ):
        assert MODELS_CONFIG_MAP[arch] is Gemma4Config


@pytest.mark.cpu_test
def test_absent_declaration_changes_nothing():
    """Stock behaviour: no flag, the config is left alone, causal everywhere."""
    text = _text_config(E4B_LAYER_TYPES, 18)
    Gemma4Config.verify_and_update_model_config(_model_config(text, runner_type="generate"))
    assert not hasattr(text, "is_causal") or text.is_causal is True
    assert noncausal_full_attention(text) is False
    assert not any(e["noncausal"] for e in noncausal_layer_plan(text))


@pytest.mark.cpu_test
@pytest.mark.parametrize("flag", ["noncausal_full_attention", "is_causal"])
def test_declaration_sets_is_causal_on_both_configs(flag):
    """The multimodal wrapper reads hf_config, the text stack hf_text_config."""
    text = _text_config(G31_LAYER_TYPES)
    setattr(text, flag, flag != "is_causal")
    wrapper = PretrainedConfig()
    wrapper.text_config = text
    Gemma4Config.verify_and_update_model_config(_model_config(wrapper, text))
    assert wrapper.is_causal is False
    assert text.is_causal is False
    assert noncausal_full_attention(text) is True


@pytest.mark.cpu_test
def test_declaration_refuses_the_generate_runner():
    text = _text_config(G31_LAYER_TYPES, noncausal_full_attention=True)
    with pytest.raises(ValueError, match="pooling runner"):
        Gemma4Config.verify_and_update_model_config(
            _model_config(text, runner_type="generate")
        )


@pytest.mark.cpu_test
def test_conflicting_declarations_are_refused():
    text = _text_config(G31_LAYER_TYPES, noncausal_full_attention=True)
    wrapper = PretrainedConfig(noncausal_full_attention=False)
    wrapper.text_config = text
    with pytest.raises(ValueError, match="conflicting"):
        Gemma4Config.verify_and_update_model_config(_model_config(wrapper, text))


@pytest.mark.cpu_test
def test_plan_31b_full_layers_only_no_kv_sharing():
    text = _text_config(G31_LAYER_TYPES, is_causal=False)
    plan = noncausal_layer_plan(text)
    assert len(plan) == 60
    assert [i for i, e in enumerate(plan) if e["noncausal"]] == list(range(5, 60, 6))
    assert all(e["kv_source"] is None for e in plan)
    assert all(
        not e["noncausal"] for e in plan if e["layer_type"] == "sliding_attention"
    )


@pytest.mark.cpu_test
def test_plan_e4b_kv_shared_full_layers_read_the_last_unshared_full_layer():
    """E4B: layers 24-41 are KV-shared. Full layers 29, 35, 41 take K/V from
    layer 23 (the last unshared full layer); sliding shared layers from 22."""
    text = _text_config(E4B_LAYER_TYPES, 18, is_causal=False)
    plan = noncausal_layer_plan(text)
    assert [i for i, e in enumerate(plan) if e["noncausal"]] == [5, 11, 17, 23, 29, 35, 41]
    assert all(e["kv_source"] is None for e in plan[:24])
    for i in (29, 35, 41):
        assert plan[i]["kv_source"] == 23
    for i in range(24, 42):
        if plan[i]["layer_type"] == "sliding_attention":
            assert plan[i]["kv_source"] == 22 and not plan[i]["noncausal"]


@pytest.mark.cpu_test
def test_kv_shared_noncausal_source_is_itself_noncausal():
    """A non-causal KV-shared layer must read K/V computed by a non-causal
    layer of the same kind, so the source is always a full-attention layer."""
    text = _text_config(E4B_LAYER_TYPES, 18, is_causal=False)
    plan = noncausal_layer_plan(text)
    for e in plan:
        if e["noncausal"] and e["kv_source"] is not None:
            assert plan[e["kv_source"]]["noncausal"]


def _fake_model(layer_types, attn_types):
    """A model skeleton: .model.layers[i].self_attn.attn.attn_type."""
    from types import SimpleNamespace

    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(attn_type=t)))
        for t in attn_types
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


@pytest.mark.cpu_test
def test_silent_causal_guard_passes_when_the_built_layers_match():
    text = _text_config(E4B_LAYER_TYPES, 18, is_causal=False)
    built = [
        AttentionType.ENCODER_ONLY if t == "full_attention" else AttentionType.DECODER
        for t in E4B_LAYER_TYPES
    ]
    assert assert_noncausal_layers(_fake_model(E4B_LAYER_TYPES, built), text) == 7
    stock = _text_config(E4B_LAYER_TYPES, 18)
    assert assert_noncausal_layers(_fake_model(E4B_LAYER_TYPES, [AttentionType.DECODER] * 42), stock) == 0


@pytest.mark.cpu_test
def test_silent_causal_guard_raises_on_a_causal_build_under_the_declaration():
    """The dangerous case: declared non-causal, built the stock way (plugin absent,
    stale file): must raise, never serve one-way quietly."""
    text = _text_config(E4B_LAYER_TYPES, 18, is_causal=False)
    with pytest.raises(RuntimeError, match="7 full-attention layers should be bidirectional, 0 are"):
        assert_noncausal_layers(_fake_model(E4B_LAYER_TYPES, [AttentionType.DECODER] * 42), text)
    # and a partial switch (one KV-shared full layer left causal)
    built = [
        AttentionType.ENCODER_ONLY if t == "full_attention" else AttentionType.DECODER
        for t in E4B_LAYER_TYPES
    ]
    built[41] = AttentionType.DECODER
    with pytest.raises(RuntimeError, match=r"\(41, 'full_attention', 'causal'\)"):
        assert_noncausal_layers(_fake_model(E4B_LAYER_TYPES, built), text)


@pytest.mark.cpu_test
def test_silent_causal_guard_raises_on_a_bidirectional_build_without_the_declaration():
    stock = _text_config(G31_LAYER_TYPES)
    built = [
        AttentionType.ENCODER_ONLY if t == "full_attention" else AttentionType.DECODER
        for t in G31_LAYER_TYPES
    ]
    with pytest.raises(RuntimeError, match="0 full-attention layers should be bidirectional, 10 are"):
        assert_noncausal_layers(_fake_model(G31_LAYER_TYPES, built), stock)
