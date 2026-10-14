# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 non-causal full attention against a transformers reference (GPU, E4B-sized: fits an L40S; ~10 min).

The reference is transformers' Gemma4 with the same mask change the checkpoint
was trained with: on the full-attention layers the causal mask becomes the
padding mask and the attention modules' ``is_causal`` is False; the
sliding-window layers keep their causal window mask. The readout is the one a
decision shim makes: the last token's hidden state against the ``lm_head``
rows of a few label tokens, Gemma's final logit softcap applied, softmax over
the labels. vLLM serves the model under the pooling runner (``--convert embed``,
LAST pooling, no normalisation) and must give the same label distributions.

Also checked: WITHOUT the declaration the same pooling readout equals the
causal transformers forward, so stock behaviour is unchanged.
"""

import pytest
import torch

from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "google/gemma-4-E4B-it"  # 42 layers, 7 full-attention, 18 KV-shared
LABELS = [" A", " B", " C", " D"]
PROMPTS = [
    "State: the order arrived damaged; the customer has a receipt.\n"
    "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
    "D: other\nAnswer with the letter of one option only.",
    "State: invoice 4471 was charged twice on the same day.\n"
    "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
    "D: other\nAnswer with the letter of one option only.",
    "State: the device reboots whenever the firmware update starts.\n"
    "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
    "D: other\nAnswer with the letter of one option only.",
]
MAX_ARGMAX_DISAGREEMENT = 0.01
MAX_ABS_DP = 0.02


def _reference_label_probs(model_name, prompts, label_ids, noncausal):
    """Label distributions from transformers, causal or with the non-causal
    full-attention mask (the training-time hook, inlined)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    text_config = model.config.get_text_config()
    text_model = model.model.language_model
    full_modules = [
        layer.self_attn
        for layer, kind in zip(text_model.layers, text_config.layer_types)
        if kind == "full_attention"
    ]
    if noncausal:
        for m in full_modules:
            m.is_causal = False

        def pre(module, args, kwargs):
            am = kwargs.get("attention_mask")
            pad = kwargs.pop("_pad_mask", None)
            if isinstance(am, dict) and pad is not None:
                am = dict(am)
                am["full_attention"] = (
                    None if bool(pad.all()) else pad.bool()[:, None, None, :]
                )
                kwargs["attention_mask"] = am
            return args, kwargs

        text_model.register_forward_pre_hook(pre, with_kwargs=True)

    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        # The wrapper builds the mask dict itself and passes it to the text
        # model; the hook needs the 2-D padding mask alongside.
        if noncausal:
            handle = text_model.register_forward_pre_hook(
                lambda m, a, kw: (a, {**kw, "_pad_mask": enc.attention_mask}),
                with_kwargs=True,
                prepend=True,
            )
        out = model.model(**enc)
        if noncausal:
            handle.remove()
    h = out.last_hidden_state[:, -1, :].float()
    W = model.lm_head.weight[label_ids].float()
    logits = h @ W.T
    cap = float(getattr(text_config, "final_logit_softcapping", 0) or 0)
    if cap:
        logits = cap * torch.tanh(logits / cap)
    probs = torch.softmax(logits, dim=-1).cpu()
    del model
    torch.cuda.empty_cache()
    return probs, W.cpu(), cap


def _vllm_label_probs(vllm_runner, model_name, prompts, W, cap, noncausal):
    hf_overrides = {"text_config": {"noncausal_full_attention": True}} if noncausal else {}
    with vllm_runner(
        model_name,
        runner="pooling",
        convert="embed",
        override_pooler_config={"pooling_type": "LAST", "normalize": False},
        hf_overrides=hf_overrides,
        max_model_len=2048,
        enable_prefix_caching=False,
    ) as vllm_model:
        vllm_config = vllm_model.llm.llm_engine.vllm_config
        if noncausal:
            assert vllm_config.model_config.attn_type == "encoder_only"
            assert not vllm_config.cache_config.enable_prefix_caching
            assert not vllm_config.scheduler_config.enable_chunked_prefill
        outputs = vllm_model.llm.embed(prompts, use_tqdm=False)
    h = torch.stack([torch.tensor(o.outputs.embedding) for o in outputs]).float()
    logits = h @ W.T
    if cap:
        logits = cap * torch.tanh(logits / cap)
    return torch.softmax(logits, dim=-1)


@pytest.mark.parametrize("noncausal", [False, True])
def test_gemma4_pooling_readout_matches_transformers(vllm_runner, noncausal):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    label_ids = [tok(label, add_special_tokens=False)["input_ids"][-1] for label in LABELS]
    ref, W, cap = _reference_label_probs(MODEL, PROMPTS, label_ids, noncausal)
    cleanup_dist_env_and_memory()
    ours = _vllm_label_probs(vllm_runner, MODEL, PROMPTS, W, cap, noncausal)

    disagreement = (ref.argmax(-1) != ours.argmax(-1)).float().mean().item()
    assert disagreement <= MAX_ARGMAX_DISAGREEMENT, (ref, ours)
    assert (ref - ours).abs().max().item() <= MAX_ABS_DP, (ref, ours)


@pytest.mark.parametrize("noncausal", [True])
def test_noncausal_readout_differs_from_causal(vllm_runner, noncausal):
    """Sanity: the declaration changes the readout (a silently causal engine
    would pass the parity test against the wrong reference only if the
    reference were wrong too)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    label_ids = [tok(label, add_special_tokens=False)["input_ids"][-1] for label in LABELS]
    causal, W, cap = _reference_label_probs(MODEL, PROMPTS, label_ids, False)
    cleanup_dist_env_and_memory()
    ours = _vllm_label_probs(vllm_runner, MODEL, PROMPTS, W, cap, True)
    assert (causal - ours).abs().max().item() > 1e-3
