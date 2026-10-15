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
LAST pooling, no activation/normalisation) and must give the same label distributions.

Also checked: WITHOUT the declaration the same pooling readout equals the
causal transformers forward, so stock behaviour is unchanged.
"""

import pytest
import torch

from vllm.config import PoolerConfig
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
# Tolerances from the measured drift (bidir-test2, 2026-10-15, 20 DI dev prompts, E4B): transformers bf16 itself sits
# max 0.136 / mean 0.005 from transformers fp32; vLLM (Triton encoder path, BLOCK 32 on the 512-wide heads, bf16,
# cudagraphs) sits max 0.089 / mean 0.004 from fp32, i.e. closer than the bf16 reference; argmax agreed on all 19
# rows with a top-2 margin >= 0.05. So the reference is fp32 (the anchor), the max-|dp| bound is 0.10 (near ties
# amplify ~0.3-logit bf16 noise), the mean bound 0.01, and argmax is judged on clear rows only.
MAX_ABS_DP = 0.10
MAX_MEAN_DP = 0.01
CLEAR_MARGIN = 0.05


def _reference_label_probs(model_name, prompts, label_ids, noncausal):
    """Label distributions from transformers, causal or with the non-causal
    full-attention mask (the training-time hook, inlined).

    ONE PROMPT PER FORWARD, no padding, no mask, in fp32. A left-padded multi-row batch
    is read wrongly by PyTorch's memory-efficient SDPA kernel on Gemma-4's
    512-wide global heads under an explicit mask (measured on Gemma-4-12B:
    probabilities off by up to 0.9); a single unpadded row takes the fused
    kernel and is exact. Row by row the non-causal change is just the module
    flag (no mask to replace), which is also what vLLM's unpadded prefill does.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="sdpa"   # the fp32 anchor (E4B: ~32 GB, fits an L40S)
    ).cuda().eval()
    text_config = model.config.get_text_config()
    text_model = model.model.language_model
    if noncausal:
        for layer, kind in zip(text_model.layers, text_config.layer_types):
            if kind == "full_attention":
                layer.self_attn.is_causal = False
    W = model.lm_head.weight[label_ids].float()
    cap = float(getattr(text_config, "final_logit_softcapping", 0) or 0)
    rows = []
    with torch.no_grad():
        for prompt in prompts:
            enc = tok(prompt, return_tensors="pt").to("cuda")
            out = model.model(**enc)
            logits = out.last_hidden_state[0, -1, :].float() @ W.T
            if cap:
                logits = cap * torch.tanh(logits / cap)
            rows.append(torch.softmax(logits, dim=-1).cpu())
    probs = torch.stack(rows)
    del model
    torch.cuda.empty_cache()
    return probs, W.cpu(), cap


def _vllm_label_probs(vllm_runner, model_name, prompts, W, cap, noncausal):
    hf_overrides = {"text_config": {"noncausal_full_attention": True}} if noncausal else {}
    with vllm_runner(
        model_name,
        runner="pooling",
        convert="embed",
        pooler_config=PoolerConfig(task="embed", seq_pooling_type="LAST", use_activation=False),
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

    clear = [r for r in range(len(PROMPTS)) if float(ref[r].topk(2).values.diff().abs()) >= CLEAR_MARGIN]
    disagreement = sum(int(ref[r].argmax() != ours[r].argmax()) for r in clear) / max(len(clear), 1)
    assert disagreement <= MAX_ARGMAX_DISAGREEMENT, (ref, ours)
    assert (ref - ours).abs().max().item() <= MAX_ABS_DP, (ref, ours)
    assert (ref - ours).abs().mean().item() <= MAX_MEAN_DP, (ref, ours)


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
    engine_gap = (causal - _vllm_label_probs(vllm_runner, MODEL, PROMPTS, W, cap, False)).abs().max().item()
    # the mode change must move the readout by clearly more than the engine-vs-reference numeric gap
    assert (causal - ours).abs().max().item() > max(2 * engine_gap, 1e-2), (engine_gap, causal, ours)
