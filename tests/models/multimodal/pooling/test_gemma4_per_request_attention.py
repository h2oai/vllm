# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 per-request attention on ONE generate engine against the pooling-runner two-way read (GPU).

A request flagged ``vllm_xargs attention_mode=bidirectional`` on a stock-config Gemma-4 (generate runner, Triton
attention, the prompt prefilled whole) must give the label distribution the pooling runner gives under the non-causal
declaration (full-attention layers two-way, sliding-window layers causal); an unflagged request in the SAME batch
must give the causal pooling readout. Before the per-group causal fix the flagged read put a two-sided window on the
sliding layers and sat far from the two-way reference.

Model: google/gemma-4-E4B-it by default (42 layers, 7 full-attention, 18 KV-shared; one 80 GB GPU), or
GEMMA4_MODEL=<path or repo> (e.g. the 31B merge with GEMMA4_QUANT=fp8). About 10 minutes on an H100.
"""

import os

import pytest
import torch

from vllm import SamplingParams
from vllm.config import PoolerConfig
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E4B-it")
QUANT = os.environ.get("GEMMA4_QUANT") or None
MAX_LEN = 2048
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
    "State: the parcel was delivered to a neighbour who kept it; the sender wants a refund.\n"
    "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
    "D: other\nAnswer with the letter of one option only.",
    "State: the app shows a blank screen after the login step on Android 14.\n"
    "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
    "D: other\nAnswer with the letter of one option only.",
    "State: a customer asks whether the store is open on public holidays.\n"
    "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
    "D: other\nAnswer with the letter of one option only.",
]
# Both sides are vLLM Triton bf16 forwards of the same weights; they differ by the kernel path only (the generate
# runner's unified attention with a per-sequence flag vs the pooling runner's encoder-only prefill kernel). The
# Triton-vs-SDPA gap measured for these readouts was 0.05 max; the same bound is used here.
MAX_ABS_DP = 0.05
MAX_MEAN_DP = 0.01


def _label_rows(llm, label_ids):
    """The lm_head rows of the labels and the final logit softcap, read inside the worker."""
    def rows(model):
        head = getattr(model, "lm_head", None) or getattr(model.language_model, "lm_head")
        cap = getattr(getattr(model, "logits_processor", None), "soft_cap", None)
        return head.weight[label_ids].float().cpu(), (float(cap) if cap else 0.0)
    return llm.apply_model(rows)[0]


def _pooling_label_probs(vllm_runner, prompts, W, cap, noncausal):
    """The pooling runner's readout: the last hidden state against the label rows W (taken from the generate
    engine, whose lm_head is certainly loaded; Gemma ties it to the embeddings) with the final softcap."""
    hf_overrides = {"text_config": {"noncausal_full_attention": True}} if noncausal else {}
    with vllm_runner(
        MODEL, runner="pooling", convert="embed",
        pooler_config=PoolerConfig(task="embed", seq_pooling_type="LAST", use_activation=False),
        hf_overrides=hf_overrides, max_model_len=MAX_LEN, enable_prefix_caching=False, quantization=QUANT,
    ) as vllm_model:
        outputs = vllm_model.llm.embed(prompts, use_tqdm=False)
    h = torch.stack([torch.tensor(o.outputs.embedding) for o in outputs]).float()
    logits = h @ W.T
    if cap:
        logits = cap * torch.tanh(logits / cap)
    return torch.softmax(logits, dim=-1)


def _one_engine_label_probs(vllm_runner, prompts, label_ids):
    """(flagged, unflagged, W, cap): the label distributions from ONE generate engine, the two modes mixed in one
    batch, plus the label rows and softcap read from it for the pooling-runner reference."""
    flagged = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(label_ids), logprob_token_ids=label_ids,
                             extra_args={"attention_mode": "bidirectional"})
    plain = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(label_ids), logprob_token_ids=label_ids)
    with vllm_runner(
        MODEL, attention_backend="TRITON_ATTN", max_model_len=MAX_LEN, max_num_batched_tokens=MAX_LEN,
        quantization=QUANT,
    ) as vllm_model:
        W, cap = _label_rows(vllm_model.llm, label_ids)
        mixed = [p for pair in zip(prompts, prompts) for p in pair]
        params = [flagged, plain] * len(prompts)
        outputs = vllm_model.llm.generate(mixed, params, use_tqdm=False)
    rows = []
    for out in outputs:
        lp = out.outputs[0].logprobs[0]
        missing = [i for i in label_ids if i not in lp]
        assert not missing, f"label tokens {missing} missing from the logprobs"
        rows.append(torch.tensor([lp[i].logprob for i in label_ids]))
    probs = torch.softmax(torch.stack(rows), dim=-1)
    return probs[0::2], probs[1::2], W, cap


def test_flagged_read_is_the_two_way_pooling_read_and_unflagged_the_causal_one(vllm_runner):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    label_ids = [tok(label, add_special_tokens=False)["input_ids"][-1] for label in LABELS]
    two_way, one_way, W, cap = _one_engine_label_probs(vllm_runner, PROMPTS, label_ids)
    cleanup_dist_env_and_memory()
    ref_two_way = _pooling_label_probs(vllm_runner, PROMPTS, W, cap, noncausal=True)
    cleanup_dist_env_and_memory()
    ref_one_way = _pooling_label_probs(vllm_runner, PROMPTS, W, cap, noncausal=False)

    d_two = (two_way - ref_two_way).abs()
    d_one = (one_way - ref_one_way).abs()
    assert d_one.max().item() <= MAX_ABS_DP and d_one.mean().item() <= MAX_MEAN_DP, (one_way, ref_one_way)
    assert d_two.max().item() <= MAX_ABS_DP and d_two.mean().item() <= MAX_MEAN_DP, (two_way, ref_two_way)
    # the flag must change the readout by clearly more than the engine-vs-reference gap (an engine that ignored
    # the flag, or read the sliding layers two-sided, would not land on both references at once)
    separation = (two_way - one_way).abs().max().item()
    assert separation > max(2 * d_two.max().item(), 1e-2), (separation, two_way, one_way)
