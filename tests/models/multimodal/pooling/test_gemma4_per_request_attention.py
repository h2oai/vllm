# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma-4 per-request attention on ONE generate engine against the pooling runner's
two-way read (GPU).

A request flagged ``vllm_xargs attention_mode=bidirectional`` on a stock-config Gemma-4
(generate runner, Triton attention, the prompt prefilled whole) must give the label
distribution the pooling runner gives under the non-causal declaration (full-attention
layers two-way, sliding-window layers causal), and an unflagged request in the SAME
batch must give the causal pooling readout. Before the per-group causal fix the flagged
read put a two-sided window on the sliding layers and sat far from the two-way
reference.

Model: google/gemma-4-E4B-it by default (42 layers, 7 full-attention, 18 KV-shared;
one 80 GB GPU), or GEMMA4_MODEL=<path or repo> (e.g. a larger Gemma-4 checkpoint with
GEMMA4_QUANT=fp8). About 10 minutes on an H100. Skipped without a CUDA GPU and when
the model is not accessible (the default repository is gated).
"""

import json
import os

import pytest
import torch

from vllm import SamplingParams
from vllm.config import PoolerConfig
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E4B-it")
QUANT = os.environ.get("GEMMA4_QUANT") or None


def _model_accessible() -> bool:
    """A local folder, or a Hub repository this environment may read (the default is
    gated)."""
    if os.path.isdir(MODEL):
        return True
    try:
        from huggingface_hub import auth_check

        auth_check(MODEL)
        return True
    except Exception:  # noqa: BLE001 - gated, missing, or offline: skip either way
        return False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory < 40 * 2**30,
        reason="needs a GPU with at least 40 GB",
    ),
]
MAX_LEN = 4096
# a text-only test: a multimodal Gemma-4 checkpoint's image items (2,496 tokens) would
# need a larger step
MM_LIMIT = {"image": 0, "video": 0}
LABELS = [" A", " B", " C", " D"]
PROMPTS = [
    (
        "State: the order arrived damaged; the customer has a receipt.\n"
        "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
        "D: other\nAnswer with the letter of one option only."
    ),
    (
        "State: invoice 4471 was charged twice on the same day.\n"
        "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
        "D: other\nAnswer with the letter of one option only."
    ),
    (
        "State: the device reboots whenever the firmware update starts.\n"
        "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
        "D: other\nAnswer with the letter of one option only."
    ),
    (
        "State: the parcel was delivered to a neighbour who kept it; the sender "
        "wants a refund.\n"
        "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
        "D: other\nAnswer with the letter of one option only."
    ),
    (
        "State: the app shows a blank screen after the login step on Android 14.\n"
        "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
        "D: other\nAnswer with the letter of one option only."
    ),
    (
        "State: a customer asks whether the store is open on public holidays.\n"
        "Question: which team handles this?\nA: returns\nB: billing\nC: technical\n"
        "D: other\nAnswer with the letter of one option only."
    ),
]
# Both sides are vLLM Triton bf16 forwards of the same weights and differ by the
# kernel path only: the generate runner's unified attention with a per-sequence flag
# against the pooling runner's encoder-only prefill kernel. The Triton-vs-SDPA gap
# measured for these readouts was 0.05 at most; the same bound is used here.
MAX_ABS_DP = 0.05
MAX_MEAN_DP = 0.01
# Under FP8 (GEMMA4_QUANT=fp8) two engines of the same weights drift on near-tie
# prompts (measured: up to 0.25 between two FP8 starts of one engine on the same
# probe), so the bounds are wider there and the argmax is judged on clear rows only
# (reference top-2 margin at or above CLEAR_MARGIN).
FP8_MAX_ABS_DP = 0.30
FP8_MAX_MEAN_DP = 0.06
CLEAR_MARGIN = 0.20


def _label_rows(model, label_ids):
    """The lm_head rows of the labels (Gemma ties lm_head to the token embeddings; the
    checkpoint's lm_head.weight
    when it has one) read straight from the safetensors shards, and the final logit
    softcap from the config.
    Read from the files, not through the engine: a function cannot cross the V1
    multiprocess executor."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    def fetch(name):
        return (
            name
            if os.path.isdir(model) and os.path.exists(os.path.join(model, name))
            else hf_hub_download(model, name)
        )

    local = os.path.isdir(model)
    index_path = (
        os.path.join(model, "model.safetensors.index.json")
        if local
        else fetch("model.safetensors.index.json")
    )
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    key = next((k for k in weight_map if k.endswith("lm_head.weight")), None) or next(
        k for k in weight_map if k.endswith("embed_tokens.weight")
    )
    shard = os.path.join(model, weight_map[key]) if local else fetch(weight_map[key])
    with safe_open(shard, "pt") as f:
        W = f.get_slice(key)[label_ids].float()
    cap = getattr(
        AutoConfig.from_pretrained(model).get_text_config(),
        "final_logit_softcapping",
        None,
    )
    return W, (float(cap) if cap else 0.0)


def _pooling_label_probs(vllm_runner, prompts, W, cap, noncausal):
    """The pooling runner's readout: the last hidden state against the label rows W
    (taken from the generate
    engine, whose lm_head is certainly loaded; Gemma ties it to the embeddings) with the
    final softcap."""
    hf_overrides = (
        {"text_config": {"noncausal_full_attention": True}} if noncausal else {}
    )
    with vllm_runner(
        MODEL,
        runner="pooling",
        convert="embed",
        pooler_config=PoolerConfig(
            task="embed", seq_pooling_type="LAST", use_activation=False
        ),
        hf_overrides=hf_overrides,
        max_model_len=MAX_LEN,
        enable_prefix_caching=False,
        quantization=QUANT,
        limit_mm_per_prompt=MM_LIMIT,
    ) as vllm_model:
        outputs = vllm_model.llm.embed(prompts, use_tqdm=False)
    h = torch.stack([torch.tensor(o.outputs.embedding) for o in outputs]).float()
    logits = h @ W.T
    if cap:
        logits = cap * torch.tanh(logits / cap)
    return torch.softmax(logits, dim=-1)


def _one_engine_label_probs(vllm_runner, prompts, label_ids):
    """(flagged, unflagged) label distributions from ONE generate engine, the two modes
    mixed in one batch."""
    flagged = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(label_ids),
        logprob_token_ids=label_ids,
        extra_args={"attention_mode": "bidirectional"},
    )
    plain = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(label_ids),
        logprob_token_ids=label_ids,
    )
    with vllm_runner(
        MODEL,
        attention_backend="TRITON_ATTN",
        max_model_len=MAX_LEN,
        max_num_batched_tokens=MAX_LEN,
        quantization=QUANT,
        limit_mm_per_prompt=MM_LIMIT,
    ) as vllm_model:
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
    return probs[0::2], probs[1::2]


def test_flagged_read_is_the_two_way_pooling_read_and_unflagged_the_causal_one(
    vllm_runner,
):
    if not _model_accessible():  # at run time, not at collection: a network call
        pytest.skip(f"model {MODEL} is not accessible here")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    label_ids = [
        tok(label, add_special_tokens=False)["input_ids"][-1] for label in LABELS
    ]
    W, cap = _label_rows(MODEL, label_ids)
    two_way, one_way = _one_engine_label_probs(vllm_runner, PROMPTS, label_ids)
    cleanup_dist_env_and_memory()
    ref_two_way = _pooling_label_probs(vllm_runner, PROMPTS, W, cap, noncausal=True)
    cleanup_dist_env_and_memory()
    ref_one_way = _pooling_label_probs(vllm_runner, PROMPTS, W, cap, noncausal=False)

    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    print("one-way (one engine, unflagged):", one_way, sep="\n")
    print("one-way reference (pooling runner, causal):", ref_one_way, sep="\n")
    print("two-way (one engine, flagged):", two_way, sep="\n")
    print("two-way reference (pooling runner, non-causal):", ref_two_way, sep="\n")
    max_dp, mean_dp = (
        (FP8_MAX_ABS_DP, FP8_MAX_MEAN_DP) if QUANT else (MAX_ABS_DP, MAX_MEAN_DP)
    )
    d_two = (two_way - ref_two_way).abs()
    d_one = (one_way - ref_one_way).abs()
    print(
        "gaps: one-way max/mean",
        d_one.max().item(),
        d_one.mean().item(),
        "two-way max/mean",
        d_two.max().item(),
        d_two.mean().item(),
    )
    for ours, ref in ((one_way, ref_one_way), (two_way, ref_two_way)):
        clear = [
            r
            for r in range(len(PROMPTS))
            if float(ref[r].topk(2).values.diff().abs()) >= CLEAR_MARGIN
        ]
        assert all(int(ours[r].argmax()) == int(ref[r].argmax()) for r in clear), (
            clear,
            ours,
            ref,
        )
    assert d_one.max().item() <= max_dp and d_one.mean().item() <= mean_dp, (
        one_way,
        ref_one_way,
    )
    assert d_two.max().item() <= max_dp and d_two.mean().item() <= mean_dp, (
        two_way,
        ref_two_way,
    )
    # the flag must change the readout by clearly more than the engine-vs-reference gap
    # (an engine that ignored
    # the flag, or read the sliding layers two-sided, would not land on both references
    # at once)
    separation = (two_way - one_way).abs().max().item()
    assert separation > max(2 * d_two.max().item(), 1e-2), (
        separation,
        two_way,
        one_way,
    )
