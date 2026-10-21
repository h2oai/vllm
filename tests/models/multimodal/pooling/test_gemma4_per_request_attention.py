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
reference. The test is decided on the answers (each path returns its own reference's
answer on every clear row, and the two-way answer where the modes disagree); the
numeric gaps are capped only as a loose sanity bound, not a regression bound, since
the pooling reference itself moves between runs; see the bounds below.

Model: google/gemma-4-E4B-it by default (42 layers, 7 full-attention, 18 KV-shared;
a 40 GB GPU is enough), or GEMMA4_MODEL=<path or repo> (e.g. a larger Gemma-4
checkpoint, 80 GB or more in bf16). About 10 minutes on an H100. Skipped without a
CUDA GPU, below 40 GB, and when the model is not accessible.
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
# the generate engine's attention backend: TRITON_ATTN (default) applies the
# per-sequence flag in its unified kernel; FLASH_ATTN applies it through
# FlashAttention 4 (GEMMA4_FA_VERSION=4; the version is resolved per model and
# GPU, so set it explicitly)
BACKEND = os.environ.get("GEMMA4_BACKEND", "TRITON_ATTN")
FA_VERSION = int(os.environ.get("GEMMA4_FA_VERSION", "0")) or None
ENGINE_EXTRA = (
    {"attention_config": {"flash_attn_version": FA_VERSION}} if FA_VERSION else {}
)


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
# Both sides are vLLM forwards of the same weights through the same Triton kernels,
# but in different batches and through different readouts (the generate runner's
# logprobs against a pooled hidden state times the label rows), so they differ by
# numerics: measured on google/gemma-4-E4B-it in bf16 up to 0.09 and 0.14 in two runs
# (mean 0.02-0.03; the engine's reads were identical between the runs, the pooling
# reference moved by up to 0.07 on near-tie rows), and on a 31B Gemma-4 checkpoint
# in FP8 up to 0.31 on a three-way near tie (mean 0.05), where the e4m3 rounding
# amplifies accumulation-order differences. The attention mode moves clear rows by
# 0.5-0.9. The test is therefore decided on the semantics -- on every row whose own
# reference is clear (top-2 margin at or above CLEAR_MARGIN) the engine returns the
# reference's answer, and on every row where the two references give different clear
# answers the flagged read returns the two-way answer and the unflagged read the
# causal one (at least one such row must exist) -- and the gaps stay under absolute
# caps (the row-wise cap on the clear rows, the mean over every row), a loose sanity
# bound rather than a regression bound.
# Run this test in bf16. Under FP8 the pooling reference itself is not reproducible:
# on the 31B checkpoint the causal reference moved by up to 0.8 on one row between
# two runs on the same GPU (the one-engine reads were identical in both), so the FP8
# caps below are kept only for exploratory runs with GEMMA4_QUANT=fp8.
MAX_ABS_DP = 0.25
MAX_MEAN_DP = 0.08
FP8_MAX_ABS_DP = 0.40
FP8_MAX_MEAN_DP = 0.12
CLEAR_MARGIN = 0.20


def _label_rows(model, label_ids):
    """The lm_head rows of the labels, read straight from the safetensors files (the
    checkpoint's lm_head.weight when it has one, else the token embeddings Gemma ties
    the lm_head to), and the final logit softcap from the config. Read from the files,
    not through the engine: a function cannot cross the V1 multiprocess executor."""
    from huggingface_hub import file_exists, hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    local = os.path.isdir(model)

    def fetch(name):
        return os.path.join(model, name) if local else hf_hub_download(model, name)

    def present(name):
        return (
            os.path.exists(os.path.join(model, name))
            if local
            else file_exists(model, name)
        )

    def pick(keys):
        return next((k for k in keys if k.endswith("lm_head.weight")), None) or next(
            k for k in keys if k.endswith("embed_tokens.weight")
        )

    if present("model.safetensors.index.json"):
        with open(fetch("model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        key = pick(weight_map)
        shard = fetch(weight_map[key])
    else:
        # a single-file checkpoint (google/gemma-4-E4B-it ships one model.safetensors)
        shard = fetch("model.safetensors")
        with safe_open(shard, "pt") as f:
            key = pick(list(f.keys()))
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
    (read from the checkpoint files by _label_rows) with the final softcap."""
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


def _iteration_histogram(llm):
    """The engine's vllm:iteration_tokens_total histogram: (count, sum, buckets),
    buckets mapping each upper limit to the cumulative count of iterations."""
    for metric in llm.get_metrics():
        if metric.name == "vllm:iteration_tokens_total":
            return int(metric.count), float(metric.sum), dict(metric.buckets)
    raise AssertionError("vllm:iteration_tokens_total not in LLM.get_metrics()")


def _iterations_above(before, after, tokens):
    """Iterations between two histogram readings that carried more than
    ``tokens`` tokens (the largest bucket limit not above ``tokens`` bounds the
    count below it)."""
    count0, _, buckets0 = before
    count1, _, buckets1 = after
    limits = [float(k) for k in buckets1 if k != "+Inf" and float(k) <= tokens]
    if not limits:
        return count1 - count0
    key = next(k for k in buckets1 if k != "+Inf" and float(k) == max(limits))
    return (count1 - count0) - (buckets1[key] - buckets0.get(key, 0))


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
        attention_backend=BACKEND,
        disable_log_stats=False,  # LLM.get_metrics() for the step count below
        enable_prefix_caching=False,
        # a hybrid model with Mamba-style state layers needs chunked prefill on
        # when prefix caching is on; a bidirectional prompt is still prefilled
        # whole (whole-or-defer in the scheduler)
        enable_chunked_prefill=True,
        **ENGINE_EXTRA,
        max_model_len=MAX_LEN,
        max_num_batched_tokens=MAX_LEN,
        quantization=QUANT,
        limit_mm_per_prompt=MM_LIMIT,
    ) as vllm_model:
        # an unflagged pass first: on FLASH_ATTN with FlashAttention 4 it leaves
        # compiled causal kernels in the JIT cache that a later per-sequence call
        # must not reuse (the vendored compile key does not record the flag)
        vllm_model.llm.generate(prompts, plain, use_tqdm=False)
        mixed = [p for pair in zip(prompts, prompts) for p in pair]
        params = [flagged, plain] * len(prompts)
        before = _iteration_histogram(vllm_model.llm)
        outputs = vllm_model.llm.generate(mixed, params, use_tqdm=False)
        after = _iteration_histogram(vllm_model.llm)
        # the mixed batch (flagged and unflagged prompts, 12 prompts of ~60
        # tokens against a 4,096-token step) is prefilled in ONE engine step:
        # the two subsets share the step, nothing is serialised. Async
        # scheduling may add one near-empty iteration (the decode step it
        # schedules before the finish is seen, one token per request), so the
        # check counts the iterations that carried more than two tokens per
        # request (the histogram's 16-token bucket for 12 prompts; the
        # buckets are Prometheus-style cumulative counts). The engine core
        # runs in-process for this (see the top of the test).
        total = sum(len(o.prompt_token_ids) for o in outputs)
        iterations = after[0] - before[0]
        big = _iterations_above(before, after, 2 * len(mixed))
        print(
            "mixed batch:",
            total,
            "prompt tokens in",
            iterations,
            "iterations;",
            big,
            "iteration(s) above two tokens per request",
        )
        assert iterations <= 2 and big == 1 and after[1] - before[1] >= total, (
            total,
            iterations,
            big,
            before,
            after,
        )
    rows = []
    for out in outputs:
        lp = out.outputs[0].logprobs[0]
        missing = [i for i in label_ids if i not in lp]
        assert not missing, f"label tokens {missing} missing from the logprobs"
        rows.append(torch.tensor([lp[i].logprob for i in label_ids]))
    probs = torch.softmax(torch.stack(rows), dim=-1)
    return probs[0::2], probs[1::2]


def test_flagged_read_is_the_two_way_pooling_read_and_unflagged_the_causal_one(
    vllm_runner, monkeypatch
):
    if not _model_accessible():  # at run time, not at collection: a network call
        pytest.skip(f"model {MODEL} is not accessible here")
    # the one-step check below needs the engine core in this process: the
    # multiprocess core schedules as soon as it holds a request, so on a slow
    # or loaded host it can start the batch before every request has arrived
    # and split it over two steps for timing reasons alone (seen on a loaded
    # workstation and in a container); in-process, generate() adds every
    # request before the first step
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
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

    def clear_rows(ref):
        return [
            r
            for r in range(len(PROMPTS))
            if float(ref[r].topk(2).values.diff().abs()) >= CLEAR_MARGIN
        ]

    # 1. each path returns its own reference's answer on every clear row
    for ours, ref in ((one_way, ref_one_way), (two_way, ref_two_way)):
        clear = clear_rows(ref)
        assert clear, ref
        assert all(int(ours[r].argmax()) == int(ref[r].argmax()) for r in clear), (
            clear,
            ours,
            ref,
        )
    # 2. where the two modes give different clear answers, the flagged read returns
    # the two-way answer and the unflagged read the causal one (an engine that ignored
    # the flag, applied it to every request in the batch, or read the sliding layers
    # two-sided would not land on both). A prompt set without such a row cannot
    # discriminate and fails here.
    discriminating = [
        r
        for r in clear_rows(ref_one_way)
        if r in clear_rows(ref_two_way)
        and int(ref_one_way[r].argmax()) != int(ref_two_way[r].argmax())
    ]
    print(
        "discriminating rows (clear, different answers):",
        len(discriminating),
        discriminating,
    )
    assert len(discriminating) >= 1, (ref_one_way, ref_two_way)
    for r in discriminating:
        assert int(one_way[r].argmax()) == int(ref_one_way[r].argmax()), (
            r,
            one_way[r],
            ref_one_way[r],
        )
        assert int(two_way[r].argmax()) == int(ref_two_way[r].argmax()), (
            r,
            two_way[r],
            ref_two_way[r],
        )
    # 3. the numeric gaps stay under the absolute caps: a loose sanity bound, not
    # the decision, which is checks 1 and 2. The row-wise cap is judged on the
    # clear rows only: on a near-tie row the pooling reference itself can land on
    # another answer in another run (measured in bf16 on the larger checkpoint:
    # the two-way reference of one near-tie row moved from 0.48 to 0.52 on a
    # different option, a 0.47 gap, with the engine's reads identical in both
    # runs), so a gap there measures the reference, not the engine. The mean is
    # over every row.
    for ours, ref, gap in (
        (one_way, ref_one_way, d_one),
        (two_way, ref_two_way, d_two),
    ):
        clear = clear_rows(ref)
        assert gap[clear].max().item() <= max_dp, (clear, gap, ours, ref)
        assert gap.mean().item() <= mean_dp, (gap, ours, ref)
