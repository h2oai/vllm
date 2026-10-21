# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma-4 E-series ``--kv-sharing-fast-prefill`` with per-request two-way reads (GPU).

With fast prefill the KV-shared upper layers (E4B: layers 24-41 of 42) run on the
logits rows only. For a request flagged ``vllm_xargs attention_mode=bidirectional``
(prefilled whole in one step) those layers read the KV the lower, two-way layers
wrote in the same step, so the readout must be the full forward's. One reference
engine without the flag and one engine with it read the same prompts, each prompt
flagged and unflagged in one mixed batch; every read must give the reference's
label argmax and stay within MAX_ABS_DP of its label distribution (bf16: the upper
layers' GEMMs run on fewer rows, so the sums are ordered differently; exact up to
that). Prompts longer than the 512-token sliding window are included. The flagged
engine runs in-process (last: it keeps its GPU memory) so the test can assert that
the KV-shared layers really ran on the reduced rows.

Model: google/gemma-4-E4B-it by default, or GEMMA4_MODEL=<path or repo> (any Gemma-4
checkpoint with num_kv_shared_layers > 0). Skipped without a CUDA GPU of 40 GB and
when the model is not accessible.
"""

import os
import random

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm import SamplingParams

MODEL = os.environ.get("GEMMA4_MODEL", "google/gemma-4-E4B-it")
MAX_LEN = 8192
MM_LIMIT = {"image": 0, "video": 0}
LABELS = [" A", " B", " C", " D"]
QUESTION = (
    "\nQuestion: which team handles this?\nA: returns\nB: billing\nC: technical\n"
    "D: other\nAnswer with the letter of one option only."
)
STATES = [
    "State: the order arrived damaged; the customer has a receipt.",
    "State: invoice 4471 was charged twice on the same day.",
    "State: the device reboots whenever the firmware update starts.",
    "State: the parcel was delivered to a neighbour who kept it; the sender "
    "wants a refund.",
    "State: the app shows a blank screen after the login step on Android 14.",
    "State: a customer asks whether the store is open on public holidays.",
]
WORDS = [
    "invoice",
    "order",
    "customer",
    "account",
    "refund",
    "billing",
    "device",
    "firmware",
    "update",
    "screen",
    "login",
    "network",
    "outage",
    "shipment",
    "parcel",
    "warehouse",
    "return",
    "receipt",
    "payment",
    "charge",
    "duplicate",
    "technical",
    "support",
    "schedule",
    "holiday",
    "store",
    "opening",
    "hours",
    "policy",
]
# bf16 bound on the label probabilities of one read, fast prefill against the
# full forward (the same kernels; only the upper layers' row count differs)
MAX_ABS_DP = 0.02


def _model_accessible() -> bool:
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


def _prompts():
    rng = random.Random(20261021)
    prompts = [s + QUESTION for s in STATES]
    for words in (700, 1500, 2600):  # past the 512-token sliding window
        body = " ".join(rng.choice(WORDS) for _ in range(words))
        prompts.append(f"State: {body}{QUESTION}")
    return prompts


def _read(llm, prompts, label_ids):
    """Label distributions (flagged, unflagged) from one mixed batch."""
    common = dict(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(label_ids),
        logprob_token_ids=label_ids,
    )
    flagged = SamplingParams(**common, extra_args={"attention_mode": "bidirectional"})
    plain = SamplingParams(**common)
    mixed = [p for pair in zip(prompts, prompts) for p in pair]
    outputs = llm.generate(mixed, [flagged, plain] * len(prompts), use_tqdm=False)
    rows = []
    for out in outputs:
        lp = out.outputs[0].logprobs[0]
        missing = [i for i in label_ids if i not in lp]
        assert not missing, f"label tokens {missing} missing from the logprobs"
        rows.append(torch.tensor([lp[i].logprob for i in label_ids]))
    probs = torch.softmax(torch.stack(rows), dim=-1)
    return probs[0::2], probs[1::2]


def _engine_kwargs(fast_prefill):
    return dict(
        attention_backend="TRITON_ATTN",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=MAX_LEN,
        max_num_batched_tokens=MAX_LEN,
        limit_mm_per_prompt=MM_LIMIT,
        kv_sharing_fast_prefill=fast_prefill,
    )


def _gemma4_text_model(llm):
    """The in-process engine's Gemma4Model (the text stack)."""
    from vllm.model_executor.models.gemma4 import Gemma4Model

    core = llm.llm_engine.engine_core.engine_core
    model = core.model_executor.driver_worker.worker.model_runner.model
    found = [m for m in model.modules() if isinstance(m, Gemma4Model)]
    assert len(found) == 1, f"expected one Gemma4Model, found {len(found)}"
    return found[0]


def test_fast_prefill_two_way_reads_match_the_full_forward(vllm_runner):
    if not _model_accessible():  # here, not in the spawned child (a skip there passes)
        pytest.skip(f"model {MODEL} is not accessible here")
    _fast_prefill_test(vllm_runner)


@create_new_process_for_each_test("spawn")
def _fast_prefill_test(vllm_runner):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from transformers import AutoTokenizer

    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.utils import KVSharingFastPrefillMetadata

    tok = AutoTokenizer.from_pretrained(MODEL)
    label_ids = [
        tok(label, add_special_tokens=False)["input_ids"][-1] for label in LABELS
    ]
    prompts = _prompts()

    with vllm_runner(MODEL, **_engine_kwargs(False)) as ref_model:
        ref_flagged, ref_plain = _read(ref_model.llm, prompts, label_ids)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    with vllm_runner(MODEL, **_engine_kwargs(True)) as fp_model:
        text_model = _gemma4_text_model(fp_model.llm)
        assert text_model.fast_prefill_enabled
        calls = []
        inner = text_model.fast_prefill_forward
        last_layer = text_model.layers[-1].self_attn.attn.layer_name

        def recording_forward(input_ids, positions, *args, **kwargs):
            md = (get_forward_context().attn_metadata or {}).get(last_layer)
            rows = (
                md.num_logits_indices
                if isinstance(md, KVSharingFastPrefillMetadata)
                else None
            )
            calls.append((positions.size(0), rows))
            return inner(input_ids, positions, *args, **kwargs)

        text_model.fast_prefill_forward = recording_forward
        fp_flagged, fp_plain = _read(fp_model.llm, prompts, label_ids)
        text_model.fast_prefill_forward = inner

    # the KV-shared layers ran on the logits rows of a multi-token step
    reduced = [(n, r) for n, r in calls if r is not None and r < n]
    print("fast-prefill steps (tokens, logits rows):", calls)
    assert reduced, f"no step ran the KV-shared layers on fewer rows: {calls}"

    for name, ref, got in (
        ("two-way", ref_flagged, fp_flagged),
        ("one-way", ref_plain, fp_plain),
    ):
        gap = (ref - got).abs().max(dim=-1).values
        same = ref.argmax(dim=-1) == got.argmax(dim=-1)
        print(
            f"{name}: max |dp| {gap.max().item():.5f}, mean {gap.mean().item():.5f}, "
            f"argmax same {int(same.sum())}/{len(same)}"
        )
        assert bool(same.all()), (name, ref, got)
        assert gap.max().item() <= MAX_ABS_DP, (name, gap.tolist())
    # the flag does something on these prompts (the reads are not trivially equal)
    assert (ref_flagged - ref_plain).abs().max().item() > MAX_ABS_DP
