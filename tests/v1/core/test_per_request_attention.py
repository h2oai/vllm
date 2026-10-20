# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request attention mode (H2O): a request that asks for bidirectional attention
through ``vllm_xargs`` / ``extra_args`` is prefill-only, never reads or writes the
prefix cache, is scheduled whole or deferred, and is finished with an error (never a
scheduler crash) when it can never fit one step. CPU only."""

from types import SimpleNamespace

import pytest

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_scheduler


def _request(
    req_id: str,
    num_tokens: int,
    mode: str | None,
    max_tokens: int = 1,
    block_size: int = 16,
):
    init_none_hash(sha256)
    extra = {"attention_mode": mode} if mode is not None else None
    sp = SamplingParams(max_tokens=max_tokens, extra_args=extra)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def test_mode_parsing_and_prefill_only():
    assert _request("a", 32, None).attention_mode == "causal"
    assert _request("b", 32, "two-way").attention_mode == "bidirectional"
    assert _request("c", 32, "ONE_WAY").attention_mode == "causal"
    with pytest.raises(ValueError, match="attention_mode"):
        _request("d", 32, "sideways")
    with pytest.raises(ValueError, match="max_tokens=1"):
        _request("e", 32, "bidirectional", max_tokens=4)


def test_bidirectional_request_has_no_block_hashes():
    causal = _request("a", 64, None)
    two_way = _request("b", 64, "bidirectional")
    assert len(causal.block_hashes) == 4
    assert two_way.block_hashes == []  # never read from or written to the prefix cache


def test_scheduled_whole_or_deferred_and_too_long_is_an_error():
    scheduler = create_scheduler(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )
    short = _request("short", 40, "bidirectional")
    long_ok = _request("long_ok", 60, "bidirectional")
    too_long = _request("too_long", 100, "bidirectional")
    for r in (short, long_ok, too_long):
        scheduler.add_request(r)
    out = scheduler.schedule()
    # the first fits whole (40 <= 64); the second (60) would be chunked after it, so it
    # is deferred, not chunked
    assert out.num_scheduled_tokens.get("short") == 40
    assert "long_ok" not in out.num_scheduled_tokens
    # the third can never fit a step: it is finished with an error for that request
    # only, through
    # update_from_output, which EMITS the finished output (finish_reason error) so the
    # client gets an answer instead
    # of waiting forever (freeing it inside schedule() left no output at all); the
    # scheduler stays alive
    assert "too_long" not in out.num_scheduled_tokens
    assert "too_long" in scheduler.bidirectional_error_reqs
    model_output = ModelRunnerOutput(
        req_ids=["short"],
        req_id_to_index={"short": 0},
        sampled_token_ids=[[1000]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_outputs = scheduler.update_from_output(out, model_output)
    finished = {
        o.request_id: o.finish_reason
        for eco in engine_outputs.values()
        for o in eco.outputs
    }
    assert finished.get("too_long") == FinishReason.ERROR
    assert too_long.status == RequestStatus.FINISHED_ERROR
    assert (
        "too_long" not in scheduler.requests and not scheduler.bidirectional_error_reqs
    )
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens.get("long_ok") == 60  # whole, in a later step


def _mm_request(req_id, num_tokens, image_offset, image_len, mode):
    """A request with one image of image_len encoder tokens at image_offset."""
    init_none_hash(sha256)
    extra = {"attention_mode": mode} if mode is not None else None
    sp = SamplingParams(max_tokens=1, extra_args=extra)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=image_offset, length=image_len),
        identifier=f"image-{req_id}",
        modality="image",
    )
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sp,
        pooling_params=None,
        mm_features=[feature],
        block_hasher=get_request_block_hasher(16, sha256),
    )


def test_the_long_prefill_threshold_never_splits_a_bidirectional_prompt():
    scheduler = create_scheduler(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=16,
    )
    two_way = _request("two_way", 40, "bidirectional")
    causal = _request("causal", 40, None, max_tokens=4)
    scheduler.add_request(two_way)
    scheduler.add_request(causal)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["two_way"] == 40  # whole, past the 16-token clamp
    assert out.num_scheduled_tokens["causal"] == 16  # the clamp still applies here


def test_the_static_step_cap_counts_the_draft_slots():
    """Parallel drafting reserves slots in every step: a bidirectional prompt longer
    than max_num_batched_tokens less those slots can never fit one step and is an
    error, not a deferral forever; one that just fits is scheduled whole."""
    scheduler = create_scheduler(
        max_num_batched_tokens=64, max_num_seqs=4, enable_chunked_prefill=True
    )
    scheduler.vllm_config.speculative_config = SimpleNamespace(
        max_num_new_slots_for_drafting=2
    )
    over = _request("over", 63, "bidirectional")
    fits = _request("fits", 62, "bidirectional")
    scheduler.add_request(over)
    scheduler.add_request(fits)
    out = scheduler.schedule()
    assert "over" in scheduler.bidirectional_error_reqs
    assert "over" not in out.num_scheduled_tokens
    assert out.num_scheduled_tokens.get("fits") == 62


def test_an_encoder_split_defers_and_a_never_fitting_encoder_input_errors():
    """A bidirectional prompt whose image no longer fits this step's encoder budget
    is deferred whole (never prefilled up to the image); one whose encoder inputs
    exceed what any step can encode is finished with an error."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256, max_num_seqs=4, enable_chunked_prefill=True
    )  # the token budget never binds here: only the encoder budget does
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    first = _mm_request("first", 48, 0, 40, None)  # takes 40 of the 64 this step
    second = _mm_request("second", 40, 4, 32, "bidirectional")  # 32 > the 24 left
    never = _mm_request("never", 100, 4, 80, "bidirectional")  # 80 > any step's 64
    for request in (first, second, never):
        scheduler.add_request(request)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens.get("first") == 48
    assert "second" not in out.num_scheduled_tokens
    assert "never" in scheduler.bidirectional_error_reqs
    model_output = ModelRunnerOutput(
        req_ids=["first"],
        req_id_to_index={"first": 0},
        sampled_token_ids=[[1000]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(out, model_output)
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens.get("second") == 40  # whole, the budget free again


def _run_until_scheduled(scheduler, req_id, max_steps):
    """Steps the scheduler with a new 8-token causal request (8 output tokens)
    arriving every step; every scheduled request samples one token per step.
    Returns the step at which req_id was scheduled, or None."""
    for step in range(max_steps):
        if step > 0:
            scheduler.add_request(_request(f"arrival-{step}", 8, None, max_tokens=8))
        out = scheduler.schedule()
        if req_id in out.num_scheduled_tokens:
            return step
        ids = list(out.num_scheduled_tokens)
        scheduler.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={r: i for i, r in enumerate(ids)},
                sampled_token_ids=[[1000]] * len(ids),
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    return None


def test_a_deferred_bidirectional_prompt_ages_and_holds_admission(monkeypatch):
    """A bidirectional prompt that fits one step (62 of 64 tokens) but not this
    step's remaining budget is deferred. With a short causal request arriving
    every step and decoding for 8 steps, the room never comes by itself: each
    admitted request keeps a token of every later step's budget. After
    BIDIRECTIONAL_MAX_DEFERRALS deferrals the scheduler admits nothing behind the
    prompt, the running set drains, and the prompt is scheduled whole."""
    from vllm.v1.core.sched import scheduler as sched_module

    def fresh():
        scheduler = create_scheduler(
            max_num_batched_tokens=64, max_num_seqs=64, enable_chunked_prefill=True
        )
        for i in range(4):
            scheduler.add_request(_request(f"first-{i}", 8, None, max_tokens=8))
        scheduler.add_request(_request("two_way", 62, "bidirectional"))
        return scheduler

    # without aging (the behaviour before the fix) the prompt never runs
    monkeypatch.setattr(sched_module, "BIDIRECTIONAL_MAX_DEFERRALS", 10**9)
    assert _run_until_scheduled(fresh(), "two_way", 40) is None
    monkeypatch.setattr(sched_module, "BIDIRECTIONAL_MAX_DEFERRALS", 8)
    scheduler = fresh()
    step = _run_until_scheduled(scheduler, "two_way", 40)
    # 8 deferrals, then at most 8 steps for the running requests to finish
    assert step is not None and step <= 8 + 8 + 1, step
    assert scheduler.requests["two_way"].bidirectional_deferrals >= 8
