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
    # only, through update_from_output, which EMITS the finished output (finish_reason
    # error) so the client gets an answer instead of waiting forever (freeing it
    # inside schedule() left no output at all); the scheduler stays alive
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
    # the deferral releases only the deferred request's own references (the plain
    # manager's free drops just those; nothing was allocated for it), and nothing
    # of any other request
    freed = []
    real_free = scheduler.encoder_cache_manager.free
    scheduler.encoder_cache_manager.free = lambda req: freed.append(req.request_id)
    out = scheduler.schedule()
    scheduler.encoder_cache_manager.free = real_free
    assert freed == ["second"]
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


def _config(
    backend_name,
    kv_connector,
    per_kind=None,
    fa_version=None,
    dcp=1,
    rswa=None,
    mm_prefix=False,
    bidirectional="none",
):
    backend = SimpleNamespace(name=backend_name) if backend_name else None
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=backend,
            backend_per_kind={
                kind: SimpleNamespace(name=name)
                for kind, name in (per_kind or {}).items()
            },
            flash_attn_version=fa_version,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        model_config=SimpleNamespace(
            rswa_window=rswa,
            is_mm_prefix_lm=mm_prefix,
            hf_text_config=SimpleNamespace(use_bidirectional_attention=bidirectional),
        ),
        kv_transfer_config=SimpleNamespace() if kv_connector else None,
        reasoning_config=None,
    )


def test_a_bidirectional_request_is_refused_where_it_cannot_be_served(monkeypatch):
    """The frontend refuses, with the reason, a bidirectional request on an engine
    that would read it causally without saying so (any backend but TRITON_ATTN or
    FLASH_ATTN with FlashAttention 4, including an unset one) or could store or
    load its non-causal KV through a KV connector; decoding (max_tokens != 1) is
    refused too. A causal request is never checked."""
    import vllm.v1.request as request_module
    from vllm.v1.request import check_bidirectional_request, parse_attention_mode

    # the platform/version part of the FLASH_ATTN check is a GPU question; here
    # it says "fine" unless a test replaces it
    monkeypatch.setattr(request_module, "_fa4_platform_reason", lambda cfg: None)
    two_way = SamplingParams(max_tokens=1, extra_args={"attention_mode": "two-way"})
    assert parse_attention_mode(two_way) == "bidirectional"
    assert parse_attention_mode(SamplingParams(max_tokens=1)) == "causal"
    assert parse_attention_mode(None) == "causal"
    with pytest.raises(ValueError, match="attention_mode"):
        parse_attention_mode(SamplingParams(extra_args={"attention_mode": "x"}))
    check_bidirectional_request(two_way, _config("TRITON_ATTN", False))  # served
    with pytest.raises(ValueError, match="TRITON_ATTN"):
        check_bidirectional_request(two_way, _config("FLASH_ATTN", False))
    with pytest.raises(ValueError, match="got auto"):
        check_bidirectional_request(two_way, _config(None, False))
    with pytest.raises(ValueError, match="got auto"):
        check_bidirectional_request(two_way, _config(None, False, fa_version=4))
    with pytest.raises(ValueError, match="KV connector"):
        check_bidirectional_request(two_way, _config("TRITON_ATTN", True))
    decoding = SamplingParams(max_tokens=4, extra_args={"attention_mode": "two-way"})
    with pytest.raises(ValueError, match="max_tokens=1"):
        check_bidirectional_request(decoding, _config("TRITON_ATTN", False))
    # a per-kind override that routes any attention kind elsewhere is refused too
    with pytest.raises(ValueError, match="backend_per_kind"):
        check_bidirectional_request(
            two_way, _config("TRITON_ATTN", False, {"full_attention": "FLASH_ATTN"})
        )
    check_bidirectional_request(
        two_way, _config("TRITON_ATTN", False, {"full_attention": "TRITON_ATTN"})
    )
    # FLASH_ATTN applies the flag through FlashAttention 4 only: accepted with the
    # version set to 4 (main backend or per kind), refused with 3 or unset
    check_bidirectional_request(two_way, _config("FLASH_ATTN", False, fa_version=4))
    with pytest.raises(ValueError, match="flash_attn_version"):
        check_bidirectional_request(two_way, _config("FLASH_ATTN", False, fa_version=3))
    check_bidirectional_request(
        two_way,
        _config("TRITON_ATTN", False, {"full_attention": "FLASH_ATTN"}, fa_version=4),
    )
    with pytest.raises(ValueError, match="backend_per_kind"):
        check_bidirectional_request(
            two_way,
            _config(
                "TRITON_ATTN", False, {"full_attention": "FLASH_ATTN"}, fa_version=3
            ),
        )
    # a FA4 call with the flag drops the mm_prefix and R-SWA masks of the whole
    # batch, and decode context parallelism has no per-sequence path: refused
    with pytest.raises(ValueError, match="decode context"):
        check_bidirectional_request(
            two_way, _config("FLASH_ATTN", False, fa_version=4, dcp=2)
        )
    with pytest.raises(ValueError, match="R-SWA"):
        check_bidirectional_request(
            two_way, _config("FLASH_ATTN", False, fa_version=4, rswa=4096)
        )
    with pytest.raises(ValueError, match="mm_prefix"):
        check_bidirectional_request(
            two_way, _config("FLASH_ATTN", False, fa_version=4, mm_prefix=True)
        )
    check_bidirectional_request(  # Gemma-4 clears the ranges on its full layers
        two_way,
        _config(
            "FLASH_ATTN", False, fa_version=4, mm_prefix=True, bidirectional="vision"
        ),
    )
    # the platform / resolved-version part is the last word
    monkeypatch.setattr(
        request_module, "_fa4_platform_reason", lambda cfg: "FlashAttention 2 resolves"
    )
    with pytest.raises(ValueError, match="FlashAttention 2 resolves"):
        check_bidirectional_request(two_way, _config("FLASH_ATTN", False, fa_version=4))
    check_bidirectional_request(two_way, _config("TRITON_ATTN", False))  # unaffected


def test_the_fa4_dynamic_causal_cache_swap(monkeypatch):
    """The FLASH_ATTN backend runs a FlashAttention-4 call that carries a
    per-sequence causal tensor under its own JIT cache (the vendored compile key
    does not record the tensor), restoring the regular cache afterwards and on
    an exception; without the tensor nothing is touched or imported."""
    import sys

    from vllm.v1.attention.backends import flash_attn as fa

    regular = {}
    fwd = SimpleNamespace(compile_cache=regular)
    iface = SimpleNamespace(_flash_attn_fwd=fwd)
    dynamic = {}
    cute = SimpleNamespace(interface=iface, cache_utils=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn.cute", cute)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn.cute.interface", iface)
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.cute.cache_utils",
        SimpleNamespace(get_jit_cache=lambda name: dynamic),
    )
    monkeypatch.setattr(fa, "_FA4_DYNAMIC_CAUSAL_CACHE", None)
    with fa._fa4_cache_for(None):
        assert fwd.compile_cache is regular
    assert fa._FA4_DYNAMIC_CAUSAL_CACHE is None  # nothing looked up without a tensor
    tensor = SimpleNamespace()  # stands in for the int32 flag tensor
    with fa._fa4_cache_for(tensor):
        assert fwd.compile_cache is dynamic
        with fa._fa4_cache_for(tensor):  # nested: still the dynamic cache
            assert fwd.compile_cache is dynamic
        assert fwd.compile_cache is dynamic
    assert fwd.compile_cache is regular
    with pytest.raises(RuntimeError):  # noqa: SIM117 (the nesting is the point)
        with fa._fa4_cache_for(tensor):
            raise RuntimeError("compile failed")
    assert fwd.compile_cache is regular


def test_the_frontend_checks_bidirectional_requests_only(monkeypatch):
    """InputProcessor._validate_params refuses a bidirectional request with a
    VLLMValidationError naming attention_mode, and never checks a causal one."""
    import vllm.v1.request as request_module
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    monkeypatch.setattr(SamplingParams, "verify", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(request_module, "_fa4_platform_reason", lambda cfg: None)

    def processor(config):
        return SimpleNamespace(
            vllm_config=config,
            model_config=SimpleNamespace(
                return_sampling_mask=False, enable_trace_replay=False
            ),
            speculative_config=None,
            structured_outputs_config=None,
            tokenizer=None,
        )

    validate = InputProcessor._validate_params
    causal = SamplingParams(max_tokens=4)
    two_way = SamplingParams(max_tokens=1, extra_args={"attention_mode": "two-way"})
    validate(processor(_config(None, False)), causal, ("generate",))
    validate(processor(_config("TRITON_ATTN", False)), two_way, ("generate",))
    with pytest.raises(VLLMValidationError, match="TRITON_ATTN") as info:
        validate(processor(_config("FLASH_ATTN", False)), two_way, ("generate",))
    assert info.value.parameter == "attention_mode"


def _mm_request_images(req_id, num_tokens, images, mode, max_tokens=1):
    """A request with several images, each (identifier, offset, length); requests
    that share an identifier share the cached encoder output."""
    init_none_hash(sha256)
    extra = {"attention_mode": mode} if mode is not None else None
    sp = SamplingParams(max_tokens=max_tokens, extra_args=extra)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            mm_position=PlaceholderRange(offset=offset, length=length),
            identifier=identifier,
            modality="image",
        )
        for identifier, offset, length in images
    ]
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sp,
        pooling_params=None,
        mm_features=features,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _step(scheduler, out):
    """One model step: every scheduled request samples one token."""
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


def test_deferred_two_way_reads_release_their_encoder_references():
    """Two two-way reads share cached images (a: 36 slots, b: 16, of 56) and each
    needs a new 12-slot image. In the step where b is encoded for a causal request,
    neither new image fits beside the two cached ones, and both reads are deferred
    at the encoder split. Before the fix each kept its reference to its cached
    image, so neither image could ever be evicted for the other's new one: both
    waited for good. Now a deferral releases the request's own references: read_b
    evicts the released image a for its new one in that same step, and read_a
    follows a step later."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256, max_num_seqs=8, enable_chunked_prefill=True
    )
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=56)
    scheduler.add_request(_mm_request_images("once_a", 40, [("image-a", 0, 36)], None))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens.get("once_a") == 40
    _step(scheduler, out)  # image a cached, unreferenced; 20 slots free
    scheduler.add_request(_mm_request_images("once_b", 20, [("image-b", 0, 16)], None))
    scheduler.add_request(
        _mm_request_images(
            "read_a", 56, [("image-a", 0, 36), ("image-e1", 40, 12)], "bidirectional"
        )
    )
    scheduler.add_request(
        _mm_request_images(
            "read_b", 36, [("image-b", 0, 16), ("image-e2", 20, 12)], "bidirectional"
        )
    )
    scheduled = {}
    for step in range(6):
        out = scheduler.schedule()
        for req_id in ("read_a", "read_b"):
            if req_id in out.num_scheduled_tokens:
                scheduled.setdefault(req_id, step)
        _step(scheduler, out)
    assert not scheduler.bidirectional_error_reqs
    assert scheduled == {"read_b": 0, "read_a": 1}, scheduled  # before the fix: {}


def test_an_encoder_split_deferral_holds_no_admission():
    """A two-way read whose new 40-slot image cannot be allocated while a queued
    request keeps a reference to the cached image a (40 of 64 slots) is deferred
    at the encoder split every step. Before the fix, after eight deferrals it held
    admission, and a plain causal request behind it never ran. Now only the
    token-budget deferral holds admission: the causal request runs at once, and
    the read runs as soon as the reference goes."""
    scheduler = create_scheduler(
        max_num_batched_tokens=256, max_num_seqs=8, enable_chunked_prefill=True
    )
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    scheduler.add_request(_mm_request_images("once", 48, [("image-a", 0, 40)], None))
    _step(scheduler, scheduler.schedule())  # image a cached, unreferenced
    # a queued request that referenced image a in an earlier step and has not run
    holder = _mm_request_images("holder", 48, [("image-a", 0, 40)], None)
    assert scheduler.encoder_cache_manager.check_and_update_cache(holder, 0)
    scheduler.add_request(
        _mm_request_images("read", 48, [("image-w", 4, 40)], "bidirectional")
    )
    for _ in range(10):
        out = scheduler.schedule()
        assert "read" not in out.num_scheduled_tokens  # 24 free, nothing evictable
        _step(scheduler, out)
    scheduler.add_request(_request("plain", 32, None, max_tokens=2))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens.get("plain") == 32  # before the fix: never
    _step(scheduler, out)
    scheduler.encoder_cache_manager.free_encoder_input(holder, 0)  # the holder ran
    out = scheduler.schedule()
    assert out.num_scheduled_tokens.get("read") == 48
