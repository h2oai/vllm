# SPDX-License-Identifier: Apache-2.0
"""Per-request attention mode (H2O): a request asking for bidirectional attention through
``vllm_xargs``/``extra_args`` is prefill-only, never reads or writes the prefix cache, is scheduled
whole or deferred, and finishes with an error (not a scheduler crash) when it can never fit a step.
CPU only."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import sha256

from .utils import EOS_TOKEN_ID, create_scheduler


def _request(req_id: str, num_tokens: int, mode: str | None, max_tokens: int = 1, block_size: int = 16):
    extra = {"attention_mode": mode} if mode is not None else None
    sp = SamplingParams(max_tokens=max_tokens, extra_args=extra)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sp,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
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
    assert two_way.block_hashes == []           # never read from or written to the prefix cache


def test_scheduled_whole_or_deferred_and_too_long_is_an_error():
    scheduler = create_scheduler(max_num_batched_tokens=64, max_num_seqs=4, enable_chunked_prefill=True,
                                 enable_prefix_caching=True)
    short = _request("short", 40, "bidirectional")
    long_ok = _request("long_ok", 60, "bidirectional")
    too_long = _request("too_long", 100, "bidirectional")
    for r in (short, long_ok, too_long):
        scheduler.add_request(r)
    out = scheduler.schedule()
    # the first fits whole (40 <= 64); the second (60) would be chunked after it, so it is deferred, not chunked
    assert out.num_scheduled_tokens.get("short") == 40
    assert "long_ok" not in out.num_scheduled_tokens
    # the third can never fit a step: finished with an error for that request, the scheduler alive
    assert too_long.status == RequestStatus.FINISHED_ERROR
    assert "too_long" not in out.num_scheduled_tokens
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens.get("long_ok") == 60    # whole, in a later step
