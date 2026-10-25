# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``no_full`` dispatch: a batch that must not replay a FULL cudagraph (a
bidirectional request, whose per-sequence causal flag a FULL graph would bake
in) still gets the PIECEWISE graph of its size; with data parallelism it runs
eagerly so the ranks' graph modes cannot disagree."""

import pytest

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

pytestmark = pytest.mark.cpu_test

FULL = BatchExecutionDescriptor(
    cg_mode=CUDAGraphMode.FULL, num_tokens=128, num_reqs=128, uniform_token_count=1
)
PIECEWISE = BatchExecutionDescriptor(
    cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=128, num_reqs=None
)


def _manager() -> CudaGraphManager:
    # only the dispatch state; no capture
    m = CudaGraphManager.__new__(CudaGraphManager)
    m._graphs_captured = True
    m._candidates = {(128, 0): [FULL, PIECEWISE]}
    m._lora_dispatch_map = {}
    m._max_lora_case = 0
    return m


def test_dispatch_prefers_full_unless_no_full():
    m = _manager()
    assert m.dispatch(100, 128, 1, num_active_loras=0) == FULL
    assert m.dispatch(100, 128, 1, num_active_loras=0, no_full=True) == PIECEWISE
    # a prefill batch never matched FULL anyway
    assert m.dispatch(2, 128, None, num_active_loras=0) == PIECEWISE
    assert m.dispatch(2, 128, None, num_active_loras=0, no_full=True) == PIECEWISE


def test_dispatch_cg_and_sync_dp_no_full():
    m = _manager()
    desc, sync = dispatch_cg_and_sync_dp(m, 2, 128, None, 1, 0, no_full=True)
    assert desc == PIECEWISE and sync is None
    desc, _ = dispatch_cg_and_sync_dp(m, 100, 128, 1, 1, 0, no_full=True)
    assert desc == PIECEWISE
    desc, _ = dispatch_cg_and_sync_dp(m, 100, 128, 1, 1, 0)
    assert desc == FULL
    # need_eager still wins
    desc, _ = dispatch_cg_and_sync_dp(m, 2, 128, None, 1, 0, need_eager=True)
    assert desc.cg_mode == CUDAGraphMode.NONE
