# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN prefill CUDA graphs and KV cache (re)binding. CPU only.

CUDA-graph memory profiling captures against a throwaway minimal KV cache
(one block per sequence) and frees it; a GDN prefill graph captured there
bakes those tensors into its kernels (the ``ssm_state[state_slots]`` gather
keeps the minimal block count as its bound). Replaying it against the real
cache reads freed memory and asserts on any block id past that count. The
registry must therefore forget its graphs when the layers' KV caches are
re-bound, and ``run`` must refuse a graph captured against another binding.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.prefill_graph import GDNPrefillGraphs

pytestmark = pytest.mark.cpu_test


def _registry() -> GDNPrefillGraphs:
    return GDNPrefillGraphs(max_tokens=2048, max_seqs=16, precapture=False)


def _layer(prefix: str, num_blocks: int) -> SimpleNamespace:
    """A layer bound to a KV cache of ``num_blocks`` state blocks (conv state,
    SSM state), the two tensors a GDN layer unpacks from its page view."""
    conv = torch.zeros((num_blocks, 4, 3), dtype=torch.bfloat16)
    ssm = torch.zeros((num_blocks, 2, 4, 4), dtype=torch.float32)
    return SimpleNamespace(prefix=prefix, kv_cache=(conv, ssm))


def _populate(reg: GDNPrefillGraphs, prefix: str) -> None:
    """Pretend a few shapes were captured for ``prefix``."""
    reg.graphs[(prefix, "body", 8, 64)] = object()
    reg.graphs[(prefix, "body", 8, 128)] = object()
    reg._graph_buffers[(prefix, "body", 8, 64)] = []
    reg.buckets[(8, 64)] = object()
    reg._io[64] = ({}, {})
    reg._precaptured.add((prefix, "body"))
    reg._shapes_seen.add((8, 64))
    reg._fi_sizes[(32, 100, 8)] = 128
    reg._step_metadata = object()
    reg._step_bucket = object()


def test_reset_forgets_every_capture_and_buffer():
    reg = _registry()
    layer = _layer("layers.0.linear_attn", num_blocks=8)
    assert reg.check_kv_binding(layer)  # records the binding
    _populate(reg, layer.prefix)

    reg.reset()

    assert reg.graphs == {}
    assert reg._graph_buffers == {}
    assert reg.buckets == {}
    assert reg._io == {}
    assert reg._precaptured == set()
    assert reg._shapes_seen == set()
    assert reg._fi_sizes == {}
    assert reg._step_metadata is None
    assert reg._step_bucket is None
    assert reg._kv_bound == {}
    # a reset is not a disablement: the next eligible prefill captures again
    assert reg.disabled_reason is None


def test_same_binding_keeps_graphs():
    reg = _registry()
    layer = _layer("layers.0.linear_attn", num_blocks=8)
    assert reg.check_kv_binding(layer)
    _populate(reg, layer.prefix)
    # the same tensors again (another step): nothing is dropped
    assert reg.check_kv_binding(layer)
    assert len(reg.graphs) == 2


def test_rebinding_drops_graphs_for_the_whole_registry():
    """The profiling pass binds a minimal cache (8 blocks here), graphs get
    captured, the real cache (many more blocks) replaces it: every graph goes,
    for every layer, and the new binding is the one recorded."""
    reg = _registry()
    l0 = _layer("layers.0.linear_attn", num_blocks=8)
    l1 = _layer("layers.1.linear_attn", num_blocks=8)
    assert reg.check_kv_binding(l0) and reg.check_kv_binding(l1)
    _populate(reg, l0.prefix)
    _populate(reg, l1.prefix)
    assert len(reg.graphs) == 4

    # the real KV cache: new tensors, more blocks
    real0 = _layer("layers.0.linear_attn", num_blocks=3391)
    assert reg.check_kv_binding(real0) is False
    assert reg.graphs == {} and reg.buckets == {} and reg._precaptured == set()
    assert reg._kv_bound == {real0.prefix: reg._kv_signature(real0)}
    # the next step on the same (real) binding is accepted
    assert reg.check_kv_binding(real0)
    # layer 1, bound to the real cache in the same rebinding, is simply recorded
    real1 = _layer("layers.1.linear_attn", num_blocks=3391)
    assert reg.check_kv_binding(real1)
    assert len(reg._kv_bound) == 2


def test_same_shape_different_storage_counts_as_rebinding():
    """A reallocated cache of the same size is still another cache: the
    captured kernels hold the old addresses."""
    reg = _registry()
    layer = _layer("layers.0.linear_attn", num_blocks=16)
    assert reg.check_kv_binding(layer)
    _populate(reg, layer.prefix)
    layer.kv_cache = _layer(layer.prefix, num_blocks=16).kv_cache
    assert reg.check_kv_binding(layer) is False
    assert reg.graphs == {}


def test_unbound_layer_never_captures():
    reg = _registry()
    for kv in (None, (), (torch.zeros(0), torch.zeros(0))):
        layer = SimpleNamespace(prefix="layers.0.linear_attn", kv_cache=kv)
        assert reg.check_kv_binding(layer) is False
        assert reg._kv_bound == {}


def test_profiling_dummy_runs_are_graph_eligible_only_for_small_max_num_seqs():
    """Why the crash needed a small --max-num-seqs: a profiling dummy run of
    S tokens has min(S, max_num_seqs) sequences, and a GDN prefill graph is
    captured only for at most max_seqs (16) prefills of 2+ tokens. With 8 or
    16 sequences the profiling pass captures (and the stale graphs bite);
    with 128 sequences every dummy run is a decode batch (sizes <= 128, one
    token per sequence) or has too many prefills, so nothing is captured
    before the real KV cache exists."""
    reg = _registry()
    for max_num_seqs, expect_capture in ((8, True), (16, True), (128, False)):
        eligible = False
        for size in (16, 24, 32, 64, 128, 256, 512, 1024, 2048):
            num_reqs = min(size, max_num_seqs)
            tokens_per_req = size // num_reqs
            if tokens_per_req < 2:  # a decode batch, never a prefill graph
                continue
            if reg.seq_bucket(num_reqs) is not None and reg.bucket_size(size):
                eligible = True
        assert eligible == expect_capture, max_num_seqs
