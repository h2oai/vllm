# SPDX-License-Identifier: Apache-2.0
"""H2O residual router, gather-first capture (CPU).

The capture returns, per request, the selected layers' residual at the request's last prompt token. Before the fix
the model kept every selected layer's residual for EVERY token of the step (8.22 GiB on a ~100k-token step of a 27B);
now it gathers the staged rows first. These tests check that
  1. the returned bytes are bit-identical to the old full-step implementation (ragged batches, chunked prefill with
     the last prompt token in / not in the step, decode steps, a preempted request recomputed with its outputs,
     bf16 / fp16 / fp32, with and without a residual), through the V1 runner's own staging / pooling methods and
     through torch.compile (dynamic token dim; the row buffer is read at run time, not baked in; stale rows past the
     step's token count are clamped);
  2. every tensor the capture path allocates is bounded by max_rows x hidden (O(requests), not O(tokens)).
"""

import copy
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.v1.worker.h2o_residual import (
    H2O_ROWS_ATTR,
    collects_through_mixin,
    enable_row_gather,
    gather_vectors,
)

HIDDEN = 48
NUM_LAYERS = 8
LAYERS = (2, 3, 5, 7)


# ----------------------------------------------------------------------------- a toy decoder with the mixin
def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class _QwenLikeLayer(nn.Module):
    """Residual pattern of Qwen3-Next / Qwen3.5 (pre-norm, the residual add inside the next norm): returns
    (hidden_states, residual), residual not None. As in the served graphs, the hidden output is a GEMM output and
    the residual is used again after a GEMM, so both are materialised tensors at the capture point."""

    def __init__(self, i: int, dtype: torch.dtype):
        super().__init__()
        g = torch.Generator().manual_seed(100 + i)
        self.w1 = nn.Parameter(torch.randn(HIDDEN, HIDDEN, generator=g).to(dtype) / HIDDEN**0.5, requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(HIDDEN, HIDDEN, generator=g).to(dtype) / HIDDEN**0.5, requires_grad=False)

    def forward(self, h, r):
        r = h if r is None else h + r
        x = _rms(r) @ self.w1                       # "attention"
        r = x + r
        return torch.nn.functional.silu(_rms(r)) @ self.w2, r      # "MLP" output, residual


class _GemmaLikeLayer(_QwenLikeLayer):
    """Gemma-4 pattern: the residual is folded into hidden_states (norm, add, scalar), residual stays None."""

    def forward(self, h, r):
        h = _rms(h @ self.w1) + h
        return (_rms(torch.nn.functional.silu(h) @ self.w2) + h) * 0.75, None


class ToyModel(nn.Module, EagleModelMixin):
    def __init__(self, dtype=torch.bfloat16, gemma=False):
        super().__init__()
        cls = _GemmaLikeLayer if gemma else _QwenLikeLayer
        self.layers = nn.ModuleList([cls(i, dtype) for i in range(NUM_LAYERS)])
        self.start_layer = 0

    def forward(self, x):
        h, r = x, None
        aux = self._maybe_add_hidden_state([], 0, h, r)
        for i, layer in enumerate(self.layers):
            h, r = layer(h, r)
            self._maybe_add_hidden_state(aux, i + 1, h, r)
        return h, aux


def _old_maybe_add_hidden_state(self, aux_hidden_states, layer_idx, hidden_states, residual):
    """The pre-fix EagleModelMixin._maybe_add_hidden_state (a36449a18), verbatim."""
    if layer_idx in self.aux_hidden_state_layers:
        value = hidden_states + residual if residual is not None else hidden_states
        aux_hidden_states.append(value)
    return aux_hidden_states


def _old_v1_pool(aux_hidden_states, todo):
    """The pre-fix V1 _h2o_pool_residual body for a computed todo: stack the rows, fp16, bytes."""
    out = {}
    for rid, row in todo:
        vec = torch.stack([h[row] for h in aux_hidden_states], 0)      # [L, d]
        out[rid] = vec.to(torch.float16).cpu().numpy().tobytes()
    return out


def _old_v2_vecs(aux_hidden_states, todo):
    """The pre-fix V2 gather (before the pinned host copy)."""
    return [(rid, torch.stack([h[row] for h in aux_hidden_states], 0).to(torch.float16)) for rid, row in todo]


def _make_model(dtype, gemma, max_rows):
    m = ToyModel(dtype, gemma)
    m._set_aux_hidden_state_layers(LAYERS)
    rows = torch.zeros(max_rows, dtype=torch.int64)
    assert enable_row_gather(m, rows)
    assert getattr(m, H2O_ROWS_ATTR) is rows
    return m, rows


def _old_full_aux(model, x):
    """The same weights run through the old (pre-fix) collection: every token of every selected layer kept."""
    ref = copy.deepcopy(model)
    ref._buffers.pop(H2O_ROWS_ATTR, None)
    ref._maybe_add_hidden_state = types.MethodType(_old_maybe_add_hidden_state, ref)
    _, aux = ref(x)
    assert all(a.shape[0] == x.shape[0] for a in aux)
    return aux


# ----------------------------------------------------------------------------- step simulation (V1 runner state)
def _step(prompt_lens, done, sched, flagged=None):
    """A V1-runner-like state for one step: req i has prompt_lens[i], done[i] computed, sched[i] scheduled."""
    n = len(prompt_lens)
    req_ids = [f"r{i}" for i in range(n)]
    flagged = set(req_ids) if flagged is None else flagged
    qsl = np.zeros(n + 1, dtype=np.int32)
    qsl[1:] = np.cumsum(sched)
    requests = {
        rid: SimpleNamespace(
            num_prompt_tokens=prompt_lens[i], num_computed_tokens=done[i],
            sampling_params=SimpleNamespace(extra_args={"residual_features": rid in flagged}))
        for i, rid in enumerate(req_ids)
    }
    sched_out = SimpleNamespace(num_scheduled_tokens={rid: int(sched[i]) for i, rid in enumerate(req_ids)})
    return req_ids, qsl, requests, sched_out, int(qsl[-1])


def _fake_v1_runner(req_ids, qsl, requests, rows_buf):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    fake = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(req_ids), req_ids=req_ids),
        query_start_loc=SimpleNamespace(np=qsl),
        requests=requests,
        _h2o_rows=rows_buf,
        _h2o_staged=[],
        _h2o_resid_ready={},
    )
    for name in ("_h2o_todo", "_h2o_stage_rows", "_h2o_pool_residual"):
        setattr(fake, name, types.MethodType(getattr(GPUModelRunner, name), fake))
    fake._h2o_wants_features = GPUModelRunner._h2o_wants_features
    return fake


def _cpu_rows_buffer(gpu_rows):
    """A CpuGpuBuffer-shaped holder whose .gpu is the model's registered buffer (CPU here)."""
    holder = SimpleNamespace(gpu=gpu_rows, np=np.zeros(gpu_rows.shape[0], dtype=np.int64))

    def copy_to_gpu(n=None):
        n = gpu_rows.shape[0] if n is None else n
        return gpu_rows[:n].copy_(torch.from_numpy(holder.np[:n]))

    holder.copy_to_gpu = copy_to_gpu
    return holder


SCENARIOS = {
    # ragged prefills, all complete in this step
    "ragged_full_prefills": ([5, 17, 1, 33], [0, 0, 0, 0], [5, 17, 1, 33]),
    # chunked prefill: r0 mid-prompt (last token NOT in step), r1 final chunk (in step), r2 first chunk
    "chunked_prefill": ([40, 30, 64], [0, 20, 0], [16, 10, 8]),
    # decode steps only (prompts complete earlier): nothing to return
    "decode_only": ([7, 9, 11], [7, 9, 12], [1, 1, 1]),
    # mixed: decode + final chunk + full prefill + cached-prefix hit leaving 1 token
    "mixed": ([12, 50, 6, 100], [13, 44, 0, 99], [1, 6, 6, 1]),
    # preempted request recomputed with its outputs (scheduled past the prompt): row = prompt end, not step end
    "preempted_resume": ([10, 8], [0, 0], [14, 8]),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("gemma", [False, True], ids=["qwen_residual", "gemma_no_residual"])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_v1_runner_bytes_identical_to_old(dtype, gemma, scenario):
    prompt_lens, done, sched = SCENARIOS[scenario]
    max_rows = 6
    model, rows = _make_model(dtype, gemma, max_rows)
    rows.fill_(10_000)                                   # stale garbage from an earlier, longer step
    req_ids, qsl, requests, sched_out, T = _step(prompt_lens, done, sched)
    x = torch.randn(T, HIDDEN, generator=torch.Generator().manual_seed(T)).to(dtype)

    fake = _fake_v1_runner(req_ids, qsl, requests, _cpu_rows_buffer(rows))
    todo = fake._h2o_todo(sched_out)
    fake._h2o_stage_rows(sched_out)
    _, aux = model(x)
    assert all(a.shape == (max_rows, HIDDEN) for a in aux)
    fake._h2o_pool_residual(sched_out, aux)
    new = fake._h2o_resid_ready

    old = _old_v1_pool(_old_full_aux(model, x), todo)
    assert set(new) == set(old)
    for rid in old:
        assert new[rid] == old[rid], rid
    expected_rids = {
        rid for i, rid in enumerate(req_ids)
        if done[i] < prompt_lens[i] <= done[i] + sched[i]
    }
    assert set(old) == expected_rids
    # V2's gather (same todo, same model output) gives the same fp16 tensors
    v2_new = gather_vectors(aux, todo, gathered=True)
    v2_old = _old_v2_vecs(_old_full_aux(model, x), todo)
    assert [r for r, _ in v2_new] == [r for r, _ in v2_old]
    for (_, a), (_, b) in zip(v2_new, v2_old):
        assert a.dtype == b.dtype == torch.float16 and torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_v1_runner_unflagged_requests_and_mismatch_guard():
    model, rows = _make_model(torch.bfloat16, False, 4)
    req_ids, qsl, requests, sched_out, T = _step([5, 6, 7], [0, 0, 0], [5, 6, 7], flagged={"r1"})
    x = torch.randn(T, HIDDEN).to(torch.bfloat16)
    fake = _fake_v1_runner(req_ids, qsl, requests, _cpu_rows_buffer(rows))
    fake._h2o_stage_rows(sched_out)
    assert fake._h2o_staged == ["r1"]
    _, aux = model(x)
    fake._h2o_pool_residual(sched_out, aux)
    assert set(fake._h2o_resid_ready) == {"r1"}
    assert fake._h2o_resid_ready["r1"] == _old_v1_pool(_old_full_aux(model, x), [("r1", 5 + 5)])["r1"]
    # a pool without the matching staging returns nothing rather than a wrong vector
    fake._h2o_resid_ready = {}
    fake._h2o_staged = ["r0"]
    fake._h2o_pool_residual(sched_out, aux)
    assert fake._h2o_resid_ready == {}


def test_random_steps_bytes_identical():
    """Random ragged batches: chunked prefills ending in / before the step, decodes, many requests."""
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = int(rng.integers(1, 12))
        prompt = rng.integers(1, 200, n)
        done = np.array([int(rng.integers(0, p + 3)) for p in prompt])
        sched = np.array([
            1 if d >= p else int(rng.integers(1, p - d + 1)) for d, p in zip(done, prompt)
        ])
        model, rows = _make_model(torch.bfloat16, bool(trial % 2), n)
        req_ids, qsl, requests, sched_out, T = _step(prompt.tolist(), done.tolist(), sched.tolist())
        x = torch.randn(T, HIDDEN, generator=torch.Generator().manual_seed(trial)).to(torch.bfloat16)
        fake = _fake_v1_runner(req_ids, qsl, requests, _cpu_rows_buffer(rows))
        todo = fake._h2o_todo(sched_out)
        fake._h2o_stage_rows(sched_out)
        _, aux = model(x)
        fake._h2o_pool_residual(sched_out, aux)
        assert fake._h2o_resid_ready == _old_v1_pool(_old_full_aux(model, x), todo)


# ----------------------------------------------------------------------------- memory: O(requests), not O(tokens)
class _AllocRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.numels: list[int] = []

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(t, torch.Tensor):
                self.numels.append(t.numel())
        return out


@pytest.mark.parametrize("gemma", [False, True])
def test_capture_allocations_bounded_by_rows_not_tokens(gemma):
    T, max_rows = 20_000, 8                     # a long step; few requests
    model, rows = _make_model(torch.bfloat16, gemma, max_rows)
    rows[:3] = torch.tensor([4_999, 12_345, T - 1])
    x = torch.randn(T, HIDDEN).to(torch.bfloat16)
    h, r = x, None
    seen: list[int] = []
    rec_new = _AllocRecorder()
    aux = []
    with rec_new:
        aux = model._maybe_add_hidden_state(aux, 0, h, r)
    seen += rec_new.numels
    for i, layer in enumerate(model.layers):
        h, r = layer(h, r)
        rec = _AllocRecorder()
        with rec:
            model._maybe_add_hidden_state(aux, i + 1, h, r)
        seen += rec.numels
    bound = max_rows * HIDDEN
    assert seen, "the capture path allocated nothing?"
    assert max(seen) <= bound, (max(seen), bound)
    # per selected layer: the clamped rows [R], the gathered hidden (and residual) [R, d], their sum [R, d]
    assert sum(seen) <= len(LAYERS) * (3 * bound + max_rows)
    assert len(aux) == len(LAYERS) and all(a.shape == (max_rows, HIDDEN) for a in aux)
    # the per-request vectors: [L, d] each
    rec = _AllocRecorder()
    with rec:
        vecs = gather_vectors(aux, [("a", 0), ("b", 1), ("c", 2)], gathered=True)
    assert max(rec.numels) <= len(LAYERS) * HIDDEN
    assert [v.shape for _, v in vecs] == [(len(LAYERS), HIDDEN)] * 3

    # the old path, for contrast: a [T, d] tensor per selected layer (Qwen pattern; the Gemma pattern aliased the
    # full layer output, which then stays alive for the whole step)
    if not gemma:
        h, r = x, None
        rec_old = _AllocRecorder()
        old_aux = []
        for i, layer in enumerate(model.layers):
            h, r = layer(h, r)
            with rec_old:
                _old_maybe_add_hidden_state(model, old_aux, i + 1, h, r)
        assert max(rec_old.numels) == T * HIDDEN


def test_capture_memory_independent_of_step_tokens():
    """Bytes the capture path allocates are the same for a 1k- and a 64k-token step."""
    def captured_bytes(T):
        model, rows = _make_model(torch.float32, False, 4)
        x = torch.randn(T, HIDDEN)
        h, r = x, None
        total = 0
        aux = []
        for i, layer in enumerate(model.layers):
            h, r = layer(h, r)
            rec = _AllocRecorder()
            with rec:
                model._maybe_add_hidden_state(aux, i + 1, h, r)
            total += sum(rec.numels)
        return total
    assert captured_bytes(1_000) == captured_bytes(64_000)


# ----------------------------------------------------------------------------- torch.compile
@pytest.mark.parametrize("gemma", [False, True], ids=["qwen_residual", "gemma_no_residual"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_reads_buffer_at_run_time_dynamic_tokens(backend, gemma):
    """Old and new collection, both compiled with the same backend (as served: the capture runs inside the
    compiled graph), give bit-identical vectors; the row buffer is read at run time (its contents change between
    calls of one compiled graph); stale slots past the step's token count are clamped, never out of bounds."""
    torch._dynamo.reset()
    max_rows = 5
    model, rows = _make_model(torch.bfloat16, gemma, max_rows)
    ref = copy.deepcopy(model)
    ref._buffers.pop(H2O_ROWS_ATTR)
    ref._maybe_add_hidden_state = types.MethodType(_old_maybe_add_hidden_state, ref)
    compiled = torch.compile(model, backend=backend, dynamic=True, fullgraph=True)
    compiled_ref = torch.compile(ref, backend=backend, dynamic=True, fullgraph=True)
    for T, staged in [(37, [36, 0, 11]), (64, [63, 5]), (9, [8, 3, 2, 1, 0]), (300, [299, 150, 7]), (3, [2])]:
        x = torch.randn(T, HIDDEN, generator=torch.Generator().manual_seed(T)).to(torch.bfloat16)
        rows[: len(staged)] = torch.tensor(staged)      # slots past len(staged) keep stale rows (maybe > T)
        _, aux = compiled(x)
        _, aux_old = compiled_ref(x)
        assert all(a.shape == (max_rows, HIDDEN) for a in aux)
        assert all(a.shape == (T, HIDDEN) for a in aux_old)
        todo = [(f"r{k}", row) for k, row in enumerate(staged)]
        new = gather_vectors(aux, todo, gathered=True)
        old = _old_v2_vecs(aux_old, todo)
        for (_, a), (_, b) in zip(new, old):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        if backend == "aot_eager":                      # no fusion: also equal to the eager old path
            eager_old = _old_v2_vecs(_old_full_aux(model, x), todo)
            for (_, a), (_, b) in zip(new, eager_old):
                assert torch.equal(a.view(torch.int16), b.view(torch.int16))


# ----------------------------------------------------------------------------- enabling
def test_enable_row_gather_refuses_models_it_cannot_gather():
    class Custom(nn.Module, EagleModelMixin):           # collects aux states itself (like deepseek_v2)
        def forward(self, x):
            return x, [x]

    m = Custom()
    m._set_aux_hidden_state_layers((1,))
    assert not enable_row_gather(m, torch.zeros(4, dtype=torch.int64))
    assert getattr(m, H2O_ROWS_ATTR, None) is None

    sp = ToyModel()
    sp._set_aux_hidden_state_layers((1,))
    sp.use_sequence_parallel = True                     # aux states chunked by token across ranks
    assert not enable_row_gather(sp, torch.zeros(4, dtype=torch.int64))

    plain = ToyModel()                                  # no aux layers set: nothing to gather
    assert not enable_row_gather(plain, torch.zeros(4, dtype=torch.int64))

    outer = nn.Module()                                 # an outer wrapper: the inner model gets the buffer
    outer.model = ToyModel()
    outer.model._set_aux_hidden_state_layers(LAYERS)
    buf = torch.zeros(4, dtype=torch.int64)
    assert enable_row_gather(outer, buf)
    assert outer.model.h2o_aux_rows is buf and "h2o_aux_rows" not in outer.model.state_dict()


@pytest.mark.parametrize("modpath,clsname", [
    ("vllm.model_executor.models.qwen3_next", "Qwen3NextModel"),
    ("vllm.model_executor.models.qwen3_5", "Qwen3_5Model"),
    ("vllm.model_executor.models.gemma4", "Gemma4Model"),
])
def test_served_families_collect_through_the_mixin(modpath, clsname):
    import importlib
    try:
        mod = importlib.import_module(modpath)
    except Exception as e:                              # pragma: no cover - platform-dependent imports
        pytest.skip(f"cannot import {modpath} here: {e!r}")
    cls = getattr(mod, clsname)
    assert issubclass(cls, EagleModelMixin)
    assert collects_through_mixin(cls)


# ----------------------------------------------------------------------------- review fixes (rel27, 11-22)
def _pc(pp=1, pcp=1, tp=1, ubatch=False):
    return SimpleNamespace(pipeline_parallel_size=pp, prefill_context_parallel_size=pcp, tensor_parallel_size=tp,
                           use_ubatching=ubatch)


def _cc(enable_sp=False):
    return SimpleNamespace(pass_config=SimpleNamespace(enable_sp=enable_sp))


def test_gather_off_reasons():
    from vllm.v1.worker.h2o_residual import gather_off_reason
    assert gather_off_reason(_pc(), _cc()) is None
    assert gather_off_reason(_pc(tp=4), _cc()) is None                  # TP without SP: full tokens per rank
    assert gather_off_reason(_pc(tp=1), _cc(enable_sp=True)) is None    # SP pass is a no-op at TP=1
    assert "sequence parallel" in gather_off_reason(_pc(tp=2), _cc(enable_sp=True))
    assert "pipeline" in gather_off_reason(_pc(pp=2), _cc())
    assert "context parallel" in gather_off_reason(_pc(pcp=2), _cc())
    assert "microbatching" in gather_off_reason(_pc(ubatch=True), _cc())


def test_env_vars_are_compile_factors(monkeypatch):
    import vllm.envs as envs
    from vllm.config.utils import hash_factors
    monkeypatch.delenv("VLLM_H2O_RESIDUAL_LAYERS", raising=False)
    monkeypatch.delenv("VLLM_H2O_RESIDUAL_ALWAYS", raising=False)
    off = envs.compile_factors()
    assert "VLLM_H2O_RESIDUAL_LAYERS" in off and "VLLM_H2O_RESIDUAL_ALWAYS" in off
    h_off = hash_factors(off)
    monkeypatch.setenv("VLLM_H2O_RESIDUAL_LAYERS", "21,32,43,53")
    h_on = hash_factors(envs.compile_factors())
    monkeypatch.setenv("VLLM_H2O_RESIDUAL_LAYERS", "20,30,40,50")
    h_other = hash_factors(envs.compile_factors())
    monkeypatch.setenv("VLLM_H2O_RESIDUAL_ALWAYS", "1")
    h_always = hash_factors(envs.compile_factors())
    assert len({h_off, h_on, h_other, h_always}) == 4


class _Outer(nn.Module):
    """An outer model with the setter the runners call, over the toy inner model."""

    def __init__(self, inner=None):
        super().__init__()
        self.model = inner if inner is not None else ToyModel()

    def set_aux_hidden_state_layers(self, layers):
        self.model._set_aux_hidden_state_layers(layers)


def _fake_runner_for_enable(runner_cls, *, spec_method=None, spec_aux=False, pc=None, cc=None, model=None):
    fake = SimpleNamespace(
        model=model if model is not None else _Outer(),
        h2o_residual_layers=LAYERS,
        use_aux_hidden_state_outputs=spec_aux,
        speculative_config=None if spec_method is None else SimpleNamespace(method=spec_method),
        parallel_config=pc or _pc(),
        compilation_config=cc or _cc(),
        max_num_reqs=6,
        device=torch.device("cpu"),
        use_pp=(pc or _pc()).pipeline_parallel_size > 1,
        _h2o_rows=None, _h2o_rows_gpu=None, _h2o_rows_pool=None, _h2o_aux_only=False,
    )
    fake.vllm_config = SimpleNamespace(compilation_config=fake.compilation_config)
    for name in ("_h2o_enable_residual_layers", "_drafter_reads_aux", "_drafter_aux"):
        if hasattr(runner_cls, name):
            setattr(fake, name, types.MethodType(getattr(runner_cls, name), fake))
    if hasattr(runner_cls, "_make_buffer"):            # V1: a CpuGpuBuffer, unpinned (no CUDA here)
        from vllm.v1.utils import CpuGpuBuffer
        fake._make_buffer = lambda *size, dtype, numpy=True: CpuGpuBuffer(
            *size, dtype=dtype, device=torch.device("cpu"), pin_memory=False, with_numpy=numpy)
    return fake


def _runner_classes():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1
    out = [pytest.param(V1, id="v1")]
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2
        out.append(pytest.param(V2, id="v2"))
    except Exception as e:                              # pragma: no cover
        out.append(pytest.param(None, id="v2", marks=pytest.mark.skip(reason=f"V2 runner import: {e!r}")))
    return out


@pytest.mark.parametrize("runner_cls", _runner_classes())
@pytest.mark.parametrize("method", ["eagle3", "dflash", "extract_hidden_states"])
def test_drafter_reading_aux_states_is_refused(runner_cls, method):
    fake = _fake_runner_for_enable(runner_cls, spec_method=method, spec_aux=True)
    with pytest.raises(ValueError, match="VLLM_H2O_RESIDUAL_LAYERS"):
        fake._h2o_enable_residual_layers()
    assert fake.model.model.aux_hidden_state_layers == ()   # nothing changed on the model


@pytest.fixture
def _cpu_uva_pool(monkeypatch):
    """V2 stages rows through a UvaBufferPool (pinned/UVA, GPU only); a plain stand-in on CPU."""
    import vllm.v1.worker.gpu.buffer_utils as bu

    class _Pool:
        def __init__(self, size, dtype, max_concurrency=None):
            pass

        def copy_to_gpu(self, x, out=None):
            return out.copy_(torch.as_tensor(x))

    monkeypatch.setattr(bu, "UvaBufferPool", _Pool)


@pytest.mark.parametrize("runner_cls", _runner_classes())
@pytest.mark.parametrize("method", ["mtp", "eagle", "draft_model", None])
def test_hidden_state_drafters_keep_hidden_states(runner_cls, method, _cpu_uva_pool):
    """Drafters that read the target's last hidden states (MTP incl. Gemma-4 / Qwen3.5, EAGLE-1, draft model) run
    with the router: gather-first is on and the drafter is handed no aux states, as without the router."""
    fake = _fake_runner_for_enable(runner_cls, spec_method=method, spec_aux=False)
    fake._h2o_enable_residual_layers()
    assert fake.use_aux_hidden_state_outputs and fake._h2o_aux_only
    rows = fake._h2o_rows.gpu if getattr(fake, "_h2o_rows", None) is not None else fake._h2o_rows_gpu
    assert rows is not None and fake.model.model.h2o_aux_rows is rows
    aux = [torch.zeros(6, HIDDEN)]
    if hasattr(fake, "_drafter_reads_aux"):
        assert fake._drafter_reads_aux() is False           # V1: target_hidden_states = hidden_states[...]
    if hasattr(fake, "_drafter_aux"):
        assert fake._drafter_aux(aux) is None               # V2: propose(..., aux_hidden_states=None)


def test_drafter_call_sites_use_the_h2o_aware_selection():
    import inspect
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1
    src = inspect.getsource(V1.propose_draft_token_ids)
    assert src.count("self._drafter_reads_aux()") == 3
    assert "if self.use_aux_hidden_state_outputs:\n" not in src
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2
    except Exception as e:                              # pragma: no cover
        pytest.skip(f"V2 runner import: {e!r}")
    v2 = inspect.getsource(V2)
    assert v2.count("self._drafter_aux(aux_hidden_states)") == 2


@pytest.mark.parametrize("runner_cls", _runner_classes())
@pytest.mark.parametrize("pc,cc,why", [
    (_pc(pp=2), _cc(), "pipeline"),
    (_pc(tp=2), _cc(enable_sp=True), "sequence parallel"),
    (_pc(ubatch=True), _cc(), "microbatching"),
])
def test_gather_off_logs_a_warning(runner_cls, pc, cc, why, monkeypatch):
    import importlib
    mod = importlib.import_module(runner_cls.__module__)
    calls = {"warning": [], "info": []}
    fake_logger = SimpleNamespace(
        warning=lambda msg, *a: calls["warning"].append(msg % a),
        info=lambda msg, *a: calls["info"].append(msg % a),
    )
    monkeypatch.setattr(mod, "logger", fake_logger)
    fake = _fake_runner_for_enable(runner_cls, pc=pc, cc=cc)
    fake._h2o_enable_residual_layers()
    assert getattr(fake, "_h2o_rows", None) is None and getattr(fake, "_h2o_rows_gpu", None) is None
    assert getattr(fake.model.model, H2O_ROWS_ATTR, None) is None
    assert any("gather-first capture OFF" in m and why in m for m in calls["warning"]), calls


@pytest.mark.parametrize("runner_cls", _runner_classes())
def test_model_without_mixin_collection_logs_a_warning(runner_cls, monkeypatch):
    import importlib
    mod = importlib.import_module(runner_cls.__module__)
    warnings = []
    monkeypatch.setattr(mod, "logger", SimpleNamespace(warning=lambda m, *a: warnings.append(m % a),
                                                       info=lambda m, *a: None))

    class Custom(nn.Module, EagleModelMixin):
        def forward(self, x):
            return x, [x]

    fake = _fake_runner_for_enable(runner_cls, model=_Outer(Custom()))
    fake._h2o_enable_residual_layers()
    assert any("gather-first capture OFF" in m and "mixin" in m for m in warnings)
