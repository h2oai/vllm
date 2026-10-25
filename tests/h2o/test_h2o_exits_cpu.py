# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the H2O early-exit heads (vllm/h2o_exits.py): policy parsing, the head math against an
independent reference (the training-side formulas), the exit op's decisions, the scheduler transport, the
request-side prefix-cache exclusion, and that the exit point is traced only when exits are enabled."""

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from vllm import h2o_exits
from vllm.sampling_params import SamplingParams

H, R = 64, 8


@pytest.fixture(autouse=True)
def _clean_exit_state():
    h2o_exits.STEP = None
    yield
    h2o_exits.STEP = None
    h2o_exits.HEADS = None


def sp(extra=None, max_tokens=1):
    return SamplingParams(max_tokens=max_tokens, extra_args=extra)


# ------------------------------------------------------------------ policy
def test_policy_default_full():
    assert h2o_exits.parse_exit_policy(sp()) is None
    assert h2o_exits.parse_exit_policy(sp({"exit_policy": "full"})) is None
    assert h2o_exits.parse_exit_policy(sp({"attention_mode": "bidirectional"})) is None


def test_policy_flat_xargs():
    p = h2o_exits.parse_exit_policy(sp({"exit_policy": "auto", "exit_temp": 0.75, "exit_tau_24": 0.5}))
    assert p == {"depths": "auto", "temp": 0.75, "tau": {24: 0.5}}
    p = h2o_exits.parse_exit_policy(sp({"exit_policy": "20"}))
    assert p["depths"] == [20] and p["temp"] == 1.0


def test_policy_rejects():
    with pytest.raises(ValueError):
        h2o_exits.parse_exit_policy(sp({"exit_policy": "fast"}))
    with pytest.raises(ValueError):
        h2o_exits.parse_exit_policy(sp({"exit_policy": "auto", "exit_temp": 0}))


# ------------------------------------------------------------------ heads
@pytest.fixture()
def heads(tmp_path):
    g = torch.Generator().manual_seed(0)
    t = {}
    for k in (20, 24):
        t[f"k{k}.A.weight"] = torch.randn(R, H, generator=g) * 0.1
        t[f"k{k}.B.weight"] = torch.randn(H, R, generator=g) * 0.1
        t[f"k{k}.B.bias"] = torch.randn(H, generator=g) * 0.01
        t[f"k{k}.agree.0.weight"] = torch.randn(32, H + 3, generator=g) * 0.1
        t[f"k{k}.agree.0.bias"] = torch.zeros(32)
        t[f"k{k}.agree.2.weight"] = torch.randn(1, 32, generator=g) * 0.1
        t[f"k{k}.agree.2.bias"] = torch.zeros(1)
    meta = {"layers": [20, 24], "tau": {"20": 0.6, "24": 0.4}, "auto": [20, 24], "eps": 1e-6}
    path = tmp_path / "exits.lightning"
    save_file(t, str(path), metadata={"h2o_exits": json.dumps(meta)})
    norm_w = torch.randn(H, generator=g) * 0.1
    lm_head = torch.randn(50, H, generator=g).to(torch.bfloat16)
    return h2o_exits.ExitHeads(str(path), H, norm_w, lm_head, torch.device("cpu")), t, norm_w, lm_head


def ref_normed(t, k, norm_w, x):
    """Training-side E1 (train_exits.py Exit 'E1') + HF Qwen3_5RMSNorm, written independently."""
    x = x.float()
    z = x + F.linear(F.linear(x, t[f"k{k}.A.weight"]), t[f"k{k}.B.weight"], t[f"k{k}.B.bias"])
    var = z.pow(2).mean(-1, keepdim=True)
    return z * torch.rsqrt(var + 1e-6) * (1.0 + norm_w.float())


def test_normed_matches_reference(heads):
    h, t, norm_w, _ = heads
    x = torch.randn(5, H)
    for k in (20, 24):
        assert torch.allclose(h.normed(k, x), ref_normed(t, k, norm_w, x), atol=1e-6)


def test_e1_zero_B_equals_plain_norm(heads):
    h, t, norm_w, _ = heads
    h.p[24]["B.weight"].zero_(); h.p[24]["B.bias"].zero_()
    x = torch.randn(3, H)
    plain = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + norm_w)
    assert torch.equal(h.normed(24, x), plain)


def test_agreement_matches_reference(heads):
    h, t, _, lm = heads
    hn = torch.randn(2, H)
    ids = [[1, 2, 3], [7, 9]]
    temps = torch.tensor([0.75, 0.8])
    got = h.agreement(20, hn, ids, temps)
    exp = []
    for i in range(2):
        lg = hn[i] @ lm[ids[i]].float().T
        p = torch.softmax(lg / temps[i], -1)
        s = p.sort(descending=True).values
        f = torch.cat([hn[i], torch.stack([s[0], s[1], s[0] - s[1]])])
        z = F.gelu(f @ t["k20.agree.0.weight"].T + t["k20.agree.0.bias"]) @ t["k20.agree.2.weight"].T + t["k20.agree.2.bias"]
        exp.append(torch.sigmoid(z)[0])
    assert torch.allclose(got, torch.stack(exp), atol=1e-6)


# ------------------------------------------------------------------ the exit op
def run_op(k, hidden, residual, flag):
    h2o_exits.h2o_exit_point(hidden, residual, k, flag)


def test_op_noop_without_step(heads):
    h2o_exits.HEADS = heads[0]; h2o_exits.STEP = None
    hid, res, flag = torch.randn(6, H), torch.randn(6, H), torch.zeros(1, dtype=torch.int32)
    h0, r0 = hid.clone(), res.clone()
    run_op(20, hid, res, flag)
    assert torch.equal(hid, h0) and torch.equal(res, r0) and int(flag) == 0


def make_step(n, tau, allowed=True):
    return h2o_exits.ExitStep(
        rows=torch.arange(n) * 2 + 1, label_ids=[[1, 2, 3]] * n, temps=torch.full((n,), 0.75),
        allowed={20: torch.full((n,), allowed), 24: torch.full((n,), True)},
        tau={20: torch.full((n,), tau), 24: torch.full((n,), tau)})


def test_op_exits_when_all_agree(heads):
    h, t, norm_w, _ = heads
    h2o_exits.HEADS = h
    h2o_exits.STEP = make_step(3, tau=0.0)
    hid, res, flag = torch.randn(6, H), torch.randn(6, H), torch.zeros(1, dtype=torch.int32)
    h0, r0 = hid.clone(), res.clone()
    with pytest.raises(h2o_exits.H2OExit) as e:
        run_op(20, hid, res, flag)
    assert e.value.k == 20 and int(flag) == 20
    x = (h0 + r0)[torch.tensor([1, 3, 5])]
    assert torch.allclose(e.value.hidden.float(), ref_normed(t, 20, norm_w, x), atol=1e-6)
    assert torch.equal(hid, h0) and torch.equal(res, r0)          # the trunk's tensors are untouched


def test_op_continues_when_one_fails(heads):
    h2o_exits.HEADS = heads[0]
    h2o_exits.STEP = make_step(3, tau=2.0)                       # nobody can reach 2.0
    hid, res, flag = torch.randn(6, H), torch.randn(6, H), torch.zeros(1, dtype=torch.int32)
    run_op(20, hid, res, flag)
    assert int(flag) == 0 and h2o_exits.STEP.tried == [20]


def test_op_respects_allowed(heads):
    h2o_exits.HEADS = heads[0]
    h2o_exits.STEP = make_step(2, tau=0.0, allowed=False)        # k=20 not in these requests' depths
    run_op(20, torch.randn(4, H), torch.randn(4, H), torch.zeros(1, dtype=torch.int32))
    with pytest.raises(h2o_exits.H2OExit):
        run_op(24, torch.randn(4, H), torch.randn(4, H), torch.zeros(1, dtype=torch.int32))


# ------------------------------------------------------------------ transport + request
def test_scheduler_transport():
    from vllm.v1.core.sched.scheduler import _h2o_with_depth

    class O:  # minimal model runner output
        h2o_depth_used = {"a": 20}
    assert _h2o_with_depth(None, O(), "a") == {"h2o_depth_used": 20}
    assert _h2o_with_depth({"x": 1}, O(), "a") == {"x": 1, "h2o_depth_used": 20}
    assert _h2o_with_depth(None, O(), "b") is None
    O.h2o_depth_used = None
    assert _h2o_with_depth({"x": 1}, O(), "a") == {"x": 1}


def test_request_policy_skips_prefix_cache():
    from vllm.v1.request import Request

    hasher = lambda r: [b"h"] * (len(r.all_token_ids) // 4)  # noqa: E731
    r = Request("r1", list(range(16)), sp({"exit_policy": "auto"}), None, block_hasher=hasher)
    assert r.exit_policy is not None and r.block_hashes == []
    r2 = Request("r2", list(range(16)), sp(), None, block_hasher=hasher)
    assert r2.exit_policy is None and len(r2.block_hashes) == 4
    with pytest.raises(ValueError):
        Request("r3", list(range(16)), sp({"exit_policy": "auto"}, max_tokens=4), None, block_hasher=hasher)


# ------------------------------------------------------------------ tracing
# The op is registered for the platform's dispatch key (CUDA on a GPU build); give it a CPU kernel here so the
# miniature loop runs on a CPU-only test host. The kernel is the same Python function.
try:
    torch.library.impl("vllm::h2o_exit_point", "CPU")(h2o_exits.h2o_exit_point)
except RuntimeError:
    pass

class Loop(torch.nn.Module):
    """A miniature of Qwen3NextModel.forward's layer loop with the same exit hook."""

    def __init__(self, layers):
        super().__init__()
        self.lin = torch.nn.ModuleList([torch.nn.Linear(H, H) for _ in range(layers)])
        self.h2o_exit_layers = h2o_exits.exit_layers_from_env()
        self.register_buffer("h2o_exit_flag", torch.zeros(1, dtype=torch.int32), persistent=False)

    def forward(self, x):
        residual = x
        for i, lin in enumerate(self.lin):
            x = lin(x + residual)
            if i + 1 in self.h2o_exit_layers:
                torch.ops.vllm.h2o_exit_point(x, residual, i + 1, self.h2o_exit_flag)
        return x


def traced_ops(model):
    seen = []

    def backend(gm, example_inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    torch._dynamo.reset()
    torch.compile(model, backend=backend, fullgraph=True)(torch.randn(3, H))
    return seen


def test_exit_point_traced_only_when_enabled(monkeypatch):
    monkeypatch.delenv("VLLM_H2O_EXITS", raising=False)
    assert not any("h2o_exit_point" in s for s in traced_ops(Loop(24)))
    monkeypatch.setenv("VLLM_H2O_EXITS", "1")
    monkeypatch.setenv("VLLM_H2O_EXIT_LAYERS", "20,24")
    ops = [s for s in traced_ops(Loop(24)) if "h2o_exit_point" in s]
    assert len(ops) == 2                                          # kept (not dead-code eliminated), once per exit


def test_exit_point_is_a_splitting_op():
    from vllm.config.compilation import CompilationConfig
    assert "vllm::h2o_exit_point" in CompilationConfig._attention_ops


def test_disabled_loop_is_numerically_identical(monkeypatch):
    """Same weights, exits off vs on with no exit step: identical outputs."""
    monkeypatch.delenv("VLLM_H2O_EXITS", raising=False)
    torch.manual_seed(1); a = Loop(24)
    monkeypatch.setenv("VLLM_H2O_EXITS", "1")
    b = Loop(24); b.load_state_dict(a.state_dict())
    h2o_exits.STEP = None
    x = torch.randn(5, H)
    assert torch.equal(a(x), b(x))
