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


# ------------------------------------------------------------------ the GDN hook
QKV = 3 * H


@pytest.fixture()
def recon_heads(heads):
    """attach a reconstruction map built like h2o_exits.load: R = diag(1/(1+w_ln)) pinv(W_qkv)."""
    h, t, norm_w, lm = heads
    g = torch.Generator().manual_seed(1)
    w_qkv = torch.randn(QKV, H, generator=g) / H ** 0.5
    ln = torch.randn(H, generator=g) * 0.1
    h.recon = {k: (torch.linalg.pinv(w_qkv) / (ln + 1.0)[:, None]) for k in (20, 24)}
    return h, t, norm_w, w_qkv, ln


def project(x, w_qkv, ln):
    """what the GDN op receives for a residual-stream state x: W_qkv (1 + w_ln) x / rms(x)"""
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (xn * (ln + 1.0)) @ w_qkv.T


def make_step(n, tau, allowed=True):
    return h2o_exits.ExitStep(
        rows=torch.arange(n) * 2 + 1, label_ids=[[1, 2, 3]] * n, temps=torch.full((n,), 0.75),
        allowed={20: torch.full((n,), allowed), 24: torch.full((n,), True)},
        tau={20: torch.full((n,), tau), 24: torch.full((n,), tau)})


def test_reconstruction_recovers_scale_free_state(recon_heads):
    h, _, _, w_qkv, ln = recon_heads
    x = torch.randn(7, H) * 3.0
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(project(x, w_qkv, ln) @ h.recon[20].T, xn, atol=1e-4)


def test_hook_noop_without_step(recon_heads):
    h2o_exits.HEADS = recon_heads[0]; h2o_exits.STEP = None
    q = torch.randn(6, QKV); q0 = q.clone()
    h2o_exits.gdn_hook(20, q)
    assert torch.equal(q, q0)


def test_hook_exits_when_all_agree(recon_heads):
    h, t, norm_w, w_qkv, ln = recon_heads
    h2o_exits.HEADS = h; h2o_exits.STEP = make_step(3, tau=0.0)
    x = torch.randn(6, H)
    q = project(x, w_qkv, ln); q0 = q.clone()
    with pytest.raises(h2o_exits.H2OExit) as e:
        h2o_exits.gdn_hook(20, q)
    assert e.value.k == 20
    xn = x[torch.tensor([1, 3, 5])]
    xn = xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(e.value.hidden.float(), ref_normed(t, 20, norm_w, xn), atol=1e-4)
    assert torch.equal(q, q0)                                     # the op's input is untouched


def test_hook_extra_columns_ignored(recon_heads):
    """the fused op passes [q, k, v, z]: only the first qkv columns are read"""
    h, t, norm_w, w_qkv, ln = recon_heads
    h2o_exits.HEADS = h; h2o_exits.STEP = make_step(2, tau=0.0)
    x = torch.randn(4, H)
    q = torch.cat([project(x, w_qkv, ln), torch.randn(4, 17)], -1)
    with pytest.raises(h2o_exits.H2OExit) as e:
        h2o_exits.gdn_hook(20, q)
    xn = x[torch.tensor([1, 3])]; xn = xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(e.value.hidden.float(), ref_normed(t, 20, norm_w, xn), atol=1e-4)


def test_hook_continues_when_one_fails(recon_heads):
    h2o_exits.HEADS = recon_heads[0]; h2o_exits.STEP = make_step(3, tau=2.0)
    h2o_exits.gdn_hook(20, torch.randn(6, QKV))
    assert h2o_exits.STEP.tried == [20]


def test_hook_respects_allowed(recon_heads):
    h2o_exits.HEADS = recon_heads[0]; h2o_exits.STEP = make_step(2, tau=0.0, allowed=False)
    h2o_exits.gdn_hook(20, torch.randn(4, QKV))                   # k=20 not in these requests' depths
    with pytest.raises(h2o_exits.H2OExit):
        h2o_exits.gdn_hook(24, torch.randn(4, QKV))


def test_hook_other_layers_ignored(recon_heads):
    h2o_exits.HEADS = recon_heads[0]; h2o_exits.STEP = make_step(2, tau=0.0)
    h2o_exits.gdn_hook(5, torch.randn(4, QKV))                    # not an exit layer: nothing happens


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


def test_commit_rule_with_prefix_cache_write():
    """the 0.30.0.3 rebase joined two rules in _commits_to_cache: an exit-policy request (no block hashes) and a
    prefix_cache_write=false request commit nothing; an ordinary causal request commits, a short one too"""
    from types import SimpleNamespace

    from vllm.v1.core.kv_cache_manager import _commits_to_cache

    def req(**kw):
        base = dict(prefix_cache_write=True, exit_policy=None, block_hashes=[b"h"], attention_mode="causal")
        base.update(kw)
        return SimpleNamespace(**base)

    assert _commits_to_cache(req())
    assert _commits_to_cache(req(block_hashes=[]))  # causal, under one block: stock commits
    assert not _commits_to_cache(req(exit_policy=object(), block_hashes=[]))
    assert not _commits_to_cache(req(prefix_cache_write=False))
    assert not _commits_to_cache(req(prefix_cache_write=False, exit_policy=object(), block_hashes=[]))
    assert not _commits_to_cache(req(attention_mode="bidirectional", block_hashes=[]))
    # the Request itself: an exit policy keeps prefix_cache_write's parse and drops the hashes
    from vllm.v1.request import Request

    hasher = lambda r: [b"h"] * (len(r.all_token_ids) // 4)  # noqa: E731
    r = Request("r1", list(range(16)), sp({"exit_policy": "auto", "prefix_cache_write": False}), None,
                block_hasher=hasher)
    assert r.exit_policy is not None and r.block_hashes == [] and not _commits_to_cache(r)


# ------------------------------------------------------------------ no graph change
def test_model_and_compile_files_have_no_exit_points():
    """the exits live inside the existing GDN op: the model loop and the splitting-op list are untouched"""
    import inspect

    from vllm.config.compilation import CompilationConfig
    from vllm.model_executor.models import qwen3_next
    assert not any("h2o" in op for op in CompilationConfig._attention_ops)
    assert "h2o" not in inspect.getsource(qwen3_next.Qwen3NextModel.forward)
