# SPDX-License-Identifier: Apache-2.0
"""H2O early-exit heads on a frozen trunk (v1).

Add-on exit heads read the residual stream after layer k at each request's answer
slot (its last prompt token) and produce a normed hidden state that the model's own
lm_head turns into logits, so the readout path (logprob_token_ids) is unchanged.
The heads never write into the trunk.

A step exits at layer k only when EVERY request in the batch asked for an exit
policy, is finishing its prefill in this step, and passes its agreement gate at k
(the v1 batch rule). The exit check is a splitting custom op
(vllm::h2o_exit_point) placed after each exit layer; it runs eagerly between the
piecewise graphs and ends the forward pass by raising H2OExit, which the model
runner catches. With no exit file loaded the op is never traced into the graph, so
the engine is the same as without this module.

Exit file: ``exits.lightning`` beside the weights (safetensors format; not named
*.safetensors so the weight loader does not read it as a trunk shard), enabled by
VLLM_H2O_EXITS=1 (or a path). Tensors ``k{K}.A.weight``, ``k{K}.B.weight``,
``k{K}.B.bias``, ``k{K}.agree.{0,2}.{weight,bias}``; metadata ``h2o_exits`` (JSON):
{"layers": [20, 24], "tau": {"20": t, "24": t}, "auto": [20, 24], "eps": 1e-6}.

Request policy (sampling_params.extra_args, i.e. vllm_xargs, flat values):
  exit_policy = "full" (default) | "auto" | a layer count; exit_temp = T (the
  shim's per-type readout temperature); exit_tau_<k> = an agreement threshold
  overriding the file's default. A dict {"depth", "temp", "tau"} also works offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

EXIT_FILE = "exits.lightning"


class H2OExit(Exception):
    """Raised by the exit op to end the forward pass at layer k."""

    def __init__(self, k: int, hidden: torch.Tensor):
        super().__init__(k)
        self.k = k
        self.hidden = hidden  # [num_reqs, hidden] normed exit states (model dtype)


def exit_layers_from_env() -> tuple[int, ...]:
    """The exit layers the model forward places exit points after (static, read
    at model construction). Empty unless VLLM_H2O_EXITS is set."""
    if not os.environ.get("VLLM_H2O_EXITS"):
        return ()
    raw = os.environ.get("VLLM_H2O_EXIT_LAYERS", "20,24")
    return tuple(sorted(int(x) for x in raw.split(",") if x.strip()))


# ---------------------------------------------------------------- policy
def parse_exit_policy(sampling_params) -> dict | None:
    """None for "full" (the default); else {"depths": [k...], "temp": T,
    "tau": {k: t}} with depths resolved against the loaded heads later."""
    extra = getattr(sampling_params, "extra_args", None) or {}
    pol = extra.get("exit_policy")
    if pol is None:
        return None
    if not isinstance(pol, dict):
        # vllm_xargs values are flat: exit_policy, exit_temp, exit_tau_<k>
        pol = {"depth": pol, "temp": extra.get("exit_temp", 1.0),
               "tau": {k[len("exit_tau_"):]: v for k, v in extra.items()
                       if k.startswith("exit_tau_")}}
    depth = pol.get("depth", "full")
    if depth in ("full", None, 32, "32"):
        return None
    if depth == "auto":
        depths = "auto"
    else:
        try:
            depths = [int(depth)]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"exit_policy depth must be 'full', 'auto' or a layer count, got {depth!r}"
            ) from e
    temp = float(pol.get("temp", 1.0))
    if not temp > 0:
        raise ValueError("exit_policy temp must be > 0")
    tau = {int(k): float(v) for k, v in (pol.get("tau") or {}).items()}
    return {"depths": depths, "temp": temp, "tau": tau}


# ---------------------------------------------------------------- heads
class ExitHeads(torch.nn.Module):
    def __init__(self, path: str, hidden: int, norm_weight: torch.Tensor,
                 lm_head_weight: torch.Tensor, device: torch.device):
        super().__init__()
        from safetensors import safe_open

        with safe_open(path, "pt", device="cpu") as f:
            meta = json.loads((f.metadata() or {}).get("h2o_exits", "{}"))
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        self.layers = [int(x) for x in meta.get("layers", [])]
        self.default_tau = {int(k): float(v) for k, v in meta.get("tau", {}).items()}
        self.auto = [int(x) for x in meta.get("auto", self.layers)]
        self.eps = float(meta.get("eps", 1e-6))
        self.p = {}
        for k in self.layers:
            self.p[k] = {n[len(f"k{k}."):]: t.to(device=device, dtype=torch.float32)
                         for n, t in tensors.items() if n.startswith(f"k{k}.")}
            need = {"A.weight", "B.weight", "B.bias", "agree.0.weight",
                    "agree.0.bias", "agree.2.weight", "agree.2.bias"}
            missing = need - set(self.p[k])
            if missing:
                raise ValueError(f"{path}: exit k={k} lacks {sorted(missing)}")
        # frozen references to the trunk (never modified): final norm weight
        # (Qwen3.5 RMSNorm scales by 1 + w) and the tied lm_head rows
        self.norm_w = norm_weight.detach().float() + 1.0
        self.lm_head = lm_head_weight
        self.hidden = hidden
        logger.info("H2O exits loaded from %s: layers %s, tau %s", path,
                    self.layers, self.default_tau)

    def normed(self, k: int, x: torch.Tensor) -> torch.Tensor:
        p = self.p[k]
        x = x.float()
        z = x + F.linear(F.linear(x, p["A.weight"]), p["B.weight"], p["B.bias"])
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w

    def agreement(self, k: int, hn: torch.Tensor, label_ids: list[list[int]],
                  temps: torch.Tensor) -> torch.Tensor:
        """P(exit top-1 == full-depth top-1) per request, as the head was trained:
        features [hn, p1, p2, p1 - p2] with p the temperature softmax over the
        request's label tokens."""
        p = self.p[k]
        feats = []
        for i, ids in enumerate(label_ids):
            w = self.lm_head[ids].float()                       # K x H
            lg = hn[i] @ w.T
            pr = torch.softmax(lg / temps[i], -1)
            t2 = pr.topk(min(2, pr.numel())).values
            p1, p2 = t2[0], t2[-1]
            feats.append(torch.stack([p1, p2, p1 - p2]))
        f = torch.cat([hn, torch.stack(feats)], -1)
        h = F.gelu(F.linear(f, p["agree.0.weight"], p["agree.0.bias"]))
        return torch.sigmoid(F.linear(h, p["agree.2.weight"], p["agree.2.bias"])).squeeze(-1)


HEADS: ExitHeads | None = None


@dataclass
class ExitStep:
    """Set by the model runner for a step in which every request asked for an
    exit and finishes its prefill; None otherwise (then the exit op is a no-op)."""
    rows: torch.Tensor                       # [num_reqs] answer-slot token indices
    label_ids: list[list[int]]
    temps: torch.Tensor                      # [num_reqs]
    allowed: dict[int, torch.Tensor]         # k -> bool[num_reqs] (k in the request's depths)
    tau: dict[int, torch.Tensor]             # k -> [num_reqs]
    tried: list[int] = field(default_factory=list)


STEP: ExitStep | None = None


def load(model, model_dir: str, device) -> None:
    """Load the exit file and bind it to the model's frozen norm and lm_head."""
    global HEADS
    env = os.environ.get("VLLM_H2O_EXITS")
    if not env:
        return
    path = env if env not in ("1", "true", "True") else os.path.join(model_dir, EXIT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VLLM_H2O_EXITS is set but {path} does not exist")
    lm = model.language_model if hasattr(model, "language_model") else model
    inner = lm.model
    lm_head = lm.lm_head.weight if hasattr(lm, "lm_head") and hasattr(lm.lm_head, "weight") \
        else inner.embed_tokens.weight
    HEADS = ExitHeads(path, inner.config.hidden_size, inner.norm.weight, lm_head, device)
    expect = tuple(getattr(inner, "h2o_exit_layers", ()))
    if tuple(sorted(HEADS.layers)) != tuple(sorted(expect)):
        raise ValueError(f"exit file layers {HEADS.layers} != model exit points {expect} "
                         "(set VLLM_H2O_EXIT_LAYERS to match)")


def h2o_exit_point(hidden_states: torch.Tensor, residual: torch.Tensor, k: int,
                   flag: torch.Tensor) -> None:
    """Splitting op after layer k: decide the exit for the whole step."""
    st = STEP
    if st is None or HEADS is None or k not in st.allowed:
        return
    allow = st.allowed[k]
    if not bool(allow.all()):
        return
    x = hidden_states[st.rows] + residual[st.rows]
    hn = HEADS.normed(k, x)
    agree = HEADS.agreement(k, hn, st.label_ids, st.temps)
    st.tried.append(k)
    if bool((agree >= st.tau[k]).all()):
        flag.fill_(k)
        raise H2OExit(k, hn.to(hidden_states.dtype))


def _h2o_exit_point_fake(hidden_states, residual, k, flag) -> None:
    return None


direct_register_custom_op(
    op_name="h2o_exit_point",
    op_func=h2o_exit_point,
    mutates_args=["flag"],
    fake_impl=_h2o_exit_point_fake,
)
