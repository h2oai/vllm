# SPDX-License-Identifier: Apache-2.0
"""H2O early-exit heads on a frozen trunk (v1).

Add-on exit heads read the trunk's state after k decoder layers at each request's
answer slot (its last prompt token) and produce a normed hidden state that the
model's own lm_head turns into logits, so the readout path (logprob_token_ids) is
unchanged. The heads never write into the trunk.

No new graph split: the exit check runs inside the Gated DeltaNet core op of
layer k (a splitting op the model already has; k is a multiple of 4, so layer k is
a GDN layer). That op receives the qkv projection of the layer's input norm,
f = W_qkv (1 + w_ln) * x_k / rms(x_k). The heads recover the scale-free state
x_k / rms(x_k) = R_k f with R_k = diag(1 / (1 + w_ln)) pinv(W_qkv), computed once at
load from the frozen weights. The compiled graph is therefore the same with or
without exits, so full-depth outputs are unchanged by construction.

A step exits at layer k only when EVERY request in the batch asked for an exit
policy, is finishing its prefill in this step, and passes its agreement gate at k
(the v1 batch rule). An exit ends the forward pass by raising H2OExit from inside
the eagerly executed core op; the model runner catches it.

Exit file: ``exits.lightning`` beside the weights (safetensors format; not named
*.safetensors so the weight loader does not read it as a trunk shard), enabled by
VLLM_H2O_EXITS=1 (or a path). Tensors ``k{K}.A.weight``, ``k{K}.B.weight``,
``k{K}.B.bias``, ``k{K}.agree.{0,2}.{weight,bias}`` (heads trained on the
scale-free state); metadata ``h2o_exits`` (JSON): {"layers": [20, 24], "tau":
{"20": t, "24": t}, "auto": [20, 24], "eps": 1e-6, "input": "xnorm"}.

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

logger = init_logger(__name__)

EXIT_FILE = "exits.lightning"


class H2OExit(Exception):
    """Raised inside layer k's core op to end the forward pass at layer k."""

    def __init__(self, k: int, hidden: torch.Tensor):
        super().__init__(k)
        self.k = k
        self.hidden = hidden  # [num_reqs, hidden] normed exit states (model dtype)


# ---------------------------------------------------------------- policy
def parse_exit_policy(sampling_params) -> dict | None:
    """None for "full" (the default); else {"depths": "auto" | [k], "temp": T,
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
                 lm_head_weight: torch.Tensor, device: torch.device,
                 recon: dict[int, torch.Tensor] | None = None):
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
        # (Qwen3.5 RMSNorm scales by 1 + w), the tied lm_head rows, and the per-exit
        # reconstruction maps R_k (qkv projection -> scale-free state)
        self.norm_w = norm_weight.detach().float() + 1.0
        self.lm_head = lm_head_weight
        self.recon = recon or {}
        self.hidden = hidden
        logger.info("H2O exits loaded from %s: layers %s, tau %s", path,
                    self.layers, self.default_tau)

    def normed(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """E1 on the scale-free state x (= x_k / rms(x_k)), then the final norm."""
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
    exit and finishes its prefill; None otherwise (then the hook is a no-op)."""
    rows: torch.Tensor                       # [num_reqs] answer-slot token indices
    label_ids: list[list[int]]
    temps: torch.Tensor                      # [num_reqs]
    allowed: dict[int, torch.Tensor]         # k -> bool[num_reqs] (k in the request's depths)
    tau: dict[int, torch.Tensor]             # k -> [num_reqs]
    tried: list[int] = field(default_factory=list)


STEP: ExitStep | None = None


def _qkv_rows(layer) -> int:
    return (layer.key_dim * 2 + layer.value_dim) // layer.tp_size


def load(model, model_dir: str, device) -> None:
    """Load the exit file and bind it to the model's frozen norm, lm_head and the
    per-exit reconstruction maps (from layer k's frozen input norm and qkv rows)."""
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
    with safe_meta(path) as meta:
        layers = [int(x) for x in meta.get("layers", [])]
    recon = {}
    for k in layers:
        layer = inner.layers[k]
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            raise ValueError(f"exit k={k}: layer {k} is not a Gated DeltaNet layer")
        n = _qkv_rows(attn)
        w_qkv = attn.in_proj_qkvz.weight[:n].detach().float()        # [qkv, hidden]
        ln = layer.input_layernorm.weight.detach().float() + 1.0      # [hidden]
        recon[k] = (torch.linalg.pinv(w_qkv) / ln[:, None]).contiguous()   # [hidden, qkv]
    HEADS = ExitHeads(path, inner.config.hidden_size, inner.norm.weight, lm_head, device, recon)


class safe_meta:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        from safetensors import safe_open
        self.f = safe_open(self.path, "pt", device="cpu")
        return json.loads((self.f.metadata() or {}).get("h2o_exits", "{}"))

    def __exit__(self, *exc):
        self.f = None
        return False


def gdn_hook(layer_idx: int, qkv: torch.Tensor) -> None:
    """Called at the top of layer `layer_idx`'s Gated DeltaNet core op with the qkv
    projection (the first qkv columns of the op's input). Decides the exit at
    k = layer_idx for the whole step; raises H2OExit to end the pass."""
    st = STEP
    if st is None or HEADS is None or layer_idx not in st.allowed:
        return
    if not bool(st.allowed[layer_idx].all()):
        return
    r = HEADS.recon[layer_idx]
    f = qkv[st.rows, : r.shape[1]].float()
    x = f @ r.T                                    # x_k / rms(x_k)
    hn = HEADS.normed(layer_idx, x)
    agree = HEADS.agreement(layer_idx, hn, st.label_ids, st.temps)
    st.tried.append(layer_idx)
    if bool((agree >= st.tau[layer_idx]).all()):
        raise H2OExit(layer_idx, hn.to(qkv.dtype))
