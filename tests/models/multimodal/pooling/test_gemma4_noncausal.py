# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 non-causal full attention against linne-oak's own engine (GPU).

The oracle is MISA-core/linne-oak-1-0 (Apache-2.0; HF revision 6db14b4a), the
Gemma-4-31B-it fine-tune trained with bidirectional attention on its ten
full-attention layers. Its repository ships the reference runtime, which this
test imports UNMODIFIED from the downloaded snapshot (attribution: the files
``linne_engine.py`` and ``nc_hook.py`` of that repository, Apache License 2.0,
copyright their authors):

- ``nc_hook.enable_noncausal``: a forward pre-hook on the text wrapper that
  replaces the mask dict's ``full_attention`` entry with the 2-D padding mask
  (only padding hidden; sliding layers keep their own mask; sdpa, no KV cache).
  This is the training-time definition of "non-causal full attention" that
  the branch's ``EncoderOnlyAttention`` on the same layers must reproduce (vLLM
  runs the prompt unpadded, so "padding only" means "nothing masked").
- ``LinneEngine``: the prompt (chat template, thinking off), the one-token
  labels (A..Z, two-letter labels beyond, each one distinct token after the
  prompt), the readout = lm_head letter logits at the last prompt position with
  Gemma's final logit softcap (30), softmax at temperature 1, right padding
  with the readout at each row's own last position.

vLLM serves the same checkpoint under the pooling runner (``--convert embed``,
LAST pooling, no normalisation) with ``noncausal_full_attention`` declared; the
test computes the same letter logits from the pooled hidden state and the
checkpoint's ``lm_head`` rows (``LinneEngine`` does exactly this, in process).

Two arms, both against the oracle:
  non-causal: ``LinneEngine(noncausal=True)``  vs vLLM with the declaration;
  forced causal: ``LinneEngine(noncausal=False)`` vs vLLM without it (also the
  "force one way" probe: the same weights read causally).
Pass: argmax agreement >= 99 %, max |dp| <= 0.02 per question; with a rows
file that carries gold answers (``LINNE_PARITY_ROWS``), ECE of both sides
within 0.01 of each other. A third check proves the two arms differ (a
silently causal engine cannot pass the non-causal arm by accident).

The snapshot is shared with the training-side probes: set ``LINNE_OAK_DIR`` to
an existing download (``hf download MISA-core/linne-oak-1-0 --revision
6db14b4a``) so the 62 GB are fetched once per box.
"""

import importlib.util
import json
import os
import sys

import pytest
import torch

from vllm.distributed import cleanup_dist_env_and_memory

LINNE_REPO = "MISA-core/linne-oak-1-0"
LINNE_REVISION = "6db14b4a"
MAX_ARGMAX_DISAGREEMENT = 0.01
MAX_ABS_DP = 0.02
MAX_ECE_GAP = 0.01

STATE = (
    "The order arrived damaged yesterday. The customer has a receipt and asks "
    "for a replacement today."
)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "returns": "Refunds, replacements and damaged deliveries",
            "billing": "Payments, invoices and charges",
            "technical": "Product setup and faults",
        },
    },
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
    "wide": {"type": "choice", "criteria": {f"department {i}": None for i in range(30)}},
}


def _linne_dir() -> str:
    d = os.environ.get("LINNE_OAK_DIR")
    if d:
        return d
    from huggingface_hub import snapshot_download

    return snapshot_download(LINNE_REPO, revision=LINNE_REVISION)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows():
    """(state, questions, gold) rows: the three inline ones, or a JSON file of
    ``{"state", "questions", "gold": {qid: option name or bool}}`` for ECE."""
    path = os.environ.get("LINNE_PARITY_ROWS")
    if path:
        return [(r["state"], r["questions"], r.get("gold", {})) for r in json.load(open(path))]
    return [(STATE, QUESTIONS, {})]


def _oracle(linne_dir, rows, noncausal):
    """linne's own answers: {(row, qid): (option names, probabilities)} plus
    the engine's prompts and label ids, so vLLM reads the same tokens."""
    sys.path.insert(0, linne_dir)
    sys.modules.pop("nc_hook", None)
    engine_mod = _load_module(os.path.join(linne_dir, "linne_engine.py"), "linne_engine")
    engine = engine_mod.LinneEngine(model=linne_dir, device="cuda", noncausal=noncausal)
    out, prompts = {}, {}
    for i, (state, questions, _) in enumerate(rows):
        answer, _ = engine(state, questions)
        for qid, q in questions.items():
            a = answer["answers"][qid]
            if q["type"] == "choice":
                names = list(q["criteria"])
                probs = [a["probabilities"][n] for n in names]
                labels = list(engine_mod.string.ascii_uppercase[: len(names)]) if len(names) <= 26 \
                    else engine.pair_labels()[: len(names)]
            else:
                names, probs, labels = ["yes", "no"], [a["noul"], 1 - a["noul"]], ["yes", "no"]
            p = engine.prompt(engine_mod.render(state, q, labels))
            ids = engine.tok(p, add_special_tokens=False)["input_ids"]
            prompts[(i, qid)] = (p, engine.label_ids(p, ids, labels))
            out[(i, qid)] = (names, torch.tensor(probs))
    W = engine.model.lm_head.weight.detach().float().cpu()
    cap = engine.cap
    del engine
    torch.cuda.empty_cache()
    sys.path.remove(linne_dir)
    return out, prompts, W, cap


def _vllm(vllm_runner, linne_dir, prompts, W, cap, noncausal):
    hf_overrides = {"text_config": {"noncausal_full_attention": True}} if noncausal else {}
    keys = list(prompts)
    with vllm_runner(
        linne_dir,
        runner="pooling",
        convert="embed",
        override_pooler_config={"pooling_type": "LAST", "normalize": False},
        hf_overrides=hf_overrides,
        max_model_len=8192,
        enable_prefix_caching=False,
    ) as vllm_model:
        cfg = vllm_model.llm.llm_engine.vllm_config
        if noncausal:
            assert cfg.model_config.attn_type == "encoder_only"
            assert not cfg.cache_config.enable_prefix_caching
            assert not cfg.scheduler_config.enable_chunked_prefill
        outputs = vllm_model.llm.embed(
            [{"prompt": prompts[k][0]} for k in keys], use_tqdm=False
        )
    probs = {}
    for k, o in zip(keys, outputs):
        h = torch.tensor(o.outputs.embedding).float()
        lg = h @ W[prompts[k][1]].T
        if cap:
            lg = cap * torch.tanh(lg / cap)
        probs[k] = torch.softmax(lg, -1)
    return probs


def _ece(pairs, bins=10):
    """Expected calibration error of (confidence, correct) pairs."""
    if not pairs:
        return 0.0
    conf = torch.tensor([c for c, _ in pairs]); ok = torch.tensor([float(x) for _, x in pairs])
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.float().mean().item() * abs(conf[m].mean().item() - ok[m].mean().item())
    return e


def _compare(rows, oracle, ours):
    dis, dp, ece_ref, ece_ours = 0, 0.0, [], []
    for (i, qid), (names, ref) in oracle.items():
        got = ours[(i, qid)]
        dis += int(ref.argmax() != got.argmax())
        dp = max(dp, (ref - got).abs().max().item())
        gold = rows[i][2].get(qid)
        if gold is not None:
            want = names.index(gold) if isinstance(gold, str) else (0 if gold else 1)
            ece_ref.append((ref.max().item(), ref.argmax().item() == want))
            ece_ours.append((got.max().item(), got.argmax().item() == want))
    return dis / len(oracle), dp, _ece(ece_ref), _ece(ece_ours)


@pytest.mark.parametrize("noncausal", [True, False])
def test_linne_oak_parity(vllm_runner, noncausal):
    linne_dir = _linne_dir()
    rows = _rows()
    oracle, prompts, W, cap = _oracle(linne_dir, rows, noncausal)
    cleanup_dist_env_and_memory()
    ours = _vllm(vllm_runner, linne_dir, prompts, W, cap, noncausal)
    disagreement, dp, ece_ref, ece_ours = _compare(rows, oracle, ours)
    assert disagreement <= MAX_ARGMAX_DISAGREEMENT, (disagreement, dp)
    assert dp <= MAX_ABS_DP, dp
    assert abs(ece_ref - ece_ours) <= MAX_ECE_GAP, (ece_ref, ece_ours)


def test_noncausal_and_forced_causal_differ(vllm_runner):
    """The declaration must change the readout: linne's two modes differ on
    these rows, and so must vLLM's two engines."""
    linne_dir = _linne_dir()
    rows = _rows()
    nc, prompts, W, cap = _oracle(linne_dir, rows, True)
    cleanup_dist_env_and_memory()
    c, _, _, _ = _oracle(linne_dir, rows, False)
    cleanup_dist_env_and_memory()
    ref_gap = max((nc[k][1] - c[k][1]).abs().max().item() for k in nc)
    ours_nc = _vllm(vllm_runner, linne_dir, prompts, W, cap, True)
    ours_c = _vllm(vllm_runner, linne_dir, prompts, W, cap, False)
    our_gap = max((ours_nc[k] - ours_c[k]).abs().max().item() for k in nc)
    assert ref_gap > 1e-3 and our_gap > 1e-3, (ref_gap, our_gap)
