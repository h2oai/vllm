# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnose a non-causal parity gap (Gemma-4 E4B): bf16 drift, kernel numerics, or a semantic difference?

Runs, on the same prompts and label ids:
  HF32-bidi   transformers fp32, non-causal full layers (the anchor: no bf16 noise)
  HF16-bidi   transformers bf16, non-causal        (how far bf16 alone drifts from the anchor)
  HF16-causal transformers bf16, causal            (the size of the mode change itself)
  vLLM-bidi-T vLLM pooling runner, declaration on, TRITON_ATTN (the served path, BLOCK 32 on the wide heads)
  vLLM-bidi-F vLLM pooling runner, declaration on, FLEX_ATTENTION (a second kernel on the same layers)
and prints max |dp| and mean |dp| for every pair, plus per row the top-2 probability margin and the letter logits.

Reading: if vLLM-bidi-T vs HF32 is within ~1.5x of HF16 vs HF32, the gap is bf16 accumulation on both sides (set the
tolerance relative to the measured bf16 drift); if vLLM-bidi-T and vLLM-bidi-F agree closely but both sit far from
HF32, the difference is semantic (mask, softcap, KV sharing) and must be found; if vLLM-bidi-T is far from both
HF32 and vLLM-bidi-F, it is the Triton encoder kernel's numerics (then try BLOCK 16 or FlexAttention for the wide
layers). A near-tie row (margin < 0.05) amplifies logit noise into |dp|; judge argmax only on rows with margin >= 0.05.

    python diag_gemma4_noncausal_parity.py [--model google/gemma-4-E4B-it] [--prompts prompts.json]
"""

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gemma4_noncausal_e4b import LABELS, PROMPTS  # noqa: E402

MARGIN = 0.05


def _load_reference(model_name, dtype):
    """transformers Gemma-4 with SDPA on the TEXT stack only: the audio tower refuses SDPA on some transformers
    versions ("Gemma4AudioModel does not support an attention implementation through sdpa yet"), so the towers get
    eager (they are not exercised: text prompts) and the language model SDPA, which is the kernel the hook relies on."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype,
            attn_implementation={"text_config": "sdpa", "vision_config": "eager", "audio_config": "eager"},
        )
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="eager")
        model.model.language_model.set_attn_implementation("sdpa")
    assert model.model.language_model.config._attn_implementation == "sdpa"
    return model.cuda().eval()


def hf_probs(model_name, prompts, label_ids, noncausal, dtype):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = _load_reference(model_name, dtype)
    tc = model.config.get_text_config()
    tm = model.model.language_model
    if noncausal:
        for layer, kind in zip(tm.layers, tc.layer_types):
            if kind == "full_attention":
                layer.self_attn.is_causal = False
    W = model.lm_head.weight[label_ids].float()
    cap = float(getattr(tc, "final_logit_softcapping", 0) or 0)
    logits = []
    with torch.no_grad():
        for p in prompts:
            enc = tok(p, return_tensors="pt").to("cuda")
            h = model.model(**enc).last_hidden_state[0, -1, :].float()
            lg = h @ W.T
            logits.append((cap * torch.tanh(lg / cap) if cap else lg).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.stack(logits), W.cpu(), cap


def vllm_probs(model_name, prompts, W, cap, backend):
    from vllm import LLM
    from vllm.config import PoolerConfig

    # the HF models' blocks stay cached in this process: release them, and ask vLLM for a share that fits what is
    # left (the first run refused to start at 0.92 utilisation with 23 of 44 GiB free)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"free before vLLM ({backend}): {free / 2**30:.1f} of {total / 2**30:.1f} GiB")
    llm = LLM(
        model=model_name,
        runner="pooling",
        convert="embed",
        pooler_config=PoolerConfig(task="embed", seq_pooling_type="LAST", use_activation=False),
        hf_overrides={"text_config": {"noncausal_full_attention": True}},
        attention_backend=backend,
        max_model_len=int(os.environ.get("DIAG_MAX_LEN", "8192")),
        enable_prefix_caching=False,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("DIAG_GPU_MEM", "0.6")),
    )
    outs = llm.embed(prompts, use_tqdm=False)
    h = torch.stack([torch.tensor(o.outputs.embedding) for o in outs]).float()
    lg = h @ W.T
    lg = cap * torch.tanh(lg / cap) if cap else lg
    del llm
    torch.cuda.empty_cache()
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--prompts", help="JSON list of prompt strings (default: the test's)")
    ap.add_argument("--skip-flex", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    prompts = json.load(open(a.prompts)) if a.prompts else PROMPTS
    tok = AutoTokenizer.from_pretrained(a.model)
    label_ids = [tok(label, add_special_tokens=False)["input_ids"][-1] for label in LABELS]
    runs = {}
    runs["HF32-bidi"], W32, cap = hf_probs(a.model, prompts, label_ids, True, torch.float32)
    runs["HF16-bidi"], W16, _ = hf_probs(a.model, prompts, label_ids, True, torch.bfloat16)
    runs["HF16-causal"], _, _ = hf_probs(a.model, prompts, label_ids, False, torch.bfloat16)
    runs["vLLM-bidi-T"] = vllm_probs(a.model, prompts, W16, cap, "TRITON_ATTN")
    if not a.skip_flex:
        try:
            runs["vLLM-bidi-F"] = vllm_probs(a.model, prompts, W16, cap, "FLEX_ATTENTION")
        except Exception as e:  # a backend that refuses this model is itself a finding
            print(f"FLEX_ATTENTION run failed: {type(e).__name__}: {str(e)[:200]}")
    probs = {k: torch.softmax(v, -1) for k, v in runs.items()}
    names = list(probs)
    print("\nmax |dp| / mean |dp| between runs (rows x labels):")
    print(" " * 14 + "".join(f"{n:>16s}" for n in names))
    for i in names:
        row = []
        for j in names:
            d = (probs[i] - probs[j]).abs()
            row.append(f"{d.max():.4f}/{d.mean():.4f}")
        print(f"{i:>14s}" + "".join(f"{x:>16s}" for x in row))
    print("\nper row: top-2 margin (HF32-bidi) and letter logits per run:")
    for r in range(len(prompts)):
        top = probs["HF32-bidi"][r].topk(2).values
        print(f"row {r}: margin {float(top[0] - top[1]):.4f}" + "".join(
            f"\n   {n:>12s} logits {[round(float(x), 3) for x in runs[n][r]]} probs {[round(float(x), 3) for x in probs[n][r]]}"
            for n in names))
    anchor = probs["HF32-bidi"]
    drift = (probs["HF16-bidi"] - anchor).abs().max().item()
    served = (probs["vLLM-bidi-T"] - anchor).abs().max().item()
    clear = [r for r in range(len(prompts)) if float(anchor[r].topk(2).values.diff().abs()) >= MARGIN]
    agree = sum(int(anchor[r].argmax() == probs["vLLM-bidi-T"][r].argmax()) for r in clear)
    print(f"\nbf16 drift (HF16 vs HF32): {drift:.4f}; served (vLLM Triton vs HF32): {served:.4f}; ratio {served / max(drift, 1e-6):.2f}")
    print(f"argmax agreement on the {len(clear)} rows with margin >= {MARGIN}: {agree}/{len(clear)}")
    verdict = "bf16 accumulation (both sides)" if served <= 1.5 * drift + 0.01 else (
        "kernel numerics (Triton far from HF32)" if "vLLM-bidi-F" in probs and (probs["vLLM-bidi-F"] - anchor).abs().max().item() <= 1.5 * drift + 0.01
        else "semantic difference or shared kernel issue: inspect")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
