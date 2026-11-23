# SPDX-License-Identifier: Apache-2.0
"""H2O residual router: gather-first capture of the last-prompt-token residuals.

The router returns, per request, the residual stream of the configured layers (VLLM_H2O_RESIDUAL_LAYERS) at the
request's LAST PROMPT TOKEN. The model hands these out through the aux-hidden-state path
(EagleModelMixin._maybe_add_hidden_state). Without a row buffer that path keeps every selected layer's residual for
EVERY token of the step ([num_tokens, hidden] per layer), which on a long prefill step is gigabytes (the 11-23 OOM:
8.22 GiB on a ~100k-token step of a 27B with 4 layers). With a row buffer the model gathers the rows first and keeps
[max_rows, hidden] per layer: O(requests x layers x hidden), static in shape (max_rows = max_num_seqs), so the
compiled piecewise graphs and CUDA graphs see one fixed-size, fixed-address input.

Contract between the runner and the model:
  * ``enable_row_gather`` registers one int64 buffer of ``max_rows`` row indices (zeros) on every model module that
    collects aux hidden states through the mixin, as the non-persistent buffer ``h2o_aux_rows``. It must run after
    load_model and before the first forward (compile / profile / capture), and returns None (gather off, old
    full-step behaviour) when the model or the parallel setup cannot gather.
  * Each step whose batch completes at least one flagged prompt, the runner writes the step's rows into slots
    ``0..n-1`` of the buffer (slot k = the k-th request of ``todo``) before the forward. Slots ``>= n`` keep stale
    values; the model clamps every index to the step's token count, so a stale or dummy-run index never reads out of
    bounds, and those slots are never read.
  * After the forward, request k's vector is ``stack([aux[k] for aux in aux_hidden_states])`` (``[L, d]``); without
    gather it is ``stack([aux[row] ...])`` as before. The values are the same elements, computed by the same add, so
    the returned bytes are bit-identical.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

H2O_ROWS_ATTR = "h2o_aux_rows"


def collects_through_mixin(cls: type) -> bool:
    """True when the class's forward collects aux states through EagleModelMixin._maybe_add_hidden_state."""
    fwd = getattr(cls, "forward", None)
    code = getattr(fwd, "__code__", None)
    return code is not None and "_maybe_add_hidden_state" in code.co_names


def aux_collecting_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The modules (inner language models) that hold a non-empty aux_hidden_state_layers."""
    return [m for m in model.modules() if tuple(getattr(m, "aux_hidden_state_layers", ()) or ())]


def enable_row_gather(model: torch.nn.Module, rows: torch.Tensor) -> bool:
    """Register ``rows`` (a 1-D int64 device tensor of max_rows indices) as ``h2o_aux_rows`` on every module that
    collects aux states; False (nothing registered) when some collecting module does not use the mixin's collection
    or runs sequence parallel (its aux states are chunked by token across ranks), so the caller keeps the old path."""
    from vllm.model_executor.models.interfaces import EagleModelMixin

    mods = aux_collecting_modules(model)
    if not mods:
        return False
    for m in mods:
        if not isinstance(m, EagleModelMixin) or not collects_through_mixin(type(m)):
            return False
        try:
            if bool(getattr(m, "use_sequence_parallel", False)):
                return False
        except Exception:
            return False
    for m in mods:
        m.register_buffer(H2O_ROWS_ATTR, rows, persistent=False)
    return True


def gather_vectors(
    aux_hidden_states: Sequence[torch.Tensor],
    todo: Sequence[tuple[str, int]],
    gathered: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Per request of ``todo`` ((req_id, row of its last prompt token in the step)), the ``[L, d]`` float16 vector.

    gathered=True: aux_hidden_states are the model's gathered rows (request k in slot k); otherwise the full-step
    states, indexed by row. Same ops as the pre-gather code: stack of the per-layer rows, then .to(float16)."""
    if gathered:
        return [(rid, torch.stack([h[k] for h in aux_hidden_states], 0).to(torch.float16))
                for k, (rid, _row) in enumerate(todo)]
    return [(rid, torch.stack([h[row] for h in aux_hidden_states], 0).to(torch.float16)) for rid, row in todo]
