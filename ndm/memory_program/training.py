"""Data-independent teacher forcing for an actual causal byte LM. No precomputed addresses."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from .machine import Decision, _objective


@dataclass
class BatchResult:
    decisions: list[list[Decision]]  # document, then time; contains private request bindings
    targets: int
    mean_nll: torch.Tensor  # every real shifted byte; deployment metric, not the surrogate
    objective: torch.Tensor | None  # backpropagate this in sample mode


def teacher_force(
    policy,
    encode,
    sessions,
    windows,
    *,
    mode="sample",
    generator=None,
    replay=None,
    key_price=0.0,
    instruction_price=0.0,
    baseline=0.0,
    observer=None,
):
    """One verified raw-byte window and fresh session per document. No padding/cross-doc packing.

    encode(x) must be causal and return [B,T,H] hidden states and [B,T,256] logits; the
    mounting application is responsible for verifying document boundaries and source hashes.
    The backbone can be batched by input length; learned memory addresses CANNOT be prefilled
    that way. Baseline is a fixed pre-episode cost baseline here (e.g. from earlier batches).
    mode='replay' executes an exact recorded trace for a parity/counterfactual check. It never
    returns an on-policy training objective.
    """
    n = len(windows)
    if (
        mode not in ("sample", "greedy", "replay")
        or not n
        or len(sessions) != n
        or len({s.id for s in sessions}) != n
    ):
        raise ValueError("invalid document batch")
    if any(s.visible or s.pending or any(p is not None for p in s.cells) for s in sessions):
        raise ValueError("fresh document sessions required")
    if any(not isinstance(w, bytes) or len(w) < 2 for w in windows):
        raise ValueError("window must contain raw inputs and targets")
    if (mode == "replay") != (replay is not None) or (
        replay is not None
        and (
            len(replay) != n
            or any(len(trace) != len(w) - 1 for trace, w in zip(replay, windows, strict=True))
        )
    ):
        raise ValueError("replay requires an exact per-document instruction trace for every target")
    predict = policy.batch_runner() if hasattr(policy, "batch_runner") else policy
    device = policy.embed.weight.device
    by_length = defaultdict(list)
    hidden = {}
    logits = {}
    for i, w in enumerate(windows):
        by_length[len(w) - 1].append(i)
    for t, ids in by_length.items():
        x = torch.tensor([list(windows[i][:-1]) for i in ids], device=device, dtype=torch.long)
        h, l = encode(x)
        if h.shape != (len(ids), t, policy.host.in_features) or l.shape != (len(ids), t, 256):
            raise ValueError("bad causal encoder output")
        for j, i in enumerate(ids):
            hidden[i] = h[j]
            logits[i] = l[j]
    traces = [[] for s in sessions]
    losses = [[] for s in sessions]
    for s, w in zip(sessions, windows, strict=True):
        s.observe(w[:1])
    for t in range(max(by_length)):
        ids = [i for i, w in enumerate(windows) if t + 1 < len(w)]
        ds = predict(
            [sessions[i] for i in ids],
            torch.stack([hidden[i][t] for i in ids]),
            torch.stack([logits[i][t] for i in ids]),
            mode="greedy" if replay is not None else mode,
            generator=generator,
            **({"replay": [replay[i][t] for i in ids]} if replay is not None else {}),
            **({"observer": observer} if observer is not None else {}),
        )
        ys = [windows[i][t + 1] for i in ids]
        if hasattr(policy, "batch_runner"):
            labels = torch.as_tensor(ys, device=device)
            nll = -torch.stack([d.log_probs for d in ds]).gather(1, labels[:, None])[:, 0]
        else:
            nll = [-d.log_probs[y] for d, y in zip(ds, ys, strict=True)]
        for i, d, y, loss in zip(ids, ds, ys, nll, strict=True):
            losses[i].append(loss)
            traces[i].append(d)
            sessions[i].observe(bytes([y]))  # score first, advance transcript/live watermark second
    losses = [torch.stack(row) for row in losses]
    targets = sum(len(x) for x in losses)
    mean_nll = torch.stack([x.sum() for x in losses]).sum() / targets
    objective = None
    if mode == "sample":
        objectives = [
            _objective(ds, x, key_price, instruction_price, baseline)
            for ds, x in zip(traces, losses, strict=True)
        ]
        objective = (
            torch.stack([o.loss * len(x) for o, x in zip(objectives, losses, strict=True)]).sum() / targets
        )
    return BatchResult(traces, targets, mean_nll, objective)
