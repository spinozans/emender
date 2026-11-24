"""Chosen-only execution, including per-round coalescing and credit across byte boundaries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

import torch

from .runtime import Atom, Frame, Instruction, Refused


@dataclass(frozen=True)
class Event:
    inst: Instruction
    status: str  # ok, empty, or refusal reason
    rows: int
    register: int | None
    reader: tuple | None  # source, version, capability, live watermark
    keys: tuple[Atom, ...]  # trace is session-private; not a cross-user cache


@dataclass
class Decision:
    log_probs: torch.Tensor
    log_policy: torch.Tensor  # includes the opcode AND every conditional operand
    trace: tuple[Event, ...]
    requests: int  # logical submitted keys, before cross-example coalescing
    session_id: str
    visible: int
    mode: str


class Views:
    """Prediction-local; never reused after training updates or across sessions."""

    def __init__(self, policy):
        self.policy = policy
        self.cache = {}

    def __call__(self, p):
        if id(p) not in self.cache:
            self.cache[id(p)] = self.policy.view(p)
        return self.cache[id(p)]


def predict_many(policy, sessions, hidden, base_logits, *, mode, replay, generator):
    batch = len(sessions)
    if not batch or len({s.id for s in sessions}) != batch:
        raise ValueError("one distinct session per batch row")
    if hidden.shape != (batch, policy.host.in_features) or base_logits.shape != (batch, 256):
        raise ValueError("host shape")
    if mode not in ("sample", "greedy"):
        raise ValueError("unknown policy mode")
    if replay is not None and (len(replay) != batch or any(not p for p in replay)):
        raise ValueError("bad replay batch")
    frames = [Frame(s) for s in sessions]
    views = [Views(policy) for s in sessions]
    traces = [[] for s in sessions]
    scores = [[] for s in sessions]
    results = [None] * batch
    active = list(range(batch))
    step = 0
    while active:
        last = [traces[i][-1].inst.op if traces[i] else None for i in active]
        status = [traces[i][-1].status if traces[i] else "start" for i in active]
        hs = policy.rebuild(
            [frames[i] for i in active], hidden[active], [views[i] for i in active], last, status, step
        )
        groups = defaultdict(dict)
        instructions = {}
        denied = {}
        routes = {}
        if replay is not None and any(step >= len(replay[i]) for i in active):
            raise ValueError("replay did not emit")
        forced = None if replay is None else [replay[i][step] for i in active]
        selected = policy.choose_many(
            [frames[i] for i in active], hs, [views[i] for i in active], last, step, mode, forced, generator
        )
        for i, (inst, lp) in zip(active, selected, strict=True):
            f = frames[i]
            scores[i].append(lp)
            instructions[i] = inst
            if inst.op == "LOOKUP":
                try:
                    reader, keys = f.request(inst.args)
                except Refused as exc:
                    denied[i] = str(exc)
                    continue
                group = (reader, f.l.rows)
                groups[group].update(dict.fromkeys(keys))
                routes[i] = (group, keys)

        # Issue ONLY sampled/greedy reads. Equal text in unrelated capabilities never coalesces.
        for (reader, cap), table in groups.items():
            keys = tuple(table)
            for pos in range(0, len(keys), reader.batch_limit):
                chunk = keys[pos : pos + reader.batch_limit]
                replies = reader.read_many(chunk, cap)
                table.update(zip(chunk, replies, strict=True))

        remaining = []
        for i, h in zip(active, hs, strict=True):
            f = frames[i]
            inst = instructions[i]
            get = views[i]
            rows = 0
            reg = None
            binding = None
            keys = ()
            table = None
            if inst.op == "LOOKUP":
                reader = f.s.readers[inst.args[-1]]
                binding = (
                    reader.source,
                    reader.version,
                    reader.operation,
                    reader.watermark() if reader.watermark else None,
                )
            if i in routes:
                group, keys = routes[i]
                table = groups[group]
            if inst.op == "EMIT":
                if replay is not None and step + 1 != len(replay[i]):
                    raise ValueError("instructions after EMIT")
                logp = policy.emit(inst, f, h, base_logits[i], get)
                traces[i].append(Event(inst, "ok", 0, None, None, ()))
                results[i] = Decision(
                    logp,
                    torch.stack(scores[i]).sum(),
                    tuple(traces[i]),
                    f.requests,
                    f.s.id,
                    f.s.visible,
                    mode if replay is None else "replay",
                )
                continue
            status = "ok"
            try:
                if i in denied:
                    raise Refused(denied[i])
                p = f.execute(inst, h, policy, table)
                reg = len(f.reg)
                f.reg.append(p)
                rows = len(p.keys)
                if not rows:
                    status = "empty"
            except Refused as exc:
                # A refusal is observable, consumes the instruction, and creates NO result register.
                status = "refused: " + str(exc)
            traces[i].append(Event(inst, status, rows, reg, binding, keys))
            remaining.append(i)
        active = remaining
        step += 1
    # Commit only after every row successfully emitted. A malformed provider is a hard error.
    for f in frames:
        f.s.cells = f.cells
        f.s.pending = True
    return results


@dataclass
class Objective:
    loss: torch.Tensor
    mean_nll: torch.Tensor
    policy_term: torch.Tensor
    cost_to_go: torch.Tensor


def episode_objective(decisions, targets, *, key_price=0.0, instruction_price=0.0, baseline=0.0):
    """One on-policy document segment; minimize with any ordinary torch optimizer.

    Exact pathwise NLL on the sampled computation plus a score-function surrogate for
    the discrete decisions. Baseline must be independent of actions at/after that step.
    Document labels enter here, never in predict_many. No gradient across separate episodes.
    """
    if not decisions or any(d.mode != "sample" for d in decisions):
        raise ValueError("sampled on-policy decisions required")
    if len({d.session_id for d in decisions}) != 1 or any(
        b.visible != a.visible + 1 for a, b in zip(decisions, decisions[1:])
    ):
        raise ValueError("one contiguous document segment required")
    lp = torch.stack([d.log_probs for d in decisions])
    y = torch.as_tensor(targets, device=lp.device)
    if y.dtype != torch.long or y.shape != (len(lp),) or bool(((y < 0) | (y >= 256)).any()):
        raise ValueError("invalid byte targets")
    nll = -lp.gather(1, y[:, None]).squeeze(1)
    return _objective(decisions, nll, key_price, instruction_price, baseline)


def _objective(decisions, nll, key_price, instruction_price, baseline):
    if not all(isfinite(c) and c >= 0 for c in (key_price, instruction_price)):
        raise ValueError("invalid work price")
    costs = nll.detach() + nll.new_tensor(
        [d.requests * key_price + (len(d.trace) - 1) * instruction_price for d in decisions]
    )
    future = costs.flip([0]).cumsum(0).flip([0])
    b = torch.as_tensor(baseline, device=nll.device).detach()
    if b.shape not in ((), future.shape):
        raise ValueError("bad baseline")
    term = ((future - b) * torch.stack([d.log_policy for d in decisions])).mean()
    return Objective(nll.mean() + term, nll.mean(), term, future)
