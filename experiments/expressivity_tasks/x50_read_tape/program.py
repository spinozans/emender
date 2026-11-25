"""A single causal attention layer over observations, parent answers and chosen child answers."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("ONE", "GREEDY", "BLIND", "ADAPT")
T, D = 64, 32


def log_prior(pdf):
    return (0.99 * pdf + 0.01 / 256).log()


class Tape(nn.Module):
    def __init__(self):
        super().__init__()
        self.byte = nn.Embedding(256, D)
        self.kind = nn.Embedding(3, D)
        self.time = nn.Embedding(T, D)
        self.answer = nn.Linear(256, D, bias=False)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm(D)
        self.act = nn.Linear(D, 256)
        self.emit = nn.Linear(D, 256)
        self.scale = nn.Parameter(torch.tensor(0.5))
        for layer in (self.act, self.emit):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def row(self, kind, byte, time, pdf=None):
        row = self.kind.weight[kind] + self.byte(byte) + self.time(time)
        return row if pdf is None else row + self.answer(pdf.sqrt())

    def attend(self, rows):
        def heads(layer):
            return layer(rows).unsqueeze(1)

        v = F.scaled_dot_product_attention(
            heads(self.q), heads(self.k), heads(self.v), is_causal=True
        )
        return self.norm(rows + v.squeeze(1))

    def frontier(self, row, keys, values):
        # The only cache entries supplied here precede (or equal) this action.
        q = self.q(row)[:, None, None, :]
        v = F.scaled_dot_product_attention(q, keys[:, None], values[:, None])
        return self.norm(row + v[:, 0, 0])


@dataclass
class Trace:
    arm: str
    observed: torch.Tensor
    parent: torch.Tensor
    action: torch.Tensor | None
    child: torch.Tensor | None
    selected_logp: torch.Tensor | None


def choose(logits, generator):
    if generator is None:
        a = logits.argmax(dim=-1)
    else:
        a = torch.multinomial(logits.flatten(0, -2).softmax(-1), 1, generator=generator)
        a = a.reshape(logits.shape[:-1])
    lp = F.log_softmax(logits, dim=-1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
    return a, lp


@torch.no_grad()
def execute(model, bank, inputs, arm, generator=None):
    """Labels are not an argument. Only exogenous causal first reads are prefetched.

    GREEDY/BLIND can batch their chosen calls. ADAPT must receive the previous
    chosen answer before its next policy step. No enumerated child feature bank.
    """
    if (
        arm not in ARMS
        or inputs.dtype != np.uint8
        or inputs.ndim != 2
        or not 1 <= inputs.shape[1] <= T
    ):
        raise ValueError("unknown program or invalid observed byte tensor")
    batch, steps = inputs.shape
    first = bank.scan(inputs)
    p = torch.from_numpy(first.pdf)
    x = torch.from_numpy(inputs.astype(np.int64))
    if arm == "ONE":
        return Trace(arm, x, p, None, None, None)
    times = torch.arange(steps)
    lp = None
    if arm == "GREEDY":
        a = p.argmax(-1)
    elif arm == "BLIND":
        logits = model.act(model.attend(model.row(0, x, times)))
        a, lp = choose(logits, generator)
    else:
        obs, parent = model.row(0, x, times), model.row(1, x, times, p)
        pair = torch.stack((obs, parent), dim=2)
        pk, pv = model.k(pair), model.v(pair)
        keys, values = (torch.empty(batch, 3 * steps, D) for _ in range(2))
        actions, answers, selected = [], [], []
        prior = log_prior(p)
        for t in range(steps):
            i = 3 * t
            keys[:, i : i + 2], values[:, i : i + 2] = pk[:, t], pv[:, t]
            ctx = model.frontier(parent[:, t], keys[:, : i + 2], values[:, : i + 2])
            b, prob = choose(model.act(ctx) + model.scale * prior[:, t], generator)
            reply = bank.follow(first.at(t), b.numpy().astype(np.uint8))
            child = torch.from_numpy(reply.pdf)
            row = model.row(2, b, times[t], child)
            keys[:, i + 2], values[:, i + 2] = model.k(row), model.v(row)
            actions.append(b)
            answers.append(child)
            selected.append(prob)
        return Trace(
            arm,
            x,
            p,
            torch.stack(actions, 1),
            torch.stack(answers, 1),
            torch.stack(selected, 1),
        )
    reply = bank.follow(first, a.numpy().astype(np.uint8))
    return Trace(arm, x, p, a, torch.from_numpy(reply.pdf), lp)


def predict(model, trace):
    """Differentiable causal replay of the actual work tape; no reads in backward."""
    x, p = trace.observed, trace.parent
    times = torch.arange(x.shape[1])
    obs = model.row(0, x, times)
    rows = [obs, model.row(1, x, times, p)]
    if trace.arm != "ONE":
        rows.append(model.row(2, trace.action, times, trace.child))
    z = torch.stack(rows, 2)
    h = model.attend(z.flatten(1, 2)).reshape(z.shape)
    prior = log_prior(p)
    policy = None
    if trace.arm == "ADAPT":
        policy = model.act(h[:, :, 1]) + model.scale * prior
    elif trace.arm == "BLIND":
        policy = model.act(model.attend(obs))
    logp = (
        None
        if policy is None
        else policy.log_softmax(-1).gather(-1, trace.action.unsqueeze(-1)).squeeze(-1)
    )
    return model.emit(h[:, :, -1]) + prior, logp


def policy_loss(selected_logp, losses, baseline):
    # A sampled reply can affect any later output and later choice. The baseline
    # depends only on exogenous observations, never on this or later actions.
    advantage = (losses - baseline).detach().flip((1,)).cumsum(1).flip((1,))
    return (selected_logp * advantage).mean()


def score(model, trace, targets):
    logits, lp = predict(model, trace)
    if targets.dtype != np.uint8 or targets.shape != tuple(trace.observed.shape):
        raise ValueError("invalid external target shape/type")
    y = torch.from_numpy(targets.astype(np.int64))
    loss = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
    baseline = -log_prior(trace.parent).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    obj = loss.mean()
    if lp is not None:
        # Sampling was online, including historical chosen replies. Refuse a
        # different policy in the parallel score-function replay.
        torch.testing.assert_close(
            lp.detach(), trace.selected_logp, atol=2e-5, rtol=2e-5
        )
        obj = obj + policy_loss(lp, loss, baseline)
    return obj, loss, baseline
