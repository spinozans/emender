"""A host-independent policy over typed instructions. No query is issued to score a choice."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import math

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import Frame, Instruction, OPS, Page, Span

ARITY = dict(SLICE=2, LOOKUP=3, FOCUS=2, PROJECT=2, CONCAT=3, JOIN=4, RELATE=2, SAVE=2, RECALL=1, EMIT=2)


def _id(x, buckets):
    # Collisions affect only a learned feature, NEVER equality, validity, or dispatch.
    if isinstance(x, Span):
        # Avoid giving the policy arbitrary session nonces or document IDs as features.
        # The exact full coordinate is still used by the executor.
        x = ("span", x.source, x.start.bit_length(), (x.stop - x.start).bit_length())
    if isinstance(x, bytes) and len(x) == 1:
        return x[0]
    if not isinstance(x, bytes):
        x = repr(x).encode()
    return 256 + int.from_bytes(blake2b(x, digest_size=8).digest(), "little") % buckets


@dataclass
class View:
    rows: torch.Tensor
    fields: torch.Tensor
    summary: torch.Tensor


class MemoryPolicy(nn.Module):
    """One vocabulary and shared parameters for any number of compatible readers.

    `forward(sessions, hidden, base_logits)` predicts but does not observe a target.
    `hidden`: [batch, host_dim], `base_logits`: [batch, 256], both causal.
    """

    def __init__(self, host_dim, dim=48, heads=4, buckets=4096):
        super().__init__()
        if dim % heads or buckets < 1:
            raise ValueError("invalid workbench size")
        self.dim = dim
        self.buckets = buckets
        self.embed = nn.Embedding(256 + buckets, dim)
        self.host = nn.Linear(host_dim, dim)
        self.meta = nn.Linear(2, dim)
        self.budget = nn.Linear(2, dim)
        self.encoder = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0, batch_first=True)
        self.op = nn.Linear(dim, len(OPS), bias=False)
        self.transitions = nn.Embedding(1 + len(OPS), len(OPS))  # last row = start of prediction
        self.pointer = nn.Linear(dim, dim)
        self.prior = nn.Parameter(torch.zeros(len(OPS), 6))
        self.query = nn.Linear(dim, dim, bias=False)
        self.writer = nn.Linear(2 * dim, dim)
        self.fuse = nn.Linear(3 * dim, dim)
        self.pair_query = nn.Linear(2 * dim, dim, bias=False)
        self.pair_key = nn.Linear(dim, dim, bias=False)
        self.payload_out = nn.Linear(dim, 256, bias=False)
        self.binding_out = nn.Linear(dim, 256, bias=False)
        self.mix = nn.Linear(2 * dim, 1)
        # Compile a direct read into weights, not a host-side plan or a restriction on exploration.
        with torch.no_grad():
            for m in (
                self.op,
                self.transitions,
                self.pointer,
                self.query,
                self.fuse,
                self.pair_query,
                self.binding_out,
                self.mix,
            ):
                m.weight.zero_()
                if getattr(m, "bias", None) is not None:
                    m.bias.zero_()
            self.transitions.weight[:, OPS.index("EMIT")] = 1
            for prev, nxt in ((None, "SLICE"), ("SLICE", "LOOKUP"), ("LOOKUP", "EMIT")):
                self.transitions.weight[len(OPS) if prev is None else OPS.index(prev), OPS.index(nxt)] = 3
            self.prior[OPS.index("SLICE"), :2] = 2  # newest endpoint, unit length
            self.prior[OPS.index("LOOKUP"), 2:4] = 2  # text key, byte-edge reader
            self.prior[OPS.index("EMIT"), 4:] = 2  # page with byte column, byte projection

    def tokens(self, labels):
        ids = [_id(x, self.buckets) for x in labels]
        return self.embed(torch.tensor(ids, dtype=torch.long, device=self.embed.weight.device))

    def view(self, p: Page):
        n, d = len(p.keys), self.dim
        fs = self.tokens(
            [
                ("field", i, f.name, f.kind, (f.origin[0], f.origin[2]) if f.origin else ())
                for i, f in enumerate(p.fields)
            ]
        )
        if not n:
            return View(fs.new_empty(0, d), fs, self.tokens(["empty"])[0] + fs.mean(0))
        atoms = [k for row in p.keys for k in row]
        content = self.tokens(atoms)
        # Share information across previously unseen strings as well as learning whole-key features.
        ends = self.tokens([k[-1:] if isinstance(k, bytes) and k else "coordinate" for k in atoms])
        bound = (content + ends).view(n, len(fs), d) * (1 + fs.tanh())
        rows = bound.mean(1) + p.value
        w = p.log_mass.float().softmax(0).to(rows.dtype)
        stats = torch.stack((p.log_mass.new_tensor(math.log1p(n)), p.log_mass.logsumexp(0))).to(rows.dtype)
        return View(rows, fs + w @ bound.transpose(0, 1), w @ rows + self.meta(stats))

    def rebuild(self, frames, hosts, views, last, status, step):
        """Single batched attention pass over scratch and persistent summaries for this round."""
        left = hosts.new_tensor([[1 - step / f.l.steps, 1 - f.requests / f.l.requests] for f in frames])
        control = (
            self.host(hosts)
            + self.budget(left)
            + self.tokens([("prev", p) for p in last])
            + self.tokens([("status", s) for s in status])
        )
        seq = []
        for f, q, get in zip(frames, control, views, strict=True):
            slots = [
                (role, i, p)
                for role, pages in (("reg", f.reg), ("cell", f.cells))
                for i, p in enumerate(pages)
                if p is not None
            ]
            tokens = q.unsqueeze(0)
            if slots:
                summaries = torch.stack([get(p).summary for _, _, p in slots])
                types = self.tokens([(role, i) for role, i, _ in slots])
                tokens = torch.cat((tokens, summaries + types))
            seq.append(tokens)
        pad = nn.utils.rnn.pad_sequence(seq, batch_first=True)
        lengths = torch.tensor([len(s) for s in seq], device=pad.device)
        mask = torch.arange(pad.shape[1], device=pad.device)[None, :] >= lengths[:, None]
        return self.encoder(pad, src_key_padding_mask=mask)[:, 0]

    def options(self, role, options, args, frame, get):
        """Features of already visible operands only. Equalities use original keys in the executor."""
        device = self.embed.weight.device
        choices = list(options)
        features = [[0.0] * 6 for _ in choices]
        vec = self.tokens([(role, i) for i in choices])
        if role == "row":
            vec = vec + get(frame.reg[args[0]]).rows[choices]
        elif role in ("back", "length"):
            tape = frame.s.recent
            keys = (
                [tape[len(tape) - 1 - i : len(tape) - i] for i in choices]
                if role == "back"
                else [tape[len(tape) - args[0] - i : len(tape) - args[0]] for i in choices]
            )
            vec = vec + self.tokens(keys)
            for f, i in zip(features, choices):
                f[0 if role == "back" else 1] = float(i == (0 if role == "back" else 1))
        elif role == "reader":
            mounts = [frame.s.readers[i] for i in choices]
            # No ordinal or source-specific set of head weights is required to add a reader.
            vec = self.tokens([("reader", m.operation) for m in mounts]) + self.tokens(
                [("source", m.source) for m in mounts]
            )
            for f, m in zip(features, mounts):
                f[3] = float(m.operation == "byte_edges")
        elif role == "field" and args[-1] >= 0:
            p = frame.reg[args[-1]]
            vec = vec + torch.stack(
                [get(p).fields[i] if i >= 0 else vec.new_zeros(self.dim) for i in choices]
            )
            for f, i in zip(features, choices):
                if i >= 0:
                    f[2] = float(p.fields[i].kind == "text")
                    f[5] = float(p.fields[i].kind == "byte")
        elif role in ("reg", "cell"):
            pages = frame.reg if role == "reg" else frame.cells
            values = []
            for f, i in zip(features, choices):
                p = pages[i] if i >= 0 else None
                values.append(get(p).summary if p is not None else vec.new_zeros(self.dim))
                f[4] = float(p is not None and any(x.kind == "byte" for x in p.fields))
            vec = vec + torch.stack(values)
        return vec, torch.tensor(features, device=device, dtype=vec.dtype)

    @staticmethod
    def _pick(logits, mode, forced, rng):
        lp = logits.float().log_softmax(0)
        if forced is not None:
            j = forced
        elif mode == "sample":
            j = int(torch.multinomial(lp.detach().exp(), 1, generator=rng))
        else:
            j = int(lp.detach().argmax())
        return j, lp[j]

    def choose(self, frame: Frame, h, get, last, step, mode, forced, rng):
        domains = {op: frame.domain(op) for op in OPS if step + 1 < frame.l.steps or op == "EMIT"}
        legal = [op for op, (_, c) in domains.items() if len(c)]
        scores = self.op(h) + self.transitions.weight[len(OPS) if last is None else OPS.index(last)]
        k, lp = self._pick(
            scores[[OPS.index(x) for x in legal]], mode, legal.index(forced.op) if forced else None, rng
        )
        op = legal[k]
        oi = OPS.index(op)
        args = []
        ctx = h + self.tokens([("op", op)])[0]
        if forced is not None and len(forced.args) != ARITY[op]:
            raise ValueError("wrong instruction arity")
        for depth in range(ARITY[op]):
            role, choices = frame.domain(op, tuple(args)) if args else domains[op]
            choices = list(choices)
            # CONCAT's second key lives in the same register as its first.
            context_args = args[:1] if op == "CONCAT" and depth == 2 else args
            vec, prior = self.options(role, choices, context_args, frame, get)
            scores = vec @ self.pointer(ctx) / math.sqrt(self.dim) + prior @ self.prior[oi]
            k, pl = self._pick(scores, mode, choices.index(forced.args[depth]) if forced else None, rng)
            lp = lp + pl
            ctx = ctx + vec[k]
            args.append(choices[k])
        return Instruction(op, tuple(args)), lp

    def choose_many(self, frames, hs, views, last, step, mode, forced, rng):
        from .sampling import choose_many

        return choose_many(self, frames, hs, views, last, step, mode, forced, rng)

    def merge(self, a, b, h):
        return a + b + self.fuse(torch.cat([a, b, h.expand(len(a), -1)], 1))

    def affinity(self, p, q, h):
        x = self.view(p).rows
        y = self.view(q).rows
        left = self.pair_query(torch.cat([x, h.expand(len(x), -1)], 1))
        return (left @ self.pair_key(y).T).float() / math.sqrt(self.dim)

    def write_values(self, h, page):
        rows = self.view(page).rows
        return page.value + self.writer(torch.cat([rows, h.expand(len(rows), -1)], 1)).tanh()

    def emit(self, inst, frame, h, base, get):
        reg, column = inst.args
        if reg < 0:
            return base.float().log_softmax(-1)
        p = frame.reg[reg]
        v = get(p)
        w = (p.log_mass.float() + (v.rows @ self.query(h)).float() / math.sqrt(self.dim)).softmax(0)
        read = w.to(h.dtype) @ v.rows
        payload = w.to(h.dtype) @ p.value
        neural = (base.float() + (self.payload_out(payload) + self.binding_out(read)).float()).log_softmax(0)
        if column < 0:
            return neural
        ids = torch.tensor([row[column][0] for row in p.keys], device=h.device)
        counts = w.new_zeros(256).scatter_add(0, ids, w)
        log_mem = counts.clamp_min(torch.finfo(counts.dtype).tiny).log().masked_fill(counts == 0, -torch.inf)
        mix = self.mix(torch.cat([h, read])).float().squeeze(0)
        return torch.logaddexp(F.logsigmoid(-mix) + neural, F.logsigmoid(mix) + log_mem)

    def forward(self, sessions, hidden, base_logits, *, mode="greedy", replay=None, generator=None):
        from .machine import predict_many

        return predict_many(
            self, sessions, hidden, base_logits, mode=mode, replay=replay, generator=generator
        )
