"""Bounded exact phrase executions; each surviving history owns its K15 feedback."""
from dataclasses import dataclass, replace

import torch
from torch import nn

from .k15_phrase import Factor, Tape, decode, phrase_source


@dataclass(frozen=True)
class Row:
    start: int
    content: bytes
    payload: torch.Tensor


@dataclass(frozen=True)
class Execution:
    rows: tuple
    feedback: torch.Tensor
    distance: int = 0
    start: int = -1


class PhraseK15(nn.Module):
    def __init__(self, backbone, compiled=True):
        super().__init__()
        self.backbone = backbone
        width = backbone.embed.embedding_dim
        self.phase = nn.Linear(width, 2)
        self.operand = nn.Linear(3, 1, bias=False)
        self.transition = nn.Embedding(3, width)  # literal, start, continue
        self.write = nn.Linear(width, width)
        if compiled:
            with torch.no_grad():
                self.phase.weight.zero_(); self.phase.bias.fill_(3.)
                self.operand.weight.copy_(torch.tensor([[3., 1., .1]]))

    def hidden(self, ids, feedback):
        if ids.shape != (1, 128):
            raise ValueError('requires singleton 128-byte replay')
        h = self.backbone.embed(ids)
        h = torch.cat((h[:, :-1], h[:, -1:] + feedback), dim=1)
        for block in self.backbone.blocks:
            h = block(h)
        return self.backbone.norm(h)[0, -1]


@dataclass
class Edge:
    state: Execution
    weight: torch.Tensor
    byte: int | None
    literal: torch.Tensor
    payload: torch.Tensor
    action: str


class RecurrentReplay:
    def __init__(self, model, session, prefix, cap=2, path_limit=8):
        if len(prefix) != 128 or cap < 1 or path_limit < 1:
            raise ValueError('invalid replay bounds')
        self.model = model
        self.tape = Tape(session); self.tape.append(prefix)
        zero = model.write.bias.new_zeros(model.write.out_features)
        rows = tuple(Row(r.start, r.content, zero) for r in decode(Factor(y) for y in prefix)[1])
        self.paths = [(Execution(rows, zero), zero.new_tensor(1.))]
        self.cap, self.path_limit = cap, path_limit
        self.pending = None
        self.calls = 0
        self.reads = []

    def predict(self, *, chosen=False):
        if self.pending is not None or (chosen and len(self.paths) != 1):
            raise RuntimeError('invalid replay phase')
        m = self.model; tape = self.tape; n = len(tape.raw)
        ids = torch.tensor(list(tape.raw[-128:]), device=m.write.weight.device)[None]
        raw = tape.sources(); edges = []
        for st, mass in self.paths:
            self.calls += 1
            h = m.hidden(ids, st.feedback)
            literal = m.backbone.head(h).softmax(0)*(1-1e-6)+1e-6/256
            payload = m.write(h).tanh()
            start, keep = m.phase(h).sigmoid().unbind()
            rows = st.rows
            def emit(op, rows, dist, origin, addr, credit, recall):
                feedback = (m.transition.weight[op] + recall + st.feedback*.25).tanh()*.125
                successor = Execution(rows, feedback, dist, origin)
                edges.append(Edge(successor, mass*credit, None if addr is None else tape.byte(addr),
                                  literal, payload, ('literal','start','continue')[op]))
            if st.distance:
                if n-st.start < self.cap:
                    if not chosen or keep >= .5:
                        emit(2, rows, st.distance, st.start, n-st.distance, 1 if chosen else keep, rows[-1].payload)
                        if chosen: continue
                    if not chosen: mass = mass*(1-keep)
                rows += (Row(st.start, bytes(tape.raw[st.start:]), payload),)
            phrase = phrase_source(rows)
            addresses = sorted(set(raw) | ({phrase.start} if phrase else set()))
            features = []
            for a in addresses:
                hit = phrase is not None and a == phrase.start
                features.append(torch.stack((h.new_tensor(float(hit)), h.new_tensor((a-n)/n),
                                             phrase.payload.mean() if hit else h.new_zeros(()))))
            scores = m.operand(torch.stack(features)).flatten()
            weights = scores.softmax(0)
            if not chosen or start >= .5:
                for i in ([int(scores.argmax())] if chosen else range(len(addresses))):
                    a = addresses[i]; hit = phrase is not None and a == phrase.start
                    if hit: self.reads.append((n, a, len(rows)))
                    emit(1, rows, n-a, n, a, 1 if chosen else start*weights[i], phrase.payload if hit else rows[-1].payload)
            if not chosen or start < .5:
                emit(0, rows, 0, -1, None, 1 if chosen else 1-start, rows[-1].payload)
        dist = sum(e.weight * (e.literal if e.byte is None else torch.nn.functional.one_hot(ids.new_tensor(e.byte),256)) for e in edges)
        self.pending = edges, dist
        return dist

    def observe(self, byte):
        edges, dist = self.pending
        if not 0 <= byte <= 255 or dist[byte] <= 0:
            raise ValueError('impossible byte')
        n = len(self.tape.raw); paths = []
        for e in edges:
            st = e.state; weight = e.weight
            if e.byte is None:
                st = replace(st, rows=st.rows+(Row(n, bytes([byte]), e.payload),))
                weight = weight*e.literal[byte]
            elif byte != e.byte: continue
            paths.append((st, weight/dist[byte]))
        if len(paths) > self.path_limit:
            raise RuntimeError(f'exact path budget: {len(paths)} > {self.path_limit} at {n}')
        self.tape.append(bytes([byte])); self.paths = paths; self.pending = None
