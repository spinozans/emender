"""One visible tape per window, one committed phrase history per execution."""
from dataclasses import dataclass
from math import log1p

import torch
from torch import nn


@dataclass(frozen=True)
class Factor:
    # None means literal (operand is byte); otherwise operand is distance.
    operand: int
    length: int | None = None


def greedy(text, cap=8):
    """Observed-stream teacher only. Never called by predict or observe."""
    if cap < 1:
        raise ValueError('nonpositive cap')
    i = 0
    while i < len(text):
        best = Factor(text[i])
        for d in range(1, i+1):
            k = 0
            while k < cap and i+k < len(text) and text[i+k] == text[i+k-d]:
                k += 1
            if k > (best.length or 1):
                best = Factor(d, k)
        yield best
        i += best.length or 1


@dataclass(frozen=True)
class Phrase:
    start: int
    content: bytes  # canonical by equality, independent of parse/source pointer
    payload: float


def commit(tape, rows, start):
    expected = rows[-1].start + len(rows[-1].content) if rows else 0
    if start != expected or not start < len(tape):
        raise ValueError('noncontiguous or empty phrase')
    content = bytes(tape[start:])
    return rows + (Phrase(start, content, sum(content)/(255*len(content))),)


def decode(stream, cap=8):
    tape = bytearray(); rows = ()
    for factor in stream:
        i = len(tape); d = factor.operand
        if factor.length is None:
            tape.append(d)
        elif 1 <= d <= i and 1 <= factor.length <= cap:
            for _ in range(factor.length):
                tape.append(tape[-d])
        else:
            raise ValueError('illegal COPY')
        rows = commit(tape, rows, i)
    return bytes(tape), rows


def phrase_source(rows):
    for depth in (2, 1):
        suffix = [r.content for r in rows[-depth:]]
        for j in range(len(rows)-1, depth-1, -1):
            if [r.content for r in rows[j-depth:j]] == suffix:
                return rows[j]  # continuation aligned to a committed span
    return None


class Tape:
    def __init__(self, session):
        self.session = session
        self.raw = bytearray()

    def append(self, chunk):
        if self.session.rpc('append '+chunk.hex())['visible'] != len(self.raw)+len(chunk):
            raise RuntimeError('native visibility mismatch')
        self.raw.extend(chunk)

    def byte(self, address):
        if not 0 <= address < len(self.raw):
            raise ValueError('address outside visible tape')
        y = self.session.rpc(f'byte {address}')['byte']
        if y != self.raw[address]:
            raise RuntimeError('native byte mismatch')
        return y

    def sources(self):
        n = len(self.raw)
        if n <= 2:
            return [n-1] if n else []
        hits = self.session.rpc(f'locate {self.raw[-2:].hex()} 0 {n-1} {n}')['positions']
        return sorted({n-1, *(i+2 for i in hits[-2:])})


class Controller(nn.Module):
    """Two binary log odds heads (idle/copy phase) and shared operand scorer."""
    def __init__(self):
        super().__init__()
        self.phase = nn.Linear(3, 2)
        self.address = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.phase.weight.copy_(torch.tensor([[.5, -.5, 0.], [0., 0., -.5]]))
            self.phase.bias.copy_(torch.tensor([.25, 1.]))
            self.address.weight.copy_(torch.tensor([[.5, .5, .25]]))

    def odds(self, phase, support, confidence, used):
        x = self.phase.weight.new_tensor([log1p(support), confidence, used])
        return self.phase(x)[phase].clamp(-8, 8).sigmoid()

    def operands(self, addresses, n, phrase):
        x = self.address.weight.new_tensor([
            [(a-n)/n, float(phrase is not None and a == phrase.start),
             phrase.payload if phrase is not None and a == phrase.start else 0.]
            for a in addresses])
        return self.address(x).flatten().clamp(-8, 8).softmax(0)


@dataclass(frozen=True)
class State:
    rows: tuple
    start: int = -1  # pending COPY destination start, not yet committed
    distance: int = 0


class Filter:
    """Explicit path sum. Size exponential in observed steps; no beam or merge."""
    def __init__(self, session, controller, prefix=b'', cap=8):
        if cap < 1:
            raise ValueError('nonpositive cap')
        self.tape = Tape(session); self.controller = controller; self.cap = cap
        _, rows = decode(Factor(y) for y in prefix)
        if prefix:
            self.tape.append(prefix)
        self.paths = [(State(rows), controller.phase.weight.new_tensor(1.))]
        self.pending = None

    def predict(self, literal_logits):
        if self.pending is not None or literal_logits.shape != (256,):
            raise ValueError('expected next prediction and 256 literal logits')
        base = literal_logits.softmax(0) * (1-1e-6) + 1e-6/256
        n = len(self.tape.raw); raw = self.tape.sources(); edges = []
        for state, mass in self.paths:
            rows = state.rows
            if state.distance:
                used = n-state.start
                if used < self.cap:
                    keep = self.controller.odds(1, 0, base.max().item(), used/self.cap)
                    edges.append((state, mass*keep, self.tape.byte(n-state.distance)))
                    mass = mass*(1-keep)
                rows = commit(self.tape.raw, rows, state.start)
            phrase = phrase_source(rows)
            addresses = sorted(set(raw) | ({phrase.start} if phrase else set()))
            if addresses:
                start = self.controller.odds(0, len(addresses), base.max().item(), 0.)
                weights = self.controller.operands(addresses, n, phrase)
                for a, w in zip(addresses, weights):
                    edges.append((State(rows, n, n-a), mass*start*w, self.tape.byte(a)))
                mass = mass*(1-start)
            edges.append((State(rows), mass, None))
        dist = base * sum(m for _, m, y in edges if y is None)
        copies = [(m, y) for _, m, y in edges if y is not None]
        if copies:
            dist = dist.index_add(0, torch.tensor([y for _, y in copies], device=base.device),
                                  torch.stack([m for m, _ in copies]))
        self.pending = edges, base, dist
        return dist

    def observe(self, y):
        if self.pending is None or not 0 <= y < 256:
            raise ValueError('predict before byte observation')
        edges, base, dist = self.pending
        n = len(self.tape.raw); self.tape.append(bytes([y])); paths = []
        for state, mass, emitted in edges:
            if emitted is None:
                state = State(commit(self.tape.raw, state.rows, n)); mass = mass*base[y]
            elif emitted != y:
                continue
            paths.append((state, mass/dist[y]))
        self.paths = paths; self.pending = None
