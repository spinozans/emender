"""Bounded decode loop. Only the model returns FOLLOW and its edge operand."""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import torch

from . import workbench_model as m


class Meter:
    def __init__(self, device='cpu', *, trace=False, profile=False):
        self.device, self.trace, self.profile = torch.device(device), trace, profile
        self.reads = self.passes = 0
        self.requests, self.views, self.stages = [], [], []

    def timed(self, stage, fn, *, fence=False):
        if not self.profile:
            return fn()
        sync = fence and self.device.type == 'cuda'
        if sync:
            torch.cuda.synchronize(self.device)
        start = perf_counter_ns()
        result = fn()
        if sync:
            torch.cuda.synchronize(self.device)
        self.stages.append((stage, perf_counter_ns()-start))
        return result


def compact(values):
    """Lossless default + deviations (indexes in byte order, END=256 for counts)."""
    v = values.tolist() if hasattr(values, 'tolist') else list(values)
    default = Counter(v).most_common(1)[0][0]
    return {'default': default, 'other': [[i,t] for i,t in enumerate(v) if t != default]}


@dataclass(frozen=True)
class Answer:
    key: bytes
    h: tuple[int, ...]


def branch(bank, key, meter, source):
    h = meter.timed('branch_rpc', lambda: bank.branch(key))
    meter.reads += 1
    if meter.trace:
        meter.requests.append({'source': source, 'key': key.hex(), 'h': compact(h)})
    return Answer(key, h)


def backoff(bank, tail, meter, source):
    last = None
    for length in dict.fromkeys(n for n in (len(tail), 4, 2) if 0 < n <= len(tail)):
        last = branch(bank, tail[-length:], meter, source)
        if any(last.h):  # includes structural END; no byte evidence if END-only
            break
    return last


@dataclass(frozen=True)
class Handle:
    source: str
    version: str
    key: bytes


@dataclass(frozen=True)
class Record:
    handle: Handle
    session_id: str
    next_visible: int
    value: torch.Tensor

    def matches(self, live, archive):
        return (archive is not None and self.handle.source == archive.name and
                self.handle.version == archive.version and self.session_id == live.session_id and
                self.next_visible == live.visible and self.handle.key[-1:] == live.tail[-1:])


class Decoder:
    def __init__(self, model, live, archive, *, speculate=True):
        self.model, self.live, self.archive = model, live, archive
        self.device = next(model.parameters()).device
        self.speculate = speculate
        self.record = None
        self.handles = ()
        self.awaiting_observation = False

    def erase_payload(self):
        r = self.record
        if r is None:
            raise ValueError('no write to erase')
        self.record = Record(r.handle, r.session_id, r.next_visible, torch.zeros_like(r.value))

    def observe(self, byte, meter):
        if not self.awaiting_observation:
            raise ValueError('observe only after prediction')
        meter.timed('append_rpc', lambda: self.live.append(bytes([byte])))
        self.awaiting_observation = False

    def predict(self, meter):
        if self.awaiting_observation:
            raise ValueError('prediction already issued at this prefix')
        tail = self.live.tail
        record = self.record
        match = record is not None and record.matches(self.live, self.archive)
        base_done = recalled = False
        live = archive = child = None
        # Only bookmarks in the active mount may be shown to this model.
        self.handles = tuple(h for h in self.handles if self.archive is not None and
                             (h.source,h.version) == (self.archive.name,self.archive.version))
        for _ in range(3):
            flags = (bool(tail), base_done, match, self.speculate, recalled, child is not None)
            keys = {i: h.key for i,h in zip((m.BOOK0,m.BOOK1), self.handles)}
            for i,ans in ((m.LIVE,live),(m.ARCH,archive),(m.CHILD,child)):
                if ans is not None:
                    keys[i] = ans.key
            if recalled:
                keys[m.ROW] = record.handle.key
            cpu = meter.timed('pack', lambda: m.pack(tail, live.h if live else None,
                              archive.h if archive else None, child.h if child else None, flags, keys))
            x = meter.timed('transfer', lambda: cpu.to(self.device), fence=True)
            packet, prob, value = meter.timed('forward', lambda: self.model(
                x, record.value if recalled else None), fence=True)
            # Return only the bounded action operands, not a candidate page for host scoring.
            op_id, first, second, greedy = meter.timed('decision_return', lambda: packet.tolist(), fence=True)
            op = m.Op(op_id)
            meter.passes += 1
            if meter.trace:
                meter.views.append({'visible': self.live.visible, 'recent': tail.hex(),
                    'archive_version': self.archive.version if self.archive else None,
                    'keys_by_slot': {str(i): k.hex() for i,k in keys.items()},
                    'flags': list(flags), 'reads_so_far': meter.reads,
                    'recall': compact(record.value.detach()) if recalled else None,
                    'op': op.name, 'edges': [first, second], 'greedy': greedy,
                    'write': compact(value.detach()) if op == m.Op.WRITE_EMIT else None})
            if op in (m.Op.BASE, m.Op.BASE_RECALL):
                if base_done or not tail or (op == m.Op.BASE_RECALL and not match):
                    raise ValueError('illegal base read')
                live = backoff(self.live, tail, meter, 'live')
                recalled = op == m.Op.BASE_RECALL
                if self.archive is not None and not recalled:
                    archive = backoff(self.archive, tail, meter, 'archive')
                base_done = True
            elif op == m.Op.FOLLOW:
                if (not self.speculate or child is not None or archive is None or
                        not 0 <= first < 256 or not archive.h[first] or second == first or
                        (second != -1 and not (0 <= second < 256 and archive.h[second]))):
                    raise ValueError('illegal dependent read')
                self.handles = tuple(Handle(self.archive.name, self.archive.version, archive.key + bytes([b]))
                                     for b in (first, second) if b >= 0)
                child = branch(self.archive, self.handles[0].key, meter, 'archive')
            else:
                self.record = None
                if op == m.Op.WRITE_EMIT:
                    if child is None or not any(child.h[:256]) or self.handles[0].key != child.key:
                        raise ValueError('write without a child value')
                    self.record = Record(self.handles[0], self.live.session_id, self.live.visible+1, value)
                self.awaiting_observation = True
                return prob, greedy
        raise ValueError('model exceeded two dependent read rounds')


class Direct:
    """Cheap baseline: same backoff and source weights, no neural model, no writes."""
    def __init__(self, live, archive):
        self.live, self.archive = live, archive

    def predict(self, meter):
        p, weight = np.full(256, 1/256, dtype=np.float64), 1
        for bank,w,name in ((self.live,1,'live'),(self.archive,4,'archive')):
            if bank is not None and self.live.tail:
                ans = backoff(bank, self.live.tail, meter, name)
                h = np.array(ans.h[:256]); n = h.sum()
                if n:
                    p += w*h/n; weight += w
        p /= weight
        return p, int(p.argmax())

    def observe(self, byte, meter):
        meter.timed('append_rpc', lambda: self.live.append(bytes([byte])))
