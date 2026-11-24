"""Exclusive one-frontier transport for two executions under the SAME admitted structural readers.

No persistent immutable cache, no cross-user deduplication, no generic pass-through of tensors.
This trusted in-process adapter supports the four registered structural schemas only. Authority
comes from the owner's raw Reader objects and live guard, never a matching source string.
"""

from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
import os
from threading import get_ident

from .runtime import Hit, Reader, Span, Unavailable, valid
from .spans import mean_byte_values

_SCHEMA = {
    "byte_edges": ("text", (("symbol", "byte"), ("extension", "text"))),
    "occurrences": ("text", (("occurrence", "span"),)),
    "contents": ("span", (("text", "text"),)),
    "following": ("span", (("first", "byte"), ("text", "text"), ("region", "span"))),
}


def schema(r):
    if not isinstance(r, Reader) or _SCHEMA.get(r.operation) != (
        r.input_kind,
        tuple((f.name, f.kind) for f in r.fields),
    ):
        raise ValueError("unsupported structural reader schema")


def for_model(raw, op, weight, meter=None):
    """The text itself is a structural binding. Values are always built in this model's own graph."""
    if op not in ("contents", "following"):
        return raw
    col = 0 if op == "contents" else 1
    text = [h.keys[col] for hits in raw if not isinstance(hits, Unavailable) for h in hits]
    if not text:
        return raw
    vec = mean_byte_values(text, weight)
    if meter is not None:
        meter.update(calls=1, fragments=len(text), text_bytes=sum(map(len, text)), components=vec.numel())
    values = iter(vec)
    return tuple(
        hits if isinstance(hits, Unavailable) else tuple(replace(h, value=next(values)) for h in hits)
        for hits in raw
    )


@dataclass(frozen=True)
class StageLimit:
    frames: int = 512
    keys: int = 1024
    rows: int = 32768
    text_bytes: int = 8 << 20

    def __post_init__(self):
        if any(
            type(v) is not int or not 0 < v <= m
            for v, m in zip(vars(self).values(), (4096, 8192, 131072, 32 << 20), strict=True)
        ):
            raise ValueError("invalid staging budget")


def _size(k):
    if type(k) is bytes:
        return len(k)
    if type(k) is Span:
        return len(k.source.encode()) + len(k.version.encode()) + 24
    raise ValueError("untyped staged atom")


class FrontierReads:
    def __init__(self, guard, limit):
        if not callable(guard) or not isinstance(limit, StageLimit):
            raise ValueError("bound reader guard and limits required")
        self.owner = os.getpid(), get_ident()
        self.guard = guard
        self.limit = limit
        self.phase = "idle"
        self._calls = []
        self._next = 0
        self._work = Counter()
        self.total = Counter()
        self.peak = Counter()
        self.encoders = {phase: Counter() for phase in ("issue", "consume")}

    def _expect(self, label):
        if self.owner != (os.getpid(), get_ident()) or self.phase != label:
            raise ValueError("foreign owner or inactive frontier read phase")

    def begin(self):
        self._expect("idle")
        self.phase = "issue"

    def handoff(self):
        self._expect("issue")
        self.phase = "consume"

    def finish(self):
        self._expect("consume")
        if self._next != len(self._calls):
            self.close()
            raise ValueError("unconsumed structural replies")
        self.total.update(self._work)
        self.peak |= self._work
        self._clear()
        self.phase = "idle"

    def _clear(self):
        self._calls.clear()
        self._next = 0
        self._work.clear()

    def close(self):
        if self.owner != (os.getpid(), get_ident()):
            raise ValueError("foreign frontier owner")
        self._clear()
        self.guard = None
        self.phase = "closed"

    def _add(self, name, count):
        self._work[name] += count
        if self._work[name] > getattr(self.limit, name):
            raise ValueError("frontier staging budget exceeded: " + name)

    def view(self, r, phase, weight):
        if self.owner != (os.getpid(), get_ident()) or self.phase == "closed":
            raise ValueError("closed or foreign frontier owner")
        schema(r)
        if phase not in ("issue", "consume"):
            raise ValueError("invalid frontier read role")
        return replace(r, read_many=partial(self._fetch, phase, r, weight))

    def _fetch(self, phase, r, weight, keys, cap):
        self._expect(phase)
        try:
            self.guard(r)
            wm = r.watermark() if r.watermark else None
            if (
                type(keys) is not tuple
                or not 0 < len(keys) <= r.batch_limit
                or type(cap) is not int
                or not 0 <= cap <= 4096
            ):
                raise ValueError("invalid structural request")
            if not all(valid(k, r.input_kind) for k in keys):
                raise ValueError("wrong structural key type")
            if phase == "issue":
                self._add("frames", 1)
                self._add("keys", len(keys))
                self._add("text_bytes", sum(map(_size, keys)))
                result = r.read_many(keys, cap)
                if type(result) not in (tuple, list) or len(result) != len(keys):
                    raise ValueError("incomplete structural reply")
                raw = []
                for hits in result:
                    if type(hits) is Unavailable:
                        if type(hits.reason) is not str or not hits.reason:
                            raise ValueError("invalid structural refusal")
                        self._add("text_bytes", len(hits.reason.encode()))
                        raw.append(hits)
                        continue
                    if type(hits) not in (tuple, list) or len(hits) > cap:
                        raise ValueError("unbounded structural reply")
                    self._add("rows", len(hits))
                    for h in hits:
                        if (
                            type(h) is not Hit
                            or h.value is not None
                            or type(h.mass) is not int
                            or not 0 < h.mass < 2**63
                            or type(h.keys) is not tuple
                            or len(h.keys) != len(r.fields)
                            or not all(valid(k, f.kind) for k, f in zip(h.keys, r.fields, strict=True))
                        ):
                            raise ValueError("nonstructural or malformed provider row")
                        self._add("text_bytes", sum(map(_size, h.keys)))
                    raw.append(tuple(hits))
                raw = tuple(raw)
                if r.watermark and wm != r.watermark():
                    raise ValueError("live reader advanced during issue")
                self.guard(r)
                self._calls.append((r, keys, cap, wm, raw))
            else:
                if self._next >= len(self._calls):
                    raise ValueError("no such issued read")
                actual, ks, c, w, raw = self._calls[self._next]
                if actual is not r or (ks, c, w) != (keys, cap, wm):
                    raise ValueError("different issued capability, keys, cap or watermark")
                self._next += 1
            return for_model(raw, r.operation, weight, self.encoders[phase])
        except BaseException:
            self.close()
            raise
