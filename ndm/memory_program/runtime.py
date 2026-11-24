"""Reference executor. The runtime knows types, costs and limits, not the policy's plan."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import log
from uuid import uuid4
from typing import Callable, Sequence

import torch


class Refused(Exception):
    """An unavailable/incomplete or over-budget operation, NOT an exact empty answer."""


@dataclass(frozen=True)
class Span:
    source: str
    version: str
    document: int
    start: int
    stop: int

    def __post_init__(self):
        if (
            not all(isinstance(s, str) and s for s in (self.source, self.version))
            or not all(type(i) is int for i in (self.document, self.start, self.stop))
            or self.document < 0
            or not 0 <= self.start < self.stop
        ):
            raise ValueError("invalid logical span")


Atom = bytes | Span


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # text, byte (one-byte text), or span; columns are addressed by index
    origin: tuple[str, str, str, int | None] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.kind not in ("text", "byte", "span"):
            raise ValueError("invalid column")


def compatible(actual, wanted):
    return actual == wanted or (actual == "byte" and wanted == "text")


def valid(value, kind):
    if kind == "span":
        return isinstance(value, Span)
    return isinstance(value, bytes) and (kind != "byte" or len(value) == 1)


@dataclass(frozen=True)
class Hit:
    keys: tuple[Atom, ...]
    mass: int = 1
    value: torch.Tensor | None = None


@dataclass(frozen=True)
class Unavailable:
    reason: str


@dataclass(frozen=True, eq=False)
class Reader:
    """One trusted local, already authorized, version-bound capability.

    Return one complete result (including empty) or Unavailable PER key. Identity
    (not matching descriptor strings) permits coalescing immutable reads in a round.
    Providers must apply scope before count/rank/cap. Do not use physical node IDs as keys.
    """

    source: str
    version: str
    operation: str
    input_kind: str
    fields: tuple[Field, ...]
    read_many: Callable[[tuple[Atom, ...], int], Sequence[Sequence[Hit] | Unavailable]]

    batch_limit: int = 256
    session_id: str | None = None
    watermark: Callable[[], int] | None = None

    def __post_init__(self):
        if (
            not all(isinstance(s, str) and s for s in (self.source, self.version, self.operation))
            or not self.fields
            or type(self.batch_limit) is not int
            or self.batch_limit < 1
        ):
            raise ValueError("unbound reader")
        if (self.watermark is None) != (self.session_id is None) or (
            self.session_id is not None and self.version != self.session_id
        ):
            raise ValueError("live reader version must bind its session and watermark")
        Field("input", self.input_kind)


@dataclass(frozen=True)
class Page:
    fields: tuple[Field, ...]
    keys: tuple[tuple[Atom, ...], ...]
    log_mass: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class Limits:
    steps: int = 6  # includes EMIT; last step only permits EMIT
    recent: int = 32  # byte span pointer horizon (saved tables can be older)
    pattern: int = 8
    rows: int = 256  # exact max per intermediate; never keep top rows silently
    columns: int = 12
    cells: int = 4  # persistent pages per document
    requests: int = 32  # sum of distinct logical keys submitted in each round

    def __post_init__(self):
        if self.columns < 2 or any(type(x) is not int or x < 1 for x in vars(self).values()):
            raise ValueError("positive budgets required")


@dataclass(frozen=True)
class Instruction:
    op: str
    args: tuple[int, ...] = ()


class Session:
    """One observation stream, one fixed mount scope. Create a fresh session at reset."""

    def __init__(self, readers=(), limits=Limits(), advance=None, *, session_id=None, document=0, start=0):
        self.readers = tuple(readers)
        self.limits = limits
        self.advance = advance
        if len({(r.source, r.version, r.operation) for r in self.readers}) != len(self.readers):
            raise ValueError("duplicate binding")
        self.id = session_id or uuid4().hex
        if not isinstance(self.id, str) or any(type(x) is not int or x < 0 for x in (document, start)):
            raise ValueError("invalid observation stream binding")
        self.document = document
        self.start = start  # origin of this window inside its document
        if any(
            r.session_id is not None and (r.session_id != self.id or r.watermark() != 0) for r in self.readers
        ):
            raise ValueError("wrong live session or nonempty initial watermark")
        self.cells = [None] * limits.cells
        self.recent = b""
        self.visible = 0
        self.pending = False

    def detach_values(self):
        """Explicit truncated-BPTT boundary; use fresh sessions when parameters change."""
        self.cells = [
            None if p is None else replace(p, value=p.value.detach(), log_mass=p.log_mass.detach())
            for p in self.cells
        ]

    def observe(self, data: bytes):
        if not isinstance(data, bytes) or not data:
            raise ValueError("observed bytes required")
        if self.advance is not None:
            self.advance(data)
        if any(r.watermark and r.watermark() != self.visible + len(data) for r in self.readers):
            raise ValueError("live append mismatch")
        self.recent = (self.recent + data)[-self.limits.recent :]
        self.visible += len(data)
        self.pending = False


OPS = ("SLICE", "LOOKUP", "FOCUS", "PROJECT", "CONCAT", "JOIN", "RELATE", "SAVE", "RECALL", "EMIT")


class Frame:
    """Fresh SSA-style working registers each prediction; commit writes only on EMIT."""

    def __init__(self, session):
        if session.pending:
            raise ValueError("observe before another prediction")
        self.s = session
        self.l = session.limits
        self.reg = []
        self.cells = list(session.cells)
        self.requests = 0
        self.read_width = {
            kind: min(
                (len(m.fields) for m in session.readers if compatible(kind, m.input_kind)),
                default=self.l.columns + 1,
            )
            for kind in ("text", "byte", "span")
        }

    def domain(self, op, args=()):
        """Type mask for ONE operand given preceding operands. No result/target lookahead."""
        r = self.reg
        n = len(args)
        a = args
        if op == "SLICE":
            if not n:
                return "back", range(len(self.s.recent))
            return "length", range(1, 1 + min(self.l.pattern, len(self.s.recent) - a[0]))
        if op == "LOOKUP":

            def readers(i, c):
                return [
                    j
                    for j, m in enumerate(self.s.readers)
                    if compatible(r[i].fields[c].kind, m.input_kind)
                    and len(r[i].fields) + len(m.fields) <= self.l.columns
                ]

            def fields(i):
                return [
                    c
                    for c, f in enumerate(r[i].fields)
                    if len(r[i].fields) + self.read_width[f.kind] <= self.l.columns
                ]

            if not n:
                return "reg", [i for i in range(len(r)) if fields(i)]
            if n == 1:
                return "field", fields(a[0])
            return "reader", readers(*a)
        if op == "PROJECT":
            return ("reg", range(len(r))) if not n else ("field", range(len(r[a[0]].fields)))
        if op == "CONCAT":

            def strings(i):
                return [j for j, f in enumerate(r[i].fields) if compatible(f.kind, "text")]

            if not n:
                return "reg", [i for i, p in enumerate(r) if len(p.fields) < self.l.columns and strings(i)]
            return "field", strings(a[0])
        if op == "RELATE":

            def rhs(i):
                return [j for j, q in enumerate(r) if len(q.fields) + len(r[i].fields) <= self.l.columns]

            shortest = min((len(p.fields) for p in r), default=self.l.columns + 1)
            return (
                ("reg", [i for i, p in enumerate(r) if len(p.fields) + shortest <= self.l.columns])
                if not n
                else ("reg", rhs(a[0]))
            )
        if op == "JOIN":

            def fields2(i, c, j):
                f = r[i].fields[c]
                q = r[j]
                if len(r[i].fields) + len(q.fields) > self.l.columns:
                    return []
                return [
                    b
                    for b, g in enumerate(q.fields)
                    if compatible(f.kind, g.kind) or compatible(g.kind, f.kind)
                ]

            def rhs(i, c):
                return [j for j in range(len(r)) if fields2(i, c, j)]

            # A schema-only existence test, not enumeration of candidate join plans.
            widths = {
                kind: min(
                    (
                        len(q.fields)
                        for q in r
                        if any(compatible(f.kind, kind) or compatible(kind, f.kind) for f in q.fields)
                    ),
                    default=self.l.columns + 1,
                )
                for kind in ("text", "byte", "span")
            }

            def lhs(i):
                return [
                    c
                    for c, f in enumerate(r[i].fields)
                    if len(r[i].fields) + widths[f.kind] <= self.l.columns
                ]

            if not n:
                return "reg", [i for i in range(len(r)) if lhs(i)]
            if n == 1:
                return "field", lhs(a[0])
            if n == 2:
                return "reg", rhs(*a)
            return "field", fields2(*a)
        if op == "FOCUS":
            if not n:
                return "reg", [i for i, p in enumerate(r) if p.keys]
            return "row", range(len(r[a[0]].keys))
        if op == "SAVE":
            return ("reg", range(len(r))) if not n else ("cell", range(self.l.cells))
        if op == "RECALL":
            return "cell", [j for j, c in enumerate(self.cells) if c is not None]
        if op == "EMIT":
            if not n:
                return "reg", [-1] + [i for i, p in enumerate(r) if p.keys]
            return "field", [-1] + (
                [j for j, f in enumerate(r[a[0]].fields) if f.kind == "byte"] if a[0] >= 0 else []
            )
        raise ValueError("unknown op")

    def request(self, args):
        i, c, j = args
        keys = tuple(dict.fromkeys(row[c] for row in self.reg[i].keys))
        if self.requests + len(keys) > self.l.requests:
            raise Refused("key budget")
        reader = self.s.readers[j]
        if reader.watermark and reader.watermark() != self.s.visible:
            raise ValueError("stale live watermark")
        self.requests += len(keys)
        return reader, keys

    def execute(self, inst, h, model, table=None, *, get=None):
        """Execute ONE already type-checked non-emit instruction, no candidate plan lookahead."""
        op, a = inst.op, inst.args
        r = self.reg
        d = h.numel()
        if op == "SLICE":
            back, k = a
            end = len(self.s.recent) - back
            key = self.s.recent[end - k : end]
            stop = self.s.visible - back
            loc = Span("tape", self.s.id, self.s.document, self.s.start + stop - k, self.s.start + stop)
            origin = ("tape", self.s.id, "slice", self.s.visible)
            return Page(
                (Field("pattern", "text", origin), Field("location", "span", origin)),
                ((key, loc),),
                h.new_zeros(1),
                h.new_zeros((1, d)),
            )
        if op == "RECALL":
            return self.cells[a[0]]
        p = r[a[0]]
        if op == "FOCUS":
            j = a[1]
            return Page(p.fields, (p.keys[j],), p.log_mass[j : j + 1], p.value[j : j + 1])
        if op == "PROJECT":
            j = a[1]
            return Page((p.fields[j],), tuple((row[j],) for row in p.keys), p.log_mass, p.value)
        if op == "CONCAT":
            _, i, j = a
            if any(len(row[i]) + len(row[j]) > self.l.pattern for row in p.keys):
                raise Refused("constructed key exceeds byte budget")
            f = Field("composed", "text", ("computed", self.s.id, "concat", self.s.visible))
            return Page(
                p.fields + (f,), tuple(row + (row[i] + row[j],) for row in p.keys), p.log_mass, p.value
            )
        if op == "SAVE":
            v = model.write_values(h, p) if get is None else model.write_values(h, p, get=get)
            out = Page(p.fields, p.keys, p.log_mass, v)
            self.cells[a[1]] = out
            return out
        if op == "JOIN":
            _, i, qid, j = a
            q = r[qid]
            index = defaultdict(list)
            for k, key in enumerate(q.keys):
                index[key[j]].append(k)
            left = []
            right = []
            for k, key in enumerate(p.keys):
                hits = index.get(key[i], ())
                if len(left) + len(hits) > self.l.rows:
                    raise Refused("join exceeds row budget")
                left.extend([k] * len(hits))
                right.extend(hits)
            keys = tuple(p.keys[l] + q.keys[b] for l, b in zip(left, right, strict=True))
            return Page(
                p.fields + q.fields,
                keys,
                p.log_mass[left] + q.log_mass[right],
                model.merge(p.value[left], q.value[right], h),
            )
        if op == "RELATE":
            q = r[a[1]]
            n = len(p.keys)
            m = len(q.keys)
            if n * m > self.l.rows:
                raise Refused("pair relation exceeds row budget")
            # All pairs under an explicit cap. Both bindings survive; no equality is asserted.
            score = model.affinity(p, q, h) if get is None else model.affinity(p, q, h, get=get)
            mass = (p.log_mass[:, None] + q.log_mass[None, :] + score).reshape(-1)
            values = model.merge(p.value.repeat_interleave(m, 0), q.value.repeat(n, 1), h)
            return Page(p.fields + q.fields, tuple(x + y for x in p.keys for y in q.keys), mass, values)
        if op != "LOOKUP":
            raise ValueError("not a read/write instruction")
        _, column, reader_id = a
        reader = self.s.readers[reader_id]
        if table is None:
            raise ValueError("lookup requires dispatched answers")
        keys = []
        idx = []
        vals = []
        mass = []
        for i, row in enumerate(p.keys):
            hits = table[row[column]]
            if isinstance(hits, Unavailable):
                raise Refused(hits.reason)
            if len(keys) + len(hits) > self.l.rows:
                raise Refused("expanded read exceeds row budget")
            for hit in hits:
                if (
                    type(hit.mass) is not int
                    or not 0 < hit.mass < 2**63
                    or len(hit.keys) != len(reader.fields)
                    or not all(valid(k, f.kind) for k, f in zip(hit.keys, reader.fields, strict=True))
                ):
                    raise ValueError("invalid provider row")
                if hit.value is not None and (
                    hit.value.shape != h.shape or hit.value.device != h.device or hit.value.dtype != h.dtype
                ):
                    raise ValueError("value provider schema/device mismatch")
                keys.append(row + hit.keys)
                idx.append(i)
                vals.append(hit.value)
                mass.append(log(hit.mass))
        v = p.value[idx]
        weights = p.log_mass[idx].float() + torch.tensor(mass, device=h.device, dtype=torch.float32)
        if any(x is not None for x in vals):
            v = v + torch.stack([h.new_zeros(d) if x is None else x for x in vals])
        watermark = reader.watermark() if reader.watermark else None
        if reader.watermark and watermark != self.s.visible:
            raise ValueError("live mount changed during lookup")
        origin = (reader.source, reader.version, reader.operation, watermark)
        return Page(
            p.fields + tuple(replace(f, origin=origin) for f in reader.fields), tuple(keys), weights, v
        )


def branch_reader(source, version, get_many, *, structural_end=False, **binding):
    """Two typed projections of a 257-bin answer. END has no byte/extension binding."""

    def read(keys, cap):
        rows = []
        for k, h in zip(keys, get_many(keys), strict=True):
            if h is None or isinstance(h, Unavailable):
                rows.append(Unavailable("branch not materialized") if h is None else h)
                continue
            if len(h) != 257 or any(type(v) is not int or v < 0 for v in h):
                raise ValueError("invalid branch histogram")
            if structural_end:
                rows.append((Hit((k,), h[256]),) if h[256] else ())
                continue
            symbols = [(bytes([b]), v) for b, v in enumerate(h[:256]) if v]
            rows.append(
                Unavailable("branch exceeds row budget")
                if len(symbols) > cap
                else tuple(Hit((s, k + s), v) for s, v in symbols)
            )
        return rows

    if structural_end:
        return Reader(source, version, "end", "text", (Field("closed_at", "text"),), read, **binding)
    return Reader(
        source,
        version,
        "byte_edges",
        "text",
        (Field("symbol", "byte"), Field("extension", "text")),
        read,
        **binding,
    )
