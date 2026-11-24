"""Checked *executed-choice* control for training values, not a generative policy or an answer cache.

A trusted owner first samples an immutable selector on real observed frontiers. Replaying its integer
instructions still executes real, scoped reads. A guard checks ordered requests and structured results
before the next dependent round. It intentionally does not compare or reuse learned values/pair weights.
Never give this private experimental control tape to the live learned arm or an untrusted decoder.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from hashlib import sha256
import torch
from .runtime import Instruction, Span, Unavailable
from .training import teacher_force


def digest(x):
    return sha256(repr(x).encode("utf-8")).digest()


@dataclass(frozen=True, slots=True)
class Choice:
    instruction: Instruction
    request: bytes | None
    result: bytes


@dataclass(frozen=True, slots=True)
class RouteTape:
    authority: str  # trusted binding for encoder, frozen selector, RNG scheme, source and mount scope
    origins: tuple
    programs: tuple  # document -> visible byte -> Choice per instruction

    def instructions(self):
        return [[[v.instruction for v in byte] for byte in doc] for doc in self.programs]


class _Window:
    def __init__(self, s, word):
        self.s = s
        self.doc = (s.document, s.start)
        # Rebind actual private readers and the two executor-owned origins, never immutable versions.
        self.local = {("tape", s.id), ("computed", s.id)} | {
            (r.source, r.version) for r in s.readers if r.session_id == s.id
        }
        if any((r.source, r.version) in self.local and r.session_id is None for r in s.readers):
            raise ValueError("ambiguous local/immutable origin")
        self.t = 0
        self.depth = 0
        self.request = None
        if word is None:  # a current-frontier observer has no future window or labels
            return
        self.length = len(word) - 1
        schema = tuple(
            (
                r.source,
                self.version(r.source, r.version),
                r.operation,
                r.input_kind,
                tuple(self.field(f) for f in r.fields),
                r.batch_limit,
                r.session_id is not None,
            )
            for r in s.readers
        )
        self.origin = (
            self.doc,
            self.length,
            sha256(word[:-1]).digest(),
            digest((astuple(s.limits), schema)),
        )

    def version(self, source, version):
        return ("private", *self.doc) if (source, version) in self.local else version

    def binding(self, o):
        return (o[0], self.version(o[0], o[1]), o[2], o[3]) if o is not None else None

    def field(self, f):
        return (f.name, f.kind, self.binding(f.origin))

    def key(self, k):
        if type(k) is bytes:
            return k
        if isinstance(k, Span):
            return (
                "span",
                k.source,
                self.version(k.source, k.version),
                k.document,
                k.start,
                k.stop,
            )
        raise ValueError("unknown structural key")

    def keys(self, ks):
        return tuple(self.key(k) for k in ks)


class _Observer:
    def __init__(self, sessions, words, authority, max_steps, expected=None):
        if (
            type(authority) is not str
            or len(authority) != 64
            or any(c not in "0123456789abcdef" for c in authority)
            or len(sessions) != len(words)
            or not words
            or any(type(w) is not bytes or len(w) < 2 for w in words)
            or type(max_steps) is not int
            or max_steps < 1
            or sum((len(w) - 1) * s.limits.steps for s, w in zip(sessions, words)) > max_steps
            or len({s.id for s in sessions}) != len(words)
            or len({(s.document, s.start) for s in sessions}) != len(words)
            or any(s.visible or s.pending or any(c is not None for c in s.cells) for s in sessions)
        ):
            raise ValueError("invalid or over-budget fresh route episode")
        self.authority = authority
        self.expected = expected
        self.rows = [_Window(s, w) for s, w in zip(sessions, words)]
        self.index = {s.id: i for i, s in enumerate(sessions)}
        self.origins = tuple(r.origin for r in self.rows)
        if expected is not None and (
            not isinstance(expected, RouteTape)
            or expected.authority != authority
            or expected.origins != self.origins
            or len(expected.programs) != len(self.rows)
            or any(
                len(doc) != r.length or any(not 0 < len(byte) <= r.s.limits.steps for byte in doc)
                for doc, r in zip(expected.programs, self.rows)
            )
        ):
            raise ValueError("wrong frozen authority, origin, schema or observation digest")
        self.out = None if expected is not None else [[[] for _ in range(r.length)] for r in self.rows]

    def _here(self, s, step):
        i = self.index[s.id]
        r = self.rows[i]
        if r.s is not s or s.visible != r.t + 1 or step != r.depth or r.t >= r.length:
            raise ValueError("stale route frontier")
        return (
            i,
            r,
            self.expected.programs[i][r.t][step] if self.expected is not None else None,
        )

    def request(self, s, step, inst, reader, cap, keys):
        _, r, expect = self._here(s, step)
        origin = (
            reader.source,
            reader.version,
            reader.operation,
            reader.watermark() if reader.watermark else None,
        )
        h = digest(
            (
                r.doc,
                s.visible,
                step,
                inst.op,
                inst.args,
                r.binding(origin),
                cap,
                r.keys(keys),
            )
        )
        if r.request is not None or (
            expect is not None and (expect.request != h or expect.instruction != inst)
        ):
            raise ValueError("changed frozen request before dispatch")
        r.request = h

    def event(self, s, step, e, page, table):
        i, r, expect = self._here(s, step)
        layout = (
            None
            if page is None
            else (
                tuple(r.field(f) for f in page.fields),
                tuple(r.keys(row) for row in page.keys),
            )
        )
        answers = []
        if table is not None:
            for k in e.keys:  # ONLY this client's chosen keys, not peers' physical batch.
                hits = table[k]
                if isinstance(hits, Unavailable):
                    answers.append(("unavailable", hits.reason))
                else:
                    if len(hits) > s.limits.rows:
                        raise ValueError("provider returned unbounded rows")
                    answers.append(tuple((r.keys(h.keys), h.mass, h.value is not None) for h in hits))
        inst = e.inst
        h = digest(
            (
                r.doc,
                s.visible,
                step,
                inst.op,
                inst.args,
                e.status,
                e.register,
                e.rows,
                r.binding(e.reader),
                r.keys(e.keys),
                layout,
                tuple(answers),
            )
        )
        choice = Choice(inst, r.request, h)
        if expect is not None:
            if expect != choice:
                raise ValueError("changed frozen result, status or executed instruction")
        else:
            self.out[i][r.t].append(choice)
        r.request = None
        if inst.op == "EMIT":
            r.t += 1
            r.depth = 0
        else:
            r.depth += 1

    def finish(self):
        if any(
            r.t != r.length or r.depth or r.request is not None or r.s.pending or r.s.visible != r.length + 1
            for r in self.rows
        ):
            raise ValueError("incomplete route observation")
        if self.expected is None:
            return RouteTape(
                self.authority,
                self.origins,
                tuple(tuple(tuple(byte) for byte in doc) for doc in self.out),
            )


class FrontierChoices(_Observer):
    """One synchronous, already observed frontier. Same request/result checks as the full control tape."""

    def __init__(self, sessions, expected=None):
        self.rows = [_Window(s, None) for s in sessions]
        self.index = {s.id: i for i, s in enumerate(sessions)}
        self.visible = tuple(s.visible for s in sessions)
        if (
            not sessions
            or len(self.index) != len(sessions)
            or any(s.pending or not s.visible for s in sessions)
        ):
            raise ValueError("current distinct unscored frontiers required")
        self.expected = expected
        self.out = [[[]] for _ in sessions] if expected is None else None
        if expected is not None and (
            len(expected) != len(sessions)
            or any(not 0 < len(v) <= s.limits.steps for v, s in zip(expected, sessions, strict=True))
        ):
            raise ValueError("invalid one-frontier choices")

    def _here(self, s, step):
        i = self.index[s.id]
        r = self.rows[i]
        if (
            r.s is not s
            or s.visible != self.visible[i]
            or r.t
            or r.depth != step
            or not 0 <= step < s.limits.steps
            or (self.expected is not None and step >= len(self.expected[i]))
        ):
            raise ValueError("stale current-frontier instruction")
        return i, r, None if self.expected is None else self.expected[i][step]

    def finish(self):
        if any(
            r.t != 1 or r.depth or r.request is not None or not r.s.pending or r.s.visible != v
            for r, v in zip(self.rows, self.visible, strict=True)
        ):
            raise ValueError("incomplete one-frontier execution")
        if self.out is not None:
            return tuple(tuple(doc[0]) for doc in self.out)


def capture_routes(selector, encode, sessions, windows, *, authority, max_steps, generator):
    """Sample the selector normally. No logits, historical answers or numeric view survives in the tape."""
    if any(p.requires_grad for p in selector.parameters()) or generator is None:
        raise ValueError("immutable selector and explicit sampling stream required")
    stamp = tuple((id(p), p._version) for p in selector.parameters())
    obs = _Observer(sessions, windows, authority, max_steps)
    with torch.no_grad():
        batch = teacher_force(
            selector,
            encode,
            sessions,
            windows,
            mode="sample",
            generator=generator,
            observer=obs,
        )
    if stamp != tuple((id(p), p._version) for p in selector.parameters()):
        raise ValueError("selector changed during capture")
    return obs.finish(), batch


def record_training_routes(
    policy,
    encode,
    sessions,
    windows,
    *,
    authority,
    max_steps,
    generator,
    key_price=0.0,
    instruction_price=0.0,
    baseline=0.0,
):
    """Observe the actual sampled training forward. Seal its choices *before* backward/update.

    Unlike a separate no-grad selector, this never changes how the training graph is built.
    The caller may backpropagate the returned normal sampled objective; the tape contains
    no tensor in that graph. Reusing a tape fixes this forward's choices, not a later
    version of the selector. As for replay, discard sessions after an exception.
    """
    if (
        not torch.is_grad_enabled()
        or not policy.training
        or generator is None
        or not all(p.requires_grad for p in policy.parameters())
    ):
        raise ValueError("recording requires a live train-mode policy, gradients and explicit sampler")

    def stamp():
        return tuple((id(p), p._version, p.requires_grad) for p in (*policy.parameters(), *policy.buffers()))

    initial = stamp()
    observer = _Observer(sessions, windows, authority, max_steps)
    batch = teacher_force(
        policy,
        encode,
        sessions,
        windows,
        mode="sample",
        generator=generator,
        observer=observer,
        key_price=key_price,
        instruction_price=instruction_price,
        baseline=baseline,
    )
    if initial != stamp():
        raise ValueError("training state changed before route tape was sealed")
    return observer.finish(), batch


def replay_routes(values, encode, sessions, windows, tape, *, authority, max_steps):
    """Train with result.mean_nll only. There is deliberately no on-policy objective for a forced route.

    Any failure invalidates these sessions and their staged gradients; discard instead of continuing.
    The caller independently verifies the encoder, source and access scope represented by authority.
    """
    obs = _Observer(sessions, windows, authority, max_steps, expected=tape)
    batch = teacher_force(
        values,
        encode,
        sessions,
        windows,
        mode="replay",
        replay=tape.instructions(),
        observer=obs,
    )
    obs.finish()
    return batch
