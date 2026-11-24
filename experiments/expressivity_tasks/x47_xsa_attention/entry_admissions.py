"""Private logical observations for a generated DIRECT/ENTER service. No archive ownership here."""

from contextlib import contextmanager
from hashlib import sha256

from ndm.memory_program.direct_entry import EntryStream
from .private_tape import PrivateTape
from .serving_verify import require


class EntryAdmissions:
    def __init__(
        self,
        rows,
        trace,
        *,
        admitted_rows,
        byte_limit,
        demand,
        encode,
        session,
        check,
        active=None,
        program_only=False,
    ):
        self.rows, self.trace = rows, trace
        self.remaining = set(admitted_rows)
        self.active = set() if active is None else active
        self.tapes, self.closed = {}, {}
        self.peak = 0
        self.byte_limit, self.demand = byte_limit, demand
        self.program_only = program_only
        self.encode, self.session, self.check = encode, session, check

    @contextmanager
    def binding(self, i):
        require(i in self.remaining, "unoffered, repeated or out-of-grant row")
        self.check()
        self.remaining.remove(i)
        r = self.rows[i]
        live = PrivateTape(
            r["doc"],
            r["start"],
            encode=self.encode,
            byte_limit=self.byte_limit,
            demand=self.demand,
        )
        q = s = None
        try:
            s = self.session(r, live)
            if not self.program_only:
                q = EntryStream(s, byte_limit=self.byte_limit)
            require(
                s.id == live.id and s.id not in self.trace.names.live,
                "reused private nonce",
            )
            self.trace.names.live[s.id] = (s.document, s.start)
            self.active.add(live.id)
            self.tapes[i] = live
            yield s, (live if self.program_only else q)
        finally:
            matched = live.data == self.trace.prompts[i] + self.trace.raw[i]
            observed = sha256(live.data).hexdigest()
            try:
                if q is not None:
                    q.close()
                elif s is not None:
                    s.cells, s.recent, s.readers, s.advance = [None] * s.limits.cells, b"", (), None
                    s.pending = True
            finally:
                try:
                    live.close()
                finally:
                    self.active.discard(live.id)
                    self.tapes.pop(i, None)
                    self.trace.names.live.pop(live.id, None)
            scrub = (
                (
                    self.program_only
                    or (q is not None and q.closed and not q._counts.rows and not q._counts.tail)
                )
                and s is not None
                and not s.recent
                and not any(s.cells)
                and not s.readers
                and s.advance is None
                and live.closed
                and not live.data
                and live.native is None
            )
            self.closed[i] = dict(
                row=i,
                observed_sha=observed,
                matched=matched,
                scrub=scrub,
                work=dict(live.stats),
                exit=live.exit,
            )
            require(
                matched and scrub,
                "different observed transcript or incomplete private teardown",
            )

    def reply(self, i, reply):
        self.peak = max(self.peak, sum(t.native is not None for t in self.tapes.values()))
        self.trace.commit(i, reply)

    def finish(self):
        require(
            not self.active
            and not self.tapes
            and not self.remaining
            and not self.trace.names.live
            and not self.trace.pending_direct
            and not self.trace.registers,
            "unfinished private admissions",
        )
        return [self.closed[i] for i in sorted(self.closed)]
