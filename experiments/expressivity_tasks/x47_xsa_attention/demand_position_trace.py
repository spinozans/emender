"""Post-selection attribution and post-score evidence; never a source of model inputs."""

from hashlib import sha256

import numpy as np
import torch
from ndm.memory_program.machine import Decision
from .arrival_service_bench import event_update
from .demand_tape_check import EntryTrace
from .direct_control_bench import add_hash
from .serving_verify import require


class NativeLog:
    """Bounded per eight-client wave. Keep raw keys only until their synchronous dispatch returns."""

    def __init__(self, trace):
        self.trace = trace
        self.calls = []
        self.pending = []

    def raw(self, chosen, cap, unresolved, result, before, begin, end):
        spans, report = result
        rows = report["rows"]
        require(
            len(self.calls) < 87 * 6 * 32 and len(spans) == len(unresolved) == len(rows) > 0,
            "unbound positional frame",
        )
        material = [
            (tuple(ss), r["raw"], r["status"], r["hits"], r["lf"], r["ranks"])
            for ss, r in zip(spans, rows, strict=True)
        ]
        h = sha256()
        add_hash(h, (chosen, cap, tuple(unresolved), material))
        c = dict(
            before=before,
            begin=begin,
            end=end,
            chosen=len(chosen),
            keys=len(unresolved),
            cap=cap,
            sha=h.hexdigest(),
            ranks=sum(r["ranks"] for r in rows),
            lf_steps=sum(sum(r["lf"]) for r in rows),
            lf_rows=sum(len(r["lf"]) for r in rows),
            verified_spans=sum(map(len, spans)),
            exposed_spans=sum(
                len(ss)
                for ss, r in zip(spans, rows, strict=True)
                if r["status"] == "complete" and len(ss) <= cap
            ),
            overcap=sum(r["status"] == "overcap" for r in rows),
        )
        self.calls.append(c)
        self.pending.append((c, tuple(unresolved)))

    def lookup(self, r, start, end, items):
        if r.source != "P100" or r.operation != "occurrences":
            return
        selected = {}
        for sid, keys in items:
            i = self.trace.origins[self.trace.names.live[sid]]
            for key in keys:
                selected.setdefault(key, set()).add(i)
        for c, ks in self.pending:
            require(
                start <= c["before"] <= c["begin"] < c["end"] <= end and all(k in selected for k in ks),
                "native work outside actual chosen immutable batch",
            )
            c.update(
                issuers=sorted(set.union(*(selected[k] for k in ks))), lookup_begin=start, lookup_end=end
            )
        self.pending.clear()

    def finish(self, ticks):
        require(not self.pending, "unattributed positional request")
        for c in self.calls:
            (i,) = [
                i
                for i, t in enumerate(ticks)
                if t["start"] <= c["lookup_begin"] <= c["lookup_end"] <= t["ack"]
                and set(c["issuers"]) <= set(t["rows"])
            ]
            c["tick"] = i
        return self.calls


class ProgramTrace(EntryTrace):
    """No gate, trust mixture or direct counts. Only actual unconditional sampled decisions."""

    def __init__(self, rows, prompts):
        super().__init__(rows, prompts)
        self.rows = rows
        self.events = [sha256() for _ in rows]

    def commit(self, i, reply):
        d = reply.decision
        require(
            isinstance(d, Decision)
            and d.mode == "sample"
            and self.origins[self.names.live[d.session_id]] == i
            and d.visible == reply.visible_before == len(self.prompts[i]) + len(self.raw[i])
            and len(self.raw[i]) < 12
            and reply.byte == int(d.log_probs.argmax())
            and d.log_probs.shape == (256,)
            and d.trace[-1].inst.op == "EMIT",
            "unbound unconditional decision",
        )
        regs = self.registers.pop(i, [])
        require(len(regs) == sum(e.register is not None for e in d.trace), "incomplete ordered registers")
        event_update(self.events[i], d, self.rows[i])
        h = sha256(self.events[i].digest() + b"".join(regs))
        xs = {"byte_logp": d.log_probs, "joint_logp": d.log_policy}
        require(
            all(
                x.device.type == "cpu"
                and x.dtype == torch.float32
                and not x.requires_grad
                and x.isfinite().all()
                for x in xs.values()
            ),
            "unbound program scores",
        )
        self.records[i].append(
            dict(
                keys=d.requests,
                nonemit=len(d.trace) - 1,
                registers=len(regs),
                digest=h.digest(),
                scores={k: v.numpy().copy() for k, v in xs.items()},
            )
        )
        self.raw[i].append(reply.byte)

    def prefix(self, i, n):
        rs = self.records[i]
        bs = self.raw[i]
        require(n == len(bs) == len(rs) > 0, "missing final prefix")
        return dict(
            length=n,
            byte_sha=sha256(bs).hexdigest(),
            event_sha=self.events[i].hexdigest(),
            evidence_sha=sha256(b"".join(r["digest"] for r in rs)).hexdigest(),
            observed_sha=sha256(self.prompts[i] + bs).hexdigest(),
            program_keys=sum(r["keys"] for r in rs),
            nonemit=sum(r["nonemit"] for r in rs),
            registers=sum(r["registers"] for r in rs),
        )


def compare_program(actual, ref):
    require(actual.origins == ref.origins and actual.prompts == ref.prompts, "unbound program comparison")
    n = bad = 0
    num = {k: dict(elements=0, violations=0, max_abs=0.0) for k in ("byte_logp", "joint_logp")}
    for i, rs in enumerate(actual.records):
        require(actual.prefix(i, len(rs)) == ref.prefix(i, len(rs)), "changed program events/registers/bytes")
        for x, y in zip(rs, ref.records[i], strict=True):
            bad += int(x["digest"] != y["digest"])
            n += 1
            for k, z in num.items():
                a, b = (r["scores"][k].astype(np.float64) for r in (x, y))
                require(
                    a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "invalid scores"
                )
                err = abs(a - b)
                z["elements"] += err.size
                z["max_abs"] = max(z["max_abs"], float(err.max()))
                z["violations"] += int((err > 5e-5 + 2e-5 * abs(b)).sum())
    return dict(
        passed=bad == 0 and not any(z["violations"] for z in num.values()),
        positions=n,
        wrong_evidence=bad,
        numeric=num,
    )
