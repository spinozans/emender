"""Private, bounded observations of *completed* S17 predictions. No oracle in the predictor."""

from collections import Counter
from hashlib import sha256

import numpy as np
import torch

from .direct_control_bench import add_hash, note
from .packed_program_profile import BindingNames, events
from .serving_verify import require


class EntryTrace:
    def __init__(self, rows, prompts):
        self.prompts = prompts
        self.origins = {(r["doc"], r["start"]): i for i, r in enumerate(rows)}
        require(len(self.origins) == len(rows) == len(prompts), "ambiguous trace origins")
        self.names = BindingNames([])
        self.raw = [bytearray() for _ in rows]
        self.records = [[] for _ in rows]
        self.pending_direct = {}
        self.registers = {}

    def _current(self, s):
        i = self.origins[self.names.live[s.id]]
        require(
            (s.document, s.start) == self.names.live[s.id]
            and s.visible == len(self.prompts[i]) + len(self.raw[i]),
            "unbound scored prefix",
        )
        return i

    def direct(self, s, v, a):
        i = self._current(s)
        require(
            s.pending and s.visible == v.visible and s.recent[-8:] == v.tail and i not in self.pending_direct,
            "stale direct witness",
        )
        h = sha256()
        note(h, v, a)
        self.pending_direct[i] = h.digest(), len(a.probes)

    def page(self, s, step, reg, p):
        i = self._current(s)
        regs = self.registers.setdefault(i, [])
        require(reg == len(regs) and reg <= step < s.limits.steps, "wrong register order")
        h = sha256()
        add_hash(h, (s.visible, step, reg, self.names.layout(p)))
        regs.append(h.digest())

    def commit(self, i, reply):
        t = len(self.raw[i])
        d = reply.decision
        enter = d.program is not None
        probes = self.pending_direct.pop(i, None)
        regs = self.registers.pop(i, [])
        require(
            t == len(self.records[i]) < 12
            and reply.visible_before == len(self.prompts[i]) + t
            and reply.byte == int(d.log_probs.argmax())
            and enter == (probes is None) == (d.trust is None),
            "unchosen or missing computation",
        )
        h = sha256()
        add_hash(h, (reply.visible_before, enter, d.keys, d.nonemit))
        numbers = dict(byte_logp=d.log_probs, gate_logp=d.gate_logp, joint_logp=d.log_policy)
        if enter:
            p = d.program
            require(
                self.origins[self.names.live[p.session_id]] == i
                and p.visible == reply.visible_before
                and len(regs) == sum(e.register is not None for e in p.trace)
                and p.requests == d.keys
                and p.mode == "sample"
                and p.trace[-1].inst.op == "EMIT",
                "incomplete program witness",
            )
            add_hash(h, events([[p]], self.names))
            h.update(b"".join(regs))
            numbers["conditional_logp"] = p.log_policy
        else:
            require(
                not regs
                and d.nonemit == 0
                and probes[1] == d.keys
                and d.trust.shape == (4,)
                and abs(d.trust.sum().item() - 1) < 5e-5,
                "incomplete direct witness",
            )
            h.update(probes[0])
            numbers["trust"] = d.trust
        require(
            all(
                x.device.type == "cpu" and x.dtype == torch.float32 and not x.requires_grad
                for x in numbers.values()
            ),
            "not fixed CPU scores",
        )
        self.records[i].append(
            dict(
                enter=enter,
                keys=d.keys,
                nonemit=d.nonemit,
                registers=len(regs),
                digest=h.digest(),
                scores={k: v.numpy().copy() for k, v in numbers.items()},
            )
        )
        self.raw[i].append(reply.byte)

    def prefix(self, i, n):
        r = self.records[i][:n]
        bs = self.raw[i][:n]
        require(len(bs) == len(r) == n > 0, "missing observed prefix")
        return dict(
            length=n,
            byte_sha=sha256(bs).hexdigest(),
            observed_sha=sha256(self.prompts[i] + bs).hexdigest(),
            evidence_sha=sha256(b"".join(x["digest"] for x in r)).hexdigest(),
            entry_sha=sha256(bytes(x["enter"] for x in r)).hexdigest(),
            enter=sum(x["enter"] for x in r),
            probes=sum(x["keys"] for x in r if not x["enter"]),
            program_keys=sum(x["keys"] for x in r if x["enter"]),
            nonemit=sum(x["nonemit"] for x in r),
            registers=sum(x["registers"] for x in r),
        )


def compare(actual, ref):
    require(actual.origins == ref.origins and actual.prompts == ref.prompts, "different admitted prompts")
    num = {
        k: dict(elements=0, violations=0, max_abs=0.0)
        for k in ("byte_logp", "gate_logp", "joint_logp", "conditional_logp", "trust")
    }
    work = Counter()
    for i, bs in enumerate(actual.raw):
        n = len(bs)
        actual.prefix(i, n)
        ref.prefix(i, n)
        for j, (a, b) in enumerate(zip(actual.records[i], ref.records[i][:n], strict=True)):
            work["positions"] += 1
            work["entries"] += int(a["enter"])
            work["wrong_byte"] += int(bs[j] != ref.raw[i][j])
            work["wrong_entry"] += int(a["enter"] != b["enter"])
            work["wrong_evidence"] += int(a["digest"] != b["digest"])
            for k in sorted(a["scores"].keys() & b["scores"].keys()):
                x, y = (r["scores"][k].astype(np.float64) for r in (a, b))
                require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), "invalid score")
                err = np.abs(x - y)
                z = num[k]
                z["elements"] += x.size
                z["violations"] += int((err > 5e-5 + 2e-5 * np.abs(y)).sum())
                z["max_abs"] = max(z["max_abs"], float(err.max()))
    return dict(
        **work,
        numeric=num,
        passed=not any(work[k] for k in ("wrong_byte", "wrong_entry", "wrong_evidence"))
        and not any(v["violations"] for v in num.values()),
    )
