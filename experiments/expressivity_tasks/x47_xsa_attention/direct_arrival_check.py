"""Bounded private post-score observations. Never supplied to the serving predictor."""

from hashlib import sha256
import numpy as np

from .direct_control_bench import note
from .serving_verify import require


class Transcript:
    def __init__(self, rows, *, horizon=12):
        self.origins = {(r["doc"], r["start"]): i for i, r in enumerate(rows)}
        require(len(self.origins) == len(rows), "diagnostic row identity collision")
        self.raw = [bytearray() for _ in rows]
        self.events = [[] for _ in rows]
        self.probes = [[] for _ in rows]
        self.scores = np.empty((len(rows), horizon, 260), dtype="<f4")

    def observe(self, s, v, a, lp, w):
        i = self.origins[(s.document, s.start)]
        t = len(self.events[i])
        require(
            t == len(self.raw[i]) and t < self.scores.shape[1] and s.visible == v.visible == 32 + t,
            "stale or speculative direct score",
        )
        h = sha256()
        note(h, v, a)
        self.events[i].append(h.digest())
        self.probes[i].append(len(a.probes))
        self.scores[i, t, :256] = lp.detach().numpy()
        self.scores[i, t, 256:] = w.detach().numpy()

    def commit(self, i, b):
        require(len(self.events[i]) == len(self.raw[i]) + 1, "delivery without exactly one prediction")
        self.raw[i].append(b)

    def prefix(self, i, n):
        require(0 < n <= len(self.raw[i]) == len(self.events[i]) == len(self.probes[i]), "unfinished prefix")
        return dict(
            length=n,
            byte_sha=sha256(self.raw[i][:n]).hexdigest(),
            evidence_sha=sha256(b"".join(self.events[i][:n])).hexdigest(),
            probes=sum(self.probes[i][:n]),
        )


def compare(actual, ref):
    require(actual.origins == ref.origins, "different row order or document scope")
    numbers = {name: dict(elements=0, max_abs=0.0, violations=0) for name in ("byte_logp", "trust")}
    wrong_byte = wrong_event = positions = 0
    for i, bs in enumerate(actual.raw):
        n = len(bs)
        actual.prefix(i, n)
        ref.prefix(i, n)
        wrong_byte += sum(a != b for a, b in zip(bs, ref.raw[i][:n], strict=True))
        wrong_event += sum(a != b for a, b in zip(actual.events[i], ref.events[i][:n], strict=True))
        positions += n
        for k, cols in (("byte_logp", slice(0, 256)), ("trust", slice(256, 260))):
            a = actual.scores[i, :n, cols].astype(np.float64)
            b = ref.scores[i, :n, cols].astype(np.float64)
            require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite scored frontier")
            delta = np.abs(a - b)
            x = numbers[k]
            x["elements"] += delta.size
            x["max_abs"] = max(x["max_abs"], float(delta.max()))
            x["violations"] += int((delta > 5e-5 + 2e-5 * np.abs(b)).sum())
    return dict(
        positions=positions,
        wrong_byte=wrong_byte,
        wrong_evidence=wrong_event,
        numeric=numbers,
        passed=not (wrong_byte or wrong_event or any(v["violations"] for v in numbers.values())),
    )
