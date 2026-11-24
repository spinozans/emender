"""Off-card actual XSA session vs literal address plan. Synthetic bytes only; no corpus or training."""

from __future__ import annotations

from collections import Counter
import json
import os

from ..x47_xsa_attention.workbench_native import EXE_SHA, Live
from .literal import DEPTHS, HORIZONS, MIN_AGE, MISS, Match, literal_plan


def _require(ok, message):
    if not ok:
        raise ValueError(message)


class _WindowSession(Live):
    def read_suffix(self, horizon, work):
        t = self.visible - 1
        lo, hi = max(0, self.visible - horizon), t - MIN_AGE
        for d in DEPTHS:
            if hi - lo < d:
                continue
            key = self.tail[-d:]
            # locate includes occurrences ending exactly at hi, whose follower is at hi.
            # Cap covers the entire admitted fresh window, not an arbitrary prefix of a result.
            r = self._rpc(f"locate {key.hex()} {lo} {hi} 4096")
            work["locate"] += 1
            loc = r.get("positions")
            _require(
                isinstance(loc, list)
                and all(type(i) is int and lo <= i <= hi - d for i in loc),
                "invalid locate",
            )
            _require(loc == sorted(set(loc)), "duplicate or unordered occurrences")
            if not loc:
                continue
            counts = Counter()
            for i in loc:
                x = self._rpc(f"byte {i + d}").get("byte")
                _require(type(x) is int and 0 <= x < 256, "invalid historical byte")
                counts[x] += 1
                work["byte"] += 1
            return Match(d, loc[-1] + d, tuple(sorted(counts.items())))
        return MISS


def run():
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD"),
        "CPU only",
    )
    key = b"\0\x1e\xff@x49!"
    short = b"aaabaabaaa" * 3 + (key + b"q") * 4 + bytes(range(32)) + key
    late = key + b"Q" + 2080 * b"z" + key
    work = Counter()
    for word, probes in (
        (short, list(range(len(short)))),
        (late, [0, 23, len(late) - 1]),
    ):
        expected = {h: literal_plan(word, h) for h in HORIZONS}
        for sealed in (False, True):
            with _WindowSession() as live:
                split = 4  # some matches span the sealed/tail seam
                for t in probes:
                    if sealed and live.visible < split <= t + 1:
                        work["appended_bytes"] += split - live.visible
                        live.append(word[live.visible : split])
                        _require(live._rpc("seal") == {"sealed": split}, "seal failed")
                        work["seals"] += 1
                    if live.visible <= t:
                        work["appended_bytes"] += t + 1 - live.visible
                        live.append(word[live.visible : t + 1])
                    _require(live.visible == t + 1, "wrong native frontier")
                    for h in HORIZONS:
                        row = live.read_suffix(h, work)
                        _require(
                            row == expected[h][t],
                            f"native mismatch at horizon {h}, frontier {t}",
                        )
                        work["matched_frontiers"] += 1
                        work["hits" if row.latest is not None else "misses"] += 1
                        if row.latest is not None and t - row.latest > 1024:
                            work["hit_older_than_1024"] += 1
            _require(live.proc.returncode == 0, "session did not exit cleanly")
            work["sessions_closed"] += 1
    _require(
        work["hit_older_than_1024"] > 0 and work["hits"] and work["misses"],
        "empty contrast",
    )
    return {
        "scope": "synthetic CPU address parity; not an LM result",
        "binary_sha256": EXE_SHA,
        "work": dict(work),
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
