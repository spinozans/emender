"""Off-card X49-Q address parity using only real session locate/byte results on invented prefixes."""

from collections import Counter
import json
import os
import random
from ..x47_xsa_attention.workbench_native import Live, EXE_SHA
from ..x48_muon_continuation.data import check
from .query_depth import addresses, Hit


def native(live, H, count):
    t = live.visible - 1
    lo = max(0, t + 1 - H)
    hi = t - 16
    seen = {}

    def byte(i):
        check(lo <= i <= t, "noncausal byte request")
        if i not in seen:
            v = live._rpc(f"byte {i}").get("byte")
            check(type(v) is int and 0 <= v <= 255, "bad native byte")
            seen[i] = v
            count["byte_requests"] += 1
        return seen[i]

    for d in (8, 4, 2, 1):
        if hi - lo < d:
            continue
        key = bytes(byte(i) for i in range(t - d + 1, t + 1))
        ps = live._rpc(f"locate {key.hex()} {lo} {hi} 4096").get("positions")
        count["locate_requests"] += 1
        check(
            type(ps) is list
            and ps == sorted(set(ps))
            and all(type(x) is int and lo <= x <= hi - d for x in ps),
            "bad native locate",
        )
        if not ps:
            continue
        j = ps[-1] + d
        code = byte(j)
        ell = {}
        for pos in ps:
            k = pos + d
            if byte(k) != code:
                continue
            n = 0
            while n < 32 and k - 1 - n >= lo and byte(k - 1 - n) == byte(t - n):
                n += 1
            check(n >= d, "native key/address mismatch")
            ell[k] = n
        e = max(ell, key=lambda k: (ell[k], k))
        controls = [
            k
            for k in ell
            if e != j
            and k not in (j, e)
            and (t - k).bit_length() == (t - e).bit_length()
            and ell[k] == ell[j]
        ]
        s = min(controls, key=lambda k: (abs(k - e), -k)) if controls else e
        return {
            "E": Hit(t, d, j, e, e, ell[j], ell[e], False),
            "M": Hit(t, d, j, e, s, ell[j], ell[e], bool(controls)),
        }
    return {"E": None, "M": None}


def run():
    check(
        os.getenv("CUDA_VISIBLE_DEVICES") == "" and not os.getenv("GPU_LEASE_HELD"),
        "off-card check only",
    )
    w = bytearray(random.Random(4906).randbytes(1620))
    key = b"q\x00r\xffstu\x1e"
    long = b"\xfepoint@Q#Z" + key
    for k, context in [
        (126, b"D" + key),
        (174, b"E" + key),
        (290, b"C" + key),
        (150, long),
    ]:
        w[k - len(context) : k + 1] = context + b"\0"
    probes = [0, 8, 17, 128, 142, 156, 290, 350, 1600]
    for t in (350, 1600):
        w[t + 1 - len(long) : t + 1] = long
    w = bytes(w)
    counts = Counter()
    for seal in (False, True):
        with Live() as live:
            for t in probes:
                seam = 146
                if seal and live.visible < seam <= t + 1:
                    live.append(w[live.visible : seam])
                    check(live._rpc("seal") == {"sealed": seam}, "seal failed")
                    counts["seals"] += 1
                if live.visible < t + 1:
                    live.append(w[live.visible : t + 1])
                check(live.visible == t + 1, "wrong visible length")
                for H in (128, 1024, 4096):
                    real = native(live, H, counts)
                    for a, r in real.items():
                        plan = addresses(w[: t + 1], policy=a, horizon=H)
                        last = plan[-1] if plan and plan[-1].t == t else None
                        check(r == last, "Q native/prefix mismatch")
                        counts["policy_frontiers"] += 1
                    m = real["M"]
                    counts["query_frontiers"] += 1
                    counts["literal_hits"] += int(m is not None)
                    if m:
                        counts["moves"] += int(m.extended != m.latest)
                        counts["matched_substitutions"] += int(m.matched)
                    if H == 4096 and t == 350:
                        check(
                            m
                            and (m.latest, m.extended, m.selected) == (290, 150, 174)
                            and m.longest >= 18,
                            "missing constructed matched pair",
                        )
        check(live.proc.returncode == 0, "unclosed native session")
        counts["sessions_closed"] += 1
    check(
        counts["matched_substitutions"]
        and counts["moves"]
        and 0 < counts["literal_hits"] < counts["query_frontiers"],
        "degenerate native fixture",
    )
    return dict(
        scope="CPU synthetic prefix address parity only, no natural targets or weights",
        binary_sha256=EXE_SHA,
        work=dict(counts),
    )


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
