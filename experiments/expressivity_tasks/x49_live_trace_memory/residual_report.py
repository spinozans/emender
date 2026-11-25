"""RB: independently sum code occupancy and fallback reads from sealed counters, never text."""

import json
import os
from ..x48_muon_continuation.data import check, sha
from . import decode_report, residual_store as store, residual_panel as p


def audit_codes(rows, z, k, d, s):
    n = store.tables(d, s, z["decoders"][k]).count.tolist()
    fit = z["stages"]["fit"]["occupancy_by_seed"][k]["fit"]

    def summary(v):
        return dict(
            count=sum(v),
            used_codes=sum(x > 0 for x in v),
            singleton_codes=v.count(1),
            max_count=max(v),
        )

    check(fit == summary(n), "wrong summary of FIT code support")
    probes = z["stages"]["probe"]
    for src in ("true", "sham"):
        counts = [0] * 256
        for row in rows:
            a = row["read_code_counts"][src]
            check(
                sum(x for x, ct in zip(a, n, strict=True) if not ct)
                == row["work"][src + "_unseen_code_reads"],
                "wrong unobserved-code count",
            )
            for i, x in enumerate(a):
                counts[i] += x
        check(
            summary(counts) == probes["occupancy_by_seed"][k][src],
            "wrong RB read occupancy",
        )
    check(
        probes["documents_with_support_by_seed"][k]["displaced_slot"]
        == sum(r["work"]["sham_displaced"] > 0 for r in rows),
        "wrong exact-triple support",
    )


def report():
    z = decode_report.table_report("RB", code_audit=audit_codes)
    z["verified"]["independent_occupancy_and_fallback"] = True
    return z


if __name__ == "__main__":
    os.umask(0o077)
    z = report()
    out = p.ROOT.parent / "residual-byte-result-1.json"
    with out.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        "X49-RB verified aggregate:", sha(out), out.stat().st_size, "bytes", flush=True
    )
