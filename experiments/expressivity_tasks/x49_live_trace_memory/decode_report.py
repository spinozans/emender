"""Independent DR close audit. No source text, decoder fitting on PROBE, or host runs."""

import json
import os
from math import fsum, isclose
from ..x48_muon_continuation.data import check, sha
from . import alternative_report as common, decode_panel as p, decode_data as data
from .panels import SEEDS
from .surprise_panel import journal


def report():
    z = common.report(study="DR")
    stage = z["stages"]["probe"]
    d = p.ROOT / p.JOB.name
    roles = data.checked_roles(
        data.bound(data.BASE / "reservation.json", z["data"]["rows_sha"])
    )
    hashes = [r["sha256"] for r in roles["probe"]]
    for s in SEEDS:
        k = str(s)
        _, rows = journal(d / f"probe-{s}.jsonl", hashes)
        n = 32 * sum(r["work"]["literal_hits"] for r in rows)
        for a, avg in stage["hit_component_MSE_by_seed"][k].items():
            check(
                (avg is None)
                if not n
                else isclose(
                    avg,
                    fsum(r["squared_error_sums"][a] for r in rows) / n,
                    abs_tol=1e-15,
                    rel_tol=2e-14,
                ),
                "wrong DR MSE denominator",
            )
        check(
            stage["documents_with_support_by_seed"][k]["changed_pair"]
            == sum(r["work"]["sham_pairs_changed"] > 0 for r in rows)
            and z["decoders"][k]["fit_writes"]
            == z["stages"]["fit"]["teacher_fit_writes_per_seed"]
            == len(roles["fit"]) * (p.T - 1),
            "incomplete DR support or FIT writes",
        )
    z["verified"]["independent_MSE_denominator"] = True
    return z


if __name__ == "__main__":
    os.umask(0o077)
    z = report()
    out = p.ROOT.parent / "decode-record-result-1.json"
    with out.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        "X49-DR verified aggregate:", sha(out), out.stat().st_size, "bytes", flush=True
    )
