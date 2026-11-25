"""Independent EH aggregate support/fit/paired-loss/charge closure. No text or host execution."""

import json
import os
from ..x48_muon_continuation.data import check, sha
from . import (
    alternative_report as common,
    error_history_data as data,
    error_history_panel as p,
)
from .panels import SEEDS
from .surprise_panel import journal


def report():
    r = common.report(study="EH")
    d = p.ROOT / p.JOB.name
    roles = data.checked_roles(
        data.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    )
    check(
        r["swap_fit"] == "ERROR"
        and r["fit_dimensions"] == p.DIMS
        and r["j_refits"] == 0,
        "wrong EH model family",
    )
    for role, rs in roles.items():
        hs = [v["sha256"] for v in rs]
        stage = r["stages"][role]
        for s in SEEDS:
            k = str(s)
            _, rows = journal(d / f"{role}-{s}.jsonl", hs)
            works = [x["work"] for x in rows]
            total = stage["work_by_seed"][k]
            support = stage["documents_with_support_by_seed"][k]
            for label, counter in [
                ("donor_displaced", "shift_older_displaced_uses"),
                ("changed_guess", "shift_older_changed_uses"),
                ("changed_histogram", "changed_histogram_rows"),
                ("older_history", "pool_active_rows"),
                ("older_mistake", "pool_older_mistake_uses"),
            ]:
                check(
                    sum(w[counter] for w in works) == total[counter]
                    and sum(w[counter] > 0 for w in works) == support[label],
                    "EH support is not a paired-document count",
                )
            peaks = stage["peak_per_window_by_seed"][k]
            check(
                all(v == max(w[name] for w in works) for name, v in peaks.items())
                and total["candidate_correct"]
                == sum(x["candidate_correct"] for x in rows)
                and total["pool_key_admissions"]
                == total["pool_committed_records"]
                + sum(w["pool_peak_keys"] for w in works),
                "wrong EH index, pending or copy work",
            )
    r["verified"]["independent_histogram_support_and_peak"] = True
    r["wrong_guess_coefficient_effect_by_seed"] = {
        s: (
            "suppress"
            if v["ERROR"][-1] > 0
            else "favor"
            if v["ERROR"][-1] < 0
            else "none"
        )
        for s, v in r["beta"].items()
    }
    return r


if __name__ == "__main__":
    os.umask(0o077)
    z = report()
    out = p.ROOT.parent / "error-history-result-1.json"
    with out.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        "X49-EH verified aggregate:", sha(out), out.stat().st_size, "bytes", flush=True
    )
