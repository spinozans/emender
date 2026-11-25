"""LS publication reduction, including a separate pooled covariance and table occupancy check."""

from __future__ import annotations
import json
import os
from decimal import Decimal, localcontext
from math import isclose
from ..x48_muon_continuation.data import check, sha
from . import (
    alternative_report as common,
    literal_mark_panel as p,
    literal_mark_data as data,
)
from .surprise_panel import journal
from .panels import SEEDS


def verify_marks(folder, role, hashes, stats):
    for s in SEEDS:
        k = str(s)
        _, rows = journal(folder / f"{role}-{s}.jsonl", hashes)
        work = [x["work"] for x in rows]
        hits = sum(w["literal_hits"] for w in work)
        with localcontext() as ctx:
            ctx.prec = 64
            # Recenter each document's moment sums around the pooled means, then add them.
            mt = (
                Decimal(sum(w["text_code_sum"] for w in work)) / hits
                if hits
                else Decimal(0)
            )
            mh = (
                Decimal(sum(w["neural_code_sum"] for w in work)) / hits
                if hits
                else Decimal(0)
            )
            xx = yy = xy = Decimal(0)
            for w in work:
                n = w["literal_hits"]
                t = w["text_code_sum"]
                h = w["neural_code_sum"]
                xx += w["text_code_square_sum"] - 2 * t * mt + n * mt * mt
                yy += w["neural_code_square_sum"] - 2 * h * mh + n * mh * mh
                xy += w["text_neural_code_product_sum"] - h * mt - t * mh + n * mt * mh
            got = stats["hit_weighted_pearson_by_seed"][k]
            check(
                (got is None)
                if not (xx and yy)
                else got is not None
                and isclose(
                    got, float(xy / (xx * yy).sqrt()), abs_tol=2e-14, rel_tol=0
                ),
                "LS correlation differs from pooled hit moments",
            )
        support = stats["documents_with_support_by_seed"][k]
        peaks = stats["peak_table_entries_per_window_by_seed"][k]
        check(
            support["text_vs_neural_changed"]
            == sum(w["text_vs_neural_changed"] > 0 for w in work)
            and all(
                peaks[a] == max(w[a] for w in work)
                for a in ("text_distinct_contexts", "text_distinct_pairs")
            ),
            "incorrect LS support/table peak",
        )


def report():
    r = common.report(study="LS")
    d = p.ROOT / p.JOB.name
    roles = data.checked_roles(
        data.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    )
    for role, rows in roles.items():
        verify_marks(d, role, [row["sha256"] for row in rows], r["stages"][role])
    r["verified"]["independent_pooled_mark_correlations"] = True
    return r


if __name__ == "__main__":
    os.umask(0o077)
    z = report()
    out = p.ROOT.parent / "literal-surprise-result-1.json"
    with out.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        "X49-LS verified aggregate:", sha(out), out.stat().st_size, "bytes", flush=True
    )
