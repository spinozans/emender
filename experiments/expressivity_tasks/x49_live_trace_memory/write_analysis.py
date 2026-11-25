"""W-only analysis of closed score receipts. Bootstrap whole docs; short docs are not padding."""

from __future__ import annotations
import numpy as np
from ..x48_muon_continuation.data import check, load, sha
from .panel_analysis import read_stage
from . import write_panel as p
from .write_data import ROWS


def document_bootstrap(lengths):
    ns = np.asarray(lengths, dtype=np.int64)
    check(
        ns.ndim == 1
        and ns.size > 0
        and np.array_equal(ns, lengths)
        and np.all((ns >= 1) & (ns < 4096)),
        "invalid real document target counts",
    )
    n = len(ns)
    draw = np.random.default_rng(4905).integers(0, n, size=(10000, n))
    den = ns[draw].sum(1)
    total = int(ns.sum())

    def summarize(delta):
        x = np.asarray(delta, dtype=np.float64)
        check(
            x.shape == (3, n) and np.isfinite(x).all(),
            "missing paired document NLL sums",
        )
        v = x.mean(0)
        lo, hi = np.quantile(v[draw].sum(1) / den, [0.025, 0.975], method="linear")
        return dict(
            gain=float(v.sum() / total),
            by_seed=(x.sum(1) / total).tolist(),
            ci95=[float(lo), float(hi)],
        )

    return summarize


def compare(table, lengths, wall):
    summary = document_bootstrap(lengths)
    contrasts = {}
    shams = {}

    def get(a, s, i, h, sh):
        return table[f"{a}-{s}@{i}"][h, sh]

    for a, b in p.PAIRS:
        k = a + ":" + b
        ws = wall[k]
        delta = {
            h: np.stack(
                [
                    get(b, s, 1024, h, "none") - get(a, s, 1024, h, "none")
                    for s in p.SEEDS
                ]
            )
            for h in (128, 1024, 4096)
        }
        clock = []
        for s in p.SEEDS:
            sel = ws[str(s)]["selected"]
            if sel[a] is not None and sel[b] is not None:
                clock.append(
                    get(b, s, sel[b]["step"], 4096, "none")
                    - get(a, s, sel[a]["step"], 4096, "none")
                )
        contrasts[k] = dict(
            full={str(h): summary(x) for h, x in delta.items()},
            horizon_interaction=summary(delta[4096] - delta[128]),
            common_wall=summary(clock) if len(clock) == 3 else None,
            wall_conclusive=len(clock) == 3
            and all(v["conclusive"] for v in ws.values()),
        )
    for a in p.ARMS:
        shams[a] = {
            sh: summary(
                [
                    get(a, s, 1024, 4096, sh) - get(a, s, 1024, 4096, "none")
                    for s in p.SEEDS
                ]
            )
            for sh in (("literal", "inner") if a == "L" else ("literal",))
        }
    return dict(contrasts=contrasts, binding=shams)


def analyze():
    root = p.ROOT
    d = root / p.FINAL
    check(sha(ROWS) == p.ROW_SHA, "changed W order")
    rows = load(ROWS)["panels"]["final_test"]
    tab, stats, plan = read_stage(root, p.FINAL, rows, job=p.JOBS[p.FINAL])
    runs = p.histories(plan["ref"])
    wall = p.clocks(runs)
    ns = [r["length"] - 1 for r in rows]
    check(
        len(ns) == p.N
        and sum(ns) == p.NT
        and wall == plan["wall_pairs"]
        and plan["training_charges"] == {k: v["charge_sha"] for k, v in runs.items()}
        and plan["lengths_sha"] == p.fingerprint(ns),
        "foreign analysis clocks/documents",
    )
    paid = load(root / "charged.json")
    check(
        set(paid) == set(p.JOBS)
        and all(
            c == load(root / k / "charge.json")
            and c["rc"] == 0
            and c["renewed"]
            and 0 < c["upper_card_s"] <= p.JOBS[k].cap
            for k, c in paid.items()
        ),
        "partial W campaign",
    )
    out = compare(tab, ns, wall)
    for key, e in stats["endpoints"].items():
        a = plan["endpoints"][key]["arm"]
        cc = e["cells"]
        clean = cc["4096:none"]["work"]
        interventions = {
            "literal_payloads_reassigned",
            "inner_writes_reassigned",
            "inner_reassignments_consumed",
        }
        for cell, info in cc.items():
            w = info["work"]
            check(
                w["literal_hits"] + w["literal_misses"] == w["literal_queries"] == p.NT
                and w["literal_value_reads"] == w["literal_hits"]
                and w["literal_value_writes"] == p.NT,
                "missing byte/write",
            )
            if a != "B":
                check(
                    w["writer_hit_enrichments"] == w["literal_hits"]
                    and w["writer_lookups_reused"] == p.NT
                    and w["linked_uses"] + w["linked_misses"] == p.NT
                    and w["writer_current_recomputations"]
                    == (w["literal_hits"] if a == "P" else 0)
                    and w["writer_old_value_reads"]
                    == (w["literal_hits"] if a == "L" else 0),
                    "changed inner workload",
                )
            if cell.startswith("4096:"):
                check(
                    {k: v for k, v in clean.items() if k not in interventions}
                    == {k: v for k, v in w.items() if k not in interventions},
                    "sham changed logical evidence",
                )
            check(
                (
                    cell.endswith(":literal")
                    or w.get("literal_payloads_reassigned", 0) == 0
                )
                and (
                    cell.endswith(":inner")
                    or w.get("inner_writes_reassigned", 0)
                    == w.get("inner_reassignments_consumed", 0)
                    == 0
                )
                and 0
                <= w.get("inner_reassignments_consumed", 0)
                <= w.get("linked_uses", 0),
                "wrong sham namespace",
            )
    for s in p.SEEDS:
        base, local, linked = [
            stats["endpoints"][f"{a}-{s}@1024"]["cells"] for a in p.ARMS
        ]
        for cell in base:
            wb, wp, wl = (x[cell]["work"] for x in (base, local, linked))
            check(
                all(wp[k] == wl[k] == v for k, v in wb.items())
                and all(
                    wl[k] == v
                    for k, v in wp.items()
                    if k
                    not in ("writer_old_value_reads", "writer_current_recomputations")
                ),
                "unmatched B/P/L evidence at shared cell",
            )
    return dict(
        scope="X49-W read-before-write; new short native final test, no v2/R final access",
        ref=plan["ref"],
        freeze_sha=sha(d / "frozen.json"),
        train_identity=p.W_ID,
        rows_sha=p.ROW_SHA,
        analysis=out,
        wall=wall,
        stage=stats,
        training={
            k: load(root / k / "complete.json")
            for k, j in p.JOBS.items()
            if j.kind == "train"
        },
        charges=paid,
        upper_card_s=sum(v["upper_card_s"] for v in paid.values()),
        shared_prep=p.prep(),
        device_short_check=load(d / "short-device-check.json"),
    )
