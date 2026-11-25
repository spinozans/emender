"""All-byte paired endpoint analysis. This consumes closed score receipts, never a corpus or learner."""

from __future__ import annotations
from collections import Counter
import json
from math import isclose
import numpy as np
from ..x48_muon_continuation.data import load, sha, write_json
from .panel_spec import ROOT, SEEDS, STEPS, T, JOBS, PAIRS, OUT, check, training
from .train_stream import V2_IDENTITY


def summarize(delta, draw):
    x = np.asarray(delta, dtype=np.float64)
    check(
        x.shape == (3, 128) and np.isfinite(x).all(),
        "need three paired seeds and all final documents",
    )
    v = x.mean(axis=0)
    lo, hi = np.quantile(v[draw].mean(axis=1), [0.025, 0.975], method="linear")
    return dict(
        gain=float(v.mean()),
        by_seed=x.mean(axis=1).tolist(),
        ci95=[float(lo), float(hi)],
    )


def contrasts(table, wall):
    draw = np.random.default_rng(4900).integers(0, 128, size=(10000, 128))
    out = {}
    shams = {}

    def get(k, s, i, h, mode):
        return table[f"{training(k, s)}@{i}"][h, mode]

    for k, b in PAIRS:
        pair = k + ":" + b
        rows = wall[pair]
        deltas = {
            h: np.stack(
                [
                    get(b, s, STEPS, h, "none") - get(k, s, STEPS, h, "none")
                    for s in SEEDS
                ]
            )
            for h in (128, 1024, 4096)
        }
        full = {str(h): summarize(v, draw) for h, v in deltas.items()}
        parts = []
        for s in SEEDS:
            p = rows[str(s)]["selected"]
            if p[b] is not None and p[k] is not None:
                parts.append(
                    get(b, s, p[b]["step"], T, "none")
                    - get(k, s, p[k]["step"], T, "none")
                )
        matched = summarize(parts, draw) if len(parts) == 3 else None
        out[pair] = dict(
            by_horizon=full,
            horizon_interaction=summarize(deltas[T] - deltas[128], draw),
            common_wall=matched,
            wall_conclusive=len(parts) == 3
            and all(r["conclusive"] for r in rows.values()),
        )
    for k in "BFC":
        shams[k] = {
            sh: summarize(
                [get(k, s, STEPS, T, sh) - get(k, s, STEPS, T, "none") for s in SEEDS],
                draw,
            )
            for sh in (("literal",) if k == "B" else ("literal", "trace"))
        }
    decisions = {}
    for k, controls in [("B", "A"), ("C", "ABF")]:
        e = [out[k + ":" + b]["by_horizon"][str(T)] for b in controls]
        a = out[k + ":A"]
        gates = dict(
            exposure=all(
                x["gain"] >= 0.005 and min(x["by_seed"]) > 0 and x["ci95"][0] > 0
                for x in e
            ),
            binding=all(x["ci95"][0] > 0 for x in shams[k].values()),
            wall=bool(a["wall_conclusive"] and a["common_wall"]["ci95"][0] > 0),
            horizon=a["horizon_interaction"]["ci95"][0] > 0,
        )
        decisions[k] = dict(gates=gates, passes=all(gates.values()))
    return dict(
        contrasts=out,
        binding=shams,
        decisions=decisions,
        simplest_nominee=next((k for k in "BC" if decisions[k]["passes"]), None),
    )


def read_stage(root, name, rows):
    d = root / name
    p = load(d / "frozen.json")
    c = load(d / "complete.json")
    paid = load(d / "charge.json")
    job = JOBS[name]
    check(
        p["split"] == job.kind
        and p["job"] == name
        and c["kind"] == job.kind
        and paid["rc"] == 0
        and paid["renewed"]
        and paid["complete_sha"] == sha(d / "complete.json")
        and c["freeze_sha"] == sha(d / "frozen.json")
        and c["stream_sha"] == sha(d / "stream.json")
        and paid["witnessed_units"]
        == c["units"]
        == p["units"]
        == len(rows) * len(p["endpoints"]),
        "stage not complete and released",
    )
    check(set(c["endpoints"]) == set(p["endpoints"]), "missing evaluation endpoint")
    table = {}
    stats = {}
    for key, endpoint in p["endpoints"].items():
        r = c["endpoints"][key]
        f = d / (key + ".jsonl")
        check(
            r["scores_sha"] == sha(f)
            and endpoint["sha256"] == r["checkpoint_sha"]
            and len(rows) == r["windows"] == p["documents"],
            "unbound score receipt",
        )
        scores = {}
        work = {}
        strata = {}
        head_s = Counter()
        prefill_s = 0.0
        arm = endpoint["arm"]
        cells = (
            [(128, "none"), (1024, "none"), (T, "none")]
            + ([(T, "literal")] if arm != "A" else [])
            + ([(T, "trace")] if arm in "FC" else [])
        )
        for row, line in zip(rows, f.read_text().splitlines(), strict=True):
            obj = json.loads(line)
            check(
                row["sha256"] == obj["document_sha"]
                and obj["host_prefills"] == 1
                and obj["host_tokens"] == T
                and [(x["horizon"], x["sham"]) for x in obj["cells"]] == cells,
                "wrong target/endpoint pairing",
            )
            prefill_s += obj["host_prefill_s"]
            for x in obj["cells"]:
                cell = x["horizon"], x["sham"]
                v = x["nll_sum"]
                w = x["work"]
                check(
                    x["targets"] == T
                    and 0 <= v < float("inf")
                    and w["literal_hits"] + w["literal_misses"] == T,
                    "wrong eval denominator",
                )
                scores.setdefault(cell, []).append(v / T)
                work.setdefault(cell, Counter()).update(w)
                head_s[cell] += x["head_score_s"]
                ds = strata.setdefault(cell, {})
                check(
                    set(x["strata"])
                    == ({"literal", "private"} if arm in "FC" else {"literal"}),
                    "missing age strata",
                )
                for src, groups in x["strata"].items():
                    check(
                        sum(y["targets"] for y in groups.values()) == T
                        and isclose(
                            sum(y["nll_sum"] for y in groups.values()),
                            v,
                            abs_tol=1e-8,
                            rel_tol=0,
                        ),
                        "strata not all bytes",
                    )
                    totals = ds.setdefault(src, {})
                    for age, y in groups.items():
                        totals.setdefault(age, Counter()).update(y)
        table[key] = {cell: np.asarray(v) for cell, v in scores.items()}
        stats[key] = dict(
            prefill_s=prefill_s,
            decode_diagnostic=r["decode_probe"],
            cells={
                str(h) + ":" + sh: dict(
                    all_byte_nll=float(table[key][h, sh].mean()),
                    targets=len(rows) * T,
                    work=work[h, sh],
                    strata=strata[h, sh],
                    head_wall_s=head_s[h, sh],
                )
                for h, sh in cells
            },
        )
    return (
        table,
        dict(
            endpoints=stats,
            charge=paid,
            worker_cpu_core_s=c["cpu_core_s"],
            source_prep=load(d / "stream.json"),
            gpu_peak_alloc=c["gpu_peak_alloc"],
            gpu_peak_reserved=c["gpu_peak_reserved"],
            process_RSS_peak_KiB=c["process_RSS_peak_KiB"],
        ),
        p,
    )


def finish(root=ROOT):
    idp = OUT / "v2/train/identity.json"
    r = OUT / "v2/reservation.json"
    check(
        sha(idp) == V2_IDENTITY and sha(r) == load(idp)["reservation_sha"],
        "foreign document order",
    )
    rows = load(r)["panels"]
    summary = {}
    final = None
    plan = None
    for name in ["cal-AB", "cal-FC", "final"]:
        t, summary[name], p = read_stage(root, name, rows[JOBS[name].kind])
        if name == "final":
            final = t
            plan = p
    decision = contrasts(final, plan["wall_pairs"])
    charges = load(root / "charged.json")
    check(set(charges) == set(JOBS), "partial panel")
    train = {
        k: load(root / k / "complete.json")
        for k, j in JOBS.items()
        if j.kind == "train"
    }
    write_json(
        root / "summary.json",
        dict(
            scope="registered X49 local panel; no Lux launch",
            decision=decision,
            stages=summary,
            wall=plan["wall_pairs"],
            train=train,
            charges=charges,
            total_card_s=sum(v["upper_card_s"] for v in charges.values()),
            native_archive_calls=0,
            shared_successful_prep_wall_s=4.45,
            shared_successful_prep_cpu_s=4.34,
        ),
    )
