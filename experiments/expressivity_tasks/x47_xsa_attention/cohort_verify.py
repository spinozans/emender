"""Offline S2 service audit; compare every trajectory to S1, not only the other S2 engines."""

import argparse
import json
from math import isclose
from pathlib import Path
from .serving_verify import HERE, ledger, load, pct, require

ROOT = HERE / "receipts/memory-program-s2"
S1 = HERE / "receipts/memory-program-s1"


def audit(root=ROOT):
    previous = load(S1 / "verified.json")["cumulative_x47_upper"]
    require(previous == 308, "bad S1 charge")
    attempts, ok = ledger(root, expected=13056)
    paid = sum(x["upper"] for x in attempts)
    require(len(ok) == 1 and 0 < paid <= 260, "unfinished/over-budget S2")
    r = ok[0]
    ref = (root / "gated-ref").read_text().strip()
    q = (root / "qualification.txt").read_text()
    require("59 passed" in q and "failed" not in q and "skipped" not in q, "CPU gate")
    idn = load(r / "identity.json")
    prior = S1 / "attempt1"
    baseline = load(prior / "identity.json")
    require(
        ref == idn["ref"] == (r / "ref").read_text().strip()
        and all(idn[k] == v for k, v in baseline.items() if k not in ("owner", "ref", "server_pid")),
        "changed source/host/workload",
    )
    dev = load(r / "device-parity.json")
    require(
        dev["cohort"]["delayed_grad_norm"] > 1e-6
        and dev["cohort"]["checks"] == dev["ready"]["checks"]
        and dev["cohort"]["pages"] == dev["ready"]["pages"]
        and dev["conditional"]["checks"]["factor_logp"] >= 100,
        "cohort parity",
    )
    total = 0
    by = {}
    mismatch = []
    for b in (1, 16):
        for mode in ("greedy", "sample"):
            for engine in ("fused", "ready", "cohort"):
                rr = []
                for block in (0, 1):
                    arm = f"{engine}-{mode}"
                    cell = f"{block}-b{b}-{arm}"
                    x = load(r / (cell + ".json"))
                    y = load(prior / (f"{block}-b{b}-fused-{mode}.json"))
                    require(
                        (x["cell"], x["train_rows"], x["emitted"], x["b"], x["block"], x["arm"])
                        == (cell, list(range(16 * block, 16 * block + b)), 64 * b, b, block, arm),
                        "cell binding",
                    )
                    require(
                        len(x["event_sha"]) == len(x["generated_sha"]) == b
                        and all(len(h) == 64 for h in x["event_sha"] + x["generated_sha"]),
                        "empty digest",
                    )
                    diff = [
                        j for j in range(b) if any(x[k][j] != y[k][j] for k in ("generated_sha", "event_sha"))
                    ]
                    if diff:
                        mismatch.append(dict(cell=cell, docs=diff))
                    else:
                        require(
                            all(x[k] == y[k] for k in ("opcodes", "statuses"))
                            and x["work"]["logical_keys"] == y["work"]["logical_keys"],
                            "work mismatch",
                        )
                    total += x["emitted"]
                    lat = x["interbyte_ns"]
                    require(
                        len(lat) == b
                        and all(len(t) == 64 and min(t) > 0 and sum(t) <= x["wall_ns"] for t in lat)
                        and [t[0] for t in lat] == x["first_ns"],
                        "lost interval",
                    )
                    require(
                        all(
                            isclose(x["interbyte"][k], pct(sum(lat, []), frac))
                            for k, frac in [
                                ("p50_ms", 0.5),
                                ("p95_ms", 0.95),
                                ("p99_ms", 0.99),
                                ("max_ms", 1),
                            ]
                        ),
                        "wrong tail",
                    )
                    require(
                        x["work"]["instructions"] == sum(x["opcodes"].values())
                        and x["opcodes"]["EMIT"] == 64 * b
                        and x["live"]["appends"] == 65 * b
                        and x["native_processes"] == 2 + b,
                        "lost clients",
                    )
                    if engine != "fused":
                        require(
                            x["ready_stats"]["emits"] == 64 * b
                            and x["backbone_batches"] == x["ready_stats"]["inputs_batches"]
                            and x["work"]["choose_batches"] == x["ready_stats"]["choose_batches"],
                            "wrong dispatch",
                        )
                    else:
                        require(
                            x["backbone_batches"] == 64 and not x["acknowledged_during_positional_worker"],
                            "bad barrier",
                        )
                    rr.append(x)

                def add(k):
                    return sum(t[k] for t in rr)

                ns = sum((sum(t["interbyte_ns"], []) for t in rr), [])
                wall = add("wall_ns") / 1e9
                by[f"b{b}/{arm}"] = dict(
                    bytes=len(ns),
                    wall_s=wall,
                    bytes_per_s=len(ns) / wall,
                    p50_ms=pct(ns, 0.5),
                    p95_ms=pct(ns, 0.95),
                    p99_ms=pct(ns, 0.99),
                    max_ms=pct(ns, 1),
                    first_p95_ms=pct(rr[0]["first_ns"] + rr[1]["first_ns"], 0.95),
                    backbone_batches=add("backbone_batches"),
                    controller_batches=sum(t["work"]["choose_batches"] for t in rr),
                    instructions=sum(t["work"]["instructions"] for t in rr),
                    native_lf_steps=sum(t["archive"].get("lf_steps", 0) for t in rr),
                    worker_s=add("position_worker_ns") / 1e9,
                    bytes_during_worker=add("acknowledged_during_positional_worker"),
                    bytes_during_other_occurrence=add("acknowledged_during_occurrence"),
                    cpu_core_s=add("host_cpu_seconds") + add("native_cpu_seconds"),
                    peak_mib=max(t["peak_allocated"] for t in rr) / 2**20,
                )
    require(total == 13056, "missing cells")

    def ratio(b, mode, numerator, denominator, k):
        return by[f"b{b}/{numerator}-{mode}"][k] / by[f"b{b}/{denominator}-{mode}"][k]

    batches = {
        k: ratio(16, "sample", "cohort", "ready", k) for k in ("backbone_batches", "controller_batches")
    }
    rate = ratio(16, "sample", "cohort", "ready", "bytes_per_s")
    greedy = ratio(16, "greedy", "cohort", "fused", "bytes_per_s")
    tails = {m: ratio(1, m, "cohort", "fused", "p99_ms") for m in ("sample", "greedy")}
    overlap = by["b16/cohort-sample"]["bytes_during_worker"]
    passed = (
        not mismatch
        and max(batches.values()) <= 0.5
        and rate >= 1.20
        and overlap > 0
        and max(tails.values()) <= 1.1
        and greedy >= 0.90
    )
    return dict(
        ref=ref,
        attempts=attempts,
        card_s_upper=paid,
        cumulative_x47_upper=previous + paid,
        remaining_s=1800 - previous - paid,
        total_emitted=total,
        mismatches=mismatch,
        metrics=by,
        b16_sampled_batch_ratios=batches,
        b16_sampled_cohort_over_ready_rate=rate,
        b16_sampled_cohort_over_barrier_rate=ratio(16, "sample", "cohort", "fused", "bytes_per_s"),
        b16_greedy_cohort_over_barrier_rate=greedy,
        b1_cohort_over_barrier_p99=tails,
        b16_cohort_overlap=overlap,
        passes=passed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(parser.parse_args().root), indent=2, allow_nan=False))
