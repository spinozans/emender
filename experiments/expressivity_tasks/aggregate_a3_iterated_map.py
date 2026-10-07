#!/usr/bin/env python3
"""Aggregate the taskA3 phase-1 8M iterated-map reference tables.

Two tables:

1. Contractive control (logistic `iterated_nonlinear_map`, the task-brief's
   named family; lambda = -0.201 so no compounding wall exists here): the
   existing e97raw battery results (e97 / e97-linear / e97-raw / gdn-1.6M
   historical) plus the new parameter-matched gdn-8m arm, seeds {42,123,456},
   per-seed and pooled error-vs-T (train T=128, eval to 1024 = 8x).

2. Wall reference (modular_quadratic p=256, the depth-wall family): the 8-seed
   grok confirmation curves (e97 flat vs e97-lin / gdn2 collapsing) that the
   1.3B phase-2 readout is scored against.

Output: paper/results/expressivity_1p3b/iterated-map/phase1_8m_control_table.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "experiments" / "expressivity_tasks" / "results"
OUT = (REPO / "paper" / "results" / "expressivity_1p3b" / "iterated-map"
       / "phase1_8m_control_table.json")

LOGISTIC_ARMS = ["e97", "e97-linear", "e97-raw", "gdn", "gdn-8m"]
SEEDS = [42, 123, 456]


def logistic_table():
    table = {}
    for arm in LOGISTIC_ARMS:
        per_seed = {}
        for seed in SEEDS:
            p = SUITE / f"e97raw_iterated_nonlinear_map__{arm}__seed{seed}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            per_seed[str(seed)] = {
                "params": d["params"],
                "final_acc": round(d["final_acc"], 6),
                "final_loss": round(d["final_loss"], 6),
                "acc_vs_T": {k: round(v["acc"], 6)
                             for k, v in d["length_extrap"].items()},
            }
        if not per_seed:
            continue
        accs_by_len: dict[str, list[float]] = {}
        for rec in per_seed.values():
            for T, a in rec["acc_vs_T"].items():
                accs_by_len.setdefault(T, []).append(a)
        pooled = {}
        for T, vals in sorted(accs_by_len.items(), key=lambda kv: int(kv[0])):
            n = len(vals)
            mean = sum(vals) / n
            std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
            pooled[T] = {"mean_acc": round(mean, 6), "std_acc": round(std, 6),
                         "n_seeds": n}
        table[arm] = {"per_seed": per_seed, "pooled_acc_vs_T": pooled,
                      "params": per_seed[next(iter(per_seed))]["params"]}
    return table


def wall_reference():
    src = REPO / "experiments" / "grok_symmetric_width" / "confirm_signature.json"
    d = json.loads(src.read_text())
    out = {}
    for key, rec in d.get("part1_mq_extrap", {}).items():
        # keys like p256_d1024_e97 (8-seed confirm; grok-filtered curves)
        out[key] = {
            "n": rec.get("n"), "n_grok": rec.get("n_grok"),
            "grok_filtered": rec.get("grok_filtered"),
            "acc_vs_T": rec["curve"],
        }
    return out


def main():
    rec = {
        "task": "taskA3-phase1-8M-reference",
        "logistic_contractive_control": {
            "note": "logistic iterated_nonlinear_map (the brief's named family): "
                    "lambda = -0.2012 < 0 (lyapunov_logistic_map.json) -> no "
                    "compounding wall; this table isolates parameters as NOT the "
                    "lever on the contractive family. gdn (1.6M) is the historical "
                    "suite arm, param-mismatched vs e97 (7.4M) -- stated verbatim; "
                    "gdn-8m (7.94M, +7.6% vs e97) closes that hole.",
            "protocol": "run_e97_raw_expressivity.py -> train_hybrid.py: 5000 "
                        "steps, schedulefree AdamW lr 3e-4, depth 4, dim 256, "
                        "batch 32, train T=128, eval {128,256,512,1024}, "
                        "8 batches/length, seeds 42/123/456",
            "arms": logistic_table(),
        },
        "modular_quadratic_wall_reference": {
            "note": "the depth-wall family (non-contracting x^2+c mod p): 8-seed "
                    "grok confirmation (experiments/grok_symmetric_width/"
                    "CONFIRM.md), grok-filtered test-acc-vs-T at the decisive "
                    "cells. e97 flat to T=4096; the linear arms (e97-lin, gdn2) "
                    "collapse toward the 1/p baseline. These are the curves the "
                    "1.3B phase-2 readout is scored against.",
            "cells": wall_reference(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2))
    print(f"-> {OUT}")
    # console summary
    print("\nlogistic control (pooled mean acc vs T):")
    for arm, t in rec["logistic_contractive_control"]["arms"].items():
        row = " ".join(f"T{k}={v['mean_acc']:.3f}" for k, v in t["pooled_acc_vs_T"].items())
        print(f"  {arm:<11} params={t['params']:>8} {row}")
    print("\nwall reference (grok-filtered, decisive cells p=256):")
    for key, c in rec["modular_quadratic_wall_reference"]["cells"].items():
        if not key.startswith("p256"):
            continue
        row = " ".join(f"T{k}={v:.3f}" for k, v in c["acc_vs_T"].items())
        print(f"  {key:<18} grok {c['n_grok']}/{c['n']} {row}")


if __name__ == "__main__":
    raise SystemExit(main())
