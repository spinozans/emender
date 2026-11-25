"""Audit the completed X49 panel and build a small public aggregate. No raw text or model loads."""

from __future__ import annotations

from collections import Counter
import json
from statistics import mean

import numpy as np

from ..x47_xsa_attention.w3_launch import fields
from ..x48_muon_continuation.data import load, sha
from .panel_analysis import contrasts, read_stage, summarize
from .panel_freeze import read_training
from .panel_spec import JOBS, OUT, ROOT, SEEDS, T, TOTAL, check, common_wall
from .train_stream import V2_IDENTITY

TRAIN_REF = "b63e3e17786d91456daf7281d6f0699b8a560c79"
SUMMARY = "6e30a78d88480d670851a4da505e18bed154c261ab7103cf87bc0e602a515bb6"


def report():
    r = ROOT
    done = load(r / "done.json")
    s = load(r / "summary.json")
    check(
        done["complete"]
        and not (r / "stop.json").exists()
        and sha(r / "summary.json") == done["summary_sha"] == SUMMARY
        and sha(r / "charged.json") == done["charges_sha"]
        and load(r / "charged.json") == s["charges"]
        and set(JOBS) == set(s["charges"]),
        "incomplete or changed panel",
    )
    check(
        sha(OUT / "v2/train/identity.json") == V2_IDENTITY
        and load(OUT / "v2/train/identity.json")["reservation_sha"]
        == sha(OUT / "v2/reservation.json"),
        "foreign reservation",
    )
    rows = load(OUT / "v2/reservation.json")["panels"]
    billed = {}
    clocks = {}
    training = {}
    stage_costs = {}
    for k, j in JOBS.items():
        d = r / k
        c = s["charges"][k]
        i = fields(d / "initial")
        after = fields(d / "after")
        end = fields(d / "released")
        term = load(d / "complete.json")
        st = load(d / "start.json")
        check(
            c == load(d / "charge.json")
            and c["job"] == k
            and c["rc"] == int(end["rc"]) == 0
            and c["renewed"]
            and int(after["heartbeat"]) > int(i["heartbeat"])
            and all(
                i[f] == after[f]
                for f in ("gpu", "pid", "starttime", "created", "adopted")
            )
            and i["adopted"] == "1"
            and (d / "watch-ack").is_file()
            and c["upper_card_s"] == int(end["released_epoch"]) - int(i["created"]) + 1
            and 0 < c["upper_card_s"] <= j.cap
            and c["witnessed_units"] == term["units"] == st["units"]
            and c["complete_sha"] == sha(d / "complete.json")
            and st["ref"] == TRAIN_REF,
            "uncertified lease: " + k,
        )
        billed[k] = {
            f: c[f]
            for f in (
                "upper_card_s",
                "inclusive_wall_s",
                "owner_and_waited_children_cpu_s",
                "witnessed_units",
            )
        }
        if j.kind == "train":
            clocks[k] = read_training(r, j.arms, j.seed, TRAIN_REF)
            check(s["train"][k] == term, "foreign summary training endpoint")
            training[k] = dict(
                targets=term["targets"],
                train_nll=term["preupdate_train_nll"],
                gpu_peak_alloc=term["gpu_peak_alloc"],
                gpu_peak_reserved=term["gpu_peak_reserved"],
                process_RSS_peak_KiB=term["process_RSS_peak_KiB"],
                recovery_sha=term["final_recovery"],
                weights_sha=term["final_weights"],
            )
        else:
            stage = s["stages"][k]
            stage_costs[k] = {
                z: stage[z]
                for z in (
                    "gpu_peak_alloc",
                    "gpu_peak_reserved",
                    "process_RSS_peak_KiB",
                    "worker_cpu_core_s",
                    "source_prep",
                )
            }
    check(
        sum(c["upper_card_s"] for c in billed.values()) == s["total_card_s"] <= TOTAL,
        "panel overcharge",
    )
    wall = common_wall(clocks)
    check(wall == s["wall"], "changed wall endpoints")
    for stage in ["cal-AB", "cal-FC", "final"]:
        scores, stats, plan = read_stage(r, stage, rows[JOBS[stage].kind])
        check(stats == s["stages"][stage], "changed closed score receipt")
    check(
        plan["ref"] == TRAIN_REF
        and plan["wall_pairs"] == wall
        and plan["units"] == 2688
        and plan["documents"] == 128,
        "foreign final",
    )
    decision = contrasts(scores, wall)
    check(decision == s["decision"], "decision reproduction mismatch")
    arms = {}
    draw = np.random.default_rng(4900).integers(0, 128, size=(10000, 128))
    for arm in "ABFC":
        eps = [stats["endpoints"][f"{arm}-{seed}@1024"] for seed in SEEDS]
        cells = {}
        for key in eps[0]["cells"]:
            cs = [e["cells"][key] for e in eps]
            work = sum((Counter(x["work"]) for x in cs), Counter())
            strata = {}
            for src, groups in cs[0]["strata"].items():
                strata[src] = {}
                for g in groups:
                    parts = [x["strata"][src][g] for x in cs]
                    n = sum(p["targets"] for p in parts)
                    strata[src][g] = {
                        "targets": n,
                        "mean_nll": sum(p["nll_sum"] for p in parts) / n if n else None,
                    }
                check(
                    sum(x["targets"] for x in strata[src].values()) == 3 * 128 * T,
                    "lost strata targets",
                )
            cells[key] = dict(
                nll=mean(x["all_byte_nll"] for x in cs),
                nll_by_seed=[x["all_byte_nll"] for x in cs],
                head_s_per_window=sum(x["head_wall_s"] for x in cs) / (3 * 128),
            )
            if key == "4096:none":
                cells[key].update(work=dict(work), strata=strata)
            if not key.endswith(":none"):
                cells[key]["changed_payloads"] = sum(
                    x["work"].get("private_payloads_reassigned", 0)
                    + x["work"].get("literal_payloads_reassigned", 0)
                    for x in cs
                )
        arms[arm] = dict(
            cells=cells,
            host_prefill_s_per_window=sum(e["prefill_s"] for e in eps) / (3 * 128),
            synthetic_full_recompute_decode_s=[
                e["decode_diagnostic"]["full_recompute_decode_s"] for e in eps
            ],
        )
    # Secondary descriptive contrast. No matched-wall selection or nomination rule for F:B.
    fb = {
        str(h): summarize(
            [
                scores[f"B-{z}@1024"][h, "none"] - scores[f"F-{z}@1024"][h, "none"]
                for z in SEEDS
            ],
            draw,
        )
        for h in (128, 1024, 4096)
    }
    return dict(
        schema="x49-natural-v2-completed-aggregate",
        trained_ref=TRAIN_REF,
        summary_sha256=SUMMARY,
        charges_sha256=done["charges_sha"],
        final_freeze_sha256=sha(r / "final/frozen.json"),
        train_identity=V2_IDENTITY,
        seeds=list(SEEDS),
        distinct_train_targets=1024 * T,
        physical_train_updates=12 * 1024,
        physical_train_targets=12 * 1024 * T,
        final_distinct_targets=128 * T,
        final_endpoints=len(scores),
        final_endpoint_windows=plan["units"],
        all_final_cells_include_misses=True,
        arms=arms,
        decision=decision,
        secondary_F_vs_B_exposure=fb,
        wall=wall,
        training=training,
        stage_costs=stage_costs,
        charges=billed,
        total_upper_card_s=s["total_card_s"],
        total_owner_and_children_cpu_s=sum(
            c["owner_and_waited_children_cpu_s"] for c in billed.values()
        ),
        driver_and_short_parents_cpu_s=done["driver_and_short_parents_cpu_s"],
        shared_train_prep_wall_s=s["shared_successful_prep_wall_s"],
        shared_train_prep_cpu_s=s["shared_successful_prep_cpu_s"],
        native_archive_calls=s["native_archive_calls"],
    )


if __name__ == "__main__":
    print(json.dumps(report(), allow_nan=False, sort_keys=True, indent=2))
