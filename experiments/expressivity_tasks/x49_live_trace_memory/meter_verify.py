"""Post-release CPU verifier and planning estimate. No checkpoint tensors, corpus reader or launcher."""

from __future__ import annotations
import json
import math
import os
from .meter import CAP, RUN, fields, validate_approval
from .panels import OUT, T
from ..x48_muon_continuation.data import check, load, sha

ARMS = ("A", "B", "F", "C")
REF = "7dd546d6b2190a70a3ec468e157bfcc0c2b0c25e"
ORIGIN = "cf7337b520958056a648e4f78a939ded3ed5a4d1bc8f871d8a900d64464b1357"
PAIRS = (("B", "A"), ("C", "A"), ("C", "B"), ("C", "F"))


def verify(root=RUN):
    validate_approval(load(OUT / "meter-authorization.json"), REF)
    info = load(root / "identity.json")
    initial = fields(root / "initial")
    end = fields(root / "released")
    last = fields(root / "after")
    o = load(root / "launch.json")
    charge = load(root / "charge.json")
    check(
        info["origin"] == ORIGIN
        and info["ref"] == REF
        and info["natural_targets"] == 0
        and o == dict(pid=int(initial["pid"]), start=initial["starttime"]),
        "foreign origin/owner",
    )
    check(
        all(
            initial[k] == last[k] == info["owner"][k]
            for k in ("pid", "starttime", "gpu", "created", "adopted")
        )
        and initial["adopted"] == "1"
        and int(initial["heartbeat"]) < int(last["heartbeat"]),
        "lease not renewed",
    )
    rows = [json.loads(s) for s in (root / "survival.jsonl").read_text().splitlines()]
    ns = [r["updates"] for r in rows]
    check(
        all(
            (r["pid"], r["start"], r["ppid"], r["trainer"], r["trainer_start"])
            == (o["pid"], o["start"], 1, info["pid"], info["trainer_start"])
            for r in rows
        )
        and ns == sorted(set(ns))
        and len(ns) >= 2
        and ns[-1] == 16
        and load(root / "live.json") in rows
        and (root / "watch-ack").is_file(),
        "incomplete detached survival evidence",
    )
    seconds = int(end["released_epoch"]) - int(initial["created"]) + 1
    check(
        int(end["rc"]) == 0
        and charge
        == dict(
            rc=0,
            upper_card_seconds=seconds,
            pid=o["pid"],
            gpu=initial["gpu"],
            witnessed_updates=16,
            renewal_and_progress=True,
        )
        and 0 < seconds <= CAP,
        "wrong terminal charge",
    )
    check(
        load(root / "complete.json")
        == dict(
            arms=list(ARMS),
            physical_updates=16,
            natural_training_targets=0,
            natural_evaluation_targets=0,
        ),
        "wrong dose",
    )
    out = {}
    for arm in ARMS:
        d = root / arm
        r = load(d / "completed.json")
        steps = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
        ev = r["synthetic_inference"]
        binding = dict(
            schema="x49-synthetic-meter-v1", origin=ORIGIN, seed=4901, ref=REF, arm=arm
        )
        check(
            r["arm"] == arm
            and r["updates"] == steps
            and [(s["step"], s["physical"], s["targets"]) for s in steps]
            == [(i, n, T) for n, i in enumerate((1, 2, 3, 3), 1)],
            "incorrect physical updates",
        )
        check(
            r["exact_successor"]
            and all(
                steps[-2][k] == steps[-1][k]
                for k in ("preupdate_nll", "preclip", "host_lr", "head_lr", "work")
            )
            and len(bytes.fromhex(r["successor_state_sha"])) == 32,
            "successor mismatch",
        )
        for s in steps:
            w = s["work"]
            check(
                w["literal_queries"] == w["literal_hits"] + w["literal_misses"] == T
                and w["private_writes"] == (T - 1 if arm in "FC" else 0)
                and w["private_queries"] == (T if arm in "FC" else 0),
                "wrong actual work",
            )
        for key, note, filename in [
            ("weights", "weights-0002.json", "weights-0002.pt"),
            ("checkpoint", "latest.json", "step-0002.pt"),
        ]:
            ck = r[key]
            p = d / filename
            check(
                ck == load(d / note)
                and ck["file"] == filename
                and ck["step"] == 2
                and ck["binding"] == binding
                and p.stat().st_size == ck["bytes"]
                and sha(p) == ck["sha256"],
                "foreign or incomplete checkpoint",
            )
        cells = ev["cells"]
        check(
            ev["host_tokens"] == T
            and ev["host_prefills"] == 1
            and [(c["horizon"], c["sham"]) for c in cells]
            == [(128, "none"), (1024, "none"), (4096, "none")]
            + ([(4096, "literal")] if arm != "A" else [])
            + ([(4096, "trace")] if arm in "FC" else [])
            and all(c["targets"] == T for c in cells),
            "wrong inference geometry",
        )
        ts = [s["wall_s"] for s in steps]
        inf = ev["host_prefill_s"] + sum(c["head_score_s"] for c in cells)
        other = max(
            0.0,
            r["arm_wall_s"]
            - (
                r["setup_s"]
                + sum(ts)
                + inf
                + sum(
                    r[k]
                    for k in (
                        "weights_save_s",
                        "full_save_s",
                        "state_hash_s",
                        "verify_restore_s",
                    )
                )
            ),
        )
        out[arm] = dict(
            arm_wall_s=r["arm_wall_s"],
            cpu_core_s=sum(r["arm_cpu_s"]),
            setup_s=r["setup_s"],
            update_wall_s=ts,
            step_accounting_proxy_s=other / 4,
            inference_window_s=inf,
            full_prefill_s=ev["host_prefill_s"],
            weights_write_s=r["weights_save_s"],
            recovery_write_s=r["full_save_s"],
            recovery_load_s=r["verify_restore_s"],
            weights_bytes=r["weights"]["bytes"],
            recovery_bytes=r["checkpoint"]["bytes"],
            peak_allocated_bytes=r["allocated_peak"],
            peak_reserved_bytes=r["reserved_peak"],
            lifetime_peak_RSS_KiB=r["process_lifetime_peak_rss_KiB"],
            successor_sha256=r["successor_state_sha"],
            completed_receipt_sha256=sha(d / "completed.json"),
        )
    return dict(
        scope="fabricated four-window real-host local meter; not natural quality or a spending grant",
        qualified_ref=REF,
        device=info["device"],
        source_checkpoint=ORIGIN,
        charge_sha256=sha(root / "charge.json"),
        upper_card_seconds=seconds,
        witnessed_physical_updates=16,
        exact_successors=4,
        natural_training_targets=0,
        evaluation_bytes_read=0,
        arms=out,
    )


def forecast(meter):
    """Point estimates from tiny synthetic samples plus explicit margins. No assertion about natural speed."""
    a = meter["arms"]
    warm = {k: max(b["update_wall_s"][1:3]) for k, b in a.items()}
    cold = max(0.0, max(a[k]["update_wall_s"][0] - v for k, v in warm.items()))
    overhead = max(
        0.0, meter["upper_card_seconds"] - sum(b["arm_wall_s"] for b in a.values())
    )
    train = {}
    evaluation = {}
    for k, b in a.items():
        train[k] = (
            overhead
            + b["setup_s"]
            + cold
            + 1024 * (warm[k] + b["step_accounting_proxy_s"])
            + 32 * b["weights_write_s"]
            + 8 * b["recovery_write_s"]
        )
        # Charge each frozen endpoint its own entire lease startup, setup and a hash-verified
        # full recovery load as proxy for weights-only loading. Charge all H/shams per doc.
        evaluation[k] = overhead + b["setup_s"] + b["recovery_load_s"]
    full = 3 * sum(
        2 * evaluation[k] + (110 + 128) * b["inference_window_s"] for k, b in a.items()
    )
    extra = 3 * sum(
        max(evaluation[k] + 128 * a[k]["inference_window_s"] for k in pair)
        for pair in PAIRS
    )
    conservative_extra = 3 * sum(
        3 * (evaluation[k] + 128 * b["inference_window_s"]) for k, b in a.items()
    )
    train_total = 3 * sum(train.values())
    disk = (
        3 * sum(32 * b["weights_bytes"] + b["recovery_bytes"] for b in a.values())
        + max(b["recovery_bytes"] for b in a.values())
        + max(b["weights_bytes"] for b in a.values())
    )
    # Round hard proposals upwards to 1 minute for trainees and five minutes for eval stages.
    # Stage checks and final use separate one-shot sessions. Bound >1.5x their largest forecast.
    train_cap = 60 * math.ceil(1.5 * max(train.values()) / 60)
    stage_check_cap = {
        label: 300
        * math.ceil(
            1.5
            * 3
            * sum(evaluation[k] + 110 * a[k]["inference_window_s"] for k in pair)
            / 300
        )
        for label, pair in [("AB", "AB"), ("FC", "FC")]
    }
    final_cap = 300 * math.ceil(
        1.5
        * (
            3 * sum(evaluation[k] + 128 * b["inference_window_s"] for k, b in a.items())
            + extra
        )
        / 300
    )
    return dict(
        scope="proposal only; synthetic costs are not natural bounds",
        assumed_trajectories=12,
        updates_per_trajectory=1024,
        analysis_milestones_per_trajectory=32,
        recovery_writes_per_trajectory=8,
        broker_overhead_proxy_s=overhead,
        cold_update_premium_s=cold,
        train_s_per_trajectory=train,
        train_s=train_total,
        full_dose_cal_and_final_s=full,
        max_additional_wall_endpoints=3 * len(PAIRS),
        additional_wall_endpoints_s=extra,
        total_projected_card_s=train_total + full + extra,
        all_other_arms_sensitivity_s=train_total + full + conservative_extra,
        estimated_panel_disk_bytes=disk,
        proposal=dict(
            train_cap_each_s=train_cap,
            calibration_caps_s=stage_check_cap,
            final_campaign_cap_s=final_cap,
            whole_panel_cap_s=12 * train_cap
            + sum(stage_check_cap.values())
            + final_cap,
            min_free_disk_GiB=320,
            prior_meter_separate=True,
            automatic_retries=0,
        ),
    )


if __name__ == "__main__":
    check(
        os.getenv("CUDA_VISIBLE_DEVICES") == "" and not os.getenv("GPU_LEASE_HELD"),
        "post-release CPU only",
    )
    m = verify()
    print(
        json.dumps(
            dict(measured=m, forecast=forecast(m)), sort_keys=True, allow_nan=False
        )
    )
