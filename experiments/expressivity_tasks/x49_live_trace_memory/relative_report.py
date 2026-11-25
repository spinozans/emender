"""Local analysis helper: only consume closed X49-R receipts, never reopen either split or write into a run."""

import json
import statistics
from collections import Counter
from experiments.expressivity_tasks.x49_live_trace_memory import relative_panel as r
from experiments.expressivity_tasks.x49_live_trace_memory.panel_analysis import (
    read_stage,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel_spec import common_wall
from experiments.expressivity_tasks.x49_live_trace_memory.relative_data import (
    BASE,
    RESERVATION,
)
from experiments.expressivity_tasks.x47_xsa_attention.w3_launch import fields
from experiments.expressivity_tasks.x48_muon_continuation.data import load, sha


def audit():
    if not __debug__:
        raise RuntimeError("do not disable audit assertions")
    d = r.ROOT
    done = load(d / "done.json")
    assert (
        done["complete"]
        and not (d / "stop.json").exists()
        and done["summary_sha"] == sha(d / "summary.json")
        and done["charges_sha"] == sha(d / "charged.json")
    )
    summary = load(d / "summary.json")
    charges = load(d / "charged.json")
    assert charges == summary["new_charges"] and set(charges) == set(r.JOBS)
    costs = {}
    for k, j in r.JOBS.items():
        base = d / k
        c = charges[k]
        first, last, end = (fields(base / f) for f in ("initial", "after", "released"))
        term = load(base / "complete.json")
        st = load(base / "start.json")
        assert (
            load(base / "charge.json") == c
            and c["job"] == k
            and c["rc"] == int(end["rc"]) == 0
            and c["renewed"]
            and int(last["heartbeat"]) > int(first["heartbeat"])
            and all(
                first[f] == last[f]
                for f in ("pid", "starttime", "created", "adopted", "gpu")
            )
            and first["adopted"] == "1"
            and (base / "watch-ack").is_file()
            and c["upper_card_s"]
            == int(end["released_epoch"]) - int(first["created"]) + 1
            and 0 < c["upper_card_s"] <= j.cap
            and c["witnessed_units"] == term["units"] == st["units"]
            and c["complete_sha"] == sha(base / "complete.json")
        )
        costs[k] = {
            f: c[f]
            for f in (
                "upper_card_s",
                "witnessed_units",
                "inclusive_wall_s",
                "owner_and_waited_children_cpu_s",
            )
        }
        costs[k].update(
            {
                f: term[f]
                for f in ("gpu_peak_alloc", "gpu_peak_reserved", "process_RSS_peak_KiB")
            }
        )
    assert (
        sum(v["upper_card_s"] for v in costs.values())
        == summary["new_card_s"]
        <= r.TOTAL
    )
    prep = load(BASE / "prepared.json")
    assert (
        sha(RESERVATION) == prep["reservation_sha"]
        and prep["raw_bytes_opened"] == 0
        and prep["fresh"] == r.N
    )
    plan = load(d / r.FINAL / "frozen.json")
    assert (
        plan["reservation_sha"] == sha(RESERVATION)
        and plan["parent_summary_sha"] == summary["parent_summary_sha"] == r.SUMMARY
        and plan["documents"] == r.N
        and plan["units"] == r.N * len(plan["endpoints"])
    )
    assert plan["ref"] == "c0f19aef96baf62697a67b034c5890385739bef2" and summary[
        "freeze_sha"
    ] == sha(d / r.FINAL / "frozen.json")
    runs = r.runs_for(plan["ref"])
    w = common_wall(runs, pairs=r.PAIRS, name=r.name)
    assert w == plan["wall_pairs"] == summary["wall"]
    assert plan["training_charges"] == {k: v["charge_sha"] for k, v in runs.items()}
    t, stage, _ = read_stage(d, r.FINAL, load(RESERVATION)["rows"], job=r.JOBS[r.FINAL])
    assert stage == summary["final_stage"]
    analysis = r.compare(t, w)
    assert analysis == summary["analysis"]
    training = {}
    final = {}
    for a in "RBFC":
        traj = []
        cells = {}
        ks = [r.name(a, s) for s in r.SEEDS]
        for k in ks:
            base = (d if a == "R" else r.OLD) / k
            cl = load(base / "complete.json")
            assert cl["steps_sha256"] == sha(base / "steps.jsonl")
            z = [
                json.loads(line)
                for line in (base / "steps.jsonl").read_text().splitlines()
            ]
            assert [v["step"] for v in z] == list(range(1, 1025))
            vals = [v["preupdate_nll"] for v in z]
            norm = [v["preclip"] for v in z]
            traj.append(
                dict(
                    first_nll=vals[0],
                    first_preclip=norm[0],
                    first128_nll=statistics.mean(vals[:128]),
                    last128_nll=statistics.mean(vals[-128:]),
                    train_nll=cl["preupdate_train_nll"],
                    median_preclip=statistics.median(norm),
                    clipped=sum(v > 1 for v in norm),
                    upper_card_s=load(base / "charge.json")["upper_card_s"],
                    terminal_weights=cl["final_weights"],
                    recovery_sha=cl["final_recovery"],
                )
            )
        training[a] = traj
        ends = [stage["endpoints"][k + "@1024"] for k in ks]
        for key in ends[0]["cells"]:
            cs = [e["cells"][key] for e in ends]
            assert all(x["targets"] == r.N * r.T for x in cs)
            cells[key] = dict(
                by_seed=[x["all_byte_nll"] for x in cs],
                nll=statistics.mean(x["all_byte_nll"] for x in cs),
                sham_changes=sum(
                    x["work"].get("private_payloads_reassigned", 0)
                    + x["work"].get("literal_payloads_reassigned", 0)
                    for x in cs
                ),
                head_s_per_window=sum(x["head_wall_s"] for x in cs) / (len(cs) * r.N),
            )
            if key == "4096:none":
                cells[key]["work_across_seeds"] = dict(
                    sum((Counter(x["work"]) for x in cs), Counter())
                )
                strata = {}
                for src, gg in cs[0]["strata"].items():
                    strata[src] = {}
                    for g in gg:
                        parts = [x["strata"][src][g] for x in cs]
                        n = sum(y["targets"] for y in parts)
                        strata[src][g] = dict(
                            targets=n,
                            nll=sum(p["nll_sum"] for p in parts) / n if n else None,
                        )
                    assert (
                        sum(y["targets"] for y in strata[src].values()) == 3 * r.N * r.T
                    )
                cells[key]["strata"] = strata
        final[a] = dict(
            cells=cells,
            host_prefill_s_per_window=sum(x["prefill_s"] for x in ends) / (3 * r.N),
            synthetic_recompute_decode_s=[
                x["decode_diagnostic"]["full_recompute_decode_s"] for x in ends
            ],
        )
    return dict(
        scope="X49-R closed-receipt audit; no corpus access in this verifier",
        ref=plan["ref"],
        summary_sha=done["summary_sha"],
        charges_sha=done["charges_sha"],
        endpoint_map_sha=summary["freeze_sha"],
        parent_summary_sha=summary["parent_summary_sha"],
        parent_training_ref=r.TRAIN_REF,
        seeds=list(r.SEEDS),
        all_scored_final_docs=r.N,
        targets_per_endpoint_cell=r.N * r.T,
        endpoint_count=len(plan["endpoints"]),
        endpoint_windows=plan["units"],
        analysis=analysis,
        wall=w,
        training=training,
        final=final,
        new_charges=costs,
        new_upper_card_s=summary["new_card_s"],
        driver_cpu_s=done["driver_and_short_parents_cpu_s"],
        metadata_prep=prep,
        stage_prep=stage["source_prep"],
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True, allow_nan=False))
