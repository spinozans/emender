"""X49-W publication audit. Read finished receipts/checkpoints only; no natural text or model execution.

Prepared as a private source draft while training, outside the qualified run tree. No score
reads before a certified terminal campaign. Output contains aggregates, not document rows.
"""

import json
from math import isclose
from statistics import mean, median
import numpy as np
from experiments.expressivity_tasks.x47_xsa_attention.w3_launch import fields
from experiments.expressivity_tasks.x48_muon_continuation.data import check, load, sha
from experiments.expressivity_tasks.x49_live_trace_memory import write_panel as p
from experiments.expressivity_tasks.x49_live_trace_memory.panel_analysis import (
    read_stage,
)
from experiments.expressivity_tasks.x49_live_trace_memory.write_analysis import analyze
from experiments.expressivity_tasks.x49_live_trace_memory.write_run import AUTH

REF = "a8eda5bf15228c1f198176cb40bfd68cfb7900fc"
PUBLIC = "939789452e969e18d9febfe557c1e1d075f2446e"


def charges(closed, plan):
    out = {}
    raw = load(p.ROOT / "charged.json")
    check(raw == closed["charges"] and raw.keys() == p.JOBS.keys(), "foreign job set")
    for k, j in p.JOBS.items():
        d = p.ROOT / k
        i, a, e = [fields(d / f) for f in ("initial", "after", "released")]
        c = raw[k]
        z = load(d / "complete.json")
        st = load(d / "start.json")
        la = load(d / "launch.json")
        live = load(d / "live.json")
        ident = load(d / "identity.json")
        n = 1024 if j.kind == "train" else plan["units"]
        nt = 4096 * n if j.kind == "train" else p.NT * len(plan["endpoints"])
        check(
            c == load(d / "charge.json")
            and c["job"] == k
            and st["name"] == k
            and st["ref"] == ident["ref"] == REF
            and st["cap_s"] == j.cap
            and z["kind"] == j.kind,
            "foreign cost/identity",
        )
        check(
            c["rc"] == int(e["rc"]) == 0
            and 0
            < c["upper_card_s"]
            == int(e["released_epoch"]) - int(i["created"]) + 1
            <= j.cap
            and c["complete_sha"] == sha(d / "complete.json"),
            "unpaid/incomplete job",
        )
        check(
            c["renewed"]
            and (d / "watch-ack").is_file()
            and i["adopted"] == "1"
            and 0 < live["units"] <= n
            and live["ppid"] == 1
            and int(a["heartbeat"]) >= int(live["heartbeat"]) > int(i["heartbeat"])
            and all(
                i[f] == a[f] == ident["owner"][f]
                for f in ("pid", "starttime", "gpu", "created", "adopted")
            ),
            "unwitnessed owner or heartbeat",
        )
        check(
            live["pid"] == la["pid"] == int(i["pid"])
            and live["start"] == la["start"] == i["starttime"]
            and live["trainer"] == ident["trainer"]
            and live["trainer_start"] == ident["trainer_start"],
            "foreign process identity",
        )
        check(
            wanted_progress(d, n, nt)
            and st["units"] == c["witnessed_units"] == z["units"] == n
            and z["targets"] == nt,
            "incorrect terminal real units/targets",
        )
        out[k] = dict(
            upper_card_s=c["upper_card_s"],
            inclusive_wall_s=c["inclusive_wall_s"],
            owner_and_waited_children_cpu_s=c["owner_and_waited_children_cpu_s"],
            worker_cpu_core_s=z["cpu_core_s"],
            peak_device_allocated=z["gpu_peak_alloc"],
            peak_device_reserved=z["gpu_peak_reserved"],
            peak_process_RSS_KiB=z["process_RSS_peak_KiB"],
            units=n,
            scored_targets=nt,
        )
    check(
        sum(r["upper_card_s"] for r in out.values())
        == closed["upper_card_s"]
        <= p.TOTAL,
        "total charge mismatch",
    )
    return out


def wanted_progress(d, n, nt):
    v = load(d / "progress.json")
    return v["units"] == n and v["scored_targets"] == nt


def trained(closed):
    out = {}
    for k, j in p.JOBS.items():
        if j.kind != "train":
            continue
        d = p.ROOT / k
        r = closed["training"][k]
        z = load(d / "latest.json")
        b = p.binding(REF, j.arms, j.seed)
        hs = sha(d / "steps.jsonl")
        check(
            r["binding"] == z["binding"] == b
            and r["steps_sha256"] == hs
            and r == load(d / "complete.json")
            and z["step"] == 1024
            and z["file"] == "step-1024.pt"
            and (d / z["file"]).stat().st_size == z["bytes"]
            and sha(d / z["file"]) == r["final_recovery"] == z["sha256"],
            "changed training bytes",
        )
        journal = list(map(json.loads, (d / "steps.jsonl").read_bytes().splitlines()))
        loss = [v["preupdate_nll"] for v in journal]
        norms = [v["preclip"] for v in journal]
        check(
            [v["step"] for v in journal] == list(range(1, 1025))
            and all(v["targets"] == 4096 for v in journal)
            and isclose(mean(loss), r["preupdate_train_nll"], abs_tol=1e-12, rel_tol=0),
            "wrong TRAIN denominator",
        )
        out[k] = dict(
            origin_sha=b["origin"],
            terminal_weights_sha=r["final_weights"],
            terminal_recovery_sha=r["final_recovery"],
            preupdate_train_nll=mean(loss),
            first_train_nll=loss[0],
            last128_train_nll=mean(loss[-128:]),
            median_preclip=median(norms),
            updates_clipped=sum(x > 1 for x in norms),
            literal_hits=r["work"]["literal_hits"],
            linked_uses=r["work"].get("linked_uses"),
        )
    for s in p.SEEDS:
        b, q, lw = (out[p.name(a, s)] for a in p.ARMS)
        check(
            b["origin_sha"] == q["origin_sha"] == lw["origin_sha"]
            and b["first_train_nll"] == q["first_train_nll"] == lw["first_train_nll"]
            and b["literal_hits"] == q["literal_hits"] == lw["literal_hits"]
            and q["linked_uses"] == lw["linked_uses"],
            "unmatched W training exposure",
        )
    return out


def independent_primary(sums, ns, analysis):
    # Alternate reduction: divide each separately resampled seed sum by that draw's own
    # observed count, THEN average seeds. Compare to the registered vectorized computation.
    n = len(ns)
    indices = np.random.default_rng(4905).integers(0, n, size=(10000, n))
    den = ns[indices].sum(1)
    for a, b in p.PAIRS:
        ds = [
            sums[f"{b}-{s}@1024"][4096, "none"] - sums[f"{a}-{s}@1024"][4096, "none"]
            for s in p.SEEDS
        ]
        values = np.mean([x[indices].sum(axis=1) / den for x in ds], axis=0)
        r = analysis["contrasts"][a + ":" + b]["full"]["4096"]
        wanted = [x.sum() / ns.sum() for x in ds]
        np.testing.assert_allclose(
            r["ci95"],
            np.quantile(values, [0.025, 0.975], method="linear"),
            rtol=0,
            atol=5e-13,
        )
        np.testing.assert_allclose(r["by_seed"], wanted, rtol=0, atol=5e-13)
        check(
            isclose(r["gain"], mean(wanted), abs_tol=5e-13, rel_tol=0),
            "wrong primary real-byte denominator",
        )


def full_cells(stage):
    ans = {}
    for arm in p.ARMS:
        eps = [stage["endpoints"][f"{arm}-{s}@1024"] for s in p.SEEDS]
        model = {}
        for k in eps[0]["cells"]:
            cs = [e["cells"][k] for e in eps]
            check(all(c["targets"] == p.NT for c in cs), "wrong endpoint size")
            ws = [c["work"] for c in cs]
            check(ws[0] == ws[1] == ws[2], "literal addresses depend on seed")
            ns = [c["all_byte_nll"] for c in cs]
            result = dict(
                by_seed_nll=ns,
                nats_per_byte=mean(ns),
                targets_per_seed=p.NT,
                head_s_per_document=sum(c["head_wall_s"] for c in cs) / (3 * p.N),
                work_per_seed=dict(ws[0]),
            )
            model[k] = result
            if k != "4096:none":
                continue
            groups = {}
            for src, parts in cs[0]["strata"].items():
                tab = {}
                for age in parts:
                    vals = [c["strata"][src][age] for c in cs]
                    cnt = vals[0]["targets"]
                    check(all(x["targets"] == cnt for x in vals), "unpaired age")
                    tab[age] = dict(
                        targets_per_seed=cnt,
                        nats_per_byte=sum(x["nll_sum"] for x in vals) / (3 * cnt)
                        if cnt
                        else None,
                    )
                check(
                    sum(x["targets_per_seed"] for x in tab.values()) == p.NT,
                    "missing no-hit targets",
                )
                groups[src] = tab
            result["by_age"] = groups
            bylen = {}
            for band in cs[0]["by_document_length"]:
                vals = [c["by_document_length"][band] for c in cs]
                v = vals[0]
                counts = {
                    f: v[f] for f in ("documents", "targets", "literal_hit_documents")
                }
                counts["two_hop_documents"] = (
                    v["double_hit_documents"] if arm in ("P", "L") else None
                )
                check(
                    all(
                        all(
                            x[f] == v[f]
                            for f in (
                                "documents",
                                "targets",
                                "double_hit_documents",
                                "literal_hit_documents",
                            )
                        )
                        for x in vals
                    ),
                    "unpaired length cell",
                )
                bylen[band] = dict(
                    **counts,
                    nats_per_byte=sum(x["nll_sum"] for x in vals) / (3 * v["targets"])
                    if v["documents"] >= 2
                    else None,
                )
            check(
                sum(x["documents"] for x in bylen.values()) == p.N
                and sum(x["targets"] for x in bylen.values()) == p.NT,
                "missing short-document mass",
            )
            result["by_length_in_targets"] = bylen
        ans[arm] = dict(
            cells=model,
            host_prefill_s_per_document=sum(e["prefill_s"] for e in eps) / (3 * p.N),
            synthetic_full_recompute_decode_s=[
                e["decode_diagnostic"]["full_recompute_decode_s"] for e in eps
            ],
        )
    return ans


def audit():
    root = p.ROOT
    d = root / p.FINAL
    # Refuse even to ingest the score files unless the detached campaign finished and released.
    check(
        not (root / "stop.json").exists() and (root / "done.json").is_file(),
        "W not successfully closed: no result",
    )
    done = load(root / "done.json")
    check(
        done["complete"]
        and done["charges_sha"] == sha(root / "charged.json")
        and done["summary_sha"] == sha(root / "summary.json")
        and (d / "watch-ack").is_file()
        and fields(d / "released")["rc"] == "0",
        "unverified W closure",
    )
    p.validate(load(AUTH), REF)
    plan = load(d / "frozen.json")
    check(
        plan["ref"] == REF
        and plan["rows_sha"] == p.ROW_SHA
        and plan["train_identity"] == p.W_ID
        and plan["spec"] == p.SPEC
        and plan["documents"] == p.N
        and plan["targets_per_endpoint"] == p.NT,
        "foreign W final",
    )
    closed = load(root / "summary.json")
    cost = charges(closed, plan)
    check(analyze() == closed, "closed analysis not reproduced")
    nf = len(plan["endpoints"])
    check(9 <= nf <= 18 and plan["units"] == nf * p.N, "wrong endpoint map")
    needed = {(a, s, 1024) for a in p.ARMS for s in p.SEEDS}
    for group in plan["wall_pairs"].values():
        for seed, pair in group.items():
            for a, v in pair["selected"].items():
                if v is not None:
                    needed.add((a, int(seed), v["step"]))
    check(
        set(plan["endpoints"]) == {f"{a}-{s}@{i}" for a, s, i in needed},
        "unrequested or missing endpoint",
    )
    for k, e in plan["endpoints"].items():
        path, b = p.resolve(e["arm"], e["seed"], e["step"], REF)
        check(
            e["binding"] == b
            and k == f"{e['arm']}-{e['seed']}@{e['step']}"
            and path.stat().st_size == e["bytes"]
            and sha(path) == e["sha256"],
            "post-freeze weight change",
        )
    rows = load(p.ROWS)["panels"]["final_test"]
    ns = np.array([r["length"] - 1 for r in rows])
    sums, stage, freeze = read_stage(root, p.FINAL, rows, job=p.JOBS[p.FINAL])
    check(stage == closed["stage"] and freeze == plan, "wrong score stage")
    independent_primary(sums, ns, closed["analysis"])
    paid_train = sum(x["upper_card_s"] for k, x in cost.items() if k != p.FINAL)
    prep = p.prep()
    check(
        stage["source_prep"]["windows"] == p.N
        and stage["source_prep"]["full_document_source_bytes"] == p.NT + p.N,
        "short source byte mismatch",
    )
    return dict(
        scope="X49-W closed-receipt audit, no natural text reopened",
        qualified_source=REF,
        qualified_public_source=PUBLIC,
        spec_sha=p.SPEC,
        summary_sha=done["summary_sha"],
        charges_sha=done["charges_sha"],
        frozen_map_sha=closed["freeze_sha"],
        rows_sha=p.ROW_SHA,
        train_identity=p.W_ID,
        seeds=list(p.SEEDS),
        fresh_short_documents=p.N,
        real_targets_per_endpoint_cell=p.NT,
        unreserved_short_documents=prep["reservation"]["unused_short"],
        distinct_endpoints=nf,
        endpoint_windows=plan["units"],
        full_and_wall=closed["analysis"],
        wall=closed["wall"],
        training=trained(closed),
        full_dose=full_cells(stage),
        charges=cost,
        training_upper_card_s=paid_train,
        final_upper_card_s=cost[p.FINAL]["upper_card_s"],
        incremental_upper_card_s=closed["upper_card_s"],
        driver_cpu_s=done["driver_and_short_parents_cpu_s"],
        shared_train_prep_wall_s=prep["total_prep_wall_s"],
        shared_train_prep_cpu_s=prep["total_prep_cpu_s"],
        final_source_prep=stage["source_prep"],
        synthetic_short_device_check=closed["device_short_check"],
    )


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2, allow_nan=False))
