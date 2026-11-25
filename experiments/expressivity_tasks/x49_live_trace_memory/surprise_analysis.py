"""Closed-receipt analysis of X49-U. No access to natural text, host packets or predictions."""

from __future__ import annotations
from collections import Counter
import json
from math import isclose, isfinite
import numpy as np
from ..x47_xsa_attention.w3_launch import fields
from ..x48_muon_continuation.data import check, load, sha
from . import surprise_data as data, surprise_panel as p
from .panels import SEEDS, T
from .surprise_panel import journal

ARMS = "BNUS"
PAIRS = (("U", "N"), ("U", "S"), ("U", "B"), ("N", "B"), ("S", "N"))


def summarize(role, logs, hashes, *, short_targets=None):
    n = len(hashes)
    final = role == "final_test"
    check(final == (short_targets is not None), "need actual final target counts")
    sizes = [T] * n if not final else list(short_targets)
    check(
        len(sizes) == n
        and (not final or all(type(v) is int and 1 <= v < T for v in sizes)),
        "invalid short lengths",
    )
    return copy_summary(
        role,
        logs,
        hashes,
        sizes,
        arms=ARMS,
        pairs=PAIRS,
        seed=4908 if final else 4907,
        responses="NUS",
    )


def copy_summary(
    role, logs, hashes, sizes, *, arms, pairs, seed, responses, extra_clocks=()
):
    """Shared conservation, support and paired reductions for predictors that adjust only copy odds."""
    n = len(hashes)
    nt = sum(sizes)
    scored = role != "fit"
    check(
        role in ("fit", "probe", "final_test")
        and n == len(set(hashes)) == len(sizes)
        and n > 0
        and set(logs) == set(SEEDS),
        "incomplete U population",
    )
    totals = {}
    support = {}
    clock = Counter()
    losses = np.zeros((len(SEEDS), n, len(arms))) if scored else None
    for si, s in enumerate(SEEDS):
        work = Counter()
        docs = Counter()
        for i, (h, x) in enumerate(zip(hashes, logs[s], strict=True)):
            length = sizes[i]
            w = x["work"]
            hit = w["literal_hits"]
            miss = w["literal_misses"]
            right = x["candidate_correct"]
            secs = x["seconds"]
            check(
                x["document_sha"] == h
                and x["targets"] == length
                and x["host_prefills"] == 1
                and ("nll_sums" in x) == scored,
                "wrong U role or pairing",
            )
            check(
                all(type(v) is int and v >= 0 for v in w.values())
                and type(right) is int
                and 0 <= right <= hit
                and hit + miss == length
                and w["literal_queries"]
                == w["literal_value_writes"]
                == w["mark_storage_bytes"]
                == length
                and w["literal_value_reads"]
                == w["U_mark_reads"]
                == w["S_mark_reads"]
                == hit
                and w["arrivals_encoded"]
                == w["mark_norms_from_current"] + w["mark_norms_extra"]
                == length - 1
                and w["marks_saturated"] <= length - 1
                and max(
                    w[k]
                    for k in (
                        "B_candidate_code_saturated",
                        "now_candidate_code_saturated",
                        "U_read_code_saturated",
                        "S_read_code_saturated",
                    )
                )
                <= hit
                and w["S_codes_changed"] <= w["S_slots_displaced"] <= hit,
                "lost target or foreign mark work",
            )
            ref = logs[SEEDS[0]][i]
            check(
                right == ref["candidate_correct"]
                and all(
                    w[k] == ref["work"][k]
                    for k in w
                    if k.startswith("literal_")
                    or k in ("S_slots_displaced", "mark_norms_from_current")
                ),
                "literal geometry/byte depends on seed",
            )
            needed = (
                {
                    "prefill_s",
                    "base_B_s",
                    "shared_current_s",
                    "shared_US_mark_build_s",
                    "U_mark_read_s",
                    "S_rotation_and_mark_read_s",
                }
                | set(extra_clocks)
                | (
                    {"fit_statistics_s"}
                    if role == "fit"
                    else {"response_" + a + "_s" for a in responses}
                )
            )
            check(
                set(secs) == needed
                and all(isfinite(v) and v >= 0 for v in secs.values()),
                "bad U clock",
            )
            clock.update(secs)
            work.update(w)
            work.update(candidate_correct=right)
            docs.update(
                {
                    k: int(v > 0)
                    for k, v in dict(
                        hits=hit,
                        misses=miss,
                        copy_correct=right,
                        changed_code=w["S_codes_changed"],
                        displaced_slot=w["S_slots_displaced"],
                        saturated_write=w["marks_saturated"],
                    ).items()
                }
            )
            if scored:
                z = x["nll_sums"]
                check(
                    set(z) == set(arms)
                    and all(isfinite(v) and v >= 0 for v in z.values()),
                    "invalid U all-byte sums",
                )
                losses[si, i, :] = [z[a] for a in arms]
        totals[str(s)] = dict(work)
        support[str(s)] = dict(docs)
    report = dict(
        paired_documents=n,
        real_targets_per_seed=nt,
        work_by_seed=totals,
        documents_with_support_by_seed=support,
        mean_seconds_per_seed_document={
            k: v / (len(SEEDS) * n) for k, v in clock.items()
        },
    )
    if losses is None:
        return report
    report.update(paired_table(losses, sizes, arms, pairs, seed))
    return report


def paired_table(losses, sizes, arms, pairs, seed):
    n, nt = len(sizes), sum(sizes)
    check(
        n > 0
        and all(x > 0 for x in sizes)
        and losses.shape == (len(SEEDS), n, len(arms))
        and np.isfinite(losses).all(),
        "bad paired-document loss tensor",
    )
    report = {}
    ix = np.random.default_rng(seed).integers(0, n, size=(10000, n))
    den = np.asarray(sizes, dtype=np.int64)[ix].sum(1)
    report["policies"] = {
        a: dict(
            nats_per_byte=float(losses[:, :, i].mean() / (nt / n)),
            by_seed=(losses[:, :, i].sum(1) / nt).tolist(),
        )
        for i, a in enumerate(arms)
    }
    report["contrasts"] = {}
    for a, b in pairs:
        dt = losses[:, :, arms.index(b)] - losses[:, :, arms.index(a)]
        bootstrap = dt.mean(0)[ix].sum(1) / den
        report["contrasts"][a + ":" + b] = dict(
            gain=float(dt.sum() / (nt * len(SEEDS))),
            by_seed=(dt.sum(1) / nt).tolist(),
            ci95=np.quantile(bootstrap, [0.025, 0.975], method="linear").tolist(),
        )
    return report


def cost(d, z, charge):
    n = p.units()
    return stage_cost(
        d, z, charge, p.JOB, n, n * T, fit_boundary=len(SEEDS) * data.N["fit"]
    )


def stage_cost(d, z, charge, job, n, targets, *, fit_boundary=None):
    ini, after, last, end = (
        fields(d / k) for k in ("initial", "after", "last", "released")
    )
    start = load(d / "start.json")
    ident = load(d / "identity.json")
    la = load(d / "launch.json")
    live = load(d / "live.json")
    obs = list(map(json.loads, (d / "survival.jsonl").read_bytes().splitlines()))
    pr = load(d / "progress.json")
    check(
        start["name"] == ident["job"] == charge["job"] == job.name
        and ident["ref"] == start["ref"]
        and z["kind"] == start["kind"] == job.kind
        and charge["complete_sha"] == sha(d / "complete.json")
        and start["cap_s"] == job.cap
        and charge["renewed"]
        and (d / "watch-ack").is_file(),
        f"invalid {job.name} terminal receipt",
    )
    check(
        ini["adopted"] == "1"
        and all(
            ini[k] == after[k] == last[k] == ident["owner"][k]
            for k in ("pid", "starttime", "created", "gpu", "adopted")
        )
        and int(after["heartbeat"]) >= int(live["heartbeat"]) > int(ini["heartbeat"])
        and all(
            o["ppid"] == 1
            and o["pid"] == la["pid"] == int(ini["pid"])
            and o["start"] == la["start"] == ini["starttime"]
            and o["trainer"] == ident["trainer"]
            and o["trainer_start"] == ident["trainer_start"]
            for o in obs
        ),
        f"foreign detached {job.name} witness",
    )
    check(
        live in obs
        and (fit_boundary is None or any(x["units"] == fit_boundary for x in obs))
        and 0 < obs[0]["units"]
        and all(
            a["units"] < b["units"] and int(a["heartbeat"]) <= int(b["heartbeat"])
            for a, b in zip(obs, obs[1:])
        )
        and obs[-1]["units"]
        == z["units"]
        == charge["witnessed_units"]
        == start["units"]
        == n
        and pr == dict(pid=ident["trainer"], units=n, scored_targets=targets)
        and z["targets"] == targets,
        "incomplete actual FIT/PROBE progress",
    )
    check(
        0
        < charge["upper_card_s"]
        == int(end["released_epoch"]) - int(ini["created"]) + 1
        <= job.cap
        and int(end["rc"]) == charge["rc"] == 0
        and isclose(
            charge["inclusive_wall_s"],
            (int(end["end_monotonic_ns"]) - start["t0"]) * 1e-9,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        f"unverified {job.name} release/charge",
    )
    return dict(
        upper_card_s=charge["upper_card_s"],
        inclusive_wall_s=charge["inclusive_wall_s"],
        owner_and_waited_children_cpu_s=charge["owner_and_waited_children_cpu_s"],
        worker_cpu_core_s=z["cpu_core_s"],
        peak_device_allocated=z["gpu_peak_alloc"],
        peak_device_reserved=z["gpu_peak_reserved"],
        peak_RSS_KiB=z["process_RSS_peak_KiB"],
        stage_seconds=z["stage_seconds"],
        progress_observations=len(obs),
        witnessed_fit_boundary=fit_boundary,
        terminal_units=n,
        device=ident["device"],
        torch=ident["torch"],
        python=ident["python"],
        rc=0,
    )


def analyze(root, *, study="U"):
    from .fit_phase import study as phase

    data, p, _ = phase(study)
    check(
        root == p.ROOT and not (root / "stop.json").exists(), f"wrong {study} closure"
    )
    d = root / p.JOB.name
    z = load(d / "complete.json")
    charge = load(d / "charge.json")
    check(load(root / "charged.json") == {p.JOB.name: charge}, f"wrong {study} ledger")
    n = p.units()
    paid = stage_cost(
        d, z, charge, p.JOB, n, n * T, fit_boundary=len(SEEDS) * data.N["fit"]
    )
    initial = data.bound(d / "frozen.json", z["freeze_sha"])
    check(
        initial == p.plan(initial["ref"], hash_weights=False)
        and z["neural_updates"] == 0
        and z["fitted_readouts"] == len(p.DIMS) * len(SEEDS),
        "foreign endpoints or fit count",
    )
    readouts = data.bound(d / "fit-frozen.json", z["fit_freeze_sha"])
    check(readouts == p.readout_map(d, initial), "unbound final readouts")
    book = data.bound(data.BASE / "reservation.json", initial["data"]["rows_sha"])
    roles = data.checked_roles(book)
    probe = data.bound(d / "probe/identity.json", z["probe_identity_sha"])
    check(
        probe["role"] == "probe"
        and probe["fit_freeze_sha"] == z["fit_freeze_sha"]
        and probe["rows_sha"] == initial["data"]["rows_sha"]
        and probe["count"] == data.N["probe"]
        and probe["targets"] == T * data.N["probe"]
        and probe["source_bytes"] == sum(r["length"] for r in roles["probe"])
        and probe["spec"] == data.SPEC,
        f"foreign {study} probe stream",
    )
    hs = data.bound(d / "probe/shifts.json", probe["shifts_sha"])
    check(
        [(x["document_sha"], x["start"]) for x in hs]
        == [(r["sha256"], r["start"]) for r in roles["probe"]],
        f"unbound {study} probe shifts",
    )
    if study in ("DR", "RB", "EH", "PC"):
        if study == "PC":
            from .paired_correction_stats import summarize as reduce
        elif study == "EH":
            from .error_history_stats import summarize as reduce
        elif study == "RB":
            from .residual_stats import summarize as reduce
        else:
            from .decode_stats import summarize as reduce
        parent = {
            k: initial[k]
            for k in (
                "jt_result_sha",
                "ls_result_sha",
                "u_confirmation_sha",
                "writer_cost_sha",
                "writer_binary_sha",
                "u_beta",
                "j_beta",
                "u_refits",
                "j_refits",
            )
        }
        if study == "EH":
            parent.update(
                {
                    k: initial[k]
                    for k in (
                        "rb_result_sha",
                        "qualification_sha",
                        "fit_dimensions",
                        "swap_fit",
                    )
                }
            )
        if study == "PC":
            parent.update(
                {
                    k: initial[k]
                    for k in (
                        "et_result_sha",
                        "et_close_sha",
                        "eh_result_sha",
                        "eh_fit_freeze_sha",
                        "qualification_sha",
                        "fit_dimensions",
                        "swap_fit",
                        "error_refits",
                        "history_refits",
                    )
                }
            )
        if study == "RB":
            parent.update(
                {
                    k: initial[k]
                    for k in (
                        "dr_result_sha",
                        "dr_fit_seal",
                        "coarse_sha",
                        "qualified_axes",
                        "d_refits",
                    )
                }
            )
    elif study == "U":
        reduce = summarize
        parent = dict(q_result_sha=data.Q_RESULT)
    else:
        if study == "V":
            from .alternative_analysis import summarize as reduce
        else:
            from .literal_mark_analysis import summarize as reduce
        parent = dict(
            **({"v_result_sha": initial["v_result_sha"]} if study == "LS" else {}),
            u_confirmation_sha=initial["u_confirmation_sha"],
            u_beta=initial["u_beta"],
            u_refits=initial["u_refits"],
        )
    stats = {}
    for role, rows in roles.items():
        hashes = [r["sha256"] for r in rows]
        logs = {}
        for s in SEEDS:
            key = str(s)
            dig, logs[s] = journal(d / f"{role}-{s}.jsonl", hashes)
            if role == "probe":
                check(
                    z["probes"][key]
                    == dict(
                        seed=s,
                        checkpoint_sha=initial["endpoints"][key]["sha256"],
                        journal_sha=dig,
                        plan_sha=z["freeze_sha"],
                        fit_freeze_sha=z["fit_freeze_sha"],
                        windows=len(rows),
                    ),
                    f"foreign {study} probe journal",
                )
        stats[role] = reduce(role, logs, hashes)
    return dict(
        scope=f"X49-{study} disjoint FIT and PROBE within native TRAIN, not a final",
        ref=initial["ref"],
        spec=data.SPEC,
        **parent,
        checkpoints={s: e["sha256"] for s, e in initial["endpoints"].items()},
        data=initial["data"],
        freeze_sha=z["freeze_sha"],
        fit_freeze_sha=z["fit_freeze_sha"],
        probe_identity_sha=z["probe_identity_sha"],
        probe_extraction_wall_s=probe["extraction_wall_s"],
        probe_source_bytes=probe["source_bytes"],
        **(
            dict(decoders=readouts["seeds"])
            if study in ("DR", "RB")
            else dict(beta={s: r["beta"] for s, r in readouts["seeds"].items()})
        ),
        neural_updates=0,
        cost=paid,
        stages=stats,
    )
