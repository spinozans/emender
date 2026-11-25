"""Native short confirmation: receipt-only weighted document inference, decision and independent audit."""

from __future__ import annotations
import json
from ..x48_muon_continuation.data import check, load, sha
from . import surprise_analysis as u
from .panels import SEEDS, fingerprint
from .short_phase import study as phase


def decision(stats):
    cs = stats["contrasts"]
    passes = {
        k: cs[k]["gain"] > 0
        and cs[k]["ci95"][0] > 0
        and all(v > 0 for v in cs[k]["by_seed"])
        for k in ("U:N", "U:S", "U:B")
    }
    return dict(
        contrasts_pass=passes,
        recommend_local_frozen_U=all(passes.values()),
        scope="only the tested frozen models and their original fitted coefficients; no scale nomination",
    )


def analyze(root, *, study="UC"):
    data, p, _, analysis = phase(study)
    check(
        root == p.ROOT and not (root / "stop.json").exists(), "foreign or partial final"
    )
    d = root / p.JOB.name
    z = load(d / "complete.json")
    initial = data.u.bound(d / "frozen.json", z["freeze_sha"])
    check(
        initial == p.plan(initial["ref"], hash_weights=False)
        and z["new_fits"] == z["neural_updates"] == 0
        and set(z["endpoints"]) == set(map(str, SEEDS)),
        "different final predictor or additional fit",
    )
    book = data.u.bound(data.BASE / "reservation.json", initial["data"]["rows_sha"])
    rows, nt = data.checked_rows(book)
    hs = [r["sha256"] for r in rows]
    sizes = [r["length"] - 1 for r in rows]
    m = initial["data"]
    check((m["documents"], m["targets"]) == (len(rows), nt), "different reserved dose")
    c = load(d / "charge.json")
    check(load(root / "charged.json") == {p.JOB.name: c}, "foreign short final charge")
    paid = u.stage_cost(d, z, c, p.JOB, len(rows) * len(SEEDS), nt * len(SEEDS))
    shape = data.u.bound(d / "shape-check.json", z["shape_sha"])
    ident = load(d / "identity.json")
    check(
        shape
        == dict(
            plan_sha=z["freeze_sha"],
            trainer=ident["trainer"],
            trainer_start=ident["trainer_start"],
            seed=SEEDS[0],
            lengths=p.LENGTHS,
            natural_targets=0,
            synthetic_prefills=len(p.LENGTHS),
        ),
        "missing device gate",
    )
    stream = data.u.bound(d / "stream.json", z["stream_sha"])
    shifts = data.u.bound(d / "shifts.json", stream["shifts_sha"])
    check(
        [v["document_sha"] for v in shifts] == hs
        and [v["target_count"] for v in shifts] == sizes
        and all(len(v["inputs"]) == len(v["targets"]) == 64 for v in shifts)
        and stream["targets_per_endpoint"] == nt
        and stream["source_bytes"] == nt + len(rows)
        and stream["windows"] == len(rows)
        and stream["freeze_sha"] == z["freeze_sha"],
        "wrong whole-document shifts",
    )
    logs = {}
    for s in SEEDS:
        k = str(s)
        h, logs[s] = p.prior.journal(d / f"{s}.jsonl", hs, sizes=sizes)
        check(
            z["endpoints"][k]
            == dict(scores_sha=h, endpoint_id=fingerprint(initial["endpoints"][k])),
            "foreign scored endpoint",
        )
    stats = (
        u.summarize("final_test", logs, hs, short_targets=sizes)
        if study == "UC"
        else analysis.summarize(logs, hs, sizes)
    )
    bands = {}
    for lo, hi in [(1, 15), (16, 127), (128, 1023), (1024, 4095)]:
        run = [x for x in logs[SEEDS[0]] if lo <= x["targets"] <= hi]
        bands[f"{lo}-{hi}"] = dict(
            documents=len(run),
            real_targets=sum(x["targets"] for x in run),
            hits=sum(x["work"]["literal_hits"] for x in run),
            misses=sum(x["work"]["literal_misses"] for x in run),
        )
    check(
        sum(v["real_targets"] for v in bands.values()) == nt
        and all(v["hits"] + v["misses"] == v["real_targets"] for v in bands.values())
        and all(
            x["work"]["literal_hits"] > 0 or len(set(x["nll_sums"].values())) == 1
            for seq in logs.values()
            for x in seq
        ),
        "lost miss or length band",
    )
    parent = (
        dict(
            u_result_sha=data.RESULT,
            u_fit_freeze_sha=data.FIT_SHA,
            final_freeze_sha=z["freeze_sha"],
        )
        if study == "UC"
        else dict(
            **p.lineage(),
            ls_fit_freeze_sha=data.FIT_SHA,
            predictor_freeze_sha=z["freeze_sha"],
        )
    )
    return dict(
        scope="new whole short native final, frozen W-B predictors and U coefficients, no refit"
        if study == "UC"
        else "new whole short native calibration, locked B/U/H/J, no fitting and no native final access",
        ref=initial["ref"],
        spec=data.SPEC,
        **parent,
        shape_check_sha=z["shape_sha"],
        data=m,
        source_bytes=stream["source_bytes"],
        stream_sha=z["stream_sha"],
        seed_order=list(SEEDS),
        predictors={
            k: dict(checkpoint=e["weights"]["sha256"], beta=e["beta"])
            for k, e in initial["endpoints"].items()
        },
        new_fits=0,
        neural_updates=0,
        cost=paid,
        analysis=stats,
        by_length_per_seed=bands,
        decision=analysis.decision(stats),
    )


def report(*, study="UC"):
    data, p, run, analysis = phase(study)
    from .panel import cpu
    from .surprise_report import check_bootstrap

    cpu()
    root = p.ROOT
    d = root / p.JOB.name
    done = load(root / "done.json")
    check(
        done["complete"] and done["charges_sha"] == sha(root / "charged.json"),
        "terminal final ledger changed",
    )
    r = data.u.bound(root / "summary.json", done["summary_sha"])
    check(r == analysis.analyze(root), "confirmation analysis changed")
    p.validate(load(run.AUTH), r["ref"])
    check(
        data.u.bound(
            d / "frozen.json",
            r["final_freeze_sha" if study == "UC" else "predictor_freeze_sha"],
        )
        == p.plan(r["ref"], hash_weights=True),
        "final bytes/coefficients changed",
    )
    rows, nt = data.checked_rows(
        data.u.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    )
    check_bootstrap(
        d,
        r["analysis"],
        [x["sha256"] for x in rows],
        short_targets=[x["length"] - 1 for x in rows],
        study="U" if study == "UC" else "JT",
    )
    reserved = data.u.bound(data.BASE / "reserved.json", r["data"]["reserved_sha"])
    check(
        nt == r["analysis"]["real_targets_per_seed"]
        and reserved["payload_documents_opened"] == 0,
        "bad final dose",
    )
    return dict(
        **r,
        metadata_reservation=reserved,
        authority_sha=sha(run.AUTH),
        bootstrap=dict(
            joint_document_draws=10000,
            seed=4908 if study == "UC" else analysis.BOOT,
            interval="linear percentile, actual drawn target totals",
        ),
        verified=dict(
            terminal_sha=sha(root / "done.json"),
            summary_sha=done["summary_sha"],
            charges_sha=done["charges_sha"],
            independent_paired_bootstrap=True,
            actual_weight_bytes_rehashed=3,
            driver_and_short_parents_cpu_s=done["driver_and_short_parents_cpu_s"],
        ),
    )


if __name__ == "__main__":
    print(json.dumps(report(), sort_keys=True, indent=2, allow_nan=False))
