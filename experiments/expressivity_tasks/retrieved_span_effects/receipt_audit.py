"""Read-only panel closure and price report. No archive, model, text or new evaluation.

The registered scorer is unmodified. Recompute it from sealed loss/count receipts;
print secondary comparisons from the time checkpoints separately from the primary.
"""

import json
from pathlib import Path
import sys

import numpy as np

from .natural_data import cpu, public_gate, sha, TOKEN_SHA, SIZES
from .natural_score import aggregate, cell, cases, BANDS
from .natural_model import HOSTS, ARMS, SEEDS
from .window import BASE, HERE, get, need, fields


def packed_result(x):
    return [round(float(y), 10) for y in (x["gain"], *x["ci"], *x["seed_gains"])] + [
        bool(x["positive"])
    ]


def time_contrast(a, b, n, mult):
    """a,b: loss sums [paired seed,document], n: target counts. Lower NLL wins."""
    need(
        a.shape == b.shape
        and a.ndim == 2
        and a.shape[1] == len(n) == mult.shape[1]
        and np.all(n > 0),
        "unpaired all-target time contrast",
    )
    delta = b - a
    seeds = delta.sum(1) / n.sum()
    avg = delta.mean(0)
    co = mult @ avg / (mult @ n)
    lo, hi = np.quantile(co, (0.025, 0.975))
    return dict(
        gain=float(avg.sum() / n.sum()),
        ci=[float(lo), float(hi)],
        seed_gains=seeds.tolist(),
        positive=bool(lo > 0 and (seeds > 0).all()),
    )


def primary(contrasts):
    base = "gdn2/g-effect/full.intact:"
    keys = [
        base + n
        for n in (
            "g-state/full.intact",
            "g-neutral/full.intact",
            "g-effect/full.wrong_effect",
        )
    ]
    return dict(
        requires=keys, established=all(contrasts[k]["all"]["positive"] for k in keys)
    )


def audit():
    cpu()
    public_gate()
    d = BASE / "panel"
    data = BASE / "natural-data"
    campaign = BASE / "campaign"
    done = get(d / "complete.json")
    plan = get(d / "plan.json")
    frozen = get(d / "frozen.json")
    ack = get(d / "freeze-ack.json")
    scope = get(BASE / "window.json")
    need(get(campaign / "exit.json")["rc"] == 0, "no successful terminal campaign")
    seal = get(campaign / "build-preprobe-seal.json")
    meter = get(HERE / "receipts/token_step_meter.json")
    need(
        sha(campaign / "build-preprobe-seal.json") == meter["build_witness_sha256"]
        and not seal["probe_open"]
        and seal["build_meta_sha"] == sha(data / "build.json")
        and seal["image_sha"] == get(data / "build.json")["image_sha"]
        and seal["plan_sha"] == frozen["plan_sha"] == sha(d / "plan.json")
        and seal["fit_routes_sha"]
        == frozen["fit_routes_sha"]
        == sha(d / "fit-routes.json"),
        "BUILD/plan changed after preprobe seal",
    )
    need(
        ack["freeze_sha"] == done["freeze_sha"] == sha(d / "frozen.json")
        and int((d / "watch-ack").read_text()) == done["units"]
        and (d / "freeze-ack.json").stat().st_mtime_ns
        <= (data / "probe-open-attempted").stat().st_mtime_ns,
        "PROBE freeze/terminal witness mismatch",
    )
    roles = {r: get(data / (r + ".json")) for r in SIZES}
    reserved = get(data / "reserved.json")
    need(
        all(
            x["tokenizer_sha"] == TOKEN_SHA
            and x["rows_sha"]
            == plan["rows_sha"]
            == done["rows_sha"]
            == reserved["reservation_sha"]
            and x["docs"] == SIZES[r]
            for r, x in roles.items()
        )
        and reserved["sizes"] == SIZES,
        "role changed",
    )
    routes = {r: get(d / (r + "-routes.json")) for r in ("fit", "probe")}
    need(
        all(
            routes[r]["targets"] == roles[r]["targets"]
            and routes[r]["windows"] == roles[r]["windows"]
            for r in routes
        ),
        "route/target mismatch",
    )
    charges = {}
    for job in ("meter", "natural-meter", "panel"):
        p = BASE / job
        c = get(p / "charge.json")
        i = fields(p / "initial")
        t = fields(p / "last")
        release = fields(p / "released")
        adm = get(p / "admission.json")
        need(
            i["gpu"] in {"0", "1"}
            and all(i[k] == t[k] for k in ("pid", "gpu", "starttime"))
            and int(t["heartbeat"]) > int(i["heartbeat"])
            and c["rc"] == int(release["rc"]) == 0
            and c["release_verified"]
            and c["renewed_progress"]
            and 0
            < c["upper_card_s"]
            == int(release["released_epoch"]) - int(i["created"]) + 1
            <= adm["cap"]
            and int(release["released_epoch"]) <= scope["deadline_epoch"],
            "failed charge",
        )
        charges[job] = c["upper_card_s"]
    need(
        sum(charges.values()) <= scope["aggregate_card_seconds_upper"],
        "window exceeded",
    )
    names = {cell(h, a, s) for h in HOSTS for a in ARMS for s in SEEDS}
    need(
        names == set(plan["cells"]) == set(done["scores"]) == set(frozen["fits"])
        and len(names) == ack["fits"] == 48,
        "wrong study size",
    )
    doses = {}
    ns = None
    times = {}
    latency = {}
    donor = None
    for h in HOSTS:
        for a in ARMS:
            key = h + "/" + a
            ts = []
            runtime = []
            for s in SEEDS:
                name = cell(h, a, s)
                fp = d / "fits" / name / "fit.json"
                f = get(fp)
                need(
                    frozen["fits"][name] == sha(fp)
                    and f["cell"] == name
                    and set(f["weights"]) == {"time", "full"},
                    "unfrozen fit",
                )
                pts = f["points"]
                full = pts["full"]
                t = pts["time"]
                sig = (full["targets"], f["order_sha"])
                need(
                    doses.setdefault(s, sig) == sig
                    and 0 < t["step"] <= full["step"] == plan["steps"]
                    and 0 < t["targets"] <= full["targets"]
                    and 0 < t["train_s"] <= full["train_s"]
                    and t["band_reached"] == (t["train_s"] >= plan["matched_train_s"]),
                    "unpaired dose",
                )
                sc = d / "scores" / (name + ".npz")
                need(done["scores"][name] == sha(sc), "score changed")
                clocks = get(d / "scores" / (name + ".json"))
                need(set(clocks) == set(cases(a)), "missing cost information")
                for v in clocks.values():
                    if donor is None:
                        donor = v["donor_eligible_targets"]
                    need(
                        v["donor_eligible_targets"] == donor and v["seconds"] > 0,
                        "different donor coverage/invalid scoring clock",
                    )
                with np.load(sc, allow_pickle=False) as z:
                    if ns is None:
                        ns = z["counts"][:, 0].copy()
                    need(
                        np.array_equal(z["counts"][:, 0], ns), "different probe targets"
                    )
                    ts.append(z["time.intact"][:, 0].copy())
                runtime.append(
                    dict(
                        full_train_s=full["train_s"],
                        full_s_per_million=full["train_s"] * 1e6 / full["targets"],
                        full_wall_s=f["wall_s"],
                        score_s_per_million=clocks["full.intact"]["seconds"]
                        * 1e6
                        / done["targets"],
                        time_step=t["step"],
                        time_targets=t["targets"],
                        time_train_s=t["train_s"],
                        time_reached=t["band_reached"],
                    )
                )
            times[h, a] = np.stack(ts)
            latency[key] = runtime
    # Original paired inference, independently regenerated without reading text or parameters.
    actual = aggregate(d)
    old = get(campaign / "aggregate.json")
    need(
        all(
            json.loads(json.dumps(v, allow_nan=False)) == old[k]
            for k, v in actual.items()
        )
        and actual["targets"]["all"] == done["targets"] == roles["probe"]["targets"]
        and old["upper_card_seconds"] == sum(charges.values())
        and old["complete_sha"] == sha(d / "complete.json"),
        "receipt-only rescore failed",
    )
    n = len(ns)
    mult = (
        np.random.default_rng(5410)
        .multinomial(n, np.full(n, 1 / n), size=10000)
        .astype(np.float64)
    )
    extra = {}
    for h in HOSTS:
        for a, b in [
            ("g-effect", "g-state"),
            ("g-effect", "g-neutral"),
            ("e-effect", "e-state"),
            ("e-effect", "e-neutral"),
            ("mix-effect", "g-effect"),
            ("mix-effect", "e-effect"),
        ]:
            extra[f"{h}/{a}:{b}"] = packed_result(
                time_contrast(times[h, a], times[h, b], ns, mult)
            )
    return dict(
        schema="span-joint-token-panel-result-v1",
        scope=actual["scope"],
        qualified=seal["qualified"],
        complete_sha=sha(d / "complete.json"),
        frozen_sha=ack["freeze_sha"],
        plan_sha=frozen["plan_sha"],
        reservation_sha=plan["rows_sha"],
        build_witness_sha256=meter["build_witness_sha256"],
        charges_upper_card_s=charges,
        checkpoint_threshold_s=plan["matched_train_s"],
        steps=plan["steps"],
        batch=plan["batch"],
        cells=len(names),
        seeds=SEEDS,
        fit_target_exposures_per_seed=[doses[s][0] for s in SEEDS],
        archive={
            k: roles["build"][k]
            for k in ("docs", "pages", "codes", "postings", "tokens", "wall_s")
        },
        windows={
            r: {
                k: roles[r][k]
                for k in ("docs", "windows", "targets", "tokens", "wall_s")
            }
            for r in routes
        },
        route_preparation={
            r: {
                k: routes[r][k] for k in ("wall_s", "targets", "windows", "hit_windows")
            }
            for r in routes
        },
        bands=BANDS,
        probe_target_counts=actual["targets"],
        equal_length_four_hit_targets=donor,
        nll=actual["absolute"],
        primary=primary(actual["contrasts"]),
        contrast_layout=[
            "gain_nats_per_token",
            "doc_ci_low",
            "doc_ci_high",
            "seed_5411",
            "seed_5412",
            "seed_5413",
            "criteria_positive",
        ],
        contrasts={
            k: {b: packed_result(x) for b, x in v.items()}
            for k, v in actual["contrasts"].items()
        },
        exploratory_matched_clock_all=extra,
        per_seed_prices=latency,
        bootstrap="10000 paired whole-document resamples, seed 5410; ratio of summed loss difference to real target count",
    )


if __name__ == "__main__":
    obj = audit()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(obj, f, sort_keys=True, allow_nan=False, indent=2)
        f.write("\n")
