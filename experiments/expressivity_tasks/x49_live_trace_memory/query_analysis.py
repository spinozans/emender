"""X49-Q all-byte, paired, frozen diagnostic inference. Closed receipts only; no text/model input."""

from __future__ import annotations
from collections import Counter
import json
from math import isclose, isfinite
import numpy as np
from ..x47_xsa_attention.w3_launch import fields
from ..x48_muon_continuation.data import check, load, sha
from .panels import SEEDS, T
from .query_score import POLICIES, STRATA, cycle
from . import query_data as data

PAIRS = (("E", "J"), ("E", "M"), ("M", "J"))


def aggregate(logs, hashes):
    n = len(hashes)
    ns = len(SEEDS)
    check(
        n > 0 and len(set(hashes)) == n and set(logs) == set(SEEDS),
        "missing paired Q population",
    )
    sums = np.zeros((ns, n, 3))
    part = np.zeros((ns, n, 3, 4))
    counts = np.zeros((n, 4), dtype=np.int64)
    geometry = {k: Counter() for k in STRATA}
    work = {a: Counter() for a in POLICIES}
    seconds = {a: Counter() for a in POLICIES}
    prefill = 0.0
    order_counts = {a: [0] * 3 for a in POLICIES}
    for si, seed in enumerate(SEEDS):
        for i, (h, x) in enumerate(zip(hashes, logs[seed], strict=True)):
            check(
                x["document_sha"] == h
                and x["order"] == cycle(seed, i)
                and x["host_prefills"] == 1
                and set(x["cells"]) == set(POLICIES)
                and set(x["strata"]) == set(STRATA)
                and x["targets"] == T
                and isfinite(x["host_prefill_s"])
                and x["host_prefill_s"] >= 0,
                "foreign Q document/order/score",
            )
            prefill += x["host_prefill_s"]
            ref = logs[SEEDS[0]][i]
            check(
                x["strata"] == ref["strata"], "different logical routes at fixed bytes"
            )
            nc = [x["strata"][k]["targets"] for k in STRATA]
            check(
                all(type(v) is int and v >= 0 for v in nc) and sum(nc) == T,
                "lost miss or real target",
            )
            counts[i] = nc
            if not si:
                for k in STRATA:
                    for field, val in x["strata"][k]["geometry"].items():
                        if field == "max_abs_control_age_gap":
                            geometry[k][field] = max(geometry[k][field], val)
                        else:
                            geometry[k][field] += val
            for pi, a in enumerate(POLICIES):
                order_counts[a][x["order"].index(a)] += 1
                c = x["cells"][a]
                w = c["work"]
                check(
                    w == ref["cells"][a]["work"]
                    and w["literal_value_reads"] == w["literal_hits"] == T - nc[0]
                    and w["literal_queries"] == w["literal_value_writes"] == T
                    and w["literal_misses"] == nc[0]
                    and all(w[f] == v for f, v in x["cells"]["J"]["work"].items()),
                    "bad literal work",
                )
                check(
                    a == "J"
                    or (
                        w["extended_moves"] == nc[2] + nc[3]
                        and w["matched_controls"] == (nc[3] if a == "M" else 0)
                    ),
                    "bad move counts",
                )
                if not si:
                    work[a].update(w)
                check(
                    isfinite(c["policy_s"])
                    and 0 <= c["extra_address_s"] <= c["policy_s"]
                    and (a != "J" or c["extra_address_s"] == 0),
                    "incorrect policy clock",
                )
                seconds[a].update(
                    policy_s=c["policy_s"], extra_address_s=c["extra_address_s"]
                )
                v = c["nll_sum"]
                parts = [c["stratum_nll_sums"][k] for k in STRATA]
                check(
                    all(
                        isfinite(y) and y >= 0 and (nt or y == 0)
                        for nt, y in zip(nc, parts, strict=True)
                    )
                    and isclose(v, sum(parts), rel_tol=0, abs_tol=1e-8),
                    "invalid Q cell denominator",
                )
                sums[si, i, pi] = v
                part[si, i, pi] = parts
    check(
        all(freq == [n, n, n] for freq in order_counts.values()),
        "unbalanced policy order",
    )
    nt = n * T
    draw = np.random.default_rng(4906).integers(0, n, size=(10000, n))
    den = np.full(n, T, dtype=np.int64)[draw].sum(1)
    out = {}
    sizes = counts.sum(0)
    support = (counts > 0).sum(0)
    sections = {
        k: dict(
            real_targets_per_seed=int(sizes[g]),
            documents_with_support=int(support[g]),
            geometry_per_seed=dict(geometry[k]),
        )
        for g, k in enumerate(STRATA)
    }
    policies = {}
    for i, a in enumerate(POLICIES):
        v = sums[:, :, i]
        policies[a] = dict(
            nats_per_byte=float(v.sum() / (ns * nt)),
            by_seed=(v.sum(1) / nt).tolist(),
            work_per_seed=dict(work[a]),
            mean_head_and_address_s_per_doc=seconds[a]["policy_s"] / (n * ns),
            mean_extra_address_s_per_doc=seconds[a]["extra_address_s"] / (n * ns),
            by_stratum={
                k: dict(
                    nll_per_target=float(part[:, :, i, g].sum() / (ns * sizes[g]))
                    if support[g] >= 2
                    else None,
                    contribution_to_all_byte_nll=float(
                        part[:, :, i, g].sum() / (ns * nt)
                    )
                    if support[g] != 1
                    else None,
                )
                for g, k in enumerate(STRATA)
            },
        )
    for a, b in PAIRS:
        ia = POLICIES.index(a)
        ib = POLICIES.index(b)
        delta = sums[:, :, ib] - sums[:, :, ia]
        dpart = part[:, :, ib] - part[:, :, ia]
        unchanged = 3 if b == "M" else 2
        check(
            not dpart[:, :, :unchanged].any(),
            "nonzero effect without a distinct value address",
        )
        check(
            np.allclose(delta, dpart.sum(-1), rtol=0, atol=2e-8), "missing contribution"
        )
        ci = np.quantile(
            delta.mean(0)[draw].sum(1) / den, [0.025, 0.975], method="linear"
        )
        out[f"{a}:{b}"] = dict(
            gain=float(delta.sum() / (ns * nt)),
            by_seed=(delta.sum(1) / nt).tolist(),
            ci95=ci.tolist(),
            strata={
                k: dict(
                    conditional_gain=float(dpart[:, :, g].sum() / (ns * sizes[g]))
                    if support[g] >= 2
                    else None,
                    contribution_to_all_byte_gain=float(
                        dpart[:, :, g].sum() / (ns * nt)
                    )
                    if support[g] != 1
                    else None,
                )
                for g, k in enumerate(STRATA)
            },
        )
    return dict(
        contrasts=out,
        policies=policies,
        strata=sections,
        paired_documents=n,
        real_targets_per_seed_and_cell=nt,
        documents_with_moves=int((counts[:, 2:].sum(1) > 0).sum()),
        mean_shared_prefill_s_per_doc=prefill / (n * ns),
        policy_order_counts=order_counts,
    )


def analyze(root):
    from . import query_panel as p

    d = root / p.JOB.name
    z = load(d / "complete.json")
    c = load(d / "charge.json")
    f = load(d / "frozen.json")
    start = load(d / "start.json")
    stream = load(d / "stream.json")
    first, after, end = (fields(d / k) for k in ("initial", "after", "released"))
    check(
        c["job"] == p.JOB.name
        and z["kind"] == p.JOB.kind
        and int(end["rc"]) == c["rc"] == 0
        and 0
        < c["upper_card_s"]
        == int(end["released_epoch"]) - int(first["created"]) + 1
        <= p.TOTAL
        and first["adopted"] == "1"
        and int(after["heartbeat"]) > int(first["heartbeat"])
        and all(
            first[k] == after[k]
            for k in ("pid", "starttime", "gpu", "created", "adopted")
        )
        and (d / "watch-ack").is_file(),
        "unwitnessed Q charge/release",
    )
    check(
        stream["windows"] == data.N
        and stream["targets_per_endpoint"] == data.N * T
        and all(stream[k] == v for k, v in f["data"].items())
        and start["units"] == f["units"]
        and start["freeze_sha"] == z["freeze_sha"],
        "unbound Q stream or freeze clock",
    )
    check(
        root == p.ROOT
        and c["rc"] == 0
        and c["renewed"]
        and c["complete_sha"] == sha(d / "complete.json")
        and load(root / "charged.json") == {p.JOB.name: c}
        and c["upper_card_s"] <= p.TOTAL
        and z["units"] == c["witnessed_units"] == f["units"] == 3 * data.N
        and z["targets"] == 3 * data.N * T
        and z["updates"] == 0
        and z["freeze_sha"] == sha(d / "frozen.json")
        and z["stream_sha"] == sha(d / "stream.json")
        and f == p.plan(f["ref"], hash_weights=False),
        "foreign/incomplete Q diagnostic",
    )
    check(
        set(z["endpoints"]) == set(f["endpoints"]) == {str(s) for s in SEEDS},
        "wrong endpoints",
    )
    hashes = [x["sha256"] for x in load(data.BASE / "reservation.json")["rows"]]
    logs = {}
    for s in SEEDS:
        e = z["endpoints"][str(s)]
        v = d / f"{s}.jsonl"
        check(
            e["scores_sha"] == sha(v)
            and e["windows"] == data.N
            and e["checkpoint_sha"] == f["endpoints"][str(s)]["sha256"],
            "wrong Q scored endpoint",
        )
        logs[s] = list(map(json.loads, v.read_bytes().splitlines()))
    return dict(
        scope="X49-Q frozen native TRAIN diagnostic, not a native final; no optimizer updates",
        ref=f["ref"],
        spec=data.SPEC,
        w_result_sha=data.W_RESULT,
        checkpoints={s: r["sha256"] for s, r in f["endpoints"].items()},
        data=f["data"],
        freeze_sha=z["freeze_sha"],
        stream=stream,
        analysis=aggregate(logs, hashes),
        charge=c,
        cpu_core_s=z["cpu_core_s"],
        max_gpu_alloc=z["gpu_peak_alloc"],
        max_gpu_reserved=z["gpu_peak_reserved"],
        process_RSS_peak_KiB=z["process_RSS_peak_KiB"],
    )
