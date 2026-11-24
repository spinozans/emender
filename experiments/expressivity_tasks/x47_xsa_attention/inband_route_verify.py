"""Verify T8 from numeric/work receipts alone (no feature, tensor, native index or tape required)."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from .frozen_packet_verify import valid_sha
from .frozen_route_verify import PURE, work
from .serving_verify import HERE, load, require

ROOT = HERE / "receipts/memory-program-t8"
ORDER = (
    "source-step",
    "value-step",
    "updated-full-live",
    "updated-full-positive",
    "updated-fixed-on-old",
    "updated-full-on-old",
)
SCORES = {"byte_vectors": 4096 * 256, "joint_scores": 4096, "nll": 1}


def authority(ctx, state):
    return hashlib.sha256(
        json.dumps(dict(ctx, selector=state), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def metric(x, source=False):
    if source:
        wall = x["times"]["wall_to_update_ns"]
        setup = x["times"]["setup_ns"]
        fwd = x["times"]["forward_ns"]
    else:
        wall = x["wall_ns"]
        setup = x["setup_ns"]
        fwd = x["forward_ns"]
        require(
            wall == sum(x[k] for k in ("forward_ns", "check_ns", "backward_ns", "update_ns")),
            "lost measured wall",
        )
    require(
        x["lifetime_ns"] >= setup + wall
        and 0 < fwd <= wall
        and x["cpu_core_s"] > 0
        and x["rss_kib"] > 0
        and x["dispatches"] == sum(v["batches"] for v in x["provider"].values()),
        "cost/dispatch evidence",
    )
    return dict(
        wall_s=wall / 1e9,
        forward_s=fwd / 1e9,
        lifecycle_s=x["lifetime_ns"] / 1e9,
        core_s=x["cpu_core_s"],
        rss_mib=x["rss_kib"] / 1024,
        child_rss_mib=x["child_rss_kib"] / 1024,
        lf_steps=x["archive"].get("lf_steps", 0),
        provider_s=sum(v["ns"] for v in x["provider"].values()) / 1e9,
        logical_keys=x["activity"]["usage"].get("logical_keys", 0),
        dispatches=x["dispatches"],
        instructions=sum(x["activity"]["opcodes"].values()),
        nll=x["nll"],
    )


def audit(root=ROOT):
    t7 = HERE / "receipts/memory-program-t7"
    old = load(t7 / "identity.json")
    hist = load(t7 / "full-update.json")
    prior = load(t7 / "verified.json")
    ref = (root / "gated-ref").read_text().strip()
    ix = load(root / "identity.json")
    ctx = ix["context"]
    init = ix["initial"]
    state = load(root / "complete.json")
    j = load(root / "progress.json")
    require(
        not (root / "stop.json").exists()
        and not prior["qualified"]
        and prior["remaining_s"] == 476
        and ix["ref"] == ref
        and ix["packet_file"] == old["prior_packet"]
        and ix["policy_file"]
        == load(HERE / "receipts/memory-program-t4/attempt1/identity.json")["checkpoint"]
        and ix["torch"] == old["torch"]
        and ctx == dict(old["context"], code=ref)
        and init == old["initial"]
        and ix["threads"] == 4
        and ix["strict"]
        and not ix["warn_only"]
        and not ix["cuda_available"]
        and ix["visible"] == ""
        and "110 passed" in (root / "qualification.txt").read_text().splitlines()[-1],
        "wrong training/CPU authority",
    )
    expected = {n: dict(targets=4096, updates=int(i < 2)) for i, n in enumerate(ORDER)}
    require(
        state["qualified"]
        and not state["cuda_initialized"]
        and state["card_s"] == 0
        and state["error"] is None
        and state["passes"] == j["passes"] == expected
        and list(state["passes"]) == list(ORDER)
        and state["scored"] == 24576
        and state["updates"] == 2
        and state["pid"] == j["pid"] == ix["pid"]
        and j["label"] == ORDER[-1]
        and 0 < state["elapsed_ns"] <= 480e9,
        "incomplete or over-time diagnostic",
    )
    runs = [load(root / (s + ".json")) for s in ORDER]
    s, fx, post, pos, fixed, ctr = runs
    c = s["checks"]
    ann = load(root / "source-vs-strict.json")
    dm = ann["differences"]
    ref_meta = load(HERE / "receipts/memory-program-t6/cpu/retained-serial/identity.json")["tensors"]
    require(
        c == hist["checks"] == load(HERE / "receipts/memory-program-t5/retained-strict-A/step.json")["checks"]
        and c["before"] == init != c["after"]
        and c["input_sha"] == ctx["features"]
        and work(s) == work(hist)
        and ann["authority"] == authority(ctx, init)
        and ann["all_t5_t7_checks_match"]
        and ann["same_work"]
        and s["mode"] == "fresh_sample"
        and not s["coupled"]
        and s["peak_allocated"] == s["peak_reserved"] == 0
        and s["targets"] == 4096
        and s["optimizer_steps"] == 1
        and dm.keys() == ref_meta.keys()
        and all(
            v["elements"] == math.prod(ref_meta[k]["shape"])
            and v["ref_sha"] == v["actual_sha"] == ref_meta[k]["sha"]
            and v["max_abs"] == v["unequal_bits"] == 0
            for k, v in dm.items()
        ),
        "source capture changed full step or work",
    )
    orig = dict(
        s,
        per_document_events=c["per_document_events"],
        dispatch_sha=c["dispatch"],
        score_sha={k: c[k] for k in SCORES},
        issuer_sha=ann["issuer_sha"],
        tape=ann["tape"],
        authority=ann["authority"],
    )
    for i, x in enumerate(runs):
        require(
            x["cell"] == ORDER[i]
            and x["targets"] == x["activity"]["opcodes"]["EMIT"] == 4096
            and x["optimizer_steps"] == int(i < 2),
            "wrong scored cell",
        )
        z = orig if i == 0 else x
        plan = z["tape"]
        require(
            valid_sha(z["issuer_sha"])
            and valid_sha(plan["tape_sha"])
            and 4096 <= plan["steps"] == sum(z["activity"]["opcodes"].values()) <= 24576
            and plan["digest_bytes"] == 32 * (plan["steps"] + plan["reads"])
            and 0 <= plan["reads"] <= z["activity"]["opcodes"].get("LOOKUP", 0)
            and len(z["per_document_events"]) == 32
            and all(valid_sha(h) for h in z["per_document_events"]),
            "incomplete trace",
        )
        if i == 0:
            continue
        oldpure = x["protected_before"]
        newpure = x["protected_after"]
        require(
            set(oldpure) == set(newpure) == PURE | {"writer.weight"}
            and all(oldpure[k] == newpure[k] and valid_sha(oldpure[k]) for k in PURE)
            and not PURE.intersection(x["used"])
            and (x["before"] != x["after"])
            == (oldpure["writer.weight"] != newpure["writer.weight"])
            == (i == 1)
            and (x["writer_grad"] > 0) == (i == 1)
            and (x["clip_norm"] > 0) == (i == 1),
            "unexpected parameter or credit path",
        )
        if i == 2:
            require(
                x["mode"] == "fresh_sample"
                and x["before"] == c["after"]
                and x["authority"] == authority(ctx, c["after"]),
                "not updated sampling",
            )
            continue
        against = post if i == 3 else orig
        gate = load(root / (ORDER[i] + "-gate.json"))
        require(
            x["mode"] == "checked_replay"
            and work(x) == work(against)
            and all(
                x[k] == against[k]
                for k in ("per_document_events", "dispatch_sha", "issuer_sha", "tape", "authority")
            )
            and gate["logical_match"] == dict(work=True, events=True, dispatch=True, requests=True)
            and gate["checked_before_step"]
            and gate["exact_scores"] == (i in (1, 3))
            and gate["scores"].keys() == SCORES.keys(),
            "wrong forced assignment",
        )
        for k, n in SCORES.items():
            v = gate["scores"][k]
            require(
                v["elements"] == n
                and 0 <= v["unequal_bits"] <= n
                and math.isfinite(v["max_abs"])
                and v["max_abs"] >= 0
                and v["ref_sha"] == against["score_sha"][k]
                and v["actual_sha"] == x["score_sha"][k]
                and (not v["unequal_bits"]) == (v["ref_sha"] == v["actual_sha"])
                and (v["unequal_bits"] or v["max_abs"] == 0)
                and (not gate["exact_scores"] or not v["unequal_bits"]),
                "incomplete score gate",
            )
    require(
        "writer.weight" in fx["used"]
        and fx["before"] == init
        and fixed["before"] == fx["after"]
        and fixed["protected_before"] == fx["protected_after"]
        and ctr["before"] == pos["before"] == post["before"] == c["after"]
        and pos["protected_before"] == ctr["protected_before"] == post["protected_before"],
        "post-policy lineage",
    )
    diff = load(root / "sensitivity.json")
    tot = Counter()
    docs = diff["per_document"]
    require(len(docs) == 32, "missing docs")
    for i, d in enumerate(docs):
        q = d["counts"]
        require(
            q["positions"] == 128
            and all(type(v) is int and 0 <= v <= 128 for v in q.values())
            and q["opcode"] <= q["instruction"] <= q["structure"] <= 128 - q["prefix_identical"]
            and q["request"] <= q["structure"]
            and q["full_live_vs_original_argmax"] <= q["full_live_vs_original_vector"]
            and q.get("prefix_tolerance_violations", 0) == 0
            and 0 <= d["prefix_max_fraction"] <= 1
            and d["prefix_max_abs"] >= 0
            and (q["structure"] or post["per_document_events"][i] == orig["per_document_events"][i]),
            "invalid same-history or route contrast",
        )
        tot.update(q)
    require(
        (tot["full_live_vs_original_vector"] > 0)
        == (ctr["score_sha"]["byte_vectors"] != post["score_sha"]["byte_vectors"])
        and (tot["request"] > 0) == (orig["issuer_sha"] != post["issuer_sha"])
        and (tot["fixed_route_value_vector"] > 0)
        == (orig["score_sha"]["byte_vectors"] != fixed["score_sha"]["byte_vectors"])
        and sum(x["lifetime_ns"] for x in runs) <= state["elapsed_ns"],
        "incomplete contrast",
    )
    metrics = {n: metric(x, i == 0) for i, (n, x) in enumerate(zip(ORDER, runs, strict=True))}
    return dict(
        ref=ref,
        qualified=True,
        scored_targets=24576,
        distinct_train_positions=4096,
        independent_steps=2,
        host_updates=0,
        card_s=0,
        cumulative_x47_upper=1324,
        remaining_s=476,
        diagnostic_wall_s=state["elapsed_ns"] / 1e9,
        source_tensors=len(dm),
        source_matches_t5_t7=True,
        exact_replay_score_elements=2 * sum(SCORES.values()),
        tape=ann["tape"],
        value_used=len(fx["used"]),
        writer_gradient=fx["writer_grad"],
        protected_selector_tensors=len(PURE),
        same_history_max_abs=max(d["prefix_max_abs"] for d in docs),
        position_contrast=dict(tot),
        metrics=metrics,
        historical_capture_plus_step_lifecycle_s=sum(x["lifecycle_s"] for x in prior["metrics"].values()),
        held_out=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
