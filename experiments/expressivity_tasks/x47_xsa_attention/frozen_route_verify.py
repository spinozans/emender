"""T7 offline: distinguish an attested fixed-route intervention from an ordinary frozen module."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .serving_verify import HERE, load, require
from .frozen_packet_verify import valid_sha

ROOT = HERE / "receipts/memory-program-t7"
LABELS = (
    "capture",
    "full-update",
    "fixed-update",
    "updated-full-live",
    "updated-fixed",
    "updated-full-on-old",
)
PURE = {"op.weight", "prior", "transitions.weight", "pointer.weight", "pointer.bias"}


def work(cell):
    # Only omit clocks, not counts, refusals, exact physical LF steps or duplicate work.
    def costs(d):
        return {k: v for k, v in d.items() if k != "ns" and not k.endswith("_ns")}

    return (
        cell["activity"],
        cell["dispatches"],
        costs(cell["archive"]),
        {k: costs(v) for k, v in cell["provider"].items()},
    )


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-t6/verified.json")
    old = load(HERE / "receipts/memory-program-t5/retained-strict-A/step.json")
    packet = load(HERE / "receipts/memory-program-t4/attempt1/retained-packet.json")
    src = load(HERE / "receipts/memory-program-t4/attempt1/identity.json")
    ref = (root / "gated-ref").read_text().strip()
    ix = load(root / "identity.json")
    stopped = not (root / "complete.json").is_file()
    labels = LABELS[:2] if stopped else LABELS
    done = None if stopped else load(root / "complete.json")
    prog = load(root / "progress.json")
    ctx = ix["context"]
    auth = hashlib.sha256(
        json.dumps(dict(ctx, selector=ix["initial"]), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        ctx
        == dict(
            encoder=packet["binding"]["encoder_sha"],
            source=packet["binding"]["source_sha"],
            features=packet["payload_sha"],
            mount=src["mount"],
            scope="whole",
            live_exe=src["position_exe_sha"],
            branch_exe=src["branch_exe_sha"],
            limits=src["limits"],
            seed=7809,
            code=ref,
        )
        and ix["authority"] == auth
        and ix["ref"] == ref
        and ix["torch"] == src["torch"]
        and ix["max_steps"] == 24576
        and ix["threads"] == 4
        and ix["strict"]
        and ix["visible"] == ""
        and not ix["cuda_available"]
        and ix["prior_packet"] == packet["sha256"]
        and ix["initial"] == old["checks"]["before"]
        and "108 passed" in (root / "qualification.txt").read_text().splitlines()[-1],
        "unbound CPU/feature/physical source",
    )
    require(
        (stopped or (not done["cuda_initialized"] and done["card_s"] == 0))
        and prev["cumulative_x47_upper"] == 1324
        and prev["remaining_s"] == 476
        and prog["pid"] == ix["pid"]
        and prog["pass_name"] == labels[-1]
        and prog["targets"] == 4096,
        "wrong device/work",
    )
    cells = [load(root / (n + ".json")) for n in labels]
    cap, live = cells[:2]
    c = live["checks"]
    orig = ix["initial"]
    metrics = {}
    require(
        live["cell"] == LABELS[1]
        and live["mode"] == "fresh_sample"
        and not live["coupled"]
        and live["peak_allocated"] == live["peak_reserved"] == 0
        and c == old["checks"]
        and live["nll"] == old["nll"]
        and c["input_sha"] == packet["payload_sha"]
        and c["before"] == orig != c["after"]
        and work(live) == work(old),
        "not the same strict initial on-policy step",
    )
    for name, s in zip(labels, cells, strict=True):
        require(
            s["targets"] == 4096 and s["optimizer_steps"] == int(name in ("full-update", "fixed-update")),
            "wrong updates/exposure",
        )
        if s is live:
            wall = s["times"]["wall_to_update_ns"]
            setup = s["times"]["setup_ns"]
        else:
            plan = s["tape"]
            acts = s["activity"]["opcodes"]
            require(
                s["pass_name"] == name
                and s["mode"]
                == (
                    "frozen-selector sample"
                    if name in ("capture", "updated-full-live")
                    else "checked forced choices"
                )
                and valid_sha(s["vectors"])
                and valid_sha(s["joint"])
                and valid_sha(plan["tape_sha"])
                and 4096 <= plan["steps"] == sum(acts.values()) <= ix["max_steps"]
                and acts["EMIT"] == 4096
                and plan["digest_bytes"] == 32 * (plan["steps"] + plan["reads"])
                and 0 < plan["reads"] <= acts["LOOKUP"]
                and len(s["per_document_events"]) == 32
                and all(valid_sha(v) for v in s["per_document_events"])
                and set(s["pure_before_after"]) == PURE
                and all(valid_sha(v) for v in s["pure_before_after"].values()),
                "incomplete verified trace",
            )
            require(
                (s["before"] != s["after"]) == (name == "fixed-update")
                and (s["writer_before"] != s["writer_after"]) == (name == "fixed-update")
                and (s["pathwise_writer_grad"] > 0) == (name == "fixed-update")
                and not PURE.intersection(s["used"]),
                "unexpected value or route adjoint",
            )
            wall = s["wall_ns"]
            setup = s["setup_ns"]
            require(
                wall == s["forward_ns"] + s["backward_checks_ns"] + s["update_ns"]
                and all(s[k] >= 0 for k in ("forward_ns", "backward_checks_ns", "update_ns")),
                "lost cost",
            )
        require(
            s["lifetime_ns"] >= setup + wall > 0
            and s["cpu_core_s"] > 0
            and s["dispatches"] == sum(v["batches"] for v in s["provider"].values())
            and s["rss_kib"] > 0,
            "bad costs",
        )
        metrics[name] = dict(
            wall_s=wall / 1e9,
            lifecycle_s=s["lifetime_ns"] / 1e9,
            cpu_core_s=s["cpu_core_s"],
            rss_mib=s["rss_kib"] / 1024,
            child_rss_mib=s["child_rss_kib"] / 1024,
            provider_s=sum(x["ns"] for x in s["provider"].values()) / 1e9,
            lf_steps=s["archive"]["lf_steps"],
            logical_keys=s["activity"]["usage"]["logical_keys"],
            instructions=sum(s["activity"]["opcodes"].values()),
            dispatches=s["dispatches"],
            nll=s["nll"],
        )
    if stopped:
        exit_info = load(root / "exit.json")
        require(
            prog["phase"] == "updated"
            and prog["targets"] == 4096
            and exit_info["rc"] == 1
            and exit_info["new_card_leases"] == 0
            and exit_info["error"]
            == (root / "run.log").read_text().strip().splitlines()[-1]
            == "ValueError: frozen selector did not reproduce the initial live computation"
            and all(not (root / (n + ".json")).exists() for n in LABELS[2:])
            and cap["before"] == orig
            and cap["vectors"] != c["byte_vectors"]
            and cap["joint"] == c["joint_scores"]
            and cap["nll_sha"] == c["nll"]
            and cap["nll"] == live["nll"]
            and cap["per_document_events"] == c["per_document_events"]
            and cap["dispatch_sha"] == c["dispatch"]
            and work(cap) == work(live),
            "not the recorded pre-replay byte-vector failure",
        )
        return dict(
            ref=ref,
            qualified=False,
            disposition=exit_info["error"],
            distinct_train_positions=4096,
            scored_targets=sum(s["targets"] for s in cells),
            independent_updates=1,
            fixed_route_updates=0,
            host_updates=0,
            forced_passes=0,
            reported_elapsed_s=exit_info["reported_elapsed_s"],
            card_s=0,
            cumulative_x47_upper=prev["cumulative_x47_upper"],
            remaining_s=prev["remaining_s"],
            same_per_document_events=32,
            same_dispatch=True,
            same_joint_scores=True,
            same_scalar_nll=True,
            same_byte_vectors=False,
            same_as_t5_strict_update=True,
            vector_difference_elements=None,
            vector_difference_max_abs=None,
            metrics=metrics,
            capture=cap["tape"],
            quality_tested=False,
        )
    fx, post, fxpost, ctr = cells[2:]
    require(
        fx["before"] == cap["before"] == orig
        and fx["vectors"] == cap["vectors"] == c["byte_vectors"]
        and cap["joint"] == fx["joint"] == c["joint_scores"]
        and fx["nll_sha"] == cap["nll_sha"] == c["nll"]
        and fx["nll"] == cap["nll"] == live["nll"]
        and fx["writer_before"] == cap["writer_before"]
        and fx["pure_before_after"] == cap["pure_before_after"] == fxpost["pure_before_after"]
        and "writer.weight" in fx["used"]
        and fx["writer_after"] == fxpost["writer_before"]
        and fxpost["before"] == fx["after"]
        and ctr["before"] == post["before"] == c["after"]
        and post["pure_before_after"] == ctr["pure_before_after"],
        "unmatched initial or post-intervention policy",
    )
    for s in (cap, fx, fxpost, ctr):
        require(
            s["per_document_events"] == c["per_document_events"]
            and s["issuer_sha"] == cap["issuer_sha"]
            and s["dispatch_sha"] == c["dispatch"]
            and s["tape"] == cap["tape"]
            and work(s) == work(live),
            "frozen assignment changed",
        )
    total = Counter()
    docs = done["per_doc"]
    require(len(docs) == 32, "missing docs")
    for i, d in enumerate(docs):
        require(
            d["positions"] == 128
            and all(type(v) is int and 0 <= v <= 128 for v in d.values())
            and d["opcode_sequence_changed"]
            <= d["instructions_or_operands_changed"]
            <= d["structural_evidence_changed"]
            and d["request_sequence_changed"] <= d["structural_evidence_changed"]
            and d["full_live_vs_old_route_argmax_changed"] <= d["full_live_vs_old_route_distribution_changed"]
            and (
                d["structural_evidence_changed"] > 0
                or post["per_document_events"][i] == cap["per_document_events"][i]
            ),
            "invalid position contrast",
        )
        total.update(d)
    require(
        (total["full_live_vs_old_route_distribution_changed"] > 0) == (ctr["vectors"] != post["vectors"])
        and (total["fixed_route_value_distribution_changed"] > 0) == (fxpost["vectors"] != cap["vectors"])
        and (total["request_sequence_changed"] > 0) == (cap["issuer_sha"] != post["issuer_sha"])
        and total["positions"] == done["distinct_target_positions"] == 4096
        and done["target_computations"] == 4096 * 6
        and done["optimizer_steps"] == 2
        and sum(s["lifetime_ns"] for s in cells) <= done["wall_ns"] <= 480e9,
        "wrong contrast or charging",
    )
    return dict(
        ref=ref,
        qualified=True,
        distinct_train_positions=4096,
        scored_targets=24576,
        independent_updates=2,
        diagnostic_wall_s=done["wall_ns"] / 1e9,
        card_s=0,
        cumulative_x47_upper=1324,
        remaining_s=476,
        metrics=metrics,
        capture=cap["tape"],
        fixed_pathwise_grad_norm=fx["clip_norm"],
        fixed_pathwise_writer_norm=fx["pathwise_writer_grad"],
        fixed_used_tensors=len(fx["used"]),
        post_update_route_and_value_changes=dict(total),
        quality_tested=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2))
