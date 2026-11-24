"""Verify T5 mode, input, progression, all-tensor bit comparisons and separate cost receipts."""

import argparse, json, math
from pathlib import Path
from .serving_verify import HERE, load, require
from .frozen_packet_verify import valid_sha

ROOT = HERE / "receipts/memory-program-t5"
ORDER = [
    "retained-ordinary-A",
    "retained-strict-A",
    "replicated-strict-A",
    "retained-strict-B",
    "replicated-strict-B",
    "retained-ordinary-B",
]
PAIRS = [(1, 3), (2, 4), (0, 5), (0, 1)]
FWD = (
    "input_sha",
    "byte_vectors",
    "joint_scores",
    "events",
    "per_document_events",
    "dispatch",
    "nll",
    "objective",
    "before",
)


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-t4/verified.json")
    require(not prev["qualified"] and prev["cumulative_x47_upper"] == 1284, "T4 disposition")
    d = load(root / "driver.json")
    done = load(root / "complete.json")
    ref = (root / "gated-ref").read_text().strip()
    require(
        d["ref"] == ref
        and d["order"] == ORDER
        and d["card_s"] == done["card_s"] == 0
        and done["wall_ns"] <= 720e9
        and done["targets"] == 24576
        and done["updates"] == 6
        and "101 passed" in (root / "qualification.txt").read_text().splitlines()[-1],
        "wrong gate/driver",
    )
    old = HERE / "receipts/memory-program-t4/attempt1"
    stamps = []
    metrics = {}
    runs = {}
    pid = set()
    identity = {}
    for cell in ORDER:
        size, mode, _ = cell.split("-")
        x = load(root / cell / "step.json")
        i = load(root / cell / "identity.json")
        a = load(old / (size + "-coupled.json"))
        p = load(old / (size + "-packet.json"))
        c = x["checks"]
        snap = i["snapshots"]
        grad = c["gradients"]
        progress = load(root / cell / "progress.json")
        require(
            i["cell"] == cell
            and i["ref"] == ref
            and i["pid"] != d["pid"] == i["parent"]
            and i["pid"] not in pid
            and i["threads"] == 4
            and i["precision"] == "highest"
            and i["strict"] == i["deterministic"] == (mode == "strict")
            and not i["warn_only"]
            and i["source_packet"] == p["sha256"]
            and i["producer"] == prev["ref"]
            and i["visible"] == ""
            and not i["cuda_initialized"]
            and not i["cuda_available"]
            and progress == dict(phase="updated", scored=4096, learner_pid=i["pid"]),
            "worker identity",
        )
        pid.add(i["pid"])
        stamps.extend([i["start_epoch_ns"], i["end_epoch_ns"]])
        require(
            x["targets"] == 4096
            and x["optimizer_steps"] == 1
            and x["mode"] == "fresh_sample"
            and not x["coupled"]
            and x["peak_allocated"] == x["peak_reserved"] == 0
            and all(c[k] == a["checks"][k] for k in FWD)
            and all(x[k] == a[k] for k in ("nll", "objective", "activity", "dispatches"))
            and c["input_sha"] == p["payload_sha"]
            and c["before"] != c["after"],
            "different forward program",
        )
        require(
            len(grad) == 35
            and all(
                isinstance(v["shape"], list) and math.prod(v["shape"]) > 0 and valid_sha(v["sha"])
                for v in snap.values()
            )
            and set(snap)
            == {
                g + ":" + k
                for k, t in grad.items()
                for g in (
                    ["param"]
                    if t is None
                    else ["grad", "param", "adam.step", "adam.exp_avg", "adam.exp_avg_sq"]
                )
            }
            and all(t is None or t == snap["grad:" + k]["sha"] for k, t in grad.items())
            and sum(math.prod(v["shape"]) for k, v in snap.items() if k.startswith("param:")) == 288987,
            "partial snapshot",
        )
        t = x["times"]
        require(
            all(v >= 0 for v in t.values())
            and t["wall_to_update_ns"]
            == sum(t[k] for k in ("forward_ns", "backward_ns", "gradient_attestation_ns", "clip_update_ns"))
            and i["end_epoch_ns"] - i["start_epoch_ns"] >= x["lifetime_ns"]
            and x["lifetime_ns"] >= t["wall_to_update_ns"] + t["setup_ns"],
            "invalid timing",
        )
        wall = t["wall_to_update_ns"] / 1e9
        metrics[cell] = dict(
            wall_s=wall,
            clean_step_s=wall - t["gradient_attestation_ns"] / 1e9,
            forward_s=t["forward_ns"] / 1e9,
            backward_s=t["backward_ns"] / 1e9,
            update_s=t["clip_update_ns"] / 1e9,
            snapshot_preclip_s=t["gradient_attestation_ns"] / 1e9,
            lifetime_s=x["lifetime_ns"] / 1e9,
            native_s=x["archive"]["validated_ns"] / 1e9,
            lf_steps=x["archive"]["lf_steps"],
            cpu_core_s=x["cpu_core_s"],
            rss_mib=x["rss_kib"] / 1024,
            grad_hash_differences_vs_historical_coupled=[
                k for k, v in grad.items() if v != a["checks"]["gradients"][k]
            ],
            same_after_as_historical=c["after"] == a["checks"]["after"],
        )
        runs[cell] = x
        identity[cell] = i
    require(
        stamps == sorted(set(stamps))
        and stamps[0] >= d["started"]
        and done["end_epoch_ns"] >= stamps[-1]
        and len({(i["torch"], i["interop"]) for i in identity.values()}) == 1,
        "not sequential/matched",
    )
    diff = load(root / "tensor-differences.json")
    results = {}
    require(set(diff) == {ORDER[j] + "_vs_" + ORDER[k] for j, k in PAIRS}, "incomplete contrasts")
    for j, k in PAIRS:
        a, b = ORDER[j], ORDER[k]
        x, y = [identity[z]["snapshots"] for z in (a, b)]
        key = a + "_vs_" + b
        entries = diff[key]
        require(entries.keys() == x.keys() == y.keys(), "parameter schema changed")
        groups = {}
        for label, v in entries.items():
            require(
                v["elements"] == math.prod(x[label]["shape"])
                and x[label]["shape"] == y[label]["shape"]
                and (v["unequal"] == 0) == (x[label]["sha"] == y[label]["sha"])
                and 0 <= v["unequal"] <= v["elements"]
                and math.isfinite(v["max_abs"])
                and math.isfinite(v["ref_absmax"])
                and min(v["max_abs"], v["ref_absmax"]) >= 0
                and (v["unequal"] or v["max_abs"] == 0.0)
                and v["within_t1"] == (v["max_abs"] <= 2e-4 + 2e-4 * v["ref_absmax"]),
                "invalid tensor difference",
            )
            g, name = label.split(":", 1)
            s = groups.setdefault(g, dict(elements=0, unequal=0, max_abs=0.0, changed=[], within_t1=True))
            s["elements"] += v["elements"]
            s["unequal"] += v["unequal"]
            s["max_abs"] = max(s["max_abs"], v["max_abs"])
            s["within_t1"] &= v["within_t1"]
            if v["unequal"]:
                s["changed"].append(name)
        exact = all(s["unequal"] == 0 for s in groups.values())
        require(not exact or runs[a]["checks"] == runs[b]["checks"], "inconsistent exact check")
        results[key] = dict(exact=exact, groups=groups)
    strict = all(results[ORDER[j] + "_vs_" + ORDER[k]]["exact"] for j, k in PAIRS[:2])
    ordinary = results[ORDER[0] + "_vs_" + ORDER[5]]["exact"]
    numeric = results[ORDER[0] + "_vs_" + ORDER[1]]["groups"]["grad"]["within_t1"]
    ratio = sum(metrics[ORDER[j]]["wall_s"] for j in (1, 3)) / sum(
        metrics[ORDER[j]]["wall_s"] for j in (0, 5)
    )
    return dict(
        ref=ref,
        cumulative_x47_upper=1284,
        remaining_s=516,
        card_s=0,
        actual_targets=24576,
        distinct_target_positions=4096,
        disposable_steps=6,
        cpu_diagnostic_wall_s=done["wall_ns"] / 1e9,
        all_forwards_match_t4=True,
        qualify_strict_cpu=strict,
        ordinary_repeat_exact=ordinary,
        new_cross_mode_gradient_tolerance=numeric,
        retained_strict_over_ordinary_wall=ratio,
        low_overhead_in_this_batch=bool(strict and numeric and ratio <= 1.25),
        metrics=metrics,
        comparisons=results,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
