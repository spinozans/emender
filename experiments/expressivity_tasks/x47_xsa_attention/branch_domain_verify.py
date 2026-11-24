"""Public receipt-only audit for S15. No GPU, local P100, packet, or private head needed."""

import argparse
import json
from pathlib import Path
from statistics import mean

from .frozen_packet_verify import valid_sha
from .serving_verify import HERE, load, require
from .terminal_branch_verify import wire

ROOT = HERE / "receipts/memory-program-s15"
ORDER = ("fresh-A", "resident-A", "resident-B", "fresh-B")


def read_panel(root, *, order, gates, resident_labels, compare_to=None):
    """Audit identical S15 wave work, independent of the registered outer lifetime contrast."""
    ref = (root / "gated-ref").read_text().strip()
    dr = load(root / "driver.json")
    end = load(root / "complete.json")
    s14 = HERE / "receipts/memory-program-s14"
    prev = load(s14 / "verified.json")
    prior = load(s14 / "attempt1/identity.json")
    prod = load(HERE / "receipts/memory-program-t4/attempt1/identity.json")
    pm = load(HERE / "receipts/memory-program-t4/attempt1/retained-packet.json")
    require(
        prev["cumulative_x47_upper"] == 1386
        and prev["remaining_s"] == 414
        and dr["ref"] == ref
        and tuple(dr["order"]) == order
        and dr["pid"] == end["pid"]
        and end["elapsed_ns"] <= 360e9
        and dr["card_s"] == end["card_s"] == end["updates"] == end["training_targets"] == 0
        and end["frontiers"] == 4096
        and end["unique_frontiers"] == 1024
        and f" {gates} passed in " in (root / "qualification.txt").read_text().splitlines()[-1],
        "unbound profile/budget",
    )
    expected = dict(
        head=load(s14 / "head.json")["after"],
        policy=prod["policy_sha"],
        checkpoint=prod["checkpoint"],
        features=pm["payload_sha"],
        packet=pm["sha256"],
        mount=prod["mount"],
        binary=prod["branch_exe_sha"],
        seed=7530,
        input128_sha=prod["input128_sha"],
        observed32_sha=prior["prompt_sha"],
        split="train",
    )
    stamps = [dr["start_epoch_ns"]]

    def signature(w):
        return dict(
            documents=w["documents"],
            logical=w["logical"],
            grant=w["grant"],
            wire={k: v for k, v in w["wire"].items() if k != "wall_ns"},
        )

    base = (
        {} if compare_to is None else {j: signature(load(compare_to / f"wave{j}.json")) for j in range(4)}
    )
    metrics = {}
    used = {dr["pid"]}
    for label in order:
        path = root / label
        cell = load(path / "cell.json")
        ex = load(path / "exit.json")
        resident = label in resident_labels
        require(
            cell["name"] == label
            and cell["ref"] == ref
            and cell["parent"] == dr["pid"]
            and cell["pid"] not in used
            and cell["identity"] == expected
            and cell["frontiers"] == 1024
            and cell["steps"] == 0
            and cell["main_core_s"] > 0
            and cell["children_core_s"] > 0
            and cell["torch"] == prior["torch"]
            and cell["threads"] == 4
            and cell["strict"]
            and not cell["warn_only"]
            and cell["visible"] == ""
            and not cell["cuda_available"]
            and not cell["cuda_initialized"]
            and ex["rc"] == 0
            and ex["process_wall_ns"] >= cell["lifecycle_ns"] >= cell["wall_ns"] > 0
            and load(path / "progress.json") == dict(pid=cell["pid"], frontiers=1024, wave=3),
            "incomplete or changed worker",
        )
        used.add(cell["pid"])
        stamps.extend([cell["start_epoch_ns"], cell["end_epoch_ns"]])
        op = cell["opens"]
        open_s = sum(a["open_ns"] for a in op) / 1e9
        require(
            len(op) == (1 if resident else 4)
            and all(
                a["open_ns"] > 0 and a["final_close_ns"] > 0 and a["grants"] == (4 if resident else 1)
                for a in op
            ),
            "open count/lifetime",
        )
        actors = []
        native = {}
        waves = []
        for a in op:
            z = a["wire"]
            pid = z["pid"]
            require(pid not in used, "reused native process identity")
            used.add(pid)
            actors.append(pid)
            native[pid] = wire(z, False, prod)
        for j in range(4):
            w = load(path / f"wave{j}.json")
            ds = w["documents"]
            sampled = j % 2 == 1
            z = w["wire"]
            l = w["logical"]
            pid = w["m20_pid"]
            require(
                w["wave"] == j
                and w["predictor"] == ("sampled" if sampled else "direct")
                and w["frontiers"] == 256
                and [d["row"] for d in ds] == list(range(8 * j, 8 * j + 8))
                and all(
                    d["frontiers"] == 32
                    and d["observed32_sha"] == expected["observed32_sha"][d["row"]]
                    and all(valid_sha(d[k]) for k in ("byte_logp", "factor", "argmax", "evidence"))
                    for d in ds
                )
                and w["expired_refused"]
                and w["sessions_cleared"]
                and len(w["actor_exits"]) == len(w["actor_peak_kib"]) == (9 if sampled else 0)
                and w["wall_ns"] >= w["private_lifecycle_ns"] >= w["learner_setup_ns"] + w["score_ns"] > 0
                and pid == actors[0 if resident else j]
                and w["m20_running_peak_kib"] / 1024 <= native[pid]["m20_peak_mib"]
                and 0 < w["grant"]["cached_before_clear"] <= 256
                and w["grant"]["rows"] == z["items"] >= z["frames"] == w["grant"]["batches"] > 0
                and valid_sha(z["sha"])
                and z["wall_ns"] > 0
                and z["send_bytes"] == 5 * z["frames"] + 4 * z["items"]
                and z["recv_bytes"] == 3080 * z["items"],
                "missing wave evidence",
            )
            for (private_pid, rc), hwm in zip(w["actor_exits"], w["actor_peak_kib"], strict=True):
                require(
                    private_pid not in used and rc == 0 and hwm > 0, "unreaped/shared private wave actor"
                )
                used.add(private_pid)
            if sampled:
                require(
                    l["opcodes"]["EMIT"] == 256
                    and valid_sha(l["dispatch_sha"])
                    and l["dispatches"] == sum(k["batches"] for k in l["provider"].values())
                    and l["live"]["appends"] == 256
                    and l["provider"]["P100/byte_edges"]["keys"] > 0
                    and w["append_ns"] > 0,
                    "sampled request mismatch",
                )
            else:
                require(
                    l["direct"]["trust_calls"] == 32
                    and l["direct"]["predictions"] == 256
                    and l["local_increments"] == 1160
                    and w["append_ns"] == 0,
                    "direct request mismatch",
                )
            sig = signature(w)
            require(base.setdefault(j, sig) == sig, "different full score, logical work, or actual R frame")
            waves.append(
                dict(
                    predictor=w["predictor"],
                    wall_s=w["wall_ns"] / 1e9,
                    scoring_s=w["score_ns"] / 1e9,
                    R_frames=z["frames"],
                    rows=z["items"],
                    wire=z["send_bytes"] + z["recv_bytes"],
                    native_r_s=z["wall_ns"] / 1e9,
                    peak_m20_so_far_mib=w["m20_running_peak_kib"] / 1024,
                    sampled_lf=l.get("archive", {}).get("lf_steps", 0),
                    private_exits=len(w["actor_exits"]),
                )
            )
        for k, pid in enumerate(actors):
            ws = waves if resident else waves[k : k + 1]
            a = native[pid]
            require(
                sum(w["R_frames"] for w in ws) == a["r_frames"]
                and sum(w["rows"] for w in ws) == a["r_rows"]
                and 16 + sum(w["wire"] for w in ws) == a["wire_bytes"],
                "unpriced inter-wave native work",
            )
        wall = cell["wall_ns"] / 1e9
        require(wall >= open_s + sum(w["wall_s"] for w in waves), "inconsistent primary wall")
        metrics[label] = dict(
            wall_s=wall,
            open_s=open_s,
            final_close_s=sum(a["final_close_ns"] for a in op) / 1e9,
            opens=len(op),
            waves=waves,
            main_peak_mib=cell["main_peak_kib"] / 1024,
            m20_peak_mib=max(a["m20_peak_mib"] for a in native.values()),
            core_s=cell["main_core_s"] + cell["children_core_s"],
            wire_including_handshakes=sum(a["wire_bytes"] for a in native.values()),
            worker_lifecycle_s=cell["lifecycle_ns"] / 1e9,
            interpreter_inclusive_s=ex["process_wall_ns"] / 1e9,
        )
    stamps.append(end["end_epoch_ns"])
    require(stamps == sorted(stamps), "workers overlapped or clock drift")

    return ref, end, metrics


def price(metrics, candidate):
    def ratio(k, reduce):
        return reduce(v[k] for l, v in metrics.items() if l.startswith(candidate)) / reduce(
            v[k] for l, v in metrics.items() if l.startswith("fresh")
        )

    ratios = dict(
        wall=ratio("wall_s", mean),
        open=ratio("open_s", mean),
        main_peak=ratio("main_peak_mib", max),
        m20_peak=ratio("m20_peak_mib", max),
    )
    ok = (
        ratios["wall"] <= 0.85
        and ratios["open"] <= 0.60
        and max(ratios["main_peak"], ratios["m20_peak"]) <= 1.20
    )
    return ratios, ok


def audit(root=ROOT):
    ref, end, metrics = read_panel(
        root, order=ORDER, gates=137, resident_labels=("resident-A", "resident-B")
    )
    ratios, ok = price(metrics, "resident")
    return dict(
        ref=ref,
        card_s=0,
        cumulative_x47_upper=1386,
        remaining_s=414,
        elapsed_s=end["elapsed_ns"] / 1e9,
        direct_prefix_predictions=2048,
        sampled_prefix_predictions=2048,
        distinct_frontiers=1024,
        training_targets=0,
        updates=0,
        generated_bytes=0,
        metrics=metrics,
        ratios=ratios,
        exact_rebinding_gate=True,
        resident_recommended=ok,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))
