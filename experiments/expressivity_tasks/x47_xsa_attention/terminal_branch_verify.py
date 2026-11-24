"""Public audit of T10: timing includes metadata requests, not just fewer full R responses."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from .frozen_packet_verify import valid_sha
from .frozen_route_verify import work
from .serving_verify import HERE, load, require

ROOT = HERE / "receipts/memory-program-t10"
ORDER = (
    "grouped-ref-A",
    "grouped-terminal-A",
    "grouped-terminal-B",
    "grouped-ref-B",
    "sampled-ref",
    "sampled-terminal",
)


def wire(w, terminal, prod):
    require(
        w["rc"] == 0
        and w["peak_kib"] > 0
        and w["binary"] == prod["branch_exe_sha"]
        and w["mount"] == prod["mount"]
        and w["handshake"] == 16
        and set(w["rpc"]) == {"S", "R"},
        "wire actor",
    )
    ss, rr = (Counter(w["rpc"][k]) for k in ("S", "R"))
    require(
        (ss["frames"] > 0) == terminal
        and ss["frames"] <= ss["items"] <= 8 * ss["frames"]
        and ss["send_bytes"] == 5 * ss["frames"] + ss["items"]
        and ss["recv_bytes"] == 32 * ss["items"]
        and 0 < rr["frames"] <= rr["items"]
        and rr["send_bytes"] == 5 * rr["frames"] + 4 * rr["items"]
        and rr["recv_bytes"] == 3080 * rr["items"]
        and rr["wall_ns"] > 0
        and (ss["wall_ns"] > 0) == terminal
        and w["bytes"] == 16 + sum(z["send_bytes"] + z["recv_bytes"] for z in (ss, rr)),
        "missing actual frame/bytes",
    )
    return dict(
        s_frames=ss["frames"],
        s_literal_bytes=ss["items"],
        r_frames=rr["frames"],
        r_rows=rr["items"],
        wire_bytes=w["bytes"],
        rpc_s=(ss["wall_ns"] + rr["wall_ns"]) / 1e9,
        m20_peak_mib=w["peak_kib"] / 1024,
    )


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-t9/verified.json")
    dref = load(HERE / "receipts/memory-program-t9/grouped-A/cell.json")
    pref = load(HERE / "receipts/memory-program-t5/retained-strict-A/step.json")
    pm = load(HERE / "receipts/memory-program-t6/cpu/retained-serial/identity.json")["tensors"]
    prod = load(HERE / "receipts/memory-program-t4/attempt1/identity.json")
    packet = load(HERE / "receipts/memory-program-t4/attempt1/retained-packet.json")
    ref = (root / "gated-ref").read_text().strip()
    dr = load(root / "driver.json")
    done = load(root / "complete.json")
    o = load(root / "oracle.json")
    require(
        tuple(dr["order"]) == ORDER
        and dr["ref"] == ref
        and dr["card_s"] == done["card_s"] == 0
        and dr["pid"] == done["pid"]
        and done["elapsed_ns"] <= 480e9
        and done["targets"] == 24576
        and done["direct_steps"] == 4
        and done["program_steps"] == 2
        and done["host_steps"] == 0
        and prev["grouped_default"]
        and prev["remaining_s"] == 476
        and "120 passed" in (root / "qualification.txt").read_text().splitlines()[-1],
        "wrong source/budget",
    )
    a, b = o["checks"]
    require(
        a["label"] == "cold"
        and b == dict(a, label="warm")
        and valid_sha(a["sha"])
        and a["keys"] == sum(a["counts"].values()) == sum(o["lengths"].values())
        and o["lengths"]["0"] == o["lengths"]["256"] == 1
        and o["input_sha"] == prod["input128_sha"][:8]
        and a["counts"].get("positive", 0) > 0
        and a["counts"].get("unavailable", 0) > 0
        and o["wall_ns"] > 0,
        "incomplete real prefix oracle",
    )
    oracle = [wire(o["reference"], False, prod), wire(o["candidate"], True, prod)]
    metrics = {}
    actors = {dr["pid"]}
    stamps = [dr["start_epoch_ns"]]
    compared = 0
    for name in ORDER:
        d = root / name
        s = load(d / "cell.json")
        w = load(d / "wire.json")
        ex = load(d / "exit.json")
        pg = load(d / "progress.json")
        proof = load(d / "comparison.json")
        term = "terminal" in name
        direct = name.startswith("grouped")
        require(
            s["pid"] not in actors
            and w["pid"] != s["pid"]
            and s["parent"] == dr["pid"]
            and s["ref"] == ref
            and ex["rc"] == 0
            and s["targets"] == 4096
            and s["steps"] == 1
            and pg == dict(pid=s["pid"], phase="updated", targets=4096)
            and s["name"] == name
            and s["threads"] == 4
            and s["strict"]
            and not s["warn_only"]
            and not s["cuda_initialized"]
            and not s["cuda_available"]
            and s["visible"] == ""
            and s["torch"] == prod["torch"]
            and s["packet"] == packet["sha256"]
            and s["features"] == packet["payload_sha"]
            and s["binary"] == prod["branch_exe_sha"]
            and s["mount"] == prod["mount"]
            and s["seed"] == (7873 if direct else 7809),
            "unbound step",
        )
        actors.add(s["pid"])
        stamps.extend([s["start_epoch_ns"], s["end_epoch_ns"]])
        z = wire(w, term, prod)
        if direct:
            require(
                s["snapshots"] == dref["snapshots"]
                and all(
                    s[k] == dref[k]
                    for k in (
                        "init",
                        "after",
                        "per_document",
                        "work",
                        "depth",
                        "availability",
                        "nll",
                        "preclip_norm",
                        "gate_mean",
                        "local",
                        "host_calls",
                        "trust_calls",
                    )
                )
                and s["pages"]["rows"] == z["r_rows"]
                and s["pages"]["batches"] == z["r_frames"]
                and (term or s["pages"] == dref["pages"])
                and s["native_pid"] == w["pid"]
                and s["native_rc"] == 0
                and s["gate_mass_err"] <= 2e-5
                and s["unit_mass_err"] <= 2e-5,
                "different direct computation",
            )
            logical = s["work"]["logical_probes"]
            native_s = s["clocks"]["archive_ns"] / 1e9
            lf = None
        else:
            q = load(d / "step.json")
            require(
                s["snapshots"] == pm
                and q["checks"] == pref["checks"]
                and work(q) == work(pref)
                and q["mode"] == "fresh_sample"
                and q["nll"] == s["nll"] == pref["nll"]
                and q["optimizer_steps"] == 1
                and not q["coupled"]
                and q["targets"] == 4096
                and s["wall_ns"] == q["times"]["wall_to_update_ns"]
                and q["peak_allocated"] == q["peak_reserved"] == 0,
                "different actual sampled addresses/adjoints",
            )
            logical = q["activity"]["usage"]["logical_keys"]
            native_s = sum(c["ns"] for c in q["provider"].values()) / 1e9
            lf = q["archive"]["lf_steps"]
        diff = proof["vs_fresh_reference"]
        require(
            proof["same_work"]
            and proof["same_historical_update"]
            and (diff is None) == (name in (ORDER[0], ORDER[4]))
            and 0 < s["ipc_bytes"] <= 12 << 20
            and ex["process_wall_ns"] >= s["lifecycle_ns"] >= s["wall_ns"] > 0
            and s["core_s"] > 0
            and s["peak_main_kib"] > 0,
            "missing reproduction or timing charge",
        )
        if diff is not None:
            require(
                diff.keys() == s["snapshots"].keys()
                and all(
                    v["max_abs"] == v["unequal_bits"] == 0
                    and v["ref_sha"] == v["actual_sha"] == s["snapshots"][k]["sha"]
                    and v["elements"] == math.prod(s["snapshots"][k]["shape"])
                    for k, v in diff.items()
                ),
                "tensor difference",
            )
            compared += sum(v["elements"] for v in diff.values())
        metrics[name] = dict(
            **z,
            wall_s=s["wall_ns"] / 1e9,
            lifecycle_s=s["lifecycle_ns"] / 1e9,
            subprocess_s=ex["process_wall_ns"] / 1e9,
            main_peak_mib=s["peak_main_kib"] / 1024,
            core_s=s["core_s"],
            provider_s=native_s,
            logical=logical,
            lf_steps=lf,
            nll=s["nll"],
        )
    stamps.append(done["end_epoch_ns"])
    require(
        stamps == sorted(set(stamps))
        and done["elapsed_ns"]
        >= o["wall_ns"] + sum(load(root / n / "exit.json")["process_wall_ns"] for n in ORDER),
        "overlap/clock",
    )
    ratios = {}
    gates = {}
    for family, ns in [("direct", ORDER[:4]), ("sampled", ORDER[4:])]:
        xx = [metrics[n] for n in ns if "terminal" in n]
        yy = [metrics[n] for n in ns if "ref" in n]
        r = {}
        for k in ("wall_s", "wire_bytes", "m20_peak_mib", "main_peak_mib"):
            fn = max if "mib" in k else sum
            r[k] = fn(x[k] for x in xx) / fn(y[k] for y in yy)
        ratios[family] = r
        gates[family] = max(r["m20_peak_mib"], r["main_peak_mib"]) <= 1.2 and (
            r["wall_s"] <= 0.9 and r["wire_bytes"] <= 0.5 if family == "direct" else r["wall_s"] <= 1.1
        )
    return dict(
        ref=ref,
        logical_and_update_gate=True,
        grouped_direct_default=gates["direct"],
        sampled_pair_nonregression=gates["sampled"],
        ratios=ratios,
        oracle_keys=a["keys"],
        oracle_cold_and_warm=oracle,
        checked_tensor_elements=compared,
        metrics=metrics,
        elapsed_s=done["elapsed_ns"] / 1e9,
        scored=24576,
        distinct_train_positions=4096,
        direct_steps=4,
        program_steps=2,
        host_steps=0,
        card_s=0,
        cumulative_x47_upper=1324,
        remaining_s=476,
        quality_tested=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
