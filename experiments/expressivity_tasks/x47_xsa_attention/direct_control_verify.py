"""Receipt-only T9 audit. Requires neither corpus/index access nor host/diagnostic tensors."""

import argparse
import json
import math
from pathlib import Path
from .serving_verify import HERE, load, require
from .frozen_packet_verify import valid_sha

ROOT = HERE / "receipts/memory-program-t9"
ORDER = ("serial-A", "grouped-A", "grouped-B", "serial-B")


def shapes():
    spec = dict(byte_logp=[32, 128, 256], gate=[32, 128, 4], nlls=[32, 128], nll=[])
    for k, shape in [
        ("hidden.weight", [64, 270]),
        ("hidden.bias", [64]),
        ("vote.weight", [4, 64]),
        ("vote.bias", [4]),
    ]:
        for a in ("grad", "param", "adam.exp_avg", "adam.exp_avg_sq", "adam.step"):
            spec[a + ":" + k] = [] if a == "adam.step" else shape
    return spec


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-t8/verified.json")
    prod = load(HERE / "receipts/memory-program-t4/attempt1/identity.json")
    meta = load(HERE / "receipts/memory-program-t4/attempt1/retained-packet.json")
    driver = load(root / "driver.json")
    done = load(root / "complete.json")
    o = load(root / "oracle.json")
    ref = (root / "gated-ref").read_text().strip()
    pid = driver["pid"]
    stamps = [driver["start_epoch_ns"]]
    workers = set()
    metrics = {}
    spec = shapes()
    refcell = None
    tot_compared = 0
    require(
        prev["qualified"]
        and prev["remaining_s"] == 476
        and prev["cumulative_x47_upper"] == 1324
        and tuple(driver["order"]) == ORDER
        and driver["ref"] == ref
        and done["pid"] == pid
        and done["targets"] == 16384
        and done["steps"] == 4
        and done["distinct_train_positions"] == 4096
        and done["card_s"] == driver["card_s"] == 0
        and done["elapsed_ns"] <= 480e9
        and "114 passed" in (root / "qualification.txt").read_text().splitlines()[-1],
        "different diagnostic/budget",
    )
    docs = meta["binding"]["rows"]
    checks = o["checks"]
    require(
        o["rc"] == [0, 0]
        and o["wall_ns"] > 0
        and [(x["document"], x["start"], x["visible"]) for x in checks]
        == [(r["document"], r["start"], t) for r in docs[:4] for t in (1, 7, 17, 63, 128)]
        and all(valid_sha(x["evidence"]) for x in checks),
        "independent native/prefix oracle",
    )
    for i, name in enumerate(ORDER):
        r = root / name
        s = load(r / "cell.json")
        progress = load(r / "progress.json")
        ex = load(r / "exit.json")
        w = s["work"]
        local = s["local"]
        p = s["pages"]
        a = s["availability"]
        d = s["depth"]
        v = s["snapshots"]
        require(
            s["pid"] not in workers
            and s["pid"] != pid == s["parent"]
            and s["name"] == name
            and s["ref"] == ref
            and s["strict"]
            and not s["warn_only"]
            and s["threads"] == 4
            and s["visible"] == ""
            and not s["cuda_initialized"]
            and not s["cuda_available"]
            and s["native_rc"] == ex["rc"] == 0
            and s["torch"] == prod["torch"]
            and s["seed"] == 7873
            and progress == dict(phase="updated", targets=4096, pid=s["pid"]),
            "worker identity/termination",
        )
        workers.add(s["pid"])
        stamps.extend((s["start_epoch_ns"], s["end_epoch_ns"]))
        require(
            s["packet"] == meta["sha256"]
            and s["features"] == meta["payload_sha"]
            and (s["mount"], s["binary"]) == (prod["mount"], prod["branch_exe_sha"])
            and s["host_calls"] == 0
            and s["trust_calls"] == 128
            and s["targets"] == 4096
            and s["steps"] == 1
            and s["parameters"] == 17604
            and len(s["per_document"]) == 32
            and all(valid_sha(h) for h in s["per_document"])
            and valid_sha(s["init"])
            and valid_sha(s["after"])
            and s["init"] != s["after"]
            and 0 < s["preclip_norm"] < math.inf
            and 0 <= s["unit_mass_err"] <= 2e-5
            and 0 <= s["gate_mass_err"] <= 2e-5
            and s["nll"] >= 0
            and math.isfinite(s["nll"]),
            "different predictor or invalid update",
        )
        require(
            v.keys() == spec.keys()
            and all(v[k]["shape"] == sh and valid_sha(v[k]["sha"]) for k, sh in spec.items())
            and 4 * sum(map(math.prod, spec.values())) <= s["ipc_bytes"] <= 8 << 20
            and abs(sum(s["gate_mean"]) - 1) <= 2e-5
            and all(0 <= z <= 1 for z in s["gate_mean"]),
            "incomplete tensor evidence",
        )
        require(
            local["bytes"] == 32 * 129
            and local["increments"] == 32 * sum(129 - k for k in (0, 1, 2, 4, 8))
            and 0 < local["rows"] <= local["edges"] <= local["increments"]
            and a["host"] == a["local"] == 4096
            and a["archive"] == w["selected"]
            and 0 <= a["latest"] <= 4096
            and sum(n for k, n in d.items() if k.startswith("archive:"))
            == sum(n for k, n in d.items() if k.startswith("local:"))
            == 4096
            and w["logical_probes"]
            == sum(w.get(k, 0) for k in ("selected", "absent", "unavailable", "end_only"))
            and 4096 <= w["logical_probes"] <= 4096 * 6
            and 0 < w["hist_calls"] <= w["submitted_keys"] <= w["logical_probes"]
            and (not name.startswith("serial") or w["hist_calls"] == w["logical_probes"])
            and 0 < p["cache_pages"] <= p["cap"] == 256
            and 0 < p["batches"] <= p["rows"]
            and p["response_bytes"] == p["rows"] * 3080
            and p["request_bytes"] == p["rows"] * 4 + p["batches"] * 5,
            "wrong chosen work/backoff/count price",
        )
        require(
            s["wall_ns"] == s["forward_ns"] + s["backward_ns"] + s["clip_attest_step_ns"]
            and s["forward_ns"] >= sum(s["clocks"].values())
            and ex["process_wall_ns"] >= s["lifecycle_ns"] >= s["wall_ns"] + s["setup_ns"]
            and s["peak_main_kib"] > 0
            and s["native_rss_kib"] > 0
            and s["core_s"] > 0,
            "bad lifecycle costs",
        )
        if i == 0:
            refcell = s
        else:
            dif = load(r / "vs-serial-A.json")
            require(
                all(
                    s[k] == refcell[k]
                    for k in (
                        "per_document",
                        "init",
                        "after",
                        "preclip_norm",
                        "nll",
                        "depth",
                        "availability",
                        "gate_mean",
                        "local",
                        "snapshots",
                    )
                )
                and all(
                    w.get(k, 0) == refcell["work"].get(k, 0)
                    for k in ("selected", "absent", "unavailable", "end_only", "logical_probes")
                )
                and dif["same_logical"]
                and dif["tensors"].keys() == spec.keys()
                and all(
                    q["elements"] == math.prod(spec[k])
                    and q["ref_sha"] == q["actual_sha"] == v[k]["sha"]
                    and q["unequal_bits"] == q["max_abs"] == 0
                    for k, q in dif["tensors"].items()
                ),
                "mismatched logical/tensor trajectory",
            )
            tot_compared += sum(q["elements"] for q in dif["tensors"].values())
        metrics[name] = dict(
            wall_s=s["wall_ns"] / 1e9,
            lifecycle_s=s["lifecycle_ns"] / 1e9,
            process_s=ex["process_wall_ns"] / 1e9,
            archive_s=s["clocks"]["archive_ns"] / 1e9,
            core_s=s["core_s"],
            peak_main_mib=s["peak_main_kib"] / 1024,
            native_rss_mib=s["native_rss_kib"] / 1024,
            hist_calls=w["hist_calls"],
            submitted_keys=w["submitted_keys"],
            page_batches=p["batches"],
            page_rows=p["rows"],
            page_wire_bytes=p["response_bytes"] + p["request_bytes"],
            gate_mass_err=s["gate_mass_err"],
            unit_mass_err=s["unit_mass_err"],
        )
    stamps.append(done["end_epoch_ns"])
    require(
        stamps == sorted(set(stamps))
        and done["elapsed_ns"]
        >= o["wall_ns"] + sum(load(root / n / "exit.json")["process_wall_ns"] for n in ORDER),
        "overlap/unaccounted processes",
    )
    serial = [metrics[n] for n in ORDER if n.startswith("serial")]
    grouped = [metrics[n] for n in ORDER if n.startswith("grouped")]
    ratio = sum(x["wall_s"] for x in grouped) / sum(x["wall_s"] for x in serial)
    rss = max(x["peak_main_mib"] for x in grouped) / max(x["peak_main_mib"] for x in serial)
    return dict(
        ref=ref,
        direct_control_qualified=True,
        grouped_default=ratio <= 0.9 and rss <= 1.2,
        grouped_over_serial_mean_wall=ratio,
        grouped_over_serial_peak_rss=rss,
        targets=16384,
        distinct_train_positions=4096,
        independent_direct_steps=4,
        program_updates=0,
        host_updates=0,
        card_s=0,
        cumulative_x47_upper=1324,
        remaining_s=476,
        parameters=17604,
        oracle_frontiers=len(checks),
        compared_tensor_elements=tot_compared,
        logical_probes=refcell["work"]["logical_probes"],
        status={k: refcell["work"].get(k, 0) for k in ("unavailable", "absent", "end_only", "selected")},
        local=refcell["local"],
        availability=refcell["availability"],
        depth=refcell["depth"],
        gate_mean=refcell["gate_mean"],
        initial_train_nll=refcell["nll"],
        metrics=metrics,
        oracle_s=o["wall_ns"] / 1e9,
        diagnostic_s=done["elapsed_ns"] / 1e9,
        held_out=False,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
