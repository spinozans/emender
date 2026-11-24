"""S17 public audit. Reconstruct admissions/gates/work and lifecycle without scores, texts or weights."""

import argparse
from collections import Counter
from hashlib import blake2b, sha256
import json
from pathlib import Path
from struct import pack

from .direct_arrival_verify import offer_metrics
from .frozen_packet_verify import valid_sha
from .serving_verify import HERE, fields, ledger, load, require
from .terminal_branch_verify import wire

ROOT = HERE / "receipts/memory-program-s17"
ORDER = dict(
    replicated=[("spaced", "eager"), ("spaced", "demand"), ("burst", "demand"), ("burst", "eager")],
    retained=[("spaced", "demand"), ("spaced", "eager"), ("burst", "eager"), ("burst", "demand")],
)
GATES = 170


def expected(row, n):
    return bytes(
        int.from_bytes(
            blake2b(pack(">QQQ", 7823, row, v), digest_size=8, person=b"byte-entry-v1").digest()[:3], "big"
        )
        < 2**20
        for v in range(32, 32 + n)
    )


def scores(proof, n, ent):
    sizes = dict(byte_logp=256 * n, gate_logp=n, joint_logp=n, conditional_logp=ent, trust=4 * (n - ent))
    require(
        proof["passed"]
        and proof["positions"] == n
        and proof["entries"] == ent
        and proof["wrong_byte"] == proof["wrong_entry"] == proof["wrong_evidence"] == 0
        and set(proof["numeric"]) == sizes.keys()
        and all(
            v["violations"] == 0 and v["elements"] == sizes[k] and 0 <= v["max_abs"] < float("inf")
            for k, v in proof["numeric"].items()
        ),
        "incomplete numeric/structural parity",
    )


def cell(x, oracle, idn, pids):
    n = 384 if x["offer"] == "closed" else 348
    eager = x["provider"] == "eager"
    z = wire(x["wire"], False, idn)
    m = offer_metrics(x)
    require(
        x["emitted"] == n
        and x["stale_cancels"] == (0 if n == 384 else 4)
        and (n == 384 or x["arrival_prefix"] == x["full"] == oracle["arrival_prefix"])
        and x["slots_zero"]
        and x["cache_empty"]
        and all(0 <= b["pages"] <= 256 for b in x["cache"].values())
        and x["cache"]["direct"]["rows"] > 0
        and z["r_rows"] == sum(b["rows"] for b in x["cache"].values()),
        "missing prefix/transport/retirement",
    )
    d, w, h, p = x["routing"], x["work"], x["host"], x["position"]
    e = sum(row["enter"] for row in x["full"])
    require(
        d["enter"] == e
        and 0 < e < n
        and d["predictions"] == w["acknowledged"] == d["direct"] + e == n
        and d["nonemit"] == sum(row["nonemit"] for row in x["full"])
        and d["program_keys"] == sum(row["program_keys"] for row in x["full"])
        and d["logical_probes"] == sum(row["probes"] for row in x["full"])
        and d["logical_probes"] >= d["submitted_keys"] >= d["hist_calls"] > 0
        and d["transfer_batches"] == d["gate_batches"] == w["batches"] == len(x["ticks"])
        and d["transfer_bytes"] == n * 2048
        and d["blocking_transfer_ns"] > 0
        and w["admissions"] == w["closed_bindings"] == h["released"] == len(x["private"]) == 32
        and w["prefill_tokens"] == h["prefill_tokens"] == 1024
        and w["step_tokens"] == h["step_tokens"] == n - 32
        and sum(int(k) * v.get("real_replays", 0) for k, v in x["buckets"].items()) == n - 32
        and not x["dispatch"].get("eager_width_miss")
        and p["rc"] == 0
        and p["peak_kib"] > 0
        and p["open_ns"] > 0,
        "wrong logical model/transfer/host work",
    )
    # Rebuild each real subset size from the receipt's actual tick memberships and initial independent gate.
    idx = Counter()
    cohorts = Counter()
    for t in x["ticks"]:
        entered = sum(expected(i, idx[i] + 1)[-1] for i in t["rows"])
        cohorts.update(program=int(bool(entered)), direct=int(entered < len(t["rows"])))
        idx.update(t["rows"])
    require(
        (d["direct_batches"], d["program_batches"]) == (cohorts["direct"], cohorts["program"]),
        "mixed-clock batch accounting",
    )
    work = Counter()
    exits = []
    actors = [p["pid"], x["wire"]["pid"]]
    for i, (q, f, u) in enumerate(zip(x["private"], x["full"], x["arrival_prefix"], strict=True)):
        for row in (f, u):
            choices = expected(i, row["length"])
            require(
                row["enter"] == sum(choices)
                and row["entry_sha"] == sha256(choices).hexdigest()
                and valid_sha(row["observed_sha"])
                and row["probes"] >= row["length"] - row["enter"]
                and 0 <= row["registers"] <= row["nonemit"] <= row["enter"] * 5,
                "changed per-client gates or evidence coverage",
            )
        v = Counter(q["work"])
        real = v["branch_rpcs"] + v["locate_rpcs"]
        require(
            q["row"] == i
            and q["scrub"]
            and q["matched"]
            and q["observed_sha"] == f["observed_sha"]
            and v["observations"] == f["length"] + 1
            and v["observed_bytes"] == 32 + f["length"]
            and v["native_bytes"] + v["unsynced_at_close"] == v["observed_bytes"]
            and v["native_starts"] == int(eager or real > 0) == int(q["exit"] is not None),
            "wrong private lifetime or append debt",
        )
        if eager:
            require(
                v["native_appends"] == v["observations"] and not v["unsynced_at_close"],
                "eager skipped an append",
            )
        else:
            require(
                v["native_starts"] <= v["native_appends"] <= min(real, v["observations"])
                and v["unsynced_at_close"] > 0
                and (32 <= v["native_bytes"] < v["observed_bytes"] if real else v["native_bytes"] == 0),
                "invalid materialization",
            )
        if q["exit"] is not None:
            ex = q["exit"]
            require(
                ex["rc"] == 0 and ex["peak_kib"] > 0 and v["open_ns"] > 0 and v["append_ns"] > 0,
                "unreaped private worker",
            )
            actors.append(ex["pid"])
            exits.append(ex)
        work.update(v)
    require(
        len(actors) == len(set(actors))
        and not pids.intersection(actors)
        and 0 <= x["peak_private_workers"] <= min(len(exits), x["peak_active"])
        and (bool(x["peak_private_workers"]) == bool(exits))
        and (not eager or x["peak_private_workers"] == x["peak_active"]),
        "unbound PID or live capacity",
    )
    pids.update(actors)
    arc = Counter(p["work"])
    return dict(
        **m,
        **z,
        enter=e,
        direct=n - e,
        direct_batches=d["direct_batches"],
        program_batches=d["program_batches"],
        direct_probes=d["logical_probes"],
        program_keys=d["program_keys"],
        nonemit=d["nonemit"],
        private_starts=len(exits),
        peak_private=x["peak_private_workers"],
        native_appends=work["native_appends"],
        native_append_bytes=work["native_bytes"],
        observed_bytes=work["observed_bytes"],
        unsynced_on_release=work["unsynced_at_close"],
        live_branches=work["branch_rpcs"],
        live_locates=work["locate_rpcs"],
        value_reads=work["value_reads"],
        value_bytes=work["bytes"],
        live_open_s=work["open_ns"] / 1e9,
        live_append_s=work["append_ns"] / 1e9,
        position_keys=arc["worker_keys"],
        certified_overcap=arc["certified_overcap"],
        lf_rows=arc["lf_rows"],
        lf_steps=arc["lf_steps"],
        live_summed_exit_hwm_mib=sum(v["peak_kib"] for v in exits) / 1024,
        position_peak_mib=p["peak_kib"] / 1024,
        server_hwm_mib=x["server_hwm_kib"] / 1024,
        peak_gpu_mib=x["peak_allocated"] / 2**20,
        transfer_bytes=d["transfer_bytes"],
        measured_core_s=x["measured_main_core_s"] + x["measured_native_core_s"],
        lifecycle_core_s=x["lifecycle_main_core_s"] + x["lifecycle_children_core_s"],
    )


def audit(root=ROOT):
    prior = load(HERE / "receipts/memory-program-t11/verified.json")
    require(
        prior["equivalence_gate"] and prior["cumulative_x47_upper"] == 1386, "unbound prior ledger/learner"
    )
    aa, good = ledger(root, 3552)
    cost = sum(a["upper"] for a in aa)
    require(len(aa) == len(good) == 1 and 0 < cost <= 130, "missing/overbudget S17")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    s14 = HERE / "receipts/memory-program-s14/attempt1"
    history = load(s14 / "identity.json")
    old = history["setup_only"]
    require(
        ref == idn["ref"] == (root / "source-ref").read_text().strip() == (r / "ref").read_text().strip()
        and f"{GATES} passed in" in (root / "qualification.txt").read_text().splitlines()[-1]
        and int(fields(r / "released")["released_epoch"]) <= 1795531800
        and all(
            idn[k] == history[k]
            for k in (
                "data",
                "prompt_sha",
                "prompt_bytes",
                "threads",
                "active_cap",
                "waiting_cap",
                "direct_head",
                "step_widths",
                "generated_length",
                "size_order",
                "device",
                "torch",
                "branch_exe_sha",
                "mount",
            )
        )
        and all(idn[k] == v for k, v in old.items())
        and idn["threads"] == 4
        and idn["gate_seed"] == 7823
        and idn["gate"] == load(HERE / "receipts/memory-program-t2/attempt1/identity.json")["entry_initial"]
        and valid_sha(idn["model"])
        and idn["size_order"] == list(ORDER)
        and all(
            idn["generation_base"][k] == v
            for k, v in history["generation_base"].items()
            if k != "model_load_s"
        )
        and load(r / "allocator.json")["expandable_segments"] is True,
        "wrong host, predictor, prompts, or gate",
    )
    build = load(r / "construction.json")
    tr = load(r / "transfer.json")
    s7 = load(HERE / "receipts/memory-program-s7/attempt1/construction.json")
    require(
        all(
            build[k] == s7[k]
            for k in ("parameters_verified", "parameter_bytes", "distinct_storages", "rules", "tensors")
        )
        and build["parameters_verified"] == 2221786584
        and build["small_to_cpu_ns"] > 0
        and build["build_verify_ns"] > 0
        and tr["wide_drop_and_small_to_gpu_ns"] > 0,
        "not verified dense host pair",
    )
    metrics = {}
    ratios = {}
    setup = {}
    errors = {}
    oracles = {}
    clocks = []
    actors = {idn["server_pid"], aa[0]["pid"]}
    high = max(build["peak_allocated"], tr["peak_allocated"])
    reserved = max(build["peak_reserved"], tr["peak_reserved"])
    for size, order in ORDER.items():
        wide = size == "replicated"
        boot = load(r / f"{size}-setup.json")
        end = load(r / f"{size}-closed.json")
        require(
            boot["parameters"] == (2221786584 if wide else 149855670)
            and boot["tensor_bytes"] == (962953216 if wide else 240762880)
            and 0 < boot["private_bytes"] <= (256 if wide else 64) << 20
            and set(boot["buckets"]) == {"1", "4", "8", "16"}
            and end["slots_freed"]
            and end["graphs_freed"]
            and end["lifecycle_ns"] > boot["prepare_ns"] > 0
            and idn["order"][size] == [list(a) for a in order],
            "wrong setup/ordering",
        )
        setup[size] = dict(
            arena_bytes=boot["tensor_bytes"],
            private_bytes=boot["private_bytes"],
            prepare_s=boot["prepare_ns"] / 1e9,
            five_cell_lifecycle_s=end["lifecycle_ns"] / 1e9,
        )
        high = max(high, boot["peak_allocated"], end["peak_allocated"])
        reserved = max(reserved, boot["peak_reserved"], end["peak_reserved"])
        oracle = load(r / f"{size}-closed-eager.json")
        oracles[size] = oracle
        for offer, provider in [("closed", "eager"), *order]:
            key = f"{size}-{offer}-{provider}"
            x = load(r / (key + ".json"))
            require(
                (x["name"], x["size"], x["offer"], x["provider"]) == (key, size, offer, provider),
                "wrong cell",
            )
            q = metrics[key] = cell(x, oracle, idn, actors)
            clocks.extend([x["begin_clock"], x["begin_clock"] + x["lifecycle_ns"]])
            high = max(high, x["peak_allocated"])
            reserved = max(reserved, x["peak_reserved"])
            if offer != "closed":
                proof = load(r / (key + "-parity.json"))
                scores(proof, 348, q["enter"])
                errors[key] = proof
        for offer in ("spaced", "burst"):
            key = f"{size}-{offer}"
            a, b = (metrics[f"{key}-{p}"] for p in ("eager", "demand"))
            require(
                all(
                    a[k] == b[k]
                    for k in (
                        "enter",
                        "direct",
                        "direct_probes",
                        "program_keys",
                        "nonemit",
                        "observed_bytes",
                        "live_branches",
                        "live_locates",
                        "value_reads",
                        "value_bytes",
                    )
                ),
                "different logical work",
            )
            starts = b["private_starts"] / a["private_starts"]
            app = b["native_appends"] / a["native_appends"]
            rate = b["bytes_s"] / a["bytes_s"]
            first = b["first_ms"]["p95"] / a["first_ms"]["p95"]
            ratios[key] = dict(
                starts=starts,
                appends=app,
                rate=rate,
                first_p95=first,
                passed=b["live_branches"] + b["live_locates"] > 0
                and starts <= 0.50
                and app <= 0.50
                and rate >= 0.95
                and first <= 1.10,
            )
    cross = load(r / "cross-size.json")
    scores(cross, 384, metrics["retained-closed-eager"]["enter"])
    total = sum(m["bytes"] for m in metrics.values())
    done = load(r / "complete.json")
    require(
        oracles["retained"]["full"] == oracles["replicated"]["full"]
        and clocks == sorted(clocks)
        and done["server_pid"] == idn["server_pid"]
        and done["steps"] == 0
        and done["model"] == idn["model"]
        and done["total_emitted"] == total == 3552
        and done["lifecycle_ns"] >= clocks[-1] - clocks[0],
        "incomplete or changed generation",
    )
    qualified = all(z["passed"] for z in ratios.values())
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=cost,
        cumulative_x47_upper=1386 + cost,
        remaining_s=414 - cost,
        updates=0,
        gradient_targets=0,
        oracle_bytes=768,
        arrival_bytes=2784,
        total_emitted=total,
        completed_arrivals=224,
        canceled_arrivals=32,
        setup=setup,
        wide_build_s=build["build_verify_ns"] / 1e9,
        small_to_cpu_s=build["small_to_cpu_ns"] / 1e9,
        wide_drop_restore_s=tr["wide_drop_and_small_to_gpu_ns"] / 1e9,
        gpu_process_inside_main_s=done["lifecycle_ns"] / 1e9,
        peak_gpu_mib=high / 2**20,
        peak_reserved_mib=reserved / 2**20,
        metrics=metrics,
        ratios=ratios,
        cross_size=cross,
        errors=errors,
        functional_gate=True,
        demand_recommended=qualified,
        panel_throughput_win=qualified and all(z["rate"] >= 1.10 for z in ratios.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))
