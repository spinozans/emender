"""S14 offline receipt audit: completed generated work, parity, native RSS and open-arrival prices."""

import argparse
from collections import Counter
import json
from pathlib import Path
from .direct_service_head import GATES, HEAD_BYTES, T9_SHA, audit_step
from .frozen_packet_verify import valid_sha
from .serving_verify import HERE, fields, ledger, load, pct, require
from .terminal_branch_verify import wire

ROOT = HERE / "receipts/memory-program-s14"
ORDERS = {
    "replicated": [("spaced", "ref"), ("spaced", "terminal"), ("burst", "terminal"), ("burst", "ref")],
    "retained": [("spaced", "terminal"), ("spaced", "ref"), ("burst", "ref"), ("burst", "terminal")],
}


def parity(proof, n):
    require(
        proof["positions"] == n
        and proof["passed"]
        and proof["wrong_byte"] == proof["wrong_evidence"] == 0
        and set(proof["numeric"]) == {"byte_logp", "trust"}
        and all(
            x["violations"] == 0 and x["max_abs"] >= 0 and x["elements"] == n * k
            for x, k in [(proof["numeric"]["byte_logp"], 256), (proof["numeric"]["trust"], 4)]
        ),
        "incomplete prefix comparison",
    )


def clock_metrics(x, oracle, prod):
    rows = x["requests"]
    ticks = x["ticks"]
    closed = x["offer"] == "closed"
    n = 384 if closed else 348
    require(
        [r["row"] for r in rows] == list(range(32))
        and len(x["full"]) == len(x["arrival_prefix"]) == 32
        and (closed or x["arrival_prefix"] == x["full"] == oracle["arrival_prefix"]),
        "different row identities/evidence",
    )
    require(
        x["slots_zero"]
        and x["cache_empty"]
        and 0 < x["cache_pages_before_close"] <= 256
        and x["emitted"] == n
        and x["stale_cancels"] == (0 if closed else 4),
        "missing work/cleanup",
    )
    w, h, d = x["work"], x["host"], x["direct_work"]
    local = x["local"]
    z = wire(x["wire"], x["provider"] == "terminal", prod)
    require(
        local["opened"] == local["closed"] == h["released"] == w["admissions"] == w["closed_bindings"] == 32
        and local["observed_bytes"] == n + 1024
        and w["acknowledged"] == d["predictions"] == n
        and w["prefill_tokens"] == h["prefill_tokens"] == 1024
        and w["step_tokens"] == h["step_tokens"] == n - 32
        and w["batches"] == d["trust_calls"] == d["transfer_batches"] == len(ticks)
        and d["transfer_bytes"] == 2048 * n
        and d["blocking_transfer_ns"] > 0
        and sum(int(k) * v.get("real_replays", 0) for k, v in x["buckets"].items()) == n - 32
        and x["dispatch"].get("eager_width_miss", 0) == 0
        and d["logical_probes"] == sum(f["probes"] for f in x["full"])
        and d["logical_probes"] >= d["submitted_keys"] >= d["hist_calls"] > 0,
        "changed causal workload",
    )
    return dict(
        **z,
        **offer_metrics(x),
        logical_probes=d["logical_probes"],
        submitted_keys=d["submitted_keys"],
        hist_calls=d["hist_calls"],
        transfer_bytes=d["transfer_bytes"],
        blocking_transfer_s=d["blocking_transfer_ns"] / 1e9,
        measured_core_s=x["measured_main_core_s"] + x["measured_native_core_s"],
        lifecycle_core_s=x["lifecycle_main_core_s"] + x["lifecycle_children_core_s"],
        server_hwm_mib=x["server_hwm_kib"] / 1024,
        peak_gpu_mib=x["peak_allocated"] / 2**20,
    )


def offer_metrics(x):
    rows, ticks = x["requests"], x["ticks"]
    closed = x["offer"] == "closed"
    n = 384 if closed else 348
    require(
        [r["row"] for r in rows] == list(range(32)) and x["offer"] in ("closed", "burst", "spaced"),
        "wrong rows/offer",
    )
    queue = []
    jitter = []
    first = []
    inter = []
    last = []
    offlive = 0
    inside = 0
    for i, (row, f, u) in enumerate(zip(rows, x["full"], x["arrival_prefix"], strict=True)):
        k = 3 if i % 8 == 0 else 12
        acks = row["acks"]
        require(
            len(acks) == f["length"] == (12 if closed else k)
            and row["status"] == ("canceled" if not closed and k == 3 else "finished")
            and u["length"] == k
            and all(valid_sha(p[v]) for p in (u, f) for v in ("byte_sha", "evidence_sha"))
            and (f == u or (closed and k == 3 and f["probes"] >= u["probes"]))
            and row["scheduled"]
            == (0 if closed else i // 4 * 60_000_000 if x["offer"] == "spaced" else i // 16 * 50_000_000)
            and row["scheduled"] <= row["offered"] <= row["opening"] <= row["admitted"] < acks[0]
            and acks == sorted(set(acks)) == [t["ack"] for t in ticks if i in t["rows"]]
            and acks[-1] <= row["released"],
            "schedule or acknowledgment drift",
        )
        queue.append(row["opening"] - row["offered"])
        jitter.append(row["offered"] - row["scheduled"])
        first.append(acks[0] - row["offered"])
        inter.extend(b - a for a, b in zip(acks, acks[1:]))
        last.append(row["released"] - row["offered"])
        offlive += any(y["admitted"] <= row["offered"] < y["released"] for y in rows)
        inside += any(t["start"] <= row["offered"] < t["ack"] for t in ticks)
    peak = max(sum(y["admitted"] <= r["admitted"] < y["released"] for y in rows) for r in rows)
    require(
        peak == x["peak_active"]
        and 0 < peak <= 16
        and (x["peak_queue"] == 0 if closed else 0 < x["peak_queue"] <= 32)
        and all(0 < len(t["rows"]) == len(set(t["rows"])) <= 16 and t["start"] < t["ack"] for t in ticks)
        and all(a["ack"] <= b["start"] for a, b in zip(ticks, ticks[1:]))
        and sum(len(t["rows"]) for t in ticks) == n
        and x["lifecycle_ns"] >= x["domain_setup_ns"] + x["wall_ns"]
        and x["wall_ns"] >= max(r["released"] for r in rows) - min(r["offered"] for r in rows)
        and x["peak_reserved"] >= x["peak_allocated"] > 0
        and x["lifecycle_main_core_s"] > 0
        and x["lifecycle_children_core_s"] >= 0,
        "incoherent timing/physical load",
    )
    if closed:
        require(
            [t["rows"] for t in ticks]
            == [list(range(16 * b, 16 * b + 16)) for b in (0, 1) for _ in range(12)],
            "changed oracle blocks",
        )

    def ms(v):
        return {name: pct(v, q) for name, q in [("p50", 0.50), ("p95", 0.95), ("max", 1)]}

    return dict(
        bytes=n,
        bytes_s=n * 1e9 / x["wall_ns"],
        wall_s=x["wall_ns"] / 1e9,
        lifecycle_s=x["lifecycle_ns"] / 1e9,
        domain_setup_s=x["domain_setup_ns"] / 1e9,
        peak_active=peak,
        peak_waiting=x["peak_queue"],
        offers_during_live=offlive,
        offers_inside_tick=inside,
        tick_widths=Counter(len(t["rows"]) for t in ticks),
        offer_jitter_ms=ms(jitter),
        queue_ms=ms(queue),
        first_ms=ms(first),
        interresponse_ms=ms(inter),
        sojourn_ms=ms(last),
    )


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-t10/verified.json")
    require(
        prev["cumulative_x47_upper"] == 1324
        and prev["grouped_direct_default"]
        and not prev["sampled_pair_nonregression"],
        "wrong prior gate",
    )
    aa, good = ledger(root, 3552)
    paid = sum(a["upper"] for a in aa)
    require(len(aa) == len(good) == 1 and 0 < paid <= 115, "incomplete or overbudget S14")
    run = good[0]
    build = audit_step(root)
    head = load(root / "head.json")
    idn = load(run / "identity.json")
    ref = (root / "gated-ref").read_text().strip()
    s12 = load(HERE / "receipts/memory-program-s12/attempt1/identity.json")
    require(
        ref == idn["ref"] == build["ref"] == head["ref"] == (run / "ref").read_text().strip()
        and f"{GATES} passed" in (root / "qualification.txt").read_text().splitlines()[-1]
        and idn["direct_head"] == head
        and head["after"] == build["after"]
        and head["card_s"] == 0
        and head["history"] == T9_SHA
        and head["bytes"] == HEAD_BYTES
        and valid_sha(head["sha"])
        and build["end_epoch_ns"]
        <= head["published_epoch_ns"]
        < (1 + int(fields(run / "initial")["created"])) * 10**9
        and idn["threads"] == 4
        and idn["active_cap"] == 16
        and idn["waiting_cap"] == 32
        and idn["step_widths"] == [1, 4, 8, 16]
        and idn["size_order"] == list(ORDERS)
        and all(
            idn[k] == s12[k]
            for k in (
                "data",
                "prompt_sha",
                "prompt_bytes",
                "generated_length",
                "mount",
                "branch_exe_sha",
                "torch",
            )
        )
        and all(
            idn["generation_base"][k] == v for k, v in s12["generation_base"].items() if k != "model_load_s"
        )
        and all(idn["setup_only"][k] == s12[k] for k in idn["setup_only"]),
        "unbound serving weights, clock, or prompts",
    )
    c = load(run / "construction.json")
    s7 = load(HERE / "receipts/memory-program-s7/attempt1/construction.json")
    require(
        all(
            c[k] == s7[k]
            for k in ("parameters_verified", "parameter_bytes", "distinct_storages", "rules", "tensors")
        )
        and c["parameters_verified"] == 2221786584
        and c["small_to_cpu_ns"] > 0
        and c["build_verify_ns"] > 0
        and load(run / "allocator.json")["expandable_segments"] is True,
        "not real verified dense replication",
    )
    setup = {}
    metrics = {}
    ratios = {}
    clocks = []
    workers = set()
    oracles = {}
    errors = {}
    high = c["peak_allocated"]
    reserved = c["peak_reserved"]
    tr = load(run / "transfer.json")
    high = max(high, tr["peak_allocated"])
    reserved = max(reserved, tr["peak_reserved"])
    for size, order in ORDERS.items():
        wide = size == "replicated"
        boot = load(run / (size + "-setup.json"))
        shutdown = load(run / (size + "-closed.json"))
        require(
            boot["tensor_bytes"] == (962953216 if wide else 240762880)
            and 0 < boot["private_bytes"] <= (256 if wide else 64) << 20
            and boot["parameters"] == (2221786584 if wide else 149855670)
            and set(boot["buckets"]) == {"1", "4", "8", "16"}
            and shutdown["slots_freed"]
            and shutdown["graphs_freed"]
            and shutdown["lifecycle_ns"] > boot["prepare_ns"] > 0
            and idn["order"][size] == [list(v) for v in order],
            "invalid resident geometry",
        )
        setup[size] = dict(
            arena_bytes=boot["tensor_bytes"],
            private_bytes=boot["private_bytes"],
            prepare_s=boot["prepare_ns"] / 1e9,
            five_cell_lifecycle_s=shutdown["lifecycle_ns"] / 1e9,
        )
        high = max(high, boot["peak_allocated"], shutdown["peak_allocated"])
        reserved = max(reserved, boot["peak_reserved"], shutdown["peak_reserved"])
        oracle = load(run / f"{size}-closed-ref.json")
        oracles[size] = oracle
        for offer, provider in [("closed", "ref"), *order]:
            name = f"{size}-{offer}-{provider}"
            x = load(run / (name + ".json"))
            pid = x["wire"]["pid"]
            require(
                x["name"] == name
                and (x["offer"], x["provider"], x["size"]) == (offer, provider, size)
                and pid not in workers
                and pid not in (idn["server_pid"], aa[0]["pid"]),
                "native domain reused or wrong case",
            )
            workers.add(pid)
            clocks.extend([x["begin_clock"], x["begin_clock"] + x["lifecycle_ns"]])
            metrics[name] = clock_metrics(x, oracle, idn)
            high = max(high, x["peak_allocated"])
            reserved = max(reserved, x["peak_reserved"])
            if offer != "closed":
                z = load(run / (name + "-parity.json"))
                parity(z, 348)
                errors[name] = z["numeric"]
        for offer in ("spaced", "burst"):
            a, b = (metrics[f"{size}-{offer}-{k}"] for k in ("ref", "terminal"))
            speed = b["bytes_s"] / a["bytes_s"]
            first = b["first_ms"]["p95"] / a["first_ms"]["p95"]
            rss = b["m20_peak_mib"] / a["m20_peak_mib"]
            ratios[f"{size}-{offer}"] = dict(
                rate=speed,
                first_p95=first,
                m20_peak=rss,
                wire=b["wire_bytes"] / a["wire_bytes"],
                passed=speed >= 1.10 and first <= 1.10 and rss <= 1.20,
            )
    cross = load(run / "cross-size.json")
    parity(cross, 384)
    end = load(run / "complete.json")
    total = sum(m["bytes"] for m in metrics.values())
    require(
        oracles["retained"]["full"] == oracles["replicated"]["full"]
        and clocks == sorted(clocks)
        and end["steps"] == 0
        and end["head"] == head["after"]
        and end["server_pid"] == idn["server_pid"]
        and end["total_emitted"] == total == 3552
        and end["lifecycle_ns"] >= clocks[-1] - clocks[0]
        and tr["wide_drop_and_small_to_gpu_ns"] > 0,
        "changed oracle or incomplete lifecycle",
    )
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=1324 + paid,
        remaining_s=476 - paid,
        cpu_head_targets=build["targets"],
        cpu_head_replayed_steps=1,
        host_or_in_lease_updates=0,
        oracle_bytes=768,
        arrival_bytes=2784,
        total_emitted=total,
        completed_arrivals=224,
        canceled_arrivals=32,
        head_prepare_s=head["elapsed_ns"] / 1e9,
        setup=setup,
        wide_build_s=c["build_verify_ns"] / 1e9,
        small_to_cpu_s=c["small_to_cpu_ns"] / 1e9,
        small_restore_incl_wide_drop_s=tr["wide_drop_and_small_to_gpu_ns"] / 1e9,
        gpu_process_lifecycle_inside_main_s=end["lifecycle_ns"] / 1e9,
        peak_gpu_mib=high / 2**20,
        peak_reserved_mib=reserved / 2**20,
        cross_size=cross,
        arrival_differences=errors,
        metrics=metrics,
        ratios=ratios,
        functional_gate=True,
        terminal_serving_default=all(q["passed"] for q in ratios.values()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
