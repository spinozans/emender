"""Audit offered timestamps, admission bounds, prefix identities and complete binding/host lifetimes."""

import argparse, json
from collections import Counter
from pathlib import Path
from .serving_verify import HERE, ledger, load, require, pct
from .cached_host_verify import check_cell

ROOT = HERE / "receipts/memory-program-s12"


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s11/verified.json")["cumulative_x47_upper"]
    require(prev == 1120, "prior charge")
    aa, good = ledger(root, 2160)
    paid = sum(x["upper"] for x in aa)
    require(len(good) == 1 and 0 < paid <= 150, "incomplete/overbudget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    s11 = HERE / "receipts/memory-program-s11/attempt1"
    old = load(s11 / "identity.json")
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        "88 passed" in q
        and not {"failed", "skipped"}.intersection(q.split())
        and ref == (r / "ref").read_text().strip() == idn["ref"]
        and all(
            idn[k] == v
            for k, v in old.items()
            if k not in ("ref", "owner", "server_pid", "generation_base", "cover_launch_limit")
        )
        and all(
            idn["generation_base"][k] == v for k, v in old["generation_base"].items() if k != "model_load_s"
        )
        and (idn["active_cap"], idn["waiting_cap"]) == (16, 32),
        "changed identities",
    )
    setup = load(r / "setup.json")
    require(setup["tensor_bytes"] == 240762880 and 0 < setup["private_bytes"] <= 64 << 20, "resident cap")
    oracle = {}
    total = 0
    metrics = {}
    mismatch = []
    for block in (0, 1):
        for mode, m in [("direct", "direct"), ("sample", "fused-sample")]:
            x = load(r / f"{block}-b16-oracle-{m}.json")
            check_cell(x, 16, 12, 32, block, f"oracle-{m}")
            y = load(s11 / f"{block}-b16-cover-{m}.json")
            require(
                all(x[k] == y[k] for k in ("event_sha", "generated_sha", "opcodes")),
                "reference differs from S11",
            )
            require(len(x["prefix_digests"]) == 16, "lost oracle stream")
            total += x["emitted"]
            for i, u in zip(x["train_rows"], x["prefix_digests"], strict=True):
                require(u["length"] == (3 if i % 8 == 0 else 12), "changed cancellation")
                oracle[mode, i] = u
    for offer, mode in [("spaced", "direct"), ("burst", "sample"), ("spaced", "sample"), ("burst", "direct")]:
        name = f"{offer}-{mode}"
        x = load(r / (name + ".json"))
        rows = x["requests"]
        ticks = x["ticks"]
        require(
            x["name"] == name
            and [z["row"] for z in rows] == list(range(32))
            and x["peak_queue"] <= 32
            and x["end_zero"]
            and x["stale_cancels"] == 4
            and x["shared_exits"] == [0, 0],
            "bad finite service",
        )
        n = sum(len(z["acks"]) for z in rows)
        total += n
        d = x["domain"]
        w = x["work"]
        h = x["host"]
        require(
            d["private_opened"] == len(d["private_closed"]) == 32
            and len(set(v for v, rc in d["private_closed"] if rc == 0)) == 32
            and d["live"]["appends"] == 380
            and x["peak_uniforms"] == (0 if mode == "direct" else d["private_peak"]),
            "bad domain/RNG cleanup",
        )
        require(
            x["emitted"] == n == 348
            and h["prefill_tokens"] == w["prefill_tokens"] == 1024
            and h["step_tokens"] == w["step_tokens"] == 316
            and sum(int(k) * v.get("real_replays", 0) for k, v in x["buckets"].items()) == 316
            and h["released"] == w["admissions"] == w["closed_bindings"] == 32
            and n == w["acknowledged"]
            and len(ticks) == w["batches"]
            and w.get("transfer_bytes", 0) == (0 if mode == "direct" else 348 * 512 * 4),
            "wrong work",
        )
        interval = []
        wait = []
        jitter = []
        sojourn = []
        first = []
        offer_while = inside = 0
        for z in rows:
            i = z["row"]
            ts = z["acks"]
            u = oracle[mode, i]
            require(
                z["scheduled"] == (i // 4 * 60 if offer == "spaced" else i // 16 * 50) * 1000000
                and z["scheduled"] <= z["offered"] <= z["opening"] <= z["admitted"] < ts[0]
                and ts == sorted(set(ts))
                and ts[-1] <= z["released"]
                and len(ts) == u["length"]
                and z["status"] == ("canceled" if i % 8 == 0 else "finished")
                and ts == [t["ack"] for t in ticks if i in t["rows"]],
                "arrival/lifetime clock",
            )
            if z["byte_sha"] != u["byte_sha"] or z["event_sha"] != u["event_sha"]:
                mismatch.append(dict(trial=name, row=i))
            jitter.append(z["offered"] - z["scheduled"])
            wait.append(z["opening"] - z["offered"])
            first.append(ts[0] - z["offered"])
            interval.extend(b - a for a, b in zip(ts, ts[1:]))
            sojourn.append(z["released"] - z["offered"])
            offer_while += int(any(y["admitted"] <= z["offered"] < y["released"] for y in rows))
            inside += int(any(t["start"] <= z["offered"] < t["ack"] for t in ticks))
        peak = max(sum(y["admitted"] <= z["admitted"] < y["released"] for y in rows) for z in rows)
        require(
            1 <= peak == d["private_peak"] <= 16
            and all(0 < len(t["rows"]) == len(set(t["rows"])) <= 16 and t["start"] < t["ack"] for t in ticks)
            and offer_while > 0
            and x["lifecycle_ns"]
            >= x["wall_ns"]
            >= max(z["released"] for z in rows) - min(z["offered"] for z in rows),
            "unverified load",
        )
        ms = lambda vals: {k: pct(vals, q) for k, q in [("p50", 0.5), ("p95", 0.95), ("max", 1)]}
        metrics[name] = dict(
            bytes=n,
            wall_s=x["wall_ns"] / 1e9,
            bytes_s=n * 1e9 / x["wall_ns"],
            domain_setup_s=x["domain_setup_ns"] / 1e9,
            lifecycle_s=x["lifecycle_ns"] / 1e9,
            offers_during_live=offer_while,
            offers_inside_tick=inside,
            peak_active=peak,
            peak_waiting=x["peak_queue"],
            jitter_ms=ms(jitter),
            queue_delay_ms=ms(wait),
            arrival_to_first_ms=ms(first),
            interresponse_ms=ms(interval),
            sojourn_ms=ms(sojourn),
            tick_widths=dict(Counter(len(t["rows"]) for t in ticks)),
            peak_gpu_mib=x["peak_allocated"] / 2**20,
            cpu_core_s=x["cpu_core_s"],
        )
    require(total == 2160, "incomplete exposure")
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        oracle_bytes=768,
        arrival_bytes=1392,
        total_emitted=total,
        completed=112,
        canceled=16,
        metrics=metrics,
        mismatches=mismatch,
        qualified=not mismatch,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
