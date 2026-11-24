"""S13 offline verifier. Rate, latency and actual native overlap separately from prefix correctness."""

import argparse, json
from collections import Counter
from pathlib import Path
from .serving_verify import HERE, ledger, load, pct, require

ROOT = HERE / "receipts/memory-program-s13"


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-t2/verified.json")["cumulative_x47_upper"]
    require(prev == 1186, "prior charge")
    aa, good = ledger(root, 1392)
    paid = sum(x["upper"] for x in aa)
    require(len(good) == 1 and 0 < paid <= 115, "incomplete/overbudget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    s12 = HERE / "receipts/memory-program-s12/attempt1"
    orig = load(s12 / "identity.json")
    qual = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        "94 passed" in qual
        and not {"failed", "skipped"}.intersection(qual.split())
        and ref == idn["ref"] == (r / "ref").read_text().strip()
        and all(
            idn[k] == v for k, v in orig.items() if k not in ("owner", "ref", "server_pid", "generation_base")
        )
        and all(
            idn["generation_base"][k] == v for k, v in orig["generation_base"].items() if k != "model_load_s"
        ),
        "sources/panel/host",
    )
    setup = load(r / "setup.json")
    require(setup["tensor_bytes"] == 240762880 and 0 < setup["private_bytes"] <= 64 << 20, "arena")
    oracle = {}
    for b in (0, 1):
        o = load(s12 / f"{b}-b16-oracle-fused-sample.json")
        oracle.update(zip(o["train_rows"], o["prefix_digests"], strict=True))
    metrics = {}
    ratios = {}
    mm = []
    ok = True
    total = 0
    for offer in ["spaced", "burst"]:
        for sched in ["barrier", "open"]:
            name = f"{offer}-{sched}"
            x = load(r / (name + ".json"))
            rows = x["requests"]
            w = x["work"]
            domain = x["domain"]
            host = x["host"]
            batches = x["batches"]
            pos = x["positions"]
            require(
                x["offer"] == offer
                and x["schedule"] == sched
                and x["name"] == name
                and [u["row"] for u in rows] == list(range(32))
                and x["end_zero"]
                and x["shared_exits"] == [0, 0]
                and x["stale_cancels"] == 4
                and 1 <= x["peak_queue"] <= 32
                and len(set(pid for pid, rc in domain["private_closed"] if rc == 0))
                == len(domain["private_closed"])
                == domain["private_opened"]
                == 32
                and domain["live"]["appends"] == 380,
                "incomplete lifetime",
            )
            require(
                host["prefill_tokens"] == w["prefill_tokens"] == 1024
                and host["step_tokens"]
                == w["step_tokens"]
                == 316
                == sum(int(k) * v.get("real_replays", 0) for k, v in x["buckets"].items())
                and w["transfer_bytes"] == 348 * 512 * 4
                and w["closed_bindings"] == w["admissions"] == host["released"] == 32
                and len(batches) == w["batches"]
                and all(0 < len(b["rows"]) == len(set(b["rows"])) <= 16 for b in batches),
                "bad work",
            )
            wait = []
            first = []
            interval = []
            life = []
            offers_live = 0
            peak = 0
            for u in rows:
                i = u["row"]
                ts = u["acks"]
                z = oracle[i]
                require(
                    u["scheduled"] == (i // 4 * 60 if offer == "spaced" else i // 16 * 50) * 1000000
                    and u["scheduled"] <= u["offered"] <= u["opening"] <= u["admitted"] < ts[0]
                    and ts == sorted(set(ts)) == [b["ack"] for b in batches if i in b["rows"]]
                    and ts[-1] <= u["released"]
                    and len(ts) == z["length"] == (3 if i % 8 == 0 else 12)
                    and u["status"] == ("canceled" if i % 8 == 0 else "finished"),
                    "bad stream",
                )
                if any(u[k] != z[k] for k in ("byte_sha", "event_sha")):
                    mm.append((name, i))
                wait.append(u["opening"] - u["offered"])
                first.append(ts[0] - u["offered"])
                life.append(u["released"] - u["offered"])
                interval.extend(b - a for a, b in zip(ts, ts[1:]))
                offers_live += int(any(v["admitted"] <= u["offered"] < v["released"] for v in rows))
                peak = max(peak, sum(v["admitted"] <= u["admitted"] < v["released"] for v in rows))
            require(
                x["emitted"] == w["acknowledged"] == sum(len(u["acks"]) for u in rows) == 348
                and peak == domain["private_peak"] == x["peak_uniforms"] <= 16
                and offers_live > 0
                and x["lifecycle_ns"]
                >= x["wall_ns"]
                >= max(u["released"] for u in rows) - min(u["offered"] for u in rows),
                "load",
            )
            total += 348
            require(
                pos
                and len(pos) == domain["archive"]["worker_batches"]
                and sum(p["keys"] for p in pos) == domain["archive"]["worker_keys"]
                and all(
                    0 < p["start"] < p["end"]
                    and p["keys"] >= 1
                    and p["issuers"]
                    and p["issuers"] == sorted(set(p["issuers"]))
                    and all(
                        rows[i]["admitted"] < p["start"] < p["end"] < rows[i]["released"]
                        and not any(p["start"] <= a < p["end"] for a in rows[i]["acks"])
                        for i in p["issuers"]
                    )
                    for p in pos
                )
                and all(a["end"] <= b["start"] for a, b in zip(pos, pos[1:])),
                "invalid native call or issuing client advanced",
            )
            inside = lambda t: any(p["start"] <= t < p["end"] for p in pos)
            overlap = sum(inside(t) for u in rows for t in u["acks"])
            reap = sum(inside(u["released"]) for u in rows)
            opened = sum(inside(u["admitted"]) for u in rows)
            require(
                sched != "barrier" or (not overlap and not reap and not opened), "inline overlap impossible"
            )
            e = x["engine"]
            require(
                (
                    not e
                    if sched == "barrier"
                    else e["emits"] == 348
                    and e["deliver_batches"] == len(batches)
                    and e["choose_batches"] == x["controller"]["choose_batches"]
                    and e["inputs_batches"] == w["transfer_batches"]
                ),
                "partial actor work",
            )
            ms = lambda vals: {k: pct(vals, q) for k, q in [("p50", 0.5), ("p95", 0.95), ("max", 1)]}
            metrics[name] = dict(
                bytes=348,
                rate=348e9 / x["wall_ns"],
                wall_s=x["wall_ns"] / 1e9,
                lifecycle_s=x["lifecycle_ns"] / 1e9,
                setup_s=x["domain_setup_ns"] / 1e9,
                offers_while_live=offers_live,
                peak_active=peak,
                peak_waiting=x["peak_queue"],
                ack_batches=len(batches),
                ack_widths=Counter(len(b["rows"]) for b in batches),
                host_batches=w["transfer_batches"],
                controller=x["controller"],
                graph_dispatch=x["dispatch"],
                cpu_core_s=x["cpu_core_s"],
                gpu_allocated_mib=x["peak_allocated"] / 2**20,
                gpu_reserved_mib=x["peak_reserved"] / 2**20,
                queue_ms=ms(wait),
                first_ms=ms(first),
                interresponse_ms=ms(interval),
                sojourn_ms=ms(life),
                worker_s=sum(p["end"] - p["start"] for p in pos) / 1e9,
                worker_calls=len(pos),
                lf_steps=domain["archive"]["lf_steps"],
                ack_during_native=overlap,
                retired_during_native=reap,
                admitted_during_native=opened,
            )
        a, b = [metrics[f"{offer}-{s}"] for s in ("open", "barrier")]
        rat = dict(
            rate=a["rate"] / b["rate"],
            sojourn_p95=a["sojourn_ms"]["p95"] / b["sojourn_ms"]["p95"],
            inter_p95=a["interresponse_ms"]["p95"] / b["interresponse_ms"]["p95"],
            inter_p50=a["interresponse_ms"]["p50"] / b["interresponse_ms"]["p50"],
        )
        ratios[offer] = rat
        ok &= bool(
            a["ack_during_native"] > 0
            and rat["rate"] >= 1.10
            and rat["sojourn_p95"] <= 1.10
            and rat["inter_p95"] <= 0.75
            and rat["inter_p50"] <= 1.25
        )
    require(total == 1392, "incomplete")
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        total_emitted=total,
        completed=112,
        canceled=16,
        mismatches=mm,
        qualified=not mm,
        promote_default=bool(ok and not mm),
        metrics=metrics,
        open_over_barrier=ratios,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
