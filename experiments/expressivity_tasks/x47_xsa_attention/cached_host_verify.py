"""Verify capacity, complete observations, trajectories and inclusive clocks without a GPU."""

import argparse
import json
from math import isclose
from pathlib import Path
from .serving_verify import HERE, ledger, load, pct, require

ROOT = HERE / "receipts/memory-program-s3"


def check_cell(row, b, steps, prompt, block, label):
    x = row
    require(
        (x["cell"], x["block"], x["arm"], x["b"], x["emitted"], x["train_rows"])
        == (f"{block}-b{b}-{label}", block, label, b, b * steps, list(range(16 * block, 16 * block + b))),
        "cell binding",
    )
    ts = x["interbyte_ns"]
    require(
        len(ts) == b
        and all(len(r) == steps and min(r) > 0 and sum(r) <= x["wall_ns"] for r in ts)
        and [t[0] for t in ts] == x["first_ns"],
        "missing interval",
    )
    require(
        all(
            isclose(x["interbyte"][k], pct(sum(ts, []), q))
            for k, q in [("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99), ("max_ms", 1)]
        )
        and isclose(x["rate"], x["emitted"] * 1e9 / x["wall_ns"]),
        "invalid clocks",
    )
    require(
        x["backbone_batches"] == steps
        and len(x["generated_sha"]) == len(x["event_sha"]) == b
        and all(len(h) == 64 for h in x["generated_sha"]),
        "not closed load",
    )
    if label.endswith("bare"):
        require(
            x["native_processes"] == 0
            and x["position_worker_ns"] == 0
            and not x["work"]
            and not x["opcodes"]
            and all(h is None for h in x["event_sha"]),
            "not bare",
        )
        return
    require(
        x["live"]["appends"] == b * (steps + 1) and x["native_processes"] == b + 2,
        "incomplete native service",
    )
    if label.endswith("direct"):
        require(
            not x["opcodes"]
            and x["work"]["archive_keys"] > 0
            and x["work"]["live_keys"] > 0
            and all(h is None for h in x["event_sha"]),
            "bad direct",
        )
    else:
        require(
            x["opcodes"]["EMIT"] == b * steps
            and x["work"]["instructions"] == sum(x["opcodes"].values())
            and all(len(h) == 64 for h in x["event_sha"]),
            "lost decisions",
        )


def summarize(rows):
    def add(k):
        return sum(x[k] for x in rows)

    ns = sum((sum(x["interbyte_ns"], []) for x in rows), [])
    sec = add("wall_ns") / 1e9
    return dict(
        bytes=len(ns),
        wall_s=sec,
        bytes_per_s=len(ns) / sec,
        p50_ms=pct(ns, 0.5),
        p95_ms=pct(ns, 0.95),
        p99_ms=pct(ns, 0.99),
        max_ms=pct(ns, 1),
        first_p95_ms=pct(sum((x["first_ns"] for x in rows), []), 0.95),
        setup_prompt_s=add("setup_prompt_ns") / 1e9,
        host_submit_s=add("backbone_submit_ns") / 1e9,
        cpu_core_s=add("native_cpu_seconds") + add("host_cpu_seconds"),
        worker_s=add("position_worker_ns") / 1e9,
        peak_gpu_mib=max(x["peak_allocated"] for x in rows) / 2**20,
        arena_mib=max(x["host"]["tensor_bytes"] for x in rows) / 2**20,
        encoded_tokens=sum(x["host"]["encoded_tokens"] for x in rows),
    )


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s2/verified.json")["cumulative_x47_upper"]
    require(prev == 393, "changed ledger")
    attempts, good = ledger(root, 13056)
    paid = sum(x["upper"] for x in attempts)
    require(len(good) == 1 and 0 < paid <= 200, "incomplete/overbudget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    s1 = HERE / "receipts/memory-program-s1/attempt1"
    base = load(s1 / "identity.json")
    require(
        idn["ref"] == ref == (r / "ref").read_text().strip()
        and all(idn[k] == v for k, v in base.items() if k not in ("owner", "ref", "server_pid")),
        "identity changes",
    )
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require("65 passed" in q and "skipped" not in q and "failed" not in q, "CPU gate")
    dev = load(r / "device-parity.json")
    require(
        dev["all_released_zero"] and min(dev["checks"].values()) == 13 and dev["stats"]["step_tokens"] == 6,
        "unfinished device oracle",
    )
    by = {}
    mismatch = []
    total = 0
    for b in (1, 16):
        for arm in ("direct", "fused-greedy", "fused-sample"):
            for host in ("full", "cache"):
                label = f"{host}-{arm}"
                rows = []
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{label}.json")
                    check_cell(x, b, 64, 32, block, label)
                    rows.append(x)
                    total += x["emitted"]
                    y = load(s1 / f"{block}-b{b}-{arm}.json")
                    diff = [
                        i for i in range(b) if any(x[k][i] != y[k][i] for k in ("event_sha", "generated_sha"))
                    ]
                    if diff:
                        mismatch.append(dict(cell=x["cell"], docs=diff))
                    else:
                        require(
                            x["opcodes"] == y["opcodes"]
                            and x["work"]["logical_keys"] == y["work"]["logical_keys"],
                            "different logical work",
                        )
                    h = x["host"]
                    require(
                        h["encoded_tokens"] == b * (95 if host == "cache" else sum(range(32, 96))),
                        "extra/missing backbone tokens",
                    )
                    if host == "cache":
                        require(
                            h["tensor_bytes"] == b * 526336
                            and h["all_released_zero"]
                            and h["stats"]
                            == dict(
                                prefill_batches=1,
                                prefill_tokens=32 * b,
                                step_batches=63,
                                step_tokens=63 * b,
                                released=b,
                            ),
                            "bad arena/advancement",
                        )
                by[f"b{b}/{label}"] = summarize(rows)
    micro = load(r / "host-only.json")
    require(
        len(micro) == 4
        and all(
            t["positions"] == 64 * t["b"]
            and t["wall_ns"] > t["prefill_ns"] > 0
            and t["encoded_tokens"] == t["b"] * (95 if t["kind"] == "cache" else 4064)
            for t in micro
        ),
        "bad host-only clocks",
    )
    require(total == 13056, "lost exposure")
    ratios = {
        arm: {
            "b16_rate": by["b16/cache-" + arm]["bytes_per_s"] / by["b16/full-" + arm]["bytes_per_s"],
            "b1_p99": by["b1/cache-" + arm]["p99_ms"] / by["b1/full-" + arm]["p99_ms"],
        }
        for arm in ("direct", "fused-greedy", "fused-sample")
    }
    return dict(
        ref=ref,
        attempts=attempts,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        total_emitted=total,
        mismatches=mismatch,
        host_only=micro,
        metrics=by,
        ratios=ratios,
        promote_c14=not mismatch
        and all(t["b16_rate"] >= 1.10 and t["b1_p99"] <= 1.10 for t in ratios.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))
