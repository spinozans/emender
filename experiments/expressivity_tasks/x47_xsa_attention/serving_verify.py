"""Offline, all-cell S1 latency/work/lease audit. No CUDA, index, or raw text needed."""

from __future__ import annotations
import argparse
import json
from math import ceil, isclose, isfinite
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "receipts/memory-program-s1"


def load(path):
    return json.loads(path.read_text())


def fields(path):
    return dict(row.split("=", 1) for row in path.read_text().splitlines())


def require(ok, why):
    if not ok:
        raise ValueError(why)


def pct(ns, frac):
    return sorted(ns)[ceil((len(ns) - 1) * frac)] / 1e6


def ledger(root, expected=17280, *, work_key="total_emitted", actor_key="server_pid"):
    attempts = []
    good = []
    for r in sorted(root.glob("attempt*/")):
        a, b, c = (fields(r / k) for k in ("initial", "last", "released"))
        launch = load(r / "launch.json")
        pid = launch["pid"]
        st = launch["start"]
        require(
            (a["pid"], a["starttime"], a["host"], a["adopted"]) == (str(pid), st, "lambda01", "1")
            and a["gpu"] in ("0", "1"),
            "invalid lease binding",
        )
        require(all(a[k] == b[k] for k in a if k != "heartbeat"), "owner changed")
        det = dict(s.split(":") for s in (r / "detachment").read_text().splitlines())
        require(det["Pid"].strip() == str(pid) and det["PPid"].strip() == "1", "not detached")
        upper = int(c["released_epoch"]) - int(a["created"]) + 1
        require(upper > 0, "invalid lease interval")
        attempts.append(dict(attempt=r.name, pid=pid, start=st, gpu=a["gpu"], upper=upper, rc=int(c["rc"])))
        if c["rc"] != "0":
            continue
        after = fields(r / "after")
        require(
            all(after[k] == b[k] for k in b if k != "heartbeat")
            and int(a["heartbeat"])
            < int(after["heartbeat"])
            <= int(b["heartbeat"])
            <= int(c["released_epoch"]),
            "unverified renewal/release",
        )
        obs = [json.loads(s) for s in (r / "survival.jsonl").read_text().splitlines()]
        idn = load(r / "identity.json")
        end = load(r / "progress.json")
        require(
            all(
                (o["owner"], o["ppid"], o["starttime"], o["progress"][actor_key])
                == (pid, 1, st, idn[actor_key])
                for o in obs
            ),
            "wrong live witness",
        )
        ts = [o["progress"][work_key] for o in obs]
        require(
            ts == sorted(ts) and min(ts) < max(ts) == expected and any(0 < t < expected for t in ts),
            "no advancing work",
        )
        require(
            idn[actor_key] != pid
            and all(idn["owner"][k] == a[k] for k in a if k != "heartbeat")
            and max(o["heartbeat"] for o in obs) > int(a["heartbeat"]),
            "wrong server",
        )
        require(
            end["phase"] == "complete" and end == obs[-1]["progress"] and (r / "watch-ack").exists(),
            "no observed terminal emit",
        )
        good.append(r)
    return attempts, good


def audit(root=ROOT):
    prior = load(HERE / "receipts/memory-program-p2/verified.json")["cumulative_x47_upper"]
    require(prior == 180, "changed P2 charge")
    attempts, good = ledger(root)
    paid = sum(a["upper"] for a in attempts)
    require(len(good) == 1 and 0 < paid <= 420, "missing or over-budget S1")
    folder = good[0]
    ident = load(folder / "identity.json")
    ref = ident["ref"]
    q = (root / "qualification.txt").read_text()
    require("55 passed" in q and "skipped" not in q and "failed" not in q, "bad CPU gate")
    require(
        ref == (folder / "ref").read_text().strip() == (root / "gated-ref").read_text().strip(),
        "source mismatch",
    )
    p2i = load(HERE / "receipts/memory-program-p2/attempt1/packed-b32-identity.json")
    p2 = load(HERE / "receipts/memory-program-p2/attempt1/packed-b32.json")
    require(
        ident["checkpoint"] == p2["checkpoint_sha"]
        and ident["policy_sha"] == p2["final_policy"]
        and all(ident[k] == p2i[k] for k in ("base_sha", "mount", "branch_exe_sha", "limits"))
        and ident["position_exe_sha"] == p2i["position_and_live_exe_sha"],
        "wrong model/mount",
    )
    require(
        ident["data"] == {k: v for k, v in p2i["data"].items() if k != "window129_sha"}
        and len(set(ident["prompt_sha"])) == 32
        and ident["prompt_bytes"] == 32
        and ident["generated_length"] == 64,
        "wrong prompts",
    )
    dev = load(folder / "device-parity.json")
    require(
        dev["kernel_checks"]["score"] == 4
        and dev["kernel_checks"]["score_grad"] == 16
        and dev["conditional"]["checks"]["factor_logp"] >= 100
        and dev["ready"]["delayed_grad_norm"] > 0,
        "incomplete CUDA oracle",
    )
    rows = {}
    total = 0
    by = {}
    matched = {}
    mismatches = []
    for b in (1, 16):
        arms = (
            ["direct"]
            + [f"{e}-{m}" for m in ("greedy", "sample") for e in ("packed", "fused", "ready")]
            + (["inline-sample"] if b == 16 else [])
        )
        for name in arms:
            rr = []
            for block in range(2):
                cell = f"{block}-b{b}-{name}"
                r = load(folder / (cell + ".json"))
                rows[block, b, name] = r
                rr.append(r)
                require(
                    (r["cell"], r["arm"], r["block"], r["b"], r["train_rows"], r["emitted"])
                    == (cell, name, block, b, list(range(16 * block, 16 * block + b)), 64 * b),
                    "cell geometry",
                )
                total += r["emitted"]
                lat = r["interbyte_ns"]
                flat = sum(lat, [])
                require(
                    len(lat) == b
                    and all(len(t) == 64 and min(t) > 0 and sum(t) <= r["wall_ns"] for t in lat)
                    and [t[0] for t in lat] == r["first_ns"],
                    "lost service interval",
                )
                for k, f in [("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99), ("max_ms", 1)]:
                    require(isclose(r["interbyte"][k], pct(flat, f)), "wrong tail")
                require(
                    r["peak_allocated"] > 0
                    and r["host_cpu_seconds"] >= 0
                    and r["native_cpu_seconds"] >= 0
                    and r["native_processes"] == 2 + b
                    and isclose(r["rate"], r["emitted"] * 1e9 / r["wall_ns"]),
                    "cost mismatch",
                )
                require(
                    len(r["generated_sha"]) == len(r["event_sha"]) == b
                    and all(len(h) == 64 for h in r["generated_sha"]),
                    "stream identities",
                )
                require(
                    0 <= r["acknowledged_during_positional_worker"] <= b * 64
                    and r["live"]["appends"] == b * 65,
                    "missing observation",
                )
                if name == "direct":
                    require(
                        not r["opcodes"]
                        and r["work"]["archive_keys"] > 0
                        and r["work"]["live_keys"] > 0
                        and all(h is None for h in r["event_sha"]),
                        "not direct",
                    )
                else:
                    require(
                        all(len(h) == 64 for h in r["event_sha"])
                        and r["opcodes"]["EMIT"] == 64 * b
                        and sum(r["opcodes"].values()) == r["work"]["instructions"],
                        "incomplete decisions",
                    )
                    kernel = "operand_eager_calls" if name.startswith("packed") else "operand_tiles"
                    require(
                        r["work"][kernel] >= r["work"]["choose_batches"]
                        and r["work"]["candidates"] >= r["work"][kernel],
                        "missing operand dispatch",
                    )
                    if name.startswith(("ready", "inline")):
                        require(r["ready_stats"]["emits"] == 64 * b, "ready progress")
                    if not name.startswith("ready"):
                        require(r["acknowledged_during_positional_worker"] == 0, "invalid overlap")
            ns = sum([sum(r["interbyte_ns"], []) for r in rr], [])
            sec = sum(r["wall_ns"] for r in rr) / 1e9
            n = len(ns)
            by[f"b{b}/{name}"] = dict(
                bytes=n,
                wall_s=sec,
                bytes_per_s=n / sec,
                p50_ms=pct(ns, 0.5),
                p95_ms=pct(ns, 0.95),
                p99_ms=pct(ns, 0.99),
                max_ms=pct(ns, 1),
                first_p95_ms=pct(sum([r["first_ns"] for r in rr], []), 0.95),
                cpu_core_s=sum(r["host_cpu_seconds"] + r["native_cpu_seconds"] for r in rr),
                peak_pytorch_mib=max(r["peak_allocated"] for r in rr) / 2**20,
                position_worker_s=sum(r["position_worker_ns"] for r in rr) / 1e9,
                emitted_during_worker=sum(r["acknowledged_during_positional_worker"] for r in rr),
            )
        for mode in ("greedy", "sample"):
            same = True
            for block in range(2):
                base = rows[block, b, "packed-" + mode]
                for variant in (
                    ["fused", "ready", "inline"] if b == 16 and mode == "sample" else ["fused", "ready"]
                ):
                    r = rows[block, b, variant + "-" + mode]
                    diff = [
                        j
                        for j in range(b)
                        if any(r[k][j] != base[k][j] for k in ("generated_sha", "event_sha"))
                    ]
                    if diff:
                        same = False
                        mismatches.append(
                            dict(block=block, b=b, mode=mode, variant=variant, different_docs=diff)
                        )
            matched[f"b{b}/{mode}"] = same
    require(total == 17280, "missing cells")
    contrast = {}
    for mode in ("greedy", "sample"):

        def metric(b, name, k):
            return by[f"b{b}/{name}-{mode}"][k]

        rate = metric(16, "ready", "bytes_per_s") / metric(16, "packed", "bytes_per_s")
        tail = metric(1, "ready", "p99_ms") / metric(1, "packed", "p99_ms")
        same = all(matched[f"b{b}/{mode}"] for b in (1, 16))
        contrast[mode] = dict(
            matched=same,
            b16_ready_over_packed_rate=rate,
            b16_fused_over_packed_rate=metric(16, "fused", "bytes_per_s")
            / metric(16, "packed", "bytes_per_s"),
            b1_ready_over_packed_p99=tail,
            passes=same and rate >= 1.2 and tail <= 1.1,
        )
    return dict(
        ref=ref,
        attempts=attempts,
        card_s_upper=paid,
        cumulative_x47_upper=prior + paid,
        remaining_s=1800 - prior - paid,
        total_emitted=total,
        metrics=by,
        matched=matched,
        mismatches=mismatches,
        contrast=contrast,
        promote=all(v["passes"] for v in contrast.values()),
        qualification=dev,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
