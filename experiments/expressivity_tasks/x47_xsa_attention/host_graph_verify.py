"""Audit every graph service cell, separately pricing startup and steady requests."""

import argparse, json
from pathlib import Path
from .cached_host_verify import check_cell, summarize
from .serving_verify import HERE, ledger, load, require

ROOT = HERE / "receipts/memory-program-s5"
S4 = HERE / "receipts/memory-program-s4"


def audit(root=ROOT):
    s4 = load(S4 / "verified.json")
    prev = s4["cumulative_x47_upper"]
    require(prev == 594, "prior charge")
    slot = s4["slot_bytes"]
    base = S4 / next(x["attempt"] for x in s4["attempts"] if x["rc"] == 0)
    attempts, good = ledger(root, 8704)
    paid = sum(x["upper"] for x in attempts)
    require(len(good) == 1 and 0 < paid <= 260, "incomplete/overbudget S5")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    y = load(base / "identity.json")
    require(
        idn["ref"] == ref == (r / "ref").read_text().strip()
        and all(
            idn[k] == v for k, v in y.items() if k not in ("ref", "owner", "server_pid", "generation_base")
        )
        and all(
            idn["generation_base"][k] == v for k, v in y["generation_base"].items() if k != "model_load_s"
        ),
        "wrong identities",
    )
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require("69 passed" in q and "skipped" not in q and "failed" not in q, "bad CPU gate")
    dev = load(r / "device-parity.json")
    require(
        dev["scrubbed"]
        and dev["work"]["width_miss"] == 1
        and dev["work"]["real_replays"] == 3
        and dev["compared"]["arena_recurrent"] >= 3 * 27 * 2 * 128 * 256
        and dev["compared"]["stock_base"] >= 256,
        "missing CUDA oracle",
    )
    by = {}
    mismatch = []
    total = 0
    ratios = {}
    for b in (1, 16):
        for m in ("bare", "direct", "fused-greedy", "fused-sample"):
            pair = {}
            for e in ("eager", "graph"):
                arm = f"{e}-{m}"
                rr = []
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{arm}.json")
                    rr.append(x)
                    check_cell(x, b, 32, 32, block, arm)
                    total += x["emitted"]
                    y = load(base / f"{block}-b{b}-cache-{m}.json")
                    h = x["host"]
                    g = h["dispatch"]
                    s = h["setup"]
                    require(
                        h["encoded_tokens"] == b * 63
                        and h["stats"] == y["host"]["stats"]
                        and h["all_released_zero"]
                        and h["tensor_bytes"] == 2 * b * slot <= 256 * 2**20
                        and h["traffic"] == y["host"]["traffic"],
                        "changed host work",
                    )
                    require(
                        x["lifetime_ns"] >= x["wall_ns"] + x["setup_prompt_ns"] + h["retire_verify_ns"]
                        and 0 < s["warmup_ns"] < x["setup_prompt_ns"]
                        and 0 < s["capture_and_init_ns"] < x["setup_prompt_ns"]
                        and s["warmup_steps"] == 2,
                        "bad lifecycle clock",
                    )
                    require(
                        g
                        == (
                            {"prefill": 1, "real_replays": 31, "neutral_retire_replays": 1}
                            if e == "graph"
                            else {"prefill": 1, "eager_step": 31}
                        )
                        and s["captured"] == (e == "graph")
                        and (
                            (s["private_pool_reserved"] >= s["private_pool_allocated"] > 0)
                            if e == "graph"
                            else s["private_pool_reserved"] == s["private_pool_allocated"] == 0
                        ),
                        "wrong dispatch/memory",
                    )
                    diff = [
                        j for j in range(b) if any(x[k][j] != y[k][j] for k in ("generated_sha", "event_sha"))
                    ]
                    if diff:
                        mismatch.append(dict(cell=x["cell"], docs=diff))
                v = summarize(rr)
                life = sum(x["lifetime_ns"] for x in rr) / 1e9
                v.update(
                    lifetime_s=life,
                    inclusive_bytes_per_s=v["bytes"] / life,
                    warm_s=sum(x["host"]["setup"]["warmup_ns"] for x in rr) / 1e9,
                    capture_init_s=sum(x["host"]["setup"]["capture_and_init_ns"] for x in rr) / 1e9,
                    retire_verify_s=sum(x["host"]["retire_verify_ns"] for x in rr) / 1e9,
                    max_private_pool_mib=max(x["host"]["setup"]["private_pool_reserved"] for x in rr) / 2**20,
                    max_reserved_mib=max(x["peak_reserved"] for x in rr) / 2**20,
                )
                pair[e] = v
                by[f"b{b}/{arm}"] = v
            a, c = pair["eager"], pair["graph"]
            saving = a["wall_s"] - c["wall_s"]
            overhead = (c["lifetime_s"] - c["wall_s"]) - (a["lifetime_s"] - a["wall_s"])
            ratios[f"b{b}/{m}"] = dict(
                rate=c["bytes_per_s"] / a["bytes_per_s"],
                inclusive_rate=c["inclusive_bytes_per_s"] / a["inclusive_bytes_per_s"],
                p99=c["p99_ms"] / a["p99_ms"],
                descriptive_break_even_responses_per_client=(
                    max(0, overhead) * 32 / saving if saving > 0 else None
                ),
            )
    require(total == 8704, "missing traffic")
    promote = not mismatch and all(
        ratios["b16/" + m]["rate"] >= 1.20 and ratios["b1/" + m]["p99"] <= 1.10
        for m in ("bare", "direct", "fused-greedy", "fused-sample")
    )
    return dict(
        ref=ref,
        attempts=attempts,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        total_emitted=total,
        metrics=by,
        ratios=ratios,
        mismatches=mismatch,
        promote_envelope=promote,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))
