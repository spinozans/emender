"""S11 offline audit. Verify exact widths actually launched before interpreting aggregate clocks."""

import argparse, json
from pathlib import Path
from collections import Counter
from .cached_host_verify import check_cell, summarize
from .serving_verify import HERE, load, ledger, require

ROOT = HERE / "receipts/memory-program-s11"
PLANS = {1: [1], 3: [1, 1, 1], 7: [4, 1, 1, 1], 13: [8, 4, 1], 15: [8, 4, 1, 1, 1], 16: [16]}


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s10/verified.json")["cumulative_x47_upper"]
    require(prev == 1017, "prior charge")
    aa, good = ledger(root, 5280)
    paid = sum(x["upper"] for x in aa)
    require(len(good) == 1 and 0 < paid <= 150, "incomplete/overbudget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    old = load(HERE / "receipts/memory-program-s8/attempt1/identity.json")
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        "84 passed" in q
        and not {"failed", "skipped"}.intersection(q.split())
        and idn["ref"] == ref == (r / "ref").read_text().strip()
        and idn["generated_length"] == 12
        and idn["step_widths"] == [1, 4, 8, 16]
        and idn["cover_launch_limit"] == 6
        and all(
            idn[k] == v
            for k, v in old.items()
            if k not in ("ref", "owner", "server_pid", "generated_length", "generation_base")
        )
        and all(
            idn["generation_base"][k] == v for k, v in old["generation_base"].items() if k != "model_load_s"
        ),
        "changed identities",
    )
    s = load(r / "setup.json")
    require(
        s["tensor_bytes"] == 240762880
        and s["private_bytes"] == sum(g["private_pool_reserved"] for g in s["buckets"].values()) <= 64 << 20
        and all(s["plans"][str(b)] == p for b, p in PLANS.items()),
        "wrong resident width set",
    )
    p = load(r / "device-parity.json")
    require(
        p["all_argmax"]
        and p["slots_reused_and_scrubbed"]
        and p["replays"] == {str(w): sum(v.count(w) for v in PLANS.values()) for w in (1, 4, 8, 16)}
        and p["elements"]["recurrent"] == 8 * 16 * 27 * 2 * 128 * 256
        and p["elements"]["output_base"] == 55 * 256,
        "missing parity",
    )
    metrics = {}
    mis = []
    ratios = {}
    total = 0
    for b, plan in PLANS.items():
        for m in ("direct", "fused-sample"):
            ys = []
            matched = True
            for kind in ("exact", "cover"):
                label = f"{kind}-{m}"
                rr = []
                exact = len(plan) == 1
                cover = not exact and kind == "cover"
                executed = plan if cover or exact else []
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{label}.json")
                    check_cell(x, b, 12, 32, block, label)
                    rr.append(x)
                    total += x["emitted"]
                    h = x["host"]
                    d = dict(prefill=1, retirements=1)
                    if cover:
                        d.update(covered_steps=11, cover_replays=11 * len(plan))
                    elif not exact:
                        d.update(eager_width_miss=11)
                    require(
                        h["dispatch"] == d
                        and h["stats"]
                        == dict(
                            prefill_batches=1,
                            prefill_tokens=32 * b,
                            step_batches=11,
                            step_tokens=11 * b,
                            released=b,
                        )
                        and h["encoded_tokens"] == 43 * b
                        and h["all_released_zero"]
                        and h["tensor_bytes"] == s["tensor_bytes"],
                        "changed host work",
                    )
                    require(
                        h["buckets"]
                        == {
                            str(w): dict(
                                neutral_retire_replays=1,
                                **({"real_replays": 11 * executed.count(w)} if w in executed else {}),
                            )
                            for w in (1, 4, 8, 16)
                        },
                        "wrong width replay count",
                    )
                    require(x["policy_device"] == ("cuda:0" if m == "direct" else "cpu"), "wrong placement")
                    if kind == "exact":
                        ys.append(x)
                    else:
                        bad = [
                            i
                            for i in range(b)
                            if any(x[k][i] != ys[block][k][i] for k in ("generated_sha", "event_sha"))
                        ]
                        if bad:
                            mis.append(dict(cell=x["cell"], docs=bad))
                            matched = False
                v = summarize(rr)
                v["release_and_teardown_s"] = (
                    sum(x["lifetime_ns"] - x["setup_prompt_ns"] - x["wall_ns"] for x in rr) / 1e9
                )
                metrics[f"b{b}/{label}"] = v
            ratios[f"b{b}/{m}"] = (
                metrics[f"b{b}/cover-{m}"]["bytes_per_s"] / metrics[f"b{b}/exact-{m}"]["bytes_per_s"]
                if matched
                else None
            )
    require(total == 5280, "lost emissions")
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        setup=s,
        oracle=p,
        total_emitted=total,
        metrics=metrics,
        mismatches=mis,
        ratios=ratios,
        promote=not mis
        and all(
            ratios[f"b{b}/{m}"] >= (0.9 if len(v) == 1 else 1.2)
            for b, v in PLANS.items()
            for m in ("direct", "fused-sample")
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
