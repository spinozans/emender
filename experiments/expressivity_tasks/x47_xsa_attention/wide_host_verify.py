"""Offline S7 audit. No wide checkpoint, GPU, or data extraction required."""

import argparse, json
from pathlib import Path
from .cached_host_verify import check_cell, summarize
from .serving_verify import HERE, load, ledger, require

ROOT = HERE / "receipts/memory-program-s7"


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s6/verified.json")["cumulative_x47_upper"]
    require(prev == 708, "prior charge")
    aa, good = ledger(root, 8704)
    paid = sum(x["upper"] for x in aa)
    require(len(good) == 1 and paid <= 240, "incomplete/overbudget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    old = HERE / "receipts/memory-program-s5/attempt1"
    y = load(old / "identity.json")
    require(
        idn["ref"] == ref == (r / "ref").read_text().strip()
        and idn["size_order"] == ["wide", "small"]
        and all(
            idn[k] == v for k, v in y.items() if k not in ("ref", "owner", "server_pid", "generation_base")
        )
        and all(
            idn["generation_base"][k] == v for k, v in y["generation_base"].items() if k != "model_load_s"
        ),
        "changed identities",
    )
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require("74 passed" in q and "failed" not in q and "skipped" not in q, "bad CPU gate")
    c = load(r / "construction.json")
    require(
        c["parameters_verified"] == idn["wide_parameters"] == 2221786584
        and c["tensors"] == c["distinct_storages"] == sum(c["rules"].values()) == 597
        and c["parameter_bytes"] == 4 * c["parameters_verified"]
        and c["width"] == 2560
        and c["layers"] == 27,
        "incorrect construction",
    )
    dev = load(r / "device-parity.json")
    require(
        dev["compared"] == dict(hidden_all_four=7 * 2560, logp=7 * 256)
        and dev["scrubbed"]
        and dev["distinct_argmax_agree"],
        "missing oracle",
    )
    total = 0
    metrics = {}
    different = []
    paired = {}
    setups = {}
    for size in ("wide", "small"):
        s = load(r / f"{size}-setup.json")
        setups[size] = s
        large = size == "wide"
        slot = 4 * (
            27 * (2 * 128 * 256 + 4 * (256 + 256 + 512)) * (4 if large else 1)
            + (2560 if large else 640)
            + 256
        )
        require(
            s["parameters"] == (2221786584 if large else 149855670)
            and s["tensor_bytes"] == 32 * slot
            and s["tensor_bytes"] <= (1 << 30 if large else 256 << 20)
            and s["private_bytes"]
            == sum(v["private_pool_reserved"] for v in s["buckets"].values())
            <= (256 if large else 64) << 20
            and load(r / f"{size}-closed.json")["slots_freed"],
            "unbound residency",
        )
        for b in (1, 16):
            for m in ("bare", "direct", "fused-greedy", "fused-sample"):
                rr = []
                label = f"{size}-{m}"
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{label}.json")
                    rr.append(x)
                    check_cell(x, b, 32, 32, block, label)
                    h = x["host"]
                    total += x["emitted"]
                    require(
                        h["all_released_zero"]
                        and h["stats"]
                        == dict(
                            prefill_batches=1,
                            prefill_tokens=32 * b,
                            step_batches=31,
                            step_tokens=31 * b,
                            released=b,
                        )
                        and h["tensor_bytes"] == s["tensor_bytes"]
                        and h["encoded_tokens"] == 63 * b
                        and h["dispatch"] == dict(prefill=1, retirements=1),
                        "changed host work",
                    )
                    for width in ("1", "16"):
                        require(
                            h["buckets"][width]
                            == (
                                {"neutral_retire_replays": 1, "real_replays": 31}
                                if int(width) == b
                                else {"neutral_retire_replays": 1}
                            ),
                            "unexpected replay",
                        )
                    y = load(old / f"{block}-b{b}-graph-{m}.json")
                    bad = [
                        j for j in range(b) if any(x[k][j] != y[k][j] for k in ("event_sha", "generated_sha"))
                    ]
                    if bad:
                        different.append(dict(cell=x["cell"], docs=bad))
                metrics[f"b{b}/{label}"] = summarize(rr)
    for b in (1, 16):
        for m in ("bare", "direct", "fused-greedy", "fused-sample"):
            same = all(
                not any(z["cell"] == f"{block}-b{b}-{sz}-{m}" for z in different)
                for block in (0, 1)
                for sz in ("small", "wide")
            )
            v = metrics[f"b{b}/wide-{m}"]
            y = metrics[f"b{b}/small-{m}"]
            paired[f"b{b}/{m}"] = dict(
                matched=same, wide_over_small_rate=v["bytes_per_s"] / y["bytes_per_s"] if same else None
            )
    require(
        total == 8704 and load(r / "transfers.json")["after_wide_removed"]["allocated"] < 2**30,
        "lost cells or wide weights still allocated",
    )
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        total_emitted=total,
        construction=c,
        oracle=dev,
        setups=setups,
        metrics=metrics,
        mismatches=different,
        paired=paired,
        feasible=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))
