"""Offline real-host receipt audit; no model execution, filesystem maps or held-out data."""

import argparse
import json
from math import prod
from pathlib import Path
from .cached_host_verify import check_cell, summarize
from .serving_verify import HERE, ledger, load, require

ROOT = HERE / "receipts/memory-program-s4"


def audit(root=ROOT):
    prior = load(HERE / "receipts/memory-program-s3/verified.json")["cumulative_x47_upper"]
    require(prior == 470, "prior charge")
    attempts, good = ledger(root, 8704)
    paid = sum(x["upper"] for x in attempts)
    require(len(good) == 1 and 0 < paid <= 360, "incomplete/overbudget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    s1 = load(HERE / "receipts/memory-program-s1/attempt1/identity.json")
    host = idn["generation_base"]
    require(
        idn["ref"] == ref == (r / "ref").read_text().strip()
        and idn["controller_training_base_sha"] == s1["base_sha"]
        and idn["generated_length"] == 32
        and all(
            idn[k] == v
            for k, v in s1.items()
            if k not in ("owner", "ref", "server_pid", "base_sha", "generated_length")
        ),
        "changed identities",
    )
    require(
        host["precision"] == "FP32"
        and host["parameters"] == 149855670
        and host["blocks"] == 27
        and host["width"] == 640
        and host["checkpoint"] == "8a214dd37172472d60fecffc6367fa29915445fd16233dc17be6360609bbbc1c"
        and host["external_ref"] == "95709fc250357c2dd109361c353192f2aa5913f9",
        "not retained host",
    )
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require("67 passed" in q and "skipped" not in q and "failed" not in q, "CPU gate")
    dev = load(r / "device-parity.json")
    require(
        dev["checks"] == dict(hidden=12, byte_logp=12)
        and dev["all_released_zero"]
        and dev["all_argmax_match"]
        and dev["stats"]["step_tokens"] == 5,
        "CUDA gate",
    )
    shapes = dev["shapes"]
    require(
        shapes == dict(recurrent=[27, 2, 128, 256], q=[27, 256, 4], k=[27, 256, 4], v=[27, 512, 4]),
        "state shapes",
    )
    state_bytes = sum(map(prod, shapes.values())) * 4
    slot_bytes = state_bytes + 4 * (640 + 256)
    require(dev["tensor_bytes"] == 3 * slot_bytes, "arena sizing")
    by = {}
    mismatch = []
    total = 0
    for b in (1, 16):
        for m in ("bare", "direct", "fused-greedy", "fused-sample"):
            rows = {}
            for e in ("full", "cache"):
                arm = f"{e}-{m}"
                rows[e] = []
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{arm}.json")
                    check_cell(x, b, 32, 32, block, arm)
                    total += x["emitted"]
                    rows[e].append(x)
                    h = x["host"]
                    require(
                        h["encoded_tokens"] == b * (63 if e == "cache" else sum(range(32, 64))),
                        "unaccounted backbone tokens",
                    )
                    if e == "cache":
                        require(
                            h["stats"]
                            == dict(
                                prefill_batches=1,
                                prefill_tokens=32 * b,
                                step_batches=31,
                                step_tokens=31 * b,
                                released=b,
                            )
                            and h["tensor_bytes"] == b * slot_bytes <= 128 * 2**20
                            and h["all_released_zero"]
                            and h["traffic"]
                            == dict(gather_bytes=31 * b * state_bytes, scatter_bytes=32 * b * state_bytes),
                            "wrong advancement/state movement",
                        )
                by[f"b{b}/{arm}"] = summarize(rows[e])
            for x, y in zip(rows["full"], rows["cache"], strict=True):
                ids = [i for i in range(b) if any(x[k][i] != y[k][i] for k in ("generated_sha", "event_sha"))]
                if ids:
                    mismatch.append(dict(block=x["block"], b=b, arm=m, different_docs=ids))
                else:
                    require(
                        x["opcodes"] == y["opcodes"]
                        and x["statuses"] == y["statuses"]
                        and x["work"].get("logical_keys") == y["work"].get("logical_keys"),
                        "different work",
                    )
    require(total == 8704, "missing service")
    ratios = {
        m: {
            "b16_rate": by["b16/cache-" + m]["bytes_per_s"] / by["b16/full-" + m]["bytes_per_s"],
            "b1_p99": by["b1/cache-" + m]["p99_ms"] / by["b1/full-" + m]["p99_ms"],
        }
        for m in ("bare", "direct", "fused-greedy", "fused-sample")
    }
    return dict(
        ref=ref,
        attempts=attempts,
        card_s_upper=paid,
        cumulative_x47_upper=prior + paid,
        remaining_s=1800 - prior - paid,
        slot_bytes=slot_bytes,
        total_emitted=total,
        mismatches=mismatch,
        metrics=by,
        ratios=ratios,
        qualified_host=True,
        recommend_speed=not mismatch
        and all(v["b16_rate"] >= 1.2 and v["b1_p99"] <= 1.1 for v in ratios.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), allow_nan=False, indent=2))
