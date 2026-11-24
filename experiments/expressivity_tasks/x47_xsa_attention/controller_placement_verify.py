"""Offline per-trajectory device-placement and transfer/cost audit."""

import argparse, json
from pathlib import Path
from .cached_host_verify import check_cell, summarize
from .serving_verify import HERE, load, ledger, require

ROOT = HERE / "receipts/memory-program-s8"


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s7/verified.json")["cumulative_x47_upper"]
    require(prev == 779, "prior charge")
    aa, good = ledger(root, 6528)
    paid = sum(x["upper"] for x in aa)
    require(len(good) == 1 and 0 < paid <= 150, "incomplete/overbudget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    old = HERE / "receipts/memory-program-s7/attempt1"
    base = load(old / "identity.json")
    require(
        idn["ref"] == ref == (r / "ref").read_text().strip()
        and all(
            idn[k] == v
            for k, v in base.items()
            if k
            not in (
                "ref",
                "owner",
                "server_pid",
                "generation_base",
                "profile",
                "wide_parameters",
                "size_order",
            )
        )
        and all(
            idn["generation_base"][k] == v for k, v in base["generation_base"].items() if k != "model_load_s"
        ),
        "wrong identities",
    )
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require("76 passed" in q and "skipped" not in q and "failed" not in q, "wrong CPU gate")
    dev = load(r / "device-parity.json")
    require(
        dev["single_packet_adjoint"]
        and dev["delayed_grad_norm"] > 0
        and dev["checks"]["factor_logp"] == 103
        and dev["checks"]["parameter_grad"] == 35,
        "missing cross-device conformance",
    )
    by = {}
    dif = []
    ratios = {}
    total = 0
    for b in (1, 16):
        for m in ("greedy", "sample"):
            group = {}
            for e in ("gpu-triton", "gpu-eager", "cpu-eager"):
                rr = []
                arm = f"{e}-{m}"
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{arm}.json")
                    rr.append(x)
                    check_cell(x, b, 32, 32, block, arm)
                    total += x["emitted"]
                    y = load(old / f"{block}-b{b}-small-fused-{m}.json")
                    cpu = e == "cpu-eager"
                    t = x["transfer"]
                    require(
                        x["host_device"] == "cuda:0"
                        and x["policy_device"] == ("cpu" if cpu else "cuda:0")
                        and (t["copies"], t["bytes"]) == ((32, 32 * b * 512 * 4) if cpu else (0, 0))
                        and (t["blocking_ns"] > 0) == cpu,
                        "bad placement/traffic",
                    )
                    require(
                        x["host"] == y["host"]
                        and x["work"].get("operand_tiles" if e == "gpu-triton" else "operand_eager_calls", 0)
                        > 0,
                        "wrong engine",
                    )
                    bad = [
                        i for i in range(b) if any(x[k][i] != y[k][i] for k in ("generated_sha", "event_sha"))
                    ]
                    if bad:
                        dif.append(dict(cell=x["cell"], docs=bad))
                v = summarize(rr)
                v.update(
                    transfer_bytes=sum(x["transfer"]["bytes"] for x in rr),
                    transfer_blocking_s=sum(x["transfer"]["blocking_ns"] for x in rr) / 1e9,
                )
                by[f"b{b}/{arm}"] = group[e] = v
            t, e, c = [group[k] for k in ("gpu-triton", "gpu-eager", "cpu-eager")]
            ratios[f"b{b}/{m}"] = dict(
                cpu_over_best_gpu_rate=c["bytes_per_s"] / max(t["bytes_per_s"], e["bytes_per_s"]),
                cpu_over_lower_gpu_p99=c["p99_ms"] / min(t["p99_ms"], e["p99_ms"]),
            )
    require(total == 6528, "missing cells")
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        total_emitted=total,
        mismatches=dif,
        metrics=by,
        ratios=ratios,
        promote=not dif
        and all(
            v["cpu_over_best_gpu_rate"] >= 1.2 and v["cpu_over_lower_gpu_p99"] <= 1.1 for v in ratios.values()
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))
