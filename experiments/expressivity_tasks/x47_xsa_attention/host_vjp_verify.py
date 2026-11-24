"""Offline T1 work, identity, finite-gradient and measured memory-relief audit."""

import argparse, json, math
from pathlib import Path
from .serving_verify import HERE, ledger, load, require

ROOT = HERE / "receipts/memory-program-t1"


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s8/verified.json")["cumulative_x47_upper"]
    require(prev == 842, "wrong prior")
    aa, good = ledger(root, 9216, work_key="total_targets", actor_key="trainer_pid")
    paid = sum(x["upper"] for x in aa)
    require(len(good) == 1 and 0 < paid <= 220, "missing or overbudget T1")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    s8 = load(HERE / "receipts/memory-program-s8/attempt1/identity.json")
    train = load(HERE / "receipts/memory-program-p2/attempt1/packed-b32-identity.json")["data"]
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        "80 passed" in q
        and "skipped" not in q
        and "failed" not in q
        and idn["ref"] == ref == (r / "ref").read_text().strip()
        and idn["data"] == train
        and all(
            idn[k] == s8[k]
            for k in (
                "checkpoint",
                "policy_sha",
                "mount",
                "branch_exe_sha",
                "position_exe_sha",
                "limits",
                "device",
                "torch",
            )
        )
        and all(
            idn["host"][k] == s8["generation_base"][k] for k in ("checkpoint", "external_ref", "files", "fla")
        ),
        "changed weights/data/gate",
    )
    require(
        (
            idn["parameters"],
            idn["precision"],
            idn["microbatch"],
            idn["seed"],
            idn["policy_device"],
            idn["host_device"],
            idn["chunk_sha"],
        )
        == (
            149855670,
            "BF16 autocast, FP32 master",
            4,
            7809,
            "cpu",
            "cuda:0",
            "51e2037b204dbd2cc9e2b9c9bf301adcb770bb23d4a825dd23032b222a68a1b7",
        ),
        "wrong host kernel/bridge",
    )
    metrics = {}
    total = 0
    proof = {}
    ratios = {}
    for n, ix in [(4, list(range(4, 8))), (32, list(range(32)))]:
        tg = n * 128
        eq = load(r / f"b{n}-equivalence.json")
        log = eq["logical"]
        require(
            log["train_rows"] == ix
            and log["window_sha"] == [train["window129_sha"][i] for i in ix]
            and len(log["event_sha"]) == len(log["dispatch_sha"]) == 64,
            "wrong sequence",
        )
        num = eq["numeric"]
        require(
            {k: v["count"] for k, v in num.items()}
            == dict(byte_logp=tg * 256, joint_logp=tg, objective=1, nll=1)
            and all(math.isfinite(v["max_abs"]) for v in num.values()),
            "wrong numeric coverage",
        )
        gs = eq["gradients"]
        require(
            all(
                0 <= v["error"] <= 2e-4 + 2e-4 * v["ref_max"] and math.isfinite(v["ref_max"])
                for v in gs.values()
            ),
            "invalid adjoint",
        )
        proof[n] = dict(
            checked_elements={
                group: sum(v["n"] for k, v in gs.items() if k.startswith(group + "/"))
                for group in ("host", "policy")
            },
            numeric=num,
            max_gradient_error=max(v["error"] for v in gs.values()),
            unused=eq["unused"],
        )
        require(
            proof[n]["checked_elements"]["host"] == 149855670
            and sum(k.startswith("host/") for k in gs) == 597,
            "incomplete host adjoints",
        )
        d = None
        for arm in ("direct", "remat"):
            x = load(r / f"b{n}-{arm}.json")
            t = x["times"]
            w = x["host_work"]
            m = x["memory"]
            require(
                x["logical"] == log
                and x["mode"] == "fresh_sample"
                and x["optimizer_steps"] == 0
                and x["targets"] == tg
                and x["activity"]["opcodes"]["EMIT"] == tg
                and w["forward_tokens"] == tg
                and w["forward_batches"] == n // 4
                and w.get("recompute_tokens", 0) == (tg if arm == "remat" else 0)
                and w.get("recompute_batches", 0) == (n // 4 if arm == "remat" else 0),
                "wrong execution",
            )
            require(
                (x["saves_at_rollout"].get("tape_logical_bytes", 0) == 0) == (arm == "remat")
                and x["saves_total"]["tape_logical_bytes"] > 0
                and m["initial"] <= m["post_rollout"] <= m["peak_allocated"] <= m["peak_reserved"]
                and 0 < t["host_first_pass_ns"] <= t["forward_ns"]
                and t["gradient_wall_ns"] == t["forward_ns"] + t["backward_ns"]
                and math.isclose(x["targets_s"], tg * 1e9 / t["gradient_wall_ns"]),
                "wrong work or timing",
            )
            require(
                all(
                    g["used_elements"] == proof[n]["checked_elements"][k]
                    and 0 < g["nonzero_elements"] <= g["used_elements"]
                    for k, g in x["gradients"].items()
                ),
                "wrong gradient counts",
            )
            total += tg
            metrics[x["cell"]] = {
                k: x[k]
                for k in (
                    "activity",
                    "dispatches",
                    "nll",
                    "objective",
                    "targets_s",
                    "times",
                    "core_s",
                    "rss_kib",
                    "children_rss_kib",
                    "memory",
                    "gradients",
                    "saves_at_rollout",
                    "saves_total",
                )
            }
            if d is not None:
                require(
                    d["activity"] == x["activity"] and d["dispatches"] == x["dispatches"],
                    "changed native work",
                )
                ratios[n] = dict(
                    post_rollout_allocated=m["post_rollout"] / d["memory"]["post_rollout"],
                    gradient_wall=t["gradient_wall_ns"] / d["times"]["gradient_wall_ns"],
                    peak_allocated=m["peak_allocated"] / d["memory"]["peak_allocated"],
                )
            d = x
    require(total == 9216, "wrong targets")
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        total_gradient_targets=total,
        optimizer_steps=0,
        proof=proof,
        metrics=metrics,
        remat_over_direct=ratios,
        promote_memory_relief=ratios[32]["post_rollout_allocated"] <= 0.80
        and ratios[32]["gradient_wall"] <= 1.50,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
