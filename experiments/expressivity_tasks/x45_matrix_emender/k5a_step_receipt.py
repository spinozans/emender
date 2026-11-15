"""K5A gate D receipt — the step metric: the fused routing region, measured.

Same measurement boundary as every step receipt in this lane (the gates.py /
cd_* protocol): one whole model training step — prepare + all microbatches'
forward and backward + clip/AdamW — at mb8, 64 windows x 2048 bytes, bf16
autocast, fp32 master weights, fused CE.  Steady walls are un-profiled (the
median of the timed steps); the device decomposition is one additional steady
step under ``torch.profiler``, attributed kernel-by-kernel.

The arms, all in ONE session (the matched pair is same-session by construction):

``x45_registered_mb8``  the registered arm's own path, unchanged:
                         ``compile_scan=True``, ``fused_backward=False`` —
                         the 7.40 s / 434,366-kernel baseline of
                         ``receipts/k5_step_metric.json`` re-measured.
``x45_fused_mb8``       gates B+C (the K4 fused scan, hosted and recomputed)
                         — the arm K5A builds on.
``x45_fused_mb8_k5a``    the full stack: the above plus the K5A routing pair
                         (``fused_routing_diff=True``) — the fused scan AND the
                         fused routing region, forward and backward.

Banks ``receipts/k5a_step_metric.json``: per arm — step walls, tokens/s (total
and steady), peak allocation, kernels/step, device busy, host gap, the kernel
attribution (the routing region's before/after and the gate-A decision kernel's
0.50 s device cost, which must be GONE in the K5A arm) — plus the honest
distance to the 32 k tok/s target and the ranked residual if short.

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5a_step_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5a_step_metric.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"

BASELINE = dict(
    registered_step_wall_s=7.40,
    registered_kernels_per_step=434366,
    registered_device_busy_s=3.329466,
    fused_bc_step_wall_s=7.992,
    fused_bc_kernels_per_step=207987,
    fused_bc_device_busy_s=3.05102,
    gate_a_kernels_per_step=219493,
    gate_a_step_wall_s=7.281,
    best_k5_tokens_per_s_steady=13932.3,
    gdn2_tokens_per_s_steady=34855,
    gdn2_kernels_per_step=59294,
    device_floor_tokens_per_s=32000,
    source=("receipts/k5_step_metric.json (gate D, its own session) -- this "
            "receipt re-measures the matched pair in ONE session with the K5A "
            "arm"),
)


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


def build_arm(fused_backward: bool, fused_routing_diff: bool):
    from experiments.expressivity_tasks.x45_matrix_emender import (
        profile_step_vs_gdn2 as P)
    import experiments.expressivity_tasks.x45_matrix_emender.model as model
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)

    def build():
        torch.manual_seed(3401)
        torch.cuda.manual_seed_all(3401)
        shape = model.matrix_config()
        policy = RoutingPolicy(ops=True)
        built = model.model_for(shape, policy, 64, compile_scan=True,
                                fused_backward=fused_backward,
                                fused_routing_diff=fused_routing_diff).cuda().train()
        summary = dict(shape=shape.__dict__,
                       parameters=model.parameter_count(shape),
                       routing=policy.record(), chunk_tokens=64,
                       fused_backward=fused_backward,
                       fused_routing_diff=fused_routing_diff)
        return built, P.LR, P.BETAS, P.WEIGHT_DECAY, summary

    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECEIPTS / "k5a_step_metric.json"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-registered", action="store_true",
                        help="reuse the banked registered arm from k5_step_metric.json")
    arguments = parser.parse_args()

    from experiments.expressivity_tasks.x45_matrix_emender import (
        profile_step_vs_gdn2 as P)

    torch.set_num_threads(1)
    P.configure_production()
    report = {"session": session_meta(), "gate": "D (K5A)",
              "baseline": BASELINE, "boundary": P.BOUNDARY}

    arms = [
        ("x45_registered_mb8", False, False),
        ("x45_fused_mb8", True, False),
        ("x45_fused_mb8_k5a", True, True),
    ]
    for label, fused_backward, fused_routing_diff in arms:
        if arguments.skip_registered and label == "x45_registered_mb8":
            continue
        print(f"[{label}] building + measuring (compile, warmup, steady, "
              f"profile) ...", flush=True)
        began = time.monotonic()
        P.X45_CONFIGS[label] = dict(
            microbatch=8, compile_scan=True, fused_backward=fused_backward,
            fused_routing_diff=fused_routing_diff,
            description=(f"compile_scan=True, fused_backward={fused_backward}, "
                         f"fused_routing_diff={fused_routing_diff} -- K5A step "
                         f"metric"))
        arm_report = P.measure_arm(
            label, build_arm(fused_backward, fused_routing_diff), 8,
            arguments.out, warmup=arguments.warmup, wall_steps=arguments.steps,
            profile_mode="full", do_cprofile=False)
        trace = P.TRACE_DIR / f"{label}_step.json"
        if trace.exists():
            trace.unlink()
        print(f"[{label}] done in {time.monotonic() - began:.0f} s: "
              f"{arm_report['ms_per_step_p50']} ms p50, "
              f"{arm_report['step_profile']['kernels']} kernels, "
              f"{arm_report['step_profile']['device_busy_ms']} ms busy",
              flush=True)

    # merge with any banked arms (k5_step_metric.json) if the registered arm
    # was skipped this session
    data = json.loads(Path(arguments.out).read_text())
    if arguments.skip_registered and "x45_registered_mb8" not in data:
        banked = json.loads((RECEIPTS / "k5_step_metric.json").read_text())
        data["x45_registered_mb8"] = banked["x45_registered_mb8"]
        data["x45_registered_mb8"]["session"] = "banked from k5_step_metric.json (NOT same-session; the matched pair is only honest without --skip-registered)"

    def kernel_ms(arm, fragment):
        for row in arm["step_profile"]["top_kernels"]:
            if fragment in row["name"]:
                return row["device_ms"]
        return None

    def summary(arm):
        profile = arm["step_profile"]
        wall_s = arm["ms_per_step_p50"] / 1000
        busy_s = profile["device_busy_ms"] / 1000
        return dict(step_wall_s=wall_s, kernels_per_step=profile["kernels"],
                    device_busy_s=busy_s, host_gap_s=wall_s - busy_s,
                    tokens_per_s=arm["tokens_per_s"],
                    tokens_per_s_steady=arm["tokens_per_s_steady"],
                    peak_gb=arm["peak_allocated_gb"],
                    components_ms=profile["components_ms"],
                    step_walls_s=arm["step_walls"])

    report["arms"] = {label: summary(data[label])
                      for label in ("x45_registered_mb8", "x45_fused_mb8",
                                    "x45_fused_mb8_k5a") if label in data}
    k5a = report["arms"].get("x45_fused_mb8_k5a", {})
    fused = report["arms"].get("x45_fused_mb8", {})

    def routing_kernels(arm):
        """Every kernel the routing region owns in this arm, from the profile:
        the K5A pair, the gate-A decision kernel, and the eager router's
        inductor fusions are distinguishable by name."""
        rows = arm.get("components_ms", {})
        return rows

    report["verdict"] = dict(
        best_arm=min(report["arms"],
                     key=lambda label: report["arms"][label]["step_wall_s"])
        if report["arms"] else None,
        k5a_step_wall_s=k5a.get("step_wall_s"),
        k5a_kernels_per_step=k5a.get("kernels_per_step"),
        k5a_device_busy_s=k5a.get("device_busy_s"),
        k5a_host_gap_s=k5a.get("host_gap_s"),
        k5a_tokens_per_s_steady=k5a.get("tokens_per_s_steady"),
        distance_to_floor_tokens_per_s=(
            BASELINE["device_floor_tokens_per_s"]
            - k5a.get("tokens_per_s_steady", 0)),
        matched_pair=dict(
            registered_step_wall_s=report["arms"]
            .get("x45_registered_mb8", {}).get("step_wall_s"),
            fused_bc_step_wall_s=fused.get("step_wall_s"),
            k5a_step_wall_s=k5a.get("step_wall_s"),
            note=("same session, same protocol: the registered arm, the B+C "
                  "fused arm, and the K5A arm (fused scan + fused routing "
                  "region)")),
        attribution=dict(
            # the gate-A decision kernel's device cost, which must be GONE
            gate_a_decision_kernel_device_ms=kernel_ms(
                data.get("x45_fused_mb8_k5a", {}), "_route_decisions"),
            k5a_kernels=dict(
                (name, kernel_ms(data.get("x45_fused_mb8_k5a", {}), name))
                for name in ("_k5a_prep_kernel", "_k5a_emit_kernel",
                            "_k5a_bwd_scores_kernel", "_k5a_bwd_rows_kernel")),
            note=("the K5A arm folds the gate-A decision walk into the "
                  "token-block-parallel prep/emit pair, so the 0.50 s "
                  "_route_decisions_kernel device cost must not appear in its "
                  "profile at all")),
        residual_attribution=dict(
            host_gap_s=k5a.get("host_gap_s"),
            note=("filled in by the K5A doc from this receipt: the ranked "
                  "owners of whatever host gap remains (the four compiled "
                  "preprocessing kernels per chunk that decisions require, "
                  "the custom-op dispatch, the backbone's own dispatch)")))

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(report)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"receipt written to {out}")
    print(json.dumps(report["verdict"], indent=1))


if __name__ == "__main__":
    main()
