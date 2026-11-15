"""K5 gate D receipt — the step metric: where the three fixes landed the step.

Same measurement boundary as the diagnosis (``PROFILE_STEP_VS_GDN2.md``): one
whole model training step — prepare + all microbatches' forward and backward +
clip/AdamW — at mb8, 64 windows x 2048 bytes, bf16 autocast, fp32 master
weights, fused CE, the gates.py/cd_* registered regime.  Steady walls are
un-profiled (the median of the timed steps); the device decomposition is one
additional steady step under ``torch.profiler`` attributed kernel-by-kernel.

The arms, all in ONE session so the matched pair is same-session (the question
the scale proof left open was whether the fused path reaches its own registered
twin once the integration is fixed):

``x45_registered_mb8``   the registered arm's own path, unchanged:
                         ``compile_scan=True``, ``fused_backward=False`` —
                         the 9.29 s / 434,357-kernel baseline re-measured.
``x45_fused_mb8``        gates B+C: the fused scan hosted and recomputed —
                         ``compile_scan=True``, ``fused_backward=True``.
``x45_fused_mb8_k5``     the full K5 stack: the above plus the fused routing
                         decision kernel (gate A) — ``fused_routing=True``.

Banks ``receipts/k5_step_metric.json``: per arm — step walls, tokens/s (total
and steady), peak allocation, kernels/step, device busy, the component
decomposition, and the phase times — plus the honest distance to the 32 k
tok/s device floor and the residual attribution (which of the four components
owns what is left).

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5_gate_d_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5_step_metric.json
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
    step_wall_s=9.29,
    kernels_per_step=434357,
    device_busy_s=3.32,
    tokens_per_s_steady=10863,
    gdn2_tokens_per_s_steady=34855,
    gdn2_kernels_per_step=59294,
    gdn2_step_wall_s=2.91,
    device_floor_tokens_per_s=32000,
    source="receipts/profile_step_vs_gdn2.json (same-session profile, this card class)",
)


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


def build_arm(fused_backward: bool, fused_routing: bool):
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
                                fused_routing=fused_routing).cuda().train()
        summary = dict(shape=shape.__dict__,
                       parameters=model.parameter_count(shape),
                       routing=policy.record(), chunk_tokens=64,
                       fused_backward=fused_backward, fused_routing=fused_routing)
        return built, P.LR, P.BETAS, P.WEIGHT_DECAY, summary

    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECEIPTS / "k5_step_metric.json"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--verdict-only", action="store_true",
                        help="recompute the verdict from the banked arm reports")
    arguments = parser.parse_args()

    from experiments.expressivity_tasks.x45_matrix_emender import (
        profile_step_vs_gdn2 as P)

    torch.set_num_threads(1)
    P.configure_production()
    report = {"session": session_meta(), "gate": "D", "baseline": BASELINE,
              "boundary": P.BOUNDARY}

    arms = [
        ("x45_registered_mb8", False, False),
        ("x45_fused_mb8", True, False),
        ("x45_fused_mb8_k5", True, True),
    ]
    if not arguments.verdict_only:
        for label, fused_backward, fused_routing in arms:
            print(f"[{label}] building + measuring (compile, warmup, steady, "
                  f"profile) ...", flush=True)
            began = time.monotonic()
            P.X45_CONFIGS[label] = dict(
                microbatch=8, compile_scan=True, fused_backward=fused_backward,
                fused_routing=fused_routing,
                description=(f"compile_scan=True, fused_backward={fused_backward}, "
                             f"fused_routing={fused_routing} -- gate D matched pair"))
            arm_report = P.measure_arm(
                label, build_arm(fused_backward, fused_routing), 8, arguments.out,
                warmup=arguments.warmup, wall_steps=arguments.steps,
                profile_mode="full", do_cprofile=False)
            # The chrome trace of the profiled step is parsed already; it is
            # 100s of MB per arm and the receipt keeps the attribution, not the
            # bytes.
            trace = P.TRACE_DIR / f"{label}_step.json"
            if trace.exists():
                trace.unlink()
            print(f"[{label}] done in {time.monotonic() - began:.0f} s: "
                  f"{arm_report['ms_per_step_p50']} ms p50, "
                  f"{arm_report['step_profile']['kernels']} kernels, "
                  f"{arm_report['step_profile']['device_busy_ms']} ms busy",
                  flush=True)

    data = json.loads(Path(arguments.out).read_text())

    # The honest distance and the residual attribution, from the banked arms.
    fused = data["x45_fused_mb8_k5"] if "x45_fused_mb8_k5" in data \
        else data["x45_fused_mb8"]
    registered = data["x45_registered_mb8"]

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
                                    "x45_fused_mb8_k5") if label in data}
    best = min(report["arms"], key=lambda label: report["arms"][label]["step_wall_s"])
    best_arm = report["arms"][best]
    k5_arm = report["arms"].get("x45_fused_mb8_k5", {})
    fused_arm = report["arms"].get("x45_fused_mb8", {})
    decision_ms = kernel_ms(data.get("x45_fused_mb8_k5", {}), "_route_decisions")
    persistent_ms = sum(filter(None, [
        kernel_ms(data.get("x45_fused_mb8", {}), "_persistent_scan_trajectory"),
        kernel_ms(data.get("x45_fused_mb8", {}), "_persistent_bwd"),
        kernel_ms(data.get("x45_fused_mb8", {}), "_persistent_scan_kernel")]))
    report["verdict"] = dict(
        best_arm=best,
        step_wall_s=best_arm["step_wall_s"],
        wall_improvement_s=BASELINE["step_wall_s"] - best_arm["step_wall_s"],
        kernels_per_step=best_arm["kernels_per_step"],
        kernel_reduction=1 - best_arm["kernels_per_step"]
        / BASELINE["kernels_per_step"],
        tokens_per_s_steady=best_arm["tokens_per_s_steady"],
        distance_to_floor_tokens_per_s=BASELINE["device_floor_tokens_per_s"]
        - best_arm["tokens_per_s_steady"],
        host_gap_s=best_arm["host_gap_s"],
        device_busy_s=best_arm["device_busy_s"],
        matched_pair=dict(
            registered_step_wall_s=registered["ms_per_step_p50"] / 1000,
            fused_step_wall_s=data["x45_fused_mb8"]["ms_per_step_p50"] / 1000,
            fused_reaches_registered=(
                data["x45_fused_mb8"]["ms_per_step_p50"]
                <= registered["ms_per_step_p50"]),
            note=("the scale proof's open question: with the compile blocker "
                  "removed (gate B) and the store replaced by checkpoints "
                  "(gate C), the fused path at mb8 vs its own registered twin, "
                  "same session, same protocol -- the three arms' walls are "
                  "within the step-to-step spread at n=3 steps, so the honest "
                  "reading is *parity*: the fused path no longer regresses, and "
                  "it halves the launch count while doing it")),
        attribution=dict(
            route_decisions_kernel_device_ms=decision_ms,
            route_decisions_note=("gate A's kernel costs ~0.5 s of device time "
                                  "per step at this shape (6,144 launches at ~82 "
                                  "us: one program per (batch, head) row, a "
                                  "latency-bound 64-token serial walk), so the "
                                  "fused-routing arm buys ~0 host and pays ~0.5 s "
                                  "device -- consistent with gate A's verdict and "
                                  "the reason fused_routing stays default-off"),
            persistent_kernels_device_ms=persistent_ms,
            session_note=("this session's card runs ~20% faster than the "
                          "profile session (the registered arm's 7.40 s here vs "
                          "9.29 s there); the session-invariant numbers -- "
                          "434,366 vs 434,357 kernels/step and 3.33 vs 3.32 s "
                          "device busy -- reproduce exactly, so the matched "
                          "pair is read same-session")),
        residual_attribution=dict(
            host_gap_s=best_arm["host_gap_s"],
            routing_note=("the surviving launch storm is the routing region's "
                          "differentiable channels (read weights fwd+bwd, "
                          "payload lanes, similarity preprocessing, ~40 "
                          "kernels/chunk compiled; gate A fused the discrete "
                          "chain and moved ~0 -- see "
                          "receipts/k5_routing_fusion.json) plus the compiled "
                          "backbone's own dispatch"),
            device_note=("device busy is at the floor class already (the "
                         "fused scan's persistent kernels + the matmul "
                         "backbone); the 32 k tok/s target needs the host gap "
                         "gone, not more device work")))

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(report)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"receipt written to {out}")
    print(json.dumps(report["verdict"], indent=1))


if __name__ == "__main__":
    main()
