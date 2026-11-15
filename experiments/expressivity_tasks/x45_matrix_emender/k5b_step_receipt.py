"""K5B gate receipt -- the step metric: the whole-scan op, measured.

Same measurement boundary as every step receipt in this lane (the gates.py /
cd_* protocol): one whole model training step -- prepare + all microbatches'
forward and backward + clip/AdamW -- at mb8, 64 windows x 2048 bytes, bf16
autocast, fp32 master weights, fused CE.  Steady walls are un-profiled (the
median of the timed steps); the device decomposition is one additional steady
step under ``torch.profiler``, attributed kernel-by-kernel.

The arms, all in ONE session (the matched pair is same-session by
construction):

``x45_registered_mb8``   the registered arm's own path, unchanged
                          (``compile_scan=True``) -- the 7.82 s / 434,366-kernel
                          baseline re-measured.
``x45_fused_mb8``        gates B+C (the K4 fused scan, hosted and recomputed).
``x45_fused_mb8_k5a``    the K5A arm: the above plus the fused routing region
                          (``fused_routing_diff=True``) -- the arm whose
                          per-chunk op dispatch owns the host gap this stage
                          removes.
``x45_fused_mb8_k5b``    the K5B arm: the whole routing+scan region as ONE op
                          per (layer, window) scan, captured into a CUDA graph
                          after its first call (``fused_whole_scan=True``).

Plus the instruments this stage's gates demand:

* ``host_ms_per_scan`` -- the routing+scan region's host issue time per scan
  (fwd+bwd, registered mixer shape), before (the per-chunk K5A arm, compiled)
  and after (the whole-scan replay), each x 192 scans per step.
* ``dispatch_counts`` -- the custom-op dispatches per step, before (counted
  from the K5A arm's profiled step: four op bodies per chunk) and after (the
  K5B ledger's fwd/bwd calls around one step: two op bodies per scan), plus
  the whole-scan arm's replay/eager/capture split.

Banks ``receipts/k5b_step_metric.json``: per arm -- step walls, tokens/s
(total and steady), peak allocation, kernels/step, device busy, host gap, the
kernel attribution -- plus the honest distance to the 32k tok/s target and the
ranked residual if short.

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5b_step_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5b_step_metric.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"

BASELINE = dict(
    registered_step_wall_s=7.819,
    registered_kernels_per_step=434366,
    registered_device_busy_s=3.348,
    fused_bc_step_wall_s=8.53,
    fused_bc_kernels_per_step=207985,
    fused_bc_device_busy_s=3.06,
    k5a_step_wall_s=9.08,
    k5a_kernels_per_step=125033,
    k5a_device_busy_s=3.10,
    k5a_host_gap_s=5.98,
    k5a_tokens_per_s_steady=11210.7,
    gdn2_tokens_per_s_steady=28124.0,
    device_floor_tokens_per_s=32000,
    source=("receipts/k5a_step_metric.json (its own session) -- this receipt "
            "re-measures the matched four arms in ONE session with the K5B "
            "arm"),
)
OP_PATTERN = re.compile(r"x45::")


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


def build_arm(fused_backward: bool, fused_routing_diff: bool,
              fused_whole_scan: bool):
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
        built = model.model_for(
            shape, policy, 64, compile_scan=True, fused_backward=fused_backward,
            fused_routing_diff=fused_routing_diff,
            fused_whole_scan=fused_whole_scan).cuda().train()
        summary = dict(shape=shape.__dict__,
                       parameters=model.parameter_count(shape),
                       routing=policy.record(), chunk_tokens=64,
                       fused_backward=fused_backward,
                       fused_routing_diff=fused_routing_diff,
                       fused_whole_scan=fused_whole_scan)
        return built, P.LR, P.BETAS, P.WEIGHT_DECAY, summary

    return build


def summary(arm) -> dict:
    profile = arm["step_profile"]
    wall_s = arm["ms_per_step_p50"] / 1000
    busy_s = (profile["device_busy_ms"] / 1000) if profile else None
    return dict(step_wall_s=wall_s,
                kernels_per_step=profile["kernels"] if profile else None,
                device_busy_s=busy_s,
                host_gap_s=(wall_s - busy_s) if busy_s is not None else None,
                tokens_per_s=arm["tokens_per_s"],
                tokens_per_s_steady=arm["tokens_per_s_steady"],
                peak_gb=arm["peak_allocated_gb"],
                components_ms=profile["components_ms"] if profile else None,
                step_walls=arm["step_walls"])


# --------------------------------------------------------- the per-scan instruments


def host_ms_per_scan(out_path: str) -> dict:
    """The routing+scan region's host issue time per scan, before and after.

    One mixer scan at the registered shape (batch 8, 2048 tokens, chunk 64,
    rank 128, bf16 autocast), forward AND backward, issued without waiting:
    the per-chunk K5A arm (compiled, the 'before') and the whole-scan op
    (compiled wrapper, the 'after').  Everything is warmed first so no compile
    lands inside the timed loop.
    """
    from experiments.expressivity_tasks.x45_matrix_emender import (
        profile_step_vs_gdn2 as P)
    from experiments.expressivity_tasks.x45_matrix_emender.k5a_routing_diff \
        import routing_diff_route_chunk
    from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan \
        import DISPATCH, compiled_whole_scan, next_slot_id, reset_slots
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        COMPILE_MODE, compiled_scan)
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)
    from experiments.expressivity_tasks.x45_matrix_emender.scan import (
        routed_chunk_scan)

    P.configure_production()
    torch.set_num_threads(1)
    batch, tokens, heads, rank, value_dim, chunk = 8, 2048, 2, 128, 128, 64
    policy = RoutingPolicy(ops=True)
    generator = torch.Generator().manual_seed(77)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).cuda() * 0.5
    directions = torch.randn(512, value_dim, generator=generator).cuda()
    choice = torch.randint(0, 512, (batch, tokens, heads),
                           generator=generator)
    payload = directions[choice]
    decay = torch.rand(batch, tokens, heads, generator=generator).cuda() \
        .clamp(0.9, 1.0)
    upstream = torch.randn(batch, tokens, heads, value_dim,
                           generator=generator).cuda()

    def fresh():
        return tuple(t.detach().clone().requires_grad_(True)
                     for t in (query, payload, decay))

    scans_per_step = 24 * 8

    def before():
        q, p, d = fresh()
        scan = compiled_scan(COMPILE_MODE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = scan(q, p, d, router=routing_diff_route_chunk,
                          policy=policy, rank=rank, rank_max=rank,
                          chunk_tokens=chunk, fused=False, trainable_fused=True)
        (result.reads * upstream).sum().backward()

    reset_slots()
    slot = next_slot_id()
    after_scan = compiled_whole_scan("default")

    def after():
        q, p, d = fresh()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = after_scan(q, p, d, policy=policy, rank=rank,
                                rank_max=rank, chunk_tokens=chunk,
                                slot_id=slot)
        (result.reads * upstream).sum().backward()

    def issue_ms(fn, repeats):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(repeats):
            fn()
        issued = time.perf_counter() - began
        torch.cuda.synchronize()
        return dict(issue_ms_per_scan=round(issued / repeats * 1000, 3),
                    repeats=repeats)

    rows = {}
    for label, fn, repeats in (("per_chunk_k5a_compiled", before, 5),
                               ("whole_scan_k5b", after, 20)):
        began = time.monotonic()
        rows[label] = issue_ms(fn, repeats)
        rows[label]["measure_s"] = round(time.monotonic() - began, 1)
        print(f"[host_ms_per_scan] {label}: "
              f"{rows[label]['issue_ms_per_scan']} ms/scan", flush=True)
    ledger = dict(DISPATCH)
    rows["whole_scan_ledger"] = ledger
    rows["scans_per_step"] = scans_per_step
    rows["per_step_seconds"] = {
        label: round(row["issue_ms_per_scan"] * scans_per_step / 1000, 3)
        for label, row in rows.items()
        if isinstance(row, dict) and "issue_ms_per_scan" in row}
    out = Path(out_path)
    data = json.loads(out.read_text()) if out.exists() else {}
    data["host_ms_per_scan"] = rows
    out.write_text(json.dumps(data, indent=2) + "\n")
    return rows


def dispatch_counts(out_path: str) -> dict:
    """The custom-op dispatches per step, before and after.

    Before: the K5A arm's step under a CPU+CUDA profiler, counting the
    ``x45::`` cpu-op events (four op bodies per chunk: K4 fwd+bwd, K5A
    fwd+bwd).  After: the K5B ledger's call counts around one whole step of
    the K5B arm (two op bodies per scan), with the replay/eager/capture split.
    """
    from experiments.expressivity_tasks.x45_matrix_emender import (
        profile_step_vs_gdn2 as P)
    from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan \
        import DISPATCH

    P.configure_production()
    torch.set_num_threads(1)
    rows = {}

    def counted_step(label, build, use_ledger):
        model, lr, betas, weight_decay, _ = build()
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr,
            betas=betas, weight_decay=weight_decay, fused=True)
        batches = P.make_batches()
        P.one_step(model, optimizer, batches, 8)          # warm + capture
        P.one_step(model, optimizer, batches, 8)
        if use_ledger:
            for key in DISPATCH:
                DISPATCH[key] = 0
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
        prof.start()
        P.one_step(model, optimizer, batches, 8)
        prof.stop()
        counts = {}
        if use_ledger:
            counts = {key: int(value) for key, value in DISPATCH.items()}
            counts["note"] = ("the K5B ledger's own counts around the "
                              "profiled step (two op bodies per scan, plus "
                              "the replay/eager/capture split)")
        else:
            for event in prof.events():
                if event.device_type == torch.autograd.DeviceType.CPU \
                        and OP_PATTERN.search(event.name):
                    counts[event.name] = counts.get(event.name, 0) + 1
            counts["note"] = ("x45:: cpu-op events in the profiled step "
                              "(four op bodies per chunk on this arm)")
        rows[label] = counts
        print(f"[dispatch_counts] {label}: "
              f"{ {k: v for k, v in counts.items() if k != 'note'} }",
              flush=True)
        del model, optimizer
        torch.cuda.empty_cache()

    counted_step("per_chunk_k5a", build_arm(True, True, False), False)
    counted_step("whole_scan_k5b", build_arm(False, False, True), True)
    out = Path(out_path)
    data = json.loads(out.read_text()) if out.exists() else {}
    data["dispatch_counts"] = rows
    out.write_text(json.dumps(data, indent=2) + "\n")
    return rows


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=str(RECEIPTS / "k5b_step_metric.json"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--arms", default="all",
                        help="'all', or a comma list of: registered,bc,k5a,k5b")
    parser.add_argument("--skip-instruments", action="store_true")
    arguments = parser.parse_args()

    from experiments.expressivity_tasks.x45_matrix_emender import (
        profile_step_vs_gdn2 as P)

    torch.set_num_threads(1)
    P.configure_production()
    report = {"session": session_meta(), "gate": "D (K5B)",
              "baseline": BASELINE, "boundary": P.BOUNDARY}

    wanted = dict(registered=("x45_registered_mb8", False, False, False),
                  bc=("x45_fused_mb8", True, False, False),
                  k5a=("x45_fused_mb8_k5a", True, True, False),
                  k5b=("x45_fused_mb8_k5b", False, False, True))
    order = ("registered", "bc", "k5a", "k5b")
    selected = order if arguments.arms == "all" \
        else [name.strip() for name in arguments.arms.split(",")]

    for key in selected:
        label, fused_backward, fused_routing_diff, fused_whole_scan = \
            wanted[key]
        print(f"[{label}] building + measuring (compile, warmup, steady, "
              f"profile) ...", flush=True)
        began = time.monotonic()
        P.X45_CONFIGS[label] = dict(
            microbatch=8, compile_scan=True, fused_backward=fused_backward,
            fused_routing_diff=fused_routing_diff,
            fused_whole_scan=fused_whole_scan,
            description=(f"compile_scan=True, fused_backward={fused_backward}, "
                         f"fused_routing_diff={fused_routing_diff}, "
                         f"fused_whole_scan={fused_whole_scan} -- K5B step "
                         f"metric"))
        arm_report = P.measure_arm(
            label, build_arm(fused_backward, fused_routing_diff,
                            fused_whole_scan), 8, arguments.out,
            warmup=arguments.warmup, wall_steps=arguments.steps,
            profile_mode="full", do_cprofile=False)
        trace = P.TRACE_DIR / f"{label}_step.json"
        if trace.exists():
            trace.unlink()
        print(f"[{label}] done in {time.monotonic() - began:.0f} s: "
              f"{arm_report['ms_per_step_p50']} ms p50, "
              f"{arm_report['step_profile']['kernels']} kernels, "
              f"{arm_report['step_profile']['device_busy_ms']} ms busy",
              flush=True)

    if not arguments.skip_instruments:
        print("[instrument] host ms/scan, before and after ...", flush=True)
        host_ms_per_scan(arguments.out)
        print("[instrument] op dispatch counts per step ...", flush=True)
        dispatch_counts(arguments.out)

    data = json.loads(Path(arguments.out).read_text())
    report["arms"] = {label: summary(data[label]) for label in
                      ("x45_registered_mb8", "x45_fused_mb8",
                       "x45_fused_mb8_k5a", "x45_fused_mb8_k5b")
                      if label in data}
    k5b = report["arms"].get("x45_fused_mb8_k5b", {})
    k5a = report["arms"].get("x45_fused_mb8_k5a", {})
    registered = report["arms"].get("x45_registered_mb8", {})

    report["verdict"] = dict(
        best_arm=min(report["arms"],
                     key=lambda label: report["arms"][label]["step_wall_s"])
        if report["arms"] else None,
        k5b_step_wall_s=k5b.get("step_wall_s"),
        k5b_kernels_per_step=k5b.get("kernels_per_step"),
        k5b_device_busy_s=k5b.get("device_busy_s"),
        k5b_host_gap_s=k5b.get("host_gap_s"),
        k5b_tokens_per_s_steady=k5b.get("tokens_per_s_steady"),
        k5a_step_wall_s=k5a.get("step_wall_s"),
        registered_step_wall_s=registered.get("step_wall_s"),
        target_tokens_per_s=BASELINE["device_floor_tokens_per_s"],
        target_reached=bool(
            (k5b.get("tokens_per_s_steady") or 0)
            >= BASELINE["device_floor_tokens_per_s"]),
        host_ms_per_scan=data.get("host_ms_per_scan"),
        dispatch_counts=data.get("dispatch_counts"),
        residual_attribution=dict(
            host_gap_s=k5b.get("host_gap_s"),
            note=("filled in by the K5B doc from this receipt: the ranked "
                  "owners of whatever host gap remains, decomposed with the "
                  "same attribution discipline")))

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(report)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"receipt written to {out}")
    print(json.dumps(report["verdict"], indent=1, default=str))


if __name__ == "__main__":
    main()
