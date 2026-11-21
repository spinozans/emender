"""K23 -- THE STEP-LEVEL SPEED PROGRAM: the operator's question, answered
with receipts (zero training, profiling + micro-benchmarks only, one card).

THE QUESTION.  K21 receipted the composed scan as LATENCY-bound: one launch
exposes batch*heads = 16 independent rows on a 142-SM card, runs at 0.7% of
DRAM peak and 17% of its own 16-SM issue ceiling, and sits 5.9x above its
own 3.74 ms floor; the step wall is 10.62 s of which the scan FORWARDS are
~4.5 s.  Where does a bit-identical step go from here?  Four levers, each
measured and receipted under the standing bar -- BIT-IDENTITY PER OUTPUT
ELEMENT OR IT IS A NEW ENGINE VERSION:

1. BATCHING/INTERLEAVING  all window-groups of a layer in ONE kernel launch
   (the K21-diagnosed lever): independent rows scheduled together is pure
   re-scheduling, per-element arithmetic order preserved -- but the Triton
   lowering's warp-layout machinery makes that an EMPIRICAL claim (the K21
   4/16-warp precedent), so every batched bracket is checked channel by
   channel (torch.equal on all 13 output channels, per group), forward AND
   backward (the adjoint's 7 gradient tensors, per group).  The K21 1.32x
   stack (lazy-norms / lazy-old-row / split-fold at 8 warps 1 stage) is
   re-validated UNDER THE BATCHED GRID -- the patch and the batching may
   interact, and the composed receipt must hold for the recommended combo.

2. FLOWS/FUSION  profile the FULL step first (torch profiler at the probe
   step shape: 64 windows, microbatch 8): where do the non-scan seconds go?
   Then torch.compile the dominant eager pieces, each under the bit-identity
   bar; a compile that cannot be made bit-identical is receipted honestly
   and parked for a new-engine-version decision.

3. GRAPHS  the launch-overhead share first (launches per step, their sizes,
   host-issue vs synced walls), then CUDA-graph capture of the scan's own
   kernel trains, bit-identity checked (same kernels, same order).

4. THE FLOOR BRACKET  K21's floor (3.74 ms at the 16-row ceiling; the same
   arithmetic at the 128-row ceiling the batch exposes costs the same 3.74
   ms for the whole 8-group launch).  Combine the validated levers, state
   the composite, update K20's exposure table.

THE BOUNDARY (unchanged discipline): the committed engines are imported
verbatim and NEVER modified -- all lever code lives in the scratch files
``k23_batch_brackets.py`` (this lane) and ``k21_micro_brackets.py`` (K21's,
imported).  Integration of validated speedups is a LATER, SEQUENCED lane
(the parallel K22 hybrid lane is hash-binding the committed engines right
now); this lane receipts and recommends, it ships nothing.

PRE-STATE: no directional prediction; the decomposition IS the result.

Usage (one leased card):

    eval "$(GPU_LEASE_VISIBLE=5 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k23_step_speed \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k23_speed.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# The probe subprocess discipline (configure_production fail-closes on a
# missing allocator knob if these are not set before CUDA initialization).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.profiler import ProfilerActivity, profile

from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
    configure_production)
from experiments.expressivity_tasks.x45_matrix_emender import (
    k20_hybrid_velocity as K20)
from experiments.expressivity_tasks.x45_matrix_emender import (
    k21_micro_brackets as MB)
from experiments.expressivity_tasks.x45_matrix_emender import (
    k23_batch_brackets as KB)
from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
    import HybridMuonX45
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
    import composed_scan, convert_model_to_composition
from experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv \
    import (CONV_MACHINE_PARAMETERS, INHERITED_LR_MULTIPLIER, TUNED_ADAMW_LR,
             split_parameters_k18)
from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
    import convert_model_to_short_conv
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
    import _round_f32
from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
    import (BETA2, GRAD_CLIP, MICROBATCH, TUNED_WINNER, _require_lease)
from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
    import _build_k10

# the committed mixer resolves ``composed_scan`` through THIS binding, so the
# recording shim (installed for the capture, removed before any timing) sees
# the exact production scan arguments.
import experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
    as _k18mod

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k23_runs"
SEED = 3401                                  # the standing seed
LAYER = 6                                    # K21's own probe layer
DEPTH = 24                                   # the registered backbone depth
N_GROUPS = 8                                 # 64 windows / microbatch 8
UNIT = ("the probe protocol's step: 64 windows (8 window-groups x batch 8) "
        "x 2048 tokens x 24 layers at the committed 150M width 576, bf16 "
        "autocast compute, fp32 params -- and the per-(layer, window-group) "
        "scan probe at batch 8 x 2048 x heads 2 x rank/value 128 x key 16, "
        "segment 64")
PROFILING_TOOLS = dict(
    torch_profiler="CUPTI kernel timings: per-kernel self CUDA time, "
                   "per-op CPU time, launch counts",
    ncu="UNAVAILABLE on this host: ERR_NVGPUCTRPERM (K21's same-session "
        "receipt) -- no hardware counters; the host-vs-device split below "
        "is wall-clock minus summed kernel self time, honestly labeled")
RECORDED = dict(
    k20_receipt=str(RECEIPTS / "k20_hybrid_velocity.json"),
    k21_receipt=str(RECEIPTS / "k21_scan_profile.json"),
    pure_ours_step_s_k20=10.624, pure_gdn2_step_s_k20=3.119,
    ours_optimizer_s_k20=0.169, gdn2_optimizer_s_k20=0.015,
    mixer_forward_ms_k20=23.403, mixer_backward_ms_k20=24.554,
    mixer_pair_ms_k20=47.983, gdn2_mixer_pair_ms_k20=6.589,
    gdn2_mixer_forward_ms_k20=2.352, gdn2_mixer_backward_ms_k20=4.252,
    committed_scan_direct_ms_k21=22.129,
    stack_bit_identical_ms_k21=16.785, stack_speedup_k21=1.32,
    scan_floor_16rows_ms_k21=3.741,           # k21 cost_model floors
    hybrid_exposure_pct_at_3_of_24_k20=73.1,
    hybrid_exposure_pct_after_stack_k21=75.4,
    pure_ours_step_s_after_stack_k21=9.61)

SCAN_FWD_KERNEL = "_k15_seq_fwd_kernel"
SCAN_BWD_KERNELS = ("_k11b_traj_kernel", "_k10_key_traj_kernel",
                    "_k13_bwd_prep_kernel", "_k11b_bwd_value_walk_kernel",
                    "_k10_bwd_key_walk_kernel")


# ------------------------------------------------------------ the machine


def build_machine_optimizer():
    """The committed K18 conv machine + the banked probe protocol's own
    optimizer split (K20's steps_ours build verbatim)."""
    model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER), SEED)
    model = convert_model_to_composition(model)
    model = convert_model_to_short_conv(model, birth="identity")
    model = model.cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != CONV_MACHINE_PARAMETERS:
        raise ValueError(f"the conv machine's parameters must be the "
                         f"registration's {CONV_MACHINE_PARAMETERS}; got "
                         f"{parameters}")
    muon_params, adamw_params, _, _ = split_parameters_k18(model, False)
    peak_lr = INHERITED_LR_MULTIPLIER * TUNED_ADAMW_LR
    optimizer = HybridMuonX45(
        muon_params, adamw_params, lr=peak_lr, betas=(0.9, BETA2),
        weight_decay=TUNED_WINNER["weight_decay"])
    return model, optimizer, optimizer_config, peak_lr, parameters, summary


def honest_step(model, optimizer, batches, peak_lr, warmup: int, i: int):
    """The committed probe loop's honest clock, verbatim (K20.steps_ours):
    sync -> stream read -> zero_grad -> lr -> every microbatch's fwd/bwd ->
    clip -> optimizer.step -> sync."""
    torch.cuda.synchronize()
    started = time.monotonic()
    batch_rows = next(batches)
    n = sum(len(r[3]) for r in batch_rows)
    optimizer.zero_grad(set_to_none=True)
    lr = peak_lr * min(1., (i + 1) / max(1, warmup))
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_sum = 0.0
    for x, y, valid in grouped(batch_rows, MICROBATCH):
        loss = model.loss(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        (loss * (valid / n)).backward()
        loss_sum += float(loss.detach()) * valid / n
    torch.cuda.synchronize()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                   error_if_nonfinite=True)
    torch.cuda.synchronize()
    opt_started = time.monotonic()
    optimizer.step()
    torch.cuda.synchronize()
    opt_s = time.monotonic() - opt_started
    wall = time.monotonic() - started
    return dict(step=i + 1, wall_s=wall, opt_s=opt_s, targets=n,
                loss=loss_sum)


# ---------------------------------------------- stage 1: walls + footprint


def step_walls_stage(stream, warmup_steps: int, measure_steps: int) -> dict:
    """The clean (unprofiled) per-step walls at the probe step shape, plus
    the per-group activation footprint -- the memory bill the batching
    lever's integration (all window-groups' graphs alive together) would
    have to pay."""
    model, optimizer, cfg, peak_lr, parameters, summary = \
        build_machine_optimizer()
    batches = iter(stream.batches("train"))
    rows = []
    for i in range(warmup_steps + measure_steps):
        row = honest_step(model, optimizer, batches, peak_lr,
                          cfg["warmup_steps"], i)
        rows.append(dict(step=row["step"], wall_s=round(row["wall_s"], 3),
                         opt_s=round(row["opt_s"], 3),
                         targets=row["targets"], loss=round(row["loss"], 6)))
        print(f"[walls] step {row['step']}: {row['wall_s']:.2f} s "
              f"(opt {row['opt_s']:.2f} s)", flush=True)
    measured = rows[warmup_steps:]
    # ---- the per-group activation footprint: one group's fwd (graph held),
    # measured on the allocator's own books after the optimizer states exist.
    batch_rows = next(batches)
    n = sum(len(r[3]) for r in batch_rows)
    x, y, valid = next(iter(grouped(batch_rows, MICROBATCH)))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    loss = model.loss(x, y)
    torch.cuda.synchronize()
    after_fwd_alloc = torch.cuda.memory_allocated()
    (loss * (valid / n)).backward()
    torch.cuda.synchronize()
    footprint = after_fwd_alloc - base_alloc
    free_b, total_b = torch.cuda.mem_get_info()
    report = dict(
        machine="the committed K18 conv machine under the banked probe "
                "protocol's step body (K20.steps_ours verbatim: "
                "HybridMuonX45, the K18 arm's optimizer split)",
        parameters=parameters, warmup_steps=warmup_steps,
        measured_steps=measure_steps, step_rows=rows,
        step_wall_s=K20.spread([r["wall_s"] * 1000 for r in measured]),
        opt_s=K20.spread([r["opt_s"] * 1000 for r in measured]),
        recorded_reference=dict(pure_ours_step_s_k20=
                                RECORDED["pure_ours_step_s_k20"],
                                ours_optimizer_s_k20=
                                RECORDED["ours_optimizer_s_k20"]),
        activation_footprint=dict(
            one_group_fwd_graph_bytes=int(footprint),
            one_group_fwd_graph_gib=round(footprint / 2**30, 2),
            eight_groups_gib=round(8 * footprint / 2**30, 2),
            note="the retained-graph bytes ONE window-group's forward holds "
                 "between its fwd and bwd (allocator books, steady state "
                 "with optimizer states resident); the batching lever's "
                 "all-groups-together integration would hold ~8x this, "
                 "against the card's free memory below",
            card_free_gib_at_measure=round(free_b / 2**30, 2),
            card_total_gib=round(total_b / 2**30, 2)))
    del model, optimizer, loss
    torch.cuda.empty_cache()
    return report


# --------------------------------------------- stage 2: the step profile


def _is_kernel_name(name: str) -> bool:
    """Device-kernel row detection (K21's own rule: op/autograd wrapper
    rows mirror their child kernel's device time and would double-count)."""
    lowered = name.lower()
    return (name.startswith("void ") or "_kernel" in lowered
            or "nvjet" in lowered or "gemm" in lowered
            or "cutlass" in lowered or name.startswith("Memcpy")
            or name.startswith("Memset")
            or SCAN_FWD_KERNEL in name
            or any(k in name for k in SCAN_BWD_KERNELS)
            or name.startswith("triton"))


def _kernel_rows(prof, iters: int) -> list[dict]:
    """Device kernels only (op/autograd wrapper rows mirror their child
    kernel's device time and would double-count -- K21's own rule)."""
    rows = []
    for evt in prof.key_averages():
        self_us = float(getattr(evt, "self_device_time_total", 0.0)
                        or getattr(evt, "self_cuda_time_total", 0.0))
        if self_us <= 0.0:
            continue
        name = evt.key
        if not _is_kernel_name(name):
            continue
        rows.append(dict(name=name[:160], count=int(evt.count),
                         self_cuda_ms_total=self_us / 1000.0))
    rows.sort(key=lambda r: -r["self_cuda_ms_total"])
    for row in rows:
        row["self_cuda_ms_per_iter"] = round(
            row["self_cuda_ms_total"] / iters, 4)
        row["avg_us_per_launch"] = round(
            row["self_cuda_ms_total"] * 1000.0 / row["count"], 2)
        row["launches_per_iter"] = row["count"] / iters
    return rows


def _bucket(name: str, optimizer_names: set) -> str:
    """Per-kernel classification.  GENERIC elementwise/copy kernel names are
    NOT attributed to the optimizer even though the optimizer launches some
    of them (they are shared with the glue) -- the optimizer's own cost is
    stated separately from its ISOLATED profile, so the shared-name time
    stays in the glue bucket with the overlap bounded by that isolated
    total (a ~1.6% effect at the measured optimizer wall)."""
    if SCAN_FWD_KERNEL in name:
        return "the composed scan FORWARD kernel"
    if any(k in name for k in SCAN_BWD_KERNELS):
        return "the composed scan BACKWARD kernels"
    if name in optimizer_names:
        return "the optimizer (Muon + fused aux AdamW)"
    lowered = name.lower()
    if ("gemm" in lowered or "cublas" in lowered or "nvjet" in lowered
            or "cutlass" in lowered or "matmul" in lowered):
        return "GEMM (projections / MLP / head, fwd+bwd)"
    if name.startswith("Memcpy") or name.startswith("Memset"):
        return "memcpy/memset"
    if ("cross_entropy" in lowered or "_ce_" in lowered or "chunk_" in lowered
            or "fla" in lowered):
        return "the fused cross-entropy head"
    return "elementwise/copy/other glue (the optimizer's shared-name " \
           "elementwise share included, bounded by the isolated total)"


def _generic_kernel_name(name: str) -> bool:
    lowered = name.lower()
    return ("elementwise" in lowered or "functor" in lowered
            or "foreach" in lowered or "reduce_kernel" in lowered
            or "copy" in lowered or name.startswith("Memcpy")
            or name.startswith("Memset"))


def step_profile_stage(stream, warmup_steps: int, steps: int,
                       clean_wall_s: float, clean_opt_s: float) -> dict:
    """The full-step decomposition: per-kernel device time by bucket, the
    launch census (how many launches, how big), and the host-vs-device
    split -- levers 2 and 3's evidence base."""
    model, optimizer, cfg, peak_lr, parameters, summary = \
        build_machine_optimizer()
    batches = iter(stream.batches("train"))
    for i in range(warmup_steps):
        honest_step(model, optimizer, batches, peak_lr, cfg["warmup_steps"],
                    i)
    with profile(activities=[ProfilerActivity.CPU,
                             ProfilerActivity.CUDA]) as prof:
        for i in range(steps):
            honest_step(model, optimizer, batches, peak_lr,
                        cfg["warmup_steps"], warmup_steps + i)
        torch.cuda.synchronize()
    # the optimizer's own kernel-name set, profiled alone on the standing
    # grads (the model is freed right after; the double step() on the same
    # grads corrupts nothing downstream)
    with profile(activities=[ProfilerActivity.CPU,
                             ProfilerActivity.CUDA]) as opt_prof:
        optimizer.step()
        torch.cuda.synchronize()
    opt_rows_all = []
    for evt in opt_prof.key_averages():
        self_us = float(getattr(evt, "self_device_time_total", 0.0)
                       or getattr(evt, "self_cuda_time_total", 0.0))
        if self_us > 0.0 and _is_kernel_name(evt.key):
            opt_rows_all.append((evt.key, self_us / 1000.0))
    opt_names_specific = {name for name, _ in opt_rows_all
                           if not _generic_kernel_name(name)}
    opt_isolated_ms = sum(ms for _, ms in opt_rows_all)

    rows = _kernel_rows(prof, steps)
    device_busy_ms = sum(row["self_cuda_ms_total"] for row in rows) / steps
    by_bucket: dict[str, dict] = {}
    for row in rows:
        key = _bucket(row["name"], opt_names_specific)
        entry = by_bucket.setdefault(key, dict(
            launches_per_step=0.0, self_cuda_ms_per_step=0.0, kernels=0))
        entry["launches_per_step"] += row["launches_per_iter"]
        entry["self_cuda_ms_per_step"] += row["self_cuda_ms_per_iter"]
        entry["kernels"] += 1
    for entry in by_bucket.values():
        entry["launches_per_step"] = round(entry["launches_per_step"], 1)
        entry["self_cuda_ms_per_step"] = round(entry["self_cuda_ms_per_step"],
                                               3)
        entry["pct_of_device_busy"] = round(
            100.0 * entry["self_cuda_ms_per_step"] / device_busy_ms, 2)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "profiler_full_step.txt").write_text(
        f"== the optimizer alone (isolated profile) ==\n"
        f"{opt_prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=30)}\n"
        f"\n== the full step x{steps} ==\n"
        f"{prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=60)}")
    launches_per_step = sum(row["launches_per_iter"] for row in rows)
    histogram = {}
    for lo, hi, label in ((0, 50, "< 50 us"), (50, 500, "50-500 us"),
                          (500, 5000, "0.5-5 ms"), (5000, 1e9, "> 5 ms")):
        selected = [r for r in rows if lo <= r["avg_us_per_launch"] < hi]
        histogram[label] = dict(
            kernel_types=len(selected),
            launches_per_step=round(sum(r["launches_per_iter"]
                                        for r in selected), 1),
            self_cuda_ms_per_step=round(sum(r["self_cuda_ms_per_iter"]
                                            for r in selected), 3))
    host_gap_ms = clean_wall_s * 1000.0 - device_busy_ms
    report = dict(
        steps_profiled=steps,
        device_busy_ms_per_step=round(device_busy_ms, 1),
        clean_step_wall_ms=round(clean_wall_s * 1000.0, 1),
        host_gap_ms_per_step=round(host_gap_ms, 1),
        host_gap_pct_of_wall=round(100.0 * host_gap_ms
                                   / (clean_wall_s * 1000.0), 2),
        host_gap_note="clean unprofiled wall minus summed device-kernel self "
                      "time: the launch/dispatch/data-pipeline share the "
                      "CUDA-graph and compile levers could target -- it "
                      "bounds BOTH levers from above",
        launches_per_step=round(launches_per_step, 1),
        launch_histogram=histogram,
        buckets=dict(sorted(by_bucket.items(),
                            key=lambda kv: -kv[1]["self_cuda_ms_per_step"])),
        top_kernels=rows[:25],
        full_table=str((RUNS / "profiler_full_step.txt").relative_to(REPO)),
        optimizer_isolated_self_cuda_ms=round(opt_isolated_ms, 1),
        optimizer_specific_bucket_ms_per_step=by_bucket.get(
            "the optimizer (Muon + fused aux AdamW)", {}).get(
                "self_cuda_ms_per_step"),
        clean_opt_s=clean_opt_s,
        counting_note="device kernels only; per-bucket rows aggregate "
                      "per-kernel self CUDA time over "
                      f"{steps} steps; the wall anchors come from the "
                      "unprofiled stage (the profiler inflates host time)")
    del model, optimizer
    torch.cuda.empty_cache()
    return report


# ------------------------------------------------- stage 3: the capture


def capture_stage(stream):
    """One real probe step's 8 window-groups driven through the committed
    machine under no_grad, with a recording shim on the committed mixer's
    own composed_scan binding: layer 6's EXACT production scan arguments per
    group, plus each group's layer-6 mixer input (the fusion lever's
    fixture).  The shim is removed before anything is timed."""
    model, _, cfg, _, parameters, summary = build_machine_optimizer()
    depth = len(model.blocks)
    if depth != DEPTH:
        raise ValueError(f"expected depth {DEPTH}, got {depth}")
    recorded: list[tuple[list, dict]] = []
    xs: list[torch.Tensor] = []
    original = _k18mod.composed_scan
    call = {"i": 0}

    def shim(*args, **kwargs):
        if call["i"] % depth == LAYER:
            recorded.append(([a.detach().clone() for a in args],
                             dict(kwargs)))
        call["i"] += 1
        return original(*args, **kwargs)

    handle = model.blocks[LAYER].mixer.register_forward_pre_hook(
        lambda module, args: xs.append(args[0].detach().clone()))
    _k18mod.composed_scan = shim
    try:
        batch_rows = next(stream.batches("train"))
        n = sum(len(r[3]) for r in batch_rows)
        losses = []
        with torch.no_grad():
            for x, y, valid in grouped(batch_rows, MICROBATCH):
                losses.append(float(model.loss(x, y)))
    finally:
        _k18mod.composed_scan = original
        handle.remove()
    if len(recorded) != N_GROUPS or len(xs) != N_GROUPS:
        raise ValueError(f"expected {N_GROUPS} recorded groups; got "
                         f"{len(recorded)} scans / {len(xs)} mixer inputs")
    groups = [[t.contiguous() for t in args] for args, _ in recorded]
    kwargs = recorded[0][1]
    policy = kwargs["policy"]
    policy_f32 = tuple(_round_f32(v) for v in
                       (policy.tau_match, policy.tau_merge, policy.tau_pop,
                        policy.tie_break, policy.read_scale))
    shape_note = dict(
        payload=tuple(groups[0][0].shape),
        payload_dtype=str(groups[0][0].dtype),
        decay_dtype=str(groups[0][3].dtype), gates_dtype=str(groups[0][4].dtype),
        rank=int(kwargs["rank"]), rank_max=int(kwargs["rank_max"]),
        segment_tokens=int(kwargs["segment_tokens"]), gated=bool(kwargs["gated"]),
        key_dim=int(groups[0][1].shape[-1]),
        policy=dict(tau_match=policy.tau_match, tau_merge=policy.tau_merge,
                    tau_pop=policy.tau_pop, tie_break=policy.tie_break,
                    read_scale=policy.read_scale, ops=bool(policy.ops)))
    print(f"[capture] {N_GROUPS} groups, layer {LAYER}; capture losses "
          f"{[round(v, 4) for v in losses]}", flush=True)
    report = dict(
        machine="the committed K18 conv machine (K21's own capture form: "
                "one real probe step's 8 window-groups driven through "
                "the machine under no_grad, hooks only)",
        capture=dict(batch_rows=len(batch_rows), targets=n,
                     losses=[round(v, 6) for v in losses],
                     layer=LAYER, scan_shape=shape_note,
                     parameters=parameters),
        scan_args_note="layer 6's EXACT production scan arguments per "
                       "window-group (a recording shim on the committed "
                       "mixer's own composed_scan binding, removed before "
                       "any timing); the mixer inputs are the same groups' "
                       "layer-6 pre-hook activations")
    live = dict(model=model, mixer=model.blocks[LAYER].mixer, groups=groups,
                xs=xs, kwargs=kwargs, policy_f32=policy_f32)
    return report, live


# ------------------------------- stage 4: lever 1 -- the batched FORWARD


def lever_batching_forward_stage(mixer, groups, kwargs, policy_f32,
                                  warmup: int, iters: int) -> dict:
    """The committed scan launched on the CONCATENATION of the step's 8
    window-groups (grid 16 -> 128 programs): pure re-scheduling, checked
    channel by channel per group; then the K21 stack under the batched
    grid."""
    rank = int(kwargs["rank"])
    seg = int(kwargs["segment_tokens"])
    gated = bool(kwargs["gated"])
    key_dim = int(groups[0][1].shape[-1])
    group_batch = int(groups[0][0].shape[0])

    refs = [MB.committed_reference(*g, policy_f32, rank, seg, gated)
            for g in groups]
    ref_twice = MB.committed_reference(*groups[0], policy_f32, rank, seg,
                                       gated)
    if not MB.bit_identity(refs[0], ref_twice)["bit_identical"]:
        raise RuntimeError("the committed scan is not deterministic across "
                           "runs at fixed inputs -- the receipts bar is void")

    def committed_call(g):
        return lambda: composed_scan(*groups[g], **kwargs)

    per_group_walls = [KB.wall_median(committed_call(g), warmup, iters)
                       for g in range(N_GROUPS)]

    def committed_loop():
        for g in range(N_GROUPS):
            composed_scan(*groups[g], **kwargs)

    committed_layer_step = KB.wall_median(committed_loop, max(2, warmup // 2),
                                          iters)
    brackets = []
    for b in (1, 2, 4, 8):
        cat = KB.cat_scan_args(groups[:b])
        state, live, keys = KB.zeros_store(cat[0], rank, key_dim)

        def batched(cat=cat, state=state, live=live, keys=keys):
            return KB.batched_fwd_reference(cat, state, live, keys,
                                           policy_f32, rank, seg, gated)

        out = batched()
        if b == 1:
            check = MB.bit_identity(refs[0], out)
            failed = [k for k, v in check["channels"].items()
                      if not v["bit_identical"]]
            identical = check["bit_identical"]
        else:
            check = KB.per_group_identity(refs[:b], out, b, group_batch)
            identical = check["bit_identical"]
            failed = [f"group {r['group']}: {ch}"
                      for r in check["per_group"] if not r["bit_identical"]
                      for ch, v in r["failed_channels"].items()
                      if not v["bit_identical"]]
        wall = KB.wall_median(batched, warmup, iters)
        brackets.append(dict(
            window_groups=b, rows_exposed=b * group_batch
            * int(groups[0][0].shape[2]),
            bit_identical=identical, failed_channels=failed,
            wall_ms=round(wall, 3),
            per_group_amortized_ms=round(wall / b, 3),
            speedup_vs_committed_layer_step=round(committed_layer_step
                                                  / wall, 3),
            committed_equivalent_ms=round(committed_layer_step * b
                                          / N_GROUPS, 3)))
        print(f"[batch fwd] {b} groups ({brackets[-1]['rows_exposed']} rows):"
              f" bit={identical} {wall:.3f} ms "
              f"({committed_layer_step * b / N_GROUPS / wall:.2f}x per-group"
              f" equivalent)", flush=True)
        del cat, state, live, keys, out
        torch.cuda.empty_cache()

    # ---- the K21 stack under the batched grid (the patch x batching
    # interaction the task names): the recommended combination must hold.
    cat8 = KB.cat_scan_args(groups)
    state8, live8, keys8 = KB.zeros_store(cat8[0], rank, key_dim)
    stack_flags = dict(lazy_norms=1, lazy_old_row=1, reduce_nr=1)
    stack_rows = []
    for stages in (1, 3):
        def stack_batched(stages=stages):
            return MB.scratch_forward_launch(
                *cat8, state8, live8, keys8, policy_f32, rank, seg, gated,
                8, stages, **stack_flags)
        out = stack_batched()
        check = KB.per_group_identity(refs, out, N_GROUPS, group_batch)
        wall = (KB.wall_median(stack_batched, warmup, iters)
                if check["bit_identical"] else
                KB.wall_median(stack_batched, 2, 4))
        stack_rows.append(dict(
            grid="batch-64 (128 rows)", stages=stages, warps=8,
            flags=stack_flags, bit_identical=check["bit_identical"],
            failed_channels=[f"group {r['group']}: {ch}"
                             for r in check["per_group"]
                             if not r["bit_identical"]
                             for ch, v in r["failed_channels"].items()
                             if not v["bit_identical"]],
            wall_ms=round(wall, 3),
            per_group_amortized_ms=round(wall / N_GROUPS, 3),
            speedup_vs_committed_layer_step=round(committed_layer_step
                                                  / wall, 3)))
        print(f"[batch fwd] stack (w8, s{stages}) batch-64: "
              f"bit={check['bit_identical']} {wall:.3f} ms "
              f"({committed_layer_step / wall:.2f}x)", flush=True)
    # the K21 re-validation at the SINGLE-group grid, same session
    state1, live1, keys1 = KB.zeros_store(groups[0][0], rank, key_dim)
    single_rows = []
    for stages in (1, 3):
        def stack_single(stages=stages):
            return MB.scratch_forward_launch(
                *groups[0], state1, live1, keys1, policy_f32, rank, seg,
                gated, 8, stages, **stack_flags)
        out = stack_single()
        check = MB.bit_identity(refs[0], out)
        wall = (KB.wall_median(stack_single, warmup, iters)
                if check["bit_identical"] else
                KB.wall_median(stack_single, 2, 4))
        single_rows.append(dict(
            grid="single-group (16 rows)", stages=stages, warps=8,
            flags=stack_flags, bit_identical=check["bit_identical"],
            failed_channels=[k for k, v in check["channels"].items()
                              if not v["bit_identical"]],
            wall_ms=round(wall, 3),
            speedup_vs_committed=round(per_group_walls[0] / wall, 3)))
        print(f"[batch fwd] stack (w8, s{stages}) single-group: "
              f"bit={check['bit_identical']} {wall:.3f} ms", flush=True)
    report = dict(
        lever="batching/interleaving: all window-groups of a layer in ONE "
              "kernel launch (pure re-scheduling of independent rows)",
        bar="torch.equal on all 13 output channels of the committed opaque "
            "op pair, PER GROUP, on layer 6's captured production "
            "arguments of all 8 window-groups",
        committed_per_group_fwd_ms=[round(w, 3) for w in per_group_walls],
        committed_per_group_fwd_median_ms=round(
            statistics.median(per_group_walls), 3),
        committed_layer_step_fwd_ms=round(committed_layer_step, 3),
        committed_determinism_selfcheck=True,
        brackets=brackets, stack_batched=stack_rows, stack_single=single_rows,
        note="the committed layer-step wall is the production form: 8 "
             "sequential composed_scan calls (wrapper overhead included), "
             "exactly what the committed loop pays per layer per step")
    stash = dict(cat8=cat8, state8=state8, live8=live8, keys8=keys8)
    return report, stash


# ------------------------------ stage 5: lever 1 -- the batched BACKWARD


def lever_batching_backward_stage(groups, kwargs, policy_f32, warmup: int,
                                  iters: int) -> dict:
    """The committed whole-window adjoint (the segment loop: replay + prep
    + both walks) on the concatenated groups -- the backward's kernel set
    launches with rows = batch*heads too, so the same re-scheduling
    question, answered on the 7 gradient tensors per group."""
    rank = int(kwargs["rank"])
    seg = int(kwargs["segment_tokens"])
    gated = bool(kwargs["gated"])
    key_dim = int(groups[0][1].shape[-1])
    group_batch = int(groups[0][0].shape[0])
    heads = int(groups[0][0].shape[2])
    value_dim = int(groups[0][0].shape[3])
    tokens = int(groups[0][0].shape[1])
    scale = policy_f32[4]

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    dreads = [torch.randn((group_batch, tokens, heads, value_dim),
                          dtype=torch.float32, device="cuda",
                          generator=generator)
                 for _ in range(N_GROUPS)]
    # per-group committed references: fwd op + bwd op on each group's own
    # zeros store
    def group_fwd(g):
        state, live, keys = KB.zeros_store(groups[g][0], rank, key_dim)
        return KB.batched_fwd_reference(groups[g], state, live, keys,
                                        policy_f32, rank, seg, gated)

    def group_pair(g):
        def fn():
            fwd = group_fwd(g)
            return KB.scan_bwd_call(groups[g], fwd, dreads[g], rank, seg,
                                   scale, gated)
        return fn

    ref_grads = [group_pair(g)() for g in range(N_GROUPS)]
    pair_walls = [KB.wall_median(group_pair(g), warmup, iters)
                  for g in range(N_GROUPS)]
    fwd_walls = [KB.wall_median(lambda g=g: group_fwd(g), warmup, iters)
                 for g in range(N_GROUPS)]
    bwd_walls = [p - f for p, f in zip(pair_walls, fwd_walls)]

    def committed_bwd_loop():
        for g in range(N_GROUPS):
            group_pair(g)()

    committed_layer_step_bwd = KB.wall_median(
        committed_bwd_loop, max(2, warmup // 2), iters)

    cat = KB.cat_scan_args(groups)
    state, live, keys = KB.zeros_store(cat[0], rank, key_dim)
    dreads_cat = torch.cat(dreads, dim=0).contiguous()

    def batched_fwd():
        return KB.batched_fwd_reference(cat, state, live, keys, policy_f32,
                                        rank, seg, gated)

    def batched_pair():
        fwd = batched_fwd()
        return KB.scan_bwd_call(cat, fwd, dreads_cat, rank, seg, scale,
                                gated)

    grads8 = batched_pair()
    checks = []
    all_equal = True
    for g in range(N_GROUPS):
        check = KB.grads_bit_identical(ref_grads[g],
                                       KB.slice_grads(grads8, g, group_batch))
        checks.append(dict(group=g, bit_identical=check["bit_identical"],
                           failed=[k for k, v in check["channels"].items()
                                   if not v["bit_identical"]]))
        all_equal &= check["bit_identical"]
    batched_fwd_wall = KB.wall_median(batched_fwd, warmup, iters)
    batched_pair_wall = (KB.wall_median(batched_pair, warmup, iters)
                         if all_equal else KB.wall_median(batched_pair, 2, 4))
    batched_bwd_wall = batched_pair_wall - batched_fwd_wall
    print(f"[batch bwd] 8 groups: grads bit={all_equal}; batched fwd "
          f"{batched_fwd_wall:.3f} ms, pair {batched_pair_wall:.3f} ms "
          f"(bwd {batched_bwd_wall:.3f} ms; committed loop bwd "
          f"{sum(bwd_walls):.3f} ms)", flush=True)
    report = dict(
        lever="backward batching: the committed adjoint op on the "
              "concatenated window-groups",
        bar="torch.equal on all 7 gradient tensors (dpayload, dwrite_keys, "
            "dquery_keys, ddecay, dgates, dstate, dkeys), PER GROUP, at "
            "fixed seeded dreads",
        bit_identical=all_equal, per_group_checks=checks,
        committed_fwd_ms=[round(w, 3) for w in fwd_walls],
        committed_pair_ms=[round(w, 3) for w in pair_walls],
        committed_bwd_ms=[round(w, 3) for w in bwd_walls],
        committed_bwd_median_ms=round(statistics.median(bwd_walls), 3),
        committed_layer_step_bwd_ms=round(committed_layer_step_bwd, 3),
        batched_fwd_ms=round(batched_fwd_wall, 3),
        batched_pair_ms=round(batched_pair_wall, 3),
        batched_bwd_ms=round(batched_bwd_wall, 3),
        batched_bwd_speedup_vs_committed_loop=round(
            sum(bwd_walls) / batched_bwd_wall, 3),
        recorded_reference=dict(
            mixer_backward_ms_k20=RECORDED["mixer_backward_ms_k20"],
            note="K20's mixer backward (24.55 ms) includes the glue's own "
                 "adjoint and autograd dispatch around this scan-adjoint "
                 "op; the difference is the mixer glue's backward, stated "
                 "in the verdict"))
    del cat, state, live, keys, dreads_cat, grads8, ref_grads
    torch.cuda.empty_cache()
    return report


# ------------------------------------ stage 6: lever 3 -- the graph census


def lever_graphs_stage(groups, kwargs, policy_f32, stash, warmup: int,
                       iters: int) -> dict:
    """The launch-overhead share, measured three ways: the host-issue cost
    of each scan call (no sync inside the timed window), the launch census
    from the step profile, and -- if the allocator permits capture -- CUDA
    graphs of the scan's own kernel trains, replayed and bit-identity
    checked."""
    rank = int(kwargs["rank"])
    seg = int(kwargs["segment_tokens"])
    gated = bool(kwargs["gated"])
    key_dim = int(groups[0][1].shape[-1])
    group_batch = int(groups[0][0].shape[0])
    heads = int(groups[0][0].shape[2])
    value_dim = int(groups[0][0].shape[3])
    tokens = int(groups[0][0].shape[1])
    scale = policy_f32[4]
    cat8 = stash.get("cat8") if stash else None
    if cat8 is None:
        return dict(
            lever="graphs: CUDA-graph capture of the scan's own kernel trains",
            skipped="the batching-forward stage did not leave its batch-64 "
                     "stash; the graph brackets were not run",
            host_issue_ms={}, graphs=[])

    def committed_call(g):
        return lambda: composed_scan(*groups[g], **kwargs)

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    dreads0 = torch.randn((group_batch, tokens, heads, value_dim),
                          dtype=torch.float32, device="cuda",
                          generator=generator)
    state1, live1, keys1 = KB.zeros_store(groups[0][0], rank, key_dim)

    def group_pair0():
        fwd = KB.batched_fwd_reference(groups[0], state1, live1, keys1,
                                       policy_f32, rank, seg, gated)
        return KB.scan_bwd_call(groups[0], fwd, dreads0, rank, seg, scale,
                                gated)

    def batched_pair8():
        fwd = KB.batched_fwd_reference(cat8, stash["state8"], stash["live8"],
                                       stash["keys8"], policy_f32, rank, seg,
                                       gated)
        dreads8 = torch.cat([dreads0] * N_GROUPS, dim=0).contiguous()
        return KB.scan_bwd_call(cat8, fwd, dreads8, rank, seg, scale, gated)

    issue = dict(
        committed_fwd_per_group_ms=round(
            KB.host_issue_ms(committed_call(0)), 3),
        committed_pair_per_group_ms=round(KB.host_issue_ms(group_pair0), 3),
        batched_pair8_ms=round(KB.host_issue_ms(batched_pair8), 3))
    print(f"[graphs] host-issue costs: {issue}", flush=True)

    def graph_bracket(label, build, eager_reference, eager_wall):
        try:
            graph, outputs = KB.capture_graph(build, warmup=3)
        except Exception as exc:                      # noqa: BLE001
            return dict(label=label, captured=False,
                         failure=f"{type(exc).__name__}: {exc}"[:500])
        graph.replay()
        torch.cuda.synchronize()
        eager_out = eager_reference()
        if isinstance(outputs, tuple) and isinstance(eager_out, tuple):
            identical = bool(all(torch.equal(a, b)
                                  for a, b in zip(eager_out, outputs)))
        else:
            identical = bool(torch.equal(eager_out, outputs))
        samples = KB.replay_median(graph, iters)
        replay = statistics.median(samples)
        return dict(label=label, captured=True, bit_identical=identical,
                    eager_wall_ms=round(eager_wall, 3),
                    replay_wall_ms=round(replay, 3),
                    replay_speedup=round(eager_wall / replay, 3),
                    timing=K20.spread(samples))

    state1b, live1b, keys1b = KB.zeros_store(groups[0][0], rank, key_dim)

    def build_group_pair():
        fwd = KB.batched_fwd_reference(groups[0], state1b, live1b, keys1b,
                                       policy_f32, rank, seg, gated)
        return KB.scan_bwd_call(groups[0], fwd, dreads0, rank, seg, scale,
                                gated)

    eager_pair_wall = KB.wall_median(group_pair0, warmup, iters)
    graphs = [graph_bracket(
        "per-group committed fwd+bwd train (the no-restructure option: "
        "same 16-row grid, launch overhead only)",
        build_group_pair, group_pair0, eager_pair_wall)]

    def build_batched_pair():
        fwd = KB.batched_fwd_reference(cat8, stash["state8"], stash["live8"],
                                       stash["keys8"], policy_f32, rank, seg,
                                       gated)
        dreads8 = torch.cat([dreads0] * N_GROUPS, dim=0).contiguous()
        return KB.scan_bwd_call(cat8, fwd, dreads8, rank, seg, scale, gated)

    eager_batched_wall = KB.wall_median(batched_pair8, warmup, iters)
    graphs.append(graph_bracket(
        "the batch-64 fwd+bwd train (the recommended grid, launch overhead "
        "only)", build_batched_pair, batched_pair8, eager_batched_wall))
    for row in graphs:
        if row.get("captured"):
            print(f"[graphs] {row['label']}: bit={row['bit_identical']} "
                  f"eager {row['eager_wall_ms']:.3f} ms -> replay "
                  f"{row['replay_wall_ms']:.3f} ms "
                  f"({row['replay_speedup']:.2f}x)", flush=True)
        else:
            print(f"[graphs] {row['label']}: capture FAILED: "
                  f"{row['failure']}", flush=True)
    return dict(
        lever="graphs: CUDA-graph capture of the scan's own kernel trains "
              "(same kernels, same arguments, same order)",
        host_issue_ms=issue,
        host_issue_note="the host-side cost of ISSUING one call without "
                        "waiting: at the committed 16-row grid the fwd is "
                        "one kernel (issue << wall), while the backward's "
                        "~32-segment launch train is the launch-census "
                        "object; the step-level host gap is in the profile "
                        "stage",
        graphs=graphs,
        allocator_note="the probe protocol's expandable-segments allocator "
                       "is load-bearing (configure_production); if capture "
                       "is refused, the refusal is receipted and the "
                       "launch-overhead evidence stands on the host-issue "
                       "and census numbers")


# -------------------------------------- stage 7: lever 2 -- flows/fusion


def lever_flows_stage(model, mixer, xs, warmup: int, iters: int) -> dict:
    """torch.compile of the eager glue OUTSIDE the Triton kernel -- the
    mixer whole (the custom scan op stays the opaque committed pair; the
    compile fuses the causal-context/conv/gate/read-gate/projection glue
    around it), and the SwiGLU MLP.  Every piece under the bit-identity
    bar: forward outputs AND the parameter/input gradients, torch.equal;
    a compile that flips bits is receipted honestly and parked."""
    x = xs[0]
    results = {}

    def autocast_call():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return mixer(x)

    def eager_mixer_fwd():
        return autocast_call()

    out_ref = eager_mixer_fwd()
    out_ref2 = eager_mixer_fwd()
    if not torch.equal(out_ref, out_ref2):
        raise RuntimeError("the eager mixer is not deterministic at fixed "
                           "inputs -- the bar is void")
    del out_ref2
    eager_fwd_wall = KB.wall_median(eager_mixer_fwd, warmup, iters)
    try:
        compiled_mixer = torch.compile(mixer, dynamic=False)

        def compiled_mixer_fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return compiled_mixer(x)

        out_c = compiled_mixer_fwd()
        identical = bool(torch.equal(out_ref, out_c))
        compiled_fwd_wall = (KB.wall_median(compiled_mixer_fwd, warmup + 5,
                                            iters)
                             if identical
                             else KB.wall_median(compiled_mixer_fwd, 2, 4))
        results["mixer_whole"] = dict(
            piece="the whole K18 mixer forward (the scan op opaque, the "
                  "eager glue around it fused by inductor)",
            bit_identical=identical,
            max_abs_diff=(float((out_ref.float() - out_c.float()).abs().max())
                          if not identical else 0.0),
            eager_ms=round(eager_fwd_wall, 3),
            compiled_ms=round(compiled_fwd_wall, 3),
            delta_ms=round(eager_fwd_wall - compiled_fwd_wall, 3))
        print(f"[flows] mixer whole: bit={identical} "
              f"{eager_fwd_wall:.3f} -> {compiled_fwd_wall:.3f} ms", flush=True)
    except Exception as exc:                          # noqa: BLE001
        results["mixer_whole"] = dict(
            piece="the whole K18 mixer forward", compiled=False,
            failure=f"{type(exc).__name__}: {exc}"[-800:])
        print(f"[flows] mixer compile FAILED: "
              f"{results['mixer_whole']['failure'][-200:]}", flush=True)
    # the backward under compile, on the SAME receipts bar: the mixer's
    # own parameter grads + dx, eager vs compiled -- its own try so the
    # forward's receipt survives a backward failure
    if results["mixer_whole"].get("bit_identical"):
        try:
            compiled_mixer = torch.compile(mixer, dynamic=False)
            xin_e = x.detach().clone().requires_grad_(True)
            xin_c = x.detach().clone().requires_grad_(True)
            mixer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mixer(xin_e).pow(2).sum().backward()
            eager_grads = {}
            for n, p in mixer.named_parameters():
                eager_grads[n] = (p.grad.detach().clone()
                                  if p.grad is not None else None)
            eager_dx = xin_e.grad.detach().clone()
            mixer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                compiled_mixer(xin_c).pow(2).sum().backward()
            grad_equal = bool(torch.equal(eager_dx, xin_c.grad))
            failed = []
            for n, p in mixer.named_parameters():
                got = p.grad
                if (eager_grads[n] is None) != (got is None) \
                        or (got is not None
                            and not torch.equal(eager_grads[n], got)):
                    failed.append(n)
            grad_equal &= not failed
            del eager_grads, eager_dx
            results["mixer_whole_backward"] = dict(
                piece="the compiled mixer's backward (glue adjoints fused; "
                      "the scan adjoint op opaque)",
                bit_identical=grad_equal, failed_params=failed,
                note="parameter grads + dx, torch.equal per tensor, at the "
                     "captured production activation")
            print(f"[flows] mixer backward: bit={grad_equal} "
                  f"(failed: {failed[:4]})", flush=True)
        except Exception as exc:                      # noqa: BLE001
            results["mixer_whole_backward"] = dict(
                piece="the compiled mixer's backward", compiled=False,
                failure=f"{type(exc).__name__}: {exc}"[-800:])
            print(f"[flows] mixer backward compile FAILED: "
                  f"{results['mixer_whole_backward']['failure'][-200:]}",
                  flush=True)

    mlp = model.blocks[LAYER].mlp
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mlp_ref = mlp(x)
    try:
        compiled_mlp = torch.compile(mlp, dynamic=False)

        def compiled_mlp_fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return compiled_mlp(x)

        mlp_c = compiled_mlp_fwd()
        mlp_identical = bool(torch.equal(mlp_ref, mlp_c))
        def eager_mlp_fwd():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return mlp(x)
        eager_mlp = KB.wall_median(eager_mlp_fwd, warmup, iters)
        compiled_mlp_wall = (KB.wall_median(compiled_mlp_fwd, warmup + 5,
                                            iters)
                             if mlp_identical
                             else KB.wall_median(compiled_mlp_fwd, 2, 4))
        results["mlp"] = dict(
            piece="the SwiGLU MLP forward", bit_identical=mlp_identical,
            eager_ms=round(eager_mlp, 3), compiled_ms=round(compiled_mlp_wall,
                                                            3),
            delta_ms=round(eager_mlp - compiled_mlp_wall, 3))
        print(f"[flows] mlp: bit={mlp_identical} {eager_mlp:.3f} -> "
              f"{compiled_mlp_wall:.3f} ms", flush=True)
    except Exception as exc:                          # noqa: BLE001
        results["mlp"] = dict(piece="the SwiGLU MLP forward", compiled=False,
                               failure=f"{type(exc).__name__}: {exc}"[-500:])
        print(f"[flows] mlp compile FAILED: "
              f"{results['mlp']['failure'][-200:]}", flush=True)
    del out_ref, out_c, mlp_ref, mlp_c
    torch.cuda.empty_cache()
    return dict(
        lever="flows/fusion: torch.compile of the eager glue OUTSIDE the "
              "Triton kernel (the scan op pair stays the committed opaque "
              "engine)",
        bar="bit-identity of the fused piece's outputs (and its parameter/"
            "input gradients for the mixer), torch.equal at the captured "
            "production activation; the K21 wrapper lever 'fused "
            "value|b_proj projections' is already receipted bit-identical "
            "(1.42x on 0.19 ms -- negligible), not re-run here",
        pieces=results)


# ------------------------------------------------- stage 8: the verdict


def verdict_stage(batch_fwd, batch_bwd, graphs, flows, walls, prof) -> dict:
    """The composite, in K20's own units, with the exposure-table update and
    the integration recommendation.  Computed, never estimated: every term
    is a same-session measurement or a recorded anchor, and every
    projection says which."""
    k20 = json.loads((RECEIPTS / "k20_hybrid_velocity.json").read_text())
    k21 = json.loads((RECEIPTS / "k21_scan_profile.json").read_text())
    committed_fwd_group = batch_fwd["committed_per_group_fwd_median_ms"]
    committed_fwd_layer = batch_fwd["committed_layer_step_fwd_ms"]
    # the best BIT-IDENTICAL batched-forward variant (stack under the
    # batched grid preferred, else the plain batched committed launch)
    stack_ok = [row for row in batch_fwd["stack_batched"]
                if row["bit_identical"]]
    plain_ok = [row for row in batch_fwd["brackets"]
                if row["bit_identical"] and row["window_groups"] == N_GROUPS]
    best_fwd = min(stack_ok + plain_ok,
                   key=lambda r: r["wall_ms"]) if (stack_ok or plain_ok) \
        else None
    batched_bwd_ms = batch_bwd["batched_bwd_ms"] \
        if batch_bwd["bit_identical"] else None
    committed_bwd_layer = sum(batch_bwd["committed_bwd_ms"])
    clean_wall_s = walls["step_wall_s"]["median_ms"] / 1000.0
    floor_16 = k21["cost_model"]["floors"][
        "compute_at_semantics_ceiling_ms"]
    # the 8-group batched launch's floor: the same arithmetic (x8 ops) at
    # the 128-row ceiling the batch exposes (x8 rate) -- numerically the
    # same 3.74 ms, for 8x the work
    floor_8group = floor_16
    glue_fwd_ms = (RECORDED["mixer_forward_ms_k20"]
                   - RECORDED["committed_scan_direct_ms_k21"])
    glue_bwd_ms = max(0.0, RECORDED["mixer_backward_ms_k20"]
                      - statistics.median(batch_bwd["committed_bwd_ms"]))

    # ---- the composite layer-step scan cost (fwd + bwd, 8 groups)
    composite_fwd_layer = best_fwd["wall_ms"] if best_fwd else None
    composite_bwd_layer = batched_bwd_ms
    graph_gain_pair = 0.0
    graph_note = "not folded (no bit-identical batch-64 graph receipt)"
    for row in (graphs or {}).get("graphs", []):
        if (row.get("captured") and row.get("bit_identical")
                and "batch-64" in row["label"]):
            graph_gain_pair = max(0.0, row["eager_wall_ms"]
                                  - row["replay_wall_ms"])
            graph_note = ("folded: the batch-64 fwd+bwd train's graph-replay "
                          "gain, measured on the committed batched launch "
                          "(not the stack) -- a lower bound on the gain at "
                          "the composite")
    fusion_mixer_delta = 0.0
    mixer_whole = (flows or {}).get("pieces", {}).get("mixer_whole", {})
    if mixer_whole.get("bit_identical"):
        fusion_mixer_delta = max(0.0, mixer_whole["delta_ms"])

    scan_savings_layer_ms = ((committed_fwd_layer - composite_fwd_layer)
                             + (committed_bwd_layer - composite_bwd_layer)
                             + graph_gain_pair)
    step_savings_s = DEPTH * scan_savings_layer_ms / 1000.0
    # the fusion lever's per-group bound: the compiled mixer's FORWARD
    # delta applies per (layer, group); the backward glue's fusion delta is
    # not separately receipted -- this is a LOWER bound, stated as one
    fusion_step_s = (DEPTH * N_GROUPS * fusion_mixer_delta / 1000.0) \
        if fusion_mixer_delta else 0.0
    pure_ours_composite_s = (clean_wall_s - step_savings_s - fusion_step_s)

    # ---- per-mixer pair, per window-group, in the batched regime
    pair_composite = None
    if composite_fwd_layer is not None and composite_bwd_layer is not None:
        pair_composite = ((composite_fwd_layer + N_GROUPS * glue_fwd_ms
                           - N_GROUPS * fusion_mixer_delta) / N_GROUPS
                          + (composite_bwd_layer
                             + N_GROUPS * glue_bwd_ms) / N_GROUPS)
    gdn2_pair = RECORDED["gdn2_mixer_pair_ms_k20"]
    t_g = k20["projection"]["measured_step_costs"]["pure_gdn2_s"]
    o_g = k20["projection"]["measured_step_costs"]["gdn2_optimizer_s"]
    o_e = k20["projection"]["measured_step_costs"]["ours_optimizer_s"]
    exposure = None
    hybrid_step_muon = hybrid_step_adamw = None
    if pair_composite is not None:
        delta_pair = pair_composite - gdn2_pair
        hybrid_step_muon = t_g + 3 * N_GROUPS * delta_pair / 1000.0 \
            + (o_e - o_g)
        hybrid_step_adamw = t_g + 3 * N_GROUPS * delta_pair / 1000.0
        exposure = dict(
            step_s_mixer_swap_banked_muon=round(hybrid_step_muon, 3),
            exposure_pct_of_gdn2_in_2700s_mixer_swap_banked_muon=round(
                100.0 * t_g / hybrid_step_muon, 1),
            step_s_mixer_swap_gdn2_adamw=round(hybrid_step_adamw, 3),
            exposure_pct_of_gdn2_in_2700s_mixer_swap_gdn2_adamw=round(
                100.0 * t_g / hybrid_step_adamw, 1))
    # ---- the floor bracket, per layer-step (8 groups of forward work)
    committed_fwd_gap = dict(
        committed_layer_step_fwd_ms=round(committed_fwd_layer, 3),
        composite_layer_step_fwd_ms=(round(composite_fwd_layer, 3)
                                     if composite_fwd_layer else None),
        floor_8group_ms=round(floor_8group, 3),
        committed_over_floor=round(committed_fwd_layer / floor_8group, 1),
        composite_over_floor=(round(composite_fwd_layer / floor_8group, 2)
                              if composite_fwd_layer else None),
        fraction_of_gap_closed=(round(
            (committed_fwd_layer - composite_fwd_layer)
            / (committed_fwd_layer - floor_8group), 3)
            if composite_fwd_layer else None),
        note="K21's floor was stated at the 16-row exposure of ONE group "
             "(3.74 ms); the batched launch's 8x work runs at the 128-row "
             "ceiling the batch legitimately exposes (the semantics forbid "
             "token-parallelism, not group-parallelism), so its floor is "
             "the SAME 3.74 ms -- the bracket is per LAYER-STEP (8 groups)")
    speedup_scan_pair = ((committed_fwd_layer + committed_bwd_layer)
                         / (composite_fwd_layer + composite_bwd_layer)) \
        if (composite_fwd_layer and composite_bwd_layer) else None
    return dict(
        the_question="where does a bit-identical step go from here -- "
                     "interleaving, flows, graphs, and the floor",
        measured=dict(
            clean_step_wall_s=round(clean_wall_s, 3),
            committed_scan_fwd_ms_per_group=round(committed_fwd_group, 3),
            committed_scan_fwd_ms_per_layer_step=round(committed_fwd_layer,
                                                       3),
            committed_scan_bwd_ms_per_group=round(
                statistics.median(batch_bwd["committed_bwd_ms"]), 3),
            committed_scan_bwd_ms_per_layer_step=round(committed_bwd_layer,
                                                       3),
            batched_fwd_ms_per_layer_step=(round(composite_fwd_layer, 3)
                                           if composite_fwd_layer else None),
            batched_bwd_ms_per_layer_step=(round(composite_bwd_layer, 3)
                                           if composite_bwd_layer else None),
            bit_identical_pair_speedup_per_layer_step=(
                round(speedup_scan_pair, 2) if speedup_scan_pair else None),
            host_gap_pct_of_wall=(prof or {}).get("host_gap_pct_of_wall"),
            launches_per_step=(prof or {}).get("launches_per_step")),
        floor_bracket=committed_fwd_gap,
        composite=dict(
            scan_savings_per_layer_step_ms=round(scan_savings_layer_ms, 3),
            scan_step_savings_s=round(step_savings_s, 3),
            graph_pair_gain_ms=round(graph_gain_pair, 3),
            graph_gain_note=graph_note,
            fusion_step_savings_s=round(fusion_step_s, 3),
            fusion_step_savings_note="the compiled mixer's FORWARD delta "
                                     "only, per (layer, group) -- a lower "
                                     "bound; the backward glue's fusion "
                                     "delta is not separately receipted",
            pure_ours_step_s_at_composite=round(pure_ours_composite_s, 3),
            step_decomposition_note=f"clean wall {clean_wall_s:.3f} s = "
            "scan fwd "
            f"{((prof or {}).get('buckets', {}).get('the composed scan FORWARD kernel', {}) or {}).get('self_cuda_ms_per_step', 0):.0f} ms + scan bwd "
            f"{((prof or {}).get('buckets', {}).get('the composed scan BACKWARD kernels', {}) or {}).get('self_cuda_ms_per_step', 0):.0f} ms + "
            "everything else per the profile buckets"),
        per_mixer_pair=dict(
            committed_pair_ms=RECORDED["mixer_pair_ms_k20"],
            composite_pair_ms_per_group=(round(pair_composite, 3)
                                          if pair_composite else None),
            gdn2_pair_ms=gdn2_pair,
            ratio_committed=round(RECORDED["mixer_pair_ms_k20"] / gdn2_pair,
                                 2),
            ratio_composite=(round(pair_composite / gdn2_pair, 2)
                             if pair_composite else None)),
        hybrid_exposure_at_3_of_24=exposure,
        escalation_check=dict(
            bit_identical_pair_speedup=speedup_scan_pair,
            threshold=2.0,
            escalated=bool(speedup_scan_pair and speedup_scan_pair >= 2.0),
            note="the task's own escalation bar: a bit-identical composite "
                 ">= 2x changes the hybrid board case materially"),
        residual_risks=dict(
            memory="the batching lever's integration holds all 8 window-"
                   "groups' graphs together: the measured per-group "
                   f"footprint is {walls['activation_footprint']['one_group_fwd_graph_gib']} GiB, "
                   f"x8 = {walls['activation_footprint']['eight_groups_gib']} GiB against "
                   f"{walls['activation_footprint']['card_free_gib_at_measure']} GiB free",
            restructure="the batched launch is receipted at the SCAN level; "
                        "the step integration is a layer-major restructure "
                        "of the committed loop (all groups' forward at a "
                        "layer before the backward), whose grad-"
                        "accumulation order CAN be preserved (backward in "
                        "group order) but must be receipted end-to-end -- "
                        "the integration lane's first falsifier, not "
                        "assumed here"),
    )


# ------------------------------------------------------------------- driver


def run_stage(out: str, key: str, fn):
    """Run one stage; accumulate its JSON-safe report; return (report,
    live).  Stages that need live tensors/modules return (report, live) --
    only the report is receipted; the live objects are the driver's."""
    try:
        result = fn()
    except Exception:
        import traceback
        failure = dict(error=traceback.format_exc()[-4000:])
        print(f"[{key}] FAILED:\n{failure['error']}", flush=True)
        K20.accumulate(out, key + "_failed", failure)
        return None, None
    if isinstance(result, tuple) and len(result) == 2:
        payload, live = result
    else:
        payload, live = result, None
    K20.accumulate(out, key, payload)
    return payload, live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=str(RECEIPTS / "k23_speed.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--step-warmup", type=int, default=2)
    parser.add_argument("--step-measure", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--stages", default="all",
                        help="comma list of stages to (re)run: walls, "
                             "profile, capture, batch_fwd, batch_bwd, "
                             "graphs, flows, verdict; 'all', or e.g. "
                             "'profile,flows,verdict' to re-run only the "
                             "failed stages and recompute the verdict "
                             "from the receipt's stored stages")
    args = parser.parse_args()
    assert args.seed == SEED, "the standing seed is 3401"
    card = _require_lease()
    configure_production()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "K23 needs CUDA"
    RUNS.mkdir(parents=True, exist_ok=True)

    receipt = dict(
        probe="k23_step_speed",
        question="the step-level speed program: interleaving, flows, "
                 "graphs, and the floor -- measured with receipts, "
                 "bit-identical or honestly parked",
        pre_state="no directional prediction; the decomposition IS the "
                  "result",
        unit=UNIT, seed=args.seed,
        session=K20.session_meta(card), git=K20.git_state(),
        bindings=K20.engine_bindings(),
        scratch_bindings=dict(
            k21_micro_brackets_sha256=K20.digest(
                ROOT / "k21_micro_brackets.py"),
            k23_batch_brackets_sha256=K20.digest(
                ROOT / "k23_batch_brackets.py")),
        profiling_tools=PROFILING_TOOLS, recorded_anchors=RECORDED)
    if receipt["git"]["relevant_paths_dirty"]:
        raise ValueError("the committed engine files must be clean before "
                         "profiling them as-committed; dirty: "
                         f"{receipt['git']['relevant_paths_dirty']}")
    # the receipt accumulates incrementally: a --stages re-run keeps the
    # earlier stages' results (and the first run's session record) and only
    # rewrites the stages it runs
    run_all = args.stages == "all"
    wanted = set() if run_all else {
        s.strip() for s in args.stages.split(",") if s.strip()}
    prior = (json.loads(Path(args.out).read_text())
             if Path(args.out).exists() else {})
    if prior.get("session"):
        receipt["first_run_session"] = prior["session"]
    # drop the stale failure markers of any stage this run re-runs
    key_of = dict(walls="step_walls", profile="step_profile",
                  capture="capture", batch_fwd="lever_batching_forward",
                  batch_bwd="lever_batching_backward",
                  graphs="lever_graphs", flows="lever_flows",
                  verdict="verdict")
    for name in (wanted if not run_all else key_of.values()):
        if name in key_of.values():
            prior.pop(name + "_failed", None)
    K20.dump(args.out, {**prior, **receipt})

    def stage(name: str):
        return run_all or name in wanted

    def load_or_run(key: str, fn, needed: bool):
        """Run the stage if selected; otherwise load it from the receipt."""
        if stage(key) or (needed and key not in prior):
            return run_stage(args.out, key, fn)
        if key in prior:
            payload = prior[key]
            if isinstance(payload, dict) and "error" in payload \
                    and set(payload) == {"error"}:
                return None, None
            return payload, None
        return None, None

    stream = K20.make_stream()

    walls, _ = load_or_run("step_walls", lambda: step_walls_stage(
        stream, args.step_warmup, args.step_measure), needed=True)
    if walls is None or "step_wall_s" not in walls:
        raise SystemExit("the step-walls stage is missing; the lane cannot "
                         "continue without the clean anchor")
    prof, _ = load_or_run("step_profile", lambda: step_profile_stage(
        stream, args.profile_warmup, args.profile_steps,
        walls["step_wall_s"]["median_ms"] / 1000.0,
        walls["opt_s"]["median_ms"] / 1000.0), needed=True)

    needs_capture = (stage("batch_fwd") or stage("batch_bwd")
                     or stage("graphs") or stage("flows"))
    capture, live = (None, None)
    if needs_capture or "capture" not in prior:
        capture, live = run_stage(args.out, "capture",
                                  lambda: capture_stage(stream))
    if (stage("batch_fwd") or stage("batch_bwd") or stage("graphs")
            or stage("flows")) and live is None:
        raise SystemExit("the capture stage failed; the lever stages need "
                         "the production scan arguments")

    if stage("batch_fwd") or "lever_batching_forward" not in prior:
        batch_fwd, stash = run_stage(
            args.out, "lever_batching_forward",
            lambda: lever_batching_forward_stage(
                live["mixer"], live["groups"], live["kwargs"],
                live["policy_f32"], args.warmup_iters, args.iters))
    else:
        batch_fwd, stash = prior["lever_batching_forward"], None
    if stage("batch_bwd") or "lever_batching_backward" not in prior:
        batch_bwd, _ = run_stage(
            args.out, "lever_batching_backward",
            lambda: lever_batching_backward_stage(
                live["groups"], live["kwargs"], live["policy_f32"],
                args.warmup_iters, args.iters))
    else:
        batch_bwd = prior["lever_batching_backward"]
    if stage("graphs") or "lever_graphs" not in prior:
        graphs, _ = run_stage(
            args.out, "lever_graphs",
            lambda: lever_graphs_stage(live["groups"], live["kwargs"],
                                      live["policy_f32"], stash,
                                      args.warmup_iters, args.iters))
    else:
        graphs = prior.get("lever_graphs")
    if stage("flows") or "lever_flows" not in prior:
        flows, _ = run_stage(
            args.out, "lever_flows",
            lambda: lever_flows_stage(live["model"], live["mixer"],
                                      live["xs"], args.warmup_iters,
                                      args.iters))
    else:
        flows = prior.get("lever_flows")

    if batch_fwd is None or batch_bwd is None:
        raise SystemExit("a batching stage is missing; the verdict needs "
                         "the batching receipts -- the receipt carries "
                         "the failure")
    verdict, _ = run_stage(args.out, "verdict", lambda: verdict_stage(
        batch_fwd, batch_bwd, graphs, flows, walls, prof))
    if verdict is None:
        raise SystemExit("the verdict stage failed -- the receipt carries "
                         "the failure")
    print(json.dumps(verdict["measured"], indent=1), flush=True)
    print(json.dumps(verdict["floor_bracket"], indent=1), flush=True)
    print(json.dumps(verdict["composite"], indent=1), flush=True)
    print(json.dumps(verdict["hybrid_exposure_at_3_of_24"], indent=1),
          flush=True)
    print(f"receipt written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
