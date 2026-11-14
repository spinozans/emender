"""Gates C and D: no replay in the profile, and tokens/s at the 150 M board.

Runs a real training step loop (AdamW, bf16 compute, fused CE -- the registered
regime) at the registered shape (64 windows x 2048 bytes = 131 072 tokens/step) for
either matrix arm, or for the registered ``gdn2`` arm through the *same* loop so the
speed comparison cannot be a harness difference.  Then it attributes device time
with ``torch.profiler`` and fails closed if any replay kernel is present.

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x45_matrix_emender.gates \
        --arm matrix_ops --steps 3 --out receipts/gate_cd_matrix_ops.json
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

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped, model_for
from experiments.expressivity_tasks.x42_pile_scale.velocity_train import configure_production
from experiments.expressivity_tasks.x45_matrix_emender.model import (
    CAPACITY, COMPILE_MODE, MatrixShape, matrix_config,
    model_for as matrix_model_for, parameter_count)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
# Kernels that must not appear: the tape arm's replay path (78 % of its step) and
# the certified slot-scan kernel the matrix arm replaces.
FORBIDDEN = ("replaybackward", "reverse_kernel", "replay", "scan_kernel")
# The registered comparison point: lattice_gdn2, 96 903 492 bytes in 2 701 s.
REGISTERED_GDN2_TOKENS_PER_S = 96_903_492 / 2_701.096692122519

MATRIX_ARMS = ("matrix_append", "matrix_ops")


def build(arm, microbatch, rank, rank_max, grow, chunk_tokens, seed, compile_scan=True):
    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if arm in MATRIX_ARMS:
        # Growth receipts need the eager path (it keeps the per-chunk telemetry).
        policy = RoutingPolicy(ops=arm == "matrix_ops", grow=grow)
        compile_scan = compile_scan and not grow
        shape = matrix_config(rank=rank, rank_max=rank_max if rank_max else rank)
        model = matrix_model_for(shape, policy, chunk_tokens,
                                 compile_scan=compile_scan).cuda().train()
        lr = 5.42e-4          # the registered gdn2 nomination's optimizer settings
        betas, weight_decay = (0.9, 0.9392004369596886), 0.20059436743465126
        summary = dict(shape=shape.__dict__, parameters=parameter_count(shape),
                       routing=policy.record(), chunk_tokens=chunk_tokens)
        return model, lr, betas, weight_decay, summary
    if arm == "gdn2":
        base, policy, _ = L.build_policy("lattice_gdn2")
        model = model_for(base, policy).cuda().train()
        summary = dict(base_arm=base, parameters=dict(
            total=sum(p.numel() for p in model.parameters())),
            policy=dict(lr=policy.lr, microbatch=policy.microbatch,
                        beta2=policy.beta2, weight_decay=policy.weight_decay))
        return (model, policy.lr, (0.9, policy.beta2), policy.weight_decay, summary)
    raise ValueError(f"unknown arm {arm!r}; expected {MATRIX_ARMS + ('gdn2',)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="matrix_ops")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--rank-max", type=int, default=None)
    parser.add_argument("--grow", action="store_true")
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--profile", action="store_true", default=True)
    parser.add_argument("--eager", action="store_true",
                        help="run the chunk loop without torch.compile")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    model, lr, betas, weight_decay, summary = build(
        args.arm, args.microbatch, args.rank, args.rank_max, args.grow,
        args.chunk_tokens, args.seed, compile_scan=not args.eager)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr,
                                  betas=betas, weight_decay=weight_decay, fused=True)
    batches = iter(stream.batches("train"))

    def one_step(rows, timed):
        # Phase labels are literal: ``prepare`` is the host-side window grouping,
        # ``fwd_bwd`` is the whole microbatch loop (every forward and its backward),
        # ``opt`` is clip + AdamW.  The flagship-stream read (``next(batches)``) is
        # outside every phase, the same boundary the registered profiler uses.
        count = sum(len(r[3]) for r in rows)
        optimizer.zero_grad(set_to_none=True)
        phases = {"prepare": 0.0, "fwd_bwd": 0.0, "opt": 0.0}
        if timed:
            torch.cuda.synchronize()
            mark = time.monotonic()
        groups = list(grouped(rows, args.microbatch))
        if timed:
            torch.cuda.synchronize()
            phases["prepare"] = time.monotonic() - mark
            mark = time.monotonic()
        loss_sum = 0.0
        for x, y, valid in groups:
            # Required before every graphed invocation in a training loop: without it
            # the cudagraph trees take the slow path ("pending, uninvoked backwards")
            # and the captured timing is not the captured path.
            torch.compiler.cudagraph_mark_step_begin()
            loss = model.loss(x, y)
            (loss * (valid / count)).backward()
            loss_sum += float(loss.detach()) * valid / count
        if timed:
            torch.cuda.synchronize()
            phases["fwd_bwd"] = time.monotonic() - mark
            mark = time.monotonic()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if timed:
            torch.cuda.synchronize()
            phases["opt"] = time.monotonic() - mark
        return count, loss_sum, phases

    for _ in range(args.warmup):
        one_step(next(batches), False)
    torch.cuda.reset_peak_memory_stats()

    phases = {"prepare": 0.0, "fwd_bwd": 0.0, "opt": 0.0}
    tokens = 0
    losses = []
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU],
        record_shapes=False, profile_memory=False)
    profiler.start()
    for _ in range(args.steps):
        count, loss, timing = one_step(next(batches), True)
        tokens += count
        losses.append(loss)
        for key in phases:
            phases[key] += timing[key]
    profiler.stop()

    total = sum(phases.values())
    kernels = []
    for event in profiler.key_averages():
        device = getattr(event, "self_device_time_total", 0) or 0
        if device > 0:
            kernels.append(dict(name=event.key[:110], cuda_ms=round(device / 1000, 3),
                                calls=event.count))
    kernels.sort(key=lambda row: -row["cuda_ms"])
    device_total = sum(row["cuda_ms"] for row in kernels) or 1.0
    for row in kernels:
        row["cuda_frac"] = round(row["cuda_ms"] / device_total, 4)
    found = sorted({name for name in FORBIDDEN
                    if any(name in row["name"].lower() for row in kernels)})
    report = dict(
        gate="C+D", arm=args.arm, parameters=parameters, summary=summary,
        microbatches=args.microbatch, windows_per_step=64, context=L.CONTEXT,
        tokens=tokens, steps=args.steps,
        phase_s={k: round(v, 3) for k, v in phases.items()},
        phase_frac={k: round(v / total, 4) for k, v in phases.items()},
        total_s=round(total, 3),
        tokens_per_s=round(tokens / total, 1),
        bytes_per_step=64 * L.CONTEXT,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        losses=losses,
        forbidden_kernels_found=found,
        replay_absent=not found,
        top_kernels=kernels[:20],
        cuda_total_ms=round(device_total, 1),
        registered_gdn2_tokens_per_s=round(REGISTERED_GDN2_TOKENS_PER_S, 1),
        capacity=CAPACITY,
        boundary=("whole model, whole step: prepare + all microbatches' forward and "
                  "backward + clip/AdamW; flagship-stream read excluded; tokens are "
                  "measured targets over the same 64-window stream batches for every "
                  "arm"),
        scan_mode=("eager chunk loop (--eager)" if args.eager else
                   f"torch.compile(mode={COMPILE_MODE}) on ONE shared scan "
                   "(inductor fusion, NO CUDA graph)"),
        seed=args.seed)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("arm", "tokens_per_s", "steps", "total_s", "peak_allocated_gb",
                       "replay_absent", "forbidden_kernels_found")}, indent=1))
    print("top kernels:")
    for row in report["top_kernels"][:8]:
        print(f"  {row['cuda_frac']:.3f} {row['cuda_ms']:9.1f}ms {row['calls']:6d}  "
              f"{row['name']}")
    if found:
        raise SystemExit(f"replay kernels present in the profile: {found}")


if __name__ == "__main__":
    main()
