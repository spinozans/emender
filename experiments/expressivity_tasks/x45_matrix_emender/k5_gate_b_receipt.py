"""K5 gate B receipt -- the K4 fused scan, compiled: the receipted blocker removed.

The blocker (``PROFILE_STEP_VS_GDN2.md`` §5, receipt ``x45_on_mb2_compile_route_failed``
inside ``receipts/profile_step_vs_gdn2.json``): with ``fused_backward=True`` *and*
``compile_scan=True``, dynamo traces into the K4 ``torch.autograd.Function`` and
inductor refuses to lower the kernels' ``bool -> int8`` views --
``InductorError: torch.bool is not supported by torch.iinfo`` on ``aten.view.dtype``
-- so the fused arm was forced to ``compile_scan=False`` and eager routing
(243 kernels/chunk, 5.98 M routing launches per mb2 step, ``route_chunk`` cum
62.6 s of a 130.5 s traced step).

The fix: the block call is re-exposed as a proper ``torch.library`` custom op
(``x45::fused_block_scan``) with ``register_autograd``, and -- because
``torch.compile`` traces a registered backward *into* the joint graph -- the
reverse walk is its own opaque op (``x45::fused_block_scan_bwd``).  Dynamo emits
one opaque call per chunk, the routing compiles around it, and the Triton
launches stay out of the trace.

What this receipt banks (``receipts/k5_k4_compile_hosting.json``):

* **the configuration compiles**: dynamo counters for the compiled routed scan
  with the fused path ON (one graph, no frames skipped), the kernel census of a
  compiled forward+backward (the persistent K4 kernels present, one pair per
  chunk), and the gradients flowing;
* **the K4 grad parity gates re-run and unchanged**: ``test_k4_fused_backward.py``
  as a subprocess, summary captured verbatim;
* **the honest fused-mb2 step wall**: the whole training step at microbatch 2
  with ``compile_scan=True`` + ``fused_backward=True`` -- the configuration that
  used to raise -- measured with the gates.py protocol (un-profiled steady
  walls), against the recorded baselines (122.2 s fused compile-off;
  29.75 s eager-scan compiled).

Usage (lease cards 0/1):

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k5_gate_b_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k5_k4_compile_hosting.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import routed_chunk_scan

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
DEVICE = "cuda"
BATCH, HEADS, TOKENS, RANK, VALUE_DIM, CHUNK = 8, 2, 2048, 128, 128, 64
RECORDED_BASELINES = dict(
    fused_mb2_step_wall_s=122.2,
    fused_mb2_tokens_per_s=880.8,
    eager_scan_mb2_step_wall_s=29.75,
    eager_scan_mb2_tokens_per_s=3579.0,
    source="receipts/profile_step_vs_gdn2.json (x45_on_mb2, x45_off_mb2), same-session p50",
)


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).to(DEVICE) * 0.5
    directions = torch.randn(distinct, value_dim, generator=generator).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    return query, payload, decay


def dynamo_snapshot() -> dict:
    from torch._dynamo.utils import counters
    return {str(key): dict(value) for key, value in dict(counters).items() if value}


def measure_compile_hosting(policy) -> dict:
    """The registered mixer shape, fused path ON, compiled: the blocker's absence."""
    query, payload, decay = distinct_stream(BATCH, TOKENS, HEADS, VALUE_DIM, 512, 77)
    compiled = torch.compile(routed_chunk_scan, dynamic=False)

    def run(scan):
        inputs = [tensor.detach().clone().requires_grad_(True)
                  for tensor in (query, payload, decay)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = scan(inputs[0], inputs[1], inputs[2], router=route_chunk,
                          policy=policy, rank=RANK, rank_max=RANK,
                          chunk_tokens=CHUNK, fused=False, trainable_fused=True)
        (result.reads.sum() + result.state.sum()).backward()
        return result, inputs

    began = time.monotonic()
    compiled_result, _ = run(compiled)
    compile_s = time.monotonic() - began
    eager_result, _ = run(routed_chunk_scan)
    for _ in range(2):
        run(compiled)
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run(compiled)
        torch.cuda.synchronize()
    counts: Counter = Counter()
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            counts[event.name[:80]] += 1
    persistent = {name: counts[name] for name in counts
                  if "persistent" in name}
    return dict(
        shape=dict(batch=BATCH, heads=HEADS, tokens=TOKENS, rank=RANK,
                   value_dim=VALUE_DIM, chunk_tokens=CHUNK,
                   chunks_per_scan=TOKENS // CHUNK),
        configuration=("torch.compile(mode='default') on routed_chunk_scan with "
                       "trainable_fused=True -- the configuration that raised "
                       "InductorError before this stage"),
        first_call_compile_s=round(compile_s, 1),
        dynamo_counters=dynamo_snapshot(),
        kernels_per_scan_fwd_bwd=sum(counts.values()),
        persistent_kernels=dict(sorted(persistent.items())),
        forward=dict(
            reads_max_abs_difference=float(
                (compiled_result.reads - eager_result.reads).abs().max()),
            reads_scale=float(eager_result.reads.abs().max()),
            state_max_abs_difference=float(
                (compiled_result.state - eager_result.state).abs().max()),
            note=("the compiled routing arithmetic differs from the eager twin's "
                  "at bf16 exactly as it does without the fused path (gate A's "
                  "cross-regime row); the block scan itself is the opaque op")))


def run_grad_gates() -> dict:
    """The K4 grad parity gates, re-run verbatim as a subprocess."""
    command = [sys.executable, "-m", "pytest", "-q",
               "experiments/expressivity_tasks/x45_matrix_emender/"
               "test_k4_fused_backward.py"]
    began = time.monotonic()
    finished = subprocess.run(command, capture_output=True, text=True,
                              cwd=str(ROOT.parents[2]))
    tail = finished.stdout.strip().splitlines()[-1] if finished.stdout.strip() else ""
    return dict(command=" ".join(command), returncode=finished.returncode,
                wall_s=round(time.monotonic() - began, 1), summary=tail)


def measure_fused_mb2_step(out_path: str) -> dict:
    """The whole training step at mb2, compile ON + fused ON: the honest wall."""
    from experiments.expressivity_tasks.x45_matrix_emender import profile_step_vs_gdn2 as P

    def build():
        return P.build_x45("x45_on_mb2_compile_route")

    # build_x45 honours X45_CONFIGS; the fused+compiled configuration is the one
    # that failed before, so build it directly with the flags the receipt means.
    def build_fused_compiled():
        import experiments.expressivity_tasks.x45_matrix_emender.model as model
        torch.manual_seed(3401)
        torch.cuda.manual_seed_all(3401)
        shape = model.matrix_config()
        policy = RoutingPolicy(ops=True)
        built = model.model_for(shape, policy, 64, compile_scan=True,
                                fused_backward=True).cuda().train()
        summary = dict(shape=shape.__dict__,
                       parameters=model.parameter_count(shape),
                       routing=policy.record(), chunk_tokens=64)
        return built, P.LR, P.BETAS, P.WEIGHT_DECAY, summary

    P.X45_CONFIGS["x45_on_mb2_compile_route"] = dict(
        microbatch=2, compile_scan=True, fused_backward=True,
        description=("fused K4 path at microbatch 2 WITH compile_scan=True -- the "
                     "configuration that raised InductorError before gate B "
                     "(receipt x45_on_mb2_compile_route_failed); routing compiled "
                     "around the opaque K4 node"))
    report = P.measure_arm("x45_on_mb2_compile_route", build_fused_compiled, 2,
                           out_path, warmup=2, wall_steps=3, profile_mode="cuda",
                           do_cprofile=True)
    report["recorded_baselines"] = RECORDED_BASELINES
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECEIPTS / "k5_k4_compile_hosting.json"))
    arguments = parser.parse_args()
    report = {"session": session_meta(), "gate": "B"}
    policy = RoutingPolicy(ops=True)

    print("[hosting] the fused path under torch.compile at the mixer shape ...",
          flush=True)
    report["compile_hosting"] = measure_compile_hosting(policy)

    print("[gates] re-running the K4 grad parity gates ...", flush=True)
    report["k4_grad_gates"] = run_grad_gates()

    print("[mb2] the honest fused+compiled step wall (this is the long one) ...",
          flush=True)
    began = time.monotonic()
    report["fused_mb2_compiled_step"] = measure_fused_mb2_step(arguments.out)
    print(f"[mb2] took {time.monotonic() - began:.0f} s", flush=True)

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(report)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"receipt written to {out}")
    step = report["fused_mb2_compiled_step"]
    print(json.dumps(dict(
        kernels_per_scan=report["compile_hosting"]["kernels_per_scan_fwd_bwd"],
        persistent_kernels=report["compile_hosting"]["persistent_kernels"],
        k4_gates=report["k4_grad_gates"]["summary"],
        mb2_step_walls_s=step["step_walls"],
        mb2_tokens_per_s=step["tokens_per_s"],
        mb2_peak_gb=step["peak_allocated_gb"]), indent=1))


if __name__ == "__main__":
    main()
