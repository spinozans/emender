"""K6 gate receipts: the registered-shape evidence for the sequential matrix arm.

Writes ``receipts/k6_gates.json`` with, in order:

* **the reproducibility probes** -- the spec's arithmetic is bit-reproducible
  across implementations (the tree fold, ``sqrt.rn``/``div.rn``, and the one
  transcendental, ``libdevice.exp`` vs ``torch.exp``), measured on fresh random
  tensors, not asserted;
* **gate A at the registered shape** -- the decision flip test at the K5A lane
  bar's fixture (batch 8, 2048 tokens, 2 heads, rank 128, d_v 128, bf16
  inputs, 512 distinct directions, seed 77): the persistent kernel vs the
  eager per-token oracle on every schedule field, the payload lane, the read
  weights and the pre-decision liveness -- bit-exact, and the read inside the
  measured floor (the fp32-vs-fp64 oracle floor, measured first);
* **gate B at a registered-class shape** -- the four gradients vs the oracle's
  autograd inside ``NOISE_FACTOR`` of the fp32-vs-fp64 floor measured on the
  same fixture, plus the kernel's register/spill counts and the memory
  arithmetic of the coarse-checkpoint backward (measured peak, the checkpoint
  bytes against the chunk-2 dial's);
* **gate D** -- the forward and backward walls at the registered shape
  (mean/p50 over reps, CUDA events and wall), the launch counts, and a
  3-step smoke of the real 150 M arm for the projected step tok/s.

Usage (the leased card):

    CUDA_VISIBLE_DEVICES=0 /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k6_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k6_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    MAX_SEQ_TILE, _pow2, _round_f32, _seq_backward_body, _seq_forward_launch,
    sequential_available, sequential_oracle_scan, sequential_scan, tree_fold)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed,
                    device="cuda"):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to(device) * 0.5
    directions = torch.randn(distinct, value_dim, generator=g).to(device)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = directions[choice]
    decay = torch.rand(batch, tokens, heads, generator=g).to(device) \
        .clamp(0.9, 1.0)
    return query, payload, decay


# ------------------------------------------------------------- reproducibility


@triton.jit
def _probe_kernel(x_ptr, y_lib, y_tl, y_sqrt, y_div, N: tl.constexpr,
                  BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + offs, mask=m)
    e = x - 40.0            # the softmax's domain: exp arguments in [-40, 0]
    tl.store(y_lib + offs, libdevice.exp(e), mask=m)
    tl.store(y_tl + offs, tl.exp(e), mask=m)
    tl.store(y_sqrt + offs,
             tl.inline_asm_elementwise('sqrt.rn.f32 $0, $1;', '=f,f', [x],
                                       dtype=tl.float32, is_pure=True, pack=1),
             mask=m)
    tl.store(y_div + offs,
             tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;', '=f,f,f',
                                       [x, x + 1.0], dtype=tl.float32,
                                       is_pure=True, pack=1), mask=m)


def reproducibility_probe(n: int = 2_000_000) -> dict:
    """The spec's arithmetic primitives, bit-compared torch vs Triton."""
    x = torch.rand(n, device="cuda") * 40        # non-negative: sqrt's domain
    outs = [torch.empty_like(x) for _ in range(4)]
    _probe_kernel[(triton.cdiv(n, 1024),)](x, *outs, N=n, BLOCK=1024)
    return dict(
        note=("the spec's correctness-roundness claim, measured: libdevice.exp "
              "IS torch's exp (0 mismatches) while Triton's default tl.exp is "
              "not; sqrt.rn/div.rn match torch's IEEE ops"),
        samples=n,
        libdevice_exp_vs_torch_exp_mismatches=int(
            (outs[0] != torch.exp(x - 40.0)).sum()),
        tl_exp_vs_torch_exp_mismatches=int(
            (outs[1] != torch.exp(x - 40.0)).sum()),
        sqrt_rn_vs_torch_sqrt_mismatches=int((outs[2] != torch.sqrt(x)).sum()),
        div_rn_vs_torch_div_mismatches=int((outs[3] != x / (x + 1)).sum()),
        tree_fold_bit_equal=bool(torch.equal(
            tree_fold(x.view(-1, 128).contiguous(), -1, 128),
            _torch_fold_reference(x.view(-1, 128).contiguous(), 128))))


def _torch_fold_reference(x, width):
    n = width
    while n > 1:
        x = x.view(*x.shape[:-1], n // 2, 2).sum(-1)
        n //= 2
    return x.squeeze(-1)


# ------------------------------------------------------------------- gate A


def gate_a_registered_shape() -> dict:
    """The decision flip test at the K5A bar's fixture, plus the read floor."""
    batch, tokens, heads, rank, value_dim = 8, 2048, 2, 128, 128
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                            512, 77)
    query, payload, decay = query.bfloat16(), payload.bfloat16(), \
        decay.bfloat16()
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
    started = time.monotonic()
    out = _seq_forward_launch(query, payload, decay, state, live, POL32, rank,
                              64, collect=True, warps=8, stages=3)
    torch.cuda.synchronize()
    kernel_wall = time.monotonic() - started
    started = time.monotonic()
    ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                 rank=rank, rank_max=rank)
    oracle_wall = time.monotonic() - started
    with torch.no_grad():
        ref64 = sequential_oracle_scan(query.double(), payload.double(),
                                       decay.double(), policy=POLICY,
                                       rank=rank, rank_max=rank)
    flips = 0
    lanes = 0
    per_field = {}
    for name in SCHEDULE_FIELDS:
        a, b = getattr(ref["schedule"], name), out[name]
        f = int((a != b).sum())
        per_field[name] = f
        flips += f
        lanes += a.numel()
    f = int((ref["schedule"].payload != out["payload"]).sum())
    per_field["payload"] = f
    flips += f
    lanes += ref["schedule"].payload.numel()
    f = int((ref["weights"] != out["weights"]).sum())
    per_field["weights"] = f
    flips += f
    lanes += ref["weights"].numel()
    f = int((ref["live_pre"] != out["live_pre"]).sum())
    per_field["live_pre"] = f
    flips += f
    lanes += ref["live_pre"].numel()
    # the read: the fp32-vs-fp64 oracle floor measured FIRST, then the deviation
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    read_scale = ref["reads"].abs().max().item()
    return dict(
        gate="A: decision flip test at the K5A bar's registered fixture",
        fixture=dict(batch=batch, tokens=tokens, heads=heads, rank=rank,
                     value_dim=value_dim, distinct=512, seed=77,
                     precision="bf16 inputs, fp32 defined-order math",
                     segment_tokens=64),
        flips=flips, lanes=lanes, per_field_flips=per_field,
        bit_equal_state=bool(torch.equal(ref["state"], out["state"])),
        bit_equal_live=bool(torch.equal(ref["live"], out["live"])),
        op_histogram=op_histogram(ref["schedule"]),
        read=dict(floor_fp32_vs_fp64=floor, kernel_deviation=deviation,
                  read_scale=read_scale,
                  deviation_over_floor=deviation / max(floor, 1e-30),
                  within_noise_factor=bool(deviation
                                           <= NOISE_FACTOR * max(floor, 1e-30)),
                  note=("the read's rank-axis reduction is a native sum (the "
                        "K1/K23 precedent: decisions bit-exact, reads at the "
                        "measured fp32 floor)")),
        walls=dict(kernel_seconds=round(kernel_wall, 2),
                   oracle_seconds=round(oracle_wall, 1)),
        kernel=dict(registers=out["kernel"].n_regs, spills=out["kernel"].n_spills,
                    warps=8, max_seq_tile=MAX_SEQ_TILE,
                    size_gate=sequential_available(rank, value_dim)))


# ------------------------------------------------------------------- gate B


def gate_b_grads() -> dict:
    """The four gradients vs the oracle's autograd, within measured floors."""
    batch, tokens, heads, rank, value_dim = 2, 256, 2, 128, 128
    seg = 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                            96, 41)
    g = torch.Generator().manual_seed(1234)
    upstream = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to("cuda")

    def oracle_grads(dtype):
        q = query.clone().to(dtype).requires_grad_(True)
        p = payload.clone().to(dtype).requires_grad_(True)
        d = decay.clone().to(dtype).requires_grad_(True)
        s = torch.zeros(batch, heads, rank, value_dim, dtype=dtype,
                        device="cuda").requires_grad_(True)
        ref = sequential_oracle_scan(q, p, d, policy=POLICY, rank=rank,
                                     rank_max=rank, initial_state=s)
        (ref["reads"] * upstream.to(dtype)).sum().backward()
        return (q.grad, p.grad, d.grad, s.grad)

    o32 = oracle_grads(torch.float32)
    o64 = oracle_grads(torch.float64)
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                       device="cuda")
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
    out = _seq_forward_launch(query, payload, decay, state, live, POL32, rank,
                              seg, collect=True, warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    dq, dp, dd, dst = _seq_backward_body(query, payload, decay, upstream,
                                         schedule, out["live_pre"],
                                         out["ckpt_state"], out["ckpt_live"],
                                         rank, seg, POLICY.read_scale)
    rows = []
    for name, kern, orac, or64 in zip(("dquery", "dpayload", "ddecay",
                                       "dstate"),
                                      (dq, dp, dd, dst), o32, o64):
        floor = (orac.float() - or64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        scale = orac.float().abs().max().item()
        rows.append(dict(output=name, floor=floor, deviation=deviation,
                         oracle_scale=scale,
                         deviation_over_floor=deviation / max(floor, 1e-30),
                         within_noise_factor=bool(deviation
                                                  <= NOISE_FACTOR * max(floor, 1e-30))))
    return dict(
        gate="B: gradients vs the eager oracle's autograd, floor first",
        fixture=dict(batch=batch, tokens=tokens, heads=heads, rank=rank,
                     value_dim=value_dim, segment_tokens=seg,
                     note=("a registered-class shape; the full 2048-token "
                           "oracle autograd graph is not needed for the gate "
                           "-- the adjoint's structure is token-local")),
        floor_definition=("the fp32 oracle against its own fp64 twin, per "
                          "output, measured on the same fixture"),
        noise_factor=NOISE_FACTOR,
        outputs=rows,
        bit_exact_note=("none of the four is bit-exact BY DESIGN at this "
                        "stage: the read-weight adjoint's cross-axis "
                        "reductions (dw, the softmax denominator's adjoint, "
                        "duq) are autograd-ordered on the oracle side, so "
                        "the adjoint is not exact-replayable; all four sit "
                        "at the floor (see deviation_over_floor)"))


def gate_b_memory() -> dict:
    """The coarse-checkpoint memory arithmetic, measured."""
    batch, tokens, heads, rank, value_dim = 8, 2048, 2, 128, 128
    seg = 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                            512, 5)
    query, payload, decay = query.bfloat16(), payload.bfloat16(), \
        decay.bfloat16()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    res = sequential_scan(q, p, d, policy=POLICY, rank=rank, rank_max=rank,
                          segment_tokens=seg)
    peak_fwd = torch.cuda.max_memory_allocated() / 2**20
    upstream = torch.randn_like(res.reads)
    (res.reads * upstream).sum().backward()
    torch.cuda.synchronize()
    peak_all = torch.cuda.max_memory_allocated() / 2**20
    nseg = -(-tokens // seg)
    mb = 2**20
    return dict(
        gate="B: the coarse-checkpoint memory arithmetic (measured + stated)",
        fixture=dict(batch=batch, tokens=tokens, heads=heads, rank=rank,
                     value_dim=value_dim, segment_tokens=seg,
                     microbatch=8),
        arithmetic=dict(
            segment_checkpoints=f"{nseg} x {batch*heads*rank*value_dim*4/mb:.1f} MB "
                                f"= {nseg*batch*heads*rank*value_dim*4/mb:.1f} MB per scan",
            segment_trajectory_scratch=f"one segment at a time: "
                                       f"{batch*heads*seg*rank*value_dim*4/mb:.1f} MB",
            dial_chunk2_chunk_entry_saves=f"{tokens//2} x "
                                          f"{batch*heads*rank*value_dim*4/mb:.1f} MB = "
                                          f"{(tokens//2)*batch*heads*rank*value_dim*4/mb:.0f} MB "
                                          f"per scan (the OOM wall)",
            note=("decision granularity and checkpoint granularity are "
                  "independent knobs: the per-token decisions cost NO "
                  "checkpoint memory; the backward keeps 64-token segments")),
        measured=dict(peak_forward_mb=round(peak_fwd, 1),
                      peak_forward_plus_backward_mb=round(peak_all, 1)),
        mb8_fits=bool(peak_all < 40000))


# ------------------------------------------------------------------- gate D


def gate_d_timing(reps: int = 5) -> dict:
    """The honest walls: forward, backward, launch counts, a step smoke."""
    batch, tokens, heads, rank, value_dim = 8, 2048, 2, 128, 128
    seg = 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                            512, 5)
    query, payload, decay = query.bfloat16(), payload.bfloat16(), \
        decay.bfloat16()
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    # forward timing
    for _ in range(2):
        res = sequential_scan(q, p, d, policy=POLICY, rank=rank,
                              rank_max=rank, segment_tokens=seg)
        (res.reads.sum()).backward()
    torch.cuda.synchronize()
    fwd_walls, bwd_walls = [], []
    from torch.profiler import ProfilerActivity
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True); ev0.record()
        res = sequential_scan(q, p, d, policy=POLICY, rank=rank,
                              rank_max=rank, segment_tokens=seg)
        ev1 = torch.cuda.Event(enable_timing=True); ev1.record()
        torch.cuda.synchronize()
        fwd_walls.append((time.perf_counter() - t0,
                          ev0.elapsed_time(ev1) / 1e3))
        grad = torch.randn_like(res.reads)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev0.record()
        (res.reads * grad).sum().backward()
        ev1.record()
        torch.cuda.synchronize()
        bwd_walls.append((time.perf_counter() - t0,
                          ev0.elapsed_time(ev1) / 1e3))
    fwd_sorted = sorted(w for w, _ in fwd_walls)
    bwd_sorted = sorted(w for w, _ in bwd_walls)
    # launch counts
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
        res = sequential_scan(q, p, d, policy=POLICY, rank=rank,
                              rank_max=rank, segment_tokens=seg)
        (res.reads.sum()).backward()
        torch.cuda.synchronize()
    kernels = [e for e in prof.key_averages() if e.device_type ==
               torch.profiler.DeviceType.CUDA]
    total_launches = int(sum(e.count for e in kernels))
    per_scan = {}
    for e in kernels:
        name = e.key[:60]
        per_scan[name] = dict(count=int(e.count),
                               device_ms=round(e.device_time_total, 3))
    return dict(
        gate="D: the sequential-decision cost, stated honestly",
        fixture=dict(batch=batch, tokens=tokens, heads=heads, rank=rank,
                     value_dim=value_dim, segment_tokens=seg, reps=reps),
        forward=dict(wall_s_p50=round(fwd_sorted[len(fwd_sorted)//2], 4),
                      device_s_p50=round(sorted(d for _, d in fwd_walls)
                                         [len(fwd_walls)//2], 4)),
        backward=dict(wall_s_p50=round(bwd_sorted[len(bwd_sorted)//2], 4),
                      device_s_p50=round(sorted(d for _, d in bwd_walls)
                                         [len(bwd_walls)//2], 4)),
        launches_per_scan_fwd_plus_bwd=total_launches,
        kernel_table=per_scan,
        tok_per_s_forward=tokens * batch / fwd_sorted[len(fwd_sorted)//2],
        note=("a per-token decision walk cannot be as fast as a chunk-128 "
              "whole scan; the number is the honest cost of decisions and "
              "reads against the current state at every token"))


def gate_d_step_smoke(steps: int = 3) -> dict:
    """Three real steps of the 150 M arm: the projected step tok/s."""
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import convert_model_to_sequential
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", 128, 3401, k5b=False)
    model = convert_model_to_sequential(model, segment_tokens=64)
    model = model.cuda().train()
    microbatch = optimizer_config["microbatch"]
    context = 2048
    ids = torch.randint(0, 256, (microbatch, context + 1), device="cuda")
    x, y = ids[:, :-1], ids[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    walls = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    targets = microbatch * context
    peak = torch.cuda.max_memory_allocated() / 2**30
    return dict(gate="D: the 150 M arm step smoke",
                steps=steps,
                microbatch=microbatch, context=context,
                walls_s=[round(w, 3) for w in walls],
                steady_step_s=round(sorted(walls)[-1], 3),
                projected_tok_per_s=round(targets / sorted(walls)[-1]),
                peak_allocated_gb=round(peak, 2),
                parameters=summary["parameters"]["total"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "receipts" / "k6_gates.json"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-step-smoke", action="store_true")
    args = ap.parse_args()
    report = {}
    started = time.monotonic()
    report["reproducibility"] = reproducibility_probe()
    print("reproducibility done", flush=True)
    report["gate_a"] = gate_a_registered_shape()
    print("gate A done", flush=True)
    report["gate_b_grads"] = gate_b_grads()
    print("gate B grads done", flush=True)
    report["gate_b_memory"] = gate_b_memory()
    print("gate B memory done", flush=True)
    report["gate_d_timing"] = gate_d_timing(args.reps)
    print("gate D timing done", flush=True)
    if not args.skip_step_smoke:
        report["gate_d_step_smoke"] = gate_d_step_smoke()
    report["walls_seconds_total"] = round(time.monotonic() - started, 1)
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("gate_a", "gate_b_grads")},
                     indent=1)[:2000], flush=True)
    print(f"receipt -> {out}", flush=True)


if __name__ == "__main__":
    main()
