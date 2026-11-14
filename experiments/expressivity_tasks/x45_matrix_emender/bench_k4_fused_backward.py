"""K4 gates C and D: launches, the trajectory's memory, and the training wall.

Boundary, stated once: **one whole block call, forward *and* backward** -- batch 8,
2048 tokens, 2 heads, rank 128, ``d_v`` 128, lanes 2, fp32 -- entry state and
liveness in, the recorded schedule and read weights as inputs, the tokens' reads
and the exit state out, and the four gradients (``dstate``, ``dweights``,
``ddecay``, ``dpayload``) back.  This is the mixer's share of a training step at
the block boundary; routing is not inside it, exactly as in
``NO_HOST_LOOP.md`` §5.

The paths measured:

``eager_C128``   the registered arm's own backward -- ordinary autograd through
                 the eager chunk scan at chunk 128 (the receipts' boundary), the
                 path the arm trains on today.  ``eager_C64`` is the same at the
                 model's own default chunk length.
``fused``        ``fused_block_scan`` -- the trajectory forward plus the reverse
                 walk.  Forward-only, backward-only and the full step are timed
                 separately, and the backward is swept at a wider warp count.
``persistent``   the forward's forward (``persistent_scan``, no trajectory), so
                 the trajectory store's cost is a measured difference and not a
                 claim.

Also in the receipt, per the stage's gates:

* **C** -- device kernels per call, forward and backward separately, the
  trajectory's bytes as a first-class number (against the block's every other
  tensor and against the card), and the peak allocated bytes of each path;
* **A at the bench shape** -- the fused gradients against the eager per-token
  autograd oracle at batch 2 (the full-batch oracle's autograd graph saves the
  whole trajectory and then some, so the parity case states its batch), with
  the measured fp32 floor per output;
* **E** -- bit-determinism of the fused backward at the full shape, the rank
  64/512 sweep, the rank-2048 refusal with the eager fallback, and the
  profiler's kernel names checked for checkpoint-replay kernels (the tape arm's
  78 % step profile is what this lane is not allowed to reintroduce).

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.bench_k4_fused_backward \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k4_fused_backward.json
"""
from __future__ import annotations

import argparse
import json
import re

import torch

from experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free import (
    head_major, kernel_count, recorded_block, synthetic_block, timed)
from experiments.expressivity_tasks.x45_matrix_emender.bench_k1_scan import (
    BATCH, CHUNK, HEADS, RANK, TOKENS, VALUE_DIM, inputs)
from experiments.expressivity_tasks.x45_matrix_emender.hostloop_free import (
    persistent_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward import (
    eager_block_scan, fused_block_scan, persistent_scan_backward,
    persistent_scan_trajectory, trainable_block_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy
from experiments.expressivity_tasks.x45_matrix_emender.test_k4_fused_backward import (
    GRAD_OUTPUTS, eager_block_walk, fused_grads, grad_floor)

DEVICE = "cuda"
LANES = 2
REPLAY_PATTERN = re.compile(r"replay|checkpoint", re.IGNORECASE)


def replay_kernels(fn, repeats: int = 1) -> dict:
    """The profiler's kernel names matched against the replay/checkpoint family."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    names = sorted({event.name for event in profiler.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA})
    matched = [name for name in names if REPLAY_PATTERN.search(name)]
    return dict(distinct_kernels=len(names), replay_kernels=matched,
                replay_kernels_absent=not matched)


def attached(state, live, weights, decay, schedule):
    """Leaf copies of the block's continuous inputs, under autograd."""
    state = state.detach().clone().requires_grad_(True)
    weights = weights.detach().clone().requires_grad_(True)
    decay = decay.detach().clone().requires_grad_(True)
    payload = schedule.payload.detach().clone().requires_grad_(True)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import MatrixSchedule
    schedule = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                             payload=payload, op=schedule.op,
                             live_after=schedule.live_after)
    return state, weights, decay, payload, schedule


def eager_step(state, live, weights, decay, schedule, chunk, dreads, dexit):
    """The registered path: eager chunked forward + autograd backward."""
    reads, exit_state, _ = eager_block_scan(state, live, weights, decay, schedule,
                                            chunk_tokens=chunk)
    return torch.autograd.grad([reads, exit_state],
                               [state, weights, decay, schedule.payload],
                               [dreads, dexit])


def fused_step(state, live, weights, decay, schedule, dreads, dexit):
    """The fused path: trajectory forward + reverse-walk backward."""
    reads, exit_state, _ = fused_block_scan(state, live, weights, decay, schedule)
    return torch.autograd.grad([reads, exit_state],
                                [state, weights, decay, schedule.payload],
                                [dreads, dexit])


def peak_bytes(fn) -> int:
    """Peak allocated bytes of one call, measured after a warm one."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def measure_case(label, state, live, weights, decay, schedule, repeats: int,
                 parity_batch: int = 2) -> dict:
    batch, heads, tokens, rank = weights.shape
    value_dim = state.shape[-1]
    dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
    dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)

    # ---- the paths, forward-only / backward-only / the whole step
    s, w, d, p, sch = attached(state, live, weights, decay, schedule)
    with torch.no_grad():
        trajectory = persistent_scan_trajectory(state, live, weights, decay,
                                                 schedule)[3]
    paths = {}

    def add(name, description, step, forward=None, backward=None):
        paths[name] = dict(description=description, step=timed(step, repeats,
                                                               label=f"{label}/{name}"),
                           peak_bytes=peak_bytes(step))
        if forward is not None:
            paths[name]["forward_only"] = dict(
                gate_c=kernel_count(forward),
                gate_d=timed(forward, repeats, label=f"{label}/{name}/fwd"))
        if backward is not None:
            paths[name]["backward_only"] = dict(
                gate_c=kernel_count(backward),
                gate_d=timed(backward, repeats, label=f"{label}/{name}/bwd"))
        paths[name]["replay"] = replay_kernels(step)

    # The registered backward, at the receipts' chunk and the model's own.
    for chunk in (CHUNK, 64):
        es, ew, ed, ep, esch = attached(state, live, weights, decay, schedule)
        graph_reads, graph_state, _ = eager_block_scan(es, live, ew, ed, esch,
                                                        chunk_tokens=chunk)

        def eager_backward(chunk=chunk, es=es, ew=ew, ed=ed, esch=esch,
                           graph_reads=graph_reads, graph_state=graph_state):
            return torch.autograd.grad([graph_reads, graph_state],
                                       [es, ew, ed, esch.payload],
                                       [dreads, dexit], retain_graph=True)

        add(f"eager_C{chunk}",
            dict(form="autograd through the eager chunk scan", chunk_tokens=chunk),
            (lambda chunk=chunk, es=es, ew=ew, ed=ed, esch=esch:
             eager_step(es, live, ew, ed, esch, chunk, dreads, dexit)),
            forward=(lambda chunk=chunk: torch.no_grad()(eager_block_scan)(
                state, live, weights, decay, schedule, chunk_tokens=chunk)),
            backward=eager_backward)

    # The fused step, plus the sweep of the backward's warp count.
    fs, fw, fd, fp, fsch = attached(state, live, weights, decay, schedule)
    add("fused", dict(form="trajectory forward + reverse walk", host_launches=dict(
            forward=1, backward="1 walk + 2 deterministic reductions")),
        (lambda: fused_step(fs, live, fw, fd, fsch, dreads, dexit)),
        forward=(lambda: torch.no_grad()(persistent_scan_trajectory)(
            state, live, weights, decay, schedule)),
        backward=(lambda: persistent_scan_backward(trajectory, weights, decay,
                                                   schedule, dreads, dexit)))
    for warps in (8,):
        add(f"fused_bwd_w{warps}", dict(form="the reverse walk at 8 warps"),
            (lambda warps=warps: persistent_scan_backward(
                trajectory, weights, decay, schedule, dreads, dexit, warps=warps)),
            backward=(lambda warps=warps: persistent_scan_backward(
                trajectory, weights, decay, schedule, dreads, dexit, warps=warps)))

    # The forward without the trajectory store: the store's cost is the difference.
    paths["persistent_no_trajectory"] = dict(
        description=dict(form="the forward's own forward, for the store's cost"),
        forward_only=dict(gate_c=kernel_count(
            lambda: torch.no_grad()(persistent_scan)(state, live, weights, decay,
                                                     schedule)),
            gate_d=timed(lambda: torch.no_grad()(persistent_scan)(
                state, live, weights, decay, schedule), repeats,
                label=f"{label}/persistent")))

    # ---- gate E: bit-determinism of the whole fused step at the full shape
    first = fused_step(fs, live, fw, fd, fsch, dreads, dexit)
    second = fused_step(fs, live, fw, fd, fsch, dreads, dexit)
    determinism = {name: bool(torch.equal(a, b)) for name, a, b in
                  zip(GRAD_OUTPUTS, first, second)}

    # ---- gate A at the bench shape (batch stated, see the module docstring)
    slice_batch = slice(0, parity_batch)
    p_state = state[slice_batch].contiguous()
    p_live = live[slice_batch].contiguous()
    p_weights = weights[slice_batch].contiguous()
    p_decay = decay[slice_batch].contiguous()
    p_schedule = type(schedule)(
        slot=schedule.slot[slice_batch], erase=schedule.erase[slice_batch],
        payload=schedule.payload[slice_batch], op=schedule.op[slice_batch],
        live_after=schedule.live_after[slice_batch])
    p_dreads = dreads[slice_batch].contiguous()
    p_dexit = dexit[slice_batch].contiguous()
    floors, oracle = grad_floor(p_state, p_live, p_weights, p_decay, p_schedule,
                                p_dreads, p_dexit)
    _, fused = fused_grads(p_state, p_live, p_weights, p_decay, p_schedule,
                          p_dreads, p_dexit)
    parity = {}
    for index, name in enumerate(GRAD_OUTPUTS):
        deviation = (fused[index].double() - oracle[index].double()).abs()
        parity[name] = dict(floor=floors[name],
                            max=float(deviation.max()),
                            median=float(deviation.median()),
                            over_floor=float(deviation.max()) / floors[name],
                            scale=float(oracle[index].abs().max()),
                            bit_equal=bool(torch.equal(fused[index], oracle[index])))

    trajectory_bytes = batch * heads * tokens * rank * value_dim * 4
    other_bytes = sum(int(t.numel()) * t.element_size() for t in
                      (state, weights, decay, schedule.payload, dreads, dexit))
    return dict(
        label=label,
        shape=dict(batch=batch, tokens=tokens, heads=heads, rank=rank,
                   value_dim=value_dim, lanes=schedule.lanes),
        trajectory_memory=dict(
            bytes=trajectory_bytes,
            gib=trajectory_bytes / 2**30,
            block_tensors_bytes=other_bytes,
            over_block_tensors=trajectory_bytes / other_bytes,
            card_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
            card_fraction=trajectory_bytes
            / torch.cuda.get_device_properties(0).total_memory),
        paths=paths, determinism=determinism, grad_parity=dict(
            batch=parity_batch, floor_and_deviation=parity),
    )


def measure(out_path: str, seed: int = 3401, repeats: int = 5,
            ranks=(64, 512)) -> dict:
    query, payload, decay, pieces = inputs(seed)
    cases = {}
    for label, policy in (("matrix_ops", RoutingPolicy(ops=True)),
                          ("matrix_append", RoutingPolicy(ops=False))):
        state, live, weights, head_first_decay, schedule = recorded_block(
            policy, query, payload, decay, pieces)
        cases[label] = measure_case(label, state, live, weights, head_first_decay,
                                    schedule, repeats)

    # ---- the rank sweep: the two ends of the persistent kernel's range
    sweep = {}
    for rank in ranks:
        state, live, weights, head_first_decay, schedule = synthetic_block(rank, seed=rank)
        batch, heads, tokens = weights.shape[0], weights.shape[1], weights.shape[2]
        value_dim = state.shape[-1]
        dreads = torch.randn(batch, heads, tokens, value_dim, device=DEVICE)
        dexit = torch.randn(batch, heads, rank, value_dim, device=DEVICE)
        s, w, d, p, sch = attached(state, live, weights, head_first_decay, schedule)
        fused = fused_step(s, live, w, d, sch, dreads, dexit)
        with torch.no_grad():
            plain = persistent_scan(state, live, weights, head_first_decay, schedule)
            recorded = persistent_scan_trajectory(state, live, weights,
                                                   head_first_decay, schedule)
        forward_bit_equal = all(torch.equal(a, b) for a, b in zip(plain, recorded[:3]))
        eager = eager_step(s, live, w, d, sch, CHUNK, dreads, dexit)
        grads_close = {name: float((a - b).abs().max())
                       for name, a, b in zip(GRAD_OUTPUTS, fused, eager)}
        sweep[f"rank{rank}"] = dict(
            shape=dict(batch=batch, tokens=tokens, rank=rank, value_dim=value_dim),
            fused_step=timed(
                (lambda s=s, w=w, d=d, sch=sch: fused_step(s, live, w, d, sch,
                                                           dreads, dexit)),
                repeats, label=f"rank{rank}/fused"),
            eager_step_C128=timed(
                (lambda s=s, w=w, d=d, sch=sch: eager_step(s, live, w, d, sch, CHUNK,
                                                           dreads, dexit)),
                repeats, label=f"rank{rank}/eager"),
            forward_bit_equal_to_persistent_scan=forward_bit_equal,
            fused_vs_eager_grad_max=grads_close)

    # ---- the size gate's fallback at rank 2048 (small shape: the state is the
    # thing that does not fit, and 8 x 2048 x 2048 x 128 fp32 would be 16 GiB).
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import MatrixSchedule
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
        random_schedule)
    schedule, query, _, decay_t = random_schedule(1, 32, 1, 2048, 8, lanes=2, seed=7)
    weights = query.permute(0, 2, 1, 3).contiguous()
    head_first = head_major(decay_t)
    state = torch.zeros(1, 1, 2048, 8, device=DEVICE)
    live = torch.zeros(1, 1, 2048, dtype=torch.bool, device=DEVICE)
    s, w, d, p, sch = attached(state, live, weights, head_first, schedule)
    reads, exit_state, _ = trainable_block_scan(s, live, w, d, sch, fused=True,
                                                chunk_tokens=16)
    fallback_grads = torch.autograd.grad([reads, exit_state],
                                         [s, w, d, sch.payload],
                                         [torch.ones_like(reads),
                                          torch.ones_like(exit_state)])
    refusal = None
    try:
        fused_block_scan(s, live, w, d, sch)
    except ValueError as error:
        refusal = str(error)
    fallback = dict(
        rank=2048, refused_by_name=refusal,
        fallback_grads_finite=all(bool(torch.isfinite(g).all()) for g in fallback_grads),
        fallback_grads_nonzero=all(float(g.abs().max()) > 0 for g in fallback_grads))

    report = dict(
        boundary=("one whole block call, forward and backward: batch 8, 2048 tokens, "
                  "2 heads, rank 128, d_v 128, lanes 2, fp32; entry state and liveness "
                  "in, the recorded schedule and read weights as inputs, reads and exit "
                  "state out, four gradients back (dstate, dweights, ddecay, dpayload). "
                  "Routing is outside this boundary, as in NO_HOST_LOOP.md §5."),
        seed=seed, repeats=repeats,
        device=torch.cuda.get_device_name(0), torch=torch.__version__,
        triton=__import__("triton").__version__,
        cases=cases, rank_sweep=sweep, fallback_rank_2048=fallback)
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    report = measure(arguments.out, arguments.seed, arguments.repeats)
    for label, case in report["cases"].items():
        print(f"== {label} {case['shape']}")
        print(f"   trajectory {case['trajectory_memory']['gib']:.2f} GiB "
              f"({case['trajectory_memory']['over_block_tensors']:.1f}x the block's "
              f"other tensors)")
        for name, row in case["paths"].items():
            if "step" not in row:
                continue
            step = row["step"]
            extra = ""
            if "forward_only" in row and "backward_only" in row:
                extra = (f" | fwd {row['forward_only']['gate_d']['latency_ms_p50']:.3f}"
                         f" bwd {row['backward_only']['gate_d']['latency_ms_p50']:.3f}"
                         f" ms, bwd kernels {row['backward_only']['gate_c']['kernels']:.0f}")
            print(f"  {name:16s} step {step['latency_ms_p50']:8.3f} ms "
                  f"(occ {step['occupancy_ms']:7.3f}, "
                  f"ratio {step['latency_over_occupancy']:5.2f})"
                  f"{extra}")
        print(f"   determinism: {case['determinism']}")
        print(f"   grad parity (batch {case['grad_parity']['batch']}): "
              + ", ".join(f"{k} {v['over_floor']:.2f}x floor"
                          for k, v in case['grad_parity']['floor_and_deviation'].items()))
    for rank, row in report["rank_sweep"].items():
        print(f"== {rank}: fused step {row['fused_step']['latency_ms_p50']:.3f} ms, "
              f"eager {row['eager_step_C128']['latency_ms_p50']:.3f} ms, "
              f"forward bit-equal {row['forward_bit_equal_to_persistent_scan']}")
    print(f"== rank 2048: refused by name: {report['fallback_rank_2048']['refused_by_name']}")


if __name__ == "__main__":
    main()
