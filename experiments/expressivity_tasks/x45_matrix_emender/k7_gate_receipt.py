"""K7 gates receipt: parity, the split-softmax law, the walk's decisions, the
refusal removed at rank 1024/2048, and the honest walls.

Three receipts in one JSON (``receipts/k7_high_rank.json``):

gate_a -- the slot-blocked chunked value scan: the canonical routed fixture and
   routed chunks at rank 128/512/1024/2048, each with the measured fp32 floor,
   the deviation of the k7 reads from the k23 twin (given identical torch
   weights -- the reduction-order-only comparison), from the per-token oracle,
   and of the in-kernel split-softmax weights from ``read.read_weights``
   (measured against that channel's own fp64 twin), with the state and the
   liveness channel asserted bit-equal.
gate_b -- the sequential decision walk at rank 128/1024/2048: every recorded
   channel (decisions, payload lanes, read weights, pre-decision liveness) bit
   compared against the K7 oracle, the read at its measured floor, the kernel's
   registers/spills, the op coverage, and the oracle pinned to the K6 draft
   registration's oracle at rank 128.
refusal_removed -- the growth snap set executes: block-call wall and device
   times (>= 5 reps) at rank 1024 and 2048 for the k7 path, the k23 twin and
   the torch weights channel it replaces, the memory arithmetic of the
   per-chunk channels and partials, and the persistent kernel's by-name
   refusal recorded alongside -- the ceiling this module exists to remove.

    eval "$(GPU_LEASE_VISIBLE=4,5 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m \
        experiments.expressivity_tasks.x45_matrix_emender.k7_gate_receipt \
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k7_high_rank.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.hostloop_free import (
    chunked_block_scan, persistent_available)
from experiments.expressivity_tasks.x45_matrix_emender.k23_read import k23_chunk
from experiments.expressivity_tasks.x45_matrix_emender.k7_high_rank_store import (
    K7_RANK_TILE, K7_STEP_TILE, k7_block_scan, k7_block_scan_routed, k7_chunk,
    k7_chunk_routed, k7_read_weights, k7_sequential_walk, sequential_oracle,
    _pow2)
from experiments.expressivity_tasks.x45_matrix_emender.read import read_weights
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    routed_fixture)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, sequential_noise_floor)

POLICY = RoutingPolicy(ops=True)
DEVICE = "cuda"


def _reps(fn, count: int, warmup: int = 2):
    """Wall times (perf_counter + sync) and CUDA-event device times, `count` reps."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    walls, devices = [], []
    for _ in range(count):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        mark = time.perf_counter()
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - mark)
        devices.append(start.elapsed_time(stop) / 1000)
    def stats(xs):
        xs = sorted(xs)
        return dict(mean=round(sum(xs) / len(xs) * 1e3, 3),
                    p50=round(xs[len(xs) // 2] * 1e3, 3),
                    min=round(xs[0] * 1e3, 3), max=round(xs[-1] * 1e3, 3))
    return stats(walls), stats(devices)


def _routed_chunk(rank, tokens=64, batch=2, heads=2, value_dim=128, seed=None):
    generator = torch.Generator().manual_seed(seed if seed is not None else rank)
    shape = (batch, heads, tokens, value_dim)
    query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(*shape, generator=generator).to(DEVICE)
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=generator).to(DEVICE) * 0.3
    live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) < 0.7
    state = state * live.unsqueeze(-1)
    decay = torch.full((batch, heads, tokens), 0.95, device=DEVICE)
    schedule, weights = route_chunk(0, 0, state.clone(), live.clone(), query,
                                    payload, POLICY, rank)
    return dict(query=query, state=state, live=live, decay=decay,
                schedule=schedule, weights=weights)


def _oracle_chunk(f):
    weights_tm = f["weights"].permute(0, 2, 1, 3).contiguous()
    reads, state = sequential_matrix_scan(
        weights_tm, f["decay"].permute(0, 2, 1), f["schedule"],
        initial_state=f["state"].clone(), initial_live=f["live"].clone())
    return reads.permute(0, 2, 1, 3), state


# ------------------------------------------------------------------------ gate A


def gate_a() -> dict:
    rows = []
    # the canonical routed fixture (rank 5, all five ops)
    fixture = routed_fixture()
    floor, oracle_state, oracle_reads_tm = sequential_noise_floor(
        fixture.result.weights, fixture.decay, fixture.schedule)
    oracle_reads = oracle_reads_tm.permute(0, 2, 1, 3)
    state = torch.zeros(2, 2, 5, 8, device=DEVICE)
    live = torch.zeros(2, 2, 5, dtype=torch.bool, device=DEVICE)
    r_k23, s_k23, l_k23 = k23_chunk(state.clone(), live.clone(), fixture.weights,
                                    fixture.decay.permute(0, 2, 1).contiguous(),
                                    fixture.schedule)
    r_k7, s_k7, l_k7 = k7_chunk(state.clone(), live.clone(), fixture.weights,
                               fixture.decay.permute(0, 2, 1).contiguous(),
                               fixture.schedule)
    rows.append(dict(
        fixture="canonical routed (rank 5, 64 tokens)", floor=round(floor, 9),
        noise_factor=NOISE_FACTOR,
        k7_vs_oracle=round(float((r_k7 - oracle_reads).abs().max()), 9),
        k7_vs_oracle_x_floor=round(float((r_k7 - oracle_reads).abs().max()) / floor, 2),
        k7_vs_k23=round(float((r_k7 - r_k23).abs().max()), 9),
        state_bit_equal_to_oracle=bool(torch.equal(s_k7, oracle_state)),
        state_bit_equal_to_k23=bool(torch.equal(s_k7, s_k23)),
        liveness_equal=bool(torch.equal(l_k7, l_k23))))
    # routed chunks across the snap set
    for rank in (128, 512, 1024, 2048):
        f = _routed_chunk(rank)
        o_reads, o_state = _oracle_chunk(f)
        weights_tm = f["weights"].permute(0, 2, 1, 3).contiguous()
        floor, _, _ = sequential_noise_floor(weights_tm, f["decay"].permute(0, 2, 1),
                                             f["schedule"])
        r_k23, s_k23, l_k23 = k23_chunk(f["state"].clone(), f["live"].clone(),
                                        f["weights"], f["decay"], f["schedule"])
        r_k7, s_k7, l_k7 = k7_chunk(f["state"].clone(), f["live"].clone(),
                                    f["weights"], f["decay"], f["schedule"])
        r_routed, s_r, l_r = k7_chunk_routed(f["query"], f["state"].clone(),
                                             f["live"].clone(), f["decay"],
                                             f["schedule"], POLICY.read_scale)
        w_k7 = k7_read_weights(f["query"], f["state"], f["live"], POLICY.read_scale)
        w64 = read_weights(f["query"].to(torch.float64), f["state"].to(torch.float64),
                           f["live"], POLICY.read_scale)
        weights_floor = float((f["weights"] - w64).abs().max())
        rows.append(dict(
            fixture=f"routed chunk (rank {rank}, 64 tokens)",
            slot_blocks=rank // K7_RANK_TILE if rank >= K7_RANK_TILE else 1,
            floor=round(floor, 9),
            k7_vs_k23_given_torch_weights=round(
                float((r_k7 - r_k23).abs().max()), 9),
            k7_vs_k23_x_floor=round(
                float((r_k7 - r_k23).abs().max()) / floor, 2),
            k7_vs_oracle=round(float((r_k7 - o_reads).abs().max()), 9),
            k7_routed_vs_k23_torch_weights=round(
                float((r_routed - r_k23).abs().max()), 9),
            weights_vs_registered=round(
                float((w_k7 - f["weights"]).abs().max()), 9),
            weights_floor_fp64=round(weights_floor, 9),
            state_bit_equal_to_k23=bool(torch.equal(s_k7, s_k23)),
            state_bit_equal_to_oracle=bool(torch.equal(s_k7, o_state)),
            liveness_equal=bool(torch.equal(l_k7, l_k23))))
    return dict(rows=rows, boundary="one chunk, forward only, fp32; the state "
                "walk is K1 (bit-exact by construction); the read gates are the "
                "measured fp32 sequential floor x NOISE_FACTOR (8)")


# ------------------------------------------------------------------------ gate B


def gate_b() -> dict:
    rows = []
    for rank in (128, 1024, 2048):
        generator = torch.Generator().manual_seed(11)
        batch, tokens, heads, value_dim = 2, 16, 2, 128
        shape = (batch, tokens, heads, value_dim)
        query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
        payload = torch.randn(*shape, generator=generator).to(DEVICE)
        state = torch.randn(batch, heads, rank, value_dim,
                           generator=generator).to(DEVICE) * 0.3
        live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) < 0.8
        state = state * live.unsqueeze(-1)
        decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
        f = dict(query=query, payload=payload, state=state, live=live, decay=decay)
        kernel = k7_sequential_walk(query, payload, decay, policy=POLICY,
                                    rank=rank, initial_state=state,
                                    initial_live=live)
        oracle = sequential_oracle(query, payload, decay, policy=POLICY, rank=rank,
                                   initial_state=state, initial_live=live)
        oracle64 = sequential_oracle(query.to(torch.float64), payload.to(torch.float64),
                                     decay.to(torch.float64), policy=POLICY,
                                     rank=rank, initial_state=state.to(torch.float64),
                                     initial_live=live)
        flips = 0
        lanes = 0
        for field in ("slot", "op", "erase", "live_after"):
            d = int((kernel["schedule"].__dict__[field]
                     != oracle["schedule"].__dict__[field]).sum())
            flips += d
            lanes += kernel["schedule"].__dict__[field].numel()
        d = int((kernel["schedule"].payload != oracle["schedule"].payload).sum())
        flips += d
        lanes += kernel["schedule"].payload.numel()
        d = int((kernel["weights"] != oracle["weights"]).sum())
        flips += d
        lanes += kernel["weights"].numel()
        d = int((kernel["live_pre"] != oracle["live_pre"]).sum())
        flips += d
        lanes += kernel["live_pre"].numel()
        floor = float((oracle["reads"]
                       - oracle64["reads"].to(torch.float32)).abs().max())
        deviation = float((kernel["reads"] - oracle["reads"]).abs().max())
        counts = torch.bincount(kernel["schedule"].op[..., 0].reshape(-1).cpu(),
                                minlength=5).tolist()
        rows.append(dict(
            rank=rank, compared_lanes=lanes, flips=flips,
            state_bit_equal=bool(torch.equal(kernel["state"], oracle["state"])),
            live_bit_equal=bool(torch.equal(kernel["live"], oracle["live"])),
            read_floor=round(floor, 9), read_deviation=round(deviation, 9),
            read_x_floor=round(deviation / max(floor, 1e-12), 2),
            op_histogram=dict(zip(("none", "append", "correct", "pop", "merge"),
                                 counts)),
            registers=int(kernel["kernel"].n_regs),
            spills=int(kernel["kernel"].n_spills)))
    # the oracle pinned to the K6 registration's own oracle at rank 128
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import sequential_oracle_scan
    generator = torch.Generator().manual_seed(2)
    rank, tokens, batch, heads, value_dim = 128, 8, 2, 2, 128
    shape = (batch, tokens, heads, value_dim)
    query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(*shape, generator=generator).to(DEVICE)
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=generator).to(DEVICE) * 0.3
    live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) < 0.6
    state = state * live.unsqueeze(-1)
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    o7 = sequential_oracle(query, payload, decay, policy=POLICY, rank=rank,
                           initial_state=state, initial_live=live)
    o6 = sequential_oracle_scan(query, payload, decay, policy=POLICY, rank=rank,
                                rank_max=rank, initial_state=state,
                                initial_live=live, collect=True)
    oracle_pin = dict(
        rank=rank,
        decision_flips=sum(int((o7["schedule"].__dict__[f]
                                != o6["schedule"].__dict__[f]).sum())
                           for f in ("slot", "op", "erase", "live_after")),
        weights_bit_equal=bool(torch.equal(o7["weights"], o6["weights"])),
        live_pre_bit_equal=bool(torch.equal(o7["live_pre"], o6["live_pre"])),
        state_bit_equal=bool(torch.equal(o7["state"], o6["state"])))
    return dict(rows=rows, oracle_pinned_to_k6=oracle_pin,
                boundary="one window scan, forward only, fp32; decisions, "
                "payload lanes, read weights and pre-decision liveness are "
                "torch.equal against the K7 oracle (written to the kernel's "
                "reduction order); the read is at the measured fp32-vs-fp64 floor")


# ------------------------------------------------------------- the refusal removed


def refusal_removed(reps: int) -> dict:
    rows = []
    batch, tokens, heads, value_dim, chunk = 8, 2048, 2, 128, 128
    for rank in (512, 1024, 2048):
        generator = torch.Generator().manual_seed(rank)
        state = torch.randn(batch, heads, rank, value_dim,
                            generator=generator).to(DEVICE) * 0.3
        live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) < 0.7
        state = state * live.unsqueeze(-1)
        query = torch.randn(batch, heads, tokens, value_dim,
                            generator=generator).to(DEVICE) * 0.5
        weights = torch.rand(batch, heads, tokens, rank,
                             generator=generator).to(DEVICE)
        # the synthetic five-op schedule the rank sweeps have always used
        from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender \
            import random_schedule
        schedule, _, _, decay_tm = random_schedule(batch, tokens, heads, rank,
                                                   value_dim, seed=rank, decay=0.95)
        decay = decay_tm.permute(0, 2, 1).contiguous()
        wall_routed, device_routed = _reps(
            lambda: k7_block_scan_routed(query, state, live, decay, schedule,
                                         POLICY.read_scale, chunk), reps)
        wall_given, device_given = _reps(
            lambda: k7_block_scan(state, live, weights, decay, schedule, chunk), reps)
        wall_k23, device_k23 = _reps(
            lambda: chunked_block_scan(state, live, weights, decay, schedule, chunk),
            reps)
        wall_torch_w, device_torch_w = _reps(
            lambda: read_weights(query[:, :, :chunk], state, live, POLICY.read_scale),
            reps)
        torch_w_block = round(wall_torch_w["mean"] * (tokens // chunk), 3)
        # the memory arithmetic of one k7 chunk (T_c = 128)
        tc, rows_ = chunk, batch * heads
        sb = max(1, _pow2(rank) // K7_RANK_TILE)
        channel = batch * heads * tc * rank * 4
        partials = batch * heads * sb * tc * _pow2(value_dim) * 4
        intra_partials = batch * heads * -(-tc * 2 // K7_STEP_TILE) * tc \
            * _pow2(value_dim) * 4
        rows.append(dict(
            rank=rank,
            persistent_kernel_refusal=not persistent_available(rank, value_dim),
            refusal_message=(f"the persistent kernel needs next_pow2({rank}) * 16 = "
                             f"{_pow2(rank) * 16} register-resident floats and its "
                             f"budget is 8192; rank {rank} at value dim {value_dim} "
                             f"does not fit"),
            k7_routed_block=dict(wall_ms=wall_routed, device_ms=device_routed),
            k7_given_weights_block=dict(wall_ms=wall_given, device_ms=device_given),
            k23_twin_block=dict(wall_ms=wall_k23, device_ms=device_k23),
            torch_weights_per_chunk_ms=wall_torch_w,
            torch_weights_per_block_ms=torch_w_block,
            per_chunk_channels_mb=dict(
                scores=round(channel / 1e6, 2), strength=round(channel / 1e6, 2),
                weights=round(channel / 1e6, 2),
                inter_partials=round(partials / 1e6, 2),
                intra_partials=round(intra_partials / 1e6, 2),
                state=round(rows_ * rank * value_dim * 4 / 1e6, 2)),
            reps=reps))
    # the walk's scan walls
    walk_rows = []
    for rank in (128, 1024, 2048):
        generator = torch.Generator().manual_seed(3)
        shape = (batch, tokens, heads, value_dim)
        query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
        payload = torch.randn(*shape, generator=generator).to(DEVICE)
        state = torch.randn(batch, heads, rank, value_dim,
                            generator=generator).to(DEVICE) * 0.3
        live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) < 0.6
        state = state * live.unsqueeze(-1)
        decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
        wall, device = _reps(
            lambda: k7_sequential_walk(query, payload, decay, policy=POLICY,
                                      rank=rank, initial_state=state,
                                      initial_live=live, collect=False), reps)
        walk_rows.append(dict(
            rank=rank, wall_ms=wall, device_ms=device,
            launches_per_scan=1,
            tokens_per_s=round(batch * tokens * heads / (wall["mean"] / 1e3)),
            traffic_gb_per_scan=round(
                3 * rank * value_dim * 4 * tokens * batch * heads / 1e9, 1)))
    return dict(
        block_boundary=dict(batch=batch, tokens=tokens, heads=heads,
                            value_dim=value_dim, chunk_tokens=chunk,
                            dtype="fp32", forward_only=True,
                            schedule="synthetic five-op, the rank-sweep fixtures' own"),
        block_rows=rows, walk_rows=walk_rows,
        note="the k23 twin was never refused (its rank axis is an in-program loop); "
             "the refusal this receipt records is the persistent kernel's by-name "
             "one, and the k7 rows show the snap set executing without it")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(0)
    report = dict(
        gate="K7: the high-rank store",
        gate_a_chunked_value_scan=gate_a(),
        gate_b_sequential_walk=gate_b(),
        refusal_removed=refusal_removed(args.reps))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}")
    for section in ("gate_a_chunked_value_scan", "gate_b_sequential_walk",
                    "refusal_removed"):
        print(f"\n== {section} ==")
        print(json.dumps(report[section], indent=1)[:2400])


if __name__ == "__main__":
    main()
