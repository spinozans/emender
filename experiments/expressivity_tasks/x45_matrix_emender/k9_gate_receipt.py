"""K9 gates receipt: parity, the number, and the growth pathology.

Writes ``receipts/k9_gates.json``:

* **gate B** -- the 0-flip bar: every recorded channel bit-equal against BOTH
  references (the K7 eager oracle and the K7 global-tiled walk) at ranks
  {256, 1024, 2048}, the exit state and liveness bit-equal, the read at the
  MEASURED fp32-vs-fp64 floor (floors re-measured first), determinism across
  two full scans, the graph-vs-eager bit-equality, the register/spill counts
  of every kernel, the honest op histograms of the random fixtures.
* **gate C** -- the number: the probe-shape walls (batch 8, 2 048 tokens,
  2 heads, fp32, forward only) before (the K7 walk re-measured, with the
  banked 374.3/414 ms receipts cited) and after (the K9 distributed walk:
  capture cost, replay-only vs full call, the tile-width measurement), at
  ranks {512, 1024, 2048}; the honest scaling law (per-token barrier cost x T
  + score compute, fitted); the projected whole-arm tok/s against the banked
  K8 arms (775 tok/s grow / 821.7 fixed-2048), with the accounting stated.
* **gate D** -- the pathology: growth re-dispatch across a snap change at the
  span-API level (0 flips span by span, the zero-injection receipts, ragged
  windows), and the per-snap occupancy table {128..2048}.

Forward only; every number one leased card, seed-stated, as the lane's.
"""
from __future__ import annotations

import json
import time

import torch

from .k7_high_rank_store import (k7_sequential_walk, sequential_oracle)
from .k9_distributed_walk import (K9_BLOCK_SLOTS, k9_sequential_walk,
                                 k9_span_walk, reset_k9_graphs)
from .schedule import RoutingPolicy

DEVICE = "cuda"
POLICY = RoutingPolicy(ops=True)


def _reps(fn, count: int, warmup: int = 2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    walls, devices = [], []
    for _ in range(count):
        t0 = time.perf_counter()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        devices.append(ev0.elapsed_time(ev1))
    walls.sort()
    devices.sort()
    mid = count // 2
    return dict(mean=round(sum(walls) / len(walls), 2),
                median=round(walls[mid], 2),
                device_median_ms=round(devices[mid], 2))


def _walk_fixture(rank: int, batch: int, tokens: int, heads: int, value_dim: int,
                  seed: int, live_fraction: float):
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, tokens, heads, value_dim)
    query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(*shape, generator=generator).to(DEVICE)
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=generator).to(DEVICE) * 0.3
    live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) \
        < live_fraction
    state = state * live.unsqueeze(-1)
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    return dict(query=query, payload=payload, state=state, live=live,
                decay=decay)


def _flips(kernel: dict, oracle: dict) -> int:
    total = 0
    for field in ("slot", "op", "erase", "live_after"):
        total += int((kernel["schedule"].__dict__[field]
                     != oracle["schedule"].__dict__[field]).sum())
    total += int((kernel["schedule"].payload != oracle["schedule"].payload).sum())
    total += int((kernel["weights"] != oracle["weights"]).sum())
    total += int((kernel["live_pre"] != oracle["live_pre"]).sum())
    return total


def _oracle64(f: dict, rank: int):
    f64 = dict(f)
    f64.update(query=f["query"].to(torch.float64),
               payload=f["payload"].to(torch.float64),
               decay=f["decay"].to(torch.float64),
               state=f["state"].to(torch.float64))
    return sequential_oracle(f64["query"], f64["payload"], f64["decay"],
                             policy=POLICY, rank=rank,
                             initial_state=f64["state"],
                             initial_live=f64["live"])


# ------------------------------------------------------------------------ gate B


def gate_b() -> dict:
    rows = []
    for rank in (256, 1024, 2048):
        f = _walk_fixture(rank, batch=2, tokens=16, heads=2, value_dim=128,
                          seed=11, live_fraction=0.8)
        kw = dict(policy=POLICY, rank=rank, initial_state=f["state"],
                  initial_live=f["live"])
        reset_k9_graphs()
        k9 = k9_sequential_walk(f["query"], f["payload"], f["decay"], **kw)
        eager = k9_sequential_walk(f["query"], f["payload"], f["decay"],
                                  graph=False, **kw)
        k7 = k7_sequential_walk(f["query"], f["payload"], f["decay"], **kw)
        oracle = sequential_oracle(f["query"], f["payload"], f["decay"], **kw)
        oracle64 = _oracle64(f, rank)
        floor = float((oracle["reads"]
                       - oracle64["reads"].to(torch.float32)).abs().max())
        deviation = float((k9["reads"] - oracle["reads"]).abs().max())
        vs_k7 = float((k9["reads"] - k7["reads"]).abs().max())
        second = k9_sequential_walk(f["query"], f["payload"], f["decay"], **kw)
        counts = torch.bincount(k9["schedule"].op[..., 0].reshape(-1).cpu(),
                               minlength=5).tolist()
        handles = k9["slot"].handles
        rows.append(dict(
            rank=rank,
            flips_vs_oracle=_flips(k9, oracle),
            flips_vs_k7_walk=_flips(k9, k7),
            state_bit_equal=bool(torch.equal(k9["state"], oracle["state"])),
            live_bit_equal=bool(torch.equal(k9["live"], oracle["live"])),
            read_floor=round(floor, 9),
            read_deviation_vs_oracle=round(deviation, 9),
            read_x_floor=round(deviation / max(floor, 1e-12), 2),
            read_deviation_vs_k7_walk=round(vs_k7, 9),
            graph_vs_eager_bit_equal=(
                _flips(k9, eager) == 0
                and bool(torch.equal(k9["reads"], eager["reads"]))),
            determinism_two_scans=(
                _flips(k9, second) == 0
                and bool(torch.equal(k9["reads"], second["reads"]))
                and bool(torch.equal(k9["state"], second["state"]))),
            op_histogram=dict(zip(("none", "append", "correct", "pop",
                                   "merge"), counts)),
            registers={name: int(h.n_regs) for name, h in handles.items()},
            spills={name: int(h.n_spills) for name, h in handles.items()}))
    return dict(rows=rows,
                boundary="one window scan, forward only, fp32; decisions, "
                "payload lanes, read weights and pre-decision liveness are "
                "torch.equal against BOTH the K7 oracle and the K7 "
                "global-tiled walk; the read is at the measured "
                "fp32-vs-fp64 floor (re-measured per fixture)",
                note="the random fixtures saturate in APPEND (the fresh-store "
                     "regime, the lane's own K6/K7 finding); the five-op "
                     "coverage is the crafted-fixture test suite's")


# ------------------------------------------------------------------------ gate C


def gate_c(reps: int) -> dict:
    batch, tokens, heads, value_dim = 8, 2048, 2, 128
    wall_rows = []
    replays = {}
    for rank in (512, 1024, 2048):
        f = _walk_fixture(rank, batch, tokens, heads, value_dim, seed=3,
                          live_fraction=0.6)
        kw = dict(policy=POLICY, rank=rank, initial_state=f["state"],
                  initial_live=f["live"], collect=False)
        reset_k9_graphs()
        t0 = time.perf_counter()
        k9_first = k9_sequential_walk(f["query"], f["payload"], f["decay"],
                                      graph=True, **kw)
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - t0) * 1e3
        k9_wall = _reps(lambda: k9_sequential_walk(
            f["query"], f["payload"], f["decay"], graph=True, **kw), reps)
        # the bare replay in a tight loop (an L2-replacement observation: the
        # full call's copy-in of the state re-populates the L2 reuse set, and
        # the bare loop's replay runs ~14% slower; NOT chased further -- the
        # full call is the deployed path and the honest number, I/O included)
        slot = k9_first["slot"]
        slot.query_s.copy_(f["query"]); slot.payload_s.copy_(f["payload"])
        slot.decay_s.copy_(f["decay"]); slot.work_state.copy_(f["state"])
        slot.work_live.copy_(f["live"].view(torch.int8))
        replay = _reps(lambda: slot.graph.replay(), reps)
        replays[rank] = replay
        k7_wall = _reps(lambda: k7_sequential_walk(
            f["query"], f["payload"], f["decay"], **kw), reps)
        rp = 1 << (rank - 1).bit_length()
        nt = rp // K9_BLOCK_SLOTS
        wall_rows.append(dict(
            rank=rank,
            k7_wall_ms=k7_wall["median"], k7_device_ms=k7_wall["device_median_ms"],
            k9_wall_ms=k9_wall["median"], k9_device_ms=k9_wall["device_median_ms"],
            k9_bare_replay_ms=replay["median"],
            k9_bare_replay_note="the bare replay in a tight loop; the full "
                                "call's state copy-in re-populates the L2 "
                                "reuse set and the full call measures FASTER "
                                "-- the full-call wall is the number",
            speedup_wall=round(k7_wall["median"] / k9_wall["median"], 2),
            tok_per_s_k9=round(batch * tokens * heads
                               / (k9_wall["median"] / 1e3)),
            capture_ms_one_time=round(capture_ms, 1),
            graph_nodes=3 * tokens + 1,
            slot_blocks_per_row=nt,
            per_token_us=round(k9_wall["median"] * 1e3 / tokens, 2)))
    # the tile-width measurement (the brief's smem-design 128 vs the lane's
    # proven 64-slot walk tile)
    rank = 2048
    f = _walk_fixture(rank, batch, tokens, heads, value_dim, seed=3,
                      live_fraction=0.6)
    kw = dict(policy=POLICY, rank=rank, initial_state=f["state"],
              initial_live=f["live"], collect=False)
    tile_rows = []
    for br in (K9_BLOCK_SLOTS, 128):
        reset_k9_graphs()
        k9_sequential_walk(f["query"], f["payload"], f["decay"], graph=True,
                           br=br, **kw)
        w = _reps(lambda: k9_sequential_walk(
            f["query"], f["payload"], f["decay"], graph=True, br=br, **kw), reps)
        tile_rows.append(dict(block_slots=br, wall_ms=w["median"]))
    # the honest scaling law: per-token = barrier + decision (NT-independent)
    # + score/apply compute (proportional to NT)
    nts = [1 << ((r - 1).bit_length() - 7) for r in (512, 1024, 2048)]
    per_token = [r["per_token_us"] for r in wall_rows]
    n0, n1, n2 = nts
    b_per_block = (per_token[2] - per_token[0]) / (n2 - n0)
    a_fixed = per_token[0] - b_per_block * n0
    scaling_law = dict(
        per_token_us_at={str(r): p for r, p in zip((512, 1024, 2048), per_token)},
        fit=f"per_token_us ~= {a_fixed:.2f} + {b_per_block:.3f} * NT",
        barrier_ms_per_scan=round(tokens * 3 * 0.56 / 1e3, 2),
        barrier_note="the graph node cost 0.56 us (gate-0 receipt "
                     "k9_barrier_economics.json) x 2048 tokens x 3 "
                     "launch-boundary syncs")

    # the whole-arm projection, from the banked K8 receipts' own accounting
    # (stated, not re-measured): the fixed-2048 arm's step wall 135.4 s =
    # 24368.7/180 s (receipts/k8_probe_fixed2048.json); the probe-shape scan
    # covers 8 windows, windows_per_step = 64, 24 layers -> 192 scans/step;
    # the banked rank-2048 forward walk 414 ms/scan (k8_gates.json gate D),
    # the backward 270 ms/scan (K9 is forward-only: untouched).
    fixed_step_s = 24368.7 / 180
    grow_step_s = 25840.6 / 180
    scans_per_step = 64 // 8 * 24
    fwd_share_s = scans_per_step * 0.414
    bwd_share_s = scans_per_step * 0.270
    k9_2048_ms = wall_rows[-1]["k9_wall_ms"]
    new_fwd = scans_per_step * k9_2048_ms / 1e3
    targets_per_step = 20_000_000 / 180
    new_fixed = fixed_step_s - fwd_share_s + new_fwd
    new_grow = grow_step_s - fwd_share_s + new_fwd
    whole_arm = dict(
        accounting=("derived from the banked receipts: fixed-2048 arm step "
                    f"{fixed_step_s:.1f} s, grow arm {grow_step_s:.1f} s; "
                    f"{scans_per_step} rank-2048 forward scans/step x 414 ms "
                    f"= {fwd_share_s:.1f} s forward-walk share (+{bwd_share_s:.0f} s "
                    "backward, untouched by K9); rest = matmuls etc."),
        fixed2048_arm=dict(banked_tok_s=821.7, banked_step_s=round(fixed_step_s, 1),
                          projected_step_s=round(new_fixed, 1),
                          projected_tok_s=round(targets_per_step / new_fixed, 1),
                          speedup=round(fixed_step_s / new_fixed, 2)),
        grow_arm=dict(banked_tok_s=774.9, banked_step_s=round(grow_step_s, 1),
                      projected_step_s=round(new_grow, 1),
                      projected_tok_s=round(targets_per_step / new_grow, 1),
                      speedup=round(grow_step_s / new_grow, 2)),
        notes=("K9 is forward-only; after it, the hi backward (270 ms/scan, "
               "k8's own kernels) becomes the dominant scan cost. The "
               "projection excludes the grow arm's probe pass (which also "
               "speeds up). The C3 saturation probe at rank 2048 becomes "
               f"affordable: a 180-step arm costs ~{180 * new_fixed / 3600:.1f} h "
               "instead of the banked ~7.4 h."))

    return dict(fixture=dict(batch=batch, tokens=tokens, heads=heads,
                             value_dim=value_dim, dtype="fp32",
                             forward_only=True,
                             block_slots_default=K9_BLOCK_SLOTS),
                walls=wall_rows,
                banked_before=dict(k7_receipted_scan_ms=374.3,
                                  k8_gate_d_scan_ms=414,
                                  source="receipts/k7_high_rank.json, "
                                         "receipts/k8_gates.json"),
                tile_width_measurement=tile_rows,
                scaling_law=scaling_law,
                whole_arm_projection=whole_arm)


# ------------------------------------------------------------------------ gate D


def gate_d() -> dict:
    from .k8_growth import _pad_live, _pad_state
    batch, tokens, heads, value_dim = 2, 960, 2, 128
    generator = torch.Generator().manual_seed(7)
    shape = (batch, tokens, heads, value_dim)
    query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(*shape, generator=generator).to(DEVICE)
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    # ragged span boundaries across snaps -- the growth events' own geometry
    spans = [(0, 192, 128), (192, 448, 256), (448, 700, 512), (700, 960, 1024)]
    result = k9_span_walk(query, payload, decay, policy=POLICY, spans=spans)
    state = torch.zeros(batch, heads, 128, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, 128, dtype=torch.bool, device=DEVICE)
    span_rows, flips = [], 0
    zero_injection = []
    for index, (t0, t1, rank) in enumerate(spans):
        oracle = sequential_oracle(query[:, t0:t1], payload[:, t0:t1],
                                   decay[:, t0:t1], policy=POLICY, rank=rank,
                                   initial_state=state, initial_live=live)
        segment = result["collected"][index]
        seg_flips = _flips(dict(schedule=segment["schedule"],
                                 weights=segment["weights"],
                                 live_pre=segment["live_pre"]), oracle)
        flips += seg_flips
        oracle64 = sequential_oracle(
            query[:, t0:t1].to(torch.float64), payload[:, t0:t1].to(torch.float64),
            decay[:, t0:t1].to(torch.float64), policy=POLICY, rank=rank,
            initial_state=state.to(torch.float64), initial_live=live)
        floor = float((oracle["reads"]
                       - oracle64["reads"].to(torch.float32)).abs().max())
        deviation = float((result["reads"][:, t0:t1]
                           - oracle["reads"]).abs().max())
        if index > 0:
            grown_from = spans[index - 1][2]
            fresh = segment["live_pre"][:, 0, :, grown_from:]
            zero_injection.append(dict(
                span=index, rank=rank, fresh_rows=int(fresh.numel()),
                dead_at_entry=bool(not fresh.any())))
        span_rows.append(dict(t0=t0, t1=t1, rank=rank, flips=seg_flips,
                              read_floor=round(floor, 9),
                              read_deviation=round(deviation, 9),
                              read_x_floor=round(deviation / max(floor, 1e-12), 2)))
        state, live = oracle["state"], oracle["live"]
        if index + 1 < len(spans) and spans[index + 1][2] > rank:
            state = _pad_state(state, spans[index + 1][2])
            live = _pad_live(live, spans[index + 1][2])
    chain = dict(total_flips=flips,
                 exit_state_bit_equal=bool(torch.equal(result["state"], state)),
                 exit_live_bit_equal=bool(torch.equal(result["live"], live)))
    # ragged single windows
    ragged = []
    for t in (1, 33, 130):
        generator = torch.Generator().manual_seed(5)
        f = _walk_fixture(256, batch=2, tokens=t, heads=2, value_dim=128,
                          seed=5, live_fraction=0.8)
        kw = dict(policy=POLICY, rank=256, initial_state=f["state"],
                  initial_live=f["live"])
        reset_k9_graphs()
        k9 = k9_sequential_walk(f["query"], f["payload"], f["decay"], **kw)
        oracle = sequential_oracle(f["query"], f["payload"], f["decay"], **kw)
        ragged.append(dict(tokens=t, flips=_flips(k9, oracle)))
    # occupancy at every ACTIVATED snap (rank x value_dim above the on-chip
    # register budget; snaps at/below route to the existing K6/K7 paths)
    occupancy = []
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    for rank in (256, 512, 1024, 2048):
        rp = 1 << (rank - 1).bit_length()
        nt = max(1, rp // K9_BLOCK_SLOTS)
        ctas = 16 * nt
        occupancy.append(dict(rank=rank, slot_blocks_per_row=nt,
                               sweep_ctas=ctas,
                               waves_on_sm=round(ctas / sm_count, 2)))
    return dict(growth_spans=span_rows, chain=chain,
                zero_injection=zero_injection,
                ragged_windows=ragged,
                occupancy_per_snap=occupancy,
                note="growth re-dispatch composes at the span-API level: "
                     "spans at/below the on-chip budget run the existing K7 "
                     "path, grown spans the K9 walk, the state chained "
                     "through k8_growth's own zero-injection pads; every "
                     "snap is its own cached graph (more blocks per snap, "
                     "state copy-in at the snap change)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    receipt = dict(
        gate="K9: the distributed-persistent walk",
        barrier_economics="receipts/k9_barrier_economics.json (gate 0)",
        device=torch.cuda.get_device_name(0),
        gate_b_sequential_walk=gate_b(),
        gate_c_the_number=gate_c(args.reps),
        gate_d_growth_pathology=gate_d(),
    )
    text = json.dumps(receipt, indent=1)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"[k9] receipt written to {args.out}")


if __name__ == "__main__":
    main()
