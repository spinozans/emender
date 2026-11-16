"""K8 gate receipts: the growth capability's evidence at the registered shapes.

Writes ``receipts/k8_gates.json`` with, in order:

* **gate A -- the legality and parity of the wired growth**:
  - the never-fires row (``grow=True``, fewer than ``grow_window`` boundaries):
    bit-equal to K6's own path on every channel (the same launch, by
    construction);
  - the grown scan at the probe shape (batch 8, 2048 tokens, start rank 128,
    cap 2048): the span table, the growth events and the per-boundary rates
    (the registration's "reported per chunk"), the zero-injection receipts
    (new rows dead until first written; the boundary state bit-equal to the
    fixed-rank twin's own segment checkpoint -- old rows unchanged by the
    event), the probe/real schedule agreement, and determinism across two
    full growing scans;
  - the per-span decision parity vs the eager oracles (0 flips, the mixed
    K6/K7 fixed points), the read at the measured floor, and the hi span
    bit-equal to ``k7_sequential_walk`` (the same kernel, launched directly).
* **gate B -- gradients**: the hi backward at rank 256 and 2048 vs the eager
  oracle that replays the same span table, floor first (fp32 oracle vs its own
  fp64 twin, NOISE_FACTOR x); the chain ACROSS a growth boundary (the
  exit-state gradient seeding the carries -- K6's own op drops it, its windows
  never chain); the prep's recomputed read weights bit-equal the forward's own
  channel.
* **gate D -- the honest walls and memory**: forward and backward per snap at
  the probe shape (through the op pair, autograd live), the probe pass's cost,
  the grown scan's cost, the peak memory arithmetic (the backward's
  recomputed checkpoints against the forward-saved alternative), and the
  launch counts.

Usage (the leased card):

    CUDA_VISIBLE_DEVICES=4 /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k8_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k8_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import NONE
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _seq_forward_launch, _seq_warps, _round_f32, sequential_oracle_scan,
    sequential_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k7_high_rank_store import (
    k7_sequential_walk, sequential_oracle)
from experiments.expressivity_tasks.x45_matrix_emender.k8_growth import (
    GROW_BOUNDARY_TOKENS, _hi_walk_launch, growing_scan,
    oracle_growing_scan, traj_checkpoints, traj_segment, _hi_prep)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
NOISE_FACTOR = 8.0
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")

PROBE_SHAPE = dict(batch=8, tokens=2048, heads=2, value_dim=128)
GROWING = RoutingPolicy(ops=True, grow=True)
QUIET = RoutingPolicy(ops=True, grow=False)


def pol32(policy):
    return (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
            _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
            _round_f32(policy.read_scale))


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed,
                   device="cuda", bf16=False):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to(device) * 0.5
    directions = torch.randn(distinct, value_dim, generator=g).to(device)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = directions[choice]
    decay = torch.rand(batch, tokens, heads, generator=g).to(device) \
        .clamp(0.9, 1.0)
    if bf16:
        query, payload, decay = (t.bfloat16() for t in (query, payload, decay))
    return query, payload, decay


# ------------------------------------------------------------------- gate A


def gate_a_never_fires() -> dict:
    """The never-fires row: bit-equal to K6's own path, every channel."""
    batch, tokens = 4, 192
    heads, rank, value_dim = 2, 128, 128
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                            512, 77)
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.99)
    grown = growing_scan(query, payload, decay, policy=policy,
                         start_rank=rank, rank_cap=2048)
    k6 = sequential_scan(query, payload, decay, policy=QUIET, rank=rank,
                         rank_max=rank)
    return dict(
        gate="A: grow=True with fewer than grow_window boundaries is K6",
        fixture=dict(batch=batch, tokens=tokens, heads=heads, rank=rank,
                     value_dim=value_dim, boundaries=tokens // 64,
                     grow_window=policy.grow_window),
        spans=[list(s) for s in grown.stats.spans], events=grown.stats.events,
        bit_equal_reads=bool(torch.equal(grown.reads, k6.reads)),
        bit_equal_state=bool(torch.equal(grown.state, k6.state)),
        bit_equal_live=bool(torch.equal(grown.live, k6.live)))


def gate_a_grown_probe_shape() -> dict:
    """The grown scan at the probe shape: events, zero-injection, prefix
    bit-equality, probe/real agreement, determinism."""
    batch, tokens, heads, value_dim = (PROBE_SHAPE["batch"],
                                       PROBE_SHAPE["tokens"],
                                       PROBE_SHAPE["heads"],
                                       PROBE_SHAPE["value_dim"])
    start_rank, cap = 128, 2048
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                            512, 5)
    policy = RoutingPolicy(ops=True, grow=True)
    result = growing_scan(query, payload, decay, policy=policy,
                          start_rank=start_rank, rank_cap=cap, collect=True)
    spans, events = result.stats.spans, result.stats.events
    collected = result.collected
    # the zero-injection receipts, per event
    zero_rows = []
    for index in range(1, len(collected)):
        prev_rank = collected[index - 1]["rank"]
        span = collected[index]
        first_write = None
        schedule = span["schedule"]
        for t in range(span["t1"] - span["t0"]):
            wrote_new = (schedule.op[:, t, :, 0] != NONE) \
                & (schedule.slot[:, t, :, 0] >= prev_rank)
            wrote_new |= (schedule.op[:, t, :, 1] != NONE) \
                & (schedule.slot[:, t, :, 1] >= prev_rank)
            if bool(wrote_new.any()):
                first_write = t
                break
        live_new_before = span["live_pre"][:, :first_write if first_write
                                           else None, :, prev_rank:]
        zero_rows.append(dict(
            boundary=span["t0"], rank_before=prev_rank, rank=span["rank"],
            first_write_into_new_rows=(None if first_write is None
                                       else span["t0"] + first_write),
            new_rows_live_before_first_write=bool(live_new_before.any()),
            new_rows_dead_at_entry=bool(
                not span["live_pre"][:, 0, :, prev_rank:].any())))
    # the prefix bit-equality: the twin at the START rank (K6's own launch)
    state = torch.zeros(batch, heads, start_rank, value_dim,
                        dtype=torch.float32, device="cuda")
    live0 = torch.zeros(batch, heads, start_rank, dtype=torch.bool,
                        device="cuda")
    twin = _seq_forward_launch(query, payload, decay, state, live0,
                               pol32(QUIET), start_rank, 64, collect=True,
                               warps=_seq_warps(start_rank, value_dim),
                               stages=3)
    t_star = events[0]["boundary"]
    k = t_star // GROW_BOUNDARY_TOKENS
    prefix_equal = True
    for field in SCHEDULE_FIELDS + ("payload",):
        a = getattr(collected[0]["schedule"], field)
        b = twin[field][:, :t_star]
        prefix_equal = prefix_equal and bool(torch.equal(a, b))
    prefix_equal = prefix_equal and bool(torch.equal(
        collected[0]["weights"], twin["weights"][:, :t_star]))
    # determinism: every channel of the collect chain, twice
    again = growing_scan(query, payload, decay, policy=policy,
                         start_rank=start_rank, rank_cap=cap, collect=True)
    determinism = again.stats.spans == spans
    for a, b in zip(collected, again.collected):
        for field in SCHEDULE_FIELDS + ("payload",):
            determinism = determinism and bool(
                torch.equal(getattr(a["schedule"], field),
                            getattr(b["schedule"], field)))
        for key in ("weights", "live_pre", "state", "live"):
            determinism = determinism and bool(torch.equal(a[key], b[key]))
    determinism = determinism and bool(torch.equal(result.reads,
                                                   again.reads))
    return dict(
        gate="A: the grown scan at the probe shape (batch 8, 2048 tokens)",
        fixture=dict(**PROBE_SHAPE, start_rank=start_rank, rank_cap=cap,
                     distinct=512, seed=5, policy=GROWING.record()),
        spans=[list(s) for s in spans],
        events=events,
        rates_note=("the per-boundary append-collision rates, reported per "
                    "chunk as registered; with the registered "
                    "grow_window_tokens=512 and the statistic's zero-initialised "
                    "last_write, every write inside the first 512 tokens of a "
                    "window counts as a collision (rate 1.0) -- the REGISTERED "
                    "implementation's own behaviour, inherited verbatim and "
                    "stated, not engineered around"),
        rates=[list(r) for r in result.stats.rates],
        zero_injection=zero_rows,
        prefix=dict(
            fire_boundary=t_star,
            span1_schedule_bit_equal_to_twin_prefix=prefix_equal,
            boundary_state_bit_equal_to_twin_checkpoint=bool(torch.equal(
                collected[0]["state"], twin["ckpt_state"][k])),
            boundary_live_bit_equal=bool(torch.equal(
                collected[0]["live"], twin["ckpt_live"][k]))),
        determinism_bit_equal=bool(determinism),
        final_rank=result.rank)


def gate_a_span_parity() -> dict:
    """Per-span decision parity vs the eager oracles, the read at the floor,
    and the hi span bit-equal to k7_sequential_walk."""
    batch, tokens, heads, value_dim = 2, 2048, 2, 128
    start_rank, cap = 128, 2048
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           512, 21)
    policy = RoutingPolicy(ops=True, grow=True)
    result = growing_scan(query, payload, decay, policy=policy,
                          start_rank=start_rank, rank_cap=cap, collect=True)
    oracle = oracle_growing_scan(query, payload, decay, policy=policy,
                                 spans=result.stats.spans, collect=True)
    with torch.no_grad():
        oracle64 = oracle_growing_scan(query.double(), payload.double(),
                                       decay.double(), policy=policy,
                                       spans=result.stats.spans,
                                       collect=False)
    rows = []
    total_flips = 0
    total_lanes = 0
    for index, span in enumerate(result.collected):
        ours, ref = span["schedule"], oracle["per_span"][index]["schedule"]
        flips = 0
        for field in SCHEDULE_FIELDS + ("payload",):
            flips += int((getattr(ref, field) != getattr(ours, field)).sum())
            total_lanes += getattr(ref, field).numel()
        total_flips += flips
        floor = (oracle["reads"][:, span["t0"]:span["t1"]]
                 - oracle64["reads"][:, span["t0"]:span["t1"]].float()) \
            .abs().max().item()
        deviation = (oracle["reads"][:, span["t0"]:span["t1"]].float()
                     - result.reads[:, span["t0"]:span["t1"]]) \
            .abs().max().item()
        rows.append(dict(span=[span["t0"], span["t1"], span["rank"]],
                         decision_flips=flips,
                         weights_bit_equal=bool(torch.equal(
                             span["weights"],
                             oracle["per_span"][index]["weights"])),
                         read_floor=floor, read_deviation=deviation,
                         deviation_over_floor=deviation / max(floor, 1e-30)))
    # the hi span bit-equal to k7_sequential_walk (the same kernel), replayed
    # from the span's own ENTRY state (the previous span's exit, padded)
    from experiments.expressivity_tasks.x45_matrix_emender.k8_growth import (
        _pad_live, _pad_state)
    hi_index = [i for i, s in enumerate(result.collected)
                if s["rank"] > 128][-1]
    hi_span = result.collected[hi_index]
    prev_rank = result.collected[hi_index - 1]["rank"]
    state = _pad_state(result.collected[hi_index - 1]["state"],
                       hi_span["rank"])
    live0 = _pad_live(result.collected[hi_index - 1]["live"], hi_span["rank"])
    t0, t1 = hi_span["t0"], hi_span["t1"]
    ref = k7_sequential_walk(query[:, t0:t1], payload[:, t0:t1],
                             decay[:, t0:t1], policy=QUIET,
                             rank=hi_span["rank"], initial_state=state,
                             initial_live=live0, collect=True)
    hi_bit_equal = bool(torch.equal(ref["reads"],
                                    result.reads[:, t0:t1])) \
        and all(torch.equal(getattr(ref["schedule"], f),
                            getattr(hi_span["schedule"], f))
                for f in SCHEDULE_FIELDS + ("payload",))
    return dict(
        gate="A: per-span parity vs the eager oracles (the mixed K6/K7 fixed "
             "points), reads at the floor",
        fixture=dict(batch=batch, tokens=tokens, heads=heads,
                     value_dim=value_dim, start_rank=start_rank,
                     rank_cap=cap, distinct=512, seed=21),
        spans=[list(s) for s in result.stats.spans],
        total_decision_flips=total_flips, total_lanes=total_lanes,
        per_span=rows,
        hi_span_bit_equal_to_k7_walk=hi_bit_equal)


# ------------------------------------------------------------------- gate B


def _oracle_grads(query, payload, decay, upstream, policy, spans, dtype,
                  initial_state=None):
    q = query.clone().to(dtype).requires_grad_(True)
    p = payload.clone().to(dtype).requires_grad_(True)
    d = decay.clone().to(dtype).requires_grad_(True)
    s = (None if initial_state is None
         else initial_state.clone().to(dtype).requires_grad_(True))
    ref = oracle_growing_scan(q, p, d, policy=policy, spans=spans,
                              collect=False, initial_state=s)
    (ref["reads"] * upstream.to(dtype)).sum().backward()
    return q.grad, p.grad, d.grad, (None if s is None else s.grad)


def _floor_rows(kern, o32, o64, names=("dquery", "dpayload", "ddecay")):
    rows = []
    for name, k, a, b in zip(names, kern, o32, o64):
        floor = (a.float() - b.float()).abs().max().item()
        deviation = (k.float() - a.float()).abs().max().item()
        rows.append(dict(output=name, floor=floor, deviation=deviation,
                         deviation_over_floor=deviation / max(floor, 1e-30),
                         within_noise_factor=bool(
                             deviation <= NOISE_FACTOR * max(floor, 1e-30))))
    return rows


def gate_b_hi_spans() -> dict:
    """The hi backward at rank 256 and 2048 vs the eager oracle, floors
    first; the prep's recomputed weights bit-equal the forward's channel."""
    rows = []
    for rank in (256, 2048):
        batch, tokens, heads, value_dim = 2, 128, 2, 128
        query, payload, decay = distinct_stream(batch, tokens, heads,
                                                value_dim, 96, 41)
        g = torch.Generator().manual_seed(1234)
        upstream = torch.randn(batch, tokens, heads, value_dim,
                               generator=g).to("cuda")
        spans = [(0, tokens, rank)]
        o32 = _oracle_grads(query, payload, decay, upstream, QUIET, spans,
                            torch.float32)
        o64 = _oracle_grads(query, payload, decay, upstream, QUIET, spans,
                            torch.float64)
        q = query.clone().requires_grad_(True)
        p = payload.clone().requires_grad_(True)
        d = decay.clone().requires_grad_(True)
        res = growing_scan(q, p, d, policy=QUIET, start_rank=rank,
                           rank_cap=rank)
        (res.reads * upstream).sum().backward()
        rows.append(dict(
            rank=rank, tokens=tokens,
            outputs=_floor_rows((q.grad, p.grad, d.grad), o32, o64)))
    # the prep's recomputed weights, bit-equal the forward's own channel
    rank, batch, tokens, heads, value_dim = 512, 2, 256, 2, 128
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           96, 43)
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device="cuda")
    live0 = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
    collect = _hi_walk_launch(query, payload, decay, state, live0,
                              pol32(QUIET), rank, collect=True)
    schedule = collect["schedule"]
    _, _, ckpt_state, ckpt_live = traj_checkpoints(state, live0, decay,
                                                  schedule, rank, 64)
    bit_equal = True
    for seg in range(ckpt_state.shape[0]):
        start = seg * 64
        stop = min(start + 64, tokens)
        traj, _, _ = traj_segment(ckpt_state[seg], ckpt_live[seg],
                                  decay[:, start:stop],
                                  schedule.chunk(start, stop), rank, heads)
        prep = _hi_prep(traj, query, torch.zeros_like(collect["reads"]),
                        collect["live_pre"], start, stop - start, rank,
                        QUIET.read_scale)
        bit_equal = bit_equal and bool(torch.equal(
            prep["w"].view(batch, heads, stop - start, rank)
            .permute(0, 2, 1, 3),
            collect["weights"][:, start:stop]))
    return dict(
        gate="B: the hi backward vs the eager oracle (floor first), and the "
             "prep weights bit-equal the forward's channel",
        noise_factor=NOISE_FACTOR,
        floor_definition=("the fp32 oracle against its own fp64 twin, per "
                          "output, on the same fixture"),
        hi_spans=rows,
        prep_weights_bit_equal_forward_channel=bool(bit_equal),
        note=("the K8 adjoint's rank-axis reductions (duq, dlam) are tile sums "
              "sequenced across slot blocks -- an order no torch expression "
              "reproduces exactly -- so the gradients are held to the floor "
              "gates, never bit-equality (the K6/K23 discipline)"))


def gate_b_across_boundary() -> dict:
    """Gradients ACROSS the growth boundary: the exit-state gradient seeds the
    span carries (K6's own op drops it -- its windows never chain); the
    zero-injection pad's adjoint in the loop, floor first."""
    batch, tokens, heads, value_dim = 2, 1024, 2, 128
    start_rank, cap = 128, 512
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           8, 51)
    g = torch.Generator().manual_seed(99)
    upstream = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to("cuda")
    policy = RoutingPolicy(ops=True, grow=True)
    state0 = torch.zeros(batch, heads, start_rank, value_dim,
                         dtype=torch.float32, device="cuda")
    live0 = torch.zeros(batch, heads, start_rank, dtype=torch.bool,
                        device="cuda")
    probe = growing_scan(query, payload, decay, policy=policy,
                         start_rank=start_rank, rank_cap=cap)
    spans = probe.stats.spans
    o32 = _oracle_grads(query, payload, decay, upstream, policy, spans,
                        torch.float32, initial_state=state0)
    o64 = _oracle_grads(query, payload, decay, upstream, policy, spans,
                        torch.float64, initial_state=state0)
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state0.clone().requires_grad_(True)
    res = growing_scan(q, p, d, policy=policy, start_rank=start_rank,
                       rank_cap=cap, initial_state=s, initial_live=live0)
    (res.reads * upstream).sum().backward()
    return dict(
        gate="B: gradients ACROSS the growth boundary (the chained carries)",
        fixture=dict(batch=batch, tokens=tokens, heads=heads,
                     value_dim=value_dim, start_rank=start_rank,
                     rank_cap=cap, spans=[list(s_) for s_ in spans],
                     distinct=8, seed=51),
        outputs=_floor_rows((q.grad, p.grad, d.grad, s.grad), o32, o64,
                            names=("dquery", "dpayload", "ddecay", "dstate")),
        note=("the entry state is attached, so dstate crosses both "
              "zero-injection pads; K6's own op pair cannot express this "
              "(its backward drops the exit-state gradient) -- the K8 span "
              "pair re-hosts K6's launch and adjoint kernels with the carry "
              "seeded"))


# ------------------------------------------------------------------- gate D


def _timed(fn, reps=3):
    walls = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    return sorted(walls)[len(walls) // 2], out


def gate_d_walls(reps: int = 3) -> dict:
    """The honest walls per snap at the probe shape, through the op pair with
    autograd live, plus the grown scan's two-pass cost and the memory."""
    batch, tokens, heads, value_dim = (PROBE_SHAPE["batch"],
                                       PROBE_SHAPE["tokens"],
                                       PROBE_SHAPE["heads"],
                                       PROBE_SHAPE["value_dim"])
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                            512, 5, bf16=True)
    rows = []
    for rank in (128, 256, 512, 1024, 2048):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fwds, bwds = [], []
        for _ in range(reps):
            q = query.clone().requires_grad_(True)
            p = payload.clone().requires_grad_(True)
            d = decay.clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = growing_scan(q, p, d, policy=QUIET, start_rank=rank,
                              rank_cap=rank)
            torch.cuda.synchronize()
            fwds.append(time.perf_counter() - t0)
            grad = torch.randn_like(res.reads)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            (res.reads * grad).sum().backward()
            torch.cuda.synchronize()
            bwds.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**30
        fwd = sorted(fwds)[len(fwds) // 2]
        bwd = sorted(bwds)[len(bwds) // 2]
        rows.append(dict(rank=rank, forward_s=round(fwd, 3),
                         backward_s=round(bwd, 3),
                         peak_fwd_bwd_gb=round(peak, 2),
                         tok_per_s_forward=round(batch * tokens / fwd)))
    # the grown scan: probe + real, and the memory of a training-shaped pass
    policy = RoutingPolicy(ops=True, grow=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fwds, bwds = [], []
    for _ in range(reps):
        q = query.clone().requires_grad_(True)
        p = payload.clone().requires_grad_(True)
        d = decay.clone().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = growing_scan(q, p, d, policy=policy, start_rank=128,
                           rank_cap=2048)
        torch.cuda.synchronize()
        fwds.append(time.perf_counter() - t0)
        grad = torch.randn_like(res.reads)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        (res.reads * grad).sum().backward()
        torch.cuda.synchronize()
        bwds.append(time.perf_counter() - t0)
    fwd = sorted(fwds)[len(fwds) // 2]
    bwd = sorted(bwds)[len(bwds) // 2]
    grown = dict(spans=[list(s) for s in res.stats.spans],
                 events=res.stats.events,
                 forward_s=round(fwd, 3), backward_s=round(bwd, 3),
                 peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 tok_per_s=round(batch * tokens / fwd))
    # the memory arithmetic, stated
    mb = 2**20
    state_mb = batch * heads * 2048 * value_dim * 4 / mb
    return dict(
        gate="D: the honest walls and memory at the probe shape",
        fixture=dict(**PROBE_SHAPE, reps=reps, precision="bf16 inputs"),
        fixed_rank=rows,
        grown=grown,
        memory_arithmetic=dict(
            state_at_2048_mb=round(state_mb, 1),
            forward_saved_checkpoints_per_scan_mb=round(33 * state_mb, 1),
            k8_backward_recomputed_checkpoints_mb=round(state_mb, 1),
            note=("at rank 2048 the K6-style forward-SAVED checkpoints would "
                  "be ~33 states per scan (~554 MB) held by autograd for "
                  "every mixer of the microbatch; K8's backward REPLAYS them "
                  "into scratch (one extra K1-class walk per span, measured "
                  "in the backward wall) and holds only the span's entry "
                  "state -- the 16x state is paid in recompute, not in "
                  "resident bytes")),
        note=("the grow arm pays the probe pass (span discovery) on top of the "
              "real pass until the cap is reached within the window; at the "
              "cap the probe is skipped and the scan is one span. 2048 is "
              "16x the registered state -- the walls say so, they are not "
              "optimised away"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "receipts" / "k8_gates.json"))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    report = {}
    started = time.monotonic()
    report["gate_a_never_fires"] = gate_a_never_fires()
    print("never-fires done", flush=True)
    report["gate_a_grown"] = gate_a_grown_probe_shape()
    print("grown probe-shape done", flush=True)
    report["gate_a_span_parity"] = gate_a_span_parity()
    print("span parity done", flush=True)
    report["gate_b_hi_spans"] = gate_b_hi_spans()
    print("gate B hi spans done", flush=True)
    report["gate_b_across_boundary"] = gate_b_across_boundary()
    print("gate B across-boundary done", flush=True)
    report["gate_d_walls"] = gate_d_walls(args.reps)
    print("gate D done", flush=True)
    report["walls_seconds_total"] = round(time.monotonic() - started, 1)
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"receipt -> {out}", flush=True)


if __name__ == "__main__":
    main()
