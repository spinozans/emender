"""K24 -- THE LONG-WINDOW gate receipts: the scan path's disciplines at the
LONGEST window, run on one leased card BEFORE any arm
(``receipts/k24_gates.json``).

The lane's own question makes the engine duties specific: the arms train
the committed composed scan on windows up to 8192 bytes -- 4x the banked
window -- so the receipts must show the scan path is THE SAME engine at
the longer sequences, not a silently different one:

* **Gate LW-A -- the composed scan at the 4x window shape** (batch 2 x
  8192 tokens, the 4x microbatch's own scan shape, rank 128, value_dim
  128, key_dim 16, bf16 inputs, gated): the kernel's schedule fields,
  payload lanes, weights, pre-decision liveness, exit state/liveness and
  exit key table BIT-EQUAL against the eager per-token composed oracle;
  the reads at the measured fp32-vs-fp64 floor x ``NOISE_FACTOR`` -- the
  K15 gate-A bar, re-asserted at the longest window.
* **Gate LW-B -- the fp64 gradchecks with LONGER sequences**: (i) the
  oracle-through-the-composed-path gradcheck at a LONGER multi-segment
  fixture (320 tokens = 5 segments; the banked gate's own fixture was 48
  tokens), every adjoint branch; (ii) the kernel path's SEVEN gradients vs
  the measured fp32-vs-fp64 floors AT THE 4X SCAN SHAPE (8192 tokens, the
  production geometry, batch 2); (iii) the segment-recompute bit-equality
  on both channels at the 4x shape (128 segments).
* **Gate LW-C -- the hybrid seam at the longest window**: the fp64
  gradcheck through the seam (``dense_out + gate * slot(x)``) at a longer
  multi-segment fixture, both gated combinations; the kernel path's seam
  gradients vs the measured floors AT THE 4X SHAPE (8192 tokens).
* **Gate LW-D -- the honest walls**: the composed scan's forward +
  backward block-call walls at all three window shapes (8x2048, 4x4096,
  2x8192 -- the same 16,384 tokens per microbatch), the matched-exposure
  wall asymmetry stated up front from measurement, not projection.

Usage (one leased card, cards 2/3):

    eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \\
    /home/erikg/emender/.venv/bin/python -m \\
    experiments.expressivity_tasks.x45_matrix_emender.k24_gate_receipt \\
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k24_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
    import (_k10_forward_launch, _k10_key_traj_kernel, separated_available,
         span_key_std)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates \
    import channel_persistent_trajectory
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    _k15_backward_body, _k15_forward_launch, composed_oracle_scan,
    composed_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k22_gate_receipt \
    import _flip_rows, _seam_outputs
from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid import (
    build_slot_mixer, slot_policy)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
    import _round_f32
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
SEGMENT_TOKENS = 64
HEADS, RANK, VALUE_DIM, KEY_DIM = 2, 128, 128, 16
# the 4x microbatch's own scan shape (the arm's production geometry)
LONG_BATCH, LONG_TOKENS = 2, 8192


def _long_fixture(seed: int = 77, batch: int = LONG_BATCH,
                  tokens: int = LONG_TOKENS,
                  dtype: torch.dtype = torch.bfloat16):
    """The K10/K5A-bar fixture at the LONG window shape: payload and
    write-key directions from a hot set of distinct fixed directions, the
    per-channel gate/decay spread."""
    generator = torch.Generator().manual_seed(seed)
    distinct = 512
    payload_dirs = torch.randn(distinct, VALUE_DIM, generator=generator)
    key_dirs = torch.randn(distinct, KEY_DIM, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, HEADS),
                           generator=generator)
    payload = payload_dirs[choice].to(dtype=dtype).cuda()
    write_keys = key_dirs[choice].to(dtype=dtype).cuda()
    query_keys = torch.randn(batch, tokens, HEADS, KEY_DIM,
                             generator=generator).to(dtype=dtype).cuda()
    spread = (torch.rand(batch, tokens, HEADS, VALUE_DIM,
                         generator=generator) * 0.15).to(dtype=dtype).cuda()
    decay_vec = (0.85 + spread).clamp(min=1e-4, max=1.0 - 1e-6)
    gates = (torch.rand(batch, tokens, HEADS, VALUE_DIM,
                        generator=generator) * 0.9 + 0.05).to(
                            dtype=dtype).cuda()
    return payload, write_keys, query_keys, decay_vec, gates


def _fresh_store(batch, heads=HEADS, rank=RANK, value_dim=VALUE_DIM,
                 key_dim=KEY_DIM):
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device="cuda")
    return state, live, keys


# ------------------------------------------------------------------ gate LW-A


def gate_lw_a(receipt: dict) -> dict:
    """The composed scan at the 4x window shape: the kernel vs the eager
    oracle, the K15 gate-A bar re-asserted at the longest window."""
    payload, write_keys, query_keys, decay_vec, gates = _long_fixture(77)
    state, live, keys = _fresh_store(LONG_BATCH)
    torch.cuda.synchronize()
    started = time.monotonic()
    out = _k15_forward_launch(payload, write_keys, query_keys, decay_vec,
                              gates, state, live, keys, POL32, RANK,
                              SEGMENT_TOKENS, True, collect=True, warps=8,
                              stages=3)
    torch.cuda.synchronize()
    kernel_wall = time.monotonic() - started
    started = time.monotonic()
    ref = composed_oracle_scan(payload, write_keys, query_keys, decay_vec,
                               gates, policy=POLICY, gated=True, rank=RANK,
                               rank_max=RANK, collect=True)
    oracle_wall = time.monotonic() - started
    with torch.no_grad():
        ref64 = composed_oracle_scan(payload.double(), write_keys.double(),
                                     query_keys.double(),
                                     decay_vec.double(), gates.double(),
                                     policy=POLICY, gated=True, rank=RANK,
                                     rank_max=RANK)
    rows = {}
    flips = lanes = 0
    for name in ("slot", "erase", "op", "live_after"):
        a, b = getattr(ref["schedule"], name), out[name]
        equal = bool(torch.equal(a, b))
        flips += int((a != b).sum())
        lanes += a.numel()
        rows[f"schedule_{name}"] = dict(lanes=a.numel(),
                                        flips=int((a != b).sum()),
                                        bit_equal=equal)
    for name in ("weights", "live_pre", "state", "live", "keys"):
        a, b = ref[name], out[name]
        equal = bool(torch.equal(a, b))
        flips += int((a != b).sum())
        lanes += a.numel()
        rows[name] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                          bit_equal=equal)
    a, b = ref["schedule"].payload, out["payload"]
    equal = bool(torch.equal(a, b))
    flips += int((a != b).sum())
    lanes += a.numel()
    rows["payload"] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                           bit_equal=equal)
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    rows["read"] = dict(floor_fp32_vs_fp64=floor, deviation=deviation,
                        ratio_vs_floor=deviation / max(floor, 1e-30),
                        within_noise_factor=bool(
                            deviation <= NOISE_FACTOR * max(floor, 1e-30)))
    gate = dict(
        name="LW-A (the composed scan at the 4x window shape)",
        fixture=dict(batch=LONG_BATCH, tokens=LONG_TOKENS, heads=HEADS,
                     rank=RANK, value_dim=VALUE_DIM, key_dim=KEY_DIM,
                     dtype="bf16 inputs, fp32 defined-order math",
                     segment_tokens=SEGMENT_TOKENS, distinct_directions=512,
                     windows=LONG_TOKENS, production_shape="the 4x "
                     "microbatch's own scan shape (2 windows x 8192 "
                     "tokens = 16,384 tokens, the memory-matched rule)"),
        channels=rows,
        total=dict(lanes=lanes, flips=flips,
                   bit_equal=bool(flips == 0)),
        read_within_noise=rows["read"]["within_noise_factor"],
        op_histogram=op_histogram(ref["schedule"]),
        walls=dict(kernel_forward_s=round(kernel_wall, 3),
                  eager_oracle_s=round(oracle_wall, 1)),
        key_spread=float(span_key_std(write_keys.float())))
    receipt["gate_lw_a"] = gate
    return gate


# ------------------------------------------------------------------ gate LW-B


def gate_lw_b(receipt: dict) -> dict:
    """The fp64 gradchecks with LONGER sequences: (i) the oracle gradcheck
    through the composed path at a 5-segment fixture (320 tokens, the
    banked gate's own was 48); (ii) the kernel path's seven gradients vs
    the measured floors AT THE 4X SCAN SHAPE; (iii) the segment-recompute
    bit-equality at the 4x shape."""
    # (i) the fp64 gradcheck at the longer multi-segment fixture
    generator = torch.Generator().manual_seed(11)
    batch, tokens, heads, rank, value_dim, key_dim = 2, 320, 2, 16, 16, 4
    distinct = 6
    payload_dirs = torch.randn(distinct, value_dim, generator=generator)
    key_dirs = torch.randn(distinct, key_dim, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, heads),
                           generator=generator)
    payload = payload_dirs[choice].double().cuda()
    write_keys = key_dirs[choice].double().cuda()
    query_keys = torch.randn(batch, tokens, heads, key_dim,
                             generator=generator).double().cuda()
    decay = (0.85 + torch.rand(batch, tokens, heads, value_dim,
                               generator=generator) * 0.1).double().cuda()
    gates = (torch.rand(batch, tokens, heads, value_dim,
                       generator=generator) * 0.9 + 0.05).double().cuda()
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    g_ = gates.clone().requires_grad_(True)
    s = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)
    k = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)

    def f(p_, wk_, qk_, d_, g__, s_, k_):
        ref = composed_oracle_scan(p_, wk_, qk_, d_, g__, policy=POLICY,
                                   gated=True, rank=rank, rank_max=rank,
                                   initial_state=s_, initial_keys=k_)
        return ref["reads"], ref["state"], ref["keys"]

    started = time.monotonic()
    gradcheck_ok = bool(torch.autograd.gradcheck(
        f, (p, wk, qk, d, g_, s, k), eps=1e-6, atol=1e-4, rtol=1e-4))
    gradcheck_wall = time.monotonic() - started

    # (ii) the kernel path's seven gradients vs the floors, at the 4x
    # production scan shape
    payload32, write_keys32, query_keys32, decay32, gates32 = _long_fixture(
        23, dtype=torch.float32)
    upstream = torch.randn(LONG_BATCH, LONG_TOKENS, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(1234)
                           ).cuda()

    def oracle_grads(dtype):
        p_ = payload32.clone().requires_grad_(True)
        wk_ = write_keys32.clone().requires_grad_(True)
        qk_ = query_keys32.clone().requires_grad_(True)
        d_ = decay32.clone().requires_grad_(True)
        g__ = gates32.clone().requires_grad_(True)
        s_ = torch.zeros(LONG_BATCH, HEADS, RANK, VALUE_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        k_ = torch.zeros(LONG_BATCH, HEADS, RANK, KEY_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        ref = composed_oracle_scan(p_.to(dtype), wk_.to(dtype),
                                   qk_.to(dtype), d_.to(dtype),
                                   g__.to(dtype), policy=POLICY, gated=True,
                                   rank=RANK, rank_max=RANK,
                                   initial_state=s_, initial_keys=k_)
        (ref["reads"] * upstream.to(dtype)).sum().backward()
        return (p_.grad, wk_.grad, qk_.grad, d_.grad, g__.grad, s_.grad,
                k_.grad)

    oracle32 = oracle_grads(torch.float32)
    oracle64 = oracle_grads(torch.float64)
    state, live, keys = _fresh_store(LONG_BATCH)
    torch.cuda.synchronize()
    started = time.monotonic()
    out = _k15_forward_launch(payload32, write_keys32, query_keys32,
                              decay32, gates32, state, live, keys, POL32,
                              RANK, SEGMENT_TOKENS, True, collect=True,
                              warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    grads = _k15_backward_body(payload32, write_keys32, query_keys32,
                               decay32, gates32, upstream, schedule,
                               out["live_pre"], out["ckpt_state"],
                               out["ckpt_live"], out["ckpt_keys"], RANK,
                               SEGMENT_TOKENS, POLICY.read_scale, True)
    torch.cuda.synchronize()
    kernel_bwd_wall = time.monotonic() - started
    names = ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dgates",
             "dstate", "dkeys")
    table = []
    for name, kern, orac, orac64 in zip(names, grads, oracle32, oracle64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        table.append(dict(output=name, floor=floor, deviation=deviation,
                          ratio_vs_floor=deviation / max(floor, 1e-30),
                          within_noise_factor=bool(
                              deviation <= NOISE_FACTOR * max(floor, 1e-6)
                              or deviation <= 1e-5)))

    # (iii) the segment-recompute bit-equality on both channels at the 4x
    # shape (128 segments)
    replay_ok = True
    decay_c = decay32.float().permute(0, 2, 1, 3).contiguous()
    rows_count = LONG_BATCH * HEADS
    for index in range(out["ckpt_state"].shape[0] - 1):
        _r, exit_state, _exit_live, _traj = channel_persistent_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index],
            decay_c[:, :, index * SEGMENT_TOKENS:
                    (index + 1) * SEGMENT_TOKENS, :],
            schedule.chunk(index * SEGMENT_TOKENS,
                           (index + 1) * SEGMENT_TOKENS))
        replay_ok &= bool(torch.equal(exit_state,
                                      out["ckpt_state"][index + 1]))
    ktraj = torch.empty((rows_count, SEGMENT_TOKENS, RANK, KEY_DIM),
                        dtype=torch.float32, device="cuda")
    _k10_key_traj_kernel[(rows_count,)](
        out["ckpt_keys"][0], write_keys32, schedule.slot, schedule.op,
        ktraj, SEGMENT_TOKENS, 256, 0, RANK, KEY_DIM,
        *write_keys32.stride(), H=HEADS,
        KP=triton_next_pow2(KEY_DIM), RP=triton_next_pow2(RANK),
        num_warps=4)
    replay_ok &= bool(torch.equal(
        ktraj[:, 0], out["ckpt_keys"][0].reshape(rows_count, RANK,
                                                  KEY_DIM)))
    gate = dict(
        name="LW-B (the fp64 gradchecks with longer sequences)",
        fixture_long_gradcheck=dict(batch=batch, tokens=tokens,
                                    segments=tokens // SEGMENT_TOKENS,
                                    heads=heads, rank=rank,
                                    value_dim=value_dim, key_dim=key_dim,
                                    note="the banked K15 gate-B gradcheck's "
                                         "own geometry with a LONGER "
                                         "sequence (320 tokens = 5 "
                                         "segments; the banked fixture "
                                         "was 48)"),
        oracle_gradcheck_fp64=dict(
            ok=gradcheck_ok, wall_seconds=round(gradcheck_wall, 1),
            note="THROUGH THE COMPOSED PATH: every adjoint branch (the "
                 "additive read's direct term into the state, the score "
                 "path into the query keys and the key table, the gated "
                 "lanes and the per-channel DECAY) at the longer "
                 "sequence"),
        floors_and_deviations=dict(
            shape=dict(batch=LONG_BATCH, tokens=LONG_TOKENS, heads=HEADS,
                       rank=RANK, value_dim=VALUE_DIM, key_dim=KEY_DIM,
                       note="the 4x microbatch's own scan shape"),
            table=table,
            kernel_backward_wall_s=round(kernel_bwd_wall, 3)),
        segment_recompute_bit_equal=bool(replay_ok),
        segments_checked=int(out["ckpt_state"].shape[0]))
    receipt["gate_lw_b"] = gate
    return gate


# ------------------------------------------------------------------ gate LW-C


def gate_lw_c(receipt: dict) -> dict:
    """The hybrid seam at the longest window: the fp64 gradcheck through
    ``dense_out + gate * slot(x)`` at a longer multi-segment fixture (both
    gated combinations), and the kernel path's seam gradients vs the
    measured floors AT THE 4X SHAPE (8192 tokens)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import ShortConvComposedMixer
    policy = slot_policy()
    gate = dict(name="LW-C (the hybrid seam at the longest window)")
    gradchecks = {}
    for gated in (True, False):
        state = torch.random.get_rng_state()
        torch.manual_seed(29 if gated else 31)
        mixer = ShortConvComposedMixer(
            width=48, heads=1, value_dim=8, rank=5, rank_max=5, key_dim=4,
            policy=policy, ctx=4, segment_tokens=64,
            birth="identity").cuda().double()
        torch.random.set_rng_state(state)
        batch, tokens, width = 2, 128, 48       # 2 segments, 2.67x the K22 gate's
        dense = torch.randn(batch, tokens, width, dtype=torch.float64,
                            device="cuda").requires_grad_(True)
        gate_value = torch.zeros((), dtype=torch.float64,
                                 device="cuda").requires_grad_(True)
        x = torch.randn(batch, tokens, width, dtype=torch.float64,
                        device="cuda").requires_grad_(True)

        def scan_fn(payload, write_keys, query_keys, lam_vec, gates_,
                    gated=gated):
            return composed_oracle_scan(
                payload, write_keys, query_keys, lam_vec, gates_,
                policy=policy, gated=gated, rank=5, rank_max=5)

        def f(dense_, gate_, x_, *params):
            out, _slot = _seam_outputs(mixer, scan_fn, x_, dense_, gate_)
            return out

        inputs = [dense, gate_value, x] + list(mixer.parameters())
        gradchecks[str(gated)] = bool(torch.autograd.gradcheck(
            f, tuple(inputs), eps=1e-6, atol=1e-4, rtol=1e-4,
            nondet_tol=0.0))
    gate["fp64_gradcheck_through_the_seam"] = dict(
        gradchecks=gradchecks,
        fixture=dict(batch=2, tokens=128, width=48, segments=2,
                     note="the K22 gate-B seam gradcheck's own geometry "
                          "with a LONGER sequence (128 tokens = 2 "
                          "segments; the banked fixture was 12)"))

    # the kernel path's seam gradients vs the floors at the 4x shape
    b, t, w = LONG_BATCH, LONG_TOKENS, 64
    dense = torch.randn(b, t, w,
                        generator=torch.Generator().manual_seed(51)).cuda()
    x = torch.randn(b, t, w,
                    generator=torch.Generator().manual_seed(52)).cuda()
    upstream = torch.randn(b, t, w,
                           generator=torch.Generator().manual_seed(1234)
                           ).cuda()

    def chain(scan_fn, dtype):
        dense_ = dense.clone().to(dtype).requires_grad_(True)
        # a FRESH leaf per chain (a shared `.to()` fixture would go
        # non-leaf on the second chain)
        gate_ = torch.tensor(0.3, device=dense_.device,
                             dtype=dtype).requires_grad_(True)
        x_ = x.clone().to(dtype).requires_grad_(True)
        state = torch.random.get_rng_state()
        torch.manual_seed(37)
        mixer_ = build_slot_mixer(width=w).cuda().to(dtype)
        torch.random.set_rng_state(state)
        out, _slot = _seam_outputs(mixer_, scan_fn, x_, dense_, gate_)
        (out * upstream.to(dtype)).sum().backward()
        return dict(ddense=dense_.grad, dgate=gate_.grad.clone(),
                    dx=x_.grad)

    kern = chain(lambda *a: composed_scan(
        *a, policy=policy, gated=True, rank=128, rank_max=128,
        segment_tokens=64), torch.float32)
    orac32 = chain(lambda *a: composed_oracle_scan(
        *a, policy=policy, gated=True, rank=128, rank_max=128),
        torch.float32)
    orac64 = chain(lambda *a: composed_oracle_scan(
        *a, policy=policy, gated=True, rank=128, rank_max=128),
        torch.float64)
    rows = {}
    for name in ("ddense", "dgate", "dx"):
        floor = (orac32[name].float()
                 - orac64[name].float()).abs().max().item()
        deviation = (kern[name].float()
                     - orac32[name].float()).abs().max().item()
        rows[name] = dict(floor=floor, deviation=deviation,
                          within_noise=bool(
            deviation <= NOISE_FACTOR * max(floor, 1e-6)
            or deviation <= 1e-5))
    gate["kernel_seam_gradients_vs_floors"] = dict(
        shape=dict(batch=b, tokens=t, width=w,
                   note="the 4x microbatch's own token count"),
        rows=rows)
    receipt["gate_lw_c"] = gate
    return gate


# ------------------------------------------------------------------ gate LW-D


def gate_lw_d(receipt: dict) -> dict:
    """The honest walls: the composed scan's forward + backward
    block-call walls at all three window shapes -- the same 16,384 tokens
    per microbatch, the sequential walk's own scaling stated from
    measurement (the matched-exposure wall asymmetry the arms run
    under)."""
    walls = {}
    for label, batch, tokens in (("1x", 8, 2048), ("2x", 4, 4096),
                                 ("4x", 2, 8192)):
        payload, write_keys, query_keys, decay_vec, gates = _long_fixture(
            91, batch=batch, tokens=tokens, dtype=torch.float32)
        state, live, keys = _fresh_store(batch)
        out = _k15_forward_launch(
            payload, write_keys, query_keys, decay_vec, gates, state, live,
            keys, POL32, RANK, SEGMENT_TOKENS, True, collect=True, warps=8,
            stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        upstream = torch.randn(batch, tokens, HEADS, VALUE_DIM,
                               generator=torch.Generator().manual_seed(
                                   5)).cuda()
        # warm in
        _k15_forward_launch(
            payload, write_keys, query_keys, decay_vec, gates, state, live,
            keys, POL32, RANK, SEGMENT_TOKENS, True, collect=True, warps=8,
            stages=3)
        _k15_backward_body(payload, write_keys, query_keys, decay_vec,
                           gates, upstream, schedule, out["live_pre"],
                           out["ckpt_state"], out["ckpt_live"],
                           out["ckpt_keys"], RANK, SEGMENT_TOKENS,
                           POLICY.read_scale, True)
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(5):
            _k15_forward_launch(
                payload, write_keys, query_keys, decay_vec, gates, state,
                live, keys, POL32, RANK, SEGMENT_TOKENS, True, collect=True,
                warps=8, stages=3)
        torch.cuda.synchronize()
        fwd = (time.monotonic() - started) / 5
        started = time.monotonic()
        for _ in range(5):
            _k15_backward_body(payload, write_keys, query_keys, decay_vec,
                               gates, upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], RANK, SEGMENT_TOKENS,
                               POLICY.read_scale, True)
        torch.cuda.synchronize()
        bwd = (time.monotonic() - started) / 5
        walls[label] = dict(
            shape=dict(batch=batch, tokens=tokens,
                       tokens_per_microbatch=batch * tokens),
            forward_s=round(fwd, 4), backward_s=round(bwd, 4),
            pair_s=round(fwd + bwd, 4))
    gate = dict(
        name="LW-D (the honest walls at the three window shapes)",
        walls=walls,
        note=("the composed scan's sequential walk: the per-microbatch wall "
              "grows ~linearly in the window length at the SAME tokens per "
              "microbatch (16,384) -- the per-program sequence depth "
              "grows with the window while the program count shrinks "
              "with the microbatch.  Matched-EXPOSURE arms eat the same "
              "bytes in DIFFERENT wall-clocks; the honest per-arm walls "
              "are receipted in each arm's own result"))
    receipt["gate_lw_d"] = gate
    return gate


def triton_next_pow2(n: int) -> int:
    import triton
    return triton.next_power_of_2(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        ROOT / "receipts" / "k24_gates.json"))
    args = ap.parse_args()
    if not separated_available(RANK, VALUE_DIM, KEY_DIM):
        raise SystemExit("the separated-key scan path is unavailable on "
                         "this host -- the K24 lane cannot run")
    receipt = dict(
        probe="k24_gate_receipt",
        note="the K24 long-window lane's engine receipts: the committed "
             "composed scan and the hybrid seam at the LONGEST window "
             "(8192 tokens), the fp64 gradchecks with longer sequences, "
             "and the honest walls at all three window shapes")
    gate_lw_a(receipt)
    gate_lw_b(receipt)
    gate_lw_c(receipt)
    gate_lw_d(receipt)
    ok = (receipt["gate_lw_a"]["total"]["bit_equal"]
          and receipt["gate_lw_a"]["read_within_noise"]
          and receipt["gate_lw_b"]["oracle_gradcheck_fp64"]["ok"]
          and all(row["within_noise_factor"] for row in
                  receipt["gate_lw_b"]["floors_and_deviations"]["table"])
          and receipt["gate_lw_b"]["segment_recompute_bit_equal"]
          and all(receipt["gate_lw_c"]
                  ["fp64_gradcheck_through_the_seam"]["gradchecks"].values())
          and all(row["within_noise"] for row in
                  receipt["gate_lw_c"]["kernel_seam_gradients_vs_floors"]
                  ["rows"].values()))
    receipt["all_checks"] = bool(ok)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("probe", "all_checks")},
                     indent=1), flush=True)
    if not ok:
        raise SystemExit("the K24 long-window gate receipts FAILED -- "
                         "escalate, not excuse")


if __name__ == "__main__":
    main()
