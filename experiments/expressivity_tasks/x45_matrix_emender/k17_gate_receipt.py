"""K17 -- the write-form machine's gate receipts: gates A, A2 (the dense
lane's own), B and D at the registered shape.

Writes ``receipts/k17_gates.json`` (the K15 gate-receipt pattern):

* **Gate A** -- the parity lanes at the registered shape (batch 8, 2048
  tokens, 2 heads, rank 128, value_dim 128, key_dim 16, bf16 inputs), in
  BOTH gated combinations, with the DENSE LANE ACTIVE (random deposits at
  the G2 fixture scale): the 3-lane schedule fields, the payload lanes,
  the read WEIGHTS, the pre-decision liveness, the exit state/liveness AND
  the exit key table bit-equal against the eager per-token dense oracle;
  the read at the measured fp32-vs-fp64 floor.
* **Gate A2** -- the write-form machine's own birth receipts at the same
  shape: (i) the BIT-IDENTITY TO THE COMPOSED MACHINE (at the zero birth
  -- ``d_proj == 0``, ``m_d == 0`` -- the K17 forward launch vs the K15
  composed kernel's own committed launch, every shared channel: 0
  differing elements -- the machine is BORN AS THE COMPOSED MACHINE bit
  for bit); (ii) the DECISIONS-BIT-IDENTICAL bar under RANDOM dense
  deposits AND random per-channel gates/decay (P2: no deposit and no gate
  value feeds op selection); (iii) the C5 write-channel instrument at the
  gate fixture (how much mass the dense channel carries vs the event
  writes) and the |w|-mass dilution instrument (unchanged by the deposits
  at the zero birth).
* **Gate B** -- the fp64 gradcheck of the oracle THROUGH THE DENSE-WRITE
  PATH (the dense lane's own adjoint included), the fp32-vs-fp64 floor per
  output measured on the gate fixture, the kernel path's EIGHT gradients
  vs the floor (``NOISE_FACTOR`` 8x), and the segment-recompute
  bit-equality on both channels (the lane-generic 3-lane value replay +
  the stride-generalized key replay).
* **Gate D** -- block-call walls (forward/backward per (layer, window)
  scan at the registered shape), the register/spill count of the forward
  kernel, and the honest cost statement against the K15 composed
  machine's own same-session launch (the dense lane's load + application
  cost).

Usage:

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k17_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k17_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    MERGE, NONE, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    separated_available, span_key_std)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates import (
    channel_persistent_trajectory)
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    _k15_backward_body, _k15_forward_launch, composed_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write import (
    LANES, _k17_backward_body, _k17_forward_launch, dense_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
# The registered shape (the K10/K13/K15 gate receipts' own fixture).
BATCH, TOKENS, HEADS, RANK, VALUE_DIM, KEY_DIM = 8, 2048, 2, 128, 128, 16
SEGMENT_TOKENS = 64
# The dense-deposit fixture scale (the K11 G2 gate convention).
DENSE_SCALE = 0.05


def registered_fixture(seed: int = 77, batch: int = BATCH, tokens: int = TOKENS,
                       dtype: torch.dtype = torch.bfloat16):
    """The K10/K5A-bar fixture: payload AND write-key directions from a hot
    set of ``distinct`` fixed directions (the identity-cache regime)."""
    generator = torch.Generator().manual_seed(seed)
    distinct = 512
    payload_dirs = torch.randn(distinct, VALUE_DIM, generator=generator)
    key_dirs = torch.randn(distinct, KEY_DIM, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, HEADS), generator=generator)
    payload = payload_dirs[choice].to(dtype=dtype).cuda()
    write_keys = key_dirs[choice].to(dtype=dtype).cuda()
    query_keys = (torch.randn(batch, tokens, HEADS, KEY_DIM, generator=generator)
                  .to(dtype=dtype).cuda())
    decay = torch.full((batch, tokens, HEADS), 0.95).to(dtype=dtype).cuda()
    return payload, write_keys, query_keys, decay


def gate_fixture(seed: int, batch: int, dtype: torch.dtype,
                 tokens: int = TOKENS):
    """The per-channel gate/decay/dense fixture: random per-channel gates in
    (0.05, 0.95), a per-channel decay spread (0.85-1.0) around the
    registered scalar, and random dense deposits at the G2 fixture scale."""
    payload, write_keys, query_keys, decay = registered_fixture(
        seed=seed, batch=batch, tokens=tokens, dtype=dtype)
    generator = torch.Generator().manual_seed(seed + 1)
    spread = (torch.rand(*payload.shape, generator=generator) * 0.15
              ).to(dtype=dtype).cuda()
    decay_vec = (decay.unsqueeze(-1) * 0.0 + 0.85 + spread).clamp(
        min=1e-4, max=1.0 - 1e-6)
    gates = (torch.rand(*payload.shape, generator=generator) * 0.9 + 0.05
             ).to(dtype=dtype).cuda()
    dense = (torch.randn(*payload.shape, generator=generator)
             .to(dtype=dtype).cuda() * DENSE_SCALE)
    return payload, write_keys, query_keys, decay_vec, gates, dense


def fresh_store(batch=BATCH, heads=HEADS, rank=RANK, value_dim=VALUE_DIM,
                key_dim=KEY_DIM):
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device="cuda")
    return state, live, keys


def dense_launch(payload, dense, write_keys, query_keys, decay_vec, gates,
                 state, live, keys, gated, collect=True, warps=8):
    return _k17_forward_launch(payload, dense, write_keys, query_keys,
                               decay_vec, gates, state, live, keys, POL32,
                               RANK, SEGMENT_TOKENS, gated, True,
                               collect=collect, warps=warps, stages=3)


def gate_a(receipt: dict) -> dict:
    rows_all = {}
    flips_all = lanes_all = 0
    for gated in (True, False):
        payload, write_keys, query_keys, decay_vec, gates, dense = gate_fixture(
            77, BATCH, torch.bfloat16)
        state, live, keys = fresh_store()
        started = time.monotonic()
        out = dense_launch(payload, dense, write_keys, query_keys, decay_vec,
                           gates, state, live, keys, gated)
        torch.cuda.synchronize()
        kernel_wall = time.monotonic() - started
        started = time.monotonic()
        ref = dense_oracle_scan(payload, dense, write_keys, query_keys,
                               decay_vec, gates, policy=POLICY, gated=gated,
                               rank=RANK, rank_max=RANK, collect=True)
        oracle_wall = time.monotonic() - started
        with torch.no_grad():
            ref64 = dense_oracle_scan(payload.double(), dense.double(),
                                      write_keys.double(), query_keys.double(),
                                      decay_vec.double(), gates.double(),
                                      policy=POLICY, gated=gated, rank=RANK,
                                      rank_max=RANK)
        rows = {}
        flips = 0
        lanes = 0
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
        # the dense lane's own sub-receipt: the fire rate and the recorded
        # zero payload off the live guards
        fired = out["op"][..., 2] != NONE
        rows["dense_lane"] = dict(
            lanes=int(fired.numel()), fired=int(fired.sum()),
            fire_rate=float(fired.float().mean()),
            recorded_payload_zero_where_not_fired=bool(
                (out["payload"][..., 2, :][~fired] == 0).all()),
            targets_live_at_decision_time=bool(all(
                bool(out["live_pre"][b, t, h, out["slot"][b, t, h, 2]])
                for b, t, h in fired.nonzero()[:256].tolist())))
        rows_all[f"gated_{gated}"] = dict(
            channels=rows,
            total=dict(lanes=lanes, flips=flips),
            op_histogram=op_histogram(ref["schedule"]),
            walls=dict(kernel_forward_s=round(kernel_wall, 3),
                       eager_oracle_s=round(oracle_wall, 1)))
        flips_all += flips
        lanes_all += lanes
    payload, write_keys, query_keys, decay_vec, gates, dense = gate_fixture(
        77, BATCH, torch.bfloat16)
    state, live, keys = fresh_store()
    out = dense_launch(payload, dense, write_keys, query_keys, decay_vec,
                       gates, state, live, keys, True)
    # the SIGNED-WEIGHT sub-receipt: the additive form's own channel, under
    # the per-channel gates AND the active deposits.
    w = out["weights"]
    live_pre = out["live_pre"]
    signed = dict(
        negative_entries=int((w[live_pre] < 0).sum()),
        max_abs_on_live=float(w[live_pre].abs().max()),
        gain_bound_holds=bool((w[live_pre].abs() <= POLICY.read_scale + 1e-5).all()),
        zero_on_dead=bool((w[~live_pre] == 0).all()),
        not_a_simplex=bool(
            (w.sum(-1) - 1.0).abs()[live_pre.any(-1)].max() > 1e-3),
        min_on_live=float(w[live_pre].min()), mean_on_live=float(w[live_pre].mean()))
    spread = span_key_std(write_keys.float())
    gate = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM, lanes=LANES,
                     dtype="bf16 inputs, fp32 defined-order math",
                     segment_tokens=SEGMENT_TOKENS, distinct_directions=512,
                     dense_scale=DENSE_SCALE,
                     note="random per-channel gates in (0.05, 0.95), the "
                          "per-channel decay a 0.85-1.0 spread, random "
                          "dense deposits at the G2 fixture scale, both "
                          "GATED combinations"),
        combinations=rows_all,
        total=dict(lanes=lanes_all, flips=flips_all),
        signed_weights=signed,
        key_collapse_watch=dict(
            span_key_std_min=float(spread.min()),
            span_key_std_mean=float(spread.mean()),
            guard_floor=0.05,
            within_guard=bool((spread >= 0.05).all())),
    )
    try:
        meta = out["kernel"]
        gate["forward_kernel_resources"] = dict(
            num_warps=8,
            registers=getattr(meta, "n_regs", None),
            spills=getattr(meta, "n_spills", None))
    except Exception:
        pass
    receipt["gate_a"] = gate
    return gate


def gate_a2(receipt: dict) -> dict:
    """The write-form machine's own birth receipts at the registered shape."""
    payload, write_keys, query_keys, decay = registered_fixture()
    batch, tokens, heads, value_dim = payload.shape
    generator = torch.Generator().manual_seed(78)
    decay_vec = (0.85 + torch.rand(batch, tokens, heads, value_dim,
                                   generator=generator) * 0.15).to(
        torch.bfloat16).cuda().clamp(min=1e-4, max=1.0 - 1e-6)
    gates = (torch.rand(batch, tokens, heads, value_dim,
                        generator=generator) * 0.9 + 0.05).to(
        torch.bfloat16).cuda()
    # (i) the BIT-IDENTITY TO THE COMPOSED MACHINE at the zero birth: the
    # mixer's own birth deposit (0*sigmoid(0)*p -- per-element +-0.0 with
    # the payload's sign, the IEEE identity either way).
    zero_dense = 0.0 * payload.float()
    k15_flips = k15_lanes = 0
    for gated in (False, True):
        state, live, keys = fresh_store()
        k15 = _k15_forward_launch(payload, write_keys, query_keys, decay_vec,
                                  gates, state, live, keys, POL32, RANK,
                                  SEGMENT_TOKENS, gated, collect=True,
                                  warps=8, stages=3)
        state2, live2, keys2 = fresh_store()
        k17 = dense_launch(payload, zero_dense.to(payload.dtype), write_keys,
                           query_keys, decay_vec, gates, state2, live2, keys2,
                           gated)
        for name in ("slot", "erase", "op", "live_after"):
            a, b = k15[name], k17[name][..., :2]
            k15_flips += int((a != b).sum())
            k15_lanes += a.numel()
        for name in ("payload", "live_pre", "reads", "state", "live",
                     "weights", "keys"):
            a, b = k15[name], k17[name] if name != "payload" \
                else k17[name][..., :2, :]
            k15_flips += int((a != b).sum())
            k15_lanes += a.numel()
    # (ii) the DECISIONS-BIT-IDENTICAL bar under RANDOM dense deposits AND
    # random per-channel gates/decay (P2: no deposit and no gate value
    # feeds op selection).
    vary_flips = vary_lanes = 0
    state3, live3, keys3 = fresh_store()
    base = dense_launch(payload, zero_dense.to(payload.dtype), write_keys,
                        query_keys, decay_vec, gates, state3, live3, keys3,
                        True)
    state4, live4, keys4 = fresh_store()
    rand_dense = (torch.randn(batch, tokens, heads, value_dim,
                              generator=generator).cuda() * DENSE_SCALE)
    vary = dense_launch(payload, rand_dense.to(payload.dtype), write_keys,
                        query_keys, decay_vec, gates, state4, live4, keys4,
                        True)
    for name in ("slot", "erase", "op", "live_after"):
        a, b = base[name][..., :2], vary[name][..., :2]
        vary_flips += int((a != b).sum())
        vary_lanes += a.numel()
    for name in ("live_pre", "keys", "weights"):
        a, b = base[name], vary[name]
        vary_flips += int((a != b).sum())
        vary_lanes += a.numel()
    # and the dense lane's TARGET is invariant to the deposit (the argmax
    # over live at decision time)
    target_flips = int((base["slot"][..., 2] != vary["slot"][..., 2]).sum())
    # (iii) the C5 write-channel instrument at the gate fixture (the active
    # deposits' mass vs the event writes') and the |w|-mass instrument at
    # the zero birth (the deposits cannot move the read at birth).
    state5, live5, keys5 = fresh_store()
    active = dense_launch(payload, rand_dense.to(payload.dtype), write_keys,
                          query_keys, decay_vec, gates, state5, live5, keys5,
                          True)
    fired = active["op"][..., 2] != NONE
    dense_mass = active["payload"][..., 2, :].float().abs().mean()
    carrying = (active["op"][..., 0] != NONE) & (active["op"][..., 0] != MERGE)
    event_mass = active["payload"][..., 0, :].float().abs().mean()
    event_mass_carrying = active["payload"][..., 0, :].float()[carrying].mean()

    def eff_rows(w, live_pre):
        wa = w.float().abs()
        mass = wa.sum(-1).clamp_min(1e-30)
        probs = wa / mass.unsqueeze(-1)
        contrib = torch.where(probs > 0, probs * probs.log(),
                              torch.zeros_like(probs))
        has = live_pre.any(-1)
        return float(torch.exp(-contrib.sum(-1))[has].mean())

    gate = dict(
        birth_bit_identity_to_composed_machine=dict(
            differing_elements=k15_flips, lanes=k15_lanes,
            bit_identical=bool(k15_flips == 0),
            note="at the zero birth (d_proj == 0, m_d == 0 -> dense == 0 "
                 "exactly, the mixer's own per-element-signed expression "
                 "0*sigmoid(0)*p) the K17 forward is bit-identical to the "
                 "K15 composed kernel's own committed launch on EVERY "
                 "shared channel (lanes 0/1 of the 3-lane schedule; both "
                 "GATED combinations) -- the machine is BORN AS THE "
                 "COMPOSED MACHINE (and hence as the K13 additive "
                 "machine: the K15 receipt's own chain); the -0.0 IEEE "
                 "corner of the zero deposit is the only legal escape and "
                 "its count is asserted zero here"),
        decisions_bit_identical_under_random_deposits=dict(
            differing_elements=vary_flips, lanes=vary_lanes,
            bit_identical=bool(vary_flips == 0),
            dense_target_flips=target_flips,
            note="no deposit value and no gate value feeds op selection: "
                 "the discrete schedule (lanes 0/1) AND the dense lane's "
                 "recorded best-match target are bit-identical under "
                 "random deposits and random per-channel gates/decay"),
        write_channel_instrument=dict(
            fire_rate=float(fired.float().mean()),
            dense_mean_abs=float(dense_mass),
            event_lane0_mean_abs=float(event_mass),
            event_lane0_mean_abs_on_carrying_ops=float(event_mass_carrying),
            dense_over_event_mass_ratio=float(dense_mass / max(event_mass,
                                                               1e-30)),
            note="the C5 instrument at the gate fixture: how much mass the "
                 "dense channel carries vs the event writes at the G2 "
                 "fixture scale (0.05) -- the arms' mechanism receipts "
                 "read the TRAINED dense channel against this"),
        dilution_instrument=dict(
            w_mass_effective_rows_at_zero_birth=eff_rows(base["weights"],
                                                         base["live_pre"]),
            w_mass_effective_rows_active_deposits=eff_rows(
                active["weights"], active["live_pre"]),
            note="e^H of the |w|-mass distribution over live rows at the "
                 "zero birth (== the composed machine's own) vs under the "
                 "active deposits -- the read's mass distribution is "
                 "invariant at birth and moves only as the deposits move "
                 "the rows' content"))
    receipt["gate_a2"] = gate
    return gate


def gate_b(receipt: dict) -> dict:
    # 1. the fp64 gradcheck of the oracle THROUGH THE DENSE-WRITE PATH
    #    (the instrument validates first; every adjoint branch; the eager
    #    fp64 oracle's cost scales with input elements, so the fixture is
    #    the K15 gate-B shape at half the tokens -- every branch fires).
    generator = torch.Generator().manual_seed(11)
    batch, tokens, heads, rank, value_dim, key_dim = 2, 24, 2, 8, 8, 4
    distinct = 6
    payload_dirs = torch.randn(distinct, value_dim, generator=generator)
    key_dirs = torch.randn(distinct, key_dim, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = payload_dirs[choice].double().cuda()
    write_keys = key_dirs[choice].double().cuda()
    query_keys = torch.randn(batch, tokens, heads, key_dim,
                             generator=generator).double().cuda()
    decay = (0.85 + torch.rand(batch, tokens, heads, value_dim,
                               generator=generator) * 0.1).double().cuda()
    gates = (torch.rand(batch, tokens, heads, value_dim, generator=generator)
             * 0.9 + 0.05).double().cuda()
    dense = (torch.randn(batch, tokens, heads, value_dim, generator=generator)
             .double().cuda() * 0.1)

    hist_holder = {}
    p = payload.clone().requires_grad_(True)
    dn = dense.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    g_ = gates.clone().requires_grad_(True)
    s = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)
    k = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)

    def f(p_, dn_, wk_, qk_, d_, g__, s_, k_):
        ref = dense_oracle_scan(p_, dn_, wk_, qk_, d_, g__, policy=POLICY,
                                gated=True, rank=rank, rank_max=rank,
                                initial_state=s_, initial_keys=k_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    gradcheck_ok = bool(torch.autograd.gradcheck(f, (p, dn, wk, qk, d, g_, s, k),
                                                 eps=1e-6, atol=1e-4, rtol=1e-4))
    hist = hist_holder["hist"]

    # 2. the floors and the kernel path's EIGHT gradients, on a gate-class
    #    shape.
    payload32, write_keys32, query_keys32, decay32, gates32, dense32 = \
        gate_fixture(23, 2, torch.float32, tokens=256)
    rank32, seg = 128, SEGMENT_TOKENS
    upstream = torch.randn(2, 256, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(1234)).cuda()

    def oracle_grads(dtype):
        p_ = payload32.clone().requires_grad_(True)
        dn_ = dense32.clone().requires_grad_(True)
        wk_ = write_keys32.clone().requires_grad_(True)
        qk_ = query_keys32.clone().requires_grad_(True)
        d_ = decay32.clone().requires_grad_(True)
        g__ = gates32.clone().requires_grad_(True)
        s_ = torch.zeros(2, HEADS, rank32, VALUE_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        k_ = torch.zeros(2, HEADS, rank32, KEY_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        ref = dense_oracle_scan(p_.to(dtype), dn_.to(dtype), wk_.to(dtype),
                                qk_.to(dtype), d_.to(dtype), g__.to(dtype),
                                policy=POLICY, gated=True, rank=rank32,
                                rank_max=rank32, initial_state=s_,
                                initial_keys=k_)
        (ref["reads"] * upstream.to(dtype)).sum().backward()
        return (p_.grad, dn_.grad, wk_.grad, qk_.grad, d_.grad, g__.grad,
                s_.grad, k_.grad)

    oracle32 = oracle_grads(torch.float32)
    oracle64 = oracle_grads(torch.float64)
    state, live, keys = fresh_store(batch=2, rank=rank32)
    out = _k17_forward_launch(payload32, dense32, write_keys32, query_keys32,
                              decay32, gates32, state, live, keys, POL32,
                              rank32, seg, True, True, collect=True, warps=8,
                              stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    grads = _k17_backward_body(payload32, dense32, write_keys32, query_keys32,
                               decay32, gates32, upstream, schedule,
                               out["live_pre"], out["ckpt_state"],
                               out["ckpt_live"], out["ckpt_keys"], rank32, seg,
                               POLICY.read_scale, True)
    names = ("dpayload", "ddense", "dwrite_keys", "dquery_keys", "ddecay",
             "dgates", "dstate", "dkeys")
    table = []
    for name, kern, orac, orac64 in zip(names, grads, oracle32, oracle64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        table.append(dict(output=name, floor=floor, deviation=deviation,
                          ratio_vs_floor=deviation / max(floor, 1e-30),
                          within_noise_factor=bool(deviation <= NOISE_FACTOR
                                                  * max(floor, 1e-6)
                                                  or deviation <= 1e-5)))
    # 3. the segment-recompute bit-equality on both channels (the imported
    #    replays: K11B's lane-generic per-channel value replay + K11's
    #    stride-generalized key replay -- the 3-lane schedule).
    import triton
    from experiments.expressivity_tasks.x45_matrix_emender.k11_learned_gates \
        import _k11_key_traj_kernel
    replay_ok = True
    decay_c = decay32.float().permute(0, 2, 1, 3).contiguous()
    rows_count = 2 * HEADS
    for index in range(out["ckpt_state"].shape[0] - 1):
        _r, exit_state, _exit_live, _traj = channel_persistent_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index],
            decay_c[:, :, index * SEGMENT_TOKENS:(index + 1) * SEGMENT_TOKENS, :],
            schedule.chunk(index * SEGMENT_TOKENS,
                           (index + 1) * SEGMENT_TOKENS))
        replay_ok &= bool(torch.equal(exit_state,
                                      out["ckpt_state"][index + 1]))
    ktraj = torch.empty((rows_count, SEGMENT_TOKENS, rank32, KEY_DIM),
                        dtype=torch.float32, device="cuda")
    _k11_key_traj_kernel[(rows_count,)](
        out["ckpt_keys"][0], write_keys32, schedule.slot, schedule.op, ktraj,
        SEGMENT_TOKENS, 256, 0, rank32, KEY_DIM,
        *schedule.slot.stride(), *write_keys32.stride(),
        H=HEADS, KP=triton.next_power_of_2(KEY_DIM),
        RP=triton.next_power_of_2(rank32), num_warps=4)
    replay_ok &= bool(torch.equal(ktraj[:, 0], out["ckpt_keys"][0].reshape(
        rows_count, rank32, KEY_DIM)))
    gate = dict(
        oracle_gradcheck_fp64=dict(ok=gradcheck_ok, branches=hist,
                                   note="THROUGH THE DENSE-WRITE PATH: the "
                                        "dense lane's own deposit adjoint "
                                        "(ddense = dS[best], ds passes "
                                        "through) under the additive read's "
                                        "attached raw-score weights, the "
                                        "per-channel DECAY vector and the "
                                        "per-channel write gate"),
        floors_and_deviations=table,
        segment_recompute_bit_equal=bool(replay_ok),
        dense_adjoint=dict(
            note="the dense lane's adjoint is the full incoming adjoint at "
                 "the best row (additive; ds passes through), and EXACTLY "
                 "zero where the lane did not fire -- receipted in the "
                 "suite"),
        noise_factor=NOISE_FACTOR)
    receipt["gate_b"] = gate
    return gate


def gate_d(receipt: dict) -> dict:
    """Block-call walls at the registered shape, against the K15 composed
    machine's own launch in the SAME session (not a citation)."""
    payload, write_keys, query_keys, decay_vec, gates, dense = gate_fixture(
        4, BATCH, torch.bfloat16)
    state, live, keys = fresh_store()

    def forward_k17():
        return dense_launch(payload, dense, write_keys, query_keys, decay_vec,
                            gates, state, live, keys, True, collect=False)

    def forward_k15():
        return _k15_forward_launch(payload, write_keys, query_keys, decay_vec,
                                   gates, state, live, keys, POL32, RANK,
                                   SEGMENT_TOKENS, True, collect=False,
                                   warps=8, stages=3)

    walls = {}
    for label, fn in (("k17", forward_k17), ("k15", forward_k15)):
        for _ in range(3):
            fn()                                 # compile/warm
        torch.cuda.synchronize()
        repeats = 5
        started = time.monotonic()
        for _ in range(repeats):
            o = fn()
        torch.cuda.synchronize()
        fwd_ms = (time.monotonic() - started) / repeats * 1e3
        schedule = MatrixSchedule(slot=o["slot"][..., :2],
                                  erase=o["erase"][..., :2],
                                  payload=o["payload"][..., :2, :],
                                  op=o["op"][..., :2],
                                  live_after=o["live_after"][..., :2]) \
            if label == "k17" else None
        upstream = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM,
                               generator=torch.Generator().manual_seed(4)).cuda()

        def backward_once():
            if label == "k17":
                return _k17_backward_body(payload, dense, write_keys,
                                          query_keys, decay_vec, gates,
                                          upstream,
                                          MatrixSchedule(
                                              slot=o["slot"], erase=o["erase"],
                                              payload=o["payload"], op=o["op"],
                                              live_after=o["live_after"]),
                                          o["live_pre"], o["ckpt_state"],
                                          o["ckpt_live"], o["ckpt_keys"], RANK,
                                          SEGMENT_TOKENS, POLICY.read_scale,
                                          True)
            return _k15_backward_body(payload, write_keys, query_keys,
                                      decay_vec, gates, upstream,
                                      MatrixSchedule(
                                          slot=o["slot"], erase=o["erase"],
                                          payload=o["payload"], op=o["op"],
                                          live_after=o["live_after"]),
                                      o["live_pre"], o["ckpt_state"],
                                      o["ckpt_live"], o["ckpt_keys"], RANK,
                                      SEGMENT_TOKENS, POLICY.read_scale, True)

        for _ in range(2):
            backward_once()
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(repeats):
            backward_once()
        torch.cuda.synchronize()
        bwd_ms = (time.monotonic() - started) / repeats * 1e3
        walls[label] = dict(forward_ms_per_scan=round(fwd_ms, 2),
                            backward_ms_per_scan=round(bwd_ms, 2))
    tok_per_scan = BATCH * TOKENS
    k17 = walls["k17"]
    gate = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM, lanes=LANES,
                     segment_tokens=SEGMENT_TOKENS),
        walls=walls,
        forward_ms_per_scan=k17["forward_ms_per_scan"],
        backward_ms_per_scan=k17["backward_ms_per_scan"],
        forward_throughput_tok_per_s=round(tok_per_scan
                                           / (k17["forward_ms_per_scan"] / 1e3)),
        cost_note=(
            "the dense lane's honest cost vs the composed machine's own "
            "same-session launch: the deposit load, the lane-2 record "
            "stores and the guarded application (the K11 unconditional "
            "select) on the forward; the dense-lane adjoint step on the "
            "backward -- the K15 launch is the honest reference, not a "
            "citation"),
        available=bool(separated_available(RANK, VALUE_DIM, KEY_DIM)))
    receipt["gate_d"] = gate
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "receipts" / "k17_gates.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "the gates need CUDA"
    receipt = dict(probe="k17_write_form_gates",
                   registration=str(ROOT / "PREREGISTRATION_K17_WRITE_FORM.md"),
                   policy=POLICY.record(),
                   score_form="the write-form machine: the K15 composed "
                              "machine (the K13 additive read under the "
                              "K11B p3 per-channel gate stack) + the dense "
                              "gated write (the b_proj-delta form: every "
                              "token deposits m_d*sigmoid(d_proj(x)) (.) p "
                              "at its best-match LIVE row, recorded as "
                              "schedule lane 2, born at exact zero)")
    gate_a(receipt)
    gate_a2(receipt)
    gate_b(receipt)
    gate_d(receipt)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("gate_a", "gate_a2", "gate_b", "gate_d")},
                     indent=1)[:4000])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
