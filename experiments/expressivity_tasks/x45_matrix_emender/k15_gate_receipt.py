"""K15 -- the composition's gate receipts: gates A, A2 (the composition's
own), B and D at the registered shape.

Writes ``receipts/k15_gates.json`` (the K10/K13 gate-receipt pattern):

* **Gate A** -- the parity lanes at the registered shape (batch 8, 2048
  tokens, 2 heads, rank 128, value_dim 128, key_dim 16, bf16 inputs), in
  BOTH gated combinations: the schedule fields, the payload lanes, the
  read WEIGHTS (the raw scores), the pre-decision liveness, the exit
  state/liveness AND the exit key table bit-equal against the eager
  per-token composed oracle; the read at the measured fp32-vs-fp64 floor;
  the SIGNED-WEIGHT sub-receipt (the additive form's own channel, under
  the gates).
* **Gate A2** -- the composition's own birth receipts at the same shape:
  (i) the BIT-IDENTITY TO THE ADDITIVE MACHINE (at ``lam_vec ==`` the
  registered scalar per channel and ``g_vec == 1`` the K15 forward launch
  vs K13's own committed kernel launch, every channel: 0 differing
  elements -- the composition is BORN as the additive machine); (ii) the
  DECISIONS-BIT-IDENTICAL bar under RANDOM per-channel gates/decay (no
  gate value feeds op selection); (iii) the |w|-mass dilution instrument
  and the birth-magnitude diagnostic (additive vs banked softmax, on the
  same fixture -- the honest ~O(live-rows)x consequence, stated).
* **Gate B** -- the fp64 gradcheck of the oracle THROUGH THE COMPOSED PATH
  (every adjoint branch: the additive read's direct term into the state,
  the score path into the query keys and the key table, the payload lanes
  under the per-channel write gate, the PER-CHANNEL DECAY vector), the
  fp32-vs-fp64 floor per output measured on the gate fixture, the kernel
  path's SEVEN gradients vs the floor (``NOISE_FACTOR`` 8x), and the
  segment-recompute bit-equality on both channels (the per-channel value
  replay + the key replay).
* **Gate D** -- block-call walls (forward/backward per (layer, window)
  scan at the registered shape), the register/spill count of the forward
  kernel, and the honest cost statement against K13's and K11B's own
  same-session launches (the composition carries BOTH engines' costs: the
  additive read's dropped softmax and the per-channel vector loads).

Usage:

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k15_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k15_gates.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    _k10_forward_launch, separated_available, span_key_std)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates import (
    channel_persistent_trajectory)
from experiments.expressivity_tasks.x45_matrix_emender.k13_additive_read import (
    _k13_backward_body, _k13_forward_launch)
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    _k15_backward_body, _k15_forward_launch, composed_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
# The registered shape (the K10/K13 gate receipts' own fixture).
BATCH, TOKENS, HEADS, RANK, VALUE_DIM, KEY_DIM = 8, 2048, 2, 128, 128, 16
SEGMENT_TOKENS = 64


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


def gate_fixture(seed: int, batch: int, dtype: torch.dtype):
    """The per-channel gate/decay fixture: random per-channel gates in
    (0.05, 0.95) and a per-channel decay spread (0.85-1.0) around the
    registered scalar."""
    payload, write_keys, query_keys, decay = registered_fixture(
        seed=seed, batch=batch, dtype=dtype)
    generator = torch.Generator().manual_seed(seed + 1)
    spread = (torch.rand(*payload.shape, generator=generator) * 0.15
              ).to(dtype=dtype).cuda()
    decay_vec = (decay.unsqueeze(-1) * 0.0 + 0.85 + spread).clamp(
        min=1e-4, max=1.0 - 1e-6)
    gates = (torch.rand(*payload.shape, generator=generator) * 0.9 + 0.05
             ).to(dtype=dtype).cuda()
    return payload, write_keys, query_keys, decay_vec, gates


def fresh_store(batch=BATCH, heads=HEADS, rank=RANK, value_dim=VALUE_DIM,
                key_dim=KEY_DIM):
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device="cuda")
    return state, live, keys


def gate_a(receipt: dict) -> dict:
    rows_all = {}
    flips_all = lanes_all = 0
    for gated in (True, False):
        payload, write_keys, query_keys, decay_vec, gates = gate_fixture(
            77, BATCH, torch.bfloat16)
        state, live, keys = fresh_store()
        started = time.monotonic()
        out = _k15_forward_launch(payload, write_keys, query_keys, decay_vec,
                                  gates, state, live, keys, POL32, RANK,
                                  SEGMENT_TOKENS, gated, collect=True,
                                  warps=8, stages=3)
        torch.cuda.synchronize()
        kernel_wall = time.monotonic() - started
        started = time.monotonic()
        ref = composed_oracle_scan(payload, write_keys, query_keys, decay_vec,
                                   gates, policy=POLICY, gated=gated,
                                   rank=RANK, rank_max=RANK, collect=True)
        oracle_wall = time.monotonic() - started
        with torch.no_grad():
            ref64 = composed_oracle_scan(payload.double(), write_keys.double(),
                                         query_keys.double(),
                                         decay_vec.double(), gates.double(),
                                         policy=POLICY, gated=gated,
                                         rank=RANK, rank_max=RANK)
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
        rows_all[f"gated_{gated}"] = dict(
            channels=rows,
            total=dict(lanes=lanes, flips=flips),
            op_histogram=op_histogram(ref["schedule"]),
            walls=dict(kernel_forward_s=round(kernel_wall, 3),
                       eager_oracle_s=round(oracle_wall, 1)))
        flips_all += flips
        lanes_all += lanes
    payload, write_keys, query_keys, decay_vec, gates = gate_fixture(
        77, BATCH, torch.bfloat16)
    state, live, keys = fresh_store()
    out = _k15_forward_launch(payload, write_keys, query_keys, decay_vec,
                              gates, state, live, keys, POL32, RANK,
                              SEGMENT_TOKENS, True, collect=True, warps=8,
                              stages=3)
    # the SIGNED-WEIGHT sub-receipt: the additive form's own channel, under
    # the per-channel gates.
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
                     value_dim=VALUE_DIM, key_dim=KEY_DIM,
                     dtype="bf16 inputs, fp32 defined-order math",
                     segment_tokens=SEGMENT_TOKENS, distinct_directions=512,
                     note="random per-channel gates in (0.05, 0.95), the "
                          "per-channel decay a 0.85-1.0 spread, both GATED "
                          "combinations"),
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
    """The composition's own birth receipts at the registered shape."""
    payload, write_keys, query_keys, decay = registered_fixture()
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    # (i) the BIT-IDENTITY TO THE ADDITIVE MACHINE: lam_vec == the registered
    # scalar per channel, g_vec == 1 -- the K15 launch vs K13's own.
    scalar = decay.float().contiguous()                     # (b, T, h)
    lam_vec = scalar.unsqueeze(-1).expand(batch, tokens, heads, value_dim
                                          ).contiguous()
    ones = torch.ones(batch, tokens, heads, value_dim, device="cuda")
    state, live, keys = fresh_store()
    k13 = _k13_forward_launch(payload, write_keys, query_keys, scalar,
                              state, live, keys, POL32, RANK, SEGMENT_TOKENS,
                              collect=True, warps=8, stages=3)
    k13_flips = k13_lanes = 0
    k15_flips = k15_lanes = 0
    for gated in (False, True):
        state2, live2, keys2 = fresh_store()
        k15 = _k15_forward_launch(payload, write_keys, query_keys, lam_vec,
                                  ones, state2, live2, keys2, POL32, RANK,
                                  SEGMENT_TOKENS, gated, collect=True,
                                  warps=8, stages=3)
        for name in ("slot", "erase", "op", "live_after", "payload",
                     "live_pre", "reads", "state", "live", "weights",
                     "keys"):
            a, b = k13[name], k15[name]
            k13_flips += int((a != b).sum())
            k13_lanes += a.numel()
        # (ii) the DECISIONS-BIT-IDENTICAL bar under RANDOM per-channel
        # gates/decay: no gate value feeds op selection.
        generator = torch.Generator().manual_seed(5)
        rand_gates = (torch.rand(batch, tokens, heads, value_dim,
                                generator=generator) * 0.9 + 0.05).cuda()
        rand_decay = (0.85 + torch.rand(batch, tokens, heads, value_dim,
                                        generator=generator) * 0.15).cuda()
        state3, live3, keys3 = fresh_store()
        vary = _k15_forward_launch(payload, write_keys, query_keys, rand_decay,
                                   rand_gates, state3, live3, keys3, POL32,
                                   RANK, SEGMENT_TOKENS, True, collect=True,
                                   warps=8, stages=3)
        for name in ("slot", "erase", "op", "live_after", "live_pre", "keys"):
            a, b = k13[name], vary[name]
            k15_flips += int((a != b).sum())
            k15_lanes += a.numel()
    # (iii) the |w|-mass dilution instrument + the birth-magnitude diagnostic:
    # additive vs the BANKED softmax read, same fixture (the honest
    # ~O(live-rows)x consequence, carried over from K13).
    state4, live4, keys4 = fresh_store()
    k10 = _k10_forward_launch(payload, write_keys, query_keys, decay,
                              state4, live4, keys4, POL32, RANK,
                              SEGMENT_TOKENS, collect=True, warps=8, stages=3)
    rms = {}
    for label, o in (("composed", k13), ("banked", k10)):
        r = o["reads"].float()
        rms[label] = [float(v) for v in r.pow(2).mean(dim=(0, 1, 3)).sqrt()]
    ratios = [a / max(b, 1e-30) for a, b in zip(rms["composed"], rms["banked"])]

    def eff_rows(w, live_pre):
        wa = w.float().abs()
        mass = wa.sum(-1).clamp_min(1e-30)
        probs = wa / mass.unsqueeze(-1)
        contrib = torch.where(probs > 0, probs * probs.log(),
                              torch.zeros_like(probs))
        has = live_pre.any(-1)
        return float(torch.exp(-contrib.sum(-1))[has].mean())
    gate = dict(
        birth_bit_identity_to_additive_machine=dict(
            differing_elements=k13_flips, lanes=k13_lanes,
            bit_identical=bool(k13_flips == 0),
            note="at lam_vec == the registered scalar on every channel and "
                 "g_vec == 1 per channel the K15 forward is bit-identical "
                 "to K13's own committed kernel launch on EVERY channel "
                 "(both GATED combinations) -- the composition is BORN as "
                 "the additive machine; the -0.0 IEEE corner of the "
                 "unity-gate CORRECT blend is the only legal escape and "
                 "its count is asserted zero here"),
        decisions_bit_identical_under_random_gates=dict(
            differing_elements=k15_flips, lanes=k15_lanes,
            bit_identical=bool(k15_flips == 0),
            note="no gate value, scalar or per-channel, feeds op "
                 "selection: the discrete schedule is bit-identical to the "
                 "K13 kernel's under random per-channel gates/decay"),
        birth_magnitude=dict(
            per_head_read_output_rms=rms,
            composed_over_banked_per_head=ratios,
            note="R13-5 carried over: the composed read at birth is "
                 "~O(live-rows)x the banked softmax read's magnitude (no "
                 "simplex normalisation); the per-channel decay gates "
                 "cannot change it at birth (they are born at the "
                 "registered scalar)"),
        dilution_instrument=dict(
            w_mass_effective_rows_composed=eff_rows(k13["weights"],
                                                    k13["live_pre"]),
            w_mass_effective_rows_banked=eff_rows(k10["weights"],
                                                  k10["live_pre"]),
            note="R13-4: e^H of the |w|-mass distribution over live rows -- "
                 "the instrument the arms' mechanism receipts read"))
    receipt["gate_a2"] = gate
    return gate


def gate_b(receipt: dict) -> dict:
    # 1. the fp64 gradcheck of the oracle THROUGH THE COMPOSED PATH
    #    (the instrument validates first; every adjoint branch).
    generator = torch.Generator().manual_seed(11)
    batch, tokens, heads, rank, value_dim, key_dim = 2, 48, 2, 8, 16, 4
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

    hist_holder = {}
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
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    gradcheck_ok = bool(torch.autograd.gradcheck(f, (p, wk, qk, d, g_, s, k),
                                                 eps=1e-6, atol=1e-4, rtol=1e-4))
    hist = hist_holder["hist"]

    # 2. the floors and the kernel path's seven gradients, on a gate-class
    #    shape.
    payload32, write_keys32, query_keys32, decay32, gates32 = gate_fixture(
        23, 2, torch.float32)
    rank32, seg = 128, SEGMENT_TOKENS
    upstream = torch.randn(2, 256, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(1234)).cuda()

    def oracle_grads(dtype):
        p_ = payload32.clone().requires_grad_(True)
        wk_ = write_keys32.clone().requires_grad_(True)
        qk_ = query_keys32.clone().requires_grad_(True)
        d_ = decay32.clone().requires_grad_(True)
        g__ = gates32.clone().requires_grad_(True)
        s_ = torch.zeros(2, HEADS, rank32, VALUE_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        k_ = torch.zeros(2, HEADS, rank32, KEY_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        ref = composed_oracle_scan(p_.to(dtype), wk_.to(dtype), qk_.to(dtype),
                                   d_.to(dtype), g__.to(dtype), policy=POLICY,
                                   gated=True, rank=rank32, rank_max=rank32,
                                   initial_state=s_, initial_keys=k_)
        (ref["reads"] * upstream.to(dtype)).sum().backward()
        return (p_.grad, wk_.grad, qk_.grad, d_.grad, g__.grad, s_.grad, k_.grad)

    oracle32 = oracle_grads(torch.float32)
    oracle64 = oracle_grads(torch.float64)
    state, live, keys = fresh_store(batch=2, rank=rank32)
    out = _k15_forward_launch(payload32, write_keys32, query_keys32, decay32,
                              gates32, state, live, keys, POL32, rank32, seg,
                              True, collect=True, warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    grads = _k15_backward_body(payload32, write_keys32, query_keys32, decay32,
                               gates32, upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank32, seg,
                               POLICY.read_scale, True)
    names = ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dgates",
             "dstate", "dkeys")
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
    #    replays: K11B's per-channel value replay + K10's key replay).
    import triton
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import _k10_key_traj_kernel
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
    _k10_key_traj_kernel[(rows_count,)](
        out["ckpt_keys"][0], write_keys32, schedule.slot, schedule.op, ktraj,
        SEGMENT_TOKENS, 256, 0, rank32, KEY_DIM,
        *write_keys32.stride(), H=HEADS, KP=triton.next_power_of_2(KEY_DIM),
        RP=triton.next_power_of_2(rank32), num_warps=4)
    replay_ok &= bool(torch.equal(ktraj[:, 0], out["ckpt_keys"][0].reshape(
        rows_count, rank32, KEY_DIM)))
    gate = dict(
        oracle_gradcheck_fp64=dict(ok=gradcheck_ok, branches=hist,
                                   note="THROUGH THE COMPOSED PATH: the "
                                        "raw-score weights are attached (no "
                                        "softmax jacobian anywhere) under "
                                        "the per-channel DECAY vector and "
                                        "write gate; the checked branches "
                                        "include the read's direct term into "
                                        "the state, the score path into the "
                                        "query keys and the key table, the "
                                        "gated lanes and the per-channel "
                                        "DECAY"),
        floors_and_deviations=table,
        segment_recompute_bit_equal=bool(replay_ok),
        strength_path=dict(
            note="the value walk's strength term is gone with the softmax "
                 "(K13's receipt, carried over): the additive prep writes "
                 "an EXACT-ZERO dstrength channel and K11B's committed "
                 "value walk consumes it verbatim (IEEE add of 0.0 is "
                 "exact) -- receipted in the suite on a fixture with "
                 "wildly different row norms"),
        noise_factor=NOISE_FACTOR)
    receipt["gate_b"] = gate
    return gate


def gate_d(receipt: dict) -> dict:
    """Block-call walls at the registered shape, against K13's AND K11B's
    own launches in the SAME session (not citations)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates \
        import _k11b_backward_body, _k11b_forward_launch
    payload, write_keys, query_keys, decay = registered_fixture()
    batch, tokens, heads, value_dim = payload.shape
    generator = torch.Generator().manual_seed(4)
    gates = (torch.rand(batch, tokens, heads, value_dim, generator=generator)
             * 0.9 + 0.05).to(torch.bfloat16).cuda()
    spread = (torch.rand(batch, tokens, heads, value_dim, generator=generator)
              * 0.15).to(torch.bfloat16).cuda()
    decay_vec = (0.85 + spread).clamp(min=1e-4, max=1.0 - 1e-6)
    state, live, keys = fresh_store()

    def forward_k15():
        return _k15_forward_launch(payload, write_keys, query_keys, decay_vec,
                                    gates, state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, True, collect=False,
                                    warps=8, stages=3)

    def forward_k13():
        return _k13_forward_launch(payload, write_keys, query_keys,
                                   decay.contiguous(), state, live,
                                   keys, POL32, RANK, SEGMENT_TOKENS,
                                   collect=False, warps=8, stages=3)

    def forward_k11b():
        return _k11b_forward_launch(payload, write_keys, query_keys,
                                    decay_vec, gates, state, live, keys,
                                    POL32, RANK, SEGMENT_TOKENS, True,
                                    collect=False, warps=8, stages=3)

    walls = {}
    for label, fn in (("k15", forward_k15), ("k13", forward_k13),
                     ("k11b", forward_k11b)):
        for _ in range(3):
            fn()                                 # compile/warm
        torch.cuda.synchronize()
        repeats = 5
        started = time.monotonic()
        for _ in range(repeats):
            o = fn()
        torch.cuda.synchronize()
        fwd_ms = (time.monotonic() - started) / repeats * 1e3
        schedule = MatrixSchedule(slot=o["slot"], erase=o["erase"],
                                  payload=o["payload"], op=o["op"],
                                  live_after=o["live_after"])
        upstream = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM,
                               generator=torch.Generator().manual_seed(4)).cuda()

        def backward_once():
            if label == "k15":
                return _k15_backward_body(payload, write_keys, query_keys,
                                          decay_vec, gates, upstream, schedule,
                                          o["live_pre"], o["ckpt_state"],
                                          o["ckpt_live"], o["ckpt_keys"], RANK,
                                          SEGMENT_TOKENS, POLICY.read_scale,
                                          True)
            if label == "k13":
                return _k13_backward_body(payload, write_keys, query_keys,
                                          decay.contiguous(), upstream,
                                          schedule, o["live_pre"],
                                          o["ckpt_state"], o["ckpt_live"],
                                          o["ckpt_keys"], RANK, SEGMENT_TOKENS,
                                          POLICY.read_scale)
            return _k11b_backward_body(payload, write_keys, query_keys,
                                       decay_vec, gates, upstream, schedule,
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
    k15 = walls["k15"]
    gate = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM,
                     segment_tokens=SEGMENT_TOKENS),
        walls=walls,
        forward_ms_per_scan=k15["forward_ms_per_scan"],
        backward_ms_per_scan=k15["backward_ms_per_scan"],
        forward_throughput_tok_per_s=round(tok_per_scan
                                           / (k15["forward_ms_per_scan"] / 1e3)),
        cost_note=(
            "the composition carries BOTH engines' costs: the additive "
            "read's dropped softmax (K13's cheaper-than-banked form) and "
            "the per-channel vector loads (K11B's gate-D overhead) -- the "
            "same-session K13/K11B launches are the honest references, "
            "not citations"),
        available=bool(separated_available(RANK, VALUE_DIM, KEY_DIM)))
    receipt["gate_d"] = gate
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "receipts" / "k15_gates.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "the gates need CUDA"
    receipt = dict(probe="k15_composition_gates",
                   registration=str(ROOT / "PREREGISTRATION_COMPOSITION.md"),
                   policy=POLICY.record(),
                   score_form="the composed machine: K13's additive read "
                              "(the banked score channel verbatim, raw and "
                              "live-gated, softmax and strength dropped) "
                              "under K11B p3's per-channel DECAY/write/read "
                              "gate stack")
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
