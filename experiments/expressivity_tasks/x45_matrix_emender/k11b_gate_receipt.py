"""K11B -- the gate receipts: gates A, E, B and D at the registered shape.

Writes ``receipts/k11b_gates.json`` (the K11 ``k11_gate_receipt.py`` pattern,
the per-channel DECAY vector and the per-channel write gate replacing the
gate scalar and the dense lane):

* **Gate A** -- the parity lanes at the registered shape (batch 8, 2048
  tokens, 2 heads, rank 128, value_dim 128, key_dim 16, bf16 inputs): the
  2-lane schedule fields, the payload lanes, the read WEIGHTS, the
  pre-decision liveness, the exit state/liveness AND the exit key table
  bit-equal against the eager per-token oracle, in both GATED combinations;
  the read at the measured fp32-vs-fp64 floor; **the 0-flip
  decision-invariance receipt (P2)** under random per-channel gates AND
  random per-channel decay vectors; **the birth-parity receipt (P5)** --
  at the registered scalar broadcast per channel and unity gates the k11b
  scan is bit-identical to K10's OWN launch on every channel; and the
  forward kernel's register/spill report.
* **Gate E** -- the K12-flagged multi-lane MERGE corner, NOT COPIED: the
  K10/K11 oracle's ``if sel["merge"].any():`` block index-puts zeros at
  ``second_index`` for EVERY (batch, head) lane whenever ANY lane merges.
  The k11b oracle (a NEW file) guards PER LANE, matching the kernel's own
  per-lane guard; this gate demonstrates the fix on the crafted
  heterogeneous two-lane fixture where the K11 oracle diverges from its
  own kernel: the k11b oracle and the k11b kernel agree bit for bit, and
  the MERGE structure of every k11b receipt fixture is measured on the
  kernel path.
* **Gate B** -- the fp64 gradcheck of the oracle (the instrument validates
  first; the PER-CHANNEL DECAY vector and the PER-CHANNEL GATES included),
  the fp32-vs-fp64 floor per output measured on the gate fixture, the
  kernel path's SEVEN gradients vs the floor (``NOISE_FACTOR`` 8x), the
  per-channel segment-recompute bit-equality through the lane-specific
  replay kernel, and the two-step wake-up receipt (the zero-born scalars
  receive gradient at step 0; the projections' step-0 gradient is exactly
  zero -- receipted so the step-0 zero is not read as a dead channel).
* **Gate D** -- block-call walls at the registered shape, measured against
  K10's AND K11's OWN launches on the same fixture and box (honest
  same-session comparisons, not citations), plus the honest per-channel
  cost statement.

Usage:

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
    experiments.expressivity_tasks.x45_matrix_emender.k11b_gate_receipt \\
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k11b_gates.json
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
    _k10_forward_launch)
from experiments.expressivity_tasks.x45_matrix_emender.k11_learned_gates import (
    _k11_forward_launch, gated_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates import (
    LANES, _k11b_backward_body, _k11b_forward_launch,
    channel_gated_oracle_scan, channel_persistent_trajectory)
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
# The registered shape (the K10/K11 gate receipts' own fixture).
BATCH, TOKENS, HEADS, RANK, VALUE_DIM, KEY_DIM = 8, 2048, 2, 128, 128, 16
SEGMENT_TOKENS = 64
GRAD_NAMES = ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dgates",
              "dstate", "dkeys")


def registered_fixture(seed: int = 77, batch: int = BATCH, tokens: int = TOKENS,
                       dtype: torch.dtype = torch.bfloat16):
    """The K5A-bar fixture: payload AND write-key directions from a hot set
    of fixed directions (the identity-cache regime).  Gate A runs it bf16
    (the registered autocast regime); gate B runs it fp32 (the floor
    instrument's regime).  The DECAY channel is the per-channel vector."""
    generator = torch.Generator().manual_seed(seed)
    distinct = 512
    payload_dirs = torch.randn(distinct, VALUE_DIM, generator=generator)
    key_dirs = torch.randn(distinct, KEY_DIM, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, HEADS), generator=generator)
    payload = payload_dirs[choice].to(dtype=dtype).cuda()
    write_keys = key_dirs[choice].to(dtype=dtype).cuda()
    query_keys = (torch.randn(batch, tokens, HEADS, KEY_DIM, generator=generator)
                  .to(dtype=dtype).cuda())
    spread = (torch.rand(batch, tokens, HEADS, VALUE_DIM, generator=generator)
              * 0.2 - 0.1)
    decay = (0.95 + spread).clamp(min=1e-4, max=1.0 - 1e-6).to(dtype=dtype).cuda()
    return payload, write_keys, query_keys, decay


def gate_stream(batch, tokens, heads, seed, dtype=torch.bfloat16):
    """Random PER-CHANNEL gates in (0.05, 0.95)."""
    generator = torch.Generator().manual_seed(seed)
    gates = (torch.rand(batch, tokens, heads, VALUE_DIM,
                        generator=generator) * 0.9 + 0.05
             ).to(dtype=dtype).cuda()
    return gates


def empty_store(batch=None, heads=None):
    batch = BATCH if batch is None else batch
    heads = HEADS if heads is None else heads
    state = torch.zeros(batch, heads, RANK, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(batch, heads, RANK, dtype=torch.bool, device="cuda")
    keys = torch.zeros(batch, heads, RANK, KEY_DIM, dtype=torch.float32,
                       device="cuda")
    return state, live, keys


def gate_a(receipt: dict) -> dict:
    payload, write_keys, query_keys, decay = registered_fixture()
    gates = gate_stream(BATCH, TOKENS, HEADS, 91)
    state, live, keys = empty_store()
    lanes = flips = 0
    rows = {}
    for gated in (True, False):
        out = _k11b_forward_launch(payload, write_keys, query_keys, decay,
                                   gates, state, live, keys, POL32,
                                   RANK, SEGMENT_TOKENS, gated,
                                   collect=True, warps=8, stages=3)
        with torch.no_grad():
            ref = channel_gated_oracle_scan(payload, write_keys, query_keys,
                                            decay, gates, policy=POLICY,
                                            gated=gated, rank=RANK,
                                            rank_max=RANK, collect=True)
        entry = dict(gated=gated)
        bad = []
        for name in ("slot", "erase", "op", "live_after"):
            if not torch.equal(getattr(ref["schedule"], name), out[name]):
                bad.append(name)
        if not torch.equal(ref["schedule"].payload, out["payload"]):
            bad.append("payload")
        for name in ("weights", "state", "live", "live_pre", "keys"):
            if not torch.equal(ref[name], out[name]):
                bad.append(name)
        entry["bit_equal_channels"] = bad == []
        entry["mismatched"] = bad
        with torch.no_grad():
            ref64 = channel_gated_oracle_scan(payload.double(),
                                              write_keys.double(),
                                              query_keys.double(),
                                              decay.double(), gates.double(),
                                              policy=POLICY, gated=gated,
                                              rank=RANK, rank_max=RANK,
                                              collect=False)
        floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
        deviation = (ref["reads"] - out["reads"]).abs().max().item()
        entry["read"] = dict(floor_fp32_vs_fp64=floor,
                            deviation=deviation,
                            within_noise_factor=bool(
                                deviation <= NOISE_FACTOR * max(floor, 1e-30)))
        rows[f"gated={gated}"] = entry
        lanes += int(ref["schedule"].slot.numel()
                     + ref["schedule"].payload.numel())
    # the 0-flip decision-invariance receipt (P2): the registered scalar
    # broadcast per channel + unity gates vs random per-channel gates AND
    # random per-channel decay vectors
    generator = torch.Generator().manual_seed(5)
    ones = torch.ones(BATCH, TOKENS, HEADS, VALUE_DIM, dtype=torch.bfloat16,
                      device="cuda")
    base_decay = torch.full((BATCH, TOKENS, HEADS), 0.95,
                           dtype=torch.bfloat16, device="cuda")
    base_decay = base_decay.unsqueeze(-1).expand(
        BATCH, TOKENS, HEADS, VALUE_DIM).contiguous()
    base = _k11b_forward_launch(payload, write_keys, query_keys, base_decay,
                                ones, state, live, keys, POL32, RANK,
                                SEGMENT_TOKENS, True, collect=True, warps=8,
                                stages=3)
    vary_decay = (torch.rand(BATCH, TOKENS, HEADS, VALUE_DIM,
                             generator=generator) * 0.9 + 0.05
                  ).to(torch.bfloat16).cuda()
    vary = _k11b_forward_launch(payload, write_keys, query_keys, vary_decay,
                                gates, state, live, keys, POL32, RANK,
                                SEGMENT_TOKENS, True, collect=True, warps=8,
                                stages=3)
    for name in ("slot", "erase", "op", "live_after"):
        flips += int((base[name] != vary[name]).sum())
    # the birth-parity receipt (P5): k11b at the registered scalar broadcast
    # per channel + unity gates vs K10's OWN launch
    k10 = _k10_forward_launch(payload, write_keys, query_keys,
                              base_decay[:, :, :, 0].contiguous(), state,
                              live, keys, POL32, RANK, SEGMENT_TOKENS,
                              collect=True, warps=8, stages=3)
    k11b = _k11b_forward_launch(payload, write_keys, query_keys, base_decay,
                                ones, state, live, keys, POL32, RANK,
                                SEGMENT_TOKENS, True, collect=True, warps=8,
                                stages=3)
    birth_diffs = {}
    for name in ("slot", "erase", "op", "live_after", "payload"):
        a = k10[name]
        birth_diffs[name] = int((a != k11b[name]).sum()) \
            if a.shape == k11b[name].shape else None
    for name in ("reads", "state", "live", "live_pre", "weights", "keys"):
        birth_diffs[name] = int((k10[name] != k11b[name]).sum())
    hist = op_histogram(MatrixSchedule(slot=k11b["slot"], erase=k11b["erase"],
                                       payload=k11b["payload"], op=k11b["op"],
                                       live_after=k11b["live_after"]))
    gate = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM, dtype="bf16 "
                     "inputs, fp32 defined-order math",
                     segment_tokens=SEGMENT_TOKENS, lanes=LANES,
                     distinct_directions=512,
                     decay_channel="per-channel vector (0.85-1.0 spread)"),
        channels=rows,
        total=dict(lanes_compared=lanes, decision_flips=flips),
        op_histogram=hist,
        birth_parity=dict(differing_elements=birth_diffs,
                          bit_identical=all(v == 0 for v in
                                           birth_diffs.values())),
    )
    # the register report from the compiled kernel's own metadata
    try:
        meta = k11b["kernel"]
        gate["forward_kernel_resources"] = dict(
            num_warps=8,
            registers=getattr(meta, "n_regs", None),
            spills=getattr(meta, "n_spills", None))
    except Exception:
        pass
    receipt["gate_a"] = gate
    return gate


def gate_e(receipt: dict) -> dict:
    """The K12-flagged multi-lane MERGE corner: NOT COPIED into the k11b
    oracle.  The K10/K11 oracle's ``if sel["merge"].any():`` block zeroes
    EVERY (batch, head) lane's second row whenever ANY lane merges; the
    k11b oracle guards per lane (the kernel's own semantics).  This gate
    demonstrates, on the K11 gate-E fixture verbatim: (a) the K11 oracle
    diverges from its own kernel on the non-merging lane's live second row
    (the committed engine's receipted corner, reproduced); (b) the k11b
    oracle agrees with the k11b kernel BIT FOR BIT on the same fixture.
    """
    V, K = 8, 8
    e = torch.eye(V, device="cuda")
    ek = torch.eye(K, device="cuda")
    key1 = 0.9 * ek[0] + (1 - 0.81) ** 0.5 * ek[1]

    def heterogeneous_fixture():
        # head 0 MERGES (second-key similarity 0.9 >= tau_merge); head 1
        # CORRECTs with a live second row (K11 gate E's own fixture)
        state = torch.zeros(1, 2, 4, V, dtype=torch.float32, device="cuda")
        keys = torch.zeros(1, 2, 4, K, dtype=torch.float32, device="cuda")
        live = torch.zeros(1, 2, 4, dtype=torch.bool, device="cuda")
        for h, rows in enumerate(([e[0], e[2], e[3], e[4]],
                                  [e[1], e[2], e[3], e[4]])):
            key_rows = ([ek[0], key1, ek[2], ek[3]] if h == 0
                        else [ek[1], ek[2], ek[3], ek[4]])
            for r, (row, krow) in enumerate(zip(rows, key_rows)):
                state[0, h, r] = row
                keys[0, h, r] = krow
                live[0, h, r] = True
        payload = torch.cat([e[0].view(1, 1, 1, V), e[1].view(1, 1, 1, V)],
                           dim=2).repeat(1, 8, 1, 1)
        write_keys = torch.cat([ek[0].view(1, 1, 1, K),
                                ek[1].view(1, 1, 1, K)],
                               dim=2).repeat(1, 8, 1, 1)
        query_keys = torch.zeros(1, 8, 2, K, device="cuda")
        decay = torch.full((1, 8, 2, V), 0.999, device="cuda")
        gates = torch.ones(1, 8, 2, V, device="cuda")
        return payload, write_keys, query_keys, decay, gates, state, live, keys

    payload, write_keys, query_keys, decay, gates, state, live, keys = \
        heterogeneous_fixture()
    # (a) the committed K11 engine's corner, reproduced: oracle vs kernel
    dense0 = torch.zeros_like(payload)
    k11_out = _k11_forward_launch(payload, dense0, write_keys, query_keys,
                                  decay[:, :, :, 0].contiguous(),
                                  gates[:, :, :, 0].contiguous(), state, live,
                                  keys, POL32, 4, 8, True, False,
                                  collect=True, warps=4, stages=3)
    with torch.no_grad():
        k11_ref = gated_oracle_scan(payload, dense0, write_keys, query_keys,
                                    decay[:, :, :, 0].contiguous(),
                                    gates[:, :, :, 0].contiguous(),
                                    policy=POLICY, gated=True, use_dense=False,
                                    rank=4, rank_max=4, initial_state=state,
                                    initial_live=live, initial_keys=keys,
                                    collect=True)
    k11_corner = dict(
        state=int((k11_ref["state"] != k11_out["state"]).sum()),
        live=int((k11_ref["live"] != k11_out["live"]).sum()),
        keys=int((k11_ref["keys"] != k11_out["keys"]).sum()),
        schedule=int(sum((getattr(k11_ref["schedule"], n) != k11_out[n]).sum()
                         for n in ("slot", "erase", "op", "live_after"))))
    # (b) the k11b oracle (per-lane guarded) vs the k11b kernel: bit-equal
    k11b_out = _k11b_forward_launch(payload, write_keys, query_keys, decay,
                                    gates, state, live, keys, POL32, 4, 8,
                                    True, collect=True, warps=4, stages=3)
    with torch.no_grad():
        k11b_ref = channel_gated_oracle_scan(payload, write_keys, query_keys,
                                             decay, gates, policy=POLICY,
                                             gated=True, rank=4, rank_max=4,
                                             initial_state=state,
                                             initial_live=live,
                                             initial_keys=keys, collect=True)
    k11b_diffs = dict(
        state=int((k11b_ref["state"] != k11b_out["state"]).sum()),
        live=int((k11b_ref["live"] != k11b_out["live"]).sum()),
        keys=int((k11b_ref["keys"] != k11b_out["keys"]).sum()),
        schedule=int(sum((getattr(k11b_ref["schedule"], n) != k11b_out[n]).sum()
                        for n in ("slot", "erase", "op", "live_after"))),
        payload=int((k11b_ref["schedule"].payload != k11b_out["payload"]).sum()))
    hist = op_histogram(k11b_ref["schedule"])
    receipt["gate_e"] = dict(
        note="the K12-flagged multi-lane MERGE corner is NOT copied: the "
             "k11b oracle (a new file) guards the MERGE-source close PER "
             "LANE, the kernel's own semantics; no committed engine is "
             "touched",
        k11_committed_corner_reproduced=dict(
            differing=k11_corner,
            verdict="the K11 oracle diverges from its own kernel at the "
                    "non-merging lane's live second row (K11_GATES.md 2's "
                    "receipt, reproduced on this fixture)"),
        k11b_oracle_vs_kernel=dict(
            differing=k11b_diffs,
            bit_identical=all(v == 0 for v in k11b_diffs.values())),
        fixture_merge_histogram=hist)
    return receipt["gate_e"]


def gate_b(receipt: dict) -> dict:
    # 1. the fp64 gradcheck of the oracle (the instrument validates first).
    generator = torch.Generator().manual_seed(11)
    batch, tokens, heads, rank, value_dim, key_dim = 2, 48, 2, 8, 16, 4
    distinct = 6
    payload_dirs = torch.randn(distinct, value_dim, generator=generator)
    key_dirs = torch.randn(distinct, key_dim, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, heads),
                           generator=generator)
    p = payload_dirs[choice].double().cuda().requires_grad_(True)
    wk = key_dirs[choice].double().cuda().requires_grad_(True)
    qk = (torch.randn(batch, tokens, heads, key_dim, generator=generator)
          * 0.5).double().cuda().requires_grad_(True)
    d = (torch.full((batch, tokens, heads, 1), 0.95).double()
         + (torch.rand(batch, tokens, heads, value_dim,
                       generator=generator) * 0.2 - 0.1).double()
         ).clamp(min=1e-4, max=1.0 - 1e-6).cuda().requires_grad_(True)
    g_ = (torch.rand(batch, tokens, heads, value_dim, generator=generator)
          * 0.6 + 0.2).double().cuda().requires_grad_(True)
    s = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)
    k = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)
    holder = {}

    def f(p_, wk_, qk_, d_, g__, s_, k_):
        ref = channel_gated_oracle_scan(p_, wk_, qk_, d_, g__, policy=POLICY,
                                        gated=True, rank=rank, rank_max=rank,
                                        initial_state=s_, initial_keys=k_)
        holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    gradcheck_ok = bool(torch.autograd.gradcheck(
        f, (p, wk, qk, d, g_, s, k), eps=1e-6, atol=1e-4, rtol=1e-4))

    # 2. the floors and the kernel path's seven gradients, on a gate-class
    #    shape (fp32 inputs: the floor instrument's regime).
    payload32, write_keys32, query_keys32, decay32 = registered_fixture(
        seed=23, batch=2, tokens=256, dtype=torch.float32)
    gates32 = gate_stream(2, 256, 2, 47, dtype=torch.float32)
    state, live, keys = empty_store(batch=2)
    with torch.no_grad():
        ref = channel_gated_oracle_scan(payload32, write_keys32, query_keys32,
                                        decay32, gates32, policy=POLICY,
                                        gated=True, rank=RANK, rank_max=RANK,
                                        collect=True)
    leaves = [payload32.clone().requires_grad_(True),
              write_keys32.clone().requires_grad_(True),
              query_keys32.clone().requires_grad_(True),
              decay32.clone().requires_grad_(True),
              gates32.clone().requires_grad_(True)]
    s32 = state.clone().requires_grad_(True)
    k32 = keys.clone().requires_grad_(True)
    ref3 = channel_gated_oracle_scan(*leaves, policy=POLICY, gated=True,
                                    rank=RANK, rank_max=RANK,
                                    initial_state=s32, initial_keys=k32,
                                    collect=False)
    upstream = torch.randn(ref3["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).cuda()
    (ref3["reads"] * upstream).sum().backward()
    s64 = state.double().requires_grad_(True)
    k64 = keys.double().requires_grad_(True)
    leaves64 = [t.double().clone().requires_grad_(True) for t in
                (payload32, write_keys32, query_keys32, decay32, gates32)]
    ref64 = channel_gated_oracle_scan(*leaves64, policy=POLICY, gated=True,
                                      rank=RANK, rank_max=RANK,
                                      initial_state=s64, initial_keys=k64,
                                      collect=False)
    (ref64["reads"] * upstream.double()).sum().backward()
    out = _k11b_forward_launch(payload32, write_keys32, query_keys32,
                               decay32, gates32, state, live, keys, POL32,
                               RANK, SEGMENT_TOKENS, True, collect=True,
                               warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    grads = _k11b_backward_body(payload32, write_keys32, query_keys32,
                                decay32, gates32, upstream, schedule,
                                out["live_pre"], out["ckpt_state"],
                                out["ckpt_live"], out["ckpt_keys"], RANK,
                                SEGMENT_TOKENS, POLICY.read_scale, True)
    oracle32 = (*[t.grad for t in leaves], s32.grad, k32.grad)
    oracle64 = (*[t.grad for t in leaves64], s64.grad, k64.grad)
    rows = {}
    for name, kern, o32, o64 in zip(GRAD_NAMES, grads, oracle32, oracle64):
        floor = (o32.float() - o64.float()).abs().max().item()
        deviation = (kern.float() - o32.float()).abs().max().item()
        rows[name] = dict(floor_fp32_vs_fp64=floor, kernel_deviation=deviation,
                          within_noise_factor=bool(
                              deviation <= NOISE_FACTOR * max(floor, 1e-6)
                              or deviation <= 1e-5))
    # 3. the 2-lane segment-recompute bit-equality through the lane-specific
    #    per-channel replay kernel.
    replay_ok = True
    decay_p = decay32.permute(0, 2, 1, 3).contiguous()
    for index, start in enumerate(range(0, 256, SEGMENT_TOKENS)):
        stop = min(start + SEGMENT_TOKENS, 256)
        _, exit_state, _exit_live, _traj = channel_persistent_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index],
            decay_p[:, :, start:stop, :], schedule.chunk(start, stop))
        if stop == 256:
            replay_ok &= bool(torch.equal(exit_state, out["state"]))
        else:
            replay_ok &= bool(torch.equal(exit_state,
                                          out["ckpt_state"][index + 1]))
    # 4. the two-step wake-up receipt: the zero-born scalars receive
    #    gradient at birth; the projections' step-0 gradient is exactly 0
    #    (receipted so the step-0 zero is not read as a dead channel).
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys
    from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates \
        import convert_model_to_channel_gates

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            from experiments.expressivity_tasks.x45_matrix_emender.\
                k10_separated_keys import convert_model_to_separated_keys
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)
            self.blocks[0].mixer = convert_model_to_separated_keys(
                self).blocks[0].mixer

    stub = Stub()
    stub = convert_model_to_channel_gates(stub, arm="p3")
    mixer = stub.blocks[0].mixer.cuda()
    x = torch.randn(2, 32, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_m, _ = mixer.mix(x)
    out_m.float().pow(2).sum().backward()
    wake = {}
    for name in ("decay_scalar", "write_scalar", "read_scalar"):
        wake[name] = dict(step0_grad_norm=float(
            getattr(mixer, name).grad.abs().sum()))
    for name in ("f_proj", "b_proj", "g_proj"):
        projection = getattr(mixer, name)
        wake[name] = dict(step0_grad_norm=float(sum(
            p.grad.abs().sum() for p in projection.parameters())))
    with torch.no_grad():
        lam_vec, lam_inc = mixer.channel_decay(x)
        birth_bit_equal = bool(torch.equal(
            lam_vec, lam_inc.unsqueeze(-1).expand_as(lam_vec)
            .to(lam_vec.dtype)))
    receipt["gate_b"] = dict(
        oracle_gradcheck_fp64=dict(passed=gradcheck_ok,
                                   hist=holder.get("hist"),
                                   branches="APPEND admission, CORRECT "
                                            "blend, the PER-CHANNEL decay "
                                            "vector, the PER-CHANNEL gates, "
                                            "keys, state"),
        gradient_floors=rows,
        replay_bit_exact=replay_ok,
        two_step_wake_up=dict(
            scalars=wake,
            note="the zero-born per-head scalars receive gradient at step 0 "
                 "(through sigmoid(0) = 0.5); the projections' step-0 "
                 "gradient is EXACTLY zero while m == 0 (the K11 two-step "
                 "wake-up pattern) -- receipted so the step-0 zero is not "
                 "read as a dead channel",
            birth_decay_bit_equal_to_registered_scalar=birth_bit_equal))
    return receipt["gate_b"]


def gate_d(receipt: dict) -> dict:
    """Block-call walls at the registered shape, K10's AND K11's own
    launches measured on the same fixture and box (honest same-session
    comparisons)."""
    payload, write_keys, query_keys, decay = registered_fixture()
    gates = gate_stream(BATCH, TOKENS, HEADS, 91)
    state, live, keys = empty_store()
    dense0 = torch.zeros_like(payload)

    def k11b_once():
        return _k11b_forward_launch(payload, write_keys, query_keys, decay,
                                     gates, state, live, keys, POL32,
                                     RANK, SEGMENT_TOKENS, True,
                                     collect=False, warps=8, stages=3)

    def k11_once():
        return _k11_forward_launch(payload, dense0, write_keys, query_keys,
                                    decay[:, :, :, 0].contiguous(),
                                    gates[:, :, :, 0].contiguous(), state,
                                    live, keys, POL32, RANK, SEGMENT_TOKENS,
                                    True, False, collect=False, warps=8,
                                    stages=3)

    def k10_once():
        return _k10_forward_launch(payload, write_keys, query_keys,
                                    decay[:, :, :, 0].contiguous(), state,
                                    live, keys, POL32, RANK, SEGMENT_TOKENS,
                                    collect=False, warps=8, stages=3)

    for _ in range(3):
        k11b_once(); k11_once(); k10_once()
    torch.cuda.synchronize()
    repeats = 5
    fwd_ms = {}
    for name, fn in (("k11b", k11b_once), ("k11", k11_once), ("k10", k10_once)):
        started = time.monotonic()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        fwd_ms[name] = (time.monotonic() - started) / repeats * 1e3

    o = k11b_once()
    schedule = MatrixSchedule(slot=o["slot"], erase=o["erase"],
                              payload=o["payload"], op=o["op"],
                              live_after=o["live_after"])
    upstream = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(4)).cuda()

    def k11b_bwd():
        return _k11b_backward_body(payload, write_keys, query_keys, decay,
                                   gates, upstream, schedule,
                                   o["live_pre"], o["ckpt_state"],
                                   o["ckpt_live"], o["ckpt_keys"], RANK,
                                   SEGMENT_TOKENS, POLICY.read_scale, True)

    from experiments.expressivity_tasks.x45_matrix_emender.k11_learned_gates \
        import _k11_backward_body
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import _k10_backward_body

    def k11_bwd():
        return _k11_backward_body(payload, dense0, write_keys, query_keys,
                                   decay[:, :, :, 0].contiguous(),
                                   gates[:, :, :, 0].contiguous(), upstream,
                                   sched11, o11["live_pre"],
                                   o11["ckpt_state"], o11["ckpt_live"],
                                   o11["ckpt_keys"], RANK, SEGMENT_TOKENS,
                                   POLICY.read_scale, True, False)

    def k10_bwd():
        return _k10_backward_body(payload, write_keys, query_keys,
                                   decay[:, :, :, 0].contiguous(), upstream,
                                   sched10, o10["live_pre"],
                                   o10["ckpt_state"], o10["ckpt_live"],
                                   o10["ckpt_keys"], RANK, SEGMENT_TOKENS,
                                   POLICY.read_scale)

    # the K10/K11 gate-D pattern: schedules computed ONCE outside the timed
    # loop, so the backward walls are backward-only on ALL sides
    o11 = k11_once()
    sched11 = MatrixSchedule(slot=o11["slot"], erase=o11["erase"],
                             payload=o11["payload"], op=o11["op"],
                             live_after=o11["live_after"])
    o10 = k10_once()
    sched10 = MatrixSchedule(slot=o10["slot"], erase=o10["erase"],
                             payload=o10["payload"], op=o10["op"],
                             live_after=o10["live_after"])

    for _ in range(2):
        k11b_bwd(); k11_bwd(); k10_bwd()
    torch.cuda.synchronize()
    bwd_ms = {}
    for name, fn in (("k11b", k11b_bwd), ("k11", k11_bwd), ("k10", k10_bwd)):
        started = time.monotonic()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        bwd_ms[name] = (time.monotonic() - started) / repeats * 1e3
    tok_per_scan = BATCH * TOKENS
    receipt["gate_d"] = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM,
                     segment_tokens=SEGMENT_TOKENS),
        forward_ms_per_scan={k: round(v, 2) for k, v in fwd_ms.items()},
        backward_ms_per_scan={k: round(v, 2) for k, v in bwd_ms.items()},
        forward_overhead=dict(vs_k10=round(fwd_ms["k11b"] / fwd_ms["k10"], 3),
                              vs_k11=round(fwd_ms["k11b"] / fwd_ms["k11"], 3)),
        backward_overhead=dict(vs_k10=round(bwd_ms["k11b"] / bwd_ms["k10"], 3),
                               vs_k11=round(bwd_ms["k11b"] / bwd_ms["k11"], 3)),
        forward_throughput_tok_per_s=round(
            tok_per_scan / (fwd_ms["k11b"] / 1e3)),
        per_channel_cost_note=(
            "the per-channel form adds per token: one (VP,) DECAY vector "
            "load and one (VP,) gate load (vs K11's two scalars), the "
            "per-channel CORRECT blend (elementwise in the value dim), and "
            "in the backward one (BV,) dlam and one (BV,) dgate store per "
            "(lane, block, token) instead of the per-block scalar "
            "partials; measured against K10's AND K11's own launches on "
            "the same fixture and box, not estimated"))
    return receipt["gate_d"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                       default=str(ROOT / "receipts" / "k11b_gates.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "the gates need CUDA"
    receipt = dict(probe="k11b_channel_gates_gates",
                   registration=str(ROOT / "PREREGISTRATION_CHANNEL_GATES.md"),
                   shape=dict(batch=BATCH, tokens=TOKENS, heads=HEADS,
                              rank=RANK, value_dim=VALUE_DIM,
                              key_dim=KEY_DIM, segment_tokens=SEGMENT_TOKENS,
                              lanes=LANES),
                   policy=POLICY.record(), torch=torch.__version__)
    gate_a(receipt)
    gate_e(receipt)
    gate_b(receipt)
    gate_d(receipt)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("gate_a", "gate_e", "gate_b",
                                             "gate_d")},
                     indent=1)[:6000])


if __name__ == "__main__":
    main()
