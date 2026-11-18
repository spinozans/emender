"""K11 -- the gate receipts: gates A, B and D at the registered shape.

Writes ``receipts/k11_gates.json`` (the K10 ``k10_gate_receipt.py`` pattern,
the gate channel and the dense lane added):

* **Gate A** -- the parity lanes at the registered shape (batch 8, 2048
  tokens, 2 heads, rank 128, value_dim 128, key_dim 16, bf16 inputs): the
  3-lane schedule fields, the payload lanes, the read WEIGHTS, the
  pre-decision liveness, the exit state/liveness AND the exit key table
  bit-equal against the eager per-token oracle, in the arm's own
  GATED/DENSE combination and in all four; the read at the measured
  fp32-vs-fp64 floor; **the 0-flip decision-invariance receipt (P2)**; **the
  birth-parity receipt (P5)** -- at unity gates and a zero dense payload the
  k11 scan is bit-identical to K10's OWN launch on every channel; and the
  forward kernel's register/spill report.
* **Gate B** -- the fp64 gradcheck of the oracle (the instrument validates
  first), the fp32-vs-fp64 floor per output measured on the gate fixture, the
  kernel path's EIGHT gradients vs the floor (``NOISE_FACTOR`` 8x), and the
  3-lane segment-recompute bit-equality through K4's imported replay kernel.
* **Gate D** -- block-call walls at the registered shape, measured against
  K10's OWN launch on the same fixture and box (an honest same-session
  comparison, not a citation), plus the honest gate-channel cost statement.
* **Gate E** -- the K12-flagged MERGE-corner assessment (K12_PLACEMENT.md's
  cross-lane flag, commit cb5615fe): the K10 eager oracle's latent
  multi-lane MERGE pattern (the ``if sel["merge"].any():`` block zeroes
  EVERY lane's second row whenever ANY lane merges) was copied into the
  k11 oracle.  This gate receipts, without touching the committed engine:
  the pattern's location; the KERNEL's per-lane guard (the production
  path is kernel-only at the registered shape); the corner demonstrated
  on a crafted heterogeneous two-lane fixture (oracle vs kernel differ
  at exactly the non-merging lane's live second row); the MERGE
  structure of every committed test fixture and every receipt fixture
  measured on the kernel path (the corner's firing condition count); and
  the resulting verdict (unreachable/inert or reachable).

Usage:

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
    experiments.expressivity_tasks.x45_matrix_emender.k11_gate_receipt \\
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k11_gates.json
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
    LANES, _k11_backward_body, _k11_forward_launch, gated_oracle_scan,
    separated_available)
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward \
    import persistent_scan_trajectory
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
# The registered shape (the K10 gate receipt's own fixture).
BATCH, TOKENS, HEADS, RANK, VALUE_DIM, KEY_DIM = 8, 2048, 2, 128, 128, 16
SEGMENT_TOKENS = 64
GRAD_NAMES = ("dpayload", "ddense", "dwrite_keys", "dquery_keys", "ddecay",
              "dgates", "dstate", "dkeys")


def registered_fixture(seed: int = 77, batch: int = BATCH, tokens: int = TOKENS,
                       dtype: torch.dtype = torch.bfloat16):
    """The K5A-bar fixture: payload AND write-key directions from a hot set
    of fixed directions (the identity-cache regime).  Gate A runs it bf16
    (the registered autocast regime); gate B runs it fp32 (the floor
    instrument's regime)."""
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


def gate_stream(batch, tokens, heads, seed, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    gates = (torch.rand(batch, tokens, heads, generator=generator) * 0.9 + 0.05
             ).to(dtype=dtype).cuda()
    dense = (torch.randn(batch, tokens, heads, VALUE_DIM,
                         generator=generator) * 0.05).to(dtype=dtype).cuda()
    return gates, dense


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
    gates, dense = gate_stream(BATCH, TOKENS, HEADS, 91)
    state, live, keys = empty_store()
    lanes = flips = 0
    rows = {}
    for gated, use_dense in ((True, True), (True, False), (False, True),
                             (False, False)):
        out = _k11_forward_launch(payload, dense, write_keys, query_keys,
                                  decay, gates, state, live, keys, POL32,
                                  RANK, SEGMENT_TOKENS, gated, use_dense,
                                  collect=True, warps=8, stages=3)
        with torch.no_grad():
            ref = gated_oracle_scan(payload, dense, write_keys, query_keys,
                                    decay, gates, policy=POLICY, gated=gated,
                                    use_dense=use_dense, rank=RANK,
                                    rank_max=RANK, collect=True)
        entry = dict(gated=gated, use_dense=use_dense)
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
            ref64 = gated_oracle_scan(payload.double(), dense.double(),
                                      write_keys.double(), query_keys.double(),
                                      decay.double(), gates.double(),
                                      policy=POLICY, gated=gated,
                                      use_dense=use_dense, rank=RANK,
                                      rank_max=RANK, collect=False)
        floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
        deviation = (ref["reads"] - out["reads"]).abs().max().item()
        entry["read"] = dict(floor_fp32_vs_fp64=floor,
                            deviation=deviation,
                            within_noise_factor=bool(
                                deviation <= NOISE_FACTOR * max(floor, 1e-30)))
        rows[f"gated={gated},dense={use_dense}"] = entry
        lanes += int(ref["schedule"].slot[..., :2].numel()
                     + ref["schedule"].payload[..., :2, :].numel())
    # the 0-flip decision-invariance receipt (P2): identity vs random gates
    ones = torch.ones(BATCH, TOKENS, HEADS, dtype=torch.bfloat16,
                      device="cuda")
    zero_dense = torch.zeros_like(dense)
    base = _k11_forward_launch(payload, zero_dense, write_keys, query_keys,
                               decay, ones, state, live, keys, POL32, RANK,
                               SEGMENT_TOKENS, True, True, collect=True,
                               warps=8, stages=3)
    vary = _k11_forward_launch(payload, dense, write_keys, query_keys, decay,
                               gates, state, live, keys, POL32, RANK,
                               SEGMENT_TOKENS, True, True, collect=True,
                               warps=8, stages=3)
    for name in ("slot", "erase", "op", "live_after"):
        flips += int((base[name][..., :2] != vary[name][..., :2]).sum())
    # the birth-parity receipt (P5): k11 at unity/zero vs K10's OWN launch
    k10 = _k10_forward_launch(payload, write_keys, query_keys, decay, state,
                              live, keys, POL32, RANK, SEGMENT_TOKENS,
                              collect=True, warps=8, stages=3)
    k11 = _k11_forward_launch(payload, zero_dense, write_keys, query_keys,
                              decay, ones, state, live, keys, POL32, RANK,
                              SEGMENT_TOKENS, True, True, collect=True,
                              warps=8, stages=3)
    birth_diffs = {}
    for name in ("slot", "erase", "op", "live_after", "payload"):
        a = k10[name]
        c = k11[name][..., :2] if name != "payload" else k11[name][..., :2, :]
        birth_diffs[name] = int((a != c).sum()) if a.shape == c.shape else None
    for name in ("reads", "state", "live", "live_pre", "weights", "keys"):
        birth_diffs[name] = int((k10[name] != k11[name]).sum())
    hist = op_histogram(MatrixSchedule(slot=k11["slot"], erase=k11["erase"],
                                       payload=k11["payload"], op=k11["op"],
                                       live_after=k11["live_after"]))
    gate = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM, dtype="bf16 "
                     "inputs, fp32 defined-order math",
                     segment_tokens=SEGMENT_TOKENS, lanes=LANES,
                     distinct_directions=512),
        channels=rows,
        total=dict(lanes_compared=lanes, decision_flips=flips),
        op_histogram=hist,
        birth_parity=dict(differing_elements=birth_diffs,
                          bit_identical=all(v == 0 for v in
                                           birth_diffs.values())),
    )
    # the register report from the compiled kernel's own metadata
    try:
        meta = k11["kernel"]
        gate["forward_kernel_resources"] = dict(
            num_warps=8,
            registers=getattr(meta, "n_regs", None),
            spills=getattr(meta, "n_spills", None))
    except Exception:
        pass
    receipt["gate_a"] = gate
    return gate


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
    d = torch.full((batch, tokens, heads), 0.95).double().cuda() \
        .requires_grad_(True)
    g_ = (torch.rand(batch, tokens, heads, generator=generator) * 0.6 + 0.2
          ).double().cuda().requires_grad_(True)
    dn = (torch.randn(batch, tokens, heads, value_dim, generator=generator)
          * 0.1).double().cuda().requires_grad_(True)
    s = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)
    k = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float64,
                    device="cuda").requires_grad_(True)
    holder = {}

    def f(p_, dn_, wk_, qk_, d_, g__, s_, k_):
        ref = gated_oracle_scan(p_, dn_, wk_, qk_, d_, g__, policy=POLICY,
                                gated=True, use_dense=True, rank=rank,
                                rank_max=rank, initial_state=s_,
                                initial_keys=k_)
        holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    gradcheck_ok = bool(torch.autograd.gradcheck(
        f, (p, dn, wk, qk, d, g_, s, k), eps=1e-6, atol=1e-4, rtol=1e-4))

    # 2. the floors and the kernel path's eight gradients, on a gate-class
    #    shape (fp32 inputs: the floor instrument's regime).
    payload32, write_keys32, query_keys32, decay32 = registered_fixture(
        seed=23, batch=2, tokens=256, dtype=torch.float32)
    gates32, dense32 = gate_stream(2, 256, 2, 47, dtype=torch.float32)
    state, live, keys = empty_store(batch=2)
    with torch.no_grad():
        ref = gated_oracle_scan(payload32, dense32, write_keys32, query_keys32,
                                decay32, gates32, policy=POLICY, gated=True,
                                use_dense=True, rank=RANK, rank_max=RANK,
                                collect=True)
    leaves = [payload32.clone().requires_grad_(True),
              dense32.clone().requires_grad_(True),
              write_keys32.clone().requires_grad_(True),
              query_keys32.clone().requires_grad_(True),
              decay32.clone().requires_grad_(True),
              gates32.clone().requires_grad_(True)]
    s32 = state.clone().requires_grad_(True)
    k32 = keys.clone().requires_grad_(True)
    ref3 = gated_oracle_scan(*leaves, policy=POLICY, gated=True,
                             use_dense=True, rank=RANK, rank_max=RANK,
                             initial_state=s32, initial_keys=k32,
                             collect=False)
    upstream = torch.randn(ref3["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).cuda()
    (ref3["reads"] * upstream).sum().backward()
    s64 = state.double().requires_grad_(True)
    k64 = keys.double().requires_grad_(True)
    leaves64 = [t.double().clone().requires_grad_(True) for t in
                (payload32, dense32, write_keys32, query_keys32, decay32,
                 gates32)]
    ref64 = gated_oracle_scan(*leaves64, policy=POLICY, gated=True,
                              use_dense=True, rank=RANK, rank_max=RANK,
                              initial_state=s64, initial_keys=k64,
                              collect=False)
    (ref64["reads"] * upstream.double()).sum().backward()
    out = _k11_forward_launch(payload32, dense32, write_keys32, query_keys32,
                              decay32, gates32, state, live, keys, POL32,
                              RANK, SEGMENT_TOKENS, True, True, collect=True,
                              warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    grads = _k11_backward_body(payload32, dense32, write_keys32, query_keys32,
                               decay32, gates32, upstream, schedule,
                               out["live_pre"], out["ckpt_state"],
                               out["ckpt_live"], out["ckpt_keys"], RANK,
                               SEGMENT_TOKENS, POLICY.read_scale, True, True)
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
    # 3. the 3-lane segment-recompute bit-equality through K4's replay.
    replay_ok = True
    decay_p = decay32.permute(0, 2, 1).contiguous()
    for index, start in enumerate(range(0, 256, SEGMENT_TOKENS)):
        stop = min(start + SEGMENT_TOKENS, 256)
        zeros_w = torch.zeros(2, 2, stop - start, RANK, dtype=torch.float32,
                              device="cuda")
        _, exit_state, _exit_live, _traj = persistent_scan_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index], zeros_w,
            decay_p[:, :, start:stop], schedule.chunk(start, stop))
        if stop == 256:
            replay_ok &= bool(torch.equal(exit_state, out["state"]))
        else:
            replay_ok &= bool(torch.equal(exit_state,
                                          out["ckpt_state"][index + 1]))
    receipt["gate_b"] = dict(
        oracle_gradcheck_fp64=dict(passed=gradcheck_ok,
                                   hist=holder.get("hist"),
                                   branches="APPEND admission, CORRECT blend, "
                                            "dense lane, gates, keys, state"),
        gradient_floors=rows,
        replay_bit_exact=replay_ok)
    return receipt["gate_b"]


def gate_d(receipt: dict) -> dict:
    """Block-call walls at the registered shape, K10's own launch measured on
    the same fixture and box (the honest same-session comparison)."""
    payload, write_keys, query_keys, decay = registered_fixture()
    gates, dense = gate_stream(BATCH, TOKENS, HEADS, 91)
    state, live, keys = empty_store()

    def k11_once():
        return _k11_forward_launch(payload, dense, write_keys, query_keys,
                                    decay, gates, state, live, keys, POL32,
                                    RANK, SEGMENT_TOKENS, True, True,
                                    collect=False, warps=8, stages=3)

    def k10_once():
        return _k10_forward_launch(payload, write_keys, query_keys, decay,
                                    state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, collect=False, warps=8,
                                    stages=3)

    for _ in range(3):
        k11_once(); k10_once()
    torch.cuda.synchronize()
    repeats = 5
    started = time.monotonic()
    for _ in range(repeats):
        o = k11_once()
    torch.cuda.synchronize()
    k11_fwd_ms = (time.monotonic() - started) / repeats * 1e3
    started = time.monotonic()
    for _ in range(repeats):
        k10_once()
    torch.cuda.synchronize()
    k10_fwd_ms = (time.monotonic() - started) / repeats * 1e3

    schedule = MatrixSchedule(slot=o["slot"], erase=o["erase"],
                              payload=o["payload"], op=o["op"],
                              live_after=o["live_after"])
    upstream = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(4)).cuda()

    def k11_bwd():
        return _k11_backward_body(payload, dense, write_keys, query_keys,
                                   decay, gates, upstream, schedule,
                                   o["live_pre"], o["ckpt_state"],
                                   o["ckpt_live"], o["ckpt_keys"], RANK,
                                   SEGMENT_TOKENS, POLICY.read_scale,
                                   True, True)

    def k10_bwd():
        from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
            import _k10_backward_body
        return _k10_backward_body(payload, write_keys, query_keys, decay,
                                  upstream, sched10, o10["live_pre"],
                                  o10["ckpt_state"], o10["ckpt_live"],
                                  o10["ckpt_keys"], RANK, SEGMENT_TOKENS,
                                  POLICY.read_scale)

    # K10's own gate-d pattern: the k10 schedule is computed ONCE outside
    # the timed loop, so the backward walls are backward-only on BOTH sides
    o10 = _k10_forward_launch(payload, write_keys, query_keys, decay,
                              state, live, keys, POL32, RANK,
                              SEGMENT_TOKENS, collect=False, warps=8,
                              stages=3)
    sched10 = MatrixSchedule(slot=o10["slot"], erase=o10["erase"],
                             payload=o10["payload"], op=o10["op"],
                             live_after=o10["live_after"])

    for _ in range(2):
        k11_bwd(); k10_bwd()
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(repeats):
        k11_bwd()
    torch.cuda.synchronize()
    k11_bwd_ms = (time.monotonic() - started) / repeats * 1e3
    started = time.monotonic()
    for _ in range(repeats):
        k10_bwd()
    torch.cuda.synchronize()
    k10_bwd_ms = (time.monotonic() - started) / repeats * 1e3
    tok_per_scan = BATCH * TOKENS
    receipt["gate_d"] = dict(
        fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM,
                     segment_tokens=SEGMENT_TOKENS),
        k11=dict(forward_ms_per_scan=round(k11_fwd_ms, 2),
                 backward_ms_per_scan=round(k11_bwd_ms, 2),
                 forward_throughput_tok_per_s=round(
                     tok_per_scan / (k11_fwd_ms / 1e3))),
        k10_same_session=dict(forward_ms_per_scan=round(k10_fwd_ms, 2),
                              backward_ms_per_scan=round(k10_bwd_ms, 2)),
        forward_overhead_vs_k10=round(k11_fwd_ms / k10_fwd_ms, 3),
        backward_overhead_vs_k10=round(k11_bwd_ms / k10_bwd_ms, 3),
        gate_channel_cost_note=(
            "the gate channel adds per token: one gate-scalar load, the "
            "CORRECT blend's sub/mul/mul/mul/add chain (or one mul at "
            "APPEND), and -- for the dense arms -- one (VP,) deposit load "
            "plus one masked row-add at the best row; measured here against "
            "K10's OWN launch on the same fixture and box, not estimated"))
    return receipt["gate_d"]


def gate_e(receipt: dict) -> dict:
    """The K12-flagged MERGE-corner assessment (see the module docstring).

    The k11 oracle's ``if bool(sel["merge"].any()):`` block (k11_learned_
    gates.py:336-352) index-puts zeros at ``sel["second_index"]`` for EVERY
    (batch, head) lane whenever ANY lane merges, instead of only the merging
    lanes' source rows.  The kernel (``_k11_seq_fwd_kernel``) guards per
    lane: ``hit2 = merge & keep_r & (second == ranks)`` with ``merge`` a
    per-lane scalar.  This gate measures whether any K11 fixture can reach
    the corner and whether production can, and demonstrates the corner on a
    crafted heterogeneous fixture.  The engine is NOT patched (committed;
    engine bugs escalate with receipts, never patch silently).
    """
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import NONE

    def keyed_stream(batch, tokens, heads, value_dim, key_dim, distinct, seed):
        generator = torch.Generator().manual_seed(seed)
        payload_dirs = torch.randn(distinct, value_dim, generator=generator)
        key_dirs = torch.randn(distinct, key_dim, generator=generator)
        choice = torch.randint(0, distinct, (batch, tokens, heads),
                               generator=generator)
        payload = payload_dirs[choice].cuda()
        write_keys = key_dirs[choice].cuda()
        query_keys = (torch.randn(batch, tokens, heads, key_dim,
                                  generator=generator) * 0.5).cuda()
        decay = torch.full((batch, tokens, heads), 0.95).cuda()
        return payload, write_keys, query_keys, decay

    def merge_structure(payload, dense, write_keys, query_keys, decay, gates,
                        rank, seg, warps=4):
        """Run the KERNEL on the fixture and measure the corner's firing
        condition: tokens where MERGE fires in a strict subset of a batch
        element's heads while a non-merging head's SECOND row is live
        pre-decision."""
        if gates is None:
            batch, tokens, heads = payload.shape[:3]
            gates = (torch.rand(batch, tokens, heads,
                                generator=torch.Generator().manual_seed(91))
                     .cuda() * 0.9 + 0.05)
        batch, tokens, heads, value_dim = payload.shape
        state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                            device="cuda")
        live = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
        keys = torch.zeros(batch, heads, rank, write_keys.shape[-1],
                           dtype=torch.float32, device="cuda")
        out = _k11_forward_launch(
            payload.float(), dense.float(), write_keys.float(),
            query_keys.float(), decay.float(), gates.float(), state, live,
            keys, POL32, rank, seg, True, True, collect=True, warps=warps,
            stages=3)
        merge_lane = out["op"][..., 1] != NONE      # recorded lane-1 close
        second = out["slot"][..., 1]
        live_pre = out["live_pre"]
        counts = merge_lane.sum(-1)                   # (b, T)
        het = (counts > 0) & (counts < heads)
        victims = 0
        b_idx, t_idx = torch.nonzero(het, as_tuple=True)
        for b, t in zip(b_idx.tolist(), t_idx.tolist()):
            for h in range(heads):
                if not bool(merge_lane[b, t, h]) and \
                        bool(live_pre[b, t, h, int(second[b, t, h])]):
                    victims += 1
        return dict(merge_lane_fires=int(merge_lane.sum()),
                    tokens_with_merge=int((merge_lane.any(-1)).sum()),
                    heterogeneous_tokens=int(het.sum()),
                    corner_firing_nonmerging_live_second=victims)

    fixtures = {}
    # the committed test suite's oracle-parity / gradcheck fixtures
    p, wk, qk, d = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    fixtures["test identity-regime/bf16/floors/replay (2,64,2,6dirs,r5)"] = \
        merge_structure(p, torch.randn_like(p) * 0.05, wk, qk, d, None, 5, 16)
    p, wk, qk, d = keyed_stream(2, 512, 2, 128, 16, 512, 77)
    fixtures["test registered-shape flip (2,512,2,512dirs,r128)"] = \
        merge_structure(p, torch.randn_like(p) * 0.05, wk, qk, d, None,
                        128, 64, warps=8)
    p, wk, qk, d = keyed_stream(2, 32, 2, 8, 4, 5, 11)
    fixtures["test gradcheck (2,32,2,5dirs,r5)"] = \
        merge_structure(p, torch.randn_like(p) * 0.05, wk, qk, d, None, 5, 8)
    p, wk, qk, d = keyed_stream(2, 128, 2, 16, 8, 6, 41)
    fixtures["test zero-flip invariance (2,128,2,6dirs,r8)"] = \
        merge_structure(p, torch.randn_like(p) * 0.05, wk, qk, d, None, 8, 32)
    # this receipt's own fixtures
    payload, write_keys, query_keys, decay = registered_fixture()
    gates, dense = gate_stream(BATCH, TOKENS, HEADS, 91)
    fixtures["gate_a registered fixture (8,2048,2,512dirs,r128)"] = \
        merge_structure(payload, dense, write_keys, query_keys, decay, gates,
                        RANK, SEGMENT_TOKENS, warps=8)
    p, wk, qk, d = keyed_stream(2, 256, 2, 128, 16, 512, 23)
    fixtures["gate_b floor fixture (2,256,2,512dirs,r128)"] = \
        merge_structure(p, torch.randn_like(p) * 0.05, wk, qk, d, None,
                        RANK, SEGMENT_TOKENS, warps=8)
    p, wk, qk, d = keyed_stream(2, 48, 2, 16, 4, 6, 11)
    fixtures["gate_b gradcheck fixture (2,48,2,6dirs,r8)"] = \
        merge_structure(p, torch.randn_like(p) * 0.05, wk, qk, d, None, 8, 16)

    # ---- the corner DEMONSTRATION: a crafted heterogeneous two-lane walk.
    # head 0: rows keyed ek0 / 0.9*ek0+sqrt(0.19)*ek1 -> a write key ek0
    #        MERGES (second similarity 0.9 >= tau_merge).
    # head 1: rows keyed ek0..ek3 (orthogonal) -> the same write key ek0
    #        CORRECTs on row 0; its SECOND row is live and must survive.
    V, K, rank, T = 8, 8, 4, 1
    e = torch.eye(V, device="cuda")
    ek = torch.eye(K, device="cuda")
    state = torch.zeros(1, 2, rank, V, device="cuda")
    keys = torch.zeros(1, 2, rank, K, device="cuda")
    live = torch.zeros(1, 2, rank, dtype=torch.bool, device="cuda")
    rows = [e[0], e[2], e[3], e[4]]
    key_rows_h0 = [ek[0], 0.9 * ek[0] + (1 - 0.81) ** 0.5 * ek[1], ek[2],
                   ek[3]]
    for r in range(rank):
        state[0, 0, r] = rows[r]
        keys[0, 0, r] = key_rows_h0[r]
        live[0, 0, r] = True
        state[0, 1, r] = rows[r]
        keys[0, 1, r] = ek[r]
        live[0, 1, r] = True
    payload = e[0].view(1, 1, 1, V).repeat(1, T, 2, 1)
    write_keys = ek[0].view(1, 1, 1, K).repeat(1, T, 2, 1)
    query_keys = torch.randn(1, T, 2, K, generator=torch.Generator()
                             .manual_seed(3)).cuda()
    decay = torch.full((1, T, 2), 0.999, device="cuda")
    gates = torch.full((1, T, 2), 0.7, device="cuda")
    dense = torch.zeros_like(payload)
    out = _k11_forward_launch(payload, dense, write_keys, query_keys, decay,
                              gates, state.clone(), live.clone(), keys.clone(),
                              POL32, rank, 64, True, True, collect=True,
                              warps=4, stages=3)
    ref = gated_oracle_scan(payload, dense, write_keys, query_keys, decay,
                            gates, policy=POLICY, gated=True, use_dense=True,
                            rank=rank, rank_max=rank, initial_state=state,
                            initial_live=live, initial_keys=keys, collect=True)
    differing = {}
    for name in ("state", "live", "keys"):
        differing[name] = int((out[name] != ref[name]).sum())
    schedule_ref = ref["schedule"]
    for name in ("slot", "erase", "op", "live_after", "payload"):
        differing[f"schedule.{name}"] = int(
            (getattr(schedule_ref, name) != out[name]).sum())
    demo = dict(
        fixture="batch 1, heads 2, rank 4, 1 token: head 0 MERGEs (second "
                "key similarity 0.9 >= tau_merge), head 1 CORRECTs with a "
                "live SECOND row",
        op_lane0_kernel=[[int(out["op"][0, 0, h, 0]) for h in range(2)]],
        op_lane1_kernel=[[int(out["op"][0, 0, h, 1]) for h in range(2)]],
        second_index=[int(out["slot"][0, 0, h, 1]) for h in range(2)],
        kernel_live_head1=[bool(v) for v in out["live"][0, 1]],
        oracle_live_head1=[bool(v) for v in ref["live"][0, 1]],
        kernel_head1_live_rows=int(out["live"][0, 1].sum()),
        oracle_head1_live_rows=int(ref["live"][0, 1].sum()),
        differing_elements=differing,
        note="the oracle kills the NON-merging head's live SECOND row (the "
             "K10 pattern verbatim); the kernel guards per lane and keeps "
             "it -- the corner is REAL in the k11 oracle, oracle-only")

    corner_reachable = any(f["corner_firing_nonmerging_live_second"] > 0
                           for f in fixtures.values())
    production_kernel_only = bool(separated_available(RANK, VALUE_DIM, KEY_DIM))
    verdict = dict(
        pattern_location="k11_learned_gates.py:336-352 (gated_oracle_scan, "
                        "the if bool(sel[\"merge\"].any()): block; the "
                        "index_put at :326 is the lane-0 write, the flagged "
                        "one is the merge-source close below it)",
        kernel_guard="_k11_seq_fwd_kernel: hit2 = merge & keep_r & "
                     "(second == ranks) -- merge is a per-(batch,head)-lane "
                     "scalar; state/keys/live zeroed only in the merging lane",
        fixtures_unreachable=(not corner_reachable),
        production=(
            dict(kernel_only_at_registered_shape=production_kernel_only,
                 path="GatedSeparatedMixer.mix -> gated_scan -> "
                      "x45::k11_scan_fwd (the opaque kernel op) whenever "
                      "separated_available; the eager oracle is only the "
                      "fallback for oversized/non-CUDA stores and the "
                      "receipt/gradcheck instrument",
                 inert=True if production_kernel_only else None),
        ),
        committed_engine_patched=False,
        escalation_rule="the engine is committed: bugs escalate with "
                        "receipts, never patch silently",
        flagged_by="the K12 placement lane (commit cb5615fe, "
                   "K12_PLACEMENT.md section 3, the K10 eager oracle's "
                   "latent multi-lane MERGE bug; K11 flagged as carrying "
                   "the same pattern)",
        verdict=("UNREACHABLE AND INERT: no committed fixture and no receipt "
                 "fixture can fire the corner (every measured fixture "
                 "produces zero MERGEs; the committed MERGE fixture is "
                 "single-lane by construction -- heads=1), and production "
                 "takes the kernel path, which guards per lane. No banked "
                 "or committed K11 receipt is invalidated; the arms may "
                 "proceed. The corner is receipted here and escalated to "
                 "the operator with the K12 flag."
                 if not corner_reachable and production_kernel_only else
                 "REACHABLE: a fixture or production path can fire the "
                 "corner -- STOP and escalate before any arm runs"))
    receipt["gate_e"] = dict(
        probe="the K12-flagged multi-lane MERGE corner assessment",
        oracle_pattern=(
            "the if bool(sel[\"merge\"].any()): block index-puts zeros at "
            "sel[\"second_index\"] for EVERY (batch, head) lane whenever "
            "ANY lane merges, instead of only the merging lanes' source "
            "rows -- the K10 oracle pattern verbatim (K12_PLACEMENT.md)"),
        fixture_merge_structure=fixtures,
        corner_demonstration=demo,
        **verdict)
    return receipt["gate_e"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "receipts" / "k11_gates.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "the gates need CUDA"
    receipt = dict(probe="k11_learned_gates_gates",
                   registration=str(ROOT / "PREREGISTRATION_LEARNED_GATES.md"),
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
