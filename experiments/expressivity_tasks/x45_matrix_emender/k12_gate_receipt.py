"""K12 -- the gate receipts: gates A, B and D at the registered shape, plus the
placement channel's own receipts.

Writes ``receipts/k12_gates.json`` (the K10 ``k10_gate_receipt.py`` pattern):

* **Gate A** -- the parity lanes at the registered shape (batch 8, 2048
  tokens, 2 heads, rank 128, value_dim 128, key_dim 16, bf16 inputs): the
  schedule fields (the PLACED slot at APPEND included), the payload lane, the
  read WEIGHTS, the pre-decision liveness, the exit state/liveness, the exit
  key table AND the exit residue table bit-equal against the eager per-token
  oracle -- at the birth weight w=0 AND at w=0.5 (placement firing); the
  read at the measured fp32-vs-fp64 floor.  The BIRTH-PARITY lane: the k12
  launch at w=+0.0 vs K10's OWN launch, bit-equal on every K10 channel, the
  flip count receipted.  The PLACEMENT lane: the fire rate (APPENDs whose
  slot differs from K10's first-free rule) and the collect receipt channels.
* **Gate B** -- the six K10 gradients through the k12 backward (K10's own
  body, imported) vs the placement oracle's autograd and the fp32-vs-fp64
  floor; the ESTIMATOR receipt: the kernel path's dL/dw vs its DEFINED torch
  reference on the same fixture (the reference replays the residue/key tables
  from the recorded channels, g_t from the oracle's own value adjoint), and
  the ZERO-GRADIENT FINDING: the oracle path's structural gradient to w is
  exactly zero on the same fixture.
* **Gate D** -- block-call walls (forward/backward per (layer, window) scan
  at the registered shape), the register/spill count of the forward kernel,
  and the honest placement-channel cost statement against K10's banked walls
  (the K10 receipt's own numbers, cited).

Usage:

    eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k12_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k12_gates.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, MERGE, POP, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    _k10_forward_launch, separated_sequential_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k12_placement import (
    _k12_backward_body, _k12_forward_launch, placement_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _norm_defined, _pow2, _round_f32, tree_fold)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import EPS, RoutingPolicy

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
# The registered shape (the K10 gate receipt's own fixture, the residue channel
# added at the same key_dim 16).
BATCH, TOKENS, HEADS, RANK, VALUE_DIM, KEY_DIM = 8, 2048, 2, 128, 128, 16
SEGMENT_TOKENS = 64


def registered_fixture(seed: int = 77, batch: int = BATCH, tokens: int = TOKENS,
                       dtype: torch.dtype = torch.bfloat16, distinct: int = 512):
    """The K10 gate receipt's fixture: payload AND write-key directions from a
    hot set of ``distinct`` fixed directions (the identity-cache regime)."""
    generator = torch.Generator().manual_seed(seed)
    payload_dirs = torch.randn(distinct, VALUE_DIM, generator=generator)
    key_dirs = torch.randn(distinct, KEY_DIM, generator=generator)
    choice = torch.randint(0, distinct, (batch, tokens, HEADS), generator=generator)
    payload = payload_dirs[choice].to(dtype=dtype).cuda()
    write_keys = key_dirs[choice].to(dtype=dtype).cuda()
    query_keys = (torch.randn(batch, tokens, HEADS, KEY_DIM, generator=generator)
                  .to(dtype=dtype).cuda())
    decay = torch.full((batch, tokens, HEADS), 0.95).to(dtype=dtype).cuda()
    return payload, write_keys, query_keys, decay


def placement_w(value: float) -> torch.Tensor:
    return torch.full((HEADS,), float(value), dtype=torch.float32,
                      device="cuda")


def placement_regime_fixture(seed: int = 41, batch: int = BATCH,
                            tokens: int = TOKENS,
                            dtype: torch.dtype = torch.bfloat16):
    """The placement lane's own fixture: the store seeded FULL with 8
    MERGEABLE KEY PAIRS (rows at cos 0.9, so matched keys MERGE and the
    source row dies) beside unique filler rows, and a token stream mixing
    pair keys (deaths) with fresh unique keys (unmatched: POP when the
    store is full, APPEND into the freed slots when it is not).  Only a
    POP-then-MERGE window leaves TWO free slots at once -- the structural
    condition for placement to differ from the first-free rule at all, and
    the honest statement of how rare that window is."""
    generator = torch.Generator().manual_seed(seed)
    n_pairs = 8
    # the pairs: A_i and B_i at cos 0.9 (>= tau_merge 0.85)
    A = torch.randn(n_pairs, KEY_DIM, generator=generator)
    A = A / A.norm(dim=-1, keepdim=True)
    U = torch.randn(n_pairs, KEY_DIM, generator=generator)
    U = U - (U * A).sum(-1, keepdim=True) * A
    U = U / U.norm(dim=-1, keepdim=True)
    B = 0.9 * A + math.sqrt(1.0 - 0.81) * U
    filler = torch.randn(RANK - 2 * n_pairs, KEY_DIM, generator=generator)
    filler = filler / filler.norm(dim=-1, keepdim=True)
    seed_rows = torch.cat([A, B, filler])            # (RANK, KEY_DIM)
    # the token stream: 1/6 pair-A keys, 1/6 pair-B keys, 2/3 fresh keys
    fresh = torch.randn(batch, tokens, HEADS, KEY_DIM, generator=generator)
    fresh = fresh / fresh.norm(dim=-1, keepdim=True)
    pair_choice = torch.randint(0, n_pairs, (batch, tokens, HEADS),
                                generator=generator)
    which = torch.rand(batch, tokens, HEADS, generator=generator)
    write_keys = fresh.clone()
    a_mask = which < (1.0 / 6.0)
    b_mask = (which >= (1.0 / 6.0)) & (which < (2.0 / 6.0))
    write_keys[a_mask] = A[pair_choice[a_mask]].to(write_keys.dtype)
    write_keys[b_mask] = B[pair_choice[b_mask]].to(write_keys.dtype)
    payload_dirs = torch.randn(256, VALUE_DIM, generator=generator)
    payload_choice = torch.randint(0, 256, (batch, tokens, HEADS),
                                   generator=generator)
    payload = payload_dirs[payload_choice.cpu()].to(dtype=dtype).cuda()
    query_keys = (torch.randn(batch, tokens, HEADS, KEY_DIM,
                              generator=generator).to(dtype=dtype).cuda())
    decay = torch.full((batch, tokens, HEADS), 0.95).to(dtype=dtype).cuda()
    seed_state = torch.randn(batch, HEADS, RANK, VALUE_DIM,
                             generator=generator).cuda()
    return (payload, write_keys.to(dtype).cuda(), query_keys, decay,
            seed_rows, seed_state)


def _parity_rows(ref, out, extra_residue: bool):
    rows = {}
    flips = 0
    lanes = 0
    for name in ("slot", "erase", "op", "live_after"):
        a, b = getattr(ref["schedule"], name), out[name]
        flips += int((a != b).sum())
        lanes += a.numel()
        rows[f"schedule_{name}"] = dict(lanes=a.numel(),
                                        flips=int((a != b).sum()),
                                        bit_equal=bool(torch.equal(a, b)))
    for name in ("weights", "live_pre", "state", "live", "keys"):
        a, b = ref[name], out[name]
        flips += int((a != b).sum())
        lanes += a.numel()
        rows[name] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                          bit_equal=bool(torch.equal(a, b)))
    if extra_residue:
        a, b = ref["residue"], out["residue"]
        flips += int((a != b).sum())
        lanes += a.numel()
        rows["residue"] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                               bit_equal=bool(torch.equal(a, b)))
    a, b = ref["schedule"].payload, out["payload"]
    flips += int((a != b).sum())
    lanes += a.numel()
    rows["payload"] = dict(lanes=a.numel(), flips=int((a != b).sum()),
                           bit_equal=bool(torch.equal(a, b)))
    return rows, flips, lanes


def gate_a(receipt: dict) -> dict:
    payload, write_keys, query_keys, decay = registered_fixture()
    state = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool, device="cuda")
    keys = torch.zeros(BATCH, HEADS, RANK, KEY_DIM, dtype=torch.float32,
                       device="cuda")
    residue = torch.zeros(BATCH, HEADS, RANK, KEY_DIM, dtype=torch.float32,
                          device="cuda")
    gate = dict(fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS,
                             rank=RANK, value_dim=VALUE_DIM, key_dim=KEY_DIM,
                             dtype="bf16 inputs, fp32 defined-order math",
                             segment_tokens=SEGMENT_TOKENS,
                             distinct_directions=512))
    # ---- the parity lanes at the birth weight and at a firing weight
    for w_value in (0.0, 0.5):
        w = placement_w(w_value)
        started = time.monotonic()
        out = _k12_forward_launch(payload, write_keys, query_keys, decay, w,
                                  state, live, keys, residue, POL32, RANK,
                                  SEGMENT_TOKENS, collect=True, warps=8,
                                  stages=3)
        torch.cuda.synchronize()
        kernel_wall = time.monotonic() - started
        started = time.monotonic()
        ref = placement_oracle_scan(payload, write_keys, query_keys, decay, w,
                                    policy=POLICY, rank=RANK, rank_max=RANK,
                                    collect=True)
        oracle_wall = time.monotonic() - started
        rows, flips, lanes = _parity_rows(ref, out, extra_residue=True)
        with torch.no_grad():
            ref64 = placement_oracle_scan(payload, write_keys, query_keys,
                                           decay, w, policy=POLICY, rank=RANK,
                                           rank_max=RANK)
        floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
        deviation = (ref["reads"] - out["reads"]).abs().max().item()
        is_append = out["op"][..., 0] == APPEND
        first_free = (~out["live_pre"]).to(torch.int64).argmax(-1)
        fired = (out["slot"][..., 0] != first_free) & is_append
        hist = op_histogram(MatrixSchedule(
            slot=out["slot"], erase=out["erase"], payload=out["payload"],
            op=out["op"], live_after=out["live_after"]))
        gate[f"w={w_value}"] = dict(
            channels=rows,
            total=dict(lanes=lanes, flips=flips),
            read=dict(floor_fp32_vs_fp64=floor, deviation=deviation,
                      ratio_vs_floor=deviation / max(floor, 1e-30),
                      within_noise_factor=bool(
                          deviation <= NOISE_FACTOR * max(floor, 1e-30))),
            placement=dict(
                append_tokens=int(is_append.sum()),
                fired_vs_first_free=int(fired.sum()),
                fire_rate=(float(fired.sum()) / max(1, int(is_append.sum()))),
                chosen_residue_sim_mean=float(out["place_sim"][is_append].mean())
                if bool(is_append.any()) else None,
                free_mean_residue_sim_mean=float(
                    out["place_free_mean"][is_append].mean())
                if bool(is_append.any()) else None),
            op_histogram=hist,
            walls=dict(kernel_forward_s=round(kernel_wall, 3),
                       eager_oracle_s=round(oracle_wall, 1)))
        try:
            meta = out["kernel"]
            gate[f"w={w_value}"]["forward_kernel_resources"] = dict(
                num_warps=8,
                registers=getattr(meta, "n_regs", None),
                spills=getattr(meta, "n_spills", None))
        except Exception:
            pass
        if w_value == 0.0:
            # ---- the BIRTH-PARITY lane: K10's own launch, bit-equal at w=0
            k10 = _k10_forward_launch(payload, write_keys, query_keys, decay,
                                       state, live, keys, POL32, RANK,
                                       SEGMENT_TOKENS, collect=True, warps=8,
                                       stages=3)
            birth_rows = {}
            birth_flips = 0
            birth_lanes = 0
            for name in ("slot", "erase", "op", "live_after"):
                a, b = k10[name], out[name]
                birth_flips += int((a != b).sum())
                birth_lanes += a.numel()
                birth_rows[f"schedule_{name}"] = dict(
                    lanes=a.numel(), flips=int((a != b).sum()),
                    bit_equal=bool(torch.equal(a, b)))
            for name in ("weights", "live_pre", "state", "live", "keys",
                         "reads"):
                a, b = k10[name], out[name]
                birth_flips += int((a != b).sum())
                birth_lanes += a.numel()
                birth_rows[name] = dict(
                    lanes=a.numel(), flips=int((a != b).sum()),
                    bit_equal=bool(torch.equal(a, b)))
            a, b = k10["payload"], out["payload"]
            birth_flips += int((a != b).sum())
            birth_lanes += a.numel()
            birth_rows["payload"] = dict(
                lanes=a.numel(), flips=int((a != b).sum()),
                bit_equal=bool(torch.equal(a, b)))
            gate["birth_parity_vs_k10_launch"] = dict(
                note="the k12 launch at w=+0.0 vs K10's own launch on every "
                     "K10 channel: the zero-born argmax tie-break reproduces "
                     "the first-free rule bit-exactly (P5)",
                channels=birth_rows,
                total=dict(lanes=birth_lanes, flips=birth_flips))
    # ---- the PLACEMENT-REGIME lane: the same registered shape with the
    # store seeded FULL (128 live rows of distinct keys, the trained machine's
    # hot-set regime): unmatched keys POP the worst row, the residue channel
    # fills at every death, and later unmatched keys APPEND into
    # residue-bearing free slots -- the regime where placement can fire.
    payload, write_keys, query_keys, decay, seed_rows, seed_state = \
        placement_regime_fixture()
    full_keys = seed_rows.view(1, 1, RANK, KEY_DIM).expand(
        BATCH, HEADS, RANK, KEY_DIM).contiguous().cuda()
    full_state = seed_state
    full_live = torch.ones(BATCH, HEADS, RANK, dtype=torch.bool,
                           device="cuda")
    w = placement_w(0.5)
    out = _k12_forward_launch(payload, write_keys, query_keys, decay, w,
                              full_state, full_live, full_keys, residue,
                              POL32, RANK, SEGMENT_TOKENS, collect=True,
                              warps=8, stages=3)
    ref = placement_oracle_scan(payload, write_keys, query_keys, decay, w,
                                policy=POLICY, rank=RANK, rank_max=RANK,
                                initial_state=full_state,
                                initial_live=full_live,
                                initial_keys=full_keys, collect=True)
    rows, flips, lanes = _parity_rows(ref, out, extra_residue=True)
    is_append = out["op"][..., 0] == APPEND
    first_free = (~out["live_pre"]).to(torch.int64).argmax(-1)
    fired = (out["slot"][..., 0] != first_free) & is_append
    nonzero_residues = int((out["residue"].norm(dim=-1) > 0).sum())
    gate["placement_regime_w=0.5"] = dict(
        note="the registered shape with the store seeded FULL (128 live rows "
             "of distinct keys): unmatched keys POP the worst row, the "
             "residue channel fills at every death, later unmatched keys "
             "APPEND into residue-bearing free slots -- the fire rate "
             "measures a regime where placement can differ from first-free",
        fixture=dict(store="full at entry, 8 mergeable key pairs at cos 0.9 "
                     "beside unique filler rows",
                     token_stream="1/6 pair-A + 1/6 pair-B + 2/3 fresh keys",
                     structural_note="only a POP-then-MERGE window leaves two "
                                    "free slots at once -- placement can "
                                    "differ from first-free only there"),
        channels=rows, total=dict(lanes=lanes, flips=flips),
        placement=dict(
            append_tokens=int(is_append.sum()),
            fired_vs_first_free=int(fired.sum()),
            fire_rate=float(fired.sum()) / max(1, int(is_append.sum())),
            residue_slots_nonzero=nonzero_residues,
            chosen_residue_sim_mean=float(out["place_sim"][is_append].mean())
            if bool(is_append.any()) else None,
            free_mean_residue_sim_mean=float(
                out["place_free_mean"][is_append].mean())
            if bool(is_append.any()) else None))
    receipt["gate_a"] = gate
    return gate


def _estimator_reference(out, payload, write_keys, w, g_chan, keys0, residue0):
    """The estimator's DEFINED torch reference, replayed from the recorded
    channels (the same construction the tests gate): returns dL/dw (heads,)."""
    batch, tokens, heads = g_chan.shape
    KD = keys0.shape[-1]
    KP = _pow2(KD)
    op0_all = out["op"][..., 0]
    slot0_all = out["slot"][..., 0]
    merge_all = out["op"][..., 1] == MERGE
    dw = torch.zeros(heads, device="cuda")
    kacc, racc = keys0.clone(), residue0.clone()
    for t in range(tokens):
        live = out["live_pre"][:, t]
        free = ~live
        wk = write_keys[:, t].detach()
        if bool((op0_all[:, t] == APPEND).any()):
            nwk = _norm_defined(wk, -1, KP)
            uk = wk / nwk.clamp_min(EPS).unsqueeze(-1)
            nres = _norm_defined(racc, -1, KP)
            simres = tree_fold(uk.unsqueeze(2) * racc, -1, KP) \
                / nres.clamp_min(EPS)
            s = w.view(1, -1, 1) * simres + 0.0
            s = torch.where(free, s, torch.full_like(s, float("-inf")))
            p = torch.softmax(s, dim=-1)
            slot0 = slot0_all[:, t]
            sim_c = simres.gather(-1, slot0.unsqueeze(-1)).squeeze(-1)
            p_c = p.gather(-1, slot0.unsqueeze(-1)).squeeze(-1)
            p_mean = (p * simres).sum(-1)
            term = g_chan[:, t] * p_c * (sim_c - p_mean)
            is_app = (op0_all[:, t] == APPEND).to(term.dtype)
            dw += (term * is_app).sum(0)
        for b in range(batch):
            for h in range(heads):
                o = int(op0_all[b, t, h])
                sl = int(slot0_all[b, t, h])
                second = int(out["slot"][b, t, h, 1])
                if o == POP:
                    racc[b, h, sl] = kacc[b, h, sl]
                if bool(merge_all[b, t, h]):
                    racc[b, h, second] = kacc[b, h, second]
                if o != 0:
                    kacc[b, h, sl] = (torch.zeros(KD, device="cuda")
                                      if o == POP else wk[b, h].detach())
                if bool(merge_all[b, t, h]):
                    kacc[b, h, second] = 0.0
    return dw


def gate_b(receipt: dict) -> dict:
    # the gate-class shape (K10's own choice: fp32 inputs, the floor
    # instrument's regime)
    payload32, write_keys32, query_keys32, decay32 = registered_fixture(
        seed=23, batch=2, tokens=256, dtype=torch.float32)
    rank, seg = 128, SEGMENT_TOKENS
    w = placement_w(0.5)
    upstream = torch.randn(2, 256, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(1234)).cuda()
    state = torch.zeros(2, HEADS, rank, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(2, HEADS, rank, dtype=torch.bool, device="cuda")
    keys = torch.zeros(2, HEADS, rank, KEY_DIM, dtype=torch.float32,
                      device="cuda")
    residue = torch.zeros_like(keys)
    out = _k12_forward_launch(payload32, write_keys32, query_keys32, decay32,
                              w, state, live, keys, residue, POL32, rank, seg,
                              collect=True, warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])

    # the placement oracle's autograd grads (fp32) and its fp64 twin (floor)
    def oracle_grads(dtype):
        p_ = payload32.clone().requires_grad_(True)
        wk_ = write_keys32.clone().requires_grad_(True)
        qk_ = query_keys32.clone().requires_grad_(True)
        d_ = decay32.clone().requires_grad_(True)
        s_ = torch.zeros(2, HEADS, rank, VALUE_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        k_ = torch.zeros(2, HEADS, rank, KEY_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        ref = placement_oracle_scan(p_.to(dtype), wk_.to(dtype),
                                     qk_.to(dtype), d_.to(dtype), w,
                                     policy=POLICY, rank=rank, rank_max=rank,
                                     initial_state=s_, initial_keys=k_)
        (ref["reads"] * upstream.to(dtype)).sum().backward()
        return ref, (p_.grad, wk_.grad, qk_.grad, d_.grad, s_.grad, k_.grad)

    ref32, grads32 = oracle_grads(torch.float32)
    _ref64, grads64 = oracle_grads(torch.float64)

    # the ZERO-GRADIENT FINDING on the same fixture: the oracle path's
    # structural gradient to w is exactly zero (the discrete choice)
    w_finding = placement_w(0.5).requires_grad_(True)
    p_f = payload32.clone().requires_grad_(True)
    ref_f = placement_oracle_scan(p_f, write_keys32, query_keys32, decay32,
                                   w_finding, policy=POLICY, rank=rank,
                                   rank_max=rank)
    finding = torch.autograd.grad((ref_f["reads"] * upstream).sum(),
                                  [w_finding, p_f], allow_unused=True)

    # the kernel path's six gradients + the estimator, via the k12 backward
    grads = _k12_backward_body(payload32, write_keys32, query_keys32, decay32,
                               w, upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], state, residue, rank, seg,
                               POLICY.read_scale)
    six_kernel = (grads[0], grads[1], grads[2], grads[3], grads[5], grads[6])
    rows = {}
    for name, kern, orac, orac64 in zip(
            ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dstate",
             "dkeys"), six_kernel, grads32, grads64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        rows[name] = dict(
            floor_fp32_vs_fp64=floor, deviation=deviation,
            ratio_vs_floor=deviation / max(floor, 1e-30),
            within_noise_factor=bool(deviation
                                     <= NOISE_FACTOR * max(floor, 1e-30)),
            bit_equal=bool(torch.equal(kern.float(), orac.float())))
    # the estimator vs its defined reference (g from the oracle's own adjoint)
    g_chan = (grads32[0] * payload32).sum(-1)
    want = _estimator_reference(out, payload32, write_keys32, w, g_chan,
                                keys, residue)
    got = grads[4]
    gate = dict(
        fixture=dict(batch=2, tokens=256, heads=HEADS, rank=rank,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM, dtype="fp32",
                     segment_tokens=seg),
        six_gradients=rows,
        zero_gradient_finding=dict(
            note="the ORACLE path's structural gradient to the placement "
                 "weight is exactly zero under P2+P3 (the discrete slot "
                 "choice is a backward constant) -- the finding; the kernel "
                 "path's straight-through estimator is the admitted "
                 "exception",
            oracle_dw_is_none=finding[0] is None,
            oracle_dw=torch.zeros(HEADS, device="cuda").tolist()
            if finding[0] is None else finding[0].tolist()),
        estimator=dict(
            note="the admitted straight-through surrogate vs its DEFINED "
                 "torch reference (replayed from the recorded channels, "
                 "g_t from the oracle's own value adjoint); a defined "
                 "estimator, not the true adjoint -- its bias is the "
                 "registration's disclosure",
            kernel_dw=got.tolist(),
            reference_dw=want.tolist(),
            max_abs_diff=float((got - want).abs().max()),
            nonzero=bool(float(got.abs().max()) > 0.0)),
        op_histogram=op_histogram(schedule))
    receipt["gate_b"] = gate
    return gate


def gate_d(receipt: dict) -> dict:
    """The honest cost statement: the k12 scan's forward/backward walls at the
    registered shape vs K10's own launch on the same fixture, same card."""
    payload, write_keys, query_keys, decay = registered_fixture(
        seed=91, batch=2, tokens=2048, dtype=torch.float32)
    state = torch.zeros(2, HEADS, RANK, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(2, HEADS, RANK, dtype=torch.bool, device="cuda")
    keys = torch.zeros(2, HEADS, RANK, KEY_DIM, dtype=torch.float32,
                       device="cuda")
    residue = torch.zeros_like(keys)
    walls = {}
    for name, launch in (("k10", None), ("k12", None)):
        # warmup then timed
        for _ in range(3):
            if name == "k10":
                _k10_forward_launch(payload, write_keys, query_keys, decay,
                                    state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, collect=False, warps=8,
                                    stages=3)
            else:
                _k12_forward_launch(payload, write_keys, query_keys, decay,
                                    placement_w(0.5), state, live, keys,
                                    residue, POL32, RANK, SEGMENT_TOKENS,
                                    collect=False, warps=8, stages=3)
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(10):
            if name == "k10":
                _k10_forward_launch(payload, write_keys, query_keys, decay,
                                    state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, collect=False, warps=8,
                                    stages=3)
            else:
                _k12_forward_launch(payload, write_keys, query_keys, decay,
                                    placement_w(0.5), state, live, keys,
                                    residue, POL32, RANK, SEGMENT_TOKENS,
                                    collect=False, warps=8, stages=3)
        torch.cuda.synchronize()
        walls[name] = round((time.monotonic() - started) / 10, 3)
    # the backward wall (the k12 backward = K10's body + the estimator)
    payload32, write_keys32, query_keys32, decay32 = registered_fixture(
        seed=91, batch=2, tokens=2048, dtype=torch.float32)
    out = _k12_forward_launch(payload32, write_keys32, query_keys32, decay32,
                              placement_w(0.5), state, live, keys, residue,
                              POL32, RANK, SEGMENT_TOKENS, collect=False,
                              warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(2, 2048, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(5)).cuda()
    _k12_backward_body(payload32, write_keys32, query_keys32, decay32,
                       placement_w(0.5), upstream, schedule, out["live_pre"],
                       out["ckpt_state"], out["ckpt_live"], out["ckpt_keys"],
                       state, residue, RANK, SEGMENT_TOKENS,
                       POLICY.read_scale)
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(5):
        _k12_backward_body(payload32, write_keys32, query_keys32, decay32,
                           placement_w(0.5), upstream, schedule,
                           out["live_pre"], out["ckpt_state"],
                           out["ckpt_live"], out["ckpt_keys"], state, residue,
                           RANK, SEGMENT_TOKENS, POLICY.read_scale)
    torch.cuda.synchronize()
    backward_wall = round((time.monotonic() - started) / 5, 3)
    receipt["gate_d"] = dict(
        note="the placement channel's honest price at the registered shape: "
             "the forward carries the residue tile and the placement scores "
             "(a scalar if appendable guard: ~2% of tokens); the backward is "
             "K10's own body plus the estimator's whole-window replay",
        forward_ms_per_scan=dict(k10=walls["k10"] * 1e3,
                                 k12=walls["k12"] * 1e3,
                                 ratio=walls["k12"] / max(walls["k10"], 1e-9)),
        backward_ms_per_scan=dict(k12=backward_wall * 1e3),
        k10_banked_walls=dict(forward_ms=19.7, backward_ms=17.2,
                              note="K10_SEPARATED_KEYS.md gate D, cited"))
    return receipt["gate_d"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        ROOT / "receipts" / "k12_gates.json"))
    args = ap.parse_args()
    torch.manual_seed(0)
    receipt = dict(probe="k12_placement",
                   registration=str(ROOT / "PREREGISTRATION_PLACEMENT.md"),
                   policy=dict(tau_match=POLICY.tau_match,
                               tau_merge=POLICY.tau_merge,
                               tau_pop=POLICY.tau_pop,
                               tie_break=POLICY.tie_break,
                               read_scale=POLICY.read_scale),
                   registered_shape=dict(batch=BATCH, tokens=TOKENS,
                                         heads=HEADS, rank=RANK,
                                         value_dim=VALUE_DIM,
                                         key_dim=KEY_DIM,
                                         segment_tokens=SEGMENT_TOKENS),
                   noise_factor=NOISE_FACTOR)
    gate_a(receipt)
    gate_b(receipt)
    gate_d(receipt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(receipt, f, indent=1)
    flips_a = receipt["gate_a"]["w=0.0"]["total"]["flips"] \
        + receipt["gate_a"]["w=0.5"]["total"]["flips"]
    birth_flips = receipt["gate_a"]["birth_parity_vs_k10_launch"]["total"]["flips"]
    print(json.dumps(dict(
        gate_a_flips=flips_a,
        birth_parity_flips=birth_flips,
        fire_rate_w0p5=receipt["gate_a"]["placement_regime_w=0.5"]["placement"]["fire_rate"],
        estimator_diff=receipt["gate_b"]["estimator"]["max_abs_diff"],
        finding_dw_none=receipt["gate_b"]["zero_gradient_finding"]["oracle_dw_is_none"],
        forward_ms=dict(k10=receipt["gate_d"]["forward_ms_per_scan"]["k10"],
                        k12=receipt["gate_d"]["forward_ms_per_scan"]["k12"]),
    ), indent=1))


if __name__ == "__main__":
    main()
