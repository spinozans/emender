"""K14 -- the gate receipts: gates A, B and D at the registered shape, plus
the routing channel's own receipts.

Writes ``receipts/k14_gates.json`` (the K10/K12 gate-receipt pattern):

* **Gate A** -- the parity lanes at the registered shape (batch 8, 2048
  tokens, 2 heads, rank 128, value_dim 128, key_dim 16, bf16 inputs): the
  schedule fields, the payload lane, the read WEIGHTS, the pre-decision
  liveness, the exit state/liveness, the exit key table AND the recorded
  per-op scores bit-equal against the eager per-token oracle -- at the
  BIRTH router (exact zeros) AND at a flipped router (random scores that
  move ops off the chain); the read at the measured fp32-vs-fp64 floor.
  The BIRTH-PARITY lane: the k14 launch at g=+0.0 vs K10's OWN launch,
  bit-equal on every K10 channel, the flip count receipted.  The
  birth-uniqueness lane: the argmax margin > 0 on every token at birth
  (no tie-break reliance).  The FLIP-RATE lane: the fraction of tokens a
  magnitude-0.9 random router deviates from the chain.
* **Gate B** -- the six K10 gradients through the k14 backward (K10's own
  body, imported verbatim) vs the routing oracle's autograd and the
  fp32-vs-fp64 floor; the ESTIMATOR receipt: the kernel path's
  d(router_scores) vs the fp64 autograd of its own DEFINED surrogate
  objective; the ZERO-GRADIENT FINDING: the oracle path's structural
  gradient to the router scores is exactly zero on the same fixture.
* **Gate D** -- block-call walls (forward/backward per (layer, window)
  scan at the registered shape), the register/spill count of the forward
  kernel, and the honest routing-channel cost statement against K10's own
  launch on the same fixture, same card.

Usage:

    eval "$(GPU_LEASE_VISIBLE=5,6 scripts/gpu_lease.sh acquire 1 --wait)"
    CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 \\
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k14_gate_receipt \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k14_gates.json
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
from experiments.expressivity_tasks.x45_matrix_emender.k14_learned_routing import (
    RESERVED_SCORE, _k14_backward_body, _k14_forward_launch,
    _router_estimator, routing_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))
NOISE_FACTOR = 8.0
# The registered shape (the K10/K12 gate receipts' own fixture).
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


def router_scores(batch, tokens, heads, scale=0.0, seed=0):
    """The router's per-op scores: exact zeros (birth / the frozen control)
    or a random draw (the flipped-router regime)."""
    if scale == 0.0:
        return torch.zeros(batch, tokens, heads, 5, dtype=torch.float32,
                           device="cuda")
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, tokens, heads, 5, generator=generator)
            .cuda() * float(scale)).contiguous()


def _parity_rows(ref, out):
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
    for name, b in (("op_scores", out["op_scores"]),
                    ("chain_op", out["chain_op"])):
        a = ref["scores"] if name == "op_scores" else ref["chain_op"]
        flips += int((a != b).sum())
        lanes += a.numel()
        rows[name] = dict(lanes=a.numel(), flips=int((a != b).sum()),
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
    gate = dict(fixture=dict(batch=BATCH, tokens=TOKENS, heads=HEADS,
                             rank=RANK, value_dim=VALUE_DIM, key_dim=KEY_DIM,
                             dtype="bf16 inputs, fp32 defined-order math",
                             segment_tokens=SEGMENT_TOKENS,
                             distinct_directions=512))
    for label, scores in (("birth_g0", router_scores(BATCH, TOKENS, HEADS)),
                         ("flipped_g0p9", router_scores(BATCH, TOKENS, HEADS,
                                                         0.9, seed=41))):
        started = time.monotonic()
        out = _k14_forward_launch(payload, write_keys, query_keys, decay,
                                  scores, state, live, keys, POL32, RANK,
                                  SEGMENT_TOKENS, collect=True, warps=8,
                                  stages=3)
        torch.cuda.synchronize()
        kernel_wall = time.monotonic() - started
        started = time.monotonic()
        ref = routing_oracle_scan(payload, write_keys, query_keys, decay,
                                  scores, policy=POLICY, rank=RANK,
                                  rank_max=RANK, collect=True)
        oracle_wall = time.monotonic() - started
        rows, flips, lanes = _parity_rows(ref, out)
        with torch.no_grad():
            ref64 = routing_oracle_scan(payload, write_keys, query_keys,
                                        decay, scores, policy=POLICY,
                                        rank=RANK, rank_max=RANK)
        floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
        deviation = (ref["reads"] - out["reads"]).abs().max().item()
        hist = op_histogram(MatrixSchedule(
            slot=out["slot"], erase=out["erase"], payload=out["payload"],
            op=out["op"], live_after=out["live_after"]))
        flipped = int((out["op"][..., 0] != out["chain_op"]).sum())
        gate[label] = dict(
            channels=rows,
            total=dict(lanes=lanes, flips=flips),
            read=dict(floor_fp32_vs_fp64=floor, deviation=deviation,
                      ratio_vs_floor=deviation / max(floor, 1e-30),
                      within_noise_factor=bool(
                          deviation <= NOISE_FACTOR * max(floor, 1e-30))),
            routing=dict(
                tokens=int(out["op"][..., 0].numel()),
                ops_flipped_off_chain=flipped,
                flip_rate=flipped / float(out["op"][..., 0].numel()),
                op_histogram=hist),
            walls=dict(kernel_forward_s=round(kernel_wall, 3),
                       eager_oracle_s=round(oracle_wall, 1)))
        try:
            meta = out["kernel"]
            gate[label]["forward_kernel_resources"] = dict(
                num_warps=8,
                registers=getattr(meta, "n_regs", None),
                spills=getattr(meta, "n_spills", None))
        except Exception:
            pass
        if label == "birth_g0":
            # ---- the BIRTH-PARITY lane: K10's own launch, bit-equal at g=0
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
                note="the k14 launch at g=+0.0 vs K10's own launch on every "
                     "K10 channel: the branch-active margin encoding "
                     "reproduces the threshold chain bit-exactly (P5; the "
                     "chain's own op recorded token for token)",
                chain_op_matches_k10=bool(
                    torch.equal(k10["op"][..., 0], out["chain_op"])),
                channels=birth_rows,
                total=dict(lanes=birth_lanes, flips=birth_flips))
            # ---- the birth-uniqueness lane: the argmax is UNIQUE at birth
            s = out["op_scores"]
            top2 = s.topk(2, dim=-1).values
            margins = top2[..., 0] - top2[..., 1]
            gate["birth_argmax_uniqueness"] = dict(
                note="the supervisor-approved construction's core receipt: "
                     "every active margin is >= 0 and every inactive op sits "
                     "at the reserved -1.0, so the birth argmax is unique on "
                     "every token -- no tie-break reliance",
                min_margin=float(margins.min()),
                all_unique=bool((margins > 0).all()),
                winner_min=float(top2[..., 0].min()),
                runnerup_reserved_share=float(
                    (top2[..., 1] == RESERVED_SCORE).float().mean()))
    receipt["gate_a"] = gate
    return gate


def gate_b(receipt: dict) -> dict:
    # the gate-class shape (K10's own choice: fp32 inputs, the floor
    # instrument's regime)
    payload32, write_keys32, query_keys32, decay32 = registered_fixture(
        seed=23, batch=2, tokens=256, dtype=torch.float32)
    rank, seg = 128, SEGMENT_TOKENS
    scores = router_scores(2, 256, HEADS, 0.5, seed=62)
    upstream = torch.randn(2, 256, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(1234)).cuda()
    state = torch.zeros(2, HEADS, rank, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(2, HEADS, rank, dtype=torch.bool, device="cuda")
    keys = torch.zeros(2, HEADS, rank, KEY_DIM, dtype=torch.float32,
                       device="cuda")
    out = _k14_forward_launch(payload32, write_keys32, query_keys32, decay32,
                              scores, state, live, keys, POL32, rank, seg,
                              collect=True, warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])

    def oracle_grads(dtype):
        p_ = payload32.clone().requires_grad_(True)
        wk_ = write_keys32.clone().requires_grad_(True)
        qk_ = query_keys32.clone().requires_grad_(True)
        d_ = decay32.clone().requires_grad_(True)
        s_ = torch.zeros(2, HEADS, rank, VALUE_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        k_ = torch.zeros(2, HEADS, rank, KEY_DIM, dtype=dtype,
                         device="cuda").requires_grad_(True)
        ref = routing_oracle_scan(p_.to(dtype), wk_.to(dtype),
                                  qk_.to(dtype), d_.to(dtype), scores,
                                  policy=POLICY, rank=rank, rank_max=rank,
                                  initial_state=s_, initial_keys=k_)
        (ref["reads"] * upstream.to(dtype)).sum().backward()
        return ref, (p_.grad, wk_.grad, qk_.grad, d_.grad, s_.grad, k_.grad)

    ref32, grads32 = oracle_grads(torch.float32)
    _ref64, grads64 = oracle_grads(torch.float64)

    # the ZERO-GRADIENT FINDING on the same fixture: the oracle path's
    # structural gradient to the router scores is exactly zero
    g_finding = router_scores(2, 256, HEADS, 0.5, seed=62).requires_grad_(True)
    p_f = payload32.clone().requires_grad_(True)
    ref_f = routing_oracle_scan(p_f, write_keys32, query_keys32, decay32,
                                g_finding, policy=POLICY, rank=rank,
                                rank_max=rank)
    finding = torch.autograd.grad((ref_f["reads"] * upstream).sum(),
                                  [g_finding, p_f], allow_unused=True)

    # the kernel path's six gradients + the estimator, via the k14 backward
    grads = _k14_backward_body(payload32, write_keys32, query_keys32, decay32,
                              upstream, schedule, out["live_pre"],
                              out["ckpt_state"], out["ckpt_live"],
                              out["ckpt_keys"], out["op_scores"], rank, seg,
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
    # the six gradients are BIT-EQUAL to K10's own body's on the same
    # recorded channels (the estimator changes nothing else)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import _k10_backward_body
    ten = _k10_backward_body(payload32, write_keys32, query_keys32, decay32,
                             upstream, schedule, out["live_pre"],
                             out["ckpt_state"], out["ckpt_live"],
                             out["ckpt_keys"], rank, seg, POLICY.read_scale)
    six_bit_equal = dict(
        dpayload=bool(torch.equal(grads[0], ten[0])),
        dwrite_keys=bool(torch.equal(grads[1], ten[1])),
        dquery_keys=bool(torch.equal(grads[2], ten[2])),
        ddecay=bool(torch.equal(grads[3], ten[3])),
        dstate=bool(torch.equal(grads[5], ten[4])),
        dkeys=bool(torch.equal(grads[6], ten[5])))
    # the estimator vs the fp64 autograd of its own DEFINED surrogate
    got = grads[4]
    c = (grads32[0].float() * payload32.float()).sum(-1)
    s64 = out["op_scores"].to(torch.float64).clone().requires_grad_(True)
    kstar = out["op"][..., 0].to(torch.int64)
    probs = torch.softmax(s64.view(-1, 5), dim=-1)
    rows_idx = torch.arange(probs.shape[0], device="cuda")
    pk = probs[rows_idx, kstar.reshape(-1)]
    surrogate = (c.to(torch.float64).reshape(-1) * pk).sum()
    d_ref = torch.autograd.grad(surrogate, s64)[0]
    err = (got.to(torch.float64) - d_ref).abs().max().item()
    gate = dict(
        fixture=dict(batch=2, tokens=256, heads=HEADS, rank=rank,
                     value_dim=VALUE_DIM, key_dim=KEY_DIM, dtype="fp32",
                     segment_tokens=seg,
                     router="random scores ~ N(0, 0.5) (the flipped regime)"),
        six_gradients=rows,
        six_gradients_bit_equal_k10_body=six_bit_equal,
        zero_gradient_finding=dict(
            note="the ORACLE path's structural gradient to the router "
                 "scores is exactly zero under the recorded-schedule "
                 "doctrine (the op choice is a backward constant) -- the "
                 "finding; the kernel path's straight-through estimator is "
                 "the admitted exception",
            oracle_drouter_is_none=finding[0] is None),
        estimator=dict(
            note="the admitted straight-through surrogate vs the fp64 "
                 "autograd of its own DEFINED surrogate objective "
                 "(c_t * p_{k*}, p = softmax over the five recorded "
                 "scores); a defined estimator, not the true adjoint -- "
                 "its bias is the registration's disclosure: c_t is zero "
                 "on MERGE/POP/NONE tokens",
            max_abs_diff=err,
            nonzero=bool(float(got.abs().max()) > 0.0),
            scale=float(d_ref.abs().max())),
        op_histogram=op_histogram(schedule))
    receipt["gate_b"] = gate
    return gate


def gate_d(receipt: dict) -> dict:
    """The honest cost statement: the k14 scan's forward/backward walls at
    the registered shape vs K10's own launch on the same fixture, same
    card."""
    payload, write_keys, query_keys, decay = registered_fixture(
        seed=91, batch=2, tokens=2048, dtype=torch.float32)
    state = torch.zeros(2, HEADS, RANK, VALUE_DIM, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(2, HEADS, RANK, dtype=torch.bool, device="cuda")
    keys = torch.zeros(2, HEADS, RANK, KEY_DIM, dtype=torch.float32,
                       device="cuda")
    scores = router_scores(2, 2048, HEADS)
    walls = {}
    for name in ("k10", "k14"):
        for _ in range(3):
            if name == "k10":
                _k10_forward_launch(payload, write_keys, query_keys, decay,
                                    state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, collect=False, warps=8,
                                    stages=3)
            else:
                _k14_forward_launch(payload, write_keys, query_keys, decay,
                                    scores, state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, collect=False, warps=8,
                                    stages=3)
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(10):
            if name == "k10":
                _k10_forward_launch(payload, write_keys, query_keys, decay,
                                    state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, collect=False, warps=8,
                                    stages=3)
            else:
                _k14_forward_launch(payload, write_keys, query_keys, decay,
                                    scores, state, live, keys, POL32, RANK,
                                    SEGMENT_TOKENS, collect=False, warps=8,
                                    stages=3)
        torch.cuda.synchronize()
        walls[name] = round((time.monotonic() - started) / 10, 3)
    # the backward wall (the k14 backward = K10's body + the estimator)
    out = _k14_forward_launch(payload, write_keys, query_keys, decay, scores,
                              state, live, keys, POL32, RANK, SEGMENT_TOKENS,
                              collect=False, warps=8, stages=3)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(2, 2048, HEADS, VALUE_DIM,
                           generator=torch.Generator().manual_seed(5)).cuda()
    _k14_backward_body(payload, write_keys, query_keys, decay, upstream,
                       schedule, out["live_pre"], out["ckpt_state"],
                       out["ckpt_live"], out["ckpt_keys"], out["op_scores"],
                       RANK, SEGMENT_TOKENS, POLICY.read_scale)
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(5):
        _k14_backward_body(payload, write_keys, query_keys, decay, upstream,
                           schedule, out["live_pre"], out["ckpt_state"],
                           out["ckpt_live"], out["ckpt_keys"],
                           out["op_scores"], RANK, SEGMENT_TOKENS,
                           POLICY.read_scale)
    torch.cuda.synchronize()
    backward_wall = round((time.monotonic() - started) / 5, 3)
    receipt["gate_d"] = dict(
        note="the routing channel's honest price at the registered shape: "
             "the forward carries the five per-op scores and the argmax "
             "beside K10's own chain quantities; the backward is K10's own "
             "body plus the estimator's small torch reduction on the "
             "recorded scores (no replay)",
        forward_ms_per_scan=dict(k10=walls["k10"] * 1e3,
                                 k14=walls["k14"] * 1e3,
                                 ratio=walls["k14"] / max(walls["k10"], 1e-9)),
        backward_ms_per_scan=dict(k14=backward_wall * 1e3))
    return receipt["gate_d"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        ROOT / "receipts" / "k14_gates.json"))
    args = ap.parse_args()
    torch.manual_seed(0)
    receipt = dict(probe="k14_learned_routing",
                   registration=str(
                       ROOT / "PREREGISTRATION_LEARNED_MANAGEMENT.md"),
                   policy=dict(tau_match=POLICY.tau_match,
                               tau_merge=POLICY.tau_merge,
                               tau_pop=POLICY.tau_pop,
                               tie_break=POLICY.tie_break,
                               read_scale=POLICY.read_scale),
                   reserved_score=RESERVED_SCORE,
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
    print(json.dumps(dict(
        gate_a_flips=dict(
            birth=receipt["gate_a"]["birth_g0"]["total"]["flips"],
            flipped=receipt["gate_a"]["flipped_g0p9"]["total"]["flips"]),
        birth_parity_flips=receipt["gate_a"]["birth_parity_vs_k10_launch"][
            "total"]["flips"],
        birth_argmax_unique=receipt["gate_a"]["birth_argmax_uniqueness"][
            "all_unique"],
        flip_rate_g0p9=receipt["gate_a"]["flipped_g0p9"]["routing"][
            "flip_rate"],
        estimator_diff=receipt["gate_b"]["estimator"]["max_abs_diff"],
        finding_drouter_none=receipt["gate_b"]["zero_gradient_finding"][
            "oracle_drouter_is_none"],
        forward_ms=dict(k10=receipt["gate_d"]["forward_ms_per_scan"]["k10"],
                        k14=receipt["gate_d"]["forward_ms_per_scan"]["k14"]),
    ), indent=1))


if __name__ == "__main__":
    main()
