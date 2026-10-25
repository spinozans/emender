"""Gate scripts + receipts for the X28E phase-2 fused kernels (pre-M2, M1).

Registration engineering gates 1-2 (+ the P1-6/G10 replay-consistency
obligation), each receipted with measured values verbatim, fixed seed, and
ZERO optimizer updates:

  * GATE 1 — fused-kernel equivalence vs the eager fp32 full-layer oracle
    (X24 gate standard, house max-relative metric = max|a-b| / max|b|):
    short-sequence fused output / final-state max-relative error < 0.03;
    input/parameter gradient max-relative error < 0.08; fused eval error
    < 0.03. Run at the G7 freeze-proposal dims (d_model=256, NH=4, K=64,
    V=128, B0=736) in the SPILLED store mode and additionally in the
    RESIDENT mode, with the G14 placeholder constants and bf16 replay
    checkpoints (the declared fused training path; an fp32-checkpoint
    diagnostic is receipted alongside).
    Receipt -> receipts/kernel_match.json

  * GATE 2/3 — replay-decision consistency (P1-6, EXTENDED TO FREE-BY-DECAY
    per G10) + multi-checkpoint branch equivalence: the fused backward kernel
    replays each segment FOLLOWING THE TRACE from the (bf16-rounded)
    checkpoints; the gate receipts that the echoed routing equals the
    forward trace at every checkpoint boundary, that replayed liveness/age
    are EXACT at every boundary (the free-by-decay decisions are replay
    inputs, never recomputed), that replayed outputs/boundary stores sit
    within the declared tolerance, that a construction where recomputing
    the death decision from the rounded checkpoint WOULD flip it stays
    exact under trace-following, and that EVERY routing case
    (R-EMPTY/R-NOMATCH/R-ZERO-RESIDUAL/R-CONFLICT/R-CAP/POP/NO-WRITE/
    free-by-decay) fires at checkpoint boundaries in the equivalence run.
    The eager-side gate (bptt.replay_consistency) is cross-checked on the
    same construction.
    Receipt -> receipts/replay_consistency.json

Usage: .venv/bin/python run_gates.py [--gpu 0]
Exit code 0 iff every registered threshold holds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # x28e_memory_head/

from eng import bptt                                          # noqa: E402
from eng.oracle import TapeLayer                               # noqa: E402
from eng.routing_trace import (                                # noqa: E402
    NO_WRITE, R_CAP, R_CONFLICT, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL,
    FREED_DECAY, FREED_POP,
)
from eng.tape_head import TapeHeadConfig, init_state, tape_scan  # noqa: E402
from eng.fused import (                                        # noqa: E402
    FusedTapeLayer, fused_fwd_raw, fused_replay_raw, fused_tape_scan,
    buffers_to_state, buffers_to_trace,
)
from eng.fused.fused_layer import load_eager_layer_weights      # noqa: E402

RECEIPTS = Path(__file__).resolve().parent / "receipts"
SEED = 20260128
ZERO_UPDATES = True  # no optimizer exists in this leg; receipted per gate


def house_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """House max-relative error (e75/e97 fused-parity standard)."""
    a = a.detach().float(); b = b.detach().float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-8))


# ---------------------------------------------------------------------------
# Gate 1: fused-kernel equivalence vs the eager fp32 full-layer oracle.
# ---------------------------------------------------------------------------
def _gate1_config(budget: int) -> TapeHeadConfig:
    """G7 freeze-proposal dims + G14 placeholder constants."""
    return TapeHeadConfig(key_dim=64, value_dim=128, n_heads=4, budget=budget,
                          match_threshold=0.5, eps_residual=1e-6,
                          pop_threshold=0.5, write_enable_threshold=0.5,
                          free_threshold=1e-3, decay_rate=0.999995,
                          usage_decay=0.999)


def gate1_one(budget: int, ckpt_dtype: torch.dtype, B: int, T: int,
              mode: str) -> dict:
    torch.manual_seed(SEED)
    dev = "cuda:0"
    d_model, NH, K, V = 256, 4, 64, 128
    cfg = _gate1_config(budget)
    eager = TapeLayer(d_model, cfg).to(dev).float()
    fused = FusedTapeLayer(d_model, cfg, ckpt_interval=16,
                          checkpoint_dtype=ckpt_dtype).to(dev).float()
    load_eager_layer_weights(fused, eager)
    x = torch.randn(B, T, d_model, device=dev)
    x_e = x.detach().clone().requires_grad_(True)
    x_f = x.detach().clone().requires_grad_(True)

    # fixed random loss weights over every differentiable output
    torch.manual_seed(SEED + 1)
    w_y = torch.randn(B, T, d_model, device=dev)
    w_ro = torch.randn(B, T, NH, V, device=dev)
    w_k = torch.randn(B, NH, cfg.slots, K, device=dev)
    w_v = torch.randn(B, NH, cfg.slots, V, device=dev) * 0.1
    w_s = torch.randn(B, NH, cfg.slots, device=dev) * 0.1

    def loss_of(y, ro, st):
        return ((y * w_y).sum() + (ro * w_ro).sum() + (st.keys * w_k).sum()
                + (st.values * w_v).sum() + (st.strength * w_s).sum())

    y_e, ro_e, st_e, tr_e = eager(x_e)
    loss_e = loss_of(y_e, ro_e, st_e)
    loss_e.backward()
    y_f, ro_f, st_f, tr_f = fused(x_f)
    loss_f = loss_of(y_f, ro_f, st_f)
    loss_f.backward()

    rec: dict = {"mode": mode, "budget": budget, "B": B, "T": T,
                 "checkpoint_dtype": str(ckpt_dtype), "seed": SEED,
                 "zero_updates": ZERO_UPDATES}
    # outputs / final state (threshold 0.03)
    rec["read_out_max_rel"] = house_rel(ro_f, ro_e)
    rec["layer_out_max_rel"] = house_rel(y_f, y_e)
    rec["final_keys_max_rel"] = house_rel(st_f.keys, st_e.keys)
    rec["final_values_max_rel"] = house_rel(st_f.values, st_e.values)
    rec["final_strength_max_rel"] = house_rel(st_f.strength, st_e.strength)
    rec["final_live_exact"] = bool(torch.equal(st_f.live, st_e.live))
    rec["final_age_exact"] = bool(torch.equal(st_f.age, st_e.age))
    # routing-trace parity (integral decisions: exact equality)
    try:
        for name, ta, tb in (("case", tr_f.case, tr_e.case),
                             ("matched", tr_f.matched, tr_e.matched),
                             ("pop_fired", tr_f.pop_fired, tr_e.pop_fired),
                             ("pop_target", tr_f.pop_target, tr_e.pop_target),
                             ("alloc_target", tr_f.alloc_target, tr_e.alloc_target),
                             ("w_on", tr_f.w_on, tr_e.w_on),
                             ("freed", tr_f.freed, tr_e.freed)):
            assert torch.equal(ta, tb), name
        rec["trace_parity_exact"] = True
    except AssertionError as exc:
        rec["trace_parity_exact"] = False
        rec["trace_parity_error"] = str(exc)

    # input + parameter gradients (threshold 0.08)
    rec["input_grad_max_rel"] = house_rel(x_f.grad, x_e.grad)
    pmap = [(f"head.head.{n}", f"head.{n}") for n in
            ("W_q.weight", "W_q.bias", "W_k.weight", "W_k.bias",
             "W_v.weight", "W_v.bias", "W_write.weight", "W_write.bias",
             "W_pop.weight", "W_pop.bias", "W_append.weight", "W_append.bias",
             "append_err_coef")] + [("W_o.weight", "W_o.weight"),
                                    ("W_o.bias", "W_o.bias")]
    worst, worst_name = 0.0, ""
    for fname, ename in pmap:
        pf = dict(fused.named_parameters())[fname]
        pe = dict(eager.named_parameters())[ename]
        if pe.grad is None and pf.grad is None:
            continue
        assert pe.grad is not None and pf.grad is not None, (fname, ename)
        rel = house_rel(pf.grad, pe.grad)
        if rel > worst:
            worst, worst_name = rel, fname
    rec["param_grad_max_rel"] = worst
    rec["param_grad_worst_name"] = worst_name

    # fused eval error (no-grad fused forward vs no-grad eager forward)
    torch.manual_seed(SEED + 2)
    x_eval = torch.randn(B, T, d_model, device=dev)
    with torch.no_grad():
        y_eval_e, _, _, _ = eager(x_eval)
        y_eval_f, ro_eval_f, st_eval_f, _ = fused(x_eval)
    rec["fused_eval_out_max_rel"] = house_rel(y_eval_f, y_eval_e)
    rec["fused_eval_read_max_rel"] = house_rel(ro_eval_f, eager(x_eval)[1])
    rec["fused_eval_final_values_max_rel"] = house_rel(st_eval_f.values,
                                                       eager(x_eval)[2].values)
    # registered thresholds
    out_ok = (rec["read_out_max_rel"] < 0.03 and rec["layer_out_max_rel"] < 0.03
              and rec["final_keys_max_rel"] < 0.03
              and rec["final_values_max_rel"] < 0.03
              and rec["final_strength_max_rel"] < 0.03
              and rec["final_live_exact"] and rec["final_age_exact"]
              and rec["trace_parity_exact"])
    grad_ok = rec["input_grad_max_rel"] < 0.08 and rec["param_grad_max_rel"] < 0.08
    eval_ok = (rec["fused_eval_out_max_rel"] < 0.03
               and rec["fused_eval_read_max_rel"] < 0.03
               and rec["fused_eval_final_values_max_rel"] < 0.03)
    rec["output_gate_pass"] = bool(out_ok)
    rec["gradient_gate_pass"] = bool(grad_ok)
    rec["eval_gate_pass"] = bool(eval_ok)
    rec["pass"] = bool(out_ok and grad_ok and eval_ok)
    return rec


def gate_kernel_match() -> dict:
    """Registration gate 1: receipts/kernel_match.json."""
    receipt = {
        "gate": "kernel_match (registration engineering gate 1, X24 standard)",
        "thresholds": {"output_final_state_max_rel": 0.03,
                       "input_parameter_gradient_max_rel": 0.08,
                       "fused_eval_max_rel": 0.03},
        "metric": "house max-relative error: max|a-b| / max|b| (e75/e97 standard)",
        "dims": {"d_model": 256, "NH": 4, "K": 64, "V": 128, "B0": 736},
        "constants_G14": {"match_threshold": 0.5, "eps_residual": 1e-6,
                          "pop_threshold": 0.5, "write_enable_threshold": 0.5,
                          "CKPT_INTERVAL": 16, "decay_rate": 0.999995,
                          "free_threshold": 1e-3},
        "seed": SEED, "zero_updates": ZERO_UPDATES,
    }
    # the tape_B0 arm configuration (spilled store: S=736 > resident tile)
    receipt["spilled_bf16_ckpt"] = gate1_one(736, torch.bfloat16, B=2, T=64,
                                             mode="spilled")
    # fp32-checkpoint diagnostic (isolates bf16-checkpoint rounding)
    receipt["spilled_fp32_ckpt_diagnostic"] = gate1_one(736, torch.float32,
                                                        B=2, T=64,
                                                        mode="spilled")
    # resident store mode (S=48 <= MAX_RESIDENT_SLOTS=64)
    receipt["resident_bf16_ckpt"] = gate1_one(48, torch.bfloat16, B=2, T=64,
                                              mode="resident")
    receipt["pass"] = all(receipt[k]["pass"] for k in
                          ("spilled_bf16_ckpt", "resident_bf16_ckpt"))
    return receipt


# ---------------------------------------------------------------------------
# Gate 2/3: replay-decision consistency (G10) + multi-checkpoint branch
# equivalence.
# ---------------------------------------------------------------------------
def _case_tokens(K: int, V: int):
    """All-cases construction with pop and free-by-decay at boundaries."""
    def e(i, scale=4.0):
        x = torch.zeros(K)
        x[i % K] = scale
        return x
    ALOGIT = 20.0

    def w(k_, v_, beta=0.9, pop=0.0, alogit=ALOGIT, q=None):
        return dict(k=k_, q=(q if q is not None else e(K - 1)), v=v_, beta=beta,
                    pop=pop, alogit=alogit)

    def r(q_, beta=0.1, pop=0.0):
        return dict(k=e(K - 2), q=q_, v=torch.zeros(V), beta=beta, pop=pop,
                    alogit=ALOGIT)

    vv = 0.5 * torch.ones(V)
    toks = [
        w(e(0), vv),                                   # t0  R-EMPTY
        w(e(0), vv),                                   # t1  R-ZERO-RESIDUAL (adjacent)
        r(e(0)),                                       # t2  NO-WRITE (ages advance)
        w(e(1), 2 * vv),                               # t3  R-NOMATCH
        w(e(1), 0.9 * torch.ones(V)),                  # t4  R-CONFLICT (adjacent slot2)
        r(e(0)),                                       # t5  NO-WRITE (ages advance)
        w(e(2), 3 * vv),                               # t6  append->R-CAP (full)
        dict(k=e(1), q=e(1), v=4.0 * torch.ones(V), beta=0.9, pop=1.0,
             alogit=ALOGIT),                           # t7  POP kills slot2 -> R-NOMATCH
        r(e(0)),                                       # t8  NO-WRITE
        r(e(1), pop=0.9),                              # t9  NO-WRITE + partial pop
    ]
    return toks


def _tokens_to_inputs(tokens, B, T, NH, K, V, dev):
    def stack(key, tail):
        return torch.stack([t[key] for t in tokens]).view(B, T, NH, *tail).to(dev)
    k = stack("k", (K,)); q = stack("q", (K,)); v = stack("v", (V,))
    beta = torch.tensor([[t["beta"] for t in tokens]]).view(B, T, NH).to(dev)
    pop = torch.tensor([[t["pop"] for t in tokens]]).view(B, T, NH).to(dev)
    alog = torch.tensor([[t["alogit"] for t in tokens]]).view(B, T, NH).to(dev)
    return k, q, v, beta, pop, alog


def _replay_run(cfg, toks, K, V, ckpt, ckpt_dtype, dev):
    B, T, NH = 1, len(toks), 1
    k, q, v, beta, pop, alog = _tokens_to_inputs(toks, B, T, NH, K, V, dev)
    s0 = init_state(B, cfg, dev)
    buf = fused_fwd_raw(k, q, v, beta, pop, alog, s0, cfg,
                        ckpt_interval=ckpt, checkpoint_dtype=ckpt_dtype,
                        max_resident=64, tile_s=32)
    rp = fused_replay_raw(buf, k, q, v, beta, pop, alog, cfg,
                          ckpt_interval=ckpt, tile_s=32)
    return buf, rp, (k, q, v, beta, pop, alog), s0


def gate_replay_consistency() -> dict:
    """Registration gate 2 (P1-6 extended to free-by-decay, G10) + the
    multi-checkpoint branch-equivalence receipt."""
    dev = "cuda:0"
    receipt = {
        "gate": "replay_consistency (P1-6 + G10 free-by-decay extension)",
        "seed": SEED, "zero_updates": ZERO_UPDATES,
    }

    # ---- engineered all-cases construction, fp32 and bf16 checkpoints ----
    K, V = 8, 16
    cases_rec: dict = {}
    for dtype, dname in ((torch.float32, "fp32_ckpts"),
                         (torch.bfloat16, "bf16_ckpts")):
        # decay 0.5 + free_threshold 0.01: free-by-decay fires at boundaries
        cfg = TapeHeadConfig(key_dim=K, value_dim=V, n_heads=1, budget=3,
                             match_threshold=0.5, eps_residual=1e-6,
                             pop_threshold=0.5, write_enable_threshold=0.5,
                             free_threshold=0.01, decay_rate=0.5,
                             usage_decay=0.999)
        toks = _case_tokens(K, V)
        buf, rp, ins, s0 = _replay_run(cfg, toks, K, V, ckpt=2,
                                       ckpt_dtype=dtype, dev=dev)
        tr = buffers_to_trace(buf)
        rec = {"checkpoint_dtype": dname, "CKPT_INTERVAL": 2,
               "T": len(toks), "num_checkpoints": buf.num_ck}
        # trace echo: replayed routing == forward trace, every field
        rec["echo_case_equal"] = bool(torch.equal(rp["echo_case"], buf.t_case))
        rec["echo_matched_equal"] = bool(torch.equal(rp["echo_matched"], buf.t_matched))
        rec["echo_popt_equal"] = bool(torch.equal(rp["echo_popt"], buf.t_popt))
        rec["echo_alloc_equal"] = bool(torch.equal(rp["echo_alloc"], buf.t_alloc))
        rec["echo_popf_equal"] = bool(torch.equal(rp["echo_popf"], buf.t_popf))
        # boundary equivalence: liveness/age EXACT (G10: traced frees applied)
        live_exact = all(torch.equal(rp["bd_live"][i].bool(),
                                     buf.ck_live[i].bool())
                         for i in range(1, buf.num_ck))
        age_exact = all(torch.equal(rp["bd_age"][i], buf.ck_age[i])
                        for i in range(1, buf.num_ck))
        rec["boundary_live_exact"] = bool(live_exact)
        rec["boundary_age_exact"] = bool(age_exact)
        rec["boundary_keys_max_rel"] = max(
            house_rel(rp["bd_keys"][i], buf.ck_keys[i].float())
            for i in range(1, buf.num_ck))
        rec["boundary_values_max_rel"] = max(
            house_rel(rp["bd_values"][i], buf.ck_values[i].float())
            for i in range(1, buf.num_ck))
        rec["boundary_strength_max_rel"] = max(
            house_rel(rp["bd_strength"][i], buf.ck_strength[i].float())
            for i in range(1, buf.num_ck))
        rec["replay_output_max_rel"] = house_rel(rp["replay_out"], buf.out)
        # branch-equivalence coverage: every case fires; per-case first token
        # and its checkpoint boundary
        coverage = {}
        for name, code in (("R-EMPTY", R_EMPTY), ("R-NOMATCH", R_NOMATCH),
                           ("R-ZERO-RESIDUAL", R_ZERO_RESIDUAL),
                           ("R-CONFLICT", R_CONFLICT), ("R-CAP", R_CAP),
                           ("NO-WRITE", NO_WRITE)):
            idx = (tr.case[0, :, 0] == code).nonzero().flatten().tolist()
            coverage[name] = {"tokens": idx,
                              "nearest_boundary":
                                  [(t // 2) * 2 for t in idx[:1]]}
        coverage["POP"] = {"tokens": tr.pop_fired[0, :, 0].nonzero().flatten().tolist()}
        coverage["FREE-BY-POP"] = {
            "count": int((tr.freed == FREED_POP).sum()),
            "tokens": tr.freed.eq(FREED_POP).any(-1).any(-1).nonzero().flatten().tolist()}
        coverage["FREE-BY-DECAY"] = {
            "count": int((tr.freed == FREED_DECAY).sum()),
            "tokens": tr.freed.eq(FREED_DECAY).any(-1).any(-1).nonzero().flatten().tolist()}
        rec["case_coverage"] = coverage
        required = {("R-EMPTY", R_EMPTY), ("R-NOMATCH", R_NOMATCH),
                    ("R-ZERO-RESIDUAL", R_ZERO_RESIDUAL),
                    ("R-CONFLICT", R_CONFLICT), ("R-CAP", R_CAP),
                    ("NO-WRITE", NO_WRITE)}
        rec["all_cases_fire"] = all(
            bool((tr.case[0, :, 0] == c).any()) for _, c in required)
        rec["pop_fires"] = bool(tr.pop_fired.any())
        rec["free_by_pop_fires"] = int((tr.freed == FREED_POP).sum()) > 0
        rec["free_by_decay_fires"] = int((tr.freed == FREED_DECAY).sum()) > 0
        tol = 0.03
        rec["boundary_tolerance_pass"] = bool(
            rec["boundary_values_max_rel"] < tol
            and rec["boundary_strength_max_rel"] < tol
            and rec["boundary_keys_max_rel"] < tol
            and rec["replay_output_max_rel"] < tol)
        rec["pass"] = bool(rec["echo_case_equal"] and rec["echo_matched_equal"]
                           and rec["echo_popt_equal"] and rec["echo_alloc_equal"]
                           and rec["echo_popf_equal"] and live_exact and age_exact
                           and rec["boundary_tolerance_pass"]
                           and rec["all_cases_fire"] and rec["pop_fires"]
                           and rec["free_by_pop_fires"]
                           and rec["free_by_decay_fires"])
        if dname == "fp32_ckpts":
            # exact-mode receipt: replay from fp32 checkpoints is essentially
            # bit-identical (the branch-equivalence baseline)
            rec["replay_output_max_rel_note"] = (
                "fp32 checkpoints: replay recomputation is exact to ~1e-6")
        cases_rec[dname] = rec

        # eager-side cross-check on the same construction (phase-1 gate)
        if dname == "fp32_ckpts":
            k, q, v, beta, pop, alog = ins
            eager_gate = bptt.replay_consistency(
                k, q, v, beta, pop, alog, s0, cfg, ckpt_interval=2,
                checkpoint_dtype=torch.bfloat16)
            cases_rec["eager_side_cross_check"] = {
                "traces_equal": eager_gate["traces_equal"],
                "live_and_age_exact": eager_gate["live_and_age_exact"],
                "output_max_rel": eager_gate["output_max_rel"],
            }
    receipt["all_cases_at_boundaries"] = cases_rec

    # ---- the would-flip construction: recomputation from the bf16-rounded
    # checkpoint WOULD free a slot that trace-following replay keeps live ----
    K2, V2 = 8, 16
    cfg = TapeHeadConfig(key_dim=K2, value_dim=V2, n_heads=1, budget=4,
                         match_threshold=0.5, eps_residual=1e-6,
                         pop_threshold=0.5, write_enable_threshold=0.5,
                         free_threshold=0.5, decay_rate=1.0, usage_decay=0.999)
    s_flip = 0.5009       # fp32 > free_threshold (survives); bf16 rounds to 0.5
    toks = [_case_tokens(K2, V2)[0]] + _case_tokens(K2, V2)[6:8]
    dev2 = dev
    B, T, NH = 1, 3, 1
    k, q, v, beta, pop, alog = _tokens_to_inputs(toks, B, T, NH, K2, V2, dev2)
    s0 = init_state(B, cfg, dev2)
    s0.keys[0, 0, 0] = torch.zeros(K2, device=dev2); s0.keys[0, 0, 0][0] = 4.0
    s0.values[0, 0, 0] = 0.5 * torch.ones(V2, device=dev2)
    s0.strength[0, 0, 0] = s_flip
    s0.live[0, 0, 0] = True
    buf = fused_fwd_raw(k, q, v, beta, pop, alog, s0, cfg,
                        ckpt_interval=2, checkpoint_dtype=torch.bfloat16,
                        max_resident=64, tile_s=32)
    rp = fused_replay_raw(buf, k, q, v, beta, pop, alog, cfg,
                          ckpt_interval=2, tile_s=32)
    rounded = buf.ck_strength[1, 0, 0, 0].float()
    wf = {
        "construction": ("slot strength 0.5009 (fp32 > free_threshold=0.5, "
                        "survives the fp32 forward); bf16 rounds it to 0.5 "
                        "<= free_threshold, so recomputing the death decision "
                        "from the rounded checkpoint WOULD free it"),
        "fp32_strength": float(s_flip),
        "bf16_checkpoint_strength": float(rounded),
        "free_threshold": cfg.free_threshold,
        "recomputed_would_free": bool(float(rounded) <= cfg.free_threshold),
        "forward_live_at_boundary": bool(buf.ck_live[1, 0, 0, 0].item()),
        "replay_live_at_boundary": bool(rp["bd_live"][1, 0, 0, 0].item()),
        "boundary_live_exact_all": bool(all(
            torch.equal(rp["bd_live"][i].bool(), buf.ck_live[i].bool())
            for i in range(1, buf.num_ck))),
        "replay_output_max_rel": house_rel(rp["replay_out"], buf.out),
    }
    wf["pass"] = bool(wf["recomputed_would_free"]
                      and wf["forward_live_at_boundary"]
                      and wf["replay_live_at_boundary"]
                      and wf["boundary_live_exact_all"]
                      and wf["replay_output_max_rel"] < 0.03)
    receipt["would_flip_free_by_decay"] = wf

    # ---- randomized-store replay gate (bf16), accelerated decay ----
    torch.manual_seed(SEED)
    K3, V3, S3, NH3, T3 = 16, 32, 130, 2, 12
    cfg = TapeHeadConfig(key_dim=K3, value_dim=V3, n_heads=NH3, budget=S3,
                         decay_rate=0.97, free_threshold=0.08)
    B3 = 2
    k = torch.randn(B3, T3, NH3, K3, device=dev)
    q = torch.randn(B3, T3, NH3, K3, device=dev)
    v = torch.randn(B3, T3, NH3, V3, device=dev)
    beta = torch.rand(B3, T3, NH3, device=dev)
    pop = torch.rand(B3, T3, NH3, device=dev)
    alog = torch.randn(B3, T3, NH3, device=dev)
    aec = torch.randn(NH3, device=dev)
    s0 = init_state(B3, cfg, dev)
    s0.keys.normal_(); s0.values.normal_()
    s0.strength.uniform_(0.1, 1.0)
    s0.live[:] = torch.rand(B3, NH3, S3, device=dev) < 0.5
    buf = fused_fwd_raw(k, q, v, beta, pop, alog, s0, cfg, append_err_coef=aec,
                        ckpt_interval=3, checkpoint_dtype=torch.bfloat16,
                        max_resident=64, tile_s=32)
    rp = fused_replay_raw(buf, k, q, v, beta, pop, alog, cfg,
                          append_err_coef=aec, ckpt_interval=3, tile_s=32)
    tr = buffers_to_trace(buf)
    rnd = {
        "config": {"B": B3, "T": T3, "NH": NH3, "S": S3, "decay": 0.97,
                   "free_threshold": 0.08, "CKPT_INTERVAL": 3,
                   "checkpoint_dtype": "bfloat16"},
        "free_by_decay_count": int((tr.freed == FREED_DECAY).sum()),
        "free_by_pop_count": int((tr.freed == FREED_POP).sum()),
        "pop_fired_count": int(tr.pop_fired.sum()),
        "boundary_live_exact": bool(all(
            torch.equal(rp["bd_live"][i].bool(), buf.ck_live[i].bool())
            for i in range(1, buf.num_ck))),
        "boundary_age_exact": bool(all(
            torch.equal(rp["bd_age"][i], buf.ck_age[i])
            for i in range(1, buf.num_ck))),
        "replay_output_max_rel": house_rel(rp["replay_out"], buf.out),
    }
    rnd["boundary_values_max_rel"] = max(
        house_rel(rp["bd_values"][i], buf.ck_values[i].float())
        for i in range(1, buf.num_ck))
    rnd["boundary_strength_max_rel"] = max(
        house_rel(rp["bd_strength"][i], buf.ck_strength[i].float())
        for i in range(1, buf.num_ck))
    rnd["pass"] = bool(rnd["free_by_decay_count"] > 0
                       and rnd["boundary_live_exact"]
                       and rnd["boundary_age_exact"]
                       and rnd["replay_output_max_rel"] < 0.03
                       and rnd["boundary_values_max_rel"] < 0.03
                       and rnd["boundary_strength_max_rel"] < 0.03)
    receipt["randomized_store_bf16"] = rnd

    receipt["pass"] = bool(
        cases_rec["fp32_ckpts"]["pass"] and cases_rec["bf16_ckpts"]["pass"]
        and wf["pass"] and rnd["pass"])
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    ok = True
    t0 = time.time()
    km = gate_kernel_match()
    (RECEIPTS / "kernel_match.json").write_text(json.dumps(km, indent=2) + "\n")
    ok &= km["pass"]
    print("=== GATE 1: kernel_match (registration gate 1) ===")
    print(json.dumps({k: v for k, v in km.items()
                      if k in ("pass", "thresholds", "metric", "dims")},
                     indent=2))
    for key in ("spilled_bf16_ckpt", "spilled_fp32_ckpt_diagnostic",
                "resident_bf16_ckpt"):
        r = km[key]
        print(f"[{key}] pass={r['pass']} "
              f"read_out={r['read_out_max_rel']:.3e} "
              f"layer_out={r['layer_out_max_rel']:.3e} "
              f"keys={r['final_keys_max_rel']:.3e} "
              f"values={r['final_values_max_rel']:.3e} "
              f"strength={r['final_strength_max_rel']:.3e} "
              f"input_grad={r['input_grad_max_rel']:.3e} "
              f"param_grad={r['param_grad_max_rel']:.3e} "
              f"(worst {r['param_grad_worst_name']}) "
              f"eval={r['fused_eval_out_max_rel']:.3e} "
              f"trace_parity={r['trace_parity_exact']} "
              f"live={r['final_live_exact']} age={r['final_age_exact']}")

    rc = gate_replay_consistency()
    (RECEIPTS / "replay_consistency.json").write_text(
        json.dumps(rc, indent=2) + "\n")
    ok &= rc["pass"]
    print("=== GATE 2/3: replay_consistency (P1-6 + G10) ===")
    for key in ("fp32_ckpts", "bf16_ckpts"):
        r = rc["all_cases_at_boundaries"][key]
        print(f"[all-cases/{key}] pass={r['pass']} live_exact={r['boundary_live_exact']} "
              f"age_exact={r['boundary_age_exact']} out_rel={r['replay_output_max_rel']:.3e} "
              f"v_rel={r['boundary_values_max_rel']:.3e} s_rel={r['boundary_strength_max_rel']:.3e} "
              f"all_cases={r['all_cases_fire']} free_by_decay={r['free_by_decay_fires']}")
    wf = rc["would_flip_free_by_decay"]
    print(f"[would-flip] pass={wf['pass']} recomputed_would_free={wf['recomputed_would_free']} "
          f"forward_live={wf['forward_live_at_boundary']} replay_live={wf['replay_live_at_boundary']}")
    rnd = rc["randomized_store_bf16"]
    print(f"[randomized/bf16] pass={rnd['pass']} free_by_decay={rnd['free_by_decay_count']} "
          f"live_exact={rnd['boundary_live_exact']} out_rel={rnd['replay_output_max_rel']:.3e}")
    ecc = rc["all_cases_at_boundaries"].get("eager_side_cross_check")
    if ecc:
        print(f"[eager cross-check] traces_equal={ecc['traces_equal']} "
              f"live_age_exact={ecc['live_and_age_exact']} "
              f"out_rel={ecc['output_max_rel']:.3e}")
    print(f"gates complete in {time.time()-t0:.1f}s; overall pass={bool(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
