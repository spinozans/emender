"""Unit tests for the X28E phase-2 fused Triton kernels (M1 engineering, pre-M2).

Coverage (registration engineering gates + supervisor resolutions):

  * routing parity vs the eager trace on RANDOMIZED STORES (resident and
    spilled kernel modes, ragged checkpoint tails): every trace field —
    including the G10 free-by-decay codes — must be EXACTLY equal, final
    liveness/age exact, outputs/states at fp32 parity;
  * every routing case firing on the fused path and matching the eager oracle
    (R-EMPTY / R-NOMATCH / R-ZERO-RESIDUAL / R-CONFLICT / R-CAP / NO-WRITE /
    POP / free-by-decay), including budget-cap behavior (G5: allocation never
    exceeds B);
  * gradient parity vs eager autograd (fp32 checkpoints ~1e-5; bf16
    checkpoints within the registered 0.08 bound), on inputs, the append_err
    coefficient and the initial store, with final-state-grad seeding and
    ragged segments;
  * bf16-replay trace consistency (G10): an engineered near-``free_threshold``
    construction where recomputing the death decision from the bf16-rounded
    checkpoint WOULD flip it, while trace-following replay keeps liveness
    exact at every checkpoint boundary;
  * G8 receipt: the final token's write is gradient-dead on the fused path.

All runs are zero-update (no optimizer touched). Requires a CUDA GPU.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # x28e_memory_head/

from eng.routing_trace import (  # noqa: E402
    NO_SLOT, NO_WRITE, R_CAP, R_CONFLICT, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL,
    FREED_DECAY, FREED_POP, assert_traces_equal,
)
from eng.tape_head import (  # noqa: E402
    TapeHeadConfig, TapeState, fade_horizon, init_state, tape_scan,
)
from eng.fused import (  # noqa: E402
    fused_fwd_raw, fused_replay_raw, fused_tape_scan, buffers_to_state,
    buffers_to_trace,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
DEV = "cuda:0"
K, V = 8, 16
ALOGIT_FULL = 20.0


def make_cfg(**over) -> TapeHeadConfig:
    base = dict(key_dim=K, value_dim=V, n_heads=1, budget=8,
                match_threshold=0.5, eps_residual=1e-6, pop_threshold=0.5,
                write_enable_threshold=0.5, free_threshold=1e-3,
                decay_rate=0.999995, usage_decay=0.999)
    base.update(over)
    return TapeHeadConfig(**base)


def e(i, scale=4.0):
    x = torch.zeros(K)
    x[i] = scale
    return x


def build_inputs(tokens, B=1, requires_grad=False):
    NH, T = 1, len(tokens)
    def leaf(x, tail):
        t = x.view(B, T, NH, *tail)
        return t.requires_grad_(True) if requires_grad else t
    return (leaf(torch.stack([t["k"] for t in tokens]), (K,)),
            leaf(torch.stack([t["q"] for t in tokens]), (K,)),
            leaf(torch.stack([t["v"] for t in tokens]), (V,)),
            leaf(torch.tensor([[t["beta"] for t in tokens]]), ()),
            leaf(torch.tensor([[t["pop"] for t in tokens]]), ()),
            leaf(torch.tensor([[t["alogit"] for t in tokens]]), ()))


def w(k_, v_, beta=0.9, pop=0.0, alogit=ALOGIT_FULL, q=None):
    return dict(k=k_, q=(q if q is not None else e(7)), v=v_, beta=beta, pop=pop,
                alogit=alogit)


def r(q_, beta=0.1, pop=0.0, k_=None, v_=None, alogit=ALOGIT_FULL):
    return dict(k=(k_ if k_ is not None else e(6)), q=q_,
                v=(v_ if v_ is not None else torch.zeros(V)), beta=beta,
                pop=pop, alogit=alogit)


def run_both(k, q, v, beta, pop, alog, s0, cfg, ckpt=4, ckpt_dtype=torch.float32,
             max_resident=64, tile_s=32, aec=None):
    """Run the eager oracle and the fused kernels on identical inputs."""
    if aec is None:
        aec = torch.zeros(cfg.n_heads, device=k.device)
    out_e, st_e, tr_e = tape_scan(k, q, v, beta, pop, alog, s0, cfg,
                                  append_err_coef=aec)
    buf = fused_fwd_raw(k, q, v, beta, pop, alog, s0, cfg, append_err_coef=aec,
                        ckpt_interval=ckpt, checkpoint_dtype=ckpt_dtype,
                        max_resident=max_resident, tile_s=tile_s)
    return (out_e, st_e, tr_e), (buf.out, buffers_to_state(buf), buffers_to_trace(buf)), buf


def assert_state_close(st_f, st_e, tol=1e-5):
    assert torch.equal(st_f.live, st_e.live), "live mask differs"
    assert torch.equal(st_f.age, st_e.age), "age differs"
    for name, a, b in (("keys", st_f.keys, st_e.keys),
                       ("values", st_f.values, st_e.values),
                       ("strength", st_f.strength, st_e.strength),
                       ("usage", st_f.usage, st_e.usage)):
        assert torch.allclose(a, b, atol=tol, rtol=tol), (name, (a - b).abs().max())


def max_rel(a, b):
    """House max-relative error (the e75/e97 fused-parity standard):
    per-tensor max-abs-diff normalized by the reference max magnitude.
    Pointwise relative error is intentionally not used: near-zero gradient
    elements make pointwise denominators meaningless (a ~1e-7 bf16-replay
    difference on a ~1e-3-scale gradient would report O(1) rel error while
    contributing nothing)."""
    a = a.float(); b = b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-8))


# ---------------------------------------------------------------------------
# Routing/trace parity on randomized stores (resident + spilled modes).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("budget,mode", [(16, "resident"), (130, "spilled")])
def test_routing_parity_random_stores(seed, budget, mode):
    torch.manual_seed(100 + seed)
    # free_threshold chosen so the accelerated decay (0.99) frees slots within
    # the 11-token window: the G10 free-by-decay codes are guaranteed to fire
    cfg = make_cfg(budget=budget, n_heads=2, decay_rate=0.99, free_threshold=0.05)
    B, T, NH, S = 2, 11, 2, budget
    k = torch.randn(B, T, NH, K, device=DEV)
    q = torch.randn(B, T, NH, K, device=DEV)
    v = torch.randn(B, T, NH, V, device=DEV) * 0.5
    beta = torch.rand(B, T, NH, device=DEV)
    pop = torch.rand(B, T, NH, device=DEV)
    alog = torch.randn(B, T, NH, device=DEV)
    aec = torch.randn(NH, device=DEV)
    s0 = init_state(B, cfg, DEV)
    s0.keys.normal_(); s0.values.normal_()
    s0.strength.uniform_(0.05, 1.0)
    s0.live[:] = torch.rand(B, NH, S, device=DEV) < 0.5
    s0.age.random_(0, 5)

    (out_e, st_e, tr_e), (out_f, st_f, tr_f), _ = run_both(
        k, q, v, beta, pop, alog, s0, cfg, ckpt=4, ckpt_dtype=torch.float32,
        max_resident=64, tile_s=32, aec=aec)

    assert_traces_equal(tr_e, tr_f, what=f"random-store parity ({mode})")
    assert_state_close(st_f, st_e)
    assert max_rel(out_f, out_e) < 1e-4
    # the fused trace must be integral and grad-free (routing is architectural)
    tr_f.assert_integral()
    # G10 coverage: pop deaths and decay deaths both fire and match the oracle
    assert int((tr_f.freed == FREED_DECAY).sum()) == int((tr_e.freed == FREED_DECAY).sum())
    assert int((tr_f.freed == FREED_POP).sum()) == int((tr_e.freed == FREED_POP).sum())
    if seed == 0 and mode == "resident":
        assert int((tr_f.freed == FREED_DECAY).sum()) > 0
        assert int(tr_f.pop_fired.sum()) > 0


def test_routing_parity_ragged_tail_spilled():
    """T=7 with CKPT=3: the final segment is ragged (length 1)."""
    torch.manual_seed(7)
    cfg = make_cfg(budget=200, n_heads=2, decay_rate=0.99)
    B, T, NH, S = 2, 7, 2, 200
    k = torch.randn(B, T, NH, K, device=DEV)
    q = torch.randn(B, T, NH, K, device=DEV)
    v = torch.randn(B, T, NH, V, device=DEV)
    beta = torch.rand(B, T, NH, device=DEV)
    pop = torch.rand(B, T, NH, device=DEV)
    alog = torch.randn(B, T, NH, device=DEV)
    s0 = init_state(B, cfg, DEV)
    s0.keys.normal_(); s0.values.normal_()
    s0.strength.uniform_(0.05, 1.0)
    s0.live[:] = torch.rand(B, NH, S, device=DEV) < 0.4
    s0.age.random_(0, 3)
    (out_e, st_e, tr_e), (out_f, st_f, tr_f), _ = run_both(
        k, q, v, beta, pop, alog, s0, cfg, ckpt=3)
    assert_traces_equal(tr_e, tr_f, what="ragged tail")
    assert_state_close(st_f, st_e)


# ---------------------------------------------------------------------------
# Every routing case on the fused path, matching the eager oracle.
# ---------------------------------------------------------------------------
def _fused_case_run(tokens, cfg, ckpt=2):
    k, q, v, beta, pop, alog = build_inputs(tokens)
    k = k.to(DEV); q = q.to(DEV); v = v.to(DEV)
    beta = beta.to(DEV); pop = pop.to(DEV); alog = alog.to(DEV)
    s0 = init_state(1, cfg, DEV)
    (out_e, st_e, tr_e), (out_f, st_f, tr_f), buf = run_both(
        k, q, v, beta, pop, alog, s0, cfg, ckpt=ckpt)
    assert_traces_equal(tr_e, tr_f, what="case parity")
    assert_state_close(st_f, st_e)
    return tr_f, st_f, out_f, buf


def test_fused_all_cases_and_cap():
    cfg = make_cfg(budget=3)
    vv = 0.5 * torch.ones(V)
    toks = [
        w(e(0), vv),                                   # t0: R-EMPTY  -> slot0
        w(e(0), vv),                                   # t1: R-ZERO-RESIDUAL -> slot1
        w(e(1), 2 * vv),                               # t2: R-NOMATCH -> slot2
        w(e(0), 0.9 * torch.ones(V)),                 # t3: R-CONFLICT (slot0)
        w(e(2), 3 * vv),                              # t4: append requested, full -> R-CAP
        dict(k=e(1), q=e(1), v=4.0 * torch.ones(V), beta=0.9, pop=1.0,
             alogit=ALOGIT_FULL),                      # t5: POP kills slot2 (G9:
        #  pop frees BEFORE write routing); k=e(1) then finds no live match
        #  (remaining keys e(0)) -> R-NOMATCH appending into the freed slot 2
        r(e(0), beta=0.1),                             # t6: NO-WRITE
    ]
    tr, st, out, _ = _fused_case_run(toks, cfg, ckpt=3)
    cases = [int(tr.case[0, t, 0]) for t in range(7)]
    assert cases == [R_EMPTY, R_ZERO_RESIDUAL, R_NOMATCH, R_CONFLICT, R_CAP,
                     R_NOMATCH, NO_WRITE], cases
    assert int(tr.alloc_target[0, 1, 0]) == 1
    assert int(tr.alloc_target[0, 4, 0]) == NO_SLOT    # R-CAP never allocates
    assert int(tr.pop_target[0, 5, 0]) == 2 and bool(tr.pop_fired[0, 5, 0])
    assert int(tr.freed[0, 5, 0, 2]) == FREED_POP      # G10: pop-death traced
    assert int(tr.alloc_target[0, 5, 0]) == 2          # G5: reuse of freed slot
    # G5: no slot is ever allocated past B
    assert int(st.live.sum()) <= 3
    assert int(tr.alloc_target.max()) < 3
    # the R-CAP merge is the delta-rule onto the best write match. With k=e(2)
    # every live key scores 0 (tie) so the best match is slot0 (lowest index,
    # matching torch.argmax): slot0 = 0.86 + 0.9*(1.5-0.86) = 1.436; slot1
    # (key e(0), value 0.5) is untouched
    assert torch.allclose(st.values[0, 0, 0],
                          torch.full((V,), 1.436, device=DEV), atol=1e-5)
    assert torch.allclose(st.values[0, 0, 1],
                          torch.full((V,), 0.5, device=DEV), atol=1e-5)


def test_fused_pop_kills_then_reuses_freed_slot():
    cfg = make_cfg()
    v0, v1 = 0.5 * torch.ones(V), 2.0 * torch.ones(V)
    toks = [
        w(e(0), v0, pop=0.0),                          # R-EMPTY -> slot0
        dict(k=e(1), q=e(0), v=v1, beta=0.9, pop=1.0, alogit=ALOGIT_FULL),
        # ^ pop kills slot0 BEFORE write routing (G9); store empty at write
        r(e(0)),                                       # reads the NEW slot only
    ]
    tr, st, out, _ = _fused_case_run(toks, cfg, ckpt=2)
    assert int(tr.case[0, 1, 0]) == R_EMPTY            # G9: pop frees before write
    assert bool(tr.pop_fired[0, 1, 0]) and int(tr.pop_target[0, 1, 0]) == 0
    assert int(tr.freed[0, 1, 0, 0]) == FREED_POP      # G10: pop-death traced
    assert int(tr.alloc_target[0, 1, 0]) == 0          # append reuses the freed slot
    assert torch.allclose(out[0, 2], (v1 * cfg.decay_rate).to(DEV), atol=1e-4)


def test_fused_free_by_decay_traced():
    # accelerated decay so slots die at the end-of-token check (G10 code 2)
    cfg = make_cfg(decay_rate=0.5)
    toks = [w(e(0), 0.5 * torch.ones(V))] + [r(e(0)) for _ in range(10)]
    tr, st, out, _ = _fused_case_run(toks, cfg, ckpt=4)
    assert int(tr.freed[0, 9, 0, 0]) == FREED_DECAY or \
           int(tr.freed[0, 10, 0, 0]) == FREED_DECAY
    assert int(st.live.sum()) == 0
    assert float(out[0, -1].abs().max()) == 0.0


def test_fused_zero_residual_requires_fp32_residual_not_bf16():
    """The R-ZERO-RESIDUAL boundary is fp32-sensitive (P1-6 motivation)."""
    cfg = make_cfg()
    v0 = torch.full((V,), 0.5004)
    v1 = v0 + 1e-6
    toks = [w(e(0), v0), w(e(0), v1), r(e(0))]
    tr, st, out, _ = _fused_case_run(toks, cfg, ckpt=1)
    assert [int(tr.case[0, t, 0]) for t in range(3)] == [R_EMPTY, R_ZERO_RESIDUAL, NO_WRITE]


# ---------------------------------------------------------------------------
# Gradient parity vs eager autograd.
# ---------------------------------------------------------------------------
_GRAD_NAMES = ["k", "q", "v", "beta", "pop", "alog", "aec", "s0k", "s0v", "s0s"]


def _grad_scenario(base, cfg, d_out, d_st):
    def run(scan_fn):
        ls = [x.detach().clone().requires_grad_(x.is_floating_point())
          for x in base]
        st = TapeState(keys=ls[7], values=ls[8], strength=ls[9],
                       live=base[10], age=base[11], usage=base[12])
        o, stf, _tr = scan_fn(ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], st, cfg,
                              append_err_coef=ls[6])
        loss = ((o * d_out).sum() + (stf.keys * d_st["keys"]).sum()
                + (stf.values * d_st["values"]).sum()
                + (stf.strength * d_st["strength"]).sum())
        # ls[:10] = the ten float grad targets (k..s0s); live/age/usage are
        # bookkeeping with no gradient by design (G13)
        return torch.autograd.grad(loss, ls[:10], allow_unused=True), o, stf, _tr
    return run


def _assert_grad_parity(ge, gf, bound):
    for name, a, b in zip(_GRAD_NAMES, ge, gf):
        if a is None:
            assert float(b.abs().sum()) == 0.0, (name, "eager unused, fused nonzero")
            continue
        err = max_rel(b, a)
        assert err < bound, (name, err, bound)


def test_gradient_parity_resident_fp32_checkpoints():
    """fp32 exact checkpoints: fused BPTT == eager autograd to ~1e-5."""
    torch.manual_seed(11)
    cfg = make_cfg(budget=16, n_heads=2, decay_rate=0.99)
    B, T, NH, S = 2, 9, 2, 16
    k = torch.randn(B, T, NH, K, device=DEV)
    q = torch.randn(B, T, NH, K, device=DEV)
    v = torch.randn(B, T, NH, V, device=DEV) * 0.5
    beta = torch.rand(B, T, NH, device=DEV)
    pop = torch.rand(B, T, NH, device=DEV)
    alog = torch.randn(B, T, NH, device=DEV)
    aec = torch.randn(NH, device=DEV)
    s0k = torch.randn(B, NH, S, K, device=DEV)
    s0v = torch.randn(B, NH, S, V, device=DEV)
    s0s = torch.rand(B, NH, S, device=DEV) * 0.8 + 0.2
    s0l = torch.rand(B, NH, S, device=DEV) < 0.5
    s0a = torch.zeros(B, NH, S, dtype=torch.int64, device=DEV)
    s0u = torch.zeros(B, NH, S, device=DEV)
    base = [k, q, v, beta, pop, alog, aec, s0k, s0v, s0s, s0l, s0a, s0u]
    d_out = torch.randn(B, T, NH, V, device=DEV)
    d_st = {"keys": torch.randn(B, NH, S, K, device=DEV),
            "values": torch.randn(B, NH, S, V, device=DEV) * 0.1,
            "strength": torch.randn(B, NH, S, device=DEV) * 0.1}
    runner = _grad_scenario(base, cfg, d_out, d_st)
    ge, oe, _, _ = runner(lambda *a, **kw: tape_scan(*a, **kw))
    gf, of, _, _ = runner(lambda *a, **kw: fused_tape_scan(
        *a, ckpt_interval=4, checkpoint_dtype=torch.float32, **kw))
    _assert_grad_parity(ge, gf, bound=1e-4)   # T=9, CKPT=4: ragged tail covered
    assert max_rel(of, oe) < 1e-4


def test_gradient_parity_spilled_bf16_checkpoints():
    """bf16 replay checkpoints: within the registered 0.08 gate bound."""
    torch.manual_seed(12)
    cfg = make_cfg(key_dim=16, value_dim=32, budget=100, n_heads=2,
                   decay_rate=0.999)
    K2, V2, B, T, NH, S = 16, 32, 2, 10, 2, 100
    k = torch.randn(B, T, NH, K2, device=DEV)
    q = torch.randn(B, T, NH, K2, device=DEV)
    v = torch.randn(B, T, NH, V2, device=DEV) * 0.5
    beta = torch.rand(B, T, NH, device=DEV)
    pop = torch.rand(B, T, NH, device=DEV)
    alog = torch.randn(B, T, NH, device=DEV)
    aec = torch.randn(NH, device=DEV)
    s0k = torch.randn(B, NH, S, K2, device=DEV)
    s0v = torch.randn(B, NH, S, V2, device=DEV)
    s0s = torch.rand(B, NH, S, device=DEV) * 0.8 + 0.2
    s0l = torch.rand(B, NH, S, device=DEV) < 0.5
    s0a = torch.zeros(B, NH, S, dtype=torch.int64, device=DEV)
    s0u = torch.zeros(B, NH, S, device=DEV)
    base = [k, q, v, beta, pop, alog, aec, s0k, s0v, s0s, s0l, s0a, s0u]
    d_out = torch.randn(B, T, NH, V2, device=DEV)
    d_st = {"keys": torch.randn(B, NH, S, K2, device=DEV),
            "values": torch.randn(B, NH, S, V2, device=DEV) * 0.1,
            "strength": torch.randn(B, NH, S, device=DEV) * 0.1}
    runner = _grad_scenario(base, cfg, d_out, d_st)
    ge, oe, _, _ = runner(lambda *a, **kw: tape_scan(*a, **kw))
    gf, of, _, _ = runner(lambda *a, **kw: fused_tape_scan(
        *a, ckpt_interval=3, checkpoint_dtype=torch.bfloat16,
        max_resident=64, tile_s=32, **kw))
    _assert_grad_parity(ge, gf, bound=0.08)
    assert max_rel(of, oe) < 0.03


def test_gradient_parity_pop_kills_target():
    """Engineered pop-kill adjoint path: pop strength 0.9 with
    free_threshold 0.3 kills every pop target at the post-pop death check
    (G1/G10), so the decay VJP must use multiplier 1 for the killed slot
    while the read VJP still adds its pre-pop contribution."""
    torch.manual_seed(9)
    cfg = make_cfg(budget=6, n_heads=1, decay_rate=0.9, free_threshold=0.3)
    B, T, S = 1, 5, 6
    k = torch.randn(B, T, 1, K, device=DEV)
    q = torch.randn(B, T, 1, K, device=DEV)
    v = torch.randn(B, T, 1, V, device=DEV) * 0.5
    beta = torch.rand(B, T, 1, device=DEV)
    pop = torch.full((B, T, 1), 0.9, device=DEV)
    alog = torch.randn(B, T, 1, device=DEV)
    aec = torch.randn(1, device=DEV)
    s0k = torch.randn(B, 1, S, K, device=DEV)
    s0v = torch.randn(B, 1, S, V, device=DEV)
    s0s = torch.rand(B, 1, S, device=DEV) * 0.6 + 0.2
    s0l = torch.ones(B, 1, S, dtype=torch.bool, device=DEV)
    base = [k, q, v, beta, pop, alog, aec, s0k, s0v, s0s,
            s0l, torch.zeros(B, 1, S, dtype=torch.int64, device=DEV),
            torch.zeros(B, 1, S, device=DEV)]
    d_out = torch.randn(B, T, 1, V, device=DEV)
    d_st = {"keys": torch.randn(B, 1, S, K, device=DEV),
            "values": torch.randn(B, 1, S, V, device=DEV) * 0.1,
            "strength": torch.randn(B, 1, S, device=DEV) * 0.1}
    runner = _grad_scenario(base, cfg, d_out, d_st)
    ge, _, _, _ = runner(lambda *a, **kw: tape_scan(*a, **kw))
    gf, _, _, tr = runner(lambda *a, **kw: fused_tape_scan(
        *a, ckpt_interval=2, checkpoint_dtype=torch.float32, **kw))
    # the construction really exercises the kill paths
    assert int((tr.freed == FREED_POP).sum()) >= 3
    assert int((tr.freed == FREED_DECAY).sum()) >= 1
    assert int((tr.alloc_target >= 0).sum()) >= 1
    _assert_grad_parity(ge, gf, bound=1e-4)


def test_final_token_write_is_gradient_dead_fused():
    """G8 receipt: read-before-write means the final token's write has no
    gradient path from the per-token read outputs."""
    cfg = make_cfg()
    toks = [w(e(0), 0.5 * torch.ones(V), alogit=1.0),
            w(e(1), 0.25 * torch.ones(V), alogit=1.0),
            dict(k=e(0), q=e(0), v=0.9 * torch.ones(V), beta=0.9, pop=0.0,
                 alogit=ALOGIT_FULL)]   # t2 is final: R-CONFLICT, unseen by reads
    k, q, v, beta, pop, alog = build_inputs(toks, requires_grad=True)
    k = k.to(DEV); q = q.to(DEV); v = v.to(DEV)
    beta = beta.to(DEV); pop = pop.to(DEV); alog = alog.to(DEV)
    s0 = init_state(1, cfg, DEV)
    out, st, tr = fused_tape_scan(k, q, v, beta, pop, alog, s0, cfg,
                                 ckpt_interval=2, checkpoint_dtype=torch.float32)
    out.sum().backward()

    def row(g):
        return float("inf") if g is None else float(g[0, 2].abs().sum())

    assert row(v.grad) in (float("inf"), 0.0)   # None (unused) or exact zero
    assert row(k.grad) in (float("inf"), 0.0)
    assert row(beta.grad) in (float("inf"), 0.0)
    assert row(alog.grad) in (float("inf"), 0.0)
    # ... but the final STATE carries the correction (a downstream loss sees
    # it): slot0 = (1-beta)*gate*v0 + beta*v2 with gate = sigmoid(1.0)
    a = float(torch.sigmoid(torch.tensor(1.0)))
    expected = (0.1 * (a * 0.5) + 0.9 * 0.9) * torch.ones(V, device=DEV)
    assert torch.allclose(st.values[0, 0, 0], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# G10: bf16 replay trace consistency (the replay-consistency gate core).
# ---------------------------------------------------------------------------
def _replay_gate(k, q, v, beta, pop, alog, s0, cfg, ckpt, ckpt_dtype,
                 aec=None, max_resident=64, tile_s=32):
    if aec is None:
        aec = torch.zeros(cfg.n_heads, device=k.device)
    buf = fused_fwd_raw(k, q, v, beta, pop, alog, s0, cfg, append_err_coef=aec,
                        ckpt_interval=ckpt, checkpoint_dtype=ckpt_dtype,
                        max_resident=max_resident, tile_s=tile_s)
    rp = fused_replay_raw(buf, k, q, v, beta, pop, alog, cfg,
                          append_err_coef=aec, ckpt_interval=ckpt, tile_s=tile_s)
    return buf, rp


def test_bf16_replay_trace_consistency_would_flip_free_by_decay():
    """Engineered construction: a slot's strength sits within bf16 rounding
    distance of free_threshold so recomputing the death decision from the
    ROUNDED checkpoint flips it; trace-following replay stays exact (G10)."""
    cfg = make_cfg(decay_rate=1.0, free_threshold=0.5, budget=4)
    # free_threshold=0.5, no decay: a slot at strength 0.5 + ulp/2 dies exactly
    # at the end-of-token check in fp32, but bf16 rounds 0.50000003 -> 0.5
    # (still <= threshold... need a value that rounds ABOVE):
    s_die = 0.5000019   # fp32; bf16 mantissa (8 bits) rounds to 0.50001526? no:
    # bf16 spacing at 0.5 is 2^-9 = 0.001953125; 0.5000019 -> nearest bf16 is
    # 0.5 exactly, so the ROUNDED strength is 0.5 <= 0.5 -> would ALSO free.
    # Use the other side: a slot just ABOVE the threshold in fp32 whose bf16
    # rounding falls to <= threshold (survives fp32, dies if recomputed).
    s_survive = 0.5009765625 + 0.0009   # 0.50188: fp32 survives (>0.5);
    # bf16 grid: ..., 0.5, 0.501953125, ... -> 0.50188 rounds to 0.501953125
    # (> 0.5, survives) -- instead pick a value that rounds DOWN to 0.5:
    s_flip = 0.5009                      # fp32 > 0.5 (survives); nearest bf16
    # grid points are 0.5 and 0.501953125; 0.5009 -> 0.5 (<= 0.5: DIES if the
    # decision is recomputed from the rounded state).
    s0 = init_state(1, cfg, DEV)
    s0.keys[0, 0, 0] = e(0)
    s0.values[0, 0, 0] = 0.5 * torch.ones(V, device=DEV)
    s0.strength[0, 0, 0] = s_flip
    s0.live[0, 0, 0] = True
    toks = [w(e(0), 0.5 * torch.ones(V))] + [r(e(0)) for _ in range(3)]
    k, q, v, beta, pop, alog = build_inputs(toks)
    k = k.to(DEV); q = q.to(DEV); v = v.to(DEV)
    beta = beta.to(DEV); pop = pop.to(DEV); alog = alog.to(DEV)

    # fp32 forward: the slot SURVIVES (> free_threshold), proven by the trace
    out_e, st_e, tr_e = tape_scan(k, q, v, beta, pop, alog, s0, cfg)
    assert bool(st_e.live[0, 0, 0])
    assert int(tr_e.freed.sum()) == 0

    # bf16 checkpoint forward: same fp32 live state; the CHECKPOINT strength
    # rounds below/at the threshold
    buf, rp = _replay_gate(k, q, v, beta, pop, alog, s0, cfg, ckpt=2,
                           ckpt_dtype=torch.bfloat16)
    rounded = buf.ck_strength[1, 0, 0, 0].float()
    assert float(rounded) <= cfg.free_threshold < float(s0.strength[0, 0, 0])
    # the forward liveness (fp32, in-register) still has the slot live:
    assert bool((buf.f_live[0, 0, 0] > 0).item())

    # replay follows the trace: boundary liveness/age EXACT at every boundary
    tr_f = buffers_to_trace(buf)
    for field_f, field_e in (("echo_case", buf.t_case), ("echo_matched", buf.t_matched),
                             ("echo_popt", buf.t_popt), ("echo_alloc", buf.t_alloc),
                             ("echo_popf", buf.t_popf)):
        assert torch.equal(rp[field_f], field_e), field_f
    for i in range(1, buf.num_ck):
        assert torch.equal(rp["bd_live"][i].bool(), buf.ck_live[i].bool()), i
        assert torch.equal(rp["bd_age"][i], buf.ck_age[i]), i
    # replayed read outputs within the bf16 tolerance
    rel = float(((rp["replay_out"] - buf.out).abs()
                 / buf.out.abs().clamp_min(1e-6)).max())
    assert rel < 0.03, rel
    # and the slot is STILL live in the replayed boundary (trace-following),
    # while recomputation from the rounded checkpoint would have freed it
    assert bool(rp["bd_live"][1, 0, 0, 0].item())
    assert bool(buf.ck_live[1, 0, 0, 0].item())


def test_bf16_replay_all_cases_at_boundaries():
    """Multi-checkpoint branch-equivalence: every routing case + pop + a
    free-by-decay event at checkpoint boundaries (fp32 and bf16 modes)."""
    # free_threshold raised so the 0.5 decay frees slots WITHIN the 8-token
    # window: free-by-decay fires at checkpoint boundaries (G10 coverage)
    cfg = make_cfg(budget=3, decay_rate=0.5, free_threshold=0.01)
    vv = 0.5 * torch.ones(V)
    toks = [
        w(e(0), vv),                                   # t0 R-EMPTY
        w(e(0), vv),                                   # t1 R-ZERO-RESIDUAL
        w(e(1), 2 * vv),                               # t2 R-NOMATCH
        w(e(0), 0.9 * torch.ones(V)),                 # t3 R-CONFLICT
        w(e(2), 3 * vv),                               # t4 R-CAP (full)
        dict(k=e(1), q=e(1), v=4.0 * torch.ones(V), beta=0.9, pop=1.0,
             alogit=ALOGIT_FULL),                      # t5 POP kills slot2 -> R-NOMATCH
        r(e(0), beta=0.1),                             # t6 NO-WRITE
        r(e(1), beta=0.1),                             # t7 (post-pop read)
    ]
    k, q, v, beta, pop, alog = build_inputs(toks)
    k = k.to(DEV); q = q.to(DEV); v = v.to(DEV)
    beta = beta.to(DEV); pop = pop.to(DEV); alog = alog.to(DEV)
    s0 = init_state(1, cfg, DEV)
    for ckpt_dtype in (torch.float32, torch.bfloat16):
        buf, rp = _replay_gate(k, q, v, beta, pop, alog, s0, cfg, ckpt=2,
                               ckpt_dtype=ckpt_dtype)
        tr = buffers_to_trace(buf)
        cases = {int(tr.case[0, t, 0]) for t in range(8)}
        assert cases == {R_EMPTY, R_ZERO_RESIDUAL, R_NOMATCH, R_CONFLICT,
                         R_CAP, NO_WRITE}
        assert bool(tr.pop_fired[0, 5, 0])
        assert int((tr.freed == FREED_POP).sum()) > 0
        assert int((tr.freed == FREED_DECAY).sum()) > 0   # decay 0.5 kills slots
        for field_f, field_e in (("echo_case", buf.t_case),
                                 ("echo_matched", buf.t_matched),
                                 ("echo_popt", buf.t_popt),
                                 ("echo_alloc", buf.t_alloc),
                                 ("echo_popf", buf.t_popf)):
            assert torch.equal(rp[field_f], field_e), (field_f, ckpt_dtype)
        for i in range(1, buf.num_ck):
            assert torch.equal(rp["bd_live"][i].bool(), buf.ck_live[i].bool())
            assert torch.equal(rp["bd_age"][i], buf.ck_age[i])
        rel = float(((rp["replay_out"] - buf.out).abs()
                     / buf.out.abs().clamp_min(1e-6)).max())
        assert rel < 0.03, (rel, ckpt_dtype)
        if ckpt_dtype == torch.float32:
            assert rel < 1e-5   # fp32 checkpoints: replay is essentially exact


# ---------------------------------------------------------------------------
# T2 decay-inertness at the placeholder constants (smoke-facing receipt).
# ---------------------------------------------------------------------------
def test_t2_horizon_at_g14_constants():
    cfg = TapeHeadConfig()   # G14 placeholder constants
    assert fade_horizon(cfg) >= 10 * 90_904
