"""Traced unit tests for the X28E slot-store tape head (phase 1, M1).

Covers the registration's zero-update trace obligations ("traced runs
receipting every routing case firing correctly — empty-store bootstrap
(R-EMPTY), repeated identical writes (R-ZERO-RESIDUAL), novel bindings
(R-NOMATCH), conflicting corrections (R-CONFLICT), cap-pressure merge
(R-CAP), and pop-on-read — before any optimizer update"), plus:

  * the replay-decision consistency gate (P1-6): bf16-rounded-checkpoint
    replay follows the trace; replayed routing == forward trace at every
    checkpoint boundary, including a construction where recomputation from
    rounded state WOULD flip the routing case;
  * zero gradient through routing/pop/decay decisions (structural
    integrality + exact-zero gradient assertions) with positive gradient
    paths for keys/values/queries/strengths/gates;
  * the reverse-replay gradient skeleton vs direct autograd (fp32 exact
    checkpoints);
  * the T1/B0 arithmetic (K=64, V=128 -> B0=736 slots <= 1024) with the
    exact parameter count at the proposed placeholder dims, and the T2
    decay-inertness horizon calculation.

No optimizer updates, no training of any kind.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # x28e_memory_head/

from eng import bptt
from eng.bptt import _load_checkpoint
from eng.oracle import TapeLayer
from eng.routing_trace import (  # noqa: E402
    NO_SLOT, NO_WRITE, R_CAP, R_CONFLICT, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL,
    assert_traces_equal,
)
from eng.tape_head import (  # noqa: E402
    TapeHead, TapeHeadConfig, fade_horizon, head_parameter_count,
    init_state, layer_parameter_count, tape_scan,
)

K, V = 8, 16
ALOGIT_FULL = 20.0  # sigmoid(20) ~ 1 - 2e-9: append gate at (essentially) full strength
B0_STATE_CELLS = 141_312  # X27 small-GDN2 receipted state cells (T1)
DECAY = 0.999995


def make_cfg(**over) -> TapeHeadConfig:
    base = dict(key_dim=K, value_dim=V, n_heads=1, budget=8,
                match_threshold=0.5, eps_residual=1e-6, pop_threshold=0.5,
                write_enable_threshold=0.25, free_threshold=1e-3,
                decay_rate=DECAY, usage_decay=0.999)
    base.update(over)
    return TapeHeadConfig(**base)


def e(i, scale=4.0):
    """One-hot key/query vector of dim K, scaled so dot products are O(16)."""
    x = torch.zeros(K)
    x[i] = scale
    return x


def run_scan(tokens, cfg, state=None, trace=None, collect_trace=True):
    """tokens: list of dicts with k, q, v, beta, pop, alogit (all required)."""
    B, T, NH = 1, len(tokens), cfg.n_heads
    k = torch.stack([t["k"] for t in tokens]).view(B, T, NH, K)
    q = torch.stack([t["q"] for t in tokens]).view(B, T, NH, K)
    v = torch.stack([t["v"] for t in tokens]).view(B, T, NH, V)
    beta = torch.tensor([[t["beta"] for t in tokens]], dtype=torch.float32).view(B, T, NH)
    pop = torch.tensor([[t["pop"] for t in tokens]], dtype=torch.float32).view(B, T, NH)
    alogit = torch.tensor([[t["alogit"] for t in tokens]], dtype=torch.float32).view(B, T, NH)
    if state is None:
        state = init_state(B, cfg, k.device)
    return tape_scan(k, q, v, beta, pop, alogit, state, cfg, trace=trace,
                     collect_trace=collect_trace) + (k, q, v, beta, pop, alogit)


def w(k, v, beta=0.9, pop=0.0, alogit=ALOGIT_FULL, q=None):
    """A W-ON write token (write-enable above the placeholder threshold)."""
    return dict(k=k, q=(q if q is not None else e(7)), v=v, beta=beta, pop=pop, alogit=alogit)


def r(q, beta=0.1, pop=0.0, k=None, v=None, alogit=ALOGIT_FULL):
    """A W-OFF read token (write-enable below the placeholder threshold)."""
    return dict(k=(k if k is not None else e(6)), q=q, v=(v if v is not None else torch.zeros(V)),
                beta=beta, pop=pop, alogit=alogit)


def build_inputs(tokens, requires_grad=False):
    B, T, NH = 1, len(tokens), 1
    def leaf(x, shape_tail):
        t = x.view(B, T, NH, *shape_tail)
        return t.requires_grad_(True) if requires_grad else t
    k = leaf(torch.stack([t["k"] for t in tokens]), (K,))
    q = leaf(torch.stack([t["q"] for t in tokens]), (K,))
    v = leaf(torch.stack([t["v"] for t in tokens]), (V,))
    beta = leaf(torch.tensor([[t["beta"] for t in tokens]]), ())
    pop = leaf(torch.tensor([[t["pop"] for t in tokens]]), ())
    alogit = leaf(torch.tensor([[t["alogit"] for t in tokens]]), ())
    return k, q, v, beta, pop, alogit


# ---------------------------------------------------------------------------
# Routing cases (the zero-update trace obligations).
# ---------------------------------------------------------------------------
def test_r_empty_bootstrap():
    cfg = make_cfg()
    out, st, tr, *_ = run_scan([w(e(0), 0.7 * torch.ones(V), q=e(1))], cfg)
    assert int(tr.case[0, 0, 0]) == R_EMPTY
    assert int(tr.alloc_target[0, 0, 0]) == 0
    assert bool(tr.w_on[0, 0, 0]) and not bool(tr.no_write[0, 0, 0])
    # read-before-write on an empty store is exactly zero
    assert float(out[0, 0].abs().max()) == 0.0
    assert int(st.live_counts()[0, 0]) == 1
    assert torch.allclose(st.values[0, 0, 0], 0.7 * torch.ones(V), atol=1e-6)
    assert torch.allclose(st.keys[0, 0, 0], e(0))
    assert int(st.age[0, 0, 0]) == 1  # allocated this token, aged at end of token


def test_r_zero_residual_append():
    cfg = make_cfg()
    v0 = 0.5 * torch.ones(V)
    out, st, tr, *_ = run_scan([w(e(0), v0), w(e(0), v0)], cfg)
    # token 1: identical content, nothing to correct -> append (accumulates)
    assert int(tr.case[0, 1, 0]) == R_ZERO_RESIDUAL
    assert int(tr.matched[0, 1, 0]) == 0
    assert int(tr.alloc_target[0, 1, 0]) == 1
    assert int(st.live_counts()[0, 0]) == 2
    assert torch.allclose(st.values[0, 0, 1], v0, atol=1e-6)
    assert torch.allclose(st.values[0, 0, 0], v0, atol=1e-6)


def test_r_nomatch_append():
    cfg = make_cfg()
    # F12g: R-NOMATCH fires when NO age-1 (adjacency) slot exists — a W-OFF
    # read token advances the ages so the novel key is non-adjacent.
    out, st, tr, *_ = run_scan([w(e(0), 0.5 * torch.ones(V)),
                                r(e(0)),
                                w(e(1), 2.0 * torch.ones(V))], cfg)
    assert int(tr.case[0, 2, 0]) == R_NOMATCH
    assert int(tr.matched[0, 2, 0]) == NO_SLOT  # no adjacency slot to correct
    assert int(tr.alloc_target[0, 2, 0]) == 1
    assert int(st.live_counts()[0, 0]) == 2


def test_r_conflict_delta_correction():
    cfg = make_cfg()
    v0, v1 = 0.5 * torch.ones(V), 2.5 * torch.ones(V)
    out, st, tr, *_ = run_scan([w(e(0), v0), w(e(0), v1)], cfg)
    assert int(tr.case[0, 1, 0]) == R_CONFLICT
    assert int(tr.matched[0, 1, 0]) == 0
    assert int(tr.alloc_target[0, 1, 0]) == NO_SLOT  # corrections never allocate
    assert int(st.live_counts()[0, 0]) == 1
    beta = 0.9
    expected = (1 - beta) * v0 + beta * v1
    assert torch.allclose(st.values[0, 0, 0], expected, atol=1e-6)


def test_r_cap_merge():
    cfg = make_cfg(budget=2)
    toks = [
        w(e(0), 1.0 * torch.ones(V)),                      # R-EMPTY -> slot0
        r(e(0)),                                          # W-OFF: ages advance
        w(e(1), 2.0 * torch.ones(V)),                     # R-NOMATCH -> slot1
        r(e(1)),                                          # W-OFF: ages advance
        # novel key, non-adjacent: append requested, budget full -> R-CAP
        w(0.01 * e(0) + 0.08 * e(1), 5.0 * torch.ones(V)),
    ]
    out, st, tr, *_ = run_scan(toks, cfg)
    assert int(tr.case[0, 4, 0]) == R_CAP
    # F12g: the merge addresses the YOUNGEST live slot (slot1, written later)
    assert int(tr.matched[0, 4, 0]) == 1
    assert int(tr.alloc_target[0, 4, 0]) == NO_SLOT
    # delta-rule merge: slot1 = 2 + 0.9 * (5 - 2) = 4.7
    assert torch.allclose(st.values[0, 0, 1], torch.full((V,), 4.7), atol=1e-6)
    assert torch.allclose(st.values[0, 0, 0], torch.full((V,), 1.0), atol=1e-6)
    # R-ZERO-RESIDUAL append request while full ALSO degrades to R-CAP
    # (adjacent identical value against the age-1 merge slot; ages advanced by
    # the token above, so this token is again adjacent to slot1's correction)
    toks.append(r(e(1)))
    toks.append(w(e(1), st.values[0, 0, 1].clone()))
    out, st, tr, *_ = run_scan(toks, cfg)
    assert int(tr.case[0, 6, 0]) == R_CAP
    assert int(st.live_counts()[0, 0]) == 2    # no slot is ever allocated past B
    assert torch.allclose(st.values[0, 0, 1], torch.full((V,), 4.7), atol=1e-6)


def test_pop_on_read_full_kill_and_reuse():
    cfg = make_cfg()
    v0, v1 = 0.5 * torch.ones(V), 2.0 * torch.ones(V)
    toks = [
        w(e(0), v0, pop=0.0),                 # R-EMPTY -> slot0
        dict(k=e(1), q=e(0), v=v1, beta=0.9, pop=1.0, alogit=ALOGIT_FULL),
        # ^ read matches slot0; pop strength 1.0 > pop_threshold -> slot0 dead;
        #   write: novel key -> R-NOMATCH append, reusing the freed slot0
        r(e(0)),                              # W-OFF read: store holds only the new slot
    ]
    out, st, tr, *_ = run_scan(toks, cfg)
    # the pop kills slot0 BEFORE write routing, so the store is empty at write
    # time: the write routes R-EMPTY (dispatch order: READ/POP -> WRITE)
    assert int(tr.case[0, 1, 0]) == R_EMPTY
    assert bool(tr.pop_fired[0, 1, 0])
    assert int(tr.pop_target[0, 1, 0]) == 0
    assert int(tr.alloc_target[0, 1, 0]) == 0  # append reuses the just-popped slot
    assert int(st.live_counts()[0, 0]) == 1
    # the read after the pop sees the NEW slot content, never the popped value
    assert torch.allclose(out[0, 2], v1 * DECAY, atol=1e-4)
    assert not torch.allclose(out[0, 2], v0, atol=1e-2)


def test_pop_target_separate_from_write_target_and_partial_pop():
    cfg = make_cfg()
    v0, v1, v2 = 0.5 * torch.ones(V), 2.0 * torch.ones(V), 3.0 * torch.ones(V)
    toks = [
        w(e(0), v0),                                   # R-EMPTY -> slot0 (key e0)
        r(e(0)),                                      # W-OFF: ages advance
        w(e(1), v1),                                    # R-NOMATCH -> slot1 (key e1)
        # read matches slot0 (query e0) -> POP target 0; post-pop the age-1
        # slot is dead -> R-NOMATCH append into the freed slot: the POP
        # target (slot0) and the WRITE target (the reused slot) differ.
        dict(k=e(1), q=e(0), v=v2, beta=0.9, pop=1.0, alogit=ALOGIT_FULL),
        r(e(0)),                                       # only slot1 remains live
    ]
    out, st, tr, *_ = run_scan(toks, cfg)
    # F12g: the pop target (read-path argmax, slot0) and the correction
    # target (the age-1 slot written at t2, slot1) are SEPARATE by
    # construction; the adjacency correction fires on slot1.
    assert int(tr.case[0, 3, 0]) == R_CONFLICT
    assert int(tr.matched[0, 3, 0]) == 1               # correction target slot1
    assert int(tr.pop_target[0, 3, 0]) == 0            # pop target slot0 (SEPARATE)
    assert int(st.live_counts()[0, 0]) == 1             # slot0 dead, slot1 live
    # partial pop (0.6) does not kill: strength 0.4, slot remains live and fades
    toks2 = [w(e(0), v0),
             dict(k=e(7), q=e(0), v=torch.zeros(V), beta=0.1, pop=0.6, alogit=ALOGIT_FULL),
             r(e(0))]
    out2, st2, tr2, *_ = run_scan(toks2, cfg)
    assert bool(tr2.pop_fired[0, 1, 0]) and int(tr2.pop_target[0, 1, 0]) == 0
    assert int(st2.live_counts()[0, 0]) == 1           # still live
    # end of t2: strength = 1 * (1 - 0.6) decayed twice
    assert torch.allclose(st2.strength[0, 0, 0], torch.tensor(0.4 * DECAY ** 2), atol=1e-9)
    # faded read: single live slot gets softmax prob 1, weighted by strength
    expected = (0.4 * DECAY ** 2) * v0
    assert torch.allclose(out2[0, 2], expected, atol=1e-5)


def test_no_write_token():
    cfg = make_cfg()
    out, st, tr, *_ = run_scan([w(e(0), 0.5 * torch.ones(V))], cfg)
    keys0, values0, live0 = st.keys.clone(), st.values.clone(), st.live.clone()
    # W-OFF token: no write at all (NO-WRITE); keys/values/liveness unchanged
    out, st2, tr2, *_ = run_scan([r(e(0), beta=0.1)], cfg, state=st)
    assert int(tr2.case[0, 0, 0]) == NO_WRITE
    assert bool(tr2.no_write[0, 0, 0]) and not bool(tr2.w_on[0, 0, 0])
    assert int(tr2.alloc_target[0, 0, 0]) == NO_SLOT
    assert torch.equal(st2.keys, keys0)
    assert torch.equal(st2.values, values0)
    assert torch.equal(st2.live, live0)
    # decay/age still run (end-of-token bookkeeping)
    assert int(st2.age[0, 0, 0]) == 2
    assert torch.allclose(st2.strength[0, 0, 0], torch.tensor(DECAY ** 2), atol=1e-9)


def test_decay_frees_slot_and_t2_horizon():
    # architectural decay with an exaggerated rate to observe freeing quickly
    cfg = make_cfg(decay_rate=0.5)
    toks = [w(e(0), 0.5 * torch.ones(V))] + [r(e(0)) for _ in range(10)]
    out, st, tr, *_ = run_scan(toks, cfg)
    # strength 0.5^n <= free_threshold=1e-3 at n=10 decays -> freed
    assert int(st.live_counts()[0, 0]) == 0
    assert float(out[0, -1].abs().max()) == 0.0
    # T2 inertness calculation with the registered default constants:
    # fade horizon >= 10x the longest registered input (90,904-token smoke)
    default_cfg = TapeHeadConfig()
    horizon = fade_horizon(default_cfg)
    assert horizon >= 10 * 90_904, horizon
    # and the accelerated config matches the same formula
    assert abs(fade_horizon(cfg) - math.log(1e-3) / math.log(0.5)) < 1e-9


# ---------------------------------------------------------------------------
# Gradient discipline: routing is architectural, always.
# ---------------------------------------------------------------------------
def _grad_scenario():
    """7-token F12g scenario: append; W-OFF+partial-pop; W-OFF read;
    nomatch-append; the ADJACENCY correction (age-1 slot from the previous
    token) that a later token reads (so it carries gradient); reads. The
    append gate uses a NON-saturated logit (sigmoid(20) rounds to exactly
    1.0 in fp32, which would zero the append-gate gradient)."""
    cfg = make_cfg()
    toks = [
        w(e(0), 0.5 * torch.ones(V), pop=0.1, alogit=1.0),   # t0: R-EMPTY append slot0
        dict(k=e(5), q=e(0), v=torch.zeros(V), beta=0.1, pop=0.6, alogit=ALOGIT_FULL),
        # ^ t1: NO-WRITE + partial pop of slot0 (read argmax by query)
        r(e(0)),                                        # t2: W-OFF read (ages advance)
        w(e(1), 0.25 * torch.ones(V), pop=0.1, alogit=1.0),  # t3: R-NOMATCH append slot1
        dict(k=e(1), q=e(1), v=0.9 * torch.ones(V), beta=0.9, pop=0.0, alogit=ALOGIT_FULL),
        # ^ t4: R-CONFLICT — the ADJACENCY correction of slot1 (age 1)
        r(e(1)),                                        # t5: W-OFF read (sees the correction)
        r(e(0)),                                        # t6: W-OFF read
    ]
    return cfg, toks


def test_zero_gradient_through_routing():
    cfg, toks = _grad_scenario()
    k, q, v, beta, pop, alogit = build_inputs(toks, requires_grad=True)
    state = init_state(1, cfg, k.device)
    out, st, tr = tape_scan(k, q, v, beta, pop, alogit, state, cfg)

    # structural: every routing decision is integral and gradient-free
    tr.assert_integral()
    assert tr.case.dtype == torch.int64

    # sanity: every token routed as designed
    cases = [int(tr.case[0, t, 0]) for t in range(7)]
    assert cases == [R_EMPTY, NO_WRITE, NO_WRITE, R_NOMATCH,
                     R_CONFLICT, NO_WRITE, NO_WRITE], cases
    assert bool(tr.pop_fired[0, 1, 0]) and int(tr.pop_target[0, 1, 0]) == 0

    wread = torch.randn_like(out)
    (out * wread).sum().backward()

    # ---- zero gradient through routing decisions (EXACT zeros) ----
    # t1/t2/t5/t6 are W-OFF: their write keys/values and write/append gates
    # have NO differentiable path (routing consumed them only detached).
    for t in (1, 2, 5, 6):
        assert float(v.grad[0, t].abs().sum()) == 0.0, t
        assert float(k.grad[0, t].abs().sum()) == 0.0, t
        assert float(beta.grad[0, t].abs().sum()) == 0.0, t
        assert float(alogit.grad[0, t].abs().sum()) == 0.0, t
    # pop did not fire at t0/t3/t4/t5/t6 -> zero pop-strength gradient there
    # (t1's PARTIAL pop fires: its strength decrement is attached by design)
    for t in (0, 3, 4, 5, 6):
        assert float(pop.grad[0, t].abs().sum()) == 0.0, t
    assert float(pop.grad[0, 1].abs().sum()) > 0.0    # the partial pop carried
    # the R-CONFLICT token's key is routing-only (corrections never touch keys)
    assert float(k.grad[0, 4].abs().sum()) == 0.0

    # ---- positive gradient paths: keys/values/queries/strengths/gates ----
    assert float(k.grad[0, 0].abs().sum()) > 0.0     # appended key in the read softmax
    assert float(k.grad[0, 3].abs().sum()) > 0.0     # appended key in the read softmax
    assert float(v.grad[0, 0].abs().sum()) > 0.0     # appended value read later
    assert float(alogit.grad[0, 0].abs().sum()) > 0.0   # append gate scales stored content
    assert float(q.grad[0, 5].abs().sum()) > 0.0     # query in the 2-slot read mixture
    assert float(pop.grad[0, 1].abs().sum()) > 0.0  # pop magnitude via slot strength
    assert float(beta.grad[0, 4].abs().sum()) > 0.0  # correction magnitude, read at t5
    assert float(v.grad[0, 4].abs().sum()) > 0.0     # correction target value


def test_final_token_write_is_gradient_dead_without_later_read():
    """Read-before-write (registered order) means a token's own write is only
    visible to LATER tokens; the FINAL token's write has no gradient path from
    the per-token read outputs alone (FLAGGED spec consequence: end-of-word
    supervision reads pre-write state)."""
    cfg, toks = _grad_scenario()
    k, q, v, beta, pop, alogit = build_inputs(toks[:5], requires_grad=True)
    state = init_state(1, cfg, k.device)
    out, st, tr = tape_scan(k, q, v, beta, pop, alogit, state, cfg)
    out.sum().backward()
    # t4 is final: its write is unseen -> the write gate receives NO gradient
    # at all, and the correction's value does not
    assert beta.grad is None or float(beta.grad[0, 4].abs().sum()) == 0.0
    assert float(v.grad[0, 4].abs().sum()) == 0.0
    # ... but the final STATE carries the correction (a downstream loss sees
    # it): the t4 correction targeted slot1 (the age-1 slot written at t3)
    a = float(torch.sigmoid(torch.tensor(1.0)))
    v_best = a * 0.25 * torch.ones(V)     # slot1's appended value (gate a, v=0.25)
    expected = v_best + 0.9 * (0.9 * torch.ones(V) - v_best)
    assert torch.allclose(st.values[0, 0, 1], expected, atol=1e-6)
    # slot0 untouched by the correction
    assert torch.allclose(st.values[0, 0, 0], a * 0.5 * torch.ones(V), atol=1e-6)


def test_replay_gradients_match_direct_autograd():
    """Reverse-replay BPTT skeleton (fp32 exact checkpoints) == full autograd."""
    cfg, toks = _grad_scenario()
    k, q, v, beta, pop, alogit = build_inputs(toks, requires_grad=True)
    state0 = init_state(1, cfg, k.device)

    # direct full-graph autograd
    out, st, tr = tape_scan(k, q, v, beta, pop, alogit, state0, cfg)
    d_out = torch.randn_like(out)
    direct = torch.autograd.grad(outputs=[out], grad_outputs=[d_out],
                                inputs=[k, q, v, beta, pop, alogit], allow_unused=True)

    # checkpointed forward + reverse-replay gradients (fp32 exact checkpoints)
    _, _, trace, ckpts = bptt.checkpoint_forward(
        k, q, v, beta, pop, alogit, state0, cfg, ckpt_interval=2,
        checkpoint_dtype=torch.float32)
    replayed = bptt.replay_gradients(
        k, q, v, beta, pop, alogit, ckpts, cfg, ckpt_interval=2, d_out=d_out)

    names = ["k", "q", "v", "write_strength", "pop_strength", "append_logit"]
    for name, g_direct in zip(names, direct):
        g_replay = replayed[name]
        if g_direct is None:
            assert float(g_replay.abs().sum()) == 0.0, name
            continue
        diff = (g_replay - g_direct).abs()
        scale = g_direct.abs().max().clamp_min(1e-8)
        assert float(diff.max() / scale) < 1e-5, (name, float(diff.max()), float(scale))


# ---------------------------------------------------------------------------
# P1-6 replay-decision consistency: trace-following survives bf16 rounding.
# ---------------------------------------------------------------------------
def test_bf16_replay_trace_consistency_and_rounding_flip():
    """Replay follows the trace; recomputation from bf16-rounded state WOULD
    flip R-ZERO-RESIDUAL to R-CONFLICT, and the replayed routing still equals
    the forward trace at every checkpoint boundary."""
    cfg = make_cfg()
    v0 = torch.full((V,), 0.5004)
    v1 = v0 + 1e-6
    toks = [w(e(0), v0),                       # t0: R-EMPTY append slot0
            w(e(0), v1),                       # t1: fp32 residual ~1.6e-11 < eps -> R-ZERO-RESIDUAL append
            r(e(0))]                           # t2: W-OFF read of slot0
    k, q, v, beta, pop, alogit = build_inputs(toks)
    state0 = init_state(1, cfg, k.device)

    out_fwd, st_fwd, trace_fwd, ckpts = bptt.checkpoint_forward(
        k, q, v, beta, pop, alogit, state0, cfg, ckpt_interval=1,
        checkpoint_dtype=torch.bfloat16)
    assert [int(trace_fwd.case[0, t, 0]) for t in range(3)] == [R_EMPTY, R_ZERO_RESIDUAL, NO_WRITE]

    # the would-flip: bf16 rounds the stored slot0 value 0.5004 -> 0.5, so the
    # residual recomputed from the ROUNDED checkpoint crosses eps_residual.
    rounded_slot0 = ckpts[1]["values"][0, 0, 0].to(torch.float32)   # state after t0
    fp32_residual = float(((v[0, 1, 0] - st_fwd.values[0, 0, 0].detach()) ** 2).sum())
    rounded_residual = float(((v[0, 1, 0] - rounded_slot0) ** 2).sum())
    assert fp32_residual < cfg.eps_residual < rounded_residual, (fp32_residual, rounded_residual)

    # recomputation-from-rounded-state (routing recomputed) WOULD flip the case
    _, _, tr_recomputed = tape_scan(
        k[:, 1:2], q[:, 1:2], v[:, 1:2], beta[:, 1:2], pop[:, 1:2], alogit[:, 1:2],
        _load_checkpoint(ckpts[1]), cfg, trace=None)
    assert int(tr_recomputed.case[0, 0, 0]) == R_CONFLICT

    # trace-following replay: replayed routing == forward trace at EVERY boundary
    out_rp, trace_rp, reports = bptt.replay_forward(
        k, q, v, beta, pop, alogit, ckpts, cfg, ckpt_interval=1)
    assert_traces_equal(trace_fwd, trace_rp, what="bf16 trace-following replay")
    assert all(rep["live_equal"] and rep["age_equal"] for rep in reports)
    for rep in reports:
        assert rep["values_max_rel"] < 0.03, rep
        assert rep["keys_max_rel"] < 0.03, rep
        assert rep["strength_max_rel"] < 0.03, rep
    out_diff = (out_rp - out_fwd).abs() / out_fwd.abs().clamp_min(1e-6)
    assert float(out_diff.max()) < 0.03, float(out_diff.max())

    # full-gate helper agrees
    gate = bptt.replay_consistency(k, q, v, beta, pop, alogit, state0, cfg,
                                   ckpt_interval=1, checkpoint_dtype=torch.bfloat16)
    assert gate["traces_equal"] and gate["live_and_age_exact"]
    assert gate["output_max_rel"] < 0.03


def test_fp32_replay_consistency_all_cases():
    """The multi-checkpoint branch-equivalence gate over every routing case."""
    cfg = make_cfg(budget=3)
    v = 0.5 * torch.ones(V)
    toks = [
        w(e(0), v),                                    # t0: R-EMPTY (slot0)
        w(e(0), v),                                    # t1: R-ZERO-RESIDUAL (slot1)
        r(e(0), beta=0.1),                             # t2: NO-WRITE (ages advance)
        w(e(1), 2 * v),                                # t3: R-NOMATCH (slot2)
        w(e(1), 0.9 * torch.ones(V)),                  # t4: R-CONFLICT (adjacent slot2)
        r(e(0), beta=0.1),                             # t5: NO-WRITE (ages advance)
        w(e(2), 3 * v),                                # t6: full -> R-CAP (merge slot2)
        dict(k=e(2), q=e(1), v=4.0 * torch.ones(V), beta=0.9, pop=1.0, alogit=ALOGIT_FULL),
        # ^ t7: read argmax = slot2 (key e1); POP fired kills it. t6 was
        # R-CAP (a merge, not an append): no age-1 slot here -> append
        # requested; post-pop slot2 freed -> the append reuses it
        r(e(0), beta=0.1),                             # t8: NO-WRITE
    ]
    k, q, vv, beta, pop, alogit = build_inputs(toks)
    state0 = init_state(1, cfg, k.device)

    out_fwd, st, trace_fwd, ckpts = bptt.checkpoint_forward(
        k, q, vv, beta, pop, alogit, state0, cfg, ckpt_interval=3,
        checkpoint_dtype=torch.float32)
    cases = {int(trace_fwd.case[0, t, 0]) for t in range(9)}
    assert cases == {R_EMPTY, R_ZERO_RESIDUAL, R_NOMATCH, R_CONFLICT, R_CAP, NO_WRITE}
    assert bool(trace_fwd.pop_fired[0, 7, 0]) and int(trace_fwd.pop_target[0, 7, 0]) == 2
    # both R-CAP provenances: R-NOMATCH-requested and R-ZERO-requested... (t4 is
    # NOMATCH-requested; the ZERO-requested cap is covered in test_r_cap_merge)

    gate = bptt.replay_consistency(k, q, vv, beta, pop, alogit, state0, cfg,
                                   ckpt_interval=3, checkpoint_dtype=torch.float32)
    assert gate["traces_equal"] and gate["live_and_age_exact"]
    assert gate["output_max_abs"] == 0.0  # fp32 exact checkpoints: bitwise replay
    for rep in gate["boundary_reports"]:
        assert rep["values_max_abs"] == 0.0 and rep["keys_max_abs"] == 0.0


# ---------------------------------------------------------------------------
# T1/B0 arithmetic + parameter count + module smoke.
# ---------------------------------------------------------------------------
def test_t1_b0_arithmetic_and_param_count():
    # K+V >= 138 required for B0 <= 1024 (registration T1/B0 arithmetic)
    Kd, Vd = 64, 128
    assert Kd + Vd >= 138
    b0 = math.ceil(B0_STATE_CELLS / (Kd + Vd))
    assert b0 == 736, b0
    assert (Kd + Vd) * b0 == B0_STATE_CELLS      # exact: no rounding residue
    assert b0 <= 1024

    # exact parameter count at the proposed dims; d_model/NH are
    # TO-BE-BOUND-AT-FREEZE placeholders (flagged: the registration does not
    # bind the input width or head multiplicity).
    d_model, NH = 256, 4
    cfg = TapeHeadConfig(key_dim=Kd, value_dim=Vd, n_heads=NH, budget=b0)
    torch.manual_seed(0)   # seed BEFORE construction: init is deterministic
    head = TapeHead(d_model, cfg)
    n_head = sum(p.numel() for p in head.parameters())
    assert n_head == head_parameter_count(d_model, cfg) == 266_256, n_head
    layer = TapeLayer(d_model, cfg)
    n_layer = sum(p.numel() for p in layer.parameters())
    assert n_layer == layer_parameter_count(d_model, cfg) == 397_584, n_layer

    # module smoke: traced forward through the full layer (no optimizer update)
    x = torch.randn(2, 6, d_model)
    y, read_out, st, tr = layer(x)
    assert y.shape == (2, 6, d_model) and read_out.shape == (2, 6, NH, Vd)
    tr.assert_integral()
    assert tr.case.shape == (2, 6, NH)
    assert int(st.live.shape[2]) == b0
    assert int(st.live_counts().sum()) > 0  # writes actually fired
    # gradient flows to the learned projections. F12 EXCEPTIONS: W_k is the
    # FROZEN key geometry (stored and match keys detached — the read path
    # cannot collapse the addressing space) and W_write is the ARCHITECTURAL
    # write gate (detached; zero learned management) — both gradients are
    # expected ABSENT, exactly like W_pop on the pop-disabled arm (F10).
    y.sum().backward()
    for pname, param in layer.named_parameters():
        if pname.endswith(("W_k.weight", "W_k.bias",
                           "W_write.weight", "W_write.bias",
                           "W_pop.weight", "W_pop.bias")):
            # F12: frozen/architectural parameters receive NO EFFECTIVE
            # gradient (None, or exact zeros when the graph touches them
            # through a zero-masked factor, e.g. an unfired pop).
            assert param.grad is None \
                or float(param.grad.abs().sum()) == 0.0, \
                f"F12 frozen/architectural parameter must not receive " \
                f"gradient ({pname})"
            continue
        assert param.grad is not None, f"no gradient reached {pname}"
        assert float(param.grad.abs().sum()) > 0.0, f"zero gradient at {pname}"


def test_trace_covers_every_case_and_summarizes():
    cfg = make_cfg(budget=3)
    v = 0.5 * torch.ones(V)
    # F12g adjacency routing: R-ZERO-RESIDUAL needs an ADJACENT identical
    # value; R-NOMATCH needs a non-adjacent novel key; R-CAP needs a full
    # budget at a non-adjacent append request.
    toks = [w(e(0), v),          # t0: R-EMPTY -> slot0
            w(e(0), v),          # t1: adjacency, residual 0 -> R-ZERO-RESIDUAL slot1
            r(e(0)),             # t2: W-OFF (ages advance)
            w(e(1), 2 * v),      # t3: non-adjacent -> R-NOMATCH slot2
            w(e(1), 3 * v),      # t4: adjacency correction -> R-CONFLICT slot2
            r(e(0)),             # t5: W-OFF (ages advance)
            w(e(2), 4 * v)]      # t6: append requested, budget full -> R-CAP
    out, st, tr, *_ = run_scan(toks, cfg)
    counts = tr.counts()
    assert counts["R-EMPTY"] == 1
    assert counts["R-ZERO-RESIDUAL"] == 1
    assert counts["R-NOMATCH"] == 1
    assert counts["R-CONFLICT"] == 1
    assert counts["R-CAP"] == 1
    text = tr.summarize()
    assert "R-EMPTY" in text and "POP fired" in text
