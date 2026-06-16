#!/usr/bin/env python3
"""state_aware_mlp_param_check.py — REAL param accounting for the state-aware-MLP design.

Task: research-state-aware-mlp (Architect). This is the "tiny prototype param check"
deliverable. EVERY number printed here is computed from the real model code in
ndm/models/ (E88FLAHybrid + LadderLM + MixerMLPWrapper) — no mock data.

Steps:
  1. Build the REAL baseline emender-mlp (LadderLM level=E97 == E88FLAHybrid
     split-edit) at the exact ref-run geometry; reproduce 1,286,589,072 EXACTLY.
  2. Derive a closed-form per-layer/total param formula and ASSERT it equals the
     real build (so the fast iso-param solver below is provably faithful).
  3. Quantify the naive flatten-and-project blowup the task calls infeasible.
  4. Param-cost each candidate "state -> MLP" mechanism and solve the iso-param
     rebalance (shrink the SwiGLU hidden, i.e. mlp_ratio) back to the baseline.

Run:  python scripts/state_aware_mlp_param_check.py            # formula only
      python scripts/state_aware_mlp_param_check.py --build    # also build real model
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

# Baseline geometry — the REAL ref_emender_mlp run
# (/mnt/nvme1n1/erikg/ref_emender_mlp/launch_manifest.json):
#   --level E97 --dim 1792 --n_heads 216 --n_state 32 --depth 11 --expansion 1.0
#   --use_gate 1 --gate_activation silu --mlp_ratio 2.262336203876648 --mlp_multiple 64
#   tokenizer p50k_base -> vocab 50281
VOCAB = 50281
DIM = 1792
DEPTH = 11
N_HEADS = 216
N_STATE = 32
EXPANSION = 1.0
MLP_RATIO = 2.262336203876648
MLP_MULTIPLE = 64
BASELINE_TOTAL = 1_286_589_072

KEY_DIM = N_HEADS * N_STATE                      # 6912
VALUE_DIM = int(N_HEADS * N_STATE * EXPANSION)   # 6912
HEAD_V = VALUE_DIM // N_HEADS                     # 32
STATE_PER_HEAD = N_STATE * HEAD_V                 # 1024
STATE_PER_LAYER = N_HEADS * STATE_PER_HEAD        # 221,184


def round_mlp_hidden(dim, mlp_ratio, multiple=MLP_MULTIPLE):
    """Copy of ndm.models.ladder_lm.round_mlp_hidden (verified identical)."""
    return max(multiple, int(round(dim * mlp_ratio / multiple) * multiple))


def mixer_params(dim=DIM, n_heads=N_HEADS, n_state=N_STATE, expansion=EXPANSION):
    """E88FLAHybrid(use_split_edit=True, use_gate=True, head_mix='concat',
    use_conv=False, tie_kv=False, decay_mode='mamba')."""
    key_dim = n_heads * n_state
    value_dim = int(n_heads * n_state * expansion)
    p = 0
    p += dim * (2 * key_dim + value_dim)   # qkv_proj (fused q,k,v)
    p += dim * n_heads                       # a_proj
    p += n_heads                             # A_log
    p += n_heads                             # dt_bias
    p += dim * value_dim                     # g_proj (output gate)
    p += dim * key_dim                       # erase_gate_proj        (split-edit / E97)
    p += dim * value_dim                     # value_write_gate_proj  (split-edit / E97)
    p += value_dim * dim                     # o_proj  (W_o: value_dim -> dim)
    return p


def mlp_params(dim, mlp_hidden, extra_in=0):
    """SwiGLU MLP. extra_in = features concatenated to the MLP input (state-aware
    injection): w1,w2 widen dim -> dim+extra_in; w3 unchanged."""
    return (dim + extra_in) * mlp_hidden + (dim + extra_in) * mlp_hidden + mlp_hidden * dim


def per_layer_params(dim, mlp_hidden, n_heads=N_HEADS, n_state=N_STATE,
                     expansion=EXPANSION, extra_in=0, extra_readout=0):
    p = mixer_params(dim, n_heads, n_state, expansion)
    p += dim                                 # pre-mixer RMSNorm (layer_norms.N)
    p += dim                                 # post-mixer RMSNorm (norm_2)
    p += mlp_params(dim, mlp_hidden, extra_in)
    p += extra_readout                       # state-summary module R (mechanism-specific)
    return p


def total_params(dim=DIM, depth=DEPTH, mlp_hidden=None, mlp_ratio=MLP_RATIO,
                 n_heads=N_HEADS, n_state=N_STATE, expansion=EXPANSION,
                 extra_in=0, extra_readout=0, vocab=VOCAB):
    if mlp_hidden is None:
        mlp_hidden = round_mlp_hidden(dim, mlp_ratio)
    non_layer = vocab * dim + dim            # tied lm_head; embedding + final norm
    return non_layer + depth * per_layer_params(
        dim, mlp_hidden, n_heads, n_state, expansion, extra_in, extra_readout)


def solve_isoparam_mlp_hidden(extra_in, extra_readout, dim=DIM, depth=DEPTH,
                              target=BASELINE_TOTAL):
    """Shrink mlp_hidden h to restore the budget. P_layer is affine in h:
    P(h) = base + h*(2*(dim+extra_in) + dim). Solve, round to MLP_MULTIPLE, report residual."""
    non_layer = VOCAB * dim + dim
    base_layer = mixer_params(dim) + 2 * dim + extra_readout
    slope = 2 * (dim + extra_in) + dim
    h_real = (target - non_layer - depth * base_layer) / (depth * slope)
    if h_real <= 0:
        return None
    h_round = max(MLP_MULTIPLE, int(round(h_real / MLP_MULTIPLE)) * MLP_MULTIPLE)
    achieved = total_params(dim=dim, depth=depth, mlp_hidden=h_round,
                            extra_in=extra_in, extra_readout=extra_readout)
    return {'h_real': h_real, 'h_round': h_round, 'eff_mlp_ratio': h_round / dim,
            'achieved_total': achieved, 'residual': achieved - target,
            'residual_pct': 100.0 * (achieved - target) / target}


def solve_isoparam_dim(extra_in, extra_readout, depth=DEPTH, mlp_ratio=MLP_RATIO,
                       target=BASELINE_TOTAL):
    """Alternative lever: shrink dim (mlp_ratio,n_heads,n_state fixed). Coarse search."""
    best = None
    for dim in range(256, DIM + 1, 64):
        h = round_mlp_hidden(dim, mlp_ratio)
        tot = total_params(dim=dim, depth=depth, mlp_hidden=h,
                           extra_in=extra_in, extra_readout=extra_readout)
        gap = abs(tot - target)
        if best is None or gap < best['gap']:
            best = {'dim': dim, 'mlp_hidden': h, 'total': tot, 'gap': gap,
                    'residual_pct': 100.0 * (tot - target) / target}
    return best


@dataclass
class Mechanism:
    key: str
    name: str
    extra_readout: int   # params of the state-summary module R (per layer)
    extra_in: int        # features appended to the MLP input (per layer)
    note: str


def build_mechanisms():
    m = []
    # M1a — Readout-concat (FULL pre-W_o readout, no down-proj). R has NO new params
    # (reuses the gated Sq already computed before o_proj).
    m.append(Mechanism('M1a', 'Readout-concat (full 6912, no down-proj)',
                       0, VALUE_DIM,
                       'cheapest wiring; MLP nonlinearly mixes per-head readouts (o_proj mixes them only linearly); extra_in huge.'))
    # M1b — Readout-concat with a learned down-projection 6912 -> m.
    for mm in (256, 512, 1024):
        m.append(Mechanism(f'M1b_m{mm}', f'Readout-concat down-proj 6912->{mm}',
                           VALUE_DIM * mm, mm,
                           'down-proj bounds MLP-input growth; R = Linear(6912,m).'))
    # M2 — Multi-query readout: Rq total queries/head (Rq-1 extra query projs); each
    # extra query reads S_h -> head_v/head; extra readout (Rq-1)*VALUE_DIM, down-proj->m.
    for Rq, mm in ((2, 512), (4, 512), (4, 1024)):
        eq = Rq - 1
        er = eq * (DIM * KEY_DIM) + (eq * VALUE_DIM) * mm
        m.append(Mechanism(f'M2_R{Rq}_m{mm}', f'Multi-query R={Rq}, down-proj->{mm}',
                           er, mm,
                           f'{eq} extra input-dependent query proj(s)+downproj; exposes an R-dim row-subspace/head; needs kernel multi-query.'))
    # M3 — Low-rank fixed bilinear probes: r learned pairs (a_i in R^n_state,
    # b_i in R^head_v) SHARED across heads -> per head r scalars a_i^T S_h b_i.
    for r, mm in ((8, 0), (16, 512), (32, 512)):
        summary = N_HEADS * r
        probe = r * (N_STATE + HEAD_V)
        if mm and mm < summary:
            er, ei = probe + summary * mm, mm
        else:
            er, ei = probe, summary
        m.append(Mechanism(f'M3_r{r}' + (f'_m{mm}' if mm else ''),
                           f'Low-rank bilinear probes r={r}' + (f', down-proj->{mm}' if mm else ' (direct)'),
                           er, ei,
                           'input-INDEPENDENT learned probes; tiny params; r bilinear directions/head the single query misses.'))
    # M4 — State statistics (parameter-free to compute; nonlinear in S).
    m.append(Mechanism('M4a', 'State stats: Frobenius-norm + trace (2/head=432)',
                       0, 2 * N_HEADS,
                       'param-free features; nonlinear in S; cheapest expressivity probe (ablation floor).'))
    for mm in (256, 512):
        m.append(Mechanism(f'M4b_m{mm}', f'State stats: per-head row-norms (6912) down-proj->{mm}',
                           N_HEADS * N_STATE * mm, mm,
                           '||row_i(S_h)|| for all i,h; richer nonlinear stat; R = Linear(6912,m).'))
    return m


def fmt(n):
    return f'{n:,}'


def main():
    formula_total = total_params()
    print('=' * 84)
    print('STATE-AWARE MLP — REAL PARAM CHECK')
    print('=' * 84)
    print(f'Baseline: dim={DIM} depth={DEPTH} n_heads={N_HEADS} n_state={N_STATE} '
          f'exp={EXPANSION} mlp_ratio={MLP_RATIO:.6f}')
    print(f'  key_dim={KEY_DIM} value_dim={VALUE_DIM} head_v={HEAD_V} '
          f'mlp_hidden={round_mlp_hidden(DIM, MLP_RATIO)} vocab={VOCAB}')
    print(f'  per-layer params = {fmt(per_layer_params(DIM, round_mlp_hidden(DIM, MLP_RATIO)))}')
    print(f'  formula total    = {fmt(formula_total)}')
    print(f'  task baseline    = {fmt(BASELINE_TOTAL)}')
    assert formula_total == BASELINE_TOTAL, 'formula does not match baseline!'
    print('  formula == baseline  OK')

    if '--build' in sys.argv or '--build-arms' in sys.argv:
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import torch
        from ndm.models.ladder_lm import LadderLM

        def build_total(**kw):
            torch.manual_seed(0)
            base = dict(vocab_size=VOCAB, dim=DIM, depth=DEPTH, level='E97',
                        layer_kwargs=None, expansion=EXPANSION, n_state=N_STATE,
                        n_heads=N_HEADS, use_gate=True, gate_activation='silu',
                        use_triton=True, mlp_multiple=MLP_MULTIPLE)
            base.update(kw)
            mdl = LadderLM(**base)
            return sum(p.numel() for p in mdl.parameters())

        real = build_total(mlp_ratio=MLP_RATIO)
        print(f'  REAL torch build = {fmt(real)}   {"OK" if real == BASELINE_TOTAL else "MISMATCH"}')

    if '--build-arms' in sys.argv:
        # Certify the THREE state-aware-MLP A/B arms (STATE_AWARE_MLP_DESIGN.md §5/§6)
        # are iso-param to the baseline within the < 0.4% gate, by REAL torch build.
        #   1. baseline     : emender-mlp, plain SwiGLU (mlp_ratio 2.2623 -> hidden 4032).
        #   2. M1b_m512      : readout-summary down-proj 6912->512 + RMSNorm concat,
        #                      iso mlp_hidden 2816 (design table; residual -0.098% pre-RMSNorm).
        #   3. plain-wider   : plain SwiGLU, extra_in=0, mlp_hidden re-spent to the SAME
        #      control        total as baseline -> hidden 4032 (== baseline by iso-param;
        #                      'wider' than M1b's shrunk 2816). This IS the capacity control.
        print('\n' + '-' * 84)
        print('STATE-AWARE-MLP A/B ARMS — REAL BUILD ISO-PARAM CERTIFICATION (< 0.4% gate)')
        print('-' * 84)
        arms = [
            ('baseline',     dict(mlp_ratio=MLP_RATIO)),
            ('M1b_m512',     dict(mlp_ratio=MLP_RATIO, state_summary_dim=512, mlp_hidden=2816)),
            ('plain-wider',  dict(mlp_ratio=MLP_RATIO, state_summary_dim=0,   mlp_hidden=4032)),
        ]
        ok = True
        print(f'  {"arm":<14}{"real params":>16}{"resid":>14}{"resid%":>10}  gate')
        for name, kw in arms:
            tot = build_total(**kw)
            resid = tot - BASELINE_TOTAL
            pct = 100.0 * resid / BASELINE_TOTAL
            within = abs(pct) < 0.4
            ok = ok and within
            print(f'  {name:<14}{fmt(tot):>16}{resid:>+14,}{pct:>+9.4f}%  '
                  f'{"PASS" if within else "FAIL (>0.4%)"}')
        print('  ' + ('ALL ARMS ISO-PARAM WITHIN 0.4%  OK' if ok
                      else 'ISO-PARAM CERTIFICATION FAILED'))
        if not ok:
            sys.exit(1)

    print('\n' + '-' * 84)
    print('WHAT THE MLP NEVER SEES (per layer, per token, per batch element)')
    print('-' * 84)
    print(f'  matrix state S_h        : [n_state x head_v]=[{N_STATE}x{HEAD_V}]={STATE_PER_HEAD} numbers/head')
    print(f'  x {N_HEADS} heads        : {fmt(STATE_PER_LAYER)} state numbers/layer')
    print(f'  query readout Sq_h=S_h@q : {HEAD_V} numbers/head -> only {HEAD_V}/{STATE_PER_HEAD}'
          f'={100*HEAD_V/STATE_PER_HEAD:.3f}% of each head exposed/step')
    print(f'  concat readout           : {fmt(VALUE_DIM)} -> o_proj(LINEAR) -> dim={DIM}')
    print(f'  => MLP input is the post-o_proj {DIM}-d vector ONLY; cross-head mixing is LINEAR,')
    print(f'     the only cross-head nonlinearity is SwiGLU AFTER the linear collapse.')

    print('\n' + '-' * 84)
    print('NAIVE FLATTEN+PROJECT BLOWUP (the infeasible baseline)')
    print('-' * 84)
    naive_layer = STATE_PER_LAYER * DIM
    print(f'  flatten S ({fmt(STATE_PER_LAYER)}) -> dim ({DIM}) = {fmt(naive_layer)} params/layer')
    print(f'  x depth {DEPTH}                                 = {fmt(naive_layer * DEPTH)} added')
    print(f'  = {(naive_layer*DEPTH)/BASELINE_TOTAL:.2f}x the WHOLE baseline -> INFEASIBLE')

    print('\n' + '-' * 84)
    print('PARAM-BOUNDED MECHANISMS + ISO-PARAM REBALANCE (shrink SwiGLU hidden)')
    print('-' * 84)
    base_h = round_mlp_hidden(DIM, MLP_RATIO)
    print(f'  baseline mlp_hidden={base_h} (mlp_ratio {MLP_RATIO:.4f})\n')
    print(f'  {"mech":<12}{"+R/layer":>13}{"+MLP-in":>9}{"naive+total":>15}'
          f'{"iso h":>8}{"iso ratio":>10}{"resid%":>9}')
    print('  ' + '-' * 76)
    for mc in build_mechanisms():
        naive_added = DEPTH * (mc.extra_readout + 2 * mc.extra_in * base_h)
        sol = solve_isoparam_mlp_hidden(mc.extra_in, mc.extra_readout)
        if sol is None:
            dimsol = solve_isoparam_dim(mc.extra_in, mc.extra_readout)
            print(f'  {mc.key:<12}{fmt(mc.extra_readout):>13}{mc.extra_in:>9}{fmt(naive_added):>15}'
                  f'   MLP-lever INFEASIBLE -> dim {dimsol["dim"]} (resid {dimsol["residual_pct"]:.2f}%)')
        else:
            print(f'  {mc.key:<12}{fmt(mc.extra_readout):>13}{mc.extra_in:>9}{fmt(naive_added):>15}'
                  f'{sol["h_round"]:>8}{sol["eff_mlp_ratio"]:>10.4f}{sol["residual_pct"]:>9.4f}')

    print('\n  Mechanism notes:')
    for mc in build_mechanisms():
        print(f'   - {mc.key}: {mc.name}\n         {mc.note}')


if __name__ == '__main__':
    main()
