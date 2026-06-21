# State-Aware MLP — M1 implementation + cheap falsification (results)

**Task:** `improve-mlp-integration` (Programmer). Implements **M1** (state→MLP head-mixing)
per `paper/review/STATE_AWARE_MLP_DESIGN.md` §5, certifies iso-param, and runs the cheap
grok-cell falsification FIRST (small scale) before any 1.3B run, per the task decision rule.

**NON-NEGOTIABLE #1 honored:** M1 consumes ONLY the fused kernel's existing pre-`o_proj`
readout + standard Linear/SwiGLU (cuBLAS). It touches **no** recurrence dynamics, so the E97
state update stays on the **fused Triton kernel** in every arm (`[fused-guard]` asserted by
`train_grok.assert_kernel`; `m1_smoke.py` confirms grads flow and no eager fallback). **M3/M4
were NOT built here** (they require in-kernel work + backward VJP).

---

## 1. What M1 does (mechanism, not capacity)

Baseline: the SwiGLU MLP sees only the post-`o_proj` `dim`-vector; cross-head mixing is the
**linear** `o_proj`, and the only cross-head nonlinearity (SwiGLU) acts *after* that linear
collapse. **M1b** down-projects the SAME pre-`o_proj` per-head readout concat
(`e88_fla_hybrid.py:1933`, the `[B,T,value_dim]` tensor `o_proj` consumes) to `m` dims +
RMSNorm, and concatenates it to the MLP input — letting SwiGLU form **nonlinear cross-head
features before** the linear collapse. It exposes **no new information** about the matrix
state `S` (same numbers `o_proj` already mixes); the lever is *pre-collapse nonlinearity*.

Wiring (committed): `E88FLAHybrid.readout_summary_dim` (concat head_mix only → returns a
3-tuple), `SwiGLUMLP.extra_in`, `MixerMLPWrapper` (LadderLM / 1.3B path) and `HybridLadderLM`
(grok path) thread + concat the summary; `LadderLM`/`HybridLadderLM` gain `state_summary_dim`
+ exact `mlp_hidden` override for iso-param arms.

## 2. Iso-param certification (REAL build)

`python scripts/state_aware_mlp_param_check.py --build-arms` (1.3B emender geometry):

| arm | real params | residual | gate |
|---|---|---|---|
| baseline (mlp_ratio 2.2623 → hidden 4032) | 1,286,589,072 | +0.0000% | PASS |
| **M1b_m512** (R 6912→512 + RMSNorm; iso hidden 2816) | 1,285,333,136 | **−0.0976%** | PASS |
| plain-wider control (extra_in 0; iso hidden 4032) | 1,286,589,072 | +0.0000% | PASS |

**The plain-wider control coincides with the baseline at iso-param** (proven by the build:
the plain-SwiGLU allocation that re-spends M1b's mechanism budget on width lands at hidden
4032 = baseline). So at iso-param, the emender baseline **is** the capacity-matched
plain-wider control; "beat BOTH" reduces to "beat the plain-MLP baseline." The grok probe
keeps an independent `control` arm (disjoint seeds) as an extra plain-MLP capacity estimate.

Grok-cell arms (dim 256, depth 2, n_state 32, n_heads 8 → value_dim 256; `m`=128): baseline
2,508,064 / m1b 2,508,320 (**+0.0102%**) / control 2,508,064 — iso within 0.02%.

## 3. Cheap falsification protocol (grok cell, FUSED E97)

`experiments/grok_symmetric_width/orchestrate_m1.py`: 3 arms × 3 seeds × p∈{48,256} on
`modular_quadratic` (mod 48 = design §6 spec; p256 = high-p separator regime with headroom),
train `--seq_len 128`, fresh length-extrap eval at T∈{128,256,512,1024,2048,4096} (Délétang
16× protocol). All arms = fused E97 (`--arm e97`, `use_triton`, `[fused-guard]`), differ ONLY
in the per-block MLP (iso-config separator). Each run logs in-dist test acc (T=128 held-out
pool) + extrap acc/T. Aggregator: `aggregate_m1.py`.

## 4. RESULTS (measured — 18 runs, fused E97, all complete, `aggregate_m1.py`)

Mean ± std over 3 seeds (in-dist test acc at T=128 held-out pool; fresh length-extrap acc):

**p=48** (random baseline 0.0208):

| metric | baseline | m1b | control |
|---|---|---|---|
| test_acc (T=128) | **1.000 ± 0.000** | 0.672 ± 0.284 | 0.838 ± 0.280 |
| extrap T=2048 | **0.998 ± 0.003** | 0.661 ± 0.293 | 0.833 ± 0.290 |
| extrap T=4096 | **0.997 ± 0.005** | 0.660 ± 0.294 | 0.829 ± 0.292 |

**p=256** (random baseline 0.0039):

| metric | baseline | m1b | control |
|---|---|---|---|
| test_acc (T=128) | **0.775 ± 0.299** | 0.600 ± 0.288 | 0.606 ± 0.267 |
| extrap T=2048 | **0.768 ± 0.307** | 0.577 ± 0.296 | 0.590 ± 0.281 |
| extrap T=4096 | **0.767 ± 0.309** | 0.579 ± 0.300 | 0.586 ± 0.283 |

**Grokking is bimodal** (per-seed: a seed either groks to the ceiling or stays at the
memorization floor ~0.43–0.51). The arm changes only the grok *probability*, never the
ceiling — when ANY arm groks it reaches the SAME ceiling (≈1.0 at p48, ≈0.93–0.95 at p256)
and that accuracy is **length-invariant to T=4096** (grok ⇒ perfect extrap; no-grok ⇒ flat at
floor). So there is no far-T separation between arms either.

Per-seed grok counts (seed reaches the ceiling):

| | baseline | control | m1b | pooled plain-MLP | m1b |
|---|---|---|---|---|---|
| p=48 | 3/3 | 2/3 | **1/3** | 5/6 | **1/3** |
| p=256 | 2/3 | 1/3 | **1/3** | 3/6 | **1/3** |

m1b's grok rate (2/6 across both p) is **≤** the plain-MLP grok rate (8/12) at every p. The
grok ceiling, the extrap profile, and the grok *value* are all arm-independent; the only
thing M1b changes is a modest **reduction** in grok reliability — consistent with paying for
the readout-summary `R` with a −28% SwiGLU-hidden cut (1024→736) whose function-class gain
does not recover the lost plain capacity. iso-param residual m1b-vs-baseline = +0.010%.

## 5. Verdict — **NULL. STOP. Do NOT escalate to a 1.3B run.**

Per the task decision rule: a GO requires **m1b to beat BOTH baseline AND the plain-wider
control** (on in-dist test acc or extrap). m1b does the opposite — it **ties-or-loses** the
plain-MLP arms at **every metric, every T, and both p**, and never raises the grok ceiling.
Therefore **NULL**, and we **do not** escalate to a 1.3B A/B.

This matches the design doc's honest prior (`STATE_AWARE_MLP_DESIGN.md` §7) precisely:

- **M1 exposes no NEW information about `S`** — it re-feeds the same 6912 numbers `o_proj`
  already consumes. If `o_proj`→fat-SwiGLU already approximates the useful cross-head
  products, M1's pre-collapse nonlinear re-mix adds nothing — and here it didn't.
- **M1 does not touch temporal dynamics** (the documented modular_quadratic length-extrap
  separator is per-step nonlinearity-IN-TIME inside the recurrence — grok-symmetric-width /
  grok-confirm). M1 changes only how the MLP *reads* an already-tanh-bounded state, so no
  extrap gain is mechanistically expected, and none was measured.
- The `m`-down-proj `R` is itself a linear collapse, so the surviving "pre-collapse
  nonlinearity" acts on a narrow linear shadow — a real but narrow function-class gain paid
  for with an MLP-hidden cut that, empirically, costs more grok reliability than it buys.

This NULL is consistent with the standing emender pattern (lb-compare ties bpb / loses modular_counter, e97delta-1p3b
token-win/wall-loss, opt-synth composition NULL, ttt-capability convergent-null,
emender-real-1p3b NULL at scale). The mechanism + iso-param certification are committed and reusable;
the 1.3B A/B is the correct thing **not** to spend GPU-days on.

### Reproduce
```
python scripts/state_aware_mlp_param_check.py --build-arms      # iso-param cert (1.3B arms)
python experiments/grok_symmetric_width/m1_smoke.py             # fused-guard + grad-flow (GPU)
python experiments/grok_symmetric_width/orchestrate_m1.py --gpus <ids>   # 18-run probe
python experiments/grok_symmetric_width/aggregate_m1.py         # table + verdict
```
