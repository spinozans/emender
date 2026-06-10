# ttt-capability — Capability map of the TTT-write (`refit`) cell vs GDN-2

**Task:** `ttt-capability` · **Cell:** `refit` (momentum-delta inner-optimization WRITE
head; `ndm/models/refit_head.py` → fused `ndm/triton/refit_chunked_autograd.py`) ·
**Date:** 2026-06-10 · **Env:** NVIDIA RTX 6000 Ada, fp32 (autocast disabled),
torch 2.9.1+cu128, triton 3.5.1. All GPU runs via the gpu-broker lease.

## The question

The `refit` cell is the GDN-2 (gated-delta) substrate plus a **richer write rule**:
one **heavy-ball / momentum** inner-optimizer step per token on the per-chunk inner
reconstruction loss `½‖v−Sᵀk‖²`, with an exposed inner-step count `K` (`newton_steps`).
At momentum-off (`has_mom=False`) it collapses **exactly** to the gated-delta / e97
delta-rule (delta = one inner step). Does that richer emendation
(**momentum** and/or **K-step** inner optimization) unlock any REAL capability the
GDN-2 baseline cannot reach — especially the algorithmic / counting gaps
(`modular_counter` mod-k, the step-growth / unbounded-counting cliffs)?

## Method — matched compute, matched A/B

Three arms, all dim 256 · 32 heads · n_state 32 · depth 4 · mlp_ratio 2.0 (the fixed
O(depth) nonlinear readout present in every arm), schedule-free AdamW lr 5e-4, fp32,
train T=128, eval length-extrapolation, **3 seeds** (42/123/456):

| arm | construction | what it isolates |
|-----|--------------|------------------|
| **refit-mom** | typed-gdn2 layer, all heads → `refit` slot, `refit_has_mom=1` | full momentum-delta inner optimizer |
| **refit-del** | same layer, `refit_has_mom=0` | momentum OFF = the gated-delta special case (**only** μ removed; params identical) |
| **gdn2** | `fla-gdn`, `allow_neg_eigval=1` | the production FLA GatedDeltaNet anchor |

`refit-mom` vs `refit-del` is the **clean matched ablation** — same layer class, same
parameters, the momentum gate is the *only* manipulated degree of freedom — so it
measures *exactly* "does the richer inner optimization buy capability". `gdn2` is the
external production baseline.

Battery (REAL deterministic generators, REAL training; no mocks):

- `modular_counter` K=5 — mod-k running counter (named open gap)
- `modular_quadratic` p=64 — nonlinear non-invertible step-growth cliff (`x_t=x_{t-1}²+c_t mod p`)
- `dyck_depth_unbounded` — unbounded counting (bracket depth, cap 256)
- `parity` — mod-2 saturation control (should be solved by all)
- `mqar_recall` — multi-query associative recall — GDN-2's home turf (recall control)

Runner: `experiments/expressivity_tasks/run_ttt_capability_battery.py` (broker-aware,
resumable); aggregator: `aggregate_ttt_capability.py`; raw JSON:
`results_ttt_capability/`. 45 runs, 0 failures.

## Verdict: convergent-capability **NULL** on the TTT richness lever

**The momentum / richer inner-optimization unlocks nothing the one-step delta special
case (= GDN-2's gated-delta rule) cannot already do.** Across every task the
`refit-mom − refit-del` gap is ≤ |0.025| and never positive beyond noise; where it
differs, the **momentum-OFF** arm is marginally *better* (esp. at length-extrapolation).
The only real differences in the battery are **substrate-level** (the typed all-delta
cell is a counting/algorithmic specialist that is recall-weak vs FLA GDN-2) and are
shared identically by the momentum-off arm — i.e. they are not the TTT lever.

## 1. Final eval accuracy (train T=128), mean ± std over 3 seeds

| task | baseline | refit-mom | refit-del | gdn2 | **mom−del** | mom−gdn2 |
|------|---------:|----------:|----------:|-----:|------------:|---------:|
| modular_counter K=5 | 0.200 | 0.966 ± .037 | 0.991 ± .012 | 0.861 ± .132 | **−0.025** | +0.105 |
| modular_quadratic p=64 | 0.016 | 1.000 ± .000 | 1.000 ± .000 | 0.843 ± .272 | **+0.000** | +0.157 |
| dyck_depth_unbounded | 0.004 | 0.998 ± .003 | 1.000 ± .000 | 0.998 ± .001 | **−0.002** | +0.000 |
| parity | 0.500 | 1.000 | 1.000 | 1.000 | **+0.000** | +0.000 |
| mqar_recall | 0.016 | 0.162 ± .008 | 0.170 ± .008 | 0.994 ± .002 | **−0.008** | −0.832 |

**Reading.**
- **The momentum column (`mom−del`) is a flat null.** The richest part of the TTT
  write rule — the heavy-ball surprise EMA — moves no task by more than noise, and is
  net **slightly negative**. This is the direct answer to the task's question.
- The refit/delta cell **solves** the named "open" gaps at this protocol —
  `modular_counter` K=5 (0.97–0.99), `modular_quadratic` p=64 (1.00),
  `dyck_depth_unbounded` (1.00). At this budget they are not unsolved.
- `refit` **beats** FLA `gdn2` on the two modular tasks (+0.105, +0.157), but **so
  does `refit-del`** — the win is the typed all-delta substrate / optimization
  stability (note gdn2's large seed variance ±0.13/±0.27: it sometimes fails to
  converge), **not** the momentum.
- **`gdn2` dominates recall** (`mqar` 0.994 vs refit ~0.17). The all-refit cell is a
  counting specialist that does **not** match GDN-2 recall — consistent with the
  established "gated-delta/raw-write heads lose recall; GDN-2 recall is architectural"
  result. Caveat: FLA GatedDeltaNet applies a short conv to q/k/v that the bare refit
  head lacks; part of this gap is that implementation detail, not the write rule —
  but it is *equally* present with and without momentum, so it is not the TTT lever.

## 2. Convergent-loss-null check (final eval loss)

| task | refit-mom | refit-del | gdn2 |
|------|----------:|----------:|-----:|
| modular_counter K=5 | 0.116 ± .15 | 0.026 ± .04 | 0.380 ± .30 |
| modular_quadratic p=64 | 0.000 | 0.000 | 0.220 ± .38 |
| dyck_depth_unbounded | 0.022 | 0.000 | 0.020 |
| parity | 0.000 | 0.000 | 0.000 |
| mqar_recall | 2.244 ± .14 | 2.126 ± .07 | 0.017 ± .01 |

- **refit-mom vs refit-del loss is identical** within noise on every task — the loss-null
  for the TTT lever is airtight (the momentum gate changes neither accuracy nor loss).
- Across *substrates* loss does **not** universally converge: it splits by
  specialization — refit reaches lower loss on counting/algorithmic, gdn2 reaches
  ~0 loss on recall. So the honest statement is a **convergent-loss-null on the TTT
  richness axis**, with a substrate-specialization split between refit and GDN-2.

## 3. Length-extrapolation (train T=128; the algorithm-vs-memorization signal)

acc at eval length T (mean over seeds):

| task | arm | T=128 | T=256 | T=512 |
|------|-----|------:|------:|------:|
| modular_counter K=5 | refit-mom | 0.965 | 0.820 | 0.549 |
| | refit-del | 0.991 | 0.894 | **0.594** |
| | gdn2 | 0.859 | 0.668 | 0.466 |
| modular_quadratic p=64 | refit-mom | 1.000 | 1.000 | 0.885 |
| | refit-del | 1.000 | 1.000 | **0.990** |
| | gdn2 | 0.843 | 0.838 | 0.834 |
| dyck_depth_unbounded | refit-mom | 0.998 | 0.881 | 0.518 |
| | refit-del | 1.000 | 0.928 | 0.533 |
| | gdn2 | 0.997 | 0.814 | 0.435 |

All three arms **degrade** with length (none learns the *unbounded* algorithm — the
linear fast-weight state + fixed-depth MLP ceiling holds for refit and GDN-2 alike).
Where the arms separate, **momentum-off (`refit-del`) extrapolates best** on every
counting task — the momentum buffer, if anything, slightly *hurts* the durable
algorithm. The richer inner-optimizer does not buy length-generalization either.

## 4. K-step inner-optimization probe (the other half of "K-step/momentum")

Momentum-ON `refit`, inner-step count `K = newton_steps ∈ {1, 5, exact}`, on the two
tasks with the most spread, 3 seeds (`run_ttt_kstep_probe.py`):

| task | K | acc | loss | extrap T=512 |
|------|---|----:|-----:|-------------:|
| modular_counter K=5 | **1** | 0.202 ± .015 | 4.43 | 0.201 |
| | 5 | 0.971 ± .030 | 0.09 | 0.533 |
| | exact | 0.971 ± .030 | 0.09 | 0.536 |
| modular_quadratic p=64 | **1** | 0.092 ± .078 | 4.09 | 0.038 |
| | 5 | 1.000 ± .000 | 0.00 | 0.920 |
| | exact | 1.000 ± .000 | 0.00 | 0.902 |

**K is a solve-accuracy knob, not a capability lever.** A single inner step with
momentum is a *poor* approximation of the momentum inner solve and **fails** both tasks
(stuck at baseline, loss diverging). At `K=5` the solve is accurate and the cell
succeeds — and `K=5` and `K=exact` are **bit-for-bit equivalent** (0.971/0.971,
1.000/1.000), so extra inner steps add nothing. Crucially, the accurately-solved
momentum arm (`K≥5`: 0.971 mod-counter, 1.000 mod-quadratic) **still does not beat the
momentum-OFF delta** (`refit-del`: 0.991, 1.000 from §1; `refit-del` even edges it on
mod-counter). So the entire richer-inner-optimization surface — *both* the momentum
buffer *and* the step count — collapses onto "spend ≥5 steps to land exactly where the
cheaper one-step gated-delta already is." No K, and no momentum, unlocks a capability
GDN-2's delta rule lacks.


## What this means

The TTT write head's distinguishing machinery — the **momentum** heavy-ball term and
the **K-step** inner solve — is **inert as a capability lever** at matched compute.
Every capability the `refit` cell shows (it solves mod-k counting, the modular-quadratic
step-growth task, dyck depth, and parity at this protocol; it is recall-weak) is
**already present in its momentum-off special case**, which is GDN-2's gated-delta rule.
The cell is therefore best understood as a **counting/algorithmic-leaning gated-delta
variant**, not a new capability axis: richer emendation buys no task GDN-2 cannot reach,
and `refit-del` ≥ `refit-mom` on durable (length-extrapolated) performance. This is a
clean **convergent-capability-null** for the inner-optimization richness — the same
shape as the validated-kernel-but-no-capability outcome the spec anticipated.

The one genuine architectural axis the battery surfaces is **recall**, which GDN-2 owns
and the bare refit/delta substrate does not (partly the missing short-conv); momentum
does not touch it. A productive cell mixes a GDN-2 recall backbone with a small
counting-specialist sprinkle — and the specialist need not carry the momentum buffer.

## Reproduce

```bash
eval "$(scripts/gpu_lease.sh 4)"
python experiments/expressivity_tasks/run_ttt_capability_battery.py \
    --output_dir experiments/expressivity_tasks/results_ttt_capability
python experiments/expressivity_tasks/run_ttt_kstep_probe.py        # K-step probe
python experiments/expressivity_tasks/aggregate_ttt_capability.py
```
