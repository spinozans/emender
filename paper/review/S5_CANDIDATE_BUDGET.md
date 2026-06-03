# S5 Candidate-Budget Calibration

**Task:** `s5sym-calibrate` — How few per-candidate training steps until configs of the
same architecture become **reliably rank-ordered** on S5 accuracy@T=128? CMA-ES needs a
stable ranking signal, not full convergence.

**Setup (REAL training + REAL S5 eval, no mocks):**
`experiments/expressivity_tasks/train_hybrid.py --task s5_permutation`,
seq_len=128, batch=32, seed=42, schedule-free AdamW, depth=4, **4000 steps**,
**dense S5 eval every 100 steps**. S5 = composition of S₅ transpositions, |classes|=120,
chance = 1/120 = **0.0083**. GPUs 2,3,4,5 only (0,1 untouched); each arm run on its own GPU.
Run JSONs: `experiments/expressivity_tasks/results/s5_calib_20260603/`.

**Three 8M arms × deliberately-spread configs (10 runs total):**

| Arm | layer_pattern | dim | H | N | configs |
|-----|---------------|-----|---|---|---------|
| E88 (E88_8M) | `E88` | 384 | 32 | 32 | lr {9e-5, 3e-4, 9e-4} tanh + lr 3e-4 **linear** (BL-1 knob) |
| M2RNN-CMA (M2RNN_8M) | `m2rnn` | 384 | 32 | 32 | lr {9e-5, 3e-4, 9e-4} |
| GDN (FLA_8M) | `fla-gdn` | 640 | 32 | 32 | lr {9e-5, 3e-4, 9e-4} |

"lr x0.3" = 9e-5, seed = 3e-4, "lr x3" = 9e-4. For E88 the BL-1 knob (`linear_state`:
tanh=0 vs linear=1) is added at the seed lr.

**Stable-rank definition:** the earliest step N at which the within-arm config ordering by
S5 eval_acc equals the ordering at step 4000 **and never flips again through 4000**, with the
stable region spanning ≥ 300 consecutive steps (≥ 4 dense eval points).

---

## Arm: E88 (dim384 / H32 / N32)

S5 eval_acc vs step:

```
step    lr9e-5_tanh   lr3e-4_tanh   lr9e-4_tanh   lr3e-4_linear
   0       0.0166        0.0166        0.0166        0.0166
 500       0.0369        0.0684        0.0952        0.0711
1000       0.0513        0.0922        0.1546        0.0942
1500       0.0710        0.1085        0.2659        0.1253
2000       0.0848        0.1404        0.3348        0.1680
2500       0.0892        0.1712        0.4133        0.2131
3000       0.0989        0.2037        0.4645        0.2458
3500       0.1008        0.2275        0.5002        0.2846
3999       0.1105        0.2466        0.5612        0.3161
final      0.1054        0.2421        0.5605        0.3072
```

- **Above chance (>0.02, ~2.4× chance):** every config by **step 100**.
- **Final order (best→worst):** `lr9e-4_tanh > lr3e-4_linear > lr3e-4_tanh > lr9e-5_tanh`.
- **STABLE-RANK STEP = 1000** for the *full 4-config* ordering.

Pairwise — which pair is binding:

| pair | nature | reliably ordered from |
|------|--------|------------------------|
| lr9e-4 vs lr9e-5 (extreme good/bad lr) | coarse hyperparam | **step 100** |
| lr3e-4 vs lr9e-5 | coarse hyperparam | **step 100** |
| lr9e-4 vs lr3e-4 | coarse hyperparam | **step 200** |
| **linear vs tanh @ lr3e-4** (BL-1 knob) | **architectural** | **step 1000** (last flip at 900: 0.0916 vs 0.0919) |

The **lr (good-vs-bad config) ranking is reliable by step 200** even in this slow arm.
The single slow pair is the *architectural* tanh-vs-linear distinction at identical lr — a
genuinely close pair that needs ~1000 steps to separate stably. E88 is the binding arm.

## Arm: M2RNN-CMA (dim384 / H32 / N32)

```
step      lr9e-5      lr3e-4      lr9e-4
   0      0.0166      0.0166      0.0166
 500      0.0283      0.0325      0.0368
1000      0.0333      0.0398      0.0576
2000      0.0398      0.0580      0.0747
3000      0.0457      0.0761      0.0847
3999      0.0498      0.0894      0.0930
final     0.0498      0.0874      0.0948
```

- Above chance: all configs by **step 100**.
- Final order: `lr9e-4 > lr3e-4 > lr9e-5`. **STABLE-RANK STEP = 200.**

## Arm: GDN (FLA_8M, dim640 / H32 / N32)

```
step      lr9e-5      lr3e-4      lr9e-4
   0      0.0166      0.0166      0.0166
 500      0.0392      0.0488      0.0645
1000      0.0473      0.0674      0.0969
2000      0.0601      0.0983      0.1352
3000      0.0753      0.1191      0.1616
3999      0.0833      0.1422      0.1838
final     0.0825      0.1389      0.1823
```

- Above chance: all configs by **step 100**.
- Final order: `lr9e-4 > lr3e-4 > lr9e-5`. **STABLE-RANK STEP = 200.**

---

## Summary

| Arm | above-chance | stable-rank step (full ordering) | lr-only ranking stable |
|-----|--------------|----------------------------------|------------------------|
| E88   | 100 | **1000** (binding: linear-vs-tanh) | 200 |
| M2RNN | 100 | 200 | 200 |
| GDN   | 100 | 200 | 200 |

- **Max stable-rank step across arms = 1000** (E88, driven by the architectural
  `linear_state` knob; all *lr* good/bad rankings are stable by step 200).
- **No arm is unstable by 4000 steps** — all three reach a stable, monotone ordering well
  before the budget; none would force a larger budget.
- All arms cross above-chance by step 100; ranking — not chance-crossing — is the binding
  criterion.

### Sufficiency of candidate budgets

| budget | sufficient for reliable within-arm ranking? |
|--------|---------------------------------------------|
| **100**  | **NO** — only the extreme lr pair separates; E88 mid/architectural pairs not yet ordered |
| **250**  | **NO** for E88's linear-vs-tanh architectural knob (needs ~1000); **YES** for all lr-only rankings |
| **500**  | **NO** for E88's architectural knob; **YES** for all lr rankings (M2RNN/GDN fully ordered) |
| **1000** | **YES** — all arms fully and stably ordered (minimum sufficient at seed 42) |
| **2000** | **YES** — fully ordered with comfortable margin |

The user's hypothesis (100–1000 steps) is **confirmed at the top of that range**: the
hyperparameter (lr) ranking signal CMA-ES primarily needs emerges by **~200 steps**, and the
*full* config ordering including the fine architectural tanh-vs-linear distinction is stable
by **1000 steps**. 100 steps is too few; 1000 is the empirical floor for the complete ordering.

**recommended_candidate_steps: 2000**

Rationale: the max stable-rank step across arms is **1000** (E88, single seed). With a safety
margin (single seed measured; protects against seed/eval-noise variance on the close
architectural pair) the recommended per-candidate budget rounds up to **2000**. If CMA-ES
explores only coarse hyperparameters (lr-scale, not the fine `linear_state` knob), **500**
steps already gives a fully stable ranking in all three arms and is a viable cheaper budget;
**1000** is the minimum that also stably orders the architectural knob.
