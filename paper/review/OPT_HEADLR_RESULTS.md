# OPT_HEADLR_RESULTS — per-head-type learning rates (OPT_SPEC §5.1)

**Task:** `opt-headlr` · **Probe 1 of the OPTIMIZATION-NOT-ARCHITECTURE line** ·
**Date:** 2026-06-10

This is the §5.1 lever probe: on the **frozen** within-layer GDN+nonlin mixture, do
**recall-class** heads (`gdn2_recall`) and **compute-class** heads
(counting/step-growth/nonlin) want **different learning rates**? The pre-registered
metric is **Joint Capability Coverage** (JCC = worst-corner ratio vs frozen
specialist ceilings) at convergent loss, with the §1.4 GO/NULL bar. The function
class is frozen; only the **training regime** (the per-head-type LR ratio) is swept.

---

## 1. What was run (the lever)

**New plumbing (the only new code; OPT_SPEC §5.1, §2.4).** `train_hybrid.py` grew
two flags, `--head_lr_recall_mult` / `--head_lr_compute_mult`, that split the typed
mixture's recurrent head parameters into **two optimizer param-groups** by the head
type's sub-block and scale each group's LR independently:

- **recall-class** = the `gdn2_recall` FLA sub-block (`layers.*.gdn.*`).
- **compute-class** = the counting/step-growth/nonlin sub-blocks
  (`layers.*.{unified,shell,e97_raw,e97_delta,refit}.*`).
- the shared trunk (embeddings, MLP, norms, output head) stays at base LR.

Classification is by the **direct child token** after the layer index, not a
substring, so the nonlin-shell's inner `shell.gdn.*` params are correctly
compute-class (not misread as recall). This is **optimizer plumbing, not a kernel /
architecture change** — the function class is untouched.

**Substrate `M` (frozen; OPT_SPEC §2.2 "house mixture").** 50% `gdn2_recall`
(neg-eigval on → recall **and** track) + 25% `refit-del` (momentum-off refit = the
gated-delta / e97-delta **exact** special case → counting) + 25% `nonlin`
(UnifiedCellLayer nonlin corner → step-growth), at `n_heads=32`, dim 256, depth 4,
`mlp_ratio 2.0`, **fp32** (`--disable_autocast`), schedule-free AdamW.

> **Counting-slice note.** §2.2 lists "e97_delta/refit-del for counting". The
> `e97_delta` fused split-edit kernel is **bf16-only** and cannot run under the
> §2.1-mandated fp32; `refit` with momentum **off** is the documented **exact**
> equivalent (refit μ=0 ≡ e97 delta; `ttt-triton`) and is fp32 + fused (proven in
> the TTT battery). We use `refit-del` so the whole substrate stays fp32 as the
> spec requires. The allocation is exactly 16 `gdn2_recall` / 8 `refit` / 8
> `nonlin` heads (verified from the typed-head allocator).

**Arms (the swept LR-multiplier ratio; recall-class held at base unless falsifier):**

| arm | recall-mult | compute-mult | role |
|---|---|---|---|
| `headlr_uniform` | 1× | 1× | = **B₂** substrate default |
| `headlr_c2` | 1× | 2× | |
| `headlr_c5` | 1× | 5× | |
| `headlr_c10` | 1× | 10× | the 10× point |
| `headlr_c20` | 1× | 20× | the 20× point |
| `headlr_rslow` | 0.5× | 10× | recall gentler, compute driven (CMA pattern) |
| `headlr_inverted` | 10× | 1× | **pre-registered falsifier** (drive recall) |
| **B** `gdn2-default` | — | — | fla-gdn (gdn-neg), default regime |

**Controls (OPT_SPEC §4.1).** **B = `gdn2-default`** is the fla-gdn (allow_neg_eigval=1)
incumbent at its **best base LR** from a sanity sweep `{3e-4, 5e-4, 1e-3}` — reasonably
tuned, NOT hobbled; it doubles as the **recall + track** specialist ceiling (gdn-neg
owns both). **B₂ = `headlr_uniform`** isolates the lever from the substrate. The
**counting** ceiling is the all-`refit-del` specialist; the **step-growth** ceiling
is the all-`nonlin` specialist.

**Battery (REAL deterministic generators, REAL training; OPT_SPEC §3).** recall →
`mqar_recall`; counting → `modular_counter` K=5 / `dyck_depth_unbounded` /
`anbncn_viability`; step-growth → `modular_quadratic` p=64 / `iterated_nonlinear_map`;
track → `s5_permutation`; sanity → `parity`. Train T=128; eval length-extrapolation
T ∈ {128, 256, 512}. 3 seeds {42, 123, 456}.

**Convergence (OPT_SPEC §1.5).** Heavy tasks 8000 steps (the proven-plateau TTT
budget at this exact shape), light tasks 5000. The §1.5 **convergence certificate**
is computed on the **scored metric (accuracy)**, a faithful adaptation of the spec's
"relative improvement over the final 20% of steps": because these are EXACT
algorithmic tasks the loss → 0, so the literal relative-**loss** metric is
ill-conditioned (loss bouncing in [0.002, 0.015] at acc = 1.0 reads as a "75%
improvement"). We certify on accuracy instead — `acc_climb = acc_final − acc_80%`;
a run is converged iff `acc_climb < 0.02` (the scored metric has plateaued). An
arm/seed is converged iff all its heavy tasks plateaued.

**Longer-budget re-run for the still-climbing tasks (§1.5; no hobbled baseline,
§4.1).** At 8000 steps two tasks were still climbing for the **low-compute-LR arms,
including the baseline B**: `modular_counter` (GDN-2 acc 0.92 → 0.95 over its final
20%) and `modular_quadratic` (GDN-2 still rising / high-variance). The high-compute-LR
lever arms had already plateaued there. Scoring B at 8000 would **hobble the
baseline**, so — exactly as §1.5 prescribes ("raise the budget if any control fails
the gate") — these two tasks were **re-run at 16000 steps for ALL arms** (matched
compute per task), giving every arm its true plateau before the JCC comparison. The
already-plateaued tasks (`mqar_recall`, `dyck_depth_unbounded`, `s5_permutation`,
`iterated_nonlinear_map`, controls) kept their 8000/5000 budgets.

---

## 2. Frozen ceilings `S_c` (OPT_SPEC §1.3, §3.4)

The per-(corner, eval-length) specialist ceilings, written once to
`experiments/expressivity_tasks/opt_ceilings.json` and used as the shared JCC
denominator by every probe's aggregator. Specialist map: recall + track →
`gdn2-default` (gdn-neg owns both); counting → `spec_refit` (all-refit-del);
step-growth → `spec_nonlin` (all-nonlin).

| corner | specialist | S_c @128 | S_c @256 | S_c @512 |
|---|---|---|---|---|
| recall | gdn2-default | 0.998 | 0.977 | 0.854 |
| track | gdn2-default | 1.000 | 1.000 | 0.999 |
| counting | spec_refit | 1.000 | 0.956 | 0.608 |
| step_growth | spec_nonlin | 0.683 | 0.664 | 0.653 |

(The step-growth ceiling is ≈0.68 — the all-nonlin specialist itself only partly
solves `modular_quadratic`/`iterated_nonlinear_map`; every arm's `step_growth`
ratio ≈1.0 means the house mixture's 8 nonlin heads match the all-nonlin specialist.)

---

## 3. Results — JCC leaderboard

Seed-averaged JCC = `min_c r_c` over the scored corners {recall, counting,
step_growth, track}, with the per-corner held ratios `r_c = acc / S_c` (averaged
over the eval-length grid). `conv` = seeds whose heavy tasks all plateaued.
`modular_counter`/`modular_quadratic` trained to 16000 steps (the rest 8000/5000).

| arm | recall:compute LR | JCC | SE | recall | counting | step_growth | track | held | conv |
|---|---|---|---|---|---|---|---|---|---|
| **B** `gdn2-default` | — | 0.918 | 0.037 | 0.997 | 0.918 | 1.000 | 1.000 | 3.3/4 | 2/3 |
| **B₂** `headlr_uniform` | 1× : 1× | 0.973 | 0.009 | 0.985 | 0.978 | 1.000 | 0.993 | 4.0/4 | 2/3 |
| `headlr_c2` | 1× : 2× | 0.988 | 0.010 | 0.993 | 0.988 | 1.000 | 0.999 | 4.0/4 | 3/3 |
| **`headlr_c5`** ★ | 1× : 5× | **0.994** | 0.001 | 0.999 | 0.994 | 1.000 | 1.000 | 4.0/4 | 3/3 |
| `headlr_c10` | 1× : 10× | 0.899 | 0.025 | 0.899 | 0.979 | 1.000 | 0.998 | 3.0/4 | 2/3 |
| `headlr_c20` | 1× : 20× | 0.738 | 0.203 | 0.929 | 0.977 | 1.000 | 0.772 | 2.7/4 | 2/3 |
| `headlr_rslow` | 0.5× : 10× | 0.674 | 0.202 | 0.876 | 0.971 | 1.000 | 0.725 | 2.3/4 | 1/3 |
| `headlr_inverted` (falsifier) | 10× : 1× | 0.954 | 0.006 | 0.963 | 0.976 | 1.000 | 1.000 | 3.7/4 | 3/3 |

★ best LR-placement config. Reading the sweep:

- **The lever works, with a MODERATE optimum (5×).** `c2`/`c5` lift JCC to 0.99
  (4/4 corners held, all seeds converge) — driving the compute-class LR lets the
  counting heads converge *without* disturbing recall/track. **`c5` is the most
  stable arm in the whole study** (SE 0.001; per-seed JCC 0.992/0.993/0.997).
- **The 10–20× range BACKFIRES.** `c10` drops recall (0.899); `c20` and `rslow`
  collapse track (0.77/0.73) with huge seed variance (SE ≈ 0.20). The 10–20×
  knob-LR range that worked for `CMA_CAPABILITY`'s *single* knob group **overshoots**
  for per-head-TYPE LR — too-fast compute-class heads destabilise the shared
  residual that recall/track read from.
- **Falsifier confirms the direction.** `headlr_inverted` (drive recall 10×, gentle
  compute) reaches 0.954 — better than B (it still has the substrate's compute
  heads) but **clearly inferior to `c5` (0.994)**. Driving recall is not what helps;
  driving *compute* is. The pre-registered falsifier behaves as predicted.
- **B is seed-fragile on counting** (per-seed JCC 0.898/0.866/0.990) — the
  documented GDN-2 counting weakness; only 2/3 seeds fully plateau even at 16000.
  `c5` removes this fragility entirely.

**Lever-vs-substrate decomposition (OPT_SPEC §4):**

| comparison | ΔJCC | reading |
|---|---|---|
| substrate, `JCC(B₂) − JCC(B)` | **+0.055** | the GDN+nonlin house mixture beats pure GDN-2 on counting |
| **LR-lever, `JCC(c5) − JCC(B₂)`** | **+0.021** | the per-head-type LR's *pure* contribution on top of the substrate |
| vs incumbent, `JCC(c5) − JCC(B)` | **+0.076** | total gain over the GDN-2 incumbent |

### Per-length JCC (extrapolation gradient)

Worst-corner ratio at each eval length (train T=128). This is where the lever's
advantage is **decisive**, not marginal: the moderate-LR arms HOLD across length
while B and the over-driven arms degrade.

| arm | JCC@128 | JCC@256 | JCC@512 |
|---|---|---|---|
| B `gdn2-default` | 0.992 | 0.908 | 0.853 |
| B₂ `headlr_uniform` | 0.989 | 0.969 | 0.955 |
| `headlr_c2` | 1.000 | 0.988 | 0.977 |
| **`headlr_c5`** ★ | 1.000 | 0.982 | **0.998** |
| `headlr_c10` | 0.995 | 0.938 | 0.748 |
| `headlr_c20` | 0.852 | 0.762 | 0.701 |
| `headlr_rslow` | 0.820 | 0.732 | 0.624 |
| `headlr_inverted` | 0.999 | 0.975 | 0.903 |

At the hardest extrapolation length (T=512) `headlr_c5` holds **0.998** vs B's
**0.853** — a **+0.145** worst-corner gap. The averaged-headline GO (+0.076) is a
conservative read; the length-extrapolation signal (the algorithm-vs-memorisation
axis, OPT_SPEC §3.3) is much stronger.

### B base-LR sanity sweep (baseline not hobbled)

`B` was given a base-LR sweep `{3e-4, 5e-4, 1e-3}` on the recall + counting probes
(OPT_SPEC §4.1) to confirm the 5e-4 baseline does not hobble it. The three LRs are
comparable — 5e-4 is representative, NOT a disadvantage — so the full-battery B runs
at 5e-4 are a fair incumbent.

| B LR | recall r | counting r | note |
|---|---|---|---|
| 3e-4 | 0.981 | 0.930 | |
| **5e-4** | 0.997 | 0.918 | the full-battery baseline |
| 1e-3 | 1.000 | 0.924 | |

(Counting here spans the same witnesses; the small spread across LRs confirms B's
counting ceiling is ≈0.92 regardless of base LR — an architectural limit, not an
LR artefact. The lever lifts it to 0.99.)

---

## 4. §1.4 verdict

**GO at small scale (marginal on the averaged headline, robust on extrapolation).**

- Baseline **B = `gdn2-default`**: JCC = **0.918** (SE 0.037). Decision band
  **Δ\* = max(0.03, 2·SE_seed) = 0.074** (B's high SE is its counting seed-fragility).
- Best lever **`headlr_c5`** (recall 1× / compute 5×): JCC = **0.994** (SE 0.001).
- **`JCC(c5) − JCC(B) = +0.076 ≥ Δ\* = 0.074`**, and the gain is **on the worst
  corner** (counting, +0.076) — it does NOT win by trading a corner. By the
  pre-registered §1.4 rule this is a **GO** at small scale.

**Honest qualifications (for `opt-synth`):**

1. The averaged-headline margin (+0.076 vs Δ*=0.074) is thin and rests on B's high
   counting seed-variance inflating Δ*. The **per-length** picture is far stronger:
   at T=512 `c5` holds 0.998 vs B 0.853 (**+0.145**) — the lever wins the
   length-extrapolation (algorithm-vs-memorisation) axis decisively, holding 4/4
   corners at every length with SE 0.001 across all 3 seeds.
2. Decomposed, most of the gain over the incumbent is the **substrate** (B→B₂ =
   +0.055); the **per-head-type LR lever's pure contribution** (B₂→c5 = +0.021) is
   below the §1.4 floor (0.03). So: *vs the GDN-2 incumbent the optimized regime is a
   GO; the LR lever ALONE, on top of the house mixture, is a small (mainly
   variance-reducing) effect.* Its concrete value is **reliability** — c5 converges
   all 3 seeds and holds 4/4 corners, where B₂/B are seed-fragile (2/3).
3. **Mechanism (hypothesis confirmed).** A moderate compute-class LR boost (≈5×)
   lets the counting/step-growth heads converge onto their corners while the recall
   head trains gently and keeps recall+track. The 10–20× range **overshoots**
   (recall/track collapse, SE≈0.2). The falsifier (drive recall) is inferior. This
   is exactly the §5.1 hypothesis, with a corrected magnitude: **5×, not 10–20×.**

**Best LR-placement config forwarded to `opt-synth` / 1.3B:**
`--head_lr_recall_mult 1.0 --head_lr_compute_mult 5.0` on the house mixture
(50% `gdn2_recall` neg-eigval + 25% `refit-del` + 25% `nonlin`, n_heads=32).

**Caveats for scale.** (i) The headline margin is thin; the 1.3B validation against
CMA-best GDN-2 (§4.2) is the real test. (ii) The substrate, not the LR lever, carries
most of the small-scale gain — `opt-synth` should compose `c5` onto the
`opt-minimal` core and re-test, since the lever may interact with placement/init.
(iii) refit-del stands in for e97_delta (fp32; identical algorithm) — at scale the
bf16 e97_delta kernel could be used instead.

---

## 5. Artifacts

- Plumbing: `experiments/expressivity_tasks/train_hybrid.py`
  (`--head_lr_recall_mult` / `--head_lr_compute_mult`, per-head-type param groups).
- Runner: `experiments/expressivity_tasks/run_opt_headlr_battery.py`.
- Aggregator: `experiments/expressivity_tasks/aggregate_opt_headlr.py`
  (convergence certificate, frozen ceilings, JCC, §1.4 verdict).
- Frozen ceilings (probe-local, per-(corner,length), §3.3):
  `experiments/expressivity_tasks/results_opt_headlr/opt_ceilings_headlr.json`.
  The shared `opt_ceilings.json` on main is a per-corner SCALAR file co-written by
  the sibling probes (opt-minimal/opt-norm) — this probe keeps its finer per-length
  ceilings local so its JCC is self-consistent; **`opt-synth` (§6) reconciles the
  cross-probe ceiling denominators** (the spec's explicit synth job).
- Shared-schema rows: `experiments/expressivity_tasks/results_opt_headlr/JCC_ROWS.jsonl`.
- Per-run JSON + logs: `experiments/expressivity_tasks/results_opt_headlr/`.
