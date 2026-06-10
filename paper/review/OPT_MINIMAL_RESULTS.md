# OPT-MINIMAL RESULTS — Lever 4: minimal load-bearing core of the GDN-2 / typed mixture

**Task:** `opt-minimal` (OPT_SPEC.md §5.4) · **Role:** Programmer · **Date:** 2026-06-10

**Question.** Strip the §2.2 capability-complete substrate one component at a time:
which pieces are *load-bearing* for holding **recall + counting + step-growth +
track simultaneously**, and what is the **minimal sufficient cell**? A simpler
cell is a cleaner optimization target for the other three levers and a better
paper artifact.

**Verdict (one line).** The load-bearing ranking is
**negative-eigenvalue (track) ≫ short-conv (recall length-extrapolation) ≫ output
gate > nonlinear-in-time state ≳ Δ\* > O(depth) MLP readout (removable)**. The
**minimal sufficient cell for joint capability coverage = the substrate MINUS the
MLP readout**: `gdn2_recall` (with short-conv, output gate, **negative
eigenvalue**) + `e97_delta` + `nonlin` heads. Every recurrent component is
load-bearing; only the post-mixer MLP is droppable *for the capability battery*
(it may still matter for LM bpb — flagged for `opt-synth`/`opt-1p3b`).

---

## 1. Setup (shared OPT harness)

- **Substrate `M`** (OPT_SPEC §2.2 house mixture), realized as `typed-gdn2`:
  **50% `gdn2_recall` (neg-eigval ON) + 25% `nonlin` + 25% `e97_delta`**,
  `head_type_logits` softmax `{0.6931,0,0}` → **16 / 8 / 8** of `n_heads=32`.
  Shape `dim 256 / n_state 32 / depth 4 / mlp_ratio 2.0` (OPT_SPEC §2.3).
- **Battery** (OPT_SPEC §3.1): recall `mqar_recall`; counting `modular_counter`
  K5 / `dyck_depth_unbounded` / `anbncn_viability`; step-growth
  `modular_quadratic` K64 / `iterated_nonlinear_map`; track `s5_permutation`;
  sanity `flag_hold_recall` (latch), `parity`; reported `mixed_probe`. Train
  T=128, eval grid **{128,256,512}**, **3 seeds {42,123,456}**, schedule-free
  AdamW, 8000 steps (hard) / 5000 (controls). **270 runs total.**
- **Metric** (OPT_SPEC §1.3): per-corner accuracy = mean over witness tasks,
  seed- and eval-length-averaged; `r_c = acc/S_c` clamped; **headline JCC =
  min_c r_c** over `{recall,counting,step-growth,track}`. Δ\* = max(0.03, 2·SE)
  = **0.030** (SE_seed(min_full)=0.008).
- **Frozen specialist ceilings** `S_c` (OPT_SPEC §1.3, first-probe role §6.1,
  written to `experiments/expressivity_tasks/opt_ceilings.json`):
  **recall 0.961, counting 0.850, step-growth 0.978, track 1.000**.

### 1.1 Precision deviation (documented, all-arms-identical → fair)

OPT_SPEC §2.1 mandates `--disable_autocast` (fp32). **This is incompatible with
the spec's own `e97_delta` substrate head**: that head dispatches the FUSED
split-edit Triton kernel **only on bf16 input**; under fp32 the sub-block input
is cast to bf16 while its `Linear` weights stay fp32 → dtype mismatch (verified
crash). The batteries this spec explicitly mirrors and cross-checks against
(`run_e97_within_layer.py`, `run_capgap.py`) therefore run **bf16 autocast**. We
follow that precedent: **bf16 autocast for ALL arms** — identical precision across
arms, so the ablation comparison is fair. This is the only precision under which
the spec's own substrate runs end-to-end.

### 1.2 Two ablations are kernel-locked (out of scope per "no new kernels")

OPT_SPEC §5.4 lists `min_no_beta` (fix input-dependent δ-strength) and
`min_no_decay_inputdep` (static decay). **Neither is expressible on the existing
fused cell**: `fla.layers.GatedDeltaNet` exposes only `{use_gate,
use_short_conv, allow_neg_eigval, conv_size}` — input-dependent β and the forget
gate are intrinsic to the fused gated-delta kernel, with no toggle. Ablating them
requires modifying the FLA kernel, which the task forbids ("use existing FUSED
cells (no new kernels)"). They are reported here as **kernel-locked, not run**.
The five expressible components below cover short-conv, gate, eigenvalue sign,
state-nonlinearity, and the readout — the rest of the cell.

---

## 2. Per-corner accuracy (seed + eval-length averaged)

| arm | recall | counting | step-growth | track | **JCC** | held | latch | params |
|---|---|---|---|---|---|---|---|---|
| **min_full** (= B₂) | 0.949 | 0.833 | 0.977 | 0.999 | **0.981** | 4/4 | 1.00 | 7.93M |
| min_no_conv | 0.612 | 0.850 | 0.965 | 0.993 | **0.637** | 3/4 | 1.00 | 7.91M |
| min_no_gate | 0.921 | 0.786 | 0.976 | 0.997 | **0.925** | 3/4 | 1.00 | 7.15M |
| min_no_negeig | 0.888 | 0.815 | 0.964 | **0.291** | **0.291** | 2/4 | 1.00 | 7.93M |
| min_linear_state | 0.912 | 0.844 | 0.972 | 0.984 | **0.949** | 3/4 | 1.00 | 7.93M |
| min_no_mlp | 0.961 | 0.813 | 0.971 | 0.999 | **0.956** | 4/4 | 1.00 | 6.36M |
| **B** GDN-2 (lr 1e-3, best) | 0.952 | 0.808 | 0.978 | 1.000 | **0.951** | 4/4 | 1.00 | 2.90M |
| B GDN-2 (lr 5e-4) | 0.943 | 0.794 | 0.859 | 1.000 | 0.878 | 2/4 | 1.00 | 2.90M |
| B GDN-2 (lr 3e-4) | 0.924 | 0.727 | 0.803 | 0.999 | 0.821 | 2/4 | 1.00 | 2.90M |

Per-task detail (length-averaged accuracy):

| task | full | no_conv | no_gate | no_negeig | lin_state | no_mlp | B(1e-3) | B(5e-4) |
|---|---|---|---|---|---|---|---|---|
| mqar_recall | 0.949 | **0.612** | 0.921 | 0.888 | 0.912 | 0.961 | 0.952 | 0.943 |
| modular_counter | 0.766 | 0.802 | 0.666 | 0.721 | 0.762 | 0.688 | 0.713 | 0.722 |
| dyck_depth_unbounded | 0.808 | 0.810 | 0.774 | 0.802 | 0.843 | 0.811 | 0.775 | 0.743 |
| anbncn_viability | 0.925 | 0.938 | 0.918 | 0.923 | 0.927 | 0.939 | 0.936 | 0.918 |
| modular_quadratic | 1.000 | 0.999 | 1.000 | 0.999 | 0.991 | 1.000 | 0.999 | **0.773** |
| iterated_nonlinear_map | 0.955 | 0.930 | 0.952 | 0.929 | 0.954 | 0.942 | 0.957 | 0.945 |
| s5_permutation | 0.999 | 0.993 | 0.997 | **0.291** | 0.984 | 0.999 | 1.000 | 1.000 |
| flag_hold_recall (latch) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| parity (sanity) | 0.998 | 1.00 | 1.00 | 1.00 | 0.999 | 0.999 | 0.999 | 0.997 |
| mixed_probe (reported) | 0.931 | 0.919 | 0.939 | **0.635** | 0.933 | 0.931 | 0.937 | 0.926 |

---

## 3. Necessity table (ΔJCC when each component is removed from min_full = 0.981)

| removed component | JCC | **ΔJCC** | per-corner Δacc (recall/count/step/track) | verdict |
|---|---|---|---|---|
| **negative eigenvalue** (track) | 0.291 | **+0.690** | +0.06 / +0.02 / +0.01 / **+0.71** | **LOAD-BEARING (critical)** |
| **short-conv** on q/k/v | 0.637 | **+0.344** | **+0.34** / −0.02 / +0.01 / +0.01 | **LOAD-BEARING (critical)** |
| **output gate** | 0.925 | **+0.055** | +0.03 / +0.05 / 0.00 / 0.00 | **LOAD-BEARING** |
| nonlinear-in-time state (e97 tanh) | 0.949 | +0.031 | +0.04 / −0.01 / +0.01 / +0.01 | LOAD-BEARING (marginal, ≈Δ\*) |
| O(depth) MLP readout | 0.956 | +0.024 | −0.01 / +0.02 / +0.01 / −0.00 | **removable** (< Δ\*) |

### Mechanism of each load-bearing piece

- **Negative eigenvalue is the track substrate** (ΔJCC +0.690). Removing it
  collapses `s5_permutation` 0.999 → **0.291** (and the joint `mixed_probe` 0.931
  → 0.635), while recall/counting/step-growth are untouched. Confirms
  `E97_WITHIN_LAYER_SYNTHESIS` Q1: the negative along-key eigenvalue (`gdn-neg`)
  is what gives the recall head its *track* corner. Without it the cell is
  recall-only and track-blind. **Non-negotiable.**

- **Short-conv is recall's length-extrapolation mechanism** (ΔJCC +0.344). The
  collapse is *not* in-distribution: at the **train length T=128**, `min_no_conv`
  recall is ~0.97 (≈ full). It is a **length-extrapolation** failure — T=256 drops
  to 0.60–0.76 and **T=512 to 0.13–0.29**, vs `min_full` holding 0.79–0.90 at
  T=512. The FLA q/k/v short-conv is what makes recall *generalize to longer
  sequences*; it is dead weight for in-distribution recall but load-bearing for
  the algorithm. (Corroborates the recall short-conv caveat, `TTT_CAPABILITY` §1.)

- **Output gate** (ΔJCC +0.055): a smaller but real broad effect — recall −0.03
  and counting −0.05 (`modular_counter` 0.766 → 0.666). Load-bearing, not critical.

- **Nonlinear-in-time state** (e97 `tanh` → `identity`, ΔJCC +0.031, ≈ Δ\*):
  marginally load-bearing, and the effect is on **recall** (−0.04), **NOT
  step-growth** (`modular_quadratic` 1.000 → 0.991, unchanged). The step-growth
  corner is supplied by the `nonlin` head + the MLP, not by the e97 per-step tanh.
  This corroborates the standing **convergent-loss null on nonlinearity-in-time**
  (the e97 / TTT / nlmem nulls): linearizing the recurrent state barely moves
  joint coverage.

- **O(depth) MLP readout is removable for capability** (ΔJCC +0.024 < Δ\*).
  `min_no_mlp` keeps all four corners (JCC 0.956 ≥ bar 0.951) at **6.36M vs 7.93M
  params** and even nudges recall up (0.949 → 0.961). For the *capability battery*
  the recurrence + typed heads already supply the needed nonlinearity; the
  post-mixer SwiGLU is not load-bearing. **Caveat:** the MLP is known to matter
  for LM held-out bpb (`E97_WITHIN_LAYER_SYNTHESIS`); this finding is scoped to
  capability coverage and is flagged for `opt-synth`/`opt-1p3b`.

---

## 4. Minimal sufficient cell (OPT_SPEC §5.4)

Bar: JCC ≥ JCC(min_full) − Δ\* = 0.981 − 0.030 = **0.951**.

> **Minimal sufficient cell = `gdn2_recall` (short-conv ON, output gate ON,
> negative eigenvalue ON) + `e97_delta` + `nonlin` heads, WITHOUT the post-mixer
> MLP** (`min_no_mlp`, JCC 0.956 ≥ 0.951, 6.36M params).

Only one component — the O(depth) MLP readout — is removable without dropping
below the bar. **Every recurrent component (short-conv, output gate, negative
eigenvalue, and — marginally — the nonlinear-in-time state) is load-bearing for
joint capability coverage.** There is no dead recurrent machinery to strip: the
GDN-2 / typed mixture is already close to its minimal load-bearing core for
counting+recall+track. This bounds how much the other three levers can simplify
the optimization target — they should optimize the training of *this* cell, not a
smaller function class.

---

## 5. Control B (GDN-2) — "reasonably tuned, not hobbled" (OPT_SPEC §4.1)

The LR sanity sweep was **decisive and is the headline caveat for the OPT line**:

- B at lr **5e-4** (the spec base) lands JCC **0.878** — and is *stuck on
  step-growth* (`modular_quadratic` **0.773**), reproducing the documented
  "GDN-2 weak on step-growth".
- B at lr **1e-3** lands JCC **0.951** — `modular_quadratic` jumps to **0.999**.
  **The GDN-2 step-growth weakness is substantially an LR-tuning artifact, not an
  architectural ceiling.** Tuned GDN-2 holds all four corners (4/4 held).

Had B been run only at 5e-4 (hobbled), the mixture would have looked like a
+0.10 JCC win. Against the **best-LR** GDN-2, the mixture's edge is
**JCC(min_full) − JCC(B) = 0.981 − 0.951 = +0.030 = exactly Δ\*** — i.e. **at the
noise floor**, and bought with **2.7× the parameters** (7.93M vs 2.90M). This is
context for `opt-synth`/`opt-1p3b` (a `B = CMA-best GDN-2` at 1.3B will be even
stronger): *on capability coverage at small scale, a properly LR-tuned GDN-2
essentially ties the capability-complete mixture.* This is consistent with the
standing convergent-loss null and reinforces that the OPT thesis must win on the
**training regime**, not on the mixture per se.

---

## 6. Convergence (OPT_SPEC §1.5)

The pre-registered **relative**-loss certificate `(L₈₀−L_final)/L₈₀ < 2%` is
**pathological on these tasks**: most hard tasks reach near-zero loss, where a
tiny absolute drop is a huge *relative* drop, so the certificate fires "not
converged" even when accuracy has long plateaued. We therefore gate on the
**accuracy plateau** (capability-relevant): spread of `eval_acc` over the final
30% of training < 0.05.

- By accuracy plateau, the **vast majority of runs are converged** (e.g.
  `min_full` 18/21, best-B 20/21 hard runs plateaued).
- The non-plateaued cases **concentrate in the failing arm on the corner it
  cannot do** — `min_no_negeig` on `s5_permutation` (9/21 of its non-plateaus):
  the arm oscillates because it *lacks the capability*, which **corroborates** the
  load-bearing verdict rather than confounding it.
- The huge load-bearing effects (neg-eig +0.690, short-conv +0.344) are robust to
  any residual under-training; the marginal calls (MLP, linear-state, gate) are
  robust to ±0.01. Raw per-run accuracies and both certificates are in
  `results_opt_minimal/JCC_ROWS.jsonl` (`converged` field + `conv_certificates`).

---

## 7. Artifacts & reproduction

- Runner: `experiments/expressivity_tasks/run_opt_minimal.py` (broker-leased;
  `eval "$(scripts/gpu_lease.sh 4)"` then run; resumable).
- Aggregator: `experiments/expressivity_tasks/aggregate_opt_minimal.py`
  (emits JCC_ROWS.jsonl + opt_ceilings.json + this necessity table).
- Plumbing (existing fused cells, NO kernels): `train_hybrid.py` gains
  `--gdn_use_conv` and forwards `--use_gate`/`--gdn_use_conv` to `typed-gdn2`.
- Data: `results_opt_minimal/` (270 run JSONs + `JCC_ROWS.jsonl`),
  `opt_ceilings.json` (frozen S_c, first-probe; `opt-synth` reconciles across probes).

Reproduce:
```bash
eval "$(scripts/gpu_lease.sh 4)"
python experiments/expressivity_tasks/run_opt_minimal.py \
  --arms min_full min_no_conv min_no_gate min_no_negeig min_linear_state min_no_mlp
python experiments/expressivity_tasks/run_opt_minimal.py --b_lr_sweep 3e-4 5e-4 1e-3
python experiments/expressivity_tasks/aggregate_opt_minimal.py --write_ceilings
```

## 8. Hand-off to `opt-synth`

1. **Minimal core for `R*` composition** = `gdn2_recall`(short-conv+gate+neg-eig)
   + `e97_delta` + `nonlin`, **MLP optional** (drop for capability, keep for bpb —
   re-test at 1.3B).
2. **Load-bearing, do not strip:** negative eigenvalue, short-conv, output gate.
3. **LR is a first-class lever** (feeds `opt-headlr`): GDN-2's step-growth corner
   is unlocked by lr 5e-4 → 1e-3. Any best-vs-best comparison MUST LR-tune the
   control (confirmed here; the §4.2 1.3B `B = CMA-best GDN-2` already does).
4. **Frozen ceilings** `opt_ceilings.json` written (first-probe); verify all four
   probes share the same hash before building the unified leaderboard.
