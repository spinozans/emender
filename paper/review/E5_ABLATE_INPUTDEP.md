# E5 — Ablating the input-dependence confound on e88-linear

**Task:** `e5-ablate-inputdep` (agent-1026). Date: 2026-06-04.
**Question:** Is the S5 state-tracking win of `e88-linear` driven by the LEADING
competing explanation in the linear-RNN literature — **input-dependence** (gating
/ selective transition) **+ eigenvalue range** — rather than by state
nonlinearity or rounding?

Literature framing: Cirone 2024 (input-dependence is what gives selective SSMs
their expressivity); Grazzi 2025 & Khavari 2025 (eigenvalues extended to the
**negative**/`[-1,1]` range unlock state-tracking that `[0,1]`-constrained decays
cannot reach); DeltaProduct / Siems 2025 (dense / multi-step transitions). The
common claim: a *linear*-state model tracks state because of **input-dependence +
eigenvalue range + dense transition**, NOT because of any state nonlinearity or
rounding.

`e88-linear` is `linear_state=1` (no `tanh` on the state) **and** `use_gate=1`
(input-dependent output gate). So it is squarely in the regime the literature
says *should* be the explanation. This experiment tests that directly.

---

## 1. What `e88-linear` actually is (model: `E88FLAHybrid`)

`--layer_pattern E88` resolves to `E88FLAHybrid`
(`ndm/models/e88_fla_hybrid.py`, via `get_ladder_level` in
`ndm/models/ladder_lm.py:424`). Per-head, per-slot the linear-state recurrence is

```
new_state = decay · S_prev + outer(k, delta)          # linear_state=1: no tanh
out       = (S_new.T @ q) · gate                       # gate present iff use_gate=1
```

The **transition** applied to the state each step is the scalar `decay` (a
diagonal `decay·I` map). Two sources of **input-dependence** feed this layer:

1. **Input-dependent transition (selectivity).** With the default
   `decay_mode='mamba'`,
   `decay = exp(g)`, `g = -exp(A_log)·softplus(a_proj(x) + dt_bias)`
   (`e88_fla_hybrid.py:1390-1398`). Because `a_proj(x)` depends on the token, the
   transition eigenvalue is itself a function of the input — this is exactly the
   Mamba/Cirone "selectivity".
2. **Input-dependent output gate.** With `use_gate=1`,
   `out = out · silu(g_proj(x))` (`g_proj` is input-dependent).

### Eigenvalue range — task part (b): ALREADY constrained to (0,1) by construction

For `decay_mode='mamba'`: `A_log.exp() > 0` and `softplus(·) > 0`, so `g < 0`
**always**, hence `decay = exp(g) ∈ (0,1)` **always**. The transition is a
positive real scalar strictly inside the unit interval. There is **no code path
in `E88FLAHybrid` (or `MoME88`) that can produce a negative or complex
eigenvalue** — `decay` is either `exp(negative)`, `sigmoid(·)`, `1`, or a
`sigmoid` constant, all in `[0,1]`.

**Consequence for the Grazzi/Khavari explanation:** "constrain the eigenvalues to
`[0,1]`" (task part b) is a **no-op** — they are *already* in `(0,1)`. The S5 win
therefore happens **inside the `[0,1]`-eigenvalue regime that Grazzi/Khavari say
is *too weak* for state-tracking.** So the negative/complex-eigenvalue mechanism
is **architecturally ruled out as the explanation**: `e88-linear` does not have
it and wins anyway.

Testing the *opposite* direction — *allowing* negative eigenvalues (the
`[-1,1]` regime Grazzi/Khavari credit for state-tracking) — would require a
**model code change** (e.g. a `decay = 2·sigmoid(·) - 1` or complex-diagonal
parameterization in `E88FLAHybrid.forward`). That is **not faked here**; it is
recorded as the one piece of part (b) that needs a code change (see §5). It is
also *not necessary* for the present conclusion: the leading explanation, as
stated, predicts the win should depend on input-dependence and/or on having
eigenvalue range beyond `[0,1]`, and we can test the input-dependence half
directly and cleanly.

---

## 2. Ablation design (REAL training, no mocks; ONLY GPUs 4,5)

Baseline and both ablations share the **e88-linear symmetric-CMA winner** shape
verbatim (`results/s5_symmetric_20260603/winners/e88-linear.args.json`):
`dim=256, depth=5, n_heads=38, n_state=32, expansion=1.0, lr=0.0026571…,
linear_state=1`, schedule-free AdamW, batch 32, seq_len 128, `K=5`. Only the
named knob moves:

| Arm | Knob change | Removes | `a_proj` (in-dep transition) | `g_proj` (in-dep gate) | params |
|-----|-------------|---------|:---:|:---:|---:|
| **baseline** `e88-linear` | — (`use_gate=1`, `decay_mode=mamba`) | nothing | ✔ present | ✔ present | 7.87 M |
| **A** `use_gate=0` | output gate off | input-dependent **output gate** | ✔ present | ✗ removed | 6.31 M |
| **B** `decay_mode=constant` | learned per-head **constant** decay `sigmoid(decay_logit)∈(0,1)` | input-dependent **transition (selectivity)** | ✗ removed | ✔ present | 7.82 M |

- **A** is task part (a): it removes the *output*-gating channel of
  input-dependence. (It also removes ~20% of params; noted as a confound.)
- **B** is the **direct** test of the *leading* explanation: it removes
  input-dependence **from the recurrence transition itself** while keeping the
  eigenvalues in `(0,1)` and staying param-matched to baseline. If the
  "input-dependence + eigenvalue range" story is right, B should collapse the S5
  win.

Both knobs are exposed cleanly in the model already; the only harness change was
adding a `--decay_mode` passthrough to `train_hybrid.py` (commit `13ac973`).
The model code (`ndm/models/*.py`) was **not** modified.

**Protocol (identical to the s5sym-eval baseline so numbers are comparable):**
3 seeds {42, 123, 456}; **S5** `s5_permutation` train T=128, **20000 steps**;
**S3** control `s3_permutation` train T=128, **10000 steps**; eval grid
T ∈ {128, 256, 512, 1024} (8 batches/length, end of training). 12 REAL runs
(2 arms × 3 seeds × {S5, S3}). Baseline numbers are **reused verbatim** from the
committed s5sym-eval run (`results/s5_symmetric_20260603/eval/e88-linear_*`).
Driver: `scripts/eval_e5_ablate_inputdep.py`; roll-up:
`scripts/aggregate_e5_ablate_inputdep.py`. Raw per-seed JSONs under
`results/e5_ablate_inputdep_20260604/eval/`.

---

## 3. Results

<!-- RESULTS_TABLE_START : filled from scripts/aggregate_e5_ablate_inputdep.py over the committed raw JSONs -->
Seed-mean accuracy ± SD over seeds {42, 123, 456}; n=3 each. All 12 runs
completed, 0 failed (~7.7 h wall on GPUs 4,5, 6-way co-located). Raw per-seed
JSONs: `results/e5_ablate_inputdep_20260604/eval/{arm}_{S5,S3}_seed{seed}.json`;
roll-up `eval/summary.json`.

### S5 — non-solvable NC¹ witness (random = 1/120 = 0.0083)

| Arm | T=128 | T=256 | T=512 | T=1024 |
|-----|------:|------:|------:|------:|
| **baseline** `e88-linear` (use_gate=1, mamba) | **0.9997**±0.0005 | 0.7515±0.1253 | 0.3909±0.0779 | 0.2002±0.0334 |
| **A** `use_gate=0` (no output gate) | **0.9999**±0.0001 | 0.8965±0.0627 | 0.4863±0.0688 | 0.2490±0.0347 |
| **B** `decay_mode=constant` (input-indep transition) | **0.3987**±0.0107 | 0.2014±0.0054 | 0.1059±0.0030 | 0.0567±0.0012 |

### S3 — solvable control (random = 1/6 = 0.1667)

| Arm | T=128 | T=256 | T=512 | T=1024 |
|-----|------:|------:|------:|------:|
| **baseline** `e88-linear` (use_gate=1, mamba) | 1.0000±0.0000 | 0.9919±0.0070 | 0.8646±0.0957 | 0.6480±0.1502 |
| **A** `use_gate=0` (no output gate) | 1.0000±0.0000 | 0.9974±0.0034 | 0.9377±0.0393 | 0.7497±0.1208 |
| **B** `decay_mode=constant` (input-indep transition) | 0.4027±0.1094 | 0.2845±0.0523 | 0.2250±0.0254 | 0.1960±0.0150 |

Per-seed (S5@T128): A = {0.99997, 0.99982, 1.0} (all at ceiling);
B = {0.3865, 0.4062, 0.4035} (tight collapse, not a single bad seed).
<!-- RESULTS_TABLE_END -->

Random baselines: S5 = 1/120 = 0.0083, S3 = 1/6 = 0.1667.

---

## 4. Interpretation

_(Decision rule, fixed before reading the numbers.)_

- **If the S5 win DISAPPEARS under A (`use_gate=0`) and/or B
  (`decay_mode=constant`)** → the win is driven by **input-dependence** (the
  leading explanation), not by rounding or state nonlinearity. B disappearing is
  the stronger statement: input-dependence *of the transition* (selectivity) is
  the mechanism.
- **If the S5 win PERSISTS** under the ablation(s) → input-dependence is **not**
  the explanation for that channel; combined with §1 (eigenvalues already in
  `(0,1)`, so the negative/complex-eigenvalue mechanism is absent), this would
  **reopen H1/H2 and the knob** as the live explanation.

<!-- INTERPRETATION_VERDICT_START -->
**Verdict: the S5 win is driven by INPUT-DEPENDENCE — specifically the
input-dependent *transition* (selectivity), NOT the output gate, and NOT (by
construction, §1) any negative/complex eigenvalue range.** A clean double
dissociation:

- **Output-gate input-dependence is NOT the driver (arm A).** Removing the
  input-dependent output gate leaves the S5 win fully intact — `0.9997 → 0.9999`
  at T=128, and it *improves* length-extrapolation at every held-out length
  (T=256 `0.75 → 0.90`, T=1024 `0.20 → 0.25`). S3 is likewise unchanged
  (`1.0000`). So this channel of input-dependence carries none of the win; if
  anything the gate slightly *hurts* extrapolation.

- **Transition input-dependence (selectivity) IS the driver (arm B).** Replacing
  the input-dependent decay `exp(-exp(A_log)·softplus(a_proj(x)+dt))` with a
  learned per-head **constant** `sigmoid(decay_logit)∈(0,1)` — the same square,
  param-matched, linear-state model with eigenvalues still in `(0,1)` — collapses
  S5@T128 from `0.9997 → 0.3987` (−60 pts), and length-extrapolation falls to
  near floor (`0.057` at T=1024 vs `0.20` baseline; random `0.0083`). The
  *solvable* S3 control collapses too (`1.0000 → 0.4027`), confirming that the
  loss is the model's core sequence-tracking ability, not an S5-specific
  artifact. The collapse is tight across seeds (SD `0.011` at S5@T128), so it is
  the removed selectivity, not seed noise.

**Mapping to the task's decision rule:** "S5 win DISAPPEARS under
constrained/removed input-dependence → the win is INPUT-DEPENDENCE, not rounding
or state nonlinearity." → **CONFIRMED**, and localized: it is the *input-dependent
transition* (Cirone 2024 selectivity), achieved entirely *within* the `[0,1]`
eigenvalue regime (so Grazzi/Khavari's negative-eigenvalue mechanism is *not*
what is operating here — see §1). The competing "input-dependence" explanation
for the S5 separation is **upheld** for the transition channel and **refuted**
for the output-gate channel.

**Implication for H1/H2 / the knob.** Because removing the input-dependent
transition (not nonlinearity, not rounding) is what kills the win, the leading
literature explanation accounts for the S5 result; the rounding / state-
nonlinearity hypotheses (H1/H2) are *not* needed to explain this separation and
are not reopened by this experiment. (Note `e88-linear` has no state nonlinearity
to begin with — `linear_state=1` — yet wins, which is itself already evidence
against a nonlinearity-driven story; E5 adds that the active ingredient is the
selective transition.)
<!-- INTERPRETATION_VERDICT_END -->

---

## 5. Caveats / what was NOT done (no fakes)

- **Eigenvalue constraint to `[0,1]` (task part b) is already satisfied by
  construction** (§1) — nothing to ablate; not faked.
- **Allowing negative/complex eigenvalues** (`[-1,1]`, the Grazzi/Khavari
  state-tracking regime) is the one part-(b) direction that would need a **model
  code change** in `E88FLAHybrid.forward` (a signed/complex `decay`
  parameterization). It is **documented here, not faked**, and is a candidate
  follow-up (would test whether *adding* eigenvalue range *raises* the ceiling,
  the complement of this experiment).
- Arm **A** changes parameter count (6.31 M vs 7.87 M); arm **B** is
  param-matched (7.82 M) and is the cleaner input-dependence test.
- Only GPUs 4,5 were used; `paper/main.typ` was not edited; not pushed.
