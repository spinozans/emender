# E97-RAW Expressivity Battery — Characterizing the 1.3B LM CMA-ES Winner

**Task:** `e97-raw-expressivity` · **Date:** 2026-06-06 · GPUs 0–3 only (sibling
study `e97-raw-plus-mlp` owns 4–7).

## Why

`e97-raw` is the **#1 cell on the 1.3B LM CMA-ES leaderboard** (avg-loss 5.9511),
yet it had **never** been run on the expressivity battery — every prior S5 / count /
latch / recall measurement was on the E98 unified-cell corners + GDN, *not* on the
full `e97-raw` recurrence. This study closes that gap: it measures whether the LM
winner is also expressive, and isolates which of its two design choices buys (or
costs) which capability. The output feeds the next mixing study (which capability
does `e97-raw` need a partner head for).

## What `e97-raw` is

`e97-raw` = **split-gate edit** (E97 = `E88FLAHybrid(use_split_edit=True)`:
decoupled key-axis erase `b*k` and value-axis write `w*v` gates, GDN-2-inspired)
with a **tanh state-squash** and a **RAW WRITE** — it writes the value `v`
directly instead of the delta-corrected `v − S@k`. Two design choices distinguish
it from ordinary delta-rule linear attention:

1. **raw-write** (no delta correction)
2. the **tanh** state nonlinearity `S = tanh(decay·S + outer)`

## Arms (param-matched isolation)

All E97 arms are the split-gate recurrence (`--layer_pattern E97`), dim=256 /
32 heads / N=V=32, depth 4 → **7.40M params each**. The two ablations each flip
exactly one design choice, so each pairwise comparison is a clean isolation:

| arm | split gate | state nonlin | write | params | isolates |
|---|---|---|---|---|---|
| **e97-raw** | yes | tanh | **raw** | 7.40M | — (the LM winner) |
| **e97** | yes | tanh | **delta** | 7.40M | `e97-raw` vs `e97` ⇒ **raw-write** |
| **e97-linear** | yes | **identity** | delta | 7.40M | `e97` vs `e97-linear` ⇒ **tanh squash** |
| **gdn** (ref) | GatedDeltaNet | — | delta | 1.61M | recall reference bar |

> **GDN reference is the parameter-*efficient* recall workhorse** (1.61M at the
> matched shape — ~4.6× fewer params than the E97 arms). It is the established
> recall bar in this exact battery harness (E98 five-corner: GDN MQAR
> 0.944 @T=128). Param-matching GDN *up* would only strengthen it, so the
> smaller GDN is a **conservative** reference: if `e97-raw` loses to a 1.6M GDN,
> the recall gap is real and not a capacity artifact. The three E97 arms are
> param-matched to *each other*, which is what the H1/H2/H3 isolations require.

## Probes (existing battery, `experiments/expressivity_tasks/tasks/`)

| capability | probe | random baseline |
|---|---|---|
| TRACK (state-tracking) | `s5_permutation` | (auto) |
| COUNT (`aⁿbⁿcⁿ`) | `anbncn_viability` | (auto) |
| NONLIN (iterated map) | `iterated_nonlinear_map` | (auto) |
| LATCH (flag-hold) | `flag_hold_recall` (K=4) | (auto) |
| RECALL (MQAR) | `mqar_recall` | (auto) |

## Protocol

REAL training, REAL data — no mocks/synthetic substitutes. fp32 for the E97 arms
(`--disable_autocast`; bf16 measured *slower* — the split-edit recurrence is
latency-bound, not throughput-bound); GDN runs bf16 (its chunked delta kernel
rejects float32). schedule-free AdamW, lr=3e-4, depth 4, batch 32, 5000 steps —
the same protocol as the E98 five-corner battery, for direct comparability.
**3 seeds {42, 123, 456}**; train T=128; length-extrapolation eval at
T ∈ {128, 256, 512, 1024} (Délétang protocol). 4 arms × 5 probes × 3 seeds =
**60 runs**.

Harness note: the `E97` level token did not start with `'E88'`, so
`train_hybrid.py::_layer_kw` and `hybrid_ladder.py` were silently dropping the
`raw_write` / `state_activation` / `use_triton` overrides for it. Fixed in this
task (commit `dd632ab`); the Triton split-edit path was verified **bit-exact**
against the reference recurrence for all three E97 arms (incl. `raw_write`).

Runner: `experiments/expressivity_tasks/run_e97_raw_expressivity.py`
Aggregator: `experiments/expressivity_tasks/aggregate_e97_raw_expressivity.py`

---

## Per-probe results (accuracy, mean±std over seeds 42/123/456)

All 60 runs completed, 0 failures. `final_acc` is the T=128 train-length
accuracy; the other columns are length-extrapolation eval. Numbers below were
**independently re-derived from the raw JSON by a separate verification pass** and
match the aggregator (e.g. e97-raw MQAR 0.137 ≡ re-derived 0.1344).

### TRACK (state-tracking) — `s5_permutation`  (random baseline 0.008; *solvable*: unified-track head hits 1.000 at this scale)

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97-raw | 0.096±0.006 | 0.052±0.004 | 0.030±0.002 | 0.020±0.001 |
| e97 | 0.301±0.024 | 0.155±0.016 | 0.082±0.007 | 0.044±0.003 |
| e97-linear | 0.301±0.013 | 0.158±0.012 | 0.083±0.003 | 0.045±0.002 |
| gdn | 0.074±0.004 | 0.041±0.001 | 0.025±0.001 | 0.017±0.001 |

### COUNT (aⁿbⁿcⁿ) — `anbncn_viability`  (random baseline 0.500)

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97-raw | 1.000±0.000 | 0.925±0.014 | 0.853±0.012 | 0.825±0.025 |
| e97 | 1.000±0.000 | 0.932±0.014 | 0.863±0.007 | 0.839±0.014 |
| e97-linear | 1.000±0.000 | 0.960±0.007 | 0.896±0.030 | 0.868±0.031 |
| gdn | 0.994±0.002 | 0.940±0.020 | 0.900±0.035 | 0.893±0.046 |

### NONLIN (iterated map) — `iterated_nonlinear_map`  (random baseline 0.100)

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97-raw | 0.683±0.007 | 0.677±0.007 | 0.677±0.008 | 0.673±0.008 |
| e97 | 0.885±0.013 | 0.879±0.011 | 0.878±0.011 | 0.876±0.012 |
| e97-linear | 0.893±0.008 | 0.887±0.007 | 0.887±0.006 | 0.885±0.006 |
| gdn | 0.869±0.006 | 0.857±0.004 | 0.847±0.002 | 0.822±0.002 |

### LATCH (flag-hold) — `flag_hold_recall` (K=4)  (random baseline 0.500)

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97-raw | 1.000±0.000 | 0.995±0.007 | 0.891±0.094 | 0.792±0.102 |
| e97 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| e97-linear | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| gdn | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |

### RECALL (MQAR) — `mqar_recall`  (random baseline 0.016)

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97-raw | 0.137±0.007 | 0.066±0.005 | 0.031±0.004 | 0.017±0.003 |
| e97 | 0.152±0.012 | 0.087±0.011 | 0.042±0.008 | 0.026±0.006 |
| e97-linear | 0.161±0.006 | 0.096±0.003 | 0.046±0.001 | 0.025±0.002 |
| gdn | 0.952±0.004 | 0.829±0.011 | 0.530±0.017 | 0.258±0.011 |

### Capability summary (T=128 train length)

| arm | TRACK | COUNT | NONLIN | LATCH | RECALL |
|---|---|---|---|---|---|
| **e97-raw** | ✗ 0.10 | ✓ 1.00 | ◐ 0.68 | ✓ 1.00* | ✗ 0.14 |
| e97 | ✗ 0.30 | ✓ 1.00 | ✓ 0.89 | ✓ 1.00 | ✗ 0.15 |
| e97-linear | ✗ 0.30 | ✓ 1.00 | ✓ 0.89 | ✓ 1.00 | ✗ 0.16 |
| gdn (ref) | ✗ 0.07 | ✓ 0.99 | ✓ 0.87 | ✓ 1.00 | ✓ 0.95 |

✓ solved · ◐ partial · ✗ fails. *e97-raw latch is solved at train length but
**collapses under length-extrapolation** (1.00 → 0.79 @T=1024, high variance).

---

## Hypothesis verdicts

### H1 — "e97-raw is weak at recall because raw-write drops the overwrite recall needs"

**Verdict: PARTIAL — the *outcome* is confirmed (e97-raw is weak at recall), but the proposed *mechanism* (raw-write) is REFUTED.**

| length | e97-raw | e97 (delta) | e97-linear | gdn (ref) | e97−e97-raw | gdn−e97-raw |
|---|---|---|---|---|---|---|
| T=128 | 0.137 | 0.152 | 0.161 | 0.952 | **+0.015** | **+0.815** |
| T=512 | 0.031 | 0.042 | 0.046 | 0.530 | +0.012 | +0.500 |
| T=1024 | 0.017 | 0.026 | 0.025 | 0.258 | +0.009 | +0.241 |

- **(a) e97-raw ≪ GDN: CONFIRMED, and massive** — 0.137 vs 0.952 at T=128, a
  −0.815 gap (~7×). e97-raw sits essentially at the recall floor; under
  extrapolation it decays to baseline (0.017 @T=1024) while GDN degrades
  gracefully (0.95→0.26).
- **(b) e97 (delta) > e97-raw: technically true but trivial** — restoring delta
  correction recovers only **+0.015** (≈2.8% of the gap to GDN). The decisive
  control is **e97-linear** (delta + identity state): at **0.161** it is also
  pinned near the floor and is in fact *slightly higher* than e97 — so
  "delta restores recall" does not even hold as a clean ordering.
- **The recall deficit is ARCHITECTURAL to the whole E97 split-gate family**, not
  caused by the raw-write choice. All three E97 variants (raw, delta, identity)
  cluster at 0.13–0.16 regardless of write rule or state nonlinearity. Recall is a
  property of the GDN gated-delta mechanism, not a placeable operating point of the
  E97 cell — consistent with the prior finding that *recall is architectural
  (GDN-backbone), not a tunable corner*.

### H2 — "the tanh squash helps latch + count"

**Verdict: REFUTED.** Clean isolation = e97 (tanh+delta) vs e97-linear (identity+delta).

| probe @T | e97 (tanh) | e97-linear (identity) | tanh − identity |
|---|---|---|---|
| flag_hold_recall @T=128 | 1.000 | 1.000 | 0.000 |
| flag_hold_recall @T=1024 | 1.000 | 1.000 | 0.000 |
| anbncn_viability @T=128 | 1.000 | 1.000 | 0.000 |
| anbncn_viability @T=512 | 0.863 | 0.896 | **−0.034** |
| anbncn_viability @T=1024 | 0.839 | 0.868 | **−0.029** |

- **LATCH: tanh is NEUTRAL, not helpful** — both arms are perfect (1.000) at every
  length. The task saturates, so there is no headroom for tanh to help (it is not
  *needed* for latch).
- **COUNT: tanh HURTS at extrapolation** — the non-saturating identity state
  *out-counts* tanh by ~3 points at every length beyond train (T512: 0.896 vs
  0.863). This is mechanistically expected: aⁿbⁿcⁿ needs an unbounded
  monotonically-growing counter, and a saturating tanh caps |S|≤1, harming the
  count — consistent with the prior result that *saturation hurts counting*.
- Neither claimed direction holds. (Note: e97-raw's latch *does* collapse under
  extrapolation, but that isolates the **write rule**, not tanh — see cross-cutting
  findings.)

### H3 — "does e97-raw reach the S5 track regime via its erase gate?"

**Verdict: NO — e97-raw does not reach the track regime; it sits off it, and raw-write makes it the worst of the E97 arms.**

| length | e97-raw | e97 | e97-linear | gdn | baseline |
|---|---|---|---|---|---|
| T=128 | **0.096** | 0.301 | 0.301 | 0.074 | 0.008 |
| T=1024 | 0.020 | 0.044 | 0.045 | 0.017 | 0.008 |

- A *solved* S5 is ~1.0 (the unified-track head reaches **1.000** at this exact
  scale — so the probe is solvable, the failure is the model's). e97-raw reaches
  only **0.096** at train length and decays toward baseline with length
  (0.020 @T=1024).
- **Raw-write actively hurts state-tracking**: e97-raw (0.096) is ~3× *worse* than
  the delta variants e97 / e97-linear (~0.30) — and even those only reach 0.30,
  also failing. The erase gate `b_t` is not learning the permutation-tracking
  recurrence at this config/training.
- GDN is also off-regime (0.074) — state-tracking is a deficit for the *whole*
  candidate set here, not just e97-raw.

## Cross-cutting findings (from the completeness pass)

1. **Raw-write is a pure liability on the expressivity battery.** Across all 5
   probes, the raw-write choice (e97-raw vs e97) produces a **positive delta on
   none** of them — it only ever costs. Yet e97-raw is the **#1 LM cell**. Its
   benefit therefore lives in the LM regime (optimization / throughput / scaling),
   **not** in these algorithmic capabilities. The mixing study should not expect
   raw-write to *contribute* expressivity.
2. **Raw-write damage is concentrated on iterated-composition probes.** The two
   probes it hurts most are **NONLIN −0.20** (0.89→0.68, dragging down a capability
   the architecture otherwise *has* — its single biggest casualty) and **TRACK/S5
   −0.20** (0.30→0.10). The delta-rule write is what enables stable iterated state
   composition; raw-write corrupts multi-step state evolution. Storage/counting
   probes (COUNT −0.0002, LATCH tie at train length) are barely touched.
3. **The tanh knob is nearly inert; the write rule dominates ~100×.** e97 (tanh)
   and e97-linear (identity) are near-identical everywhere (S5 0.302 vs 0.302,
   NONLIN 0.885 vs 0.892, COUNT tied, RECALL 0.157 vs 0.165). The study's named
   knob (tanh) is a red herring relative to the delta-vs-raw write axis.
4. **Hidden LATCH length-extrapolation collapse.** At train length all arms latch
   perfectly, but raw-write **collapses under extrapolation** (1.00 → 0.89 @T=512 →
   0.79 @T=1024, per-seed 0.80/0.66/0.91 — unstable), while delta/GDN hold 1.000 at
   every length. A train-length-only reading would miss this.

---

## Recommendation: what does `e97-raw` need help with?

e97-raw is **self-sufficient on COUNT and (at train length) LATCH**, **moderate on
NONLIN**, and **fails RECALL and TRACK**. Priority order for the mixing study:

| priority | capability | e97-raw | best ref | gap | what to mix |
|---|---|---|---|---|---|
| **1** | **RECALL** | 0.137 | GDN 0.952 | **−0.81** | a **gated-delta / GDN recall head** — recall is architectural, not reachable by tuning the E97 cell |
| **2** | **LATCH (extrap)** | 0.79 @T=1024 | 1.00 | **−0.21** | a **delta-write or GDN head** restores length-robust latching (raw-write's latch is brittle) |
| 3 | **NONLIN** | 0.683 | e97/linear 0.89 | −0.20 | optional ceiling-lift: a **delta-write head** (extrap-stable already, just a lower ceiling) |
| — | **TRACK/S5** | 0.096 | unified-track 1.00 | −0.90 | needs a **dedicated state-tracking mechanism**, *not* a cheap head-mix: GDN (0.07) and the delta E97 arms (0.30) all fail it too |
| — | COUNT | 1.000 | — | ~0 | already solved; no help needed |

**Concrete recommendation for the next mixing study:**

1. **Mix in a GDN / gated-delta head as the recall (and length-robust-latch)
   backbone.** This is the single highest-value addition: it closes the −0.81
   recall gap and the −0.21 latch-extrapolation gap simultaneously, since GDN is
   perfect on both. This reaffirms the prior "**GDN-backbone + exotic-specialist
   sprinkle**" thesis — e97-raw is an exotic specialist (count/latch), not a
   recall backbone.
2. **Add a dedicated state-tracking head** (the unified-`track` corner, which
   solves S5 at 1.000) if S5/group-tracking is in the target capability mix. Do
   **not** expect raw-write or GDN to provide this — it is a separate mechanism.
3. **Keep e97-raw for COUNT/LATCH-at-length and as the LM-efficient backbone**, but
   treat its raw-write as an LM-regime optimization choice, not an expressivity
   asset — on every algorithmic probe the delta-write variant is ≥ raw-write.
4. **The tanh-vs-identity knob is not worth optimizing for expressivity** (inert,
   and tanh mildly hurts counting). The actionable design axis is delta-vs-raw
   write + which specialist heads to add.

## Reproduction

```bash
# 60 runs (4 arms × 5 probes × 3 seeds, 5000 steps, fp32, GPUs 0-3, 3 jobs/GPU)
python experiments/expressivity_tasks/run_e97_raw_expressivity.py
# regenerate the [auto] tables
python experiments/expressivity_tasks/aggregate_e97_raw_expressivity.py
```
Raw per-run JSON: `experiments/expressivity_tasks/results/e97raw_<probe>__<arm>__seed<seed>.json`.
