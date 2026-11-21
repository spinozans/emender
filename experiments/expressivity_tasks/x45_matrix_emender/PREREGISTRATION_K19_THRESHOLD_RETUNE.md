Pre-registration draft — K19: THE THRESHOLD RE-TUNE ON THE COMPOSED
MACHINE — a bounded CMA-ES staleness re-check of the op-routing thresholds
under the operator's overnight authorization ("run experiments to explore
potential additional wins")

**Date:** 2026-11-25 · **Status: DRAFT, operator sign-off pending.** This
registration is written and committed BEFORE any K19 probe measurement is
claimed. It does not touch any `k10_*`/`k11*`/`k12_*`/`k13_*`/`k15_*`/`x46_*`
source, any committed PREREGISTRATION, `lattice.py`, or GDN2 model code —
those are imported or read, never edited. New files only:
`k19_*.py`, `test_k19_*.py`, `run_k19_*`, the receipts, this document.
**Cards:** this lane owns cards 5/6 only (the K18 lane owns 0/1, the X46
re-run owns 2/3 — both untouched).

## 0. The question this stage exists for

The composed machine (K15: the K10 winner + the K13 ADDITIVE READ + the
K11B p3 per-channel gate stack — the program's best absolute rows,
post-cooldown **1.3811104467936925 / 1.383232252938407** at seeds
3401/3402, +0.029/+0.031 remaining vs GDN2's **1.3521734476089478**)
carries its op-routing thresholds from a machine that no longer exists.
The thresholds (`tau_match` 0.45, `tau_pop` 0.10, `tau_merge` 0.85, and
the read channel's `read_scale` 2.787 — the K10 tuned winner's own,
`optimizer_bakeoff.TUNED_WINNER`) were tuned on the PRE-read-law machine:
the committed CMA-ES search (`CMAES_TUNING.md`, `cmaes_search.py`) ran on
the K5B chunked arm with the SOFTMAX read, where `read_scale` was a
softmax LOGIT TEMPERATURE and the store held strength-normalized rows.
K15 changed the read/write interface fundamentally (the additive read's
raw `w_j = read_scale·cos(q_key, K_j)` live-gated channel; per-channel
decay/write/read gates; identity-CORRECT share, effective rows and
occupancy all moved — the K15 mechanism receipts), and **the thresholds
were never re-searched**. The K16 config bracket discharged the
LR/optimizer/family axis on the composed machine but NOT the thresholds.

**No directional prediction is pre-stated on the outcome.** Both readings
are receipts: if the re-tune closes ≥ 0.01 at the protocol, the thresholds
were STALE across the interface change and the win banks; if parity, the
thresholds were ROBUST across the interface change and the search-symmetry
duty is discharged by receipt (the prior search closed only ~0.076 of the
old machine's gap, and the interface change plausibly MOVED the threshold
optimum — either way the axis is now receipted on the machine that
actually runs it).

## 1. The parameterization (the original search's own, nothing invented;
supervisor-approved 2026-11-25)

The committed CMA search machinery is IMPORTED (`cmaes_search.py`'s
`decode`/`encode` unit-cube conventions, its subprocess-per-eval driver
shape, its per-eval fsynced log + commit-as-it-lands discipline). The
searched space is EXACTLY the original search's own dims and bounds for
these knobs (`CMAES_TUNING.md`):

| knob | scale | bounds | the original's own receipt |
| --- | --- | --- | --- |
| `tau_match` | linear | 0.2 … 0.8 | phase 2's dim, bounds verbatim |
| `tau_pop` | linear | −0.2 … 0.4 | phase 2's dim, bounds verbatim |
| `read_scale` | linear | 0.5 … 4.0 | phase 1's dim, bounds verbatim — the registered read channel scale, the "read temperature" knob; under the additive law it scales the raw `w_j` channel (a magnitude, not a softmax temperature — its optimum plausibly moved with the read law) |

* **`tau_merge` is HELD at the banked 0.85** — the original search's own
  choice, with its receipted reason standing verbatim ("merges fire plenty
  but mostly on near-duplicate rows, so threshold-tuning the merge gate
  had no evidence demanding it"). Including it would require inventing
  bounds the original never set. If K19's per-eval op-mix telemetry
  contradicts that receipt on the composed machine, that is a DOCUMENTED
  FOLLOW-UP, not a silent widening.
* **Every non-searched knob is the composed machine's banked value,
  asserted at launch** (lr 3.3427141575371284e-4 at the 4× hybrid-Muon
  lineage, betas (0.9, 0.9392004369596886), weight decay 0.17422187309224696,
  warmup 45, microbatch 8, grad-clip 1.0, `tie_break` 1e-6) — the K16
  bracket's own drift guards, applied to the threshold axis.
* **x0 = the composed machine's incumbent values** (0.45 / 0.10 / 2.787 —
  all inside the bounds), **sigma0 0.15** in the unit cube (the original
  phase-2's own), **popsize 8, ONE generation** (8 evaluations — the
  original phase-2's own shape; a bounded staleness re-check, not an
  exhaustive search), **CMA-ES seed 20261125** (stated here; the
  original's was 20261115), pycma 4.4.4 from the approved venv,
  rank-based by construction, deterministic objective, no additional
  noise handling claimed.
* **Budget: 9 evaluations** = 1 incumbent-anchor re-run (eval 0) + 8 CMA-ES
  candidates, each one matched-exposure probe (~40-50 min/eval, ~7 h
  total), inside the operator's ~9-12-eval authorization.

## 2. The evaluation convention (the original's own, horizon stated honestly)

One eval = ONE matched-exposure probe of the COMPOSED MACHINE at seed
**3401** under the CORRECTED COOLDOWN-PRIMARY SCORING: the
20,022,866-target matched-exposure landing (180 constant-LR steps, the
registered flagship stream, the document-disjoint calibration eval
boundary, recorded-read eval path), THEN the 24-step linear cooldown, and
the objective is the **post-cooldown calibration NLL** — the same
protocol, stream, seed and scoring the K15/K16 anchors live at
(`protocol_correction._train_corrected`, imported verbatim). The machine
is the K15 composition byte for byte (`velocity_ladder._build_k10` with
the eval's config → `convert_model_to_composition`, the K15 engine
IMPORTED, never modified; the p3 gate routing split + the hybrid-Muon 4×
optimizer, the K15 arm's own). The ONLY difference vs the K15 composition
arm is the searched policy thresholds.

**The honest horizon caveat, stated as before: the search tunes to its own
budget.** Here the search's eval IS the registered probe protocol itself
(180+24 steps at the matched exposure) — the same horizon the composed
anchors 1.3811/1.3832 and the GDN2 row 1.3522 were measured at — so the
confirm below is a determinism + full-receipt pass at the SAME protocol,
not a horizon transfer. No board-scale (2700 s) claim is made or implied;
the probe-horizon optimum is receipted as a probe-horizon optimum.

**The fixed seed, stated cost (the original's own sentence, verbatim in
spirit):** every evaluation uses seed 3401 and the same deterministic
stream prefix — the search tunes the seed and the data draw as well as
the thresholds. Residual cross-eval noise is inductor-autotune kernel
choice (mitigated by the shared inductor cache; the anchor re-run below
is the receipt).

**The incumbent anchor (eval 0) is a RE-RUN, not a citation** — the
original phase-2's own convention: the incumbent config through the K19
probe must reproduce the composed machine's banked seed-3401 post-cooldown
row **1.3811104467936925** and the K15 arm's own step-1 row
(5.782775391046648, the nearest-kin parity) — the harness receipt. **If
it does NOT reproduce, the lane STOPS and escalates immediately: no
search result is interpretable until the composed machine's banked row is
re-established.**

## 3. The pre-stated adjudication (the clause-conflict precedent;
supervisor's addition of 2026-11-25)

* **THE SEARCH-INTERNAL SCOREBOARD IS NEVER A LADDER ROW** (the standing
  rule). The CMA pass's internal best config is a search receipt only;
  only the PROTOCOL-CONFIRMED WINNER at seed 3401 banks anything.
* **The winner's protocol-confirm** re-runs the best config at the full
  probe protocol (matched exposure + cooldown, seed 3401) with the FULL
  receipts — the constant-180 curve, EMA-180 (ranking-only), the
  post-cooldown PRIMARY row, the mechanism instrument (identity share,
  effective rows, occupancy, the op mix) and the decorative-store sweep
  at BOTH the step-180 and post-cooldown states — and must reproduce the
  search eval's own NLL **bit-for-bit** (the determinism receipt; a
  mismatch is escalated, not excused).
* **THE WIN BAR: ≥ 0.01 at seed 3401** vs the composed machine's own
  same-seed banked row (1.3811104467936925): the winner's post-cooldown
  row must be ≤ 1.3711104467936925. If met → the thresholds were STALE
  and **the second seed (3402) is owed** and run, compared against the
  composed machine's own 1.383232252938407 with the same bar; the
  mixed-seed conservative reading governs any "new best machine" claim
  (both rows quoted next to each other; the K13/K15 clause-conflict
  precedent — the supervisor adjudicates the banking, this lane does not).
  If NOT met → **PARITY: the thresholds were ROBUST across the interface
  change** — a receipt; the search-symmetry duty is discharged either
  way. If the winner is WORSE than the incumbent by ≥ 0.01, the
  incumbent's thresholds beat the searched point at this budget — also
  receipted, and escalated.
* **Every margin is read against BOTH bands** (the endpoint-variance band
  0.075 and the winner's seed delta 0.0119), quoted with every delta. All
  anchors are quoted on every receipt: composed **1.3811/1.3832**, GDN2
  **1.3522**.
* **ESCALATION (pre-stated):** (a) any row at or past GDN2's
  1.3521734476089478; (b) any result that inverts the standing
  conclusions — in particular a threshold re-tune win large enough to
  challenge the "the plateau is architectural" reading, or a searched-
  routing result that contradicts the K14 learned-routing receipts'
  direction; (c) the anchor failing to reproduce; (d) the confirm failing
  the determinism receipt.

## 4. The mechanism duties (receipted, no bar)

* **The op-mix instrument per eval** (the search's own telemetry): the K15
  mechanism instrument (`k15_probe_diagnostics`, imported verbatim) runs
  at every eval's post-cooldown state — op mix (lane 0 rates, the op
  histogram), occupancy, the |w|-mass effective rows + the collapse
  detector, the identity-rewrite share, all three gate channels' drifts.
* **The winner's full mechanism receipts** (the confirm run): everything
  above at BOTH states PLUS the decorative-store sweep — the K15/K16
  arms' own receipt convention.
* **The question the mechanism answers, pre-stated:** did the re-tuned
  thresholds CHANGE THE OP MIX? The K14 receipts said the model REJECTS
  LEARNED routing (the structural zero-gradient finding); K19 tests
  whether SEARCHED routing on the new interface does better, and the op
  mix + occupancy + identity-share rows vs the K15 arm's own
  (identity 0.1266/0.1220, occupancy 27.20/25.12, effective rows
  16.83/15.07, top-row share 0.155/0.175) are the comparison receipts.
* **The `read_scale` trajectory is quoted in the receipts** (the
  supervisor's own request: the single most interesting dimension — the
  read law changed its mechanical role from softmax temperature to raw
  magnitude scale).

## 5. The machine, the protocol, the provenance

The composed machine VERBATIM: `velocity_ladder._build_k10` (the tuned
winner's build, the searched thresholds set on the converted mixers —
the builder's own searched-knobs path) + `convert_model_to_composition`
(the K15 conversion; the gates born at the registered scalar
equivalents; BORN-AS-THE-ADDITIVE-MACHINE receipted by the K15 lane at
BOTH seeds). Optimizer: `HybridMuonX45` at 4× the tuned AdamW base, the
p3 gate tensors on the AdamW side (`split_parameters_gated`, the
routing cross-check — the K15 arm's own plumbing, imported). Stopping
rule: matched-EXPOSURE (20,022,866 targets, constant LR) then the
24-step linear cooldown; PRIMARY the post-cooldown calibration eval;
EMA-0.99 at step 180 SECONDARY (ranking-only); constant-180 retained for
banked comparability (the confirm run). Parameters: 158,638,848,
asserted. Searched evals carry NO step-1 parity claim (different
thresholds are a different function); the anchor eval asserts the
nearest-kin step-1 parity.

**Provenance:** the K15 engine and every committed engine are READ-ONLY;
`cmaes_search.py`'s conventions are imported, not restated; the
corrected-scoring machinery, the K15/K16 anchors, the mechanism
instrument and the decorative sweep are imported verbatim. The K18 lane
(cards 0/1) and the X46 re-run (cards 2/3) are untouched.

## 6. Deliverables

`k19_threshold_search.py` (probe / search / confirm / assemble),
`run_k19_threshold_search_leased.sh` (the detached leased driver, cards
5/6, commit per eval — the ceiling is real, a committed partial beats an
uncommitted complete), `test_k19_threshold_search.py` (CPU-only,
source-named checks), the per-eval receipts
(`receipts/k19_probes/eval_*.json`, `receipts/k19_threshold_evals.jsonl`),
the search result (`receipts/k19_threshold_search_result.json`), the
confirm runs (`k19_runs/winner_seed{3401,3402}/`), the assemble
(`receipts/k19_threshold_search_summary.json`) and this lane's result
document.
