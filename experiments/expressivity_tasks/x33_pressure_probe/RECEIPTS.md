# X33 — the pressure-predicts-collapse probe: receipts

**Status discipline:** this file's PRE-REGISTRATION section was written and
committed BEFORE any X33 evaluation ran. The X28E package
(`experiments/expressivity_tasks/x28e_memory_head/`) is hash-FROZEN and is
imported READ-ONLY; `verify_freeze` passes under the arms' environment
(`/home/erikg/emender/.venv/bin/python`, fla 0.5.1, torch 2.9.1+cu128) and is
re-verified by the probe at run start. No file in the frozen package, in
`x30_tape_cell`, `x31_tape_cell_fused`, `x32_comparables`, or any `results_*`
directory is modified by X33; the only reads from the results directories are
the two `best.pt` checkpoints (hashes below).

## Eval-only scope

- **Arms:** `tape_B0` (budget B = 736 slots) and `tape_B0x4` (B = 2944 slots).
- **Checkpoints (read-only inputs):**
  - `results_x28e_memory_head/tape_B0_seed0/best.pt`
    sha256 `29ad3a96cde65398e6a91f635bb4be0f8111c6f8f5cee09c0090a6263231f146`
    (stage-3 selected update 5000, validation 0.9865)
  - `results_x28e_memory_head/tape_B0x4_seed0/best.pt`
    sha256 `ced743eb7da4657a5cae837e4d78a3d950fbd6629af249e25a32f9cd17e744e1`
    (stage-3 selected update 7000, validation 0.9830)
- **Pools (all from the frozen generator, immutable seeds):** the four depth
  bins (`bin_id` k∈[1,128], `bin_2x` [129,256], `bin_4x` [257,512],
  `bin_9x` [1025,1152]; 512 balanced words each), each arm's own registered
  fine sweep (B0: k=608..864 step 16; B0x4: k=2816..3072 step 16; 256 balanced
  words each), AND — for the same-k comparison — B0's sweep points
  k=608..864 evaluated on B0x4 as well (sweep pools are seed-fixed functions
  of k, identical across arms).
- **Transparency note:** the X28E aggregate accuracies per bin/sweep point are
  already on disk (`final_eval.json`) and were known when this pre-registration
  was written. NOTHING about pressure (R-CAP fractions, windowed pressure,
  first-saturation indices) has ever been measured before this experiment;
  every pressure claim below is a genuine prediction.

## PRE-REGISTRATION (bound before measurement)

### The hypothesis under test (as directed, verbatim)

> **H1:** "At B0, per-example cumulative R-CAP fraction rises steeply once
> required live slots approach the 736-slot budget and saturates for k where
> accuracy collapses; at B0x4 the same k values show pressure well below
> saturation."

**The honest null (registered with equal force):** if pressure saturates
uniformly regardless of accuracy (e.g. B0x4 also saturates where it holds),
or accuracy collapses without pressure saturation, REPORT THAT — the verdict
is then REFUTED with the exact numbers, not reinterpreted.

### Operational definitions (bound now, before any pressure number exists)

All quantities are computed per example, within the word only (tokens before
the word's EOS; PAD-region tokens excluded), from the frozen package's
`RoutingTrace`:

- **Append request** (per token, per head): routing case ∈ {R-EMPTY, R-NOMATCH,
  R-ZERO-RESIDUAL, R-CAP}. R-CAP is an append request that degraded to a merge
  because the store was FULL (the architectural saturation policy).
- **R-CAP fire:** routing case == R-CAP.
- **Cumulative pressure** `P(ex) = (#R-CAP fires in word) / (#append requests
  in word)`, pooled over tokens and heads; defined as 0 when the denominator
  is 0. Also recorded per head.
- **Windowed pressure register** `W64(t)`: over the trailing 64-token window
  ending at t, `(#R-CAP fires) / (#append requests)`; defined only when the
  window contains ≥ 1 append request (sensitive register, PRIMARY). A
  **conservative variant** `W64c` requires ≥ 16 requests in the window
  (registered as secondary).
- **Saturation event:** `W64(t) ≥ 0.9`. **first_sat(ex)** = smallest in-word
  token index t with a saturation event (else none). **Sustained saturation
  index** `sust_sat(ex)` = smallest t such that every DEFINED register value
  at t' ≥ t (up to the word's EOS) is ≥ 0.9 (else none).
- **Lead/lag:** for accept words (a^k b^k, token layout SOS | a's at
  1..k | b's at k+1..2k | EOS at 2k+1): `lead_b = b_start − first_sat` with
  b_start = k+1 (positive = pressure saturates BEFORE any b — i.e. before the
  answer's evidence begins to arrive); `lead_eos = eos_idx − first_sat`
  (positive = saturation precedes the verdict position).
- **Peak live slots:** per (example, head) from the trainer's own
  `trace_live_series` occupancy instrumentation, mean over heads (the F7
  receipt machinery, reused read-only).

### Adjudication thresholds (bound now, no post-hoc moves)

- **COLLAPSE at k:** balanced accuracy < 0.75. **HOLD:** ≥ 0.99.
- **Cumulative pressure saturation:** mean P over accept words ≥ 0.9.
  **"Rises steeply":** mean P increases monotonically (up to noise ≤ one
  inverted pair) across consecutive probed k past the budget.
- **"Well below saturation" (the B0x4 comparison):** mean P < 0.5 at every
  probed k in the comparison set, and zero W64 saturation events.
- **k_landmarks** (per arm, from its own sweep + bins): `k_p50` = smallest
  probed k with mean accept-word P ≥ 0.5; `k_psat` = smallest with P ≥ 0.9;
  `k_acc99` = smallest with balanced accuracy < 0.99; `k_acc75` = smallest
  with accuracy < 0.75.

### Registered predictions (falsifiable, numbered)

- **P1 (B0 boundary behavior):** B0 shows ZERO W64 saturation events on
  accept words at all probed k ≤ 720; at EVERY probed k with collapse
  (accuracy < 0.75) a majority of accept words show a saturation event.
- **P2 (B0 cumulative rise):** B0's mean accept-word cumulative pressure is
  ≈ 0 for all probed k ≤ 720 and rises steeply for k > 736, with k_p50 inside
  [736, 1152] (i.e. pressure ≥ 0.5 somewhere in the 9x band) — and the
  directed claim's strict form (cumulative ≥ 0.9 at collapse k's) is tested
  as written; if cumulative stays below 0.9 at collapsing k's while W64
  saturates there, H1 is adjudicated MIXED with both facts reported.
- **P3 (B0x4 at B0's collapse depths):** at the same k values where B0
  collapses (the sweep band 736–864), B0x4 holds (accuracy ≥ 0.99), shows
  zero saturation events and mean accept-word P < 0.05.
- **P4 (B0x4 at the 9x band):** at k ∈ [1025,1152] B0x4 holds (accuracy
  ≥ 0.9), with mean accept-word P < 0.5 and no sustained saturation.
- **P5 (generalization at B0x4's own budget):** B0x4's saturation events and
  cumulative-pressure rise begin only beyond its 2944-slot budget (first
  events at k > 2944), and its collapse (accuracy < 0.75) coincides with
  pressure saturation.
- **P6 (lead — the "predictor" claim):** on B0 accept words at collapsing k,
  the median first_sat precedes the first b token (lead_b > 0) — pressure is
  observable before any answer evidence arrives, and lead_eos ≥ 64 tokens.

### What would count as the capability claim

If P1, P3, P4, P5, P6 hold and P2 holds at least in its "steep rise + windowed
saturation at collapse" form, the finding is: **the pressure register is an
inference-time observable that fires exactly at the budget boundary, before
the answer region, and separates holding from collapsing depths at both
budgets — a predictor of imminent capacity collapse.** Any failure of P1/P3/P4
at the separation claim or a null outcome forces the honest null.

---

## Measured values (filled after the run — nothing below existed before it)

Run identity: freeze manifest `0b674c71b5d9…aa11b4` re-verified by the probe
(185 files); checkpoints loaded read-only, sha256s as bound above; all 8 bin
balanced accuracies reproduce the X28E `final_eval.json` receipts
**byte-identically** (B0: 0.9941/0.9805/0.9844/0.5000; B0x4:
0.9902/0.9805/0.9707/0.9473). Extractor self-test vs brute-force oracle:
24/24 PASS. Walls: B0 22.6 s, B0x4 1307.6 s on leased card 5 (cards 2/3/4
at 99–100% util with other experiments during the run; card 7 excluded).

**B0 (B=736):** zero pressure at every holding depth (k ≤ 720: cumP 0.0000,
zero events, peak = k+1). Collapse (acc 0.50, accept recall 0.0) begins at
**k=736 exactly**, where the first R-CAPs fire (8 per word — the final
b-append and the EOS). Cumulative pressure then rises steeply and
monotonically: 0.0027 (736), 0.0442 (752), 0.0823 (768), 0.1175 (784),
0.1501 (800), 0.1804 (816), 0.2086 (832), 0.2349 (848), 0.2596 (864);
bin_9x 0.4407 (k=1025) → 0.5312 (k=1152). **It never reaches 0.9.**
Windowed W64 saturation fires on 100% of accept words at every k ≥ 768
(median first_sat = 1524 = 2B+52) but NOT at k=736/752 (collapse without a
windowed event; raw R-CAP does fire there). Word-level: any-R-CAP ⟺
accept-incorrect, 1408/1408, PPV 1.000, sensitivity 1.000, zero false
alarms on 1,792 correct accepts.

**B0x4 (B=2944):** bins hold with pressure 0.0000 (9x: acc 0.9473, peak
1134 ≪ 2944 — P4 supported). Own sweep: acc 0.832 for k=2816–3032 with
cumP 0.0000 through k=2928 — the 0.832 is entirely count_mismatch
(j=k±1) failures, off-by-one count blindness pressure cannot see. First
R-CAP at **k=2944** (demand ~k+2 crosses the budget); W64 saturation from
k=2976 (first_sat 5940 = 2B+52); accept-collapse (acc 0.50) at **k=3040** —
pressure leads collapse by 96 depth, saturation by 64; every collapsing
word is saturated. P5 supported.

**The honest null fires (finding of record):** at the shared sweep points
B0x4 **collapses at k=608–816 with pressure identically 0.0000** and peak
live ≈ k+1 ≤ 849 (3.5× under budget), holding again at 832–864 — a
non-capacity collapse band (confirmed by an independent minimal forward:
near-flat accept margins −0.003 in the band, +0.000..0.001 above it).
This refutes P3's accuracy clause (“B0x4 holds at B0's collapse depths”)
and half of H1's implied contrast; the pressure clause of P3 holds (0.0000,
zero events at all shared k). Additionally B0x4 k=2976–3024 shows windowed
saturation at acc 0.832 (saturation without registered-criterion collapse).

**Mechanism (single-word dump, B0 k=864):** all 4 heads in lockstep — SOS
R-EMPTY, then alternating R-CONFLICT (no allocation) / R-NOMATCH (append)
on every token through BOTH phases; POP never fires in any traced word of
either arm; the store fills at fixed position ≈ 2B−2 (token 1470), after
which every token on every head fires R-CAP. The word's append demand is
~k+2 slots, so the budget binds at k ≈ B (measured cliffs: B0 k=736,
B0x4 pressure onset k=2944). P6's registered form is refuted: saturation is
mid-b-phase (lead_b < 0); it precedes the EOS verdict by 13–205 tokens
(B0 sweep, deterministic 2k−1471), 527–781 tokens (B0 9x accepts),
141–205 tokens (B0x4 collapsing words).

Landmarks: B0 k_acc99 = k_acc75 = 736 = first R-CAP; k_p50 = 1103; k_psat =
none (max cumP 0.531). B0x4 k_acc99 = 2816; k_acc75 = 3040; first R-CAP =
2944; k_p50 = k_psat = none (max 0.081).

## Verdict

**H1: MIXED** — precisely the pre-registered mixed case: cumulative R-CAP
fraction rises steeply at the budget (and k_p50 = 1103 lies inside the
registered [736, 1152]) but never saturates ≥ 0.9 at any probed depth,
while the windowed register saturates exactly across the collapse region
(k ≥ 768 at B0). The honest null fires: accuracy collapses WITHOUT
pressure (B0x4 hole band k∈[560,816], 1,792 accept words, pressure 0,
occupancy ≤ 29% of budget), and pressure saturates without the registered
collapse criterion at B0x4 k=2976–3024 (acc 0.832, saturation leading
collapse by 64 depth).

Per-prediction: P1 partial (holds at k ≥ 768; k=736/752 collapse without
windowed events though raw R-CAP fires); P2 mixed (steep rise ✓, k_p50 ✓,
cumulative ≥ 0.9 ✗ — max 0.53); P3 refuted as stated (pressure clause ✓,
accuracy clause ✗ — B0x4 collapses at 608–816 with zero pressure); P4
supported; P5 supported; P6 refuted as registered (saturation mid-b-phase),
reformulated: the register precedes the EOS verdict by 13–781 tokens.

**The defensible claim:** the pressure register is a *specific*
(inference-time observable, one-directional) predictor of **capacity**
collapse — at B0 it separates collapsing from holding words perfectly
(PPV 1.000 / sensitivity 1.000 over 6,400 words) and the cliff coincides
exactly with the first degraded append; at B0x4 it fires at the budget and
leads accept-collapse by 96 depth. It is NOT a sensitive predictor of
collapse in general: a large non-capacity collapse band exists that it
correctly does not flag. Full analysis: REPORT.md; scatter data:
data/*.jsonl, data/*.json.
