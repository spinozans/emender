# WHEELER-ORDER CAPACITY PROBE (SPAN-RUN): does an ordered, run-coalescing store hold Theta(r x run) of context mass?

Status: **predictions pre-stated; sweep COMPLETE — the verdict is in §6.**
The predictions below were committed before any full-protocol training run
(two mid-sweep engineering recoveries are disclosed in §4 and §5.4's
lineage); nothing in the prediction section was edited after the fact.

Design reference: [`DESIGN_WHEELER_ORDER.md`](DESIGN_WHEELER_ORDER.md)
(operator-confirmed semantics: derived 1-D key order; the interval read with
full-population softmax denominator and outside-interval numerator exactly 0;
update-in-run accumulates — the registration-level accumulate-vs-overwrite
distinction; MERGE-before-POP pressure priority with `tau_run = w` — the
losslessness rule). Probe: `wheeler_capacity_probe.py`; machine-checks:
`test_wheeler_capacity_probe.py` (15 tests: interval-read invariance/omitted
mass, accumulate-vs-replace, MERGE-before-POP, POP, NONE, task construction).

## 1. The claim under test (the operator's capacity frame)

A pointwise store holds `r` memories; an ordered, run-coalescing store holds
`Theta(r x avg-run-length)` of context mass — runs are the compression
multiplier the identity-cache degeneration (RLE with run length 1, the
receipted 150 M diagnosis) measurably failed to achieve.

**Dependency (stated per the lane's discipline).** The parallel SEP-KV lane
(cards 2/3) tests the orthogonal axis — whether key-value separation is
necessary at all. Both arms here are built on the SEP-KV minimal
separated-key base (the pointwise arm IS that arm verbatim at rank 8); if
SEP-KV refutes separation entirely, these results RECLASSIFY (the order-axis
mechanisms carry, the arm definitions no longer sit on the minimal base).

## 2. Task SPAN-RUN and arms

Window: `BOS, PAD, PAD, (m_1,v_1)..(m_S,v_S), d_1, d_2, probe(m_j) -> v_j`
— a contiguous span of S combined (member, value) binding tokens (members
without replacement from a 64-token pool, values per window), a 2-token
distractor run from a separate pool, one probe at a uniform random span
position j, target = that member's registered value, value-restricted argmax
over 32, **chance = 1/32 = 0.03125**. Every binding is probeable, so the
pointwise arm cannot drop span writes: distinct bindings need distinct live
rows (the SEP-KV collision lemma).

Why per-member continuations (the confirmed v6): a single span-wide
continuation is solvable by the identity cache alone — one row, serial
CORRECT, the latest write carries the whole answer — so it cannot separate
the arms. Here the binding is the member→value MAP of the span: the
"compressed function of the span" the probe demands is a per-address read
out of the span's compressed representation.

Arms (both rank r = 8, eager oracles, ~70k params, depth 2):
* **POINTWISE** — the SEP-KV separated arm verbatim: free 16-D keys,
  registered five-op policy, CORRECT = erase-and-replace (run length 1).
* **ORDERED** — scalar keys (derived order), interval read (w = 0.25,
  full-population denominator), update-in-run accumulate, MERGE-before-POP
  (tau_run = w).

## 3. The store arithmetic (the derived collapse point, verbatim)

The pointwise arm's slot demand before the probe is S (bindings) + 2
(distractor writes) + 2 (BOS/PAD rows) = **S + 4**, against a budget of
r = 8. The **collapse point is therefore S_collapse = 8**: S = 2 gives 6
rows (holds), S = 4 gives 8 rows (holds at exactly the budget), S = 8 gives
12 rows (the first swept S past capacity). The sweep {2, 4, 8, 16, 32, 64}
brackets it.

**Refinement, disclosed before any run.** The confirmed frame called S = 8
the collapse point; deriving the quantitative curves refines what "collapse"
means mechanically. The registered POP chain destroys the most-dissimilar row
under pressure, and an erasing CORRECT replaces a row's content — so past
capacity each additional span write kills or replaces ~one binding row, but
the PROBED binding is hit with probability ~1 − r_eff/(S + 2), not ~1. The
predicted pointwise curve is therefore an **eviction-onset decay**, not a
cliff: an accuracy mixture `q·L(S) + chance·(1 − L(S))` with live fraction
`L(S) = min(1, r_eff/(S + 2))`, `r_eff ∈ [6, 8]` (8 slots minus 0–2
sacrificial/junk rows). The eviction ONSET is still exactly S = 8 (the
derived point), receipted mechanically below (P-3): the destroyed-fraction
jumps from ~0 at S ≤ 4 to ≥ 0.2 at S = 8 and grows ~1 − r_eff/(S + 2).

## 4. PRE-STATED PREDICTIONS (committed before any training run)

Protocol (as amended below, before the sweep): per-arm best-effort LR sweep
{3e-3, 1e-3, 3e-4} x 3 seeds {3401, 3402, 3403}, 3000 steps x **batch 1024**,
fixed eval set per span (seed 4999, 4096 windows) shared by every cell,
receipt windows 512 (fresh batch, seed 6999, in the mechanism pass, run
per arm when the sweep is split by arm across the 4/5 card pool).
Comparison statistic: the best-LR MEAN over seeds, per (arm, S), with
per-seed values reported.

* **P-1 (precondition: the task is learnable at probe budget).** BOTH arms
  reach value accuracy >= 0.85 at best LR at S = 2 (mean over seeds, every
  seed >= 0.70). If this fails the probe gradient is unresolved at this
  scale: report INCONCLUSIVE, tune within the probe budget, make no claim.
* **P-2 (pointwise: the eviction-onset decay, quantitatively).** At best LR:
  S = 2, 4: >= 0.85. S = 8: in [0.50, 0.78]. S = 16: in [0.28, 0.45].
  S = 32: in [0.15, 0.25]. S = 64: in [0.09, 0.16]. Every cell at S = 64
  <= 0.25 (no cell may exceed the r_eff/S ceiling).
* **P-3 (the mechanical saturation receipt — the operator lane's
  requirement).** The pointwise occupancy/eviction logs show: probed-binding
  destroyed-or-never-filed fraction <= 0.15 at S ∈ {2, 4}; >= 0.2 at S = 8;
  >= 0.55 at S = 16; >= 0.75 at S = 32; >= 0.85 at S = 64 — the answer
  evicted by span members' writes, as the store arithmetic says. Stratified
  accuracy: acc_when_intact within 0.1 of the arm's S = 2 accuracy at every
  S; acc_when_destroyed <= 3x chance; the deficit
  `1 − acc` tracks `1 − L(S)` within 0.15 at every S.
* **P-4 (ordered: the dilution curve, quantitatively — the operator lane's
  requirement).** The ordered arm holds the probed binding's MASS at every S
  (row_live_frac >= 0.9 at every S — nothing is evicted); accuracy is
  limited by DILUTION, with the true value's share of the probe's interval
  read `s(S) ≈ rho · K_span / S`, K_span ∈ [4, 6] rows for the span, rho ∈
  [0.5, 1] interval mass concentration. Predicted share bands: S = 8:
  [0.20, 0.75]; S = 16: [0.10, 0.40]; S = 32: [0.05, 0.20]; S = 64:
  [0.03, 0.10]. Predicted accuracy bands (share -> win-probability under the
  value-collision heuristic, ~B ≈ 2S/K_span co-read bindings): S = 8:
  [0.50, 0.90]; S = 16: [0.35, 0.80]; S = 32: [0.25, 0.65]; S = 64:
  [0.18, 0.50].
* **P-5 (the separation — the claim).** At best-LR means, ordered > pointwise
  at every S ∈ {8, 16, 32, 64}, with ratio >= 1.1x (S = 8), >= 1.3x (S = 16),
  >= 1.6x (S = 32), >= 2.0x (S = 64). The ordered curve also decays —
  per-address resolution is row-limited in BOTH arms (stated honestly in the
  design doc §8); the capacity claim under test is that MASS-PRESERVING
  dilution beats MASS-DESTROYING eviction, at every probeable span position.
* **P-6 (run-structure receipts — does the store actually build runs?).** In
  the trained ordered arm: the run-length histogram at the probe is
  non-degenerate from S = 8 (max run length >= 2) and grows with S (mean run
  length >= 2 at S >= 16, max >= S/8 at every S); the span occupies <= 6
  rows at every S. **Wheeler-ness (operational)**: mean key-order contiguity
  of the span's writes >= 0.5 at every S in the ordered arm, with < 2
  interleaved foreign writes on average; the pointwise arm receipts its own
  key channel for contrast. The interval read's omitted mass is receipted
  per probe (retained vs omitted disclosed, per the Lean truncation law).
* **Refutation clause (verbatim commitment).** If (i) any pointwise (LR,
  seed) cell reaches value accuracy >= 0.5 at any S >= 16 — an escape beyond
  the r_eff/S slot arithmetic, i.e. a compression mechanism this design did
  not grant it — or (ii) the ordered arm's best-LR mean does NOT exceed the
  pointwise arm's at S = 32 or S = 64, the capacity-multiplier claim is
  WRONG AT THIS SCALE and is reported as a refutation, not reframed.
* **Inconclusive clause.** P-1 failure at S = 2 for either arm -> no claim;
  tune within the probe budget and say so.

**Honest pre-registration disclosure (recorded before the sweep).** (a) A
harness smoke run (200 steps, one ordered-arm S = 8 cell, CPU, far from
convergence, value_acc 0.045 vs chance 0.031) confirmed the plumbing end to
end. (b) Two calibration cells at the ORIGINAL committed budget (3000 steps
x batch 256, lr 3e-3, seed 3401, S = 2): pointwise 0.3125, ordered 0.5159 —
BOTH curves still rising monotonically at step 3000 (loss 2.34 -> 1.83 and
2.39 -> 1.71), i.e. the budget, not the task, was the binding constraint.
Per the inconclusive clause ("tune LR/steps within the probe budget"), the
protocol was amended to 3000 steps x batch 1024 (4x the exposure, ~identical
walls: the eager per-token oracle is latency-bound, measured 233 s/cell at
both batch 256 and 1024) BEFORE the sweep, and the amendment is committed
before any sweep cell runs. (c) Feasibility at the amended budget (the P-1
fix, the separator lane's disclosed-calibration role): S = 2, seed 3401,
3000 x 1024: pointwise 0.9717 (lr 3e-3) and 0.9744 (lr 1e-3); ordered 0.9795
(lr 3e-3). No other cell has been run; no S >= 4 cell at any budget has
been run at the time of this amendment; the predictions above are not
conditioned on any S >= 4 training result. The stale calibration receipts
(`wheeler_*_s2_*` at batch 256) are superseded by the sweep's own cells and
are not part of the evidence for §4.

**(d) Mid-sweep engineering recovery, disclosed.** The pointwise arm's
receipt harness crashed at its first S = 4 cell (a CUDA device-side assert:
the probed-binding fate loop gathered slot/op at indices that can run past
the window for later-probed bindings — the per-window gate was applied after
the gather; the 9 committed S = 2 cells all landed before the crash and are
unaffected). The fix clamps the gather indices before the mask
(`_pointwise_binding_fate`); the relaunched driver REUSES the committed
per-cell receipts via a `--resume` path (cells are deterministic; the reused
receipts are byte-identical to what their own runs wrote — an engineering
recovery, never a protocol change; resumed cells are flagged `resumed` in
the summaries). The ordered arm's receipt path never had the bug and its
sweep ran uninterrupted.

## 4b. MIMETIC-INIT EXTENSION (operator-directed; pre-registered before its
## cells run; the standard-init P-1 PARTIAL receipt stands untouched)

**Variant `ordered_mimetic`.** The key/query-key projections are born AT the
fixed-family geometry: only the CURRENT-token context block is non-zero,
a fixed seeded unit direction (seed 3401) scaled by gain 1.5, so at step 0
the emitted scalar key is `tanh(1.5 <e_token, u>)` — a FIXED function of
token identity (the registered fixed-key behaviour: address = identity)
whose spread over the vocabulary is an invariant of the embeddings, not a
training outcome. Training then drifts the projections freely (weight decay
applies as everywhere). Motivation: the free-key arm's collapse receipt
(seed 3402: `span_key_std -> 0.000`, one run absorbing the window, chance
accuracy — init luck decided the entry race between key spread and store
collapse; succeeding seeds were born at `span_key_std ~ 0.016-0.018`).
Program precedent: the X42 mimetic-geometry verdict
(`experiments/expressivity_tasks/x42_pile_scale/receipts/mimetic_geometry/VERDICT.md`
— "mimicry improves materially while unfreezing does not: early-training
optimization is the stronger lead").

**Pre-stated predictions (committed before any mimetic cell runs):**
* **M-1 (safe birth repairs the precondition at S = 2).** The mimetic arm at
  S = 2: EVERY seed >= 0.70 at its best LR (including the collapsed seed
  3402's condition), best-LR mean over seeds >= 0.85 — P-1's bound, repaired
  by the init, not by tuning.
* **M-2 (no basin entry, receipted).** End-of-training `span_key_std` >=
  0.016 (the succeeding-seed range) in EVERY (LR, seed) cell at S = 2; no
  cell shows the collapse signature (span_key_std < 0.005 with one run
  absorbing the window).
* **M-3 (no adverse interaction with the capacity mechanism).** At S = 32
  (3 seeds at the mimetic arm's own best S = 2 LR — a spot-check, not a
  comparison arm): accuracy within 0.05 of the FREE-KEY ordered arm's S = 32
  best-LR mean counts as "no interaction"; a deficit > 0.10 is an adverse
  interaction, reported honestly; anything in between is reported as
  inconclusive-on-interaction with the receipts.
* **Refutation clause (verbatim from the operator's directive).** If the
  mimetic arm ALSO collapses, report it — the basin would be init-independent
  and the rotation-constrained family becomes the only safe form.

**Scheduling disclosure.** The mimetic cells run AFTER the committed S = 8..64
capacity cells finish (they do not preempt them; the extension is gated on
the committed sweeps' exit). Protocol: 3 seeds x 3 LRs at S = 2 (9 cells),
then 3 seeds x 1 LR at S = 32 (3 cells), same 3000 x 1024 budget, same
receipt machinery; the mechanism pass receipts the mimetic arm's run-length
histogram and Wheeler-ness contiguity like any ordered arm.

## 5. RESULTS (INTERIM — the pointwise arm complete and committed; the
## ordered arm's cells and the mimetic extension still landing on their
## detached leased drivers; this section records only landed, committed
## evidence; the verdict waits for the full grid)

### 5.1 The pointwise saturation curve (54/54 cells; best-LR mean over
### seeds, per-seed values; chance = 0.03125)

| S | best LR | mean | seed 3401 | seed 3402 | seed 3403 | P-2 band | in band? |
|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 1e-3 | **0.976** | 0.974 | 0.982 | 0.971 | >= 0.85 | yes |
| 4 | 3e-3 | **0.904** | 0.880 | 0.893 | 0.938 | >= 0.85 | yes |
| 8 | 3e-3 | **0.610** | 0.575 | 0.609 | 0.644 | [0.50, 0.78] | yes |
| 16 | 3e-3 | **0.303** | 0.303 | 0.324 | 0.283 | [0.28, 0.45] | yes |
| 32 | 3e-3 | **0.064** | 0.050 | 0.069 | 0.072 | [0.15, 0.25] | **below** |
| 64 | 3e-3 | **0.033** | 0.035 | 0.031 | 0.033 | [0.09, 0.16] | **below** |

P-1 MET for the pointwise arm (S = 2: mean 0.976, every seed >= 0.70). The
eviction onset lands at the derived S_collapse = 8 (the first swept S past
the S + 4 > 8 budget), and the decay past it is STEEPER than the committed
r_eff/(S + 2) law (S = 32/64 below the predicted bands, at ~chance by 64):
the honest reading is that the empirical surviving-binding fraction at
large S is ~0, not ~6/34. The refutation clause (any pointwise cell >= 0.5
at S >= 16) is NOT triggered — the pointwise arm found no escape.

### 5.2 The pointwise mechanism receipts (P-3; the seed-6999 batch, best
### models, both layers; `receipts/wheeler_mechanism_pointwise.json`)

The committed strata (layer 0 / layer 1): probed-binding destroyed fraction
0.22/0.19 (S=2), 0.35/1.00 (S=8), 0.72/1.00 (S=32), 0.87/1.00 (S=64) — the
predicted eviction growth (P-3's >= 0.2 at 8, >= 0.75 at 32, >= 0.85 at 64)
is MET at layer 0. The accuracy mixture at S = 8 is exactly the predicted
eviction-onset form: intact 0.63 of probes answer at 0.97, destroyed
probes at ~chance (0.05), giving the observed 0.61. At S >= 32 a second,
unpredicted-in-form failure mode compounds eviction: the intact rows stop
answering (layer-0 intact-accuracy 0.26 at 32, 0.02 at 64) because the
probe's read mass on the row falls to 0.15/0.13 — the free-key geometry
cannot give 32+ bindings distinct-enough addresses, so the softmax splits
its mass over ~7.5 live rows (all holding CORRECT-chained mixtures). The
span's rows are 94% live at S = 32-64 (the store HOLDS rows; their CONTENT
is the chained latest-write and the probe cannot ISOLATE it). P-3's last
sub-prediction (intact accuracy within 0.1 of the S = 2 accuracy) therefore
holds at S = 8 and fails at S >= 32 — reported as measured, with the
retrieval-dilution mechanism receipted above.

**Nothing in this interim section evaluates the capacity claim**: that
needs the ordered arm's S = 8..64 cells (in flight) against §4's committed
bands and refutation clause.

### 5.3 The mimetic-init extension (COMPLETE; §4b's cells all landed,
### committed by its own driver)

**M-2 (no basin entry) — MET, in full.** End-of-training `span_key_std` in
every (LR, seed) S = 2 cell: 0.68–0.90 (floor 0.016; the collapse signature
— std < 0.005 with one run absorbing the window — appears NOWHERE, at any
LR, any seed, and the S = 32 spot-check cells hold 0.87–0.90 too). Safe
birth eliminated the basin: **the key-collapse is init-addressable, and the
operator's diagnosis (init luck decided the entry race between key spread
and store collapse) is confirmed by the receipts** — the previously
collapsed seed 3402 learns in every mimetic cell (0.46–0.67 vs 0.045 free-
key at 3e-3), and runs form (max run length 4–7, contiguity 0.35–0.69).

**M-1 (every seed >= 0.70, mean >= 0.85 at S = 2) — NOT MET as written.**
Per-seed bests: 0.817 (3401), 0.672 (3402), 0.707 (3403); best-LR mean
0.72. Every seed learns far above chance and the collapse is repaired, but
the arm does not reach the free-key arm's lucky-seed accuracy (0.98) at
this budget: the mimetic init guarantees SPREAD, not the learned ALIGNMENT
(query key lands on its binding's row) that the lucky free-key seeds
found; 3000 x 1024 moves it partway (0.52–0.82 across cells). The bound
was over-optimistic; the honest reading is a partial repair: basin gone,
alignment still to be earned by training. **The refutation clause (mimetic
also collapses) is NOT triggered** — no collapse occurred.

**M-3 (S = 32 spot-check) — pending the free-key ordered arm's S = 32
cells** (still landing): mimetic S = 32 at its own best S = 2 LR (3e-3):
0.069 / 0.067 / 0.082 (mean 0.073). The comparison against the free-key
arm's S = 32 best-LR mean is made in the final verdict. Notably the
mimetic cells show no basin at S = 32 either (key_std 0.87–0.90), so the
long-span dilution is init-independent — consistent with dilution being
the capacity mechanism's honest price rather than a training pathology.


### 5.4 INTERIM — the ordered arm's landed cells (S = 2..16 complete,
### S = 32 in flight, S = 64 pending; the verdict WAITS for the full grid)

The ordered arm's best-LR means so far (collapsed seed 3402 included in
every aggregate, per the supervisor's direction): S = 2: 0.624, S = 4:
0.269, S = 8: 0.439, S = 16: 0.138 — against pointwise's 0.976 / 0.904 /
0.610 / 0.303. **P-5's separation is currently FAILING at S = 8 and 16**
(ordered below pointwise at both), and P-4's bands are missed (S = 8: 0.439
vs [0.50, 0.90]; S = 16: 0.138 vs [0.35, 0.80]). The S = 32 comparison is
neck-and-neck so far (pointwise 0.064; the first ordered cell 0.063); S =
32/64 decide the committed refutation clause (ii) — the verdict is written
only when they land.

The receipts give the mechanism, and it is NOT the predicted one. **The
run-structure receipts are POSITIVE (P-6):** runs form (mean run length
~3.9-4.2, max >= 7 at S = 16), the span compresses into ~5 rows of the 8
(occupancy 5.0-5.4), NOTHING is evicted (probed binding's row live in
100% of probes at every S — the mass-preserving semantics work as
designed), and the Wheeler-ness contiguity is real (0.71-0.73 at S = 16
vs 0.42-0.47 at S = 4 — the trained key order places the span's writes
contiguously). **But the dilution is steeper than the pre-registered
rho*K_span/S model:** the measured law is `true_value_share ~=
probe_mass_on_row / run_length` (0.37/4 ~ 0.09 at S = 16, measured
0.06-0.09; P-4's band [0.10, 0.40]), because (a) the probe's interval
(registered w = 0.25 against a learned key spread of ~0.9) catches several
rows, splitting the read mass (mass-on-row only 0.26-0.38, not rho ~ 1),
and (b) each row accumulates ~4 bindings. The per-address resolution
decays with S as 1/(rows in interval x run length) — the same
row-granularity ceiling the design doc §8 predicted for BOTH arms, but the
ordered arm pays it through blending while the pointwise arm pays it
through eviction-plus-retrieval-dilution, and at this budget the pointwise
payment is smaller at S = 8-16.

The S = 4 receipts also show the resolution-vs-alignment tradeoff the
mimetic section flagged: the ordered seeds' key spread GROWS from 0.018
(S = 2, where near-collapsed keys put every row inside every query's
interval — alignment free, resolution irrelevant at 2 bindings) to 0.5-0.9
(S = 4+), after which the probe must LEARN to place its query key within
the interval of its own binding's row — and at 3000 x 1024 it only
partway does (probe mass-on-row 0.26-0.38). This is a training-budget
fact, not a store-capacity fact; the receipts separate the two.

---

## 6. THE FINAL RESULTS AND VERDICT (108 committed cells + the 12-cell
## mimetic extension; every number below is from the committed receipts)

Interim sections 5.1-5.3 stand as final for the pointwise arm and the mimetic
extension; 5.4's landed-cell numbers are superseded by the complete grid below.

### 6.1 The saturation curve, both arms (best-LR mean over seeds, chance 1/32)

| S | pointwise | ordered | ord/pt | committed P-5 ratio | met? |
|---:|---:|---:|---:|---:|---|
| 2 | 0.976 | 0.624 | 0.64x | (P-1 only) | pt: P-1 MET; ord: PARTIAL |
| 4 | 0.904 | 0.269 | 0.30x | — | — |
| 8 | 0.610 | 0.439 | 0.72x | >= 1.1x | **NO** |
| 16 | 0.303 | 0.138 | 0.46x | >= 1.3x | **NO** |
| 32 | 0.064 | 0.064 | 1.01x | >= 1.6x | **NO** |
| 64 | 0.033 | 0.038 | 1.15x | >= 2.0x | **NO** (both ~chance) |

Per-seed cells at each arm's best LR are in the committed per-cell receipts
(`receipts/wheeler_{arm}_s{S}_lr*_seed*.json`) and the canonical sweeps
(`receipts/wheeler_{arm}_sweep.json`).

### 6.2 Evaluation against the pre-registered predictions

* **P-1 (precondition).** Pointwise: MET (0.976 mean, every seed >= 0.71).
  Ordered: **PARTIAL** — seeds 3401/3403 learn (0.979/0.846) and seed 3402
  fell into the receipted key-collapse basin (`span_key_std = 0.000`, one
  run absorbing the window, chance accuracy; §5.4, the supervisor-directed
  handling: the collapsed seed stays in every aggregate, no bound-rescue
  tuning). 2 of 3 ordered seeds learn the task; the task IS learnable by
  the ordered machine — but not by every init at this budget.
* **P-2 (pointwise decay).** S = 2, 4: >= 0.85 MET (0.976, 0.904). S = 8:
  0.610, in band. S = 16: 0.303, in band. S = 32: 0.064 **below** band;
  S = 64: 0.033 **below** band (at chance) — the empirical collapse past
  S = 16 is steeper than the committed r_eff/(S+2) law.
* **P-3 (the mechanical saturation receipt).** MET in its essentials: the
  pointwise store's probed-binding destruction grows with S (layer-0
  destroyed fractions 0.22 / 0.36 / 0.35 / 0.54 / 0.72 / 0.87 at
  S = 2..64), the S = 8 accuracy IS the predicted eviction-onset mixture
  (intact 63% of probes answering 0.97, destroyed probes ~chance -> 0.61),
  and the committed growth thresholds are met at S = 8 and 64 with marginal
  misses at 16 (0.535 vs 0.55) and 32 (0.723 vs 0.75). The committed
  "intact rows keep answering" sub-prediction FAILS at S >= 32 (intact
  accuracy 0.26 / 0.02): the receipted second failure mode is retrieval
  dilution — the probe's read-mass share on its row falls to ~0.13 as the
  free-key geometry cannot isolate 32+ bindings across ~7.5 live rows.
* **P-4 (ordered dilution).** MISSED: accuracy below the committed bands at
  every long span (0.439 vs [0.50,0.90]; 0.138 vs [0.35,0.80]; 0.064 vs
  [0.25,0.65]; 0.038 vs [0.18,0.50]). The share receipts are mixed (in band
  at S = 32: 0.093 vs [0.05,0.20]; ~top of band at S = 64: 0.109 vs
  [0.03,0.10]; below at S = 8/16), and the share -> accuracy mapping was
  too optimistic: the read blends ~5-13 co-held values at shares comparable
  to the true value's, so the argmax loses far more often than the
  committed collision heuristic predicted. The MEASURED dilution law, from
  the receipts: `true_value_share ~= probe_mass_on_row / run_length` —
  the interval (fixed w = 0.25) splits its mass across the rows it catches
  against a learned key spread of ~0.9 (mass-on-row 0.24-0.53), and each
  row blends its accumulated run (~4-9 bindings at S = 16..64).
* **P-5 (the separation — the claim).** **FAILED at every long span: the
  ordered arm does not beat the pointwise arm at S = 8 or 16, is a
  statistical tie at S = 32** (0.0643 vs 0.0635; the pointwise per-seed
  cells 0.050-0.072 overlap the ordered 0.062-0.068), and at S = 64 both
  arms sit at/near chance (0.038 vs 0.033; the committed >= 2.0x ratio is
  1.15x).
* **P-6 (run-structure receipts — did the store build runs?).** **MET, and
  this is the probe's positive finding:** mean run length grows 2.4 / 2.7 /
  3.9 / 5.0 / 8.6 at S = 2/8/16/32/64 (max 39 at S = 64), the span
  compresses into a handful of rows, **nothing is ever evicted**
  (probed-binding row live in 100.0% of probes at every span — the
  mass-preserving semantics work exactly as designed), the Wheeler-ness
  contiguity is real and GROWS with span (0.68 / 0.60 / 0.73 / 0.83 / 0.92
  at S = 2/8/16/32/64; the one sub-0.5 value is 0.47 at S = 4), and the
  interval read's disclosed omitted mass grows with occupancy (0.08 -> 3.9)
  as the Lean truncation law requires it to.
* **Mimetic extension (§4b).** M-2 MET IN FULL (no basin entry anywhere;
  key spread held 0.68-0.90 in every cell — the collapse is init-addressable
  and the operator's entry-race diagnosis is confirmed). M-1 NOT MET as
  written (best-LR mean 0.72 vs 0.85): safe birth guarantees spread, not
  the learned query-to-row alignment the lucky free-key seeds found.
  Refutation clause (mimetic also collapses) NOT triggered. M-3: the
  S = 32 spot-check (mimetic mean 0.073) is within 0.009 of the free-key
  ordered arm (0.064) — **no adverse interaction with the capacity
  mechanism**, and no basin at S = 32 either (key spread 0.87-0.90): the
  long-span dilution is init-independent.

### 6.3 The verdict (per the committed clauses, not reframed)

**REFUTED AT THIS SCALE.** The capacity-multiplier claim, as pre-registered
by P-5 (the ordered, run-coalescing store beats the pointwise store at long
spans with growing margins), is wrong at this probe's scale and budget: the
ordered arm loses at S = 8 and 16, ties at 32, and sits at chance with a
1.15x nominal edge at 64. The refutation clause's trigger (i) — a pointwise
escape — did NOT fire (the pointwise collapse was honest). Trigger (ii)'s
letter (ordered <= pointwise at S = 32 OR 64) evaluates as NOT fired only
because of nominal exceedances of 0.0008 and 0.005 at chance level —
statistical ties, disclosed here rather than claimed; P-5's committed
ratios failed at all four long spans regardless.

**What the refutation does NOT refute, because the receipts say it holds:**
the ordered store's mass-preservation machinery worked — runs built (mean
length ~r-scaling with S, max 39), the probed binding's mass was NEVER
evicted (100% row-live vs the pointwise arm's 13-87% destruction), and the
learned key order was measurably Wheeler-like with contiguity growing to
0.92. What failed is the CONVERSION of held mass into per-address
resolution at this budget: the interval read pays dilution
(~ mass-on-row / run-length) for every binding it holds, and the pointwise
arm's eviction pays less at mid spans because an intact pointwise row
answers cleanly while a diluted run row blends ~4-9 values the head cannot
argmax apart. Both arms hit the same row-granularity ceiling the design doc
§8 flagged as unproven; at this scale the ceiling binds the ordered arm
EARLIER in accuracy terms.

### 6.4 What oracle level does not establish (the honest list)

* **No kernels.** The ordered oracle costs ~7x the pointwise oracle's walls
  (per-token sorts and a larger autograd graph); that is an oracle
  artifact, not a property of the variant.
* **No real-text claims.** A synthetic binding task, 2154-token vocabulary,
  ~70k parameters, rank 8, windows of 9-71 tokens.
* **No board claims.** Nothing here touches nats/byte on natural text or
  any 150M-class comparison.
* **Training-scale caveat, receipted not assumed.** The ordered arm's
  probe must LEARN the query-to-row alignment once keys spread (mass-on-row
  only 0.24-0.53 at 3000 x 1024); the probe cannot separate "the
  architecture cannot resolve" from "this budget did not finish learning
  to". The pointwise arm had no comparable training-limited channel
  receipted.
* **The Theta(r x run) MASS claim was not directly tested.** SPAN-RUN
  probes per-address resolution, the resolution-side worst case for the
  ordered arm. The mass side — aggregate functions of a run retrieved by
  the interval read — is exactly the probe this refutation leaves open,
  and the mass-preservation receipts above are consistent with it.

### 6.5 The registration questions this leaves open (interleaved with SEP-KV's)

1. **The mass-vs-resolution fork (the headline).** The ordered variant's
   interval read blends a run's mass by design; a task that ASKS for the
   blend (the run's compressed function — counts, aggregate presence, the
   "compressed function of the span" the operator's frame names) is the
   right probe for the Theta(r x run) claim. SPAN-RUN's per-address probe
   refuted the resolution-side reading; the mass-side reading is untested
   and its machinery is receipted working.
2. **Safe birth vs rotation-constrained keys.** Now with receipts on both
   sides: the free-key basin (collapse) is init-addressable by mimetic
   birth (M-2), but neither form of spread automatically earns the learned
   query-to-row alignment the read needs (M-1, mass-on-row receipts). A
   registration should decide the key family WITH the alignment-training
   story attached.
3. **Interval width vs key scale.** The registered constant w = 0.25 against
   a learned spread of ~0.9 is what splits the probe's read mass across
   rows; an adaptive width, or a key-scale normalisation that pins the
   learned spread to the interval, is an open registration question the
   omitted-mass receipts make measurable.
4. **The SEP-KV dependency (§1).** If the SEP-KV lane refutes key-value
   separation entirely, these arm definitions reclassify per the committed
   clause; the order-axis mechanisms (interval read, accumulate,
   contiguity) carry, the base does not.

Files: `receipts/wheeler_{pointwise,ordered}_sweep.json` (the canonical
54-cell aggregates), `receipts/wheeler_mechanism_{pointwise,ordered}.json`
(the P-3/P-6 receipts), the 108 per-cell receipts, the 12 mimetic cells +
`receipts/wheeler_{ordered_mimetic_sweep,wheeler_mechanism_ordered_mimetic}.json`,
and the driver consoles in `wheeler_runs/`. The `.pt` best states are NOT
committed (program rule).
