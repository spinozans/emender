# WHEELER-ORDER CAPACITY PROBE (SPAN-RUN): does an ordered, run-coalescing store hold Theta(r x run) of context mass?

Status: **predictions pre-stated (this revision); sweep pending.** The
predictions below were committed before any full-protocol training run.
Results, receipts and the verdict are appended after the sweep lands; nothing
in the prediction section is edited after the fact.

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

## 5. RESULTS (pending the sweep)

To be appended: the saturation curve (accuracy vs span length, both arms,
best-LR means with per-seed cells), the P-3 eviction receipts, the P-4
share/accuracy receipts, the run-length histograms and the Wheeler-ness
contiguity, and the verdict against §4 (confirmed / refuted / partial /
inconclusive), with the honest list of what oracle level does not establish.
