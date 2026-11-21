# RESULT X46 — OURS RERUN2 (the dwell-respecting re-run) + THE COMPLETE FOUR-MACHINE ADJUDICATION

**Status: COMPLETE.** The ten ours arms of the dwell-respecting re-run
(2,000-step curriculum stages) are done (`x46_ours_rerun2.log`, 2026-11-21
03:24–06:51 UTC, card 2, rc=0 on every arm); the reference rows (n=64
controls, the GDN2 arms) are banked from the corrected full grid
(commit `ed8e2304` and its arm receipts). This document resolves the
growth-telemetry receipt contradiction, adjudicates the table against
the OPERATOR-SIGNED registration
(`PREREGISTRATION_X46_GROWTH_SEPARATOR.md`, signed 2026-11-19, §8 rulings
amended 2026-11-30), and states the X46 program's closing position.
The void history is receipted in `X46_SWEEP_VOID_RECEIPT.md`: the
supervisor's chain ran cold (`curriculum: []`) and is VOID; the ours
arms then iterated the ladder dwell — 300-step stages (chance at every
LR, the corrected-grid receipt `ed8e2304`), 400-step stages
(`x46_ours_rerun.log`: three of ten cells at chance, the rest still
climbing at cell end), and finally the 2,000-step stages of THIS run,
which the saddle receipts demand and which every number below uses.

## 1. The protocol of this run (the driver, from the log's own command lines)

The v2 driver is the `run_x46_leased.sh` wrapper pattern (the
`leased card: N (x46 arm ...)` echo is the wrapper's own line), invoked
once per arm, sequentially on leased card 2, each arm self-committing
its receipt and logging a `[rerun2 <date>] <machine> n=<n> lr=<lr>
rc=0` line. The exact invocation family, verbatim from the log:

```
x46 arm --machine {ours-fixed,ours-growable} --n {320,640} \
   --lr {3e-3,1e-3,3e-4} --seed 3401 --steps 1500 --batch 32 \
   --curriculum 64,128,192,256[,320,448,576,640] --curriculum-steps 2000
```

(the `n`-equal ladder stage is skipped by the runner; stages run BELOW
the target only). Ten arms: n=320 at all three LRs for both machines,
n=640 at {3e-3, 1e-3} for both. Measured phase: 1,500 steps, batch 32 —
matched exposure `steps × batch × n` on every arm, identical optimizer
(non-fused AdamW 0.9/0.95, wd 0.01, clip 1.0), constant-LR 90% +
cooldown 10%, PRIMARY = final value-restricted probe accuracy on the
fixed seed-4999 eval set (chance 1/128 = 0.0078). Receipts:
`receipts/x46_arm_ours-*_seed3401.json` (the v2 cells overwrite the
superseded same-name v1 cells; v1's printed blobs survive in
`x46_ours_rerun.log`, v1's receipts in git history).

**Protocol disclosures (before any number is read):**

* **Ladder asymmetry across families.** The ours v2 arms received
  2,000-step ladder stages (8,000 curriculum steps at n=320; 14,000 at
  n=640); the banked GDN2 rows ran the 300-step ladder (n=320: one
  300-step n=64 stage; n=640: 300 steps each at n=64 and n=320). The
  asymmetry is the saddle, receipted: the ours arms DWELL without the
  long stages (the 300- and 400-step rows are chance-or-climbing), the
  GDN2 arms are dwell-insensitive (0.93+ with the short ladder — their
  n=64 stage loss 1.70 shows the read sharpened immediately). The
  curriculum is protocol, not machine difference (§5, ruling 6); the
  stage-count difference is disclosed here and does not favor the ours
  arms' headline numbers — it is what makes them non-at-chance at all.
* **LR grid.** The n=640 ours cells cover {3e-3, 1e-3} of the
  registered {3e-3, 1e-3, 3e-4, 1e-4}; the ours plateau is LR-flat
  (below), so the best-cell reading is robust. The registered 3402
  seed confirmations exist for the GDN2 rows (3402 bests: gdn2-fixed
  0.8870, gdn2-growth 0.9602 at n=320 — confirming 3401) but NOT for
  the ours v2 cells (the 3402 ours rows in the receipts are superseded
  short-dwell cells). The ours v2 rows are single-seed (3401).

## 2. FIRST DUTY — THE RECEIPT CONTRADICTION, RESOLVED FROM THE CODE

**The contradiction.** Every ours-growable console blob in the log
prints `"growth_events": []` — zero events — yet prints
`"final_rank": [512, 1024]` (n=320) / `[1024, 2048]` (n=640) and ran
3.2–3.3x slower than ours-fixed at n=320 (809–825 s vs 256–263 s) and
6.9–7.1x at n=640 (4,074–4,131 s vs 584–597 s). Something grew.

**The resolution: (a) — growth fired and is fully receipted; the
console blob prints the WRONG KEY. It is a receipt-print bug, not
missing telemetry and not a machinery difference.** The code is
unambiguous:

* `run_arm`'s console summary prints exactly the keys `(machine, n, lr,
  seed, metrics, growth_events, final_rank, final_heads, ceiling,
  walls_s)`. The `growth_events` field of the receipt is populated by
  `_process_events` — the **GDN2/C3 optimizer-surgery event channel**,
  whose docstring states it is a *"No-op for the ours arms (rank growth
  is in-scan, zero-parameter)."* It is structurally `[]` for every ours
  arm, fixed or growable. The ours arms' in-scan fires live in a
  DIFFERENT key — `last_scan_growth_events` (from
  `block.mixer.growth_events`) and `channels.per_layer[].growth_events`
  — which the console print OMITS. The v1 result doc's §0.5
  (`growth events receipted`) referred to those keys.
* The full receipts carry the fires. n=320 (layer 0): fire at token
  boundary 256 (rank 128→256, smoothed 1.0) and 512 (256→512, smoothed
  1.0); layer 1 additionally at 768 (512→1024, smoothed 0.712,
  window rates [0.016, 0.944, 0.942, 0.946] — genuine collisions);
  final rank `[512, 1024]`. n=640: layer 0 → 1024 (fires at 256, 512,
  1024; the 1024 fire smoothed 0.54, genuine), layer 1 → 2048 (fires at
  256, 512, 768, 1024; smoothed 0.64/0.75 on the late fires); final
  rank `[1024, 2048]`.
* The wall-clock is the growth itself, corroborating: the growable
  pays a per-forward no-grad probe pass (the span-table discovery) and
  re-dispatches the window's spans at grown ranks on the K7 hi-walk
  kernel (K6 kernels exist only at rank ≤ 128), so per-token cost rises
  with the grown rank — 3.3x at n=320 (ranks to 512/1024), 7x at n=640
  (ranks to 1024/2048, windows 1,921 tokens).

**(b) is EXCLUDED — ours-growable differs from ours-fixed in NO way
except the registered growth machinery.** Code receipts:

* Same init and same parameters: `build_matrix_arm` seeds
  `torch.manual_seed(seed)` and builds the same `X46StoreModel` for
  both arms; `convert_model_to_growing` shares the very parameter
  objects ("the K6 conversion pattern verbatim ... the parameters the
  optimizer sees, and the init the seed produced"). `rank_max`/
  `rank_cap` are capacity knobs, never parameters.
* Same store size at birth: both arms start every window's scan at
  `start_rank = 128` (`MATRIX_RANK`; `GrowingMatrixMixer.mix` always
  passes `self.rank`, which growth never mutates — growth is re-decided
  per window, in-scan).
* Same projections, same head, same loss, same curriculum, same
  optimizer; the only deltas are `policy.grow=True` and
  `rank_cap=2048` (vs `grow=False`, `rank_cap=128`).
* Bit-equality at equal rank is the machinery's own contract: the
  grow-never-fires forward is *"bit-equal to the banked K6 path by
  construction"* (`k8_growth.py`, the span op pair).
* **The direct experimental receipt: the n=64 twin cells.** At n=64 the
  window is 193 tokens = 3 statistic boundaries — fewer than the
  trigger's 4-boundary smoothing window — so the trigger STRUCTURALLY
  cannot fire, and the two arms are the same machine: identical
  accuracy at every LR (0.8297 at 1e-3, 0.9683 at 3e-3, 0.0092 at
  3e-4 — equal to every digit), `final_rank [128, 128]`, zero scan
  events. The arms diverge ONLY through growth.

Therefore the 0.22→0.59 (n=320) and 0.06→0.59 (n=640) deltas belong
to the growth machinery — the in-scan fires, the zero-injection pads,
and the grown-rank spans they engage. One scope note, disclosed: the
receipts cannot decompose "more rows" from "different kernel at grown
rank" (capacity above 128 requires the hi path); both are the same
registered machinery, so the causal attribution to growth is unaffected.

**Growth-telemetry reliability (the v2 rows), stated plainly:**

* The fires, ranks, boundaries, and timing are RELIABLE (three
  independent keys agree: `last_scan_growth_events`,
  `channels.per_layer[].growth_events`, `final_rank`; the op counts and
  wall-clock corroborate).
* `mixer.growth_events` is reset every forward (it holds the LAST
  forward's span table), so `last_scan_growth_events` documents the
  final eval-window forward — representative, since the construction is
  identical every window and the fires are deterministic given the
  trained weights. The training-phase fires per step are not retained.
* The per-fire RATE numbers are NOT reliable for the early fires: the
  registered statistic's `last_write` bookkeeping is zero-initialized
  (`scan.py:263`, `k8_growth.py:1323` — the registered chunked
  implementation's own convention), so never-written slots read as
  "written at step 0" and EVERY write counts as a repeat while absolute
  step ≤ `grow_window_tokens`=512. The early fires' `window_rates`
  [1,1,1,1] are that artifact, not measured collisions; the selftest's
  machine-check (`check_growth_trigger_on_overflow`) used a −1 sentinel
  (fresh slots never count — its rate profile is the honest "0.0 → 1.0
  across the fill"). On THIS construction the fire times coincide
  under either init (the store genuinely fills at ~token 128; the
  first fire lands at boundary 256, the earliest the 4-boundary
  smoothing allows), so the FIRES are genuine; fires past step 512
  carry honest collision rates (0.54–0.75 smoothed).
* The printed `"growth_events": []` console line is unreliable as
  printed — a completeness bug in `run_arm`'s summary key list. The
  fix for any future console read is to add `last_scan_growth_events`
  to the printed keys; no data was lost.

## 3. THE COMPLETE FOUR-MACHINE TABLE

Best headline cell per machine per n (seed 3401; the LR grid per cell
in the receipts; chance 0.0078; ceilings per the signed registration's
derived table, band +0.05):

| machine | N=64 | N=320 | N=640 | N=320 ceiling (+band) | N=640 ceiling (+band) |
|---|---:|---:|---:|---:|---:|
| **ours-fixed** (rank 128) | **0.9683** (3e-3) | **0.2255** (3e-4) | **0.0600** (3e-3) | 0.4047 (0.4547) | 0.2063 (0.2563) |
| **ours-growable** (grown 512/1024–2048) | **0.9683** (3e-3) | **0.5879** (1e-3) | **0.5867** (1e-3) | none — "capacity ≥ n: no binding ceiling" | none |
| **gdn2-fixed** (H=2) | **0.9832** (1e-4) | **0.9308** (1e-3) | **0.8731** (1e-3) | 0.8016 (0.8516) | 0.4047 (0.4547) |
| **gdn2-growth** (grown H=8) | **0.9914** (1e-4) | **0.9735** (3e-4) | **0.9490** (3e-4) | none (grown) | none (grown) |

The ours v2 grid in full — n=320 ours-fixed: 0.2193 (3e-3) / 0.2220
(1e-3) / 0.2255 (3e-4); n=320 ours-growable: 0.5864 (3e-3) / 0.5879
(1e-3) / 0.5847 (3e-4); n=640 ours-fixed: 0.0600 (3e-3) / 0.0588
(1e-3); n=640 ours-growable: 0.5864 (3e-3) / 0.5867 (1e-3). Walls:
256–263 s (fixed, n=320), 809–825 s (growable, n=320), 584–597 s
(fixed, n=640), 4,074–4,131 s (growable, n=640). (One rounding note:
the briefing table's gdn2-growth n=320 figure 0.9736 is this same
cell — the receipt value is 0.97353, quoted here as 0.9735.)

Mechanistic strata (the receipts' own keys):

* **Lag quartiles** (short→long lag): ours-fixed n=320
  [0.341, 0.249, 0.180, 0.117]; n=640 [0.141, 0.059, 0.022, 0.012] —
  the recency gradient of eviction. ours-growable n=320
  [0.602, 0.584, 0.582, 0.584]; n=640 [0.588, 0.587, 0.588, 0.584] —
  FLAT: no recency penalty, no eviction signature. gdn2-fixed n=320
  [0.974, 0.960, 0.926, 0.864]; gdn2-growth n=640 [0.959, 0.951,
  0.937, 0.948] — mild or no gradient.
* **Layer-0 store ops per window** (APPEND/POP at the pair writes):
  ours-fixed n=320 ~223/~97 (30% of bindings destroyed at write),
  n=640 ~383/~256 (40%); ours-growable ~256/~64 at n=320 and ~511/~128
  at n=640 — **20% at both scales**. The grown machine's POP fraction
  is scale-free.
* **Layer 1 collapses to a circulating single row in EVERY trained
  arm** (n=640 growable: 0 APPENDs, 81,920 CORRECTs — every pair write
  overwrites one matched row): the deeper store contributes no binding
  information; the association work is layer 0's.
* **Read dilution**: the probe's read mass on its answer row at
  layer 0: 0.46 (fixed n=320), 0.43 (growable n=320), 0.30 (both,
  n=640) — the majority of read mass is crosstalk onto other rows.
* **The ladder itself receipts the learnability collapse** (final
  stage losses, 2,000-step stages): ours-fixed n=640 climbs 0.311
  (n=64) → 4.557 (n=576, ≈ chance ln 128 = 4.852); ours-growable
  saturates 0.311 → ~2.0–2.25 from stage n=320 on — the plateau is
  already scale-free INSIDE the ladder.

## 4. THE ADJUDICATION — each signed clause, quoted, with its verdict

Per the clause-conflict precedent: specific clauses govern; the 0.05
band governs phrasing (beats/ties) only. No pre-stated section of the
registration is edited; every verdict below is against the SIGNED
text.

### P-0 (the learnability precondition)

> "At N=64 (below every ceiling) every machine reaches ≥ 0.90 at its
> best LR (mean over its headline seeds). If a machine cannot, its
> rows are reported INCONCLUSIVE for that machine."

**VERDICT: HELD.** ours-fixed 0.9683, ours-growable 0.9683, gdn2-fixed
0.9832, gdn2-growth 0.9914 — all ≥ 0.90 at best LR; no machine's rows
are inconclusive. Caveats disclosed: single seed (3401 — the
registered 3402 confirmation was not run at n=64), and the ours arms
are sharply LR-sensitive at n=64 (1e-3: 0.830; 3e-4/1e-4: chance) —
the best-LR reading is what the clause registers.

### P-1 (the fixed machines pin — the primary)

> "At N=320: ours-fixed's best headline cell ≤ 0.4047 + 0.05;
> gdn2-fixed's best headline cell ≤ 0.8016 + 0.05. Refutation clause:
> a fixed machine beating its ceiling + band at any LR or seed
> REFUTES its ceiling claim and is reported as such — with the
> depth-2 superposition escape ... as the first candidate mechanism to
> receipt, a first-class finding either way."

**VERDICT: HELD for ours-fixed; REFUTED for gdn2-fixed — the
refutation clause FIRES, a first-class finding, exactly as the clause
pre-stated.**

* ours-fixed: 0.2255 ≤ 0.4547 at n=320 (all three LRs 0.219–0.226);
  0.0600 ≤ 0.2563 at n=640. **Pinned** — and pinned BELOW the bare
  ceiling (0.2255 < 0.4047; 0.0600 < 0.2063): the ceiling is
  valid-but-not-tight, as ruling 2 disclosed at sign-off (the v1
  calibration pin 0.2226 is confirmed at 0.2255 by the dwell-respecting
  run). The v1→v2 delta also receipts that the at-chance v1 rows were
  the cold-start saddle, not the pin: the 2,000-step ladder moves
  ours-fixed to 0.22 (n=320) / 0.06 (n=640) — off chance, still
  pinned, still far under the capacity bound.
* gdn2-fixed: 0.9308 > 0.8516 at n=320 and 0.8731 > 0.4547 at n=640 —
  **the derived Welch-bound ceiling (effective capacity H·d_k = 256)
  is REFUTED at both pressure points** (confirmed at seed 3402:
  0.8870). The trained dense machine answers ~93%/87% of probes where
  its linear-associative-memory bound allows 80%/40%. First candidate
  mechanism, per the clause's own naming: the depth-2 superposition
  escape — the key-conditioned nonlinear decoder (the inter-block
  nonlinearity and MLP between state and readout) resolving crosstalk
  the linear channel cannot order. The linear idealization was an
  artifact of the bound, not a property of the trained machine. This
  is the clause working as designed, not an inversion.

### P-2 (the growables escape — the amended two-reading form)

> "At N=320 both growable machines beat their fixed twins by MORE
> than the band AND reach ≥ 0.90, with growth events receipted.... A
> growable that ties its fixed twin, or whose trigger never fires, is
> the declared informative null for that machine, reported as such."
> (§8 ruling 7: the mechanism-based escape — beats the twin beyond
> the band + events + flat lag gradient — is PRIMARY; ≥ 0.90 is the
> non-gating SECONDARY.)

**VERDICT: ours-growable — PRIMARY SATISFIED, SECONDARY NOT MET.
gdn2-growth at n=320 — the declared informative null for the mechanism
claim; escape beyond the band at n=640.**

* ours-growable n=320: beats its twin by 0.362 (0.5879 vs 0.2255) —
  far beyond the band; events receipted (§2: the fires are in the full
  receipts; the console blob's `[]` is the print bug); lag gradient
  FLAT [0.602, 0.584, 0.582, 0.584] against the twin's
  [0.341, 0.249, 0.180, 0.117] — **the mechanism-based escape claim
  is SATISFIED on all three components.** The ≥ 0.90 solvability
  SECONDARY is NOT met (0.5879) — non-gating per ruling 7, reported
  as such: the growable escapes the PIN, not the task.
* gdn2-growth n=320: 0.9735 vs 0.9308 — +0.043, INSIDE the 0.05 band:
  a TIE with its fixed twin. Per the clause's own text this is the
  **informative null** for the dense machine's mechanism claim at
  n=320 (the twin already flies; growth adds nothing beyond the band),
  while the ≥ 0.90 secondary is met (0.9735). At n=640: 0.9490 vs
  0.8731 — +0.076 beyond the band, with events (heads 2→4→8, fire-PR
  31.3–41.0 vs committed τ_cal 16.99): escape beyond the band at the
  pressure cell.

### P-3 (the matched-grown-state prize — no directional prediction)

> "At N=640 with both growables at grown state 512 (ours rank 512,
> GDN2 H=4): a TIE ... REFUTES the slots-earn-their-keep claim ...; an
> ours win beyond the band ...; a GDN2 win beyond the band receipts
> the opposite. All three outcomes are pre-stated findings; none is
> preferred."

**VERDICT: the registered PREMISE did not occur; at the actual grown
states the outcome is the pre-stated GDN2 win, with the matched-state
comparison reported INCONCLUSIVE-AS-REGISTERED.** The triggers, run
verbatim, overshot 512: ours grew to [1024, 2048] (fires at
boundaries 256/512/768/1024), GDN2 snapped to H=8 (= 1,024
concatenated). No cell exists with both machines at 512. At the actual
states — ours holding at least the dense machine's capacity (1,024 and
2,048 rows vs 8×128) — **GDN2 wins beyond the band: 0.9490 vs 0.5867**
(+0.362). Honest scope: this is not the registered matched-state
read; the ours arm sits at a scale-free plateau far below solvability,
so the prize read is dominated by the ours machine's learnability
limit (§5), not by a clean state comparison. The pre-stated finding
stands as stated: at this scale, under forced addressing, dense
superposition beats addressable slots — and the slots machine's
deficit is not its state's size.

### P-4 (the growth mechanics)

> "ours-growable's registered trigger fires in-window on the
> constructed stream ..., with the grown rank ≥ n at N=320 by the
> escape; gdn2-growth's calibrated trigger fires with the per-event PR
> receipts, and its τ_cal is committed BEFORE its arms run."

**VERDICT: HELD, both halves.** ours: fires in-window at boundaries
256/512(/768) with grown ranks [512, 1024] ≥ n=320; [1024, 2048] at
n=640 — with §2's telemetry note (fires, ranks, timing reliable; the
early fires' RATE numbers carry the zero-init statistic artifact).
gdn2: τ_cal = 16.99 committed in `receipts/x46_gdn2_calibration.json`
before the arms; 4 events per growth cell, per-event PR 31.3–41.0
receipted, all above τ_cal.

### P-5 (the mechanistic stratification)

> "On the ours arms at N=320 the live-at-probe fraction is ≤ rank/n +
> ε (the slot-count bound made countable), the live-stratum accuracy
> dominates the dead-stratum accuracy, and the lag quartiles show the
> recency gradient the eviction policy predicts."

**VERDICT: the countable live-at-probe form is REFUTED by the
receipts; the stratification instrument degenerates; the recency
gradient HELD and its vanishing is the escape's direct receipt.**

* Live-at-probe ≤ rank/n + ε: measured 0.995–1.0 on BOTH ours arms at
  n=320 (0.4 + ε bound) and 1.0 at n=640. The pre-stated form modeled
  evicted slots as staying dead; the trained machines CHURN slots —
  evicted rows are re-appended into within the same window (the op
  counts receipt the churn: ~97 POPs then fresh APPENDs into freed
  rows), so slot-LIVENESS saturates while BINDING survival is what
  pins. The slot-count bound itself is untouched — it bounds live
  bindings, not live slots — and is carried by the op counts, not by
  this stratification.
* Live-stratum dominance: DEGENERATE — the dead-at-probe stratum is
  near-empty under slot reuse (e.g. fixed n=320 layer 0: dead acc
  0.011 on ~0.5% of probes), so the instrument does not separate. An
  instrument failure, reported as such; not a mechanism failure.
* Recency gradient: HELD for ours-fixed ([0.341 → 0.117] at n=320;
  [0.141 → 0.012] at n=640 — later-lagged probes answered worse, the
  eviction signature) and VANISHES for the growable (flat at both n) —
  the escape's mechanism receipted directly, as P-2's flat-gradient
  component requires.

## 5. WHAT THE SEPARATOR NOW SEPARATES (the honest mechanism statements)

**(1) The dense machine's advantage on this task is NOT capacity —
and the slot machine's failure at scale is LEARNABILITY.** Two
independent receipts force this reading:

* The ours-growable cells sit in the "no binding ceiling" regime by
  their own receipt (capacity 512 ≥ 320; 1024 ≥ 640 — the ceiling
  rows say *"capacity >= n: no binding ceiling (the control
  regime)"*), yet plateau at 0.586: a machine whose state holds
  several times the required bindings and which demonstrably learns
  the same construction to 0.97 at n=64, but cannot learn to USE
  unbound capacity at n ≥ 320. Capacity is not the binding
  constraint; the learnable write/read policy is.
* The ours-fixed machine pins BELOW its capacity ceiling (0.2255 <
  0.4047 at n=320; 0.0600 < 0.2063 at n=640): the slot-count bound is
  a valid upper bound the machine does not even reach. The curriculum
  removed the cold-start saddle (v1 at-chance → v2 0.22/0.06) but not
  the scale limit — the binding constraints are slot churn (30–40% of
  bindings destroyed at write), read dilution (majority of probe mass
  on non-answer rows), and the recency gradient, i.e., the policy the
  machine can LEARN, not the state it HAS.
* Meanwhile the dense machine EXCEEDS its own derived capacity
  ceiling at both scales (0.9308 > 0.8016; 0.8731 > 0.4047 — the
  P-1 refutation): its advantage is precisely that its write/read
  policy is learnable past the linear bound at this scale. Its
  effective key space (256 fixed, 1,024 grown) is not why it wins; the
  ours machines hold comparable or larger state (128–2,048 rows) and
  lose by 0.29–0.81.

**The separator's final word: on the constructed exact-recall family,
the machines separate on the LEARNABILITY of the write/read policy at
scale — the dense machine's nonlinear readout learns past its
linear-channel bound, the slot machine cannot learn to exploit even
unbound state — and NOT on state capacity.**

**(2) The ours-growable > ours-fixed delta (0.2255→0.5879 at n=320;
0.0588→0.5867 at n=640) — what it does and does not show.**

* DOES show: growth is CAUSAL for the escape from the pin regime. The
  two arms are the same machine at the same init, parameters, birth
  rank, projections, curriculum and optimizer (§2's code receipts +
  the n=64 twin cells, bit-identical when the trigger cannot fire);
  the only difference is the registered growth machinery; the escape
  signature is exactly the registered mechanism prediction (the recency
  gradient vanishes; the ceiling row moves from the 0.4047 bound to the
  no-binding-ceiling regime; the delta is 7–10x the band).
* DOES NOT show: solvability (0.5879 < the 0.90 bar — P-2's secondary
  unmet, non-gating); that the grown machine approaches what its
  capacity regime allows (it plateaus at 0.586, 75x chance but 0.34
  under the fixed dense machine); that growth is SUFFICIENT (it is
  necessary-and-partial on this family); or that the dense machine's
  win is a capacity win (the dense machine refutes its own capacity
  ceiling — see (1)).

**(3) The scale-free plateau at ~0.586 — the signature, stated
honestly.** One value to the third decimal — 0.5847/0.5864/0.5879
(n=320, 10x LR spread), 0.5864/0.5867 (n=640, a 2x scale jump) — flat
across lag quartiles, flat across the LR grid, flat on the training
curve from step 250 on, and already flat inside the curriculum ladder
(stage losses saturate ~2.0–2.25 from n=320 up while the fixed
machine's stage losses run to chance). What the receipts show of the
mechanism: the per-window failure structure does not scale with n —
(i) the store restarts at birth rank 128 EVERY window (growth is
re-decided in-scan per window; `mixer.rank` never persists), so every
window pays the same fill-and-fire transient, and the layer-0 POP
fraction is 20% at BOTH n (8,192/40,960 and 16,384/81,920 —
scale-free, set by the fire cadence vs the write stream, not by n);
(ii) read dilution persists at scale (answer-row mass 0.43 at n=320,
0.30 at n=640); (iii) layer 1 learns a degenerate circulating single
row at every scale, contributing nothing. What the receipts do NOT do
is decompose the 0.586 — which of these sets the number, and why it is
exactly n-independent, is UNRESOLVED by this table. What the
scale-freeness rules OUT: capacity (no binding ceiling at the grown
states), lag/eviction (flat quartiles), budget and LR (flat curve,
flat grid), chance (75x). What it points to: a per-window structural
limit of the machinery's learnable policy — the same learnability
separator as the fixed machine, one tier up. This is stated as an
open mechanism question with its receipts, not as a settled
explanation.

## 6. THE X46 PROGRAM'S CLOSING STATE

The X41 theorem's empirical question — does a fixed-state machine pin
at its own derived ceiling on a task that provably forces aliasing,
and does growth escape it? — has its answer, and it is not the
theorem's clean shape:

1. **The fixed slot machine pins** (P-1 held) — but BELOW its derived
   capacity ceiling: the pin is a learnability pin before it is a
   capacity pin. The derived bound is valid-not-tight (ruling 2's
   disclosed outcome, confirmed).
2. **The fixed dense machine REFUTES its derived ceiling** (P-1's
   refutation clause fired): the linear-associative idealization does
   not bound the trained machine; the depth-2 nonlinear decoder
   escape is real and load-bearing on this family. Derived
   linear-channel ceilings, as a class, upper-bound the READOUT
   family, not the machine.
3. **Growth is causal and necessary on the slot machine** (P-2 primary
   satisfied: +0.36 beyond the band, events receipted, flat lags) —
   **and not sufficient**: the growable plateaus at a scale-free
   0.586, far under the 0.90 bar and the dense machine. On the dense
   machine growth is an informative null at n=320 (tie inside the
   band) and a marginal escape at n=640 (+0.076).
4. **The matched-grown-state prize cell did not occur as registered**
   (the triggers overshoot 512); at the actual grown states the
   pre-stated GDN2-win finding holds (+0.362), with the honest caveat
   that the slots machine's limit at this scale is learnability, not
   the state comparison.
5. **The separator's final word**: on the constructed exact-recall
   family, the machines separate on the LEARNABILITY of the
   write/read policy at scale — the dense machine learns past its
   linear bound (0.93/0.87 fixed, 0.97/0.95 grown), the slot machine
   cannot learn to exploit even unbound state (0.59 grown, 0.22/0.06
   fixed), and capacity — the entire original axis of the X41
   theorem's state algebra — separates NOTHING on this table: both
   derived capacity ceilings are either not-reached (ours) or
   refuted (GDN2).

The X46 program is CLOSED at probe scale. Ruling 4's disclosed
alternative (the 150M-machine cell) and the one open mechanism
question (what sets 0.586) are follow-ups under their own sign-off,
not open items of this registration.

## 7. Provenance

This document: `RESULT_X46_OURS_RERUN2.md`, written after the ten v2
arms landed; every number above is from
`experiments/expressivity_tasks/x45_matrix_emender/receipts/x46_arm_*.json`
(the v2 ours cells, committed by the arms' self-committing driver) or
the banked corrected-grid rows (commit `ed8e2304`), read against the
console log `/home/erikg/emender/x46_ours_rerun2.log`. Adjudicated
against `PREREGISTRATION_X46_GROWTH_SEPARATOR.md` (OPERATOR-SIGNED
2026-11-19, §8 rulings amended 2026-11-30) — no pre-stated section
edited; refutations reported as refutations. The void chain and the
supersession are receipted in `X46_SWEEP_VOID_RECEIPT.md`; the
superseded 300/400-step ours rows are in git history and
`x46_ours_rerun.log`. No GPU work was launched for this document; the
growth-machinery citations are from `x46_growth_separator.py`,
`k8_growth.py`, `scan.py`, and `run_x46_leased.sh` as committed.
