# PREREGISTRATION -- K24: THE LONG-WINDOW PROBE

**The horizon hypothesis's first test, at probe scale: does memory become
load-bearing when the regime demands state?**

Committed BEFORE any arm (the standing lesson).  The engines are committed
and read-only: `k15_composition` / `velocity_ladder._build_k10` (the ours
machine), `matched_arms.build_gdn2` (GDN2), `k22_hybrid` (the hybrid
engine), `bakeoff_optimizers.HybridMuonX45`, `muon_control` /
`k11b_probe_channel_gates` (the routing splits),
`protocol_correction` (the corrected-scoring conventions),
`x42_pile_scale/lattice` (the stream) -- all imported VERBATIM, never
modified.  New files only: `k24_long_window.py` (the probe),
`k24_gate_receipt.py` (the long-window engine gates), the lane's tests,
the leased driver.

## 1. THE QUESTION

The uniformity finding (4-5 load-bearing layers of 24 on ours; the slot
reads decorative even inside the K22 hybrid: 0 of 3 slot layers over the
0.02 bar, all-slots dNLL -0.00009) was measured at the banked probe
window length.  If the state-demand lever -- WINDOW LENGTH -- is what the
slots are FOR, the load-bearing mask should light up as windows grow.
This lane is that first test, at probe scale: one window-length bracket
over the machines, matched exposure, the corrected scoring, the
mechanism instruments.

## 2. THE ARMS (SEVEN, seed 3401, matched 20,000,000-target stopping,
the corrected cooldown-primary scoring)

| arm | machine | window | role |
|-----|---------|--------|------|
| `ours_1x` | the committed K15 composition machine | 2048 B | **THE ANCHOR** |
| `gdn2_1x` | the committed GDN2 arm | 2048 B | **THE ANCHOR** |
| `ours_2x` | the committed composed machine | 4096 B | long |
| `gdn2_2x` | the committed GDN2 arm | 4096 B | long |
| `ours_4x` | the committed composed machine | 8192 B | long |
| `gdn2_4x` | the committed GDN2 arm | 8192 B | long |
| `hybrid_4x` | the K22 hybrid engine (GDN2 + the composed slot mixers at {6, 12, 18} through the zero-born read gates) | 8192 B | long |

The hybrid's 1x cell is NOT re-run: the K22 lane's own landed row
(`k22_runs/hybrid_seed3401`, post-cooldown 1.3547673906598772, gates to
|0.13|, 0 of 3 slot layers over the bar) is the curve's hybrid 1x point,
cited from its committed receipt.

**THE ANCHOR CONVENTION (non-negotiable).**  The 1x cells MUST reproduce
the banked rows bit for bit FIRST, before any long window runs:

* `ours_1x`: step-1 `5.782775391046648`, EVERY training-log row, the
  constant-180 `1.433674931526184`, the EMA-180 `1.424389226096017`, the
  exposure landing (step 180 / 20,022,866 / total 22,745,198) and the
  post-cooldown landing `1.3811104467936925` -- the K15 composition run's
  own committed receipts (`k15_runs/composition_seed3401`).
* `gdn2_1x`: step-1 `5.948033220866492`, every row, the constant-180
  `1.3950541530336653`, the EMA-180 `1.3803383282252721`, the same
  landing, the post-cooldown `1.3521734476089478` -- the
  protocol-correction rescore's own committed receipts
  (`protocol_correction_runs/gdn2_speedrun_rescored`).

ANY mismatch (a single diverging step's loss, a single digit of the
landing) STOPS the lane and escalates.  The anchors also pin the lane's
one new piece of plumbing -- the K24 training loop, a character-faithful
copy of `protocol_correction._train_corrected` whose ONLY differences are
the threaded (context, windows_per_step) pair: at 1x the threaded values
are the banked protocol's own, so an anchor that lands bit-for-bit is the
loop's own plumbing receipt.

## 3. THE WINDOW-LENGTH LEVER (the standing discipline, never violated)

The longer windows are the corpus pipeline's own boundary-respecting
windowing: the `lattice.Stream` constructor's own `context` parameter
(2048 / 4096 / 8192).  Every window is a slice of exactly ONE document;
windows TRUNCATE at the boundary; the document-disjoint
train/calibration/final_test splits stay intact; every document slice is
SHA-256-verified at consumption; the stream is hash-bound to the flagship
manifest registration.  NEVER pack across document boundaries; no model
state carries across windows or documents.  If a machine could not honor
the boundaries at longer windows, the lane would STOP, never force it.

## 4. THE STEP/WINDOW ARITHMETIC (matched exposure is the invariant)

The banked 1x protocol: 64 windows/step x 2048 B, microbatch 8 (16,384
tokens per microbatch), landing at step 180 / 20,022,866 targets.  The
longer arms hold (a) the ~111K-targets-per-step shape and (b) the
~180-step landing by scaling windows/step DOWN as the context grows, and
hold the microbatch's TOKEN count at the banked 16,384 -- the
memory-matched rule (the banked ours arm's peak allocation is 27.9 GiB
at 16,384 tokens/microbatch; the cards are 48 GiB):

| context | windows/step | microbatch | tokens/microbatch | constant landing | exposure (targets) |
|---------|--------------|------------|-------------------|------------------|--------------------|
| 2048 (1x) | 64 | 8  | 16,384 | step 180 | 20,022,866 |
| 4096 (2x) | 38 | 4  | 16,384 | step 182 | 20,067,118 |
| 8192 (4x) | 27 | 2  | 16,384 | step 178 | 20,088,160 |

These numbers are COMPUTED from the manifest at registration time and
committed as `receipts/k24_window_arithmetic.json` (never hardcoded);
each arm's `start.json` re-derives its own row from the manifest and
fails closed on a drift.  The cooldown is the corrected scoring's own 24
linear steps for every arm (the landing-step snapshot is added to the
checkpoint grid so the EMA and the constant-phase reads sit at the
landing for every arm).

**THE EFFECTIVE WINDOW-LENGTH DISTRIBUTION, receipted honestly per arm**
(the lever is only as strong as what was actually achieved -- the corpus's
median document is ~1,850 bytes, SHORTER than one banked window):

| mult | mean (B) | p50 | p90 | full-length fraction | windows consumed | mean / 1x mean |
|------|----------|-----|-----|----------------------|------------------|----------------|
| 1x | 1738.1 | 2048 | 2048 | 0.718 | 11,520 | 1.000 |
| 2x | 2901.5 | 4096 | 4096 | 0.530 | 6,916 | 1.669 |
| 4x | 4179.8 | 3042 | 8192 | 0.323 | 4,806 | 2.405 |

The 4x arm's ACHIEVED mean window is 2.405x the banked mean (not 4x).
Every arm's receipt states its own achieved distribution; the curves are
read against what was achieved.

## 5. THE EVAL RULE

The banked calibration eval is `L.evaluate`'s own: the first 24
calibration windows in split order, keeping the full ones (7 windows /
14,336 targets at 1x).  The K24 rule at the longer windows is the same
rule's intent at the arm's own window: THE FIRST 7 FULL WINDOWS of the
arm's own context in calibration order (28,672 targets at 2x; 57,344 at
4x).  At 1x the two rules pick the IDENTICAL window set (receipted from
the manifest; asserted by the lane's tests), and the 1x cells evaluate
through `L.evaluate` VERBATIM.  Every arm is additionally CROSS-EVALUATED
at all three window lengths on its post-cooldown final state (the
conditioning-vs-regime decomposition: the fixed-1x-eval column separates
"trained longer" from "conditioned longer").

The primary score is the post-cooldown eval at the arm's own window;
secondary the EMA at the constant landing (ranking-only); the
constant-phase curve is retained on the banked grid (1, 5, 10, 20, 40,
80, 120, 160, 180 + the landing).

## 6. THE INSTRUMENTS (the decisive duties, per arm)

* **THE DECORATIVE SWEEP -- THE HEADLINE** (ours, every mult; the hybrid,
  at 4x): each layer's store read masked at eval time, one layer at a
  time, plus the all-layers cell -- the `side_by_side.k10_read_mask_sweep`
  body with the own-window eval.  The load-bearing count (dNLL >= 0.02)
  vs window length IS the result.  On the hybrid: each slot layer's read
  gate zeroed, per slot layer + the all-slots cell (the K22 body).
* **The C5 instrument** (ours, every mult; the hybrid's slot layers, at
  4x): the composed walk's effective rows, occupancy, the op mix
  (CORRECT/APPEND/MERGE/POP pressure vs window length), the identity
  share, the p3 gate telemetry -- the `k15_probe_diagnostics` hook bodies
  at the arm's own token count.
* **GDN2's own state saturation telemetry** (every mult): the retention
  exp(g), the implied memory horizon 1/(1-retention), the erase/write
  gates per layer -- the `side_by_side.gdn2_gate_telemetry` body at the
  arm's own windows (the committed arm's exposed state surface; the
  recurrent state itself is internal to the FLA chunk kernel).
* **The hybrid's gate trajectories at 4x**: the read gates' values at the
  constant landing and post-cooldown states, vs the banked 1x hybrid's
  own final gates (|gate| ~ 0.13) -- do the gates open FURTHER when state
  is demanded?
* **The at-birth receipts for `hybrid_4x`** (the K22 discipline, at the
  long window): born as GDN2 bit for bit on the arm's own first 4x
  microbatch (vs a pure-GDN2 twin of the same build), the open signal
  (dLoss/dgate nonzero at the zero-born gate), the inert birth (every
  other slot gradient exactly zero), the store fills (APPEND/CORRECT
  fire while nothing reads out).
* **The engine gates** (`k24_gate_receipt.py`, run BEFORE any arm): the
  composed scan at the 4x window shape (batch 2 x 8192) bit-equal to the
  eager oracle on every schedule channel with the reads at the measured
  floor; the fp64 gradchecks with LONGER sequences (the composed path at
  a 5-segment fixture; the hybrid seam at a 2-segment fixture); the
  kernel gradients vs the measured fp32-vs-fp64 floors AT THE 4X SCAN
  SHAPE; the segment-recompute bit-equality at 128 segments; the honest
  walls at all three window shapes.  The committed suites stay green
  with the K24 module loaded.

## 7. PRE-STATE (directional, the hypothesis under test, with the
refutation clause)

* **L-1 (the horizon hypothesis)**: the slot layers' load-bearing count
  RISES with window length -- the decorative sweep lights up (more layers
  clear the 0.02 dNLL bar at 4x than at 1x; the banked 1x regime's own
  count is 4 of 24 on ours -- layers 0-3, the top delta 0.081 -- and 0
  of 3 slot layers on the hybrid), and/or the occupancy/effective rows
  rise on the slot machines.
* **L-2**: the quality gap (ours - gdn2) SHRINKS with window length (the
  dense state saturates as sequences grow; the addressable store does
  not).  Read on the own-window post-cooldown rows at each mult, with
  the fixed-1x-eval cross-grid as the decomposition.
* **REFUTATION, receipted plainly**: if the mask stays dark (the
  load-bearing count does not rise at 4x) and the gap stays flat at 4x
  (the 4x-vs-1x gap change inside the 0.0119 winner-seed band), the
  horizon hypothesis is dead at probe scale -- the board run's horizon
  rationale goes with it, and the remaining board value is
  scale/exposure alone.
* **No band-worship**: every delta is stated against the 0.0119
  winner-seed band (and the 0.075 endpoint-variance band cross-machine);
  a delta inside the seed band is PARITY, phrased as parity.

## 8. ESCALATION TRIGGERS (the operator hears it from the receipts)

1. **Any 1x-anchor mismatch**: STOP the lane (the runner raises; nothing
   further runs), escalate.
2. **Any engine that cannot honor document boundaries at longer
   windows**: do NOT force it; STOP, escalate.
3. **A 4x result that INVERTS an existing conclusion** (e.g. ours beats
   GDN2 at 4x on the own-window rows, or the hybrid's slot layers become
   load-bearing): flag it in the assemble, escalate with the receipts.

## 9. RUN ORDER (commit per arm; detached chains, the PPID=1 playbook)

gates -> suites -> smoke (all three machines at 4x, tiny exposure) ->
`ours_1x` -> `gdn2_1x` (both anchors MUST land bit-for-bit before any
long window runs; the driver fails closed) -> `ours_2x` -> `gdn2_2x` ->
`ours_4x` -> `gdn2_4x` -> `hybrid_4x` -> assemble ->
`RESULT_LONG_WINDOW.md` (the gap-vs-window curve, the
load-bearing-count-vs-window curve, the effective-window telemetry, the
clause adjudications, and the verdict on the horizon hypothesis at probe
scale).  Cards 2/3 (`GPU_LEASE_VISIBLE=2,3`, one leased card at a time,
`--wait`), `OMP_NUM_THREADS=4`, the approved
`/home/erikg/emender/.venv/bin/python`.

The honest walls are stated first-class per arm (matched-EXPOSURE: the
arms eat the same bytes in different wall-clocks; the composed scan's
sequential walk makes the longer-window arms materially slower per step
at the same per-step bytes -- the wall asymmetry is receipted by the
engine gates, not projected).
