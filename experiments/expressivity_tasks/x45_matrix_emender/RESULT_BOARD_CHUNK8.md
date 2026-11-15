# RESULT — the 2700 s chunk-8 board arm: the sequentiality dial measured at board scale

The extension lane's own verdict ([`RESULT_CHUNK_SEQUENTIALITY.md`](RESULT_CHUNK_SEQUENTIALITY.md),
next-steps item 2) named this arm and budgeted it honestly: **chunk 8 is the right
matched-time board point** (at 2700 s it sees ~56 % of the chunk-128 arm's bytes;
chunk 4 would see only ~35 %). This is that arm — the matrix_ops board arm with
**only `chunk_tokens` changed from 128 to 8**, everything else identical to the banked
board arm by construction: seed **3401**, the transferred gdn2-nomination optimizer
(lr 5.42e-4, β₂ 0.939, wd 0.201, warmup 69 steps, microbatch 8), the **K5B
whole-scan fast path**, the registered flagship stream (hash-bound in the receipt),
the **honest 2700 active-second clock** (starts at step 1; includes the stream read,
any compile/capture, fwd/bwd, clip, AdamW; no untimed warmup steps), 64 windows ×
2048 bytes per step, and the 150 M board's own calibration eval boundary
(`lattice.evaluate`, 7 full windows / **14,336 targets**, batch 1, recorded-read
path — every row below scored that same boundary).

**The question it answers: how much of the 0.5600-nat board gap to GDN2 does the
chunk-size sequentiality dial close AT BOARD SCALE?**

**Answer: −0.0240 nats/byte — 1.7576 → 1.7336 — about 4.3 % of the 0.5600-nat gap,
and the probe-scale promise does not transfer.** At matched *exposure* the probe
sweep measured the 128→8 dial step at −0.1987 nats (2.4173 → 2.2186, 20 MB arms);
at board scale, matched *time*, it delivers an order of magnitude less. The
remaining ~0.5360-nat gap to GDN2 (1.7336 vs 1.1976) is not a chunk-staleness
number: it belongs to the two lanes already running — the **unsearched LR** (the
CMA-ES lane) and the **per-token character the dial cannot reach** (the K6 lane,
whose per-token sequential kernel is exactly the form the chunk-2 memory wall said
the dial can never become).

## The board table (three rows, same protocol family, same seed)

| arm | chunk | params | calibration nats/byte | steps | bytes seen | tok/s (bytes/s) | active s | exposure vs c128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gdn2` (same-session) | — | 149,855,670 | **1.1976** | 831 | 91,751,599 | 33,958 | 2701.9 | — |
| `matrix_ops` (banked board arm) | 128 | 149,981,808 | **1.7576** | 832 | 91,867,164 | 33,991 | 2702.7 | 100 % |
| `matrix_ops` (this arm) | **8** | 149,981,808 | **1.7336** | 475 | 52,599,177 | **19,449** | 2704.4 | **57.3 %** |

All three rows: seed 3401, the honest 2700 s clock, the same flagship stream, the
same optimizer family (the matrix arms inherit the gdn2 nomination's settings —
the standing deconfound), the same eval boundary (7 windows / 14,336 targets).
The gdn2 and chunk-128 rows are the banked same-session pair
(`x43_fork_resolved/runs_150m_matched/`, commits b9d6402f / b52d4472); the chunk-8
row is this arm's receipt, `receipts/board_chunk8.json`
(config sha256 `5cfdece8…c3a7a25`; the banked arms' config hashes:
gdn2 `6f27da3c…4d9cc`, matrix_ops chunk-128 `afdd629e…43323c`).

## The arm's receipt (what actually ran, at what cost)

* **Stopping-rule ledger**: 2700 active seconds requested, 2704.4 honest active
  seconds at stop (the clock includes the step that crosses the boundary), 475
  steps, 52,599,177 bytes (19,449 bytes/s), last step's loss 1.9282. The parity
  phase (7.8 s) is a separate receipted phase outside the clock, identical
  discipline to every banked arm.
* **Per-chunk-size bit-equality + 0-decision-flip receipt, BEFORE timing**
  (`receipts/board_chunk8_parity.json`, sha256 `703d065e…55f9`): the whole-scan op
  pair (eager body, capture+replay, replay) bit-equal to the per-chunk K5A+K4 arm
  on the recorded schedule fields, weights, reads, state, liveness and the three
  gradients at chunk 8; and the decision flip test re-run at chunk 8 — the K5A
  router the walk runs vs the registered eager `route_chunk` on the same pinned
  chunk-entry states — **0 flips in 8,650,752 lanes**, the banked A-phase bar.
* **Dispatch ledger** (the K5B discipline): 25 captures (24 mixers + the parity
  fixture) per direction, 91,178 forward and 91,178 backward graph replays, 25
  eager first-calls per direction, **0 fallbacks** — every training scan a graph
  replay, no registered path touched.
* **Peak allocated 28.32 GB** — the memory law's chunk-8 point (peak GB =
  21.54 + 0.0265 × chunks/window; 256 chunks/window), measured not projected.
* Walls: parity 7.8 s / train 2704.4 s / diagnostics probe 42.1 s / eval 1.5 s.
  Full trajectory, config and bindings in the receipt and in
  `board_runs/matrix_ops_c8/{start.json,training.jsonl}`.

## The honest reading

**1. The dial effect at board scale is real but small — and it is a LOWER bound.**
The eval delta (−0.0240) is near the single-eval resolution, so it is read with
its corroborations, which all agree in sign: the chunk-8 arm's training loss sits
below the chunk-128 arm's at **every matched-time cross-section from 600 s on**
despite seeing only 57 % of its bytes (smoothed over the final 300 s: 2.034 vs
2.062; last logged step 1.928 vs 2.180), under identical init, data order and LR
schedule. Matched-time cross-sections (step nearest each time point):

| active s | gdn2 step / loss | matrix_ops c128 step / loss | matrix_ops c8 step / loss |
|---:|---|---|---|
| 300 | 93 / 2.144 | 89 / 2.735 | 44 / 2.974 |
| 600 | 185 / 1.996 | 180 / 2.807 | 98 / 2.630 |
| 900 | 276 / 1.977 | 273 / 2.505 | 152 / 2.672 |
| 1200 | 367 / 1.561 | 365 / 3.219 | 205 / 2.587 |
| 1800 | 552 / 1.423 | 551 / 2.278 | 313 / 2.080 |
| 2400 | 737 / 1.508 | 737 / 2.166 | 420 / 2.080 |
| 2700 | 830 / 1.381 | 831 / 2.180 | 474 / 1.970 |

**2. The matched-time vs matched-exposure distinction, stated plainly: this arm is
matched-TIME, and the exposure shortfall biases AGAINST it.** Chunk 8 runs the K5B
graph at 19.4k tok/s against chunk-128's 34.0k (the whole-scan checkpoint memory and the 16× denser
chunk-entry decision cadence), so the same 2700 s bought only 52.6 MB —
**57.3 %** of the chunk-128 arm's 91.9 MB. Two handicaps follow: less optimisation
progress at the boundary, and a longer warmup fraction (69 warmup steps are 14.5 %
of this arm's 475 steps vs 8.3 % of the chunk-128 arm's 832). Any dial gain shown
here is therefore a **lower bound** on the dial's matched-exposure effect at board
scale. But the bound cuts the other way too: even granting the most generous
extrapolation of the 43 % exposure shortfall, a −0.02-nat matched-time reading
cannot hide a −0.20-nat matched-exposure effect of the probe's size. **The honest
conclusion is that the probe-scale sequentiality gradient (−0.199 at 20 MB) does
NOT transfer to 150 M board scale: at board scale the chunk dial alone closes a
small fraction — ~4 % as measured, and well short of a majority even
exposure-corrected — of the 0.5600-nat gap.** The clean deconfound, named and not
run: a matched-EXPOSURE chunk-8 board twin (~91.9 MB at 19.4k tok/s ≈ 1.3 h), or
the chunk-4 twin the probe's still-falling curve (8→4 buys −0.0395) would licence.

**3. What remains is exactly the two lanes already running.** The residual
~0.5360-nat gap (1.7336 vs 1.1976) decomposes onto the board's own stated
confounds, none of which the dial touches:
* **The unsearched LR** — both matrix arms inherit the gdn2 nomination's
  hyperparameters verbatim; the gdn2 baseline's were CMA-ES-searched for itself.
  The **CMA-ES lane (running)** is that search; until it lands, the board's
  architecture-vs-optimiser split is unresolved, and the 0.56-nat headline stays a
  matched-*protocol* comparison, not a matched-*quality* one.
* **The per-token character** — the dial's floor is chunk 4 by memory (the
  checkpoint-scaling law; chunk 2 OOMs the card), so chunk-constant reads against
  stale entry state survive even this arm. The **K6 lane (running)** — the
  per-token sequential kernel with segment checkpoints (32 MB vs the chunk-2
  dial's ~1 GB) — is the form that can actually reach per-token decisions; the
  dial's board-scale result here says the value of K6 must be argued from its own
  gates, not inherited from the dial's probe-scale gradient.

## Op-rate diagnostics of the trained board arm (the receipt's probe)

First board-scale diagnostics of a chunk-8 model (the banked chunk-128 board arm
ran no diagnostics; its op-rate columns come from the 20 MB probe — cross-scale,
stated): lane-0 firings per token **append 0.0646 / correct 0.8965 / pop 0.0207 /
merge 0.0064 / none 0.0118**; occupancy **76.7 live slots of 128** at chunk entries
(59.9 %, final 100.8); read-weight entropy **1.836 nats**. Against the probe-scale
chunk-8 arm (20 MB: merge 0.0092, occupancy 51.2, entropy 1.546) the board-scale
store is fuller and reads are more diffuse — more training populates more live
rows — and against chunk 128 at any scale the MERGE collapse the probe found
(0.2484 → 0.0064, 39×) persists at board scale: fresh chunk-entry state still
starves the store's garbage collector. Capacity never binds (pop + none ≈ 3 %).

## Honesty ledger

Single seed (3401); one transferred LR (the CMA-ES confirmation is running, not
banked); matched-TIME arm with the exposure shortfall stated (57.3 % of the
chunk-128 arm's bytes — biases against this arm); the eval delta is near the
single-eval resolution and is read with its trajectory corroborations, not alone;
the chunk-128 op-rate comparisons are cross-scale (board vs 20 MB probe); the
parity phase's kernel warm-up sits outside the honest clock, identical discipline
to every banked arm; no registered semantics, PREREGISTRATION file or banked
receipt was modified — new receipts only (`board_chunk8.json`,
`board_chunk8_parity.json`).

## Commands (the arm's provenance)

```bash
# detached leased driver (cards 4/5; the K6 lane's 0/1 and the CMA-ES lane's
# 2/3 are never touched), PPID chain to init, lease owner + heartbeat verified
setsid bash experiments/expressivity_tasks/x45_matrix_emender/run_board_chunk8_leased.sh \
    > experiments/expressivity_tasks/x45_matrix_emender/board_runs/driver.console.txt \
    2>&1 < /dev/null &
# the arm itself: parity gate first, then the honest-clock training loop, the
# diagnostics probe and the calibration eval, all in one leased process
/home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.board_chunk8 arm
```

The driver (`run_board_chunk8_leased.sh`) acquires one lease from cards 4/5, runs
the arm, commits its own artifacts before releasing, and survives the attended
session; `board_chunk8.py` composes the banked harness pieces verbatim
(`matched_arms.build_matrix` + the `matched_arms.train` loop, the sweep's
`parity_receipt` and `probe_diagnostics`), changing only the stopping rule's
seconds discipline (the board's) and the chunk length (8).
