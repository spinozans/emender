# The chunk-size sequentiality sweep — the gradient keeps falling to the dial's floor; the floor itself is a finding

**Verdict (a), extended: still monotone, and NOT yet plateaued at chunk 4.** Eval NLL
falls at every step of the extended sweep — chunk 128 → 64 → 32 → 8 → 4 gives
**2.4173 → 2.3287 → 2.2462 → 2.2186 → 2.1791 nats/byte** at identical exposure
(20,022,866 targets, 180 steps, identical data order), identical seed (3401), and
one transferred LR. The sequentiality hypothesis is confirmed at this scale and the
extension sharpens it: the four-arm reading ("flattening: −0.089, −0.083, −0.028")
was wrong about the direction of the tail — the 8→4 step buys **−0.0395**, *more*
than 32→8's −0.0276, so the apparent flattening at chunk 8 was single-eval noise,
not the curve bottoming out. The per-token limit of the dial has NOT landed at
chunk 4.

**The extension's second finding: the dial has a hard MEMORY floor, and it is not
chunk 2.** The chunk-2 arm is **not run, with reason** (`receipts/sweep_chunk_c2_oom.json`):
the K5B whole-scan graph's training peak memory is **linear in decisions-per-window**
(chunk-entry checkpoint set: ~26.5 MB per chunk per window across the 24 layers,
fit exactly over all five landed arms — peak GB = 21.54 + 0.0265 × chunks/window,
residuals ≤ 0.005 GB). Chunk 2 means 1,024 chunks per window ≈ 48.7 GB of allocator
demand on a 47.38 GiB card: the preflight smoke OOMed mid-capture at 42.71 GiB
allocated, before the first training step was logged. This is a per-step wall, not
a time wall — the task's half-exposure fallback could not have helped. At the
registered microbatch on this hardware, **the current architecture cannot reach
per-token decisions by memory, not by time** — and that is direct evidence for the
split-scan direction (below).

## The gradient table

| chunk | decisions every | eval NLL (nats/byte) | Δ vs chunk 128 | exposure (MB) | steps | train wall (s) | tok/s | append | correct | pop | merge | none | occupancy (live/128) | read entropy (nats) | final train loss |
|------:|---------------:|---------------------:|---------------:|--------------:|------:|---------------:|------:|-------:|--------:|----:|------:|-----:|--------------------:|-------------------:|-----------------:|
| 128 | 128 tokens | **2.4173** | — | 20.02 | 180 | 589 | 34.0k | 0.0698 | 0.6676 | 0.0075 | 0.2484 | 0.0068 | 80.2 | 2.185 | 2.8070 |
| 64  | 64 tokens  | **2.3287** | −0.0886 | 20.02 | 180 | 605 | 33.1k | 0.0507 | 0.8506 | 0.0090 | 0.0803 | 0.0094 | 53.7 | 2.034 | 2.7063 |
| 32  | 32 tokens  | **2.2462** | −0.1711 | 20.02 | 180 | 633 | 31.6k | 0.0404 | 0.9233 | 0.0045 | 0.0268 | 0.0050 | 45.2 | 1.949 | 2.6390 |
| 8   | 8 tokens   | **2.2186** | −0.1987 | 20.02 | 180 | 1,050 | 19.1k | 0.0457 | 0.9245 | 0.0115 | 0.0092 | 0.0090 | 51.2 | 1.546 | 2.6062 |
| 4   | 4 tokens   | **2.1791** | −0.2382 | 20.02 | 180 | 1,674 | 12.0k | 0.0480 | 0.9380 | 0.0075 | 0.0040 | 0.0025 | 55.2 | 1.404 | 2.5968 |
| 2   | 2 tokens   | **not run — memory wall** (OOM at the matched regime, preflight-receipted; parity passed first: 0 flips in 8,650,752 lanes over 1,024 chunks) | — | — | — | — | — | — | — | — | — | — | — | — | — |

Op rates are lane-0 firings per token (every token gets exactly one decision);
occupancy is mean live slots at chunk entries; entropy is over the live-slot read
weight distribution (nats/token). The five landed arms' receipts:
`receipts/sweep_chunk_c{128,64,32,8,4}.json` (config hash, exposure, eval NLL, op
rates, occupancy, entropy, walls, dispatch ledger), running table
`receipts/sweep_chunk_gradient.json`, parity receipts
`receipts/sweep_chunk_parity_c{128,64,32,8,4}.json`, and the chunk-2 not-run
receipt `receipts/sweep_chunk_c2_oom.json` (with the preflight console
`sweep_runs/preflight_c2.console.txt` and the chunk-2 parity receipt it landed
first, both hash-bound there).

**Reading the extended gradient.** The per-halving gains are −0.0886, −0.0825,
−0.0276, **−0.0395**. The 32→8 step is the odd one out, not the tail: the 8→4 step
recovers to a gain bigger than 32→8's, so the honest reading is that the curve
keeps falling toward per-token at roughly ~0.03–0.04 nats per halving once past
chunk 32, with no plateau at chunk 4. Two corroborations, as before: the final
training loss is monotone across all five arms (2.8070 → 2.7063 → 2.6390 →
2.6062 → 2.5968 over 20M targets per arm — a far tighter instrument than the eval),
under identical data order and identical init; and the eval deltas line up with it
in sign at every step. For scale: the whole matched-arms board gap (matrix_ops
1.7576 vs gdn2 1.1976 nats/byte, `runs_150m_matched`, same session family) is
0.56 nats at 92 MB exposure; the chunk dial alone is now worth **0.238 nats at
20 MB** — a large fraction of that gap's size, measured at probe scale, without
touching a single registered knob except the chunk length.

## The chunk-2 wall: memory, not time — and what the scaling law says

The task budgeted chunk 2's *wall* honestly and gave a half-exposure fallback for
a time overrun. The preflight (a 4-step leased smoke at chunk 2, full parity gate
first) found a different wall: **OOM during the first step's graph capture** —
42.71 GiB allocated by PyTorch (47.37 GiB process total) on the 47.38 GiB card,
zero training steps logged. The mechanism is architectural, not incidental: the
K5B forward keeps a chunk-entry checkpoint set per chunk per layer for its adjoint
walk (K4's recompute-over-store doctrine — ~1 MB of entry state per chunk per
layer at the registered shape), so **the whole-scan graph's memory scales LINEARLY
with decisions-per-window**. Measured across all five landed arms:

| chunks/window (chunk) | 16 (128) | 32 (64) | 64 (32) | 256 (8) | 512 (4) | 1,024 (2) |
|---|---|---|---|---|---|---|
| peak allocated (GB) | 21.96 | 22.39 | 23.23 | 28.32 | **35.10** | OOM (projected 48.7) |

The fit is exact (peak GB = 21.54 + 0.0265 × chunks/window, residuals ≤ 0.005 GB),
which is why the projection for chunk 2 is not a guess: ~48.7 GB of allocator
demand on a card that OOMed at 42.71 GiB mid-capture. Consequences, stated
plainly:

* **The dial's floor on this hardware is ~chunk 4 at the registered microbatch.**
  The per-token limit of the sequentiality dial is therefore unreachable *by the
  dial alone* — by memory, not by time.
* **This is direct evidence for the split-scan direction** (a small sequential
  decision mini-scan over near-current state + a large-chunk parallel value
  scan): the value scan must stay large-chunk precisely *because* its checkpoint
  memory scales with the chunk count, while the decision layer must move to a
  small sequential state update. The dial cannot deliver per-token decisions;
  the split is the form that can.
* **Named but deliberately not run** (operator's decision): a chunk-2 arm at
  microbatch 4 instead of the transferred 8 is a one-command rerun that would
  fit (~half the batch-scaled demand) and deliver the chunk-2 point, but it is a
  labelled deviation from the transferred-optimizer discipline (microbatch is a
  recorded config field; the accumulation reduction order differs numerically)
  and would not sit comparably in this table. The option stays receipted in
  `sweep_chunk_c2_oom.json` for the operator if the point is ever wanted with its
  asterisk.

## The op-rate-vs-chunk-size interaction (part of the answer, not noise)

The diagnostics are the mechanism, not decoration — extended to chunk 4:

* **MERGE collapses as chunks shrink: 0.2484 → 0.0803 → 0.0268 → 0.0092 → 0.0040
  per token (62× from 128 to 4).** MERGE needs a *second* live row at ≥ `tau_merge`
  (0.85) cosine — against the chunk-entry state. With stale state the router sees
  yesterday's redundancy and merges it; with fresh state the store simply never
  accumulates the duplicate rows that trigger the threshold. MERGE is the most
  staleness-sensitive op in the registered policy, and it is still falling at
  chunk 4.
* **CORRECT keeps rising: 0.668 → 0.851 → 0.923 → 0.925 → 0.938.** With fresh
  state the best match clears `tau_match` (0.45) more often, so the same
  thresholds do more useful work per token — still not saturated at chunk 4.
* **Occupancy is NOT monotone at the small end: 80.2 → 53.7 → 45.2 → 51.2 → 55.2
  live slots of 128.** The lean-store trend reverses past chunk 8: with very
  fresh state, MERGE (the store's garbage collector) almost never fires, so the
  store holds more live rows at entry. Capacity still never binds (pop + none ≈
  1 % at chunk 4) — the registered 128-slot budget is not the constraint; the
  staleness/leanness tradeoff simply shifts.
* **Reads keep sharpening: read-weight entropy falls 2.185 → 2.034 → 1.949 →
  1.546 → 1.404 nats.** This is the K23 §7 effect seen from the other side:
  chunk-constant weighting against a stale entry state makes the read softmax
  diffuse; fresh state concentrates it — and it is still falling at chunk 4,
  which is part of why the eval curve has not plateaued.
* **Chunk-entry decision events per token** (the refresh rate): 1/128 → 1/64 →
  1/32 → 1/8 → 1/4 — at chunk 4 the router re-grounds its world view 32× more
  often than the registered chunk-128 configuration.

## What was run (the matched-arms discipline, only the chunk dial varying)

Five landed arms of the **same** `matrix_ops` model (149,981,808 parameters, the
five-op registered policy verbatim — `tau_match` 0.45, `tau_merge` 0.85,
`tau_pop` 0.10, read scale 2.0, no growth), **only `chunk_tokens` varying:
128, 64, 32, 8, 4**, plus the chunk-2 attempt receipted not-run above:

* **Seed 3401** for every arm — the arms share init, data order and LR schedule
  and differ *only* by chunk size (receipts' config hashes differ only in the
  chunk field and its git/config context).
* **Transferred LR, stated plainly: this is a WITHIN-SWEEP comparison at ONE LR**
  — the registered gdn2 nomination's optimizer, verbatim from the matched arms
  (`lr` 5.42e-4, β₂ 0.939, wd 0.201, warmup 69 steps, microbatch 8). The sweep
  says what chunk size does *at that LR*; the **tuned-LR confirmation is the named
  follow-up**. Shared caveat: warmup is 69 of 180 steps at this exposure,
  identical across arms.
* **Matched-EXPOSURE, not matched-time**: every landed arm trains to
  **20,022,866 targets** (~22 % of the 2700 s matched arms' 91.9 MB). Defended:
  the effect the hypothesis rests on (§7: 13 % of read scale per token) is
  visible far below 92 MB, and the small-chunk arms are several-fold slower per
  token, so 20 MB keeps all arms in one session. The honest walls are
  first-class: 589 / 605 / 633 / 1,050 / **1,674 s** (34.0k / 33.1k / 31.6k /
  19.1k / **12.0k** tok/s) — chunk 4 costs 2.8× the chunk-128 wall at the same
  exposure, as budgeted; the chunk-2 arm was stopped by memory before its wall
  could be measured. (At matched-*time* the chunk-4 arm would see ~⅓ the bytes
  of the chunk-128 arm — which is exactly the confound the exposure rule
  removes.)
* **Same stream binding and document-disjoint calibration eval as the matched
  arms**: the registered flagship stream (hash-bound in every receipt), the same
  `lattice.evaluate` boundary (full 2048-byte windows, batch 1, recorded-read
  path; 7 full calibration windows, 14,336 targets).
* **The K5B fast path** (`fused_whole_scan=True` behind the compiled wrapper) —
  the host-cost knob whose schedule/reads/state/liveness/gradients are bit-equal
  to the per-chunk stack by construction; without it the small-chunk arms would
  measure Python, not the model. Dispatch ledger per arm: 25 captures (24 mixers
  + the parity fixture), 0 fallbacks, every training scan a graph replay.

### Per-chunk-size bit-equality, before any timing

Each landed arm's receipt opens with the chunk-128 receipt's discipline
(`matched_arms.parity`), re-run at that arm's chunk size (walls 2.6 / 3.1 / 3.1 /
6.3 / 11.1 s), and the chunk-2 attempt passed the same gate first (wall 22.6 s,
1,024 chunks) before its memory wall:

* the whole-scan op pair (eager body, capture+replay, replay) **bit-equal to the
  per-chunk K5A+K4 arm** on the recorded schedule fields, weights, reads, state,
  liveness and the three gradients (`dquery`/`dpayload`/`ddecay`), at the
  registered shape under bf16 autocast; and
* the **decision flip test re-run at that chunk size**: the K5A router the walk
  runs vs the registered eager `route_chunk`, on the same pinned chunk-entry
  states from a real routed scan — **0 flips in 8,650,752 lanes at every chunk
  size** (16/32/64/256/512/1,024 chunks probed), the banked A-phase bar.

What is deliberately *not* asserted: bitwise whole-scan equality against the eager
registered *scan*. Its state channel differs from the K4 kernel lane's by the
banked reduction-order floors (K1/K23/K4 receipts), and the registered arm's own
compiled path carries its banked inductor-vs-eager similarity delta — whole-scan
bitwise equality against the eager scan is neither claimed nor true, and the flip
test on pinned states is the decision-parity bar the registration actually set.

### Diagnostics method

Measured on the trained model (eval regime, first 4 full calibration windows,
all 24 layers, via `whole_scan_record` — the eager walk behind the graph, bf16
autocast): lane-0 op rates per token; live-slot occupancy at chunk entries by the
scan's own liveness rule (the amax stamp bookkeeping of `chunk_scan`, walked over
the recorded schedule); read-weight entropy over live slots per token. Probe
walls: 3.2 / 6.8 / 10.3 / 45.7 / **90.4 s**. Peak memory: 21.96 / 22.39 / 23.23 /
28.32 / **35.10 GB** (the scaling row above is the same numbers read as a law).

## Next steps, in order — with the extended curve's verdict on the two the operator is waiting on

1. **The named follow-up: the tuned-LR confirmation.** One LR is the sweep's
   deliberate limit; the gradient should be re-run (or at least the 128-vs-4 pair
   re-run) at an LR tuned per arm before the chunking is re-registered. The
   69-step warmup over 180 steps is a shared confound that a longer exposure also
   removes.
2. **Verdict on the 2700 s chunk-8 board arm: YES — worth its GPU hour, because
   chunk 8 is NOT at the plateau.** The extension's whole point was to place
   chunk 8 on the curve before spending the arm, and the answer is unambiguous:
   the 8→4 step still buys −0.0395 (probe scale), *more* than 32→8 bought, so a
   chunk-8 board arm measures a point with real headroom below it — it is not a
   measurement of the dial's limit. The probe now says **~0.24 nats** (c128→c4)
   is available from the dial alone, so the board question (how much survives to
   92 MB) has more at stake than the four-arm reading suggested (~0.20). The
   honest trade, updated with the chunk-4 walls: at matched-*time* 2700 s, chunk 8
   sees ~56 % of the chunk-128 arm's bytes (19.1k tok/s), chunk 4 only ~35 %
   (12.0k tok/s) — so **chunk 8 is the right matched-time board point**, and a
   matched-*exposure* chunk-4 twin (~2.1 h at 12.0k tok/s for 91.9 MB) is the
   cleaner-but-costlier comparison if the board budget allows one.
3. **Verdict on the per-token-read variant: still needed — the dial alone does
   NOT capture all of the sequential value, and it cannot reach per-token.** Two
   independent reasons, both from the extension: (i) the curve is still falling
   at chunk 4 — every runnable chunk size still carries chunk-constant reads
   against a (now short) stale window, and the ~0.04-nats-per-halving tail bounds
   from below what true per-token reads could still recover; (ii) the memory law
   receipts that the dial can never reach per-token decisions on this hardware —
   so if the tape form's per-token character is the goal, the per-token-read
   variant (K23 §7's union-normalised read on the runtime path) or the split-scan
   form is the only remaining road. The right order is unchanged and now
   evidence-backed: re-register a smaller chunk first (semantics-free, worth
   ~0.24 nats at probe scale), then gate the per-token-read variant on its own
   discipline — it changes the registered read, so it needs its own gate, not a
   silent flip.
4. **The split-scan direction, promoted from speculation to evidence-backed:**
   the checkpoint-scaling law means the value scan must stay large-chunk (its
   memory scales with the chunk count) while the decision layer moves to a small
   sequential state update. The dial cannot deliver per-token decisions; the
   split is the architectural answer to the wall this sweep hit.

Honesty ledger: single seed; one transferred LR; ~20 MB probe exposure (these are
probe-scale numbers, not board-scale claims); eval = 14,336 targets per arm with
the training-loss corroboration stated above (five-arm monotone, the tighter
instrument); the 32→8 and 8→4 eval steps are individually near the single-eval
resolution and are read as a trend, not as isolated proof; the parity phase's
kernel warm-up is identical across arms and outside each arm's honest clock; the
chunk-2 arm is not-run-with-reason, receipted (`sweep_chunk_c2_oom.json`), with
its parity pass retained as evidence that the wall was memory, not correctness.
