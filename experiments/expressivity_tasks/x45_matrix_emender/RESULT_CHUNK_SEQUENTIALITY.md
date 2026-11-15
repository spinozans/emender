# The chunk-size sequentiality sweep — the gradient says the registered chunking IS a quality culprit

**Verdict (a): monotone improvement as chunks shrink.** Eval NLL falls at every step
of the sweep — chunk 128 → 64 → 32 → 8 gives **2.4173 → 2.3287 → 2.2462 → 2.2186
nats/byte** at identical exposure (20,022,866 targets, 180 steps, identical data
order), identical seed (3401), and one transferred LR. The sequentiality hypothesis
is confirmed at this scale: the registered chunk-granular decisions/reads (chunk-entry
state, chunk-constant read weights) cost measurable quality, and shrinking the chunk —
the one dial that interpolates sequentiality without touching registered semantics —
climbs monotonically. The named follow-ups are at the end.

## The gradient table

| chunk | decisions every | eval NLL (nats/byte) | Δ vs chunk 128 | exposure (MB) | steps | train wall (s) | tok/s | append | correct | pop | merge | none | occupancy (live/128) | read entropy (nats) | final train loss |
|------:|---------------:|---------------------:|---------------:|--------------:|------:|---------------:|------:|-------:|--------:|----:|------:|-----:|--------------------:|-------------------:|-----------------:|
| 128 | 128 tokens | **2.4173** | — | 20.02 | 180 | 589 | 34.0k | 0.0698 | 0.6676 | 0.0075 | 0.2484 | 0.0068 | 80.2 | 2.185 | 2.8070 |
| 64  | 64 tokens  | **2.3287** | −0.0886 | 20.02 | 180 | 605 | 33.1k | 0.0507 | 0.8506 | 0.0090 | 0.0803 | 0.0094 | 53.7 | 2.034 | 2.7063 |
| 32  | 32 tokens  | **2.2462** | −0.1711 | 20.02 | 180 | 633 | 31.6k | 0.0404 | 0.9233 | 0.0045 | 0.0268 | 0.0050 | 45.2 | 1.949 | 2.6390 |
| 8   | 8 tokens   | **2.2186** | −0.1987 | 20.02 | 180 | 1,050 | 19.1k | 0.0457 | 0.9245 | 0.0115 | 0.0092 | 0.0090 | 51.2 | 1.546 | 2.6062 |

Op rates are lane-0 firings per token (every token gets exactly one decision);
occupancy is mean live slots at chunk entries; entropy is over the live-slot read
weight distribution (nats/token). All four receipts:
`receipts/sweep_chunk_c{128,64,32,8}.json` (config hash, exposure, eval NLL, op
rates, occupancy, entropy, walls, dispatch ledger), running table
`receipts/sweep_chunk_gradient.json`, parity receipts
`receipts/sweep_chunk_parity_c{N}.json`.

**Reading the gradient.** The improvement is monotone but decelerating:
128→64 buys −0.0886, 64→32 buys −0.0825, 32→8 buys −0.0276. That shape is what the
hypothesis predicts — the damage comes from *staleness* (decisions/reads taken against
chunk-entry state), so halving the chunk halves the staleness until the per-token
limit is approached; chunk 8 is still 8 tokens of staleness, not zero. For scale: the
whole matched-arms board gap (matrix_ops 1.7576 vs gdn2 1.1976 nats/byte,
`runs_150m_matched`, same session family) is 0.56 nats at 92 MB exposure; the chunk
dial alone is worth **0.199 nats at 20 MB** — a large fraction of that gap's size,
measured at probe scale, without touching a single registered knob except the chunk
length.

**Evidence quality, stated honestly.** The eval boundary is 7 full windows
(14,336 targets), so a *single* 32→8 step (−0.028) is marginal on its own. The trend
is not: all four points are monotone in eval NLL **and** in final training loss
(2.807 → 2.706 → 2.639 → 2.606 over 20M targets per arm — a far tighter instrument),
under identical data order and identical init. Single seed (the standing one, 3401);
one transferred LR (below).

## The op-rate-vs-chunk-size interaction (part of the answer, not noise)

The diagnostics are the mechanism, not decoration:

* **MERGE collapses as chunks shrink: 0.2484 → 0.0803 → 0.0268 → 0.0092 per token
  (27× from 128 to 8).** MERGE needs a *second* live row at ≥ `tau_merge` (0.85)
  cosine — against the chunk-entry state. With stale state the router sees
  yesterday's redundancy and merges it; with fresh state the store simply never
  accumulates the duplicate rows that trigger the threshold. MERGE is the most
  staleness-sensitive op in the registered policy.
* **CORRECT rises to saturation: 0.668 → 0.851 → 0.923 → 0.925.** With fresh state
  the best match clears `tau_match` (0.45) far more often, so the same thresholds do
  more useful work per token.
* **The store stays leaner: occupancy at chunk entry falls 80.2 → 53.7 → 45.2 → 51.2
  live slots of 128.** Capacity pressure never binds at this scale (pop + none ≈
  1–2 % at every chunk size), so the registered 128-slot budget is not the binding
  constraint here — staleness is.
* **Reads get sharper: read-weight entropy falls 2.185 → 2.034 → 1.949 → 1.546
  nats.** This is the K23 §7 effect seen from the other side: chunk-constant
  weighting against a stale entry state makes the read softmax diffuse; fresh state
  concentrates it. The 13 %-of-read-scale median distortion that §7 measured at
  chunk 128 is the same phenomenon this sweep shows moving with the dial.
* **Chunk-entry decision events per token** (the refresh rate): 1/128 → 1/64 → 1/32
  → 1/8 — at chunk 8 the router re-grounds its world view 16× more often than the
  registered chunk-128 configuration.

## What was run (the matched-arms discipline, only the chunk dial varying)

Four arms of the **same** `matrix_ops` model (149,981,808 parameters, the five-op
registered policy verbatim — `tau_match` 0.45, `tau_merge` 0.85, `tau_pop` 0.10,
read scale 2.0, no growth), **only `chunk_tokens` varying: 128, 64, 32, 8**:

* **Seed 3401** for every arm — the arms share init, data order and LR schedule and
  differ *only* by chunk size (receipts' config hashes differ only in the chunk
  field and its git/config context).
* **Transferred LR, stated plainly: this is a WITHIN-SWEEP comparison at ONE LR**
  — the registered gdn2 nomination's optimizer, verbatim from the matched arms
  (`lr` 5.42e-4, β₂ 0.939, wd 0.201, warmup 69 steps, microbatch 8). The sweep says
  what chunk size does *at that LR*; the **tuned-LR confirmation is the named
  follow-up**. Shared caveat: warmup is 69 of 180 steps at this exposure, identical
  across arms.
* **Matched-EXPOSURE, not matched-time**: every arm trains to **20,022,866 targets**
  (~22 % of the 2700 s matched arms' 91.9 MB). Defended: the effect the hypothesis
  rests on (§7: 13 % of read scale per token) is visible far below 92 MB, and the
  chunk-8 arm is several-fold slower per token, so 20 MB keeps all four arms in one
  session. The honest walls are first-class: 589 / 605 / 633 / **1,050 s**
  (34.0k / 33.1k / 31.6k / **19.1k** tok/s) — chunk 8 costs 1.8× the chunk-128 wall
  at the same exposure, as budgeted. (At matched-*time* the chunk-8 arm would see
  ~⅓ the bytes of the chunk-128 arm — which is exactly the confound the exposure
  rule removes.)
* **Same stream binding and document-disjoint calibration eval as the matched
  arms**: the registered flagship stream (hash-bound in every receipt), the same
  `lattice.evaluate` boundary (full 2048-byte windows, batch 1, recorded-read path;
  7 full calibration windows, 14,336 targets).
* **The K5B fast path** (`fused_whole_scan=True` behind the compiled wrapper) — the
  host-cost knob whose schedule/reads/state/liveness/gradients are bit-equal to the
  per-chunk stack by construction; without it the chunk-8 arm would measure Python,
  not the model. Dispatch ledger per arm: 25 captures (24 mixers + the parity
  fixture), 0 fallbacks, every training scan a graph replay.

### Per-chunk-size bit-equality, before any timing

Each arm's receipt opens with the chunk-128 receipt's discipline
(`matched_arms.parity`), re-run at that arm's chunk size (walls 2.6 / 3.1 / 3.1 /
6.3 s):

* the whole-scan op pair (eager body, capture+replay, replay) **bit-equal to the
  per-chunk K5A+K4 arm** on the recorded schedule fields, weights, reads, state,
  liveness and the three gradients (`dquery`/`dpayload`/`ddecay`), at the
  registered shape under bf16 autocast; and
* the **decision flip test re-run at that chunk size**: the K5A router the walk
  runs vs the registered eager `route_chunk`, on the same pinned chunk-entry states
  from a real routed scan — **0 flips in 8,650,752 lanes at every chunk size**
  (16/32/64/256 chunks probed), the banked A-phase bar.

What is deliberately *not* asserted: bitwise whole-scan equality against the eager
registered *scan*. Its state channel differs from the K4 kernel lane's by the banked
reduction-order floors (K1/K23/K4 receipts), and the registered arm's own compiled
path carries its banked inductor-vs-eager similarity delta — whole-scan bitwise
equality against the eager scan is neither claimed nor true, and the flip test on
pinned states is the decision-parity bar the registration actually set.

### Diagnostics method

Measured on the trained model (eval regime, first 4 full calibration windows, all 24
layers, via `whole_scan_record` — the eager walk behind the graph, bf16 autocast):
lane-0 op rates per token; live-slot occupancy at chunk entries by the scan's own
liveness rule (the amax stamp bookkeeping of `chunk_scan`, walked over the recorded
schedule); read-weight entropy over live slots per token. Probe walls: 3.2 / 6.8 /
10.3 / 45.7 s. Peak memory: 21.96 / 22.39 / 23.23 / 28.32 GB (the 256-chunk
checkpoint set is the chunk-8 increment; the card holds 47 GB).

## Next steps, in order

1. **The named follow-up: the tuned-LR confirmation.** One LR is the sweep's
   deliberate limit; the gradient should be re-run (or at least the 128-vs-8 pair
   re-run) at an LR tuned per arm before the chunking is re-registered. The 69-step
   warmup over 180 steps is a shared confound that a longer exposure also removes.
2. **Scale confirmation at the board's rule**: a 2700 s matched-time (or
   matched-exposure) matrix_ops arm at a small chunk (8 or 32) against the banked
   chunk-128 arm's 1.7576 — the probe says ~0.2 nats is available; the board says
   how much survives to 92 MB. Note the honest trade: chunk 8 pays 1.8× the wall at
   matched exposure, so a matched-time small-chunk arm sees ~55 % of the chunk-128
   arm's bytes — the matched-exposure twin is the cleaner comparison.
3. **The per-token-read variant**: K23 §7's union-normalised read on the runtime
   path — the direct fix for the measured 13 %-of-read-scale chunk-constant
   weighting cost, and the natural companion to a small-chunk re-registration.
   (It changes the registered read, so it needs its own gate discipline — not a
   silent flip.)

Honesty ledger: single seed; one transferred LR; ~20 MB probe exposure (these are
probe-scale numbers, not board-scale claims); eval = 14,336 targets per arm with the
training-loss corroboration stated above; the parity phase's kernel warm-up is
identical across arms and outside each arm's honest clock.
