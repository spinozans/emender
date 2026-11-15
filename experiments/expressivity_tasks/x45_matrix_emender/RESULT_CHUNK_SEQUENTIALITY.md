# The chunk-size sequentiality sweep — is the registered chunking the quality culprit?

**Status: RUNNING — arms land one at a time; the table fills as receipts commit.**

## The hypothesis, and the dial

The operator's hypothesis: the matrix form's quality loss comes from disrupting the
**sequential character** of the algorithm — decisions and reads taken against the
chunk-*entry* state at chunk granularity instead of per token. The receipted evidence
it rests on is `K23_READ.md` §7: the union-normalised diagnostic recomputes the read
**at every token** against the state as it is at that token, and at the arm's shape
(rank 128, chunk 128) from a populated store the per-token-weighted read differs from
the registered chunk-constant one by a **median 13 % / max 32 % of the read scale** —
a first-order effect, not noise. The registered read is chunk-constant *by
construction* (`schedule.route_chunk` takes its softmax once per chunk against the
entry state; that is what keeps the scan affine and chunk-parallel), so the effect is
a property of the registered architecture, not a bug.

**Chunk size is the one dial that interpolates sequentiality without touching
registered semantics**: chunk *C* = decisions every *C* tokens and reads blind to
in-chunk writes for *C* tokens; small chunks approach per-token behaviour with the
same thresholds, the same ops, the same read definition. If quality climbs as chunks
shrink, sequential disruption is confirmed and the registered chunking is the named
culprit; if the gradient is flat, the hypothesis is dead at this scale and the
LR/architecture lines own the gap.

## The sweep

Four arms of the **same** `matrix_ops` model (149,981,808 parameters, the five-op
registered policy, rank 128), **only `chunk_tokens` varying: 128, 64, 32, 8**:

* **Seed 3401** (the standing seed) for every arm — the arms share init, data order
  and LR schedule, so they differ *only* by chunk size.
* **Transferred LR, stated plainly: this is a WITHIN-SWEEP comparison at ONE LR** —
  the registered gdn2 nomination's optimizer, verbatim from the matched arms
  (`lr` 5.42e-4, β₂ 0.939, wd 0.201, warmup 69 steps, microbatch 8). The sweep says
  what chunk size does *at that LR*; the **tuned-LR confirmation is the named
  follow-up**. One shared caveat stated up front: warmup is 69 of ~165 steps at this
  exposure, identical across arms.
* **Matched-EXPOSURE, not matched-time**: every arm trains to **20,000,000 targets**
  (~22 % of the 2700 s matched arms' 91.9 MB exposure). Defended: the effect the
  hypothesis rests on is 13 % of the read scale per token — visible far below 92 MB
  — and the chunk-8 arm is several-fold slower per token, so 20 MB keeps all four
  arms inside one session. The honest elapsed wall is a first-class column.
* **Same stream binding and document-disjoint calibration eval as the matched
  arms**: the registered flagship stream (hash-bound in every receipt), the same
  `lattice.evaluate` boundary (full 2048-byte windows, batch 1, recorded-read path).
* **The K5B fast path** (`fused_whole_scan=True` behind the compiled wrapper) —
  the host-cost knob whose schedule/reads/state/liveness/gradients are bit-equal to
  the per-chunk K5A+K4 stack by construction. Without it the chunk-8 arm would be
  host-bound, not device-bound, and the sweep would measure Python.

### Per-chunk-size bit-equality, before any timing

Each arm's receipt (`receipts/sweep_chunk_parity_c{N}.json`) opens with the chunk-128
receipt's discipline (`matched_arms.parity`), re-run at that arm's chunk size:

* the whole-scan op pair (eager body, capture+replay, replay) **bit-equal to the
  per-chunk K5A+K4 arm** on the recorded schedule fields, weights, reads, state,
  liveness and the three gradients, at the registered shape under bf16 autocast; and
* the **decision flip test re-run at that chunk size** (added by this sweep): the
  K5A router the walk runs vs the registered eager `route_chunk`, on the same pinned
  chunk-entry states from a real routed scan — **zero flips in 8,650,752 lanes**
  required at every chunk size (the banked A-phase bar).

What is deliberately *not* asserted: bitwise whole-scan equality against the eager
registered *scan*. Its state channel differs from the K4 kernel lane's by the banked
reduction-order floors (K1/K23/K4 receipts), and the registered arm's own compiled
path carries its banked inductor-vs-eager similarity delta — whole-scan bitwise
equality against the eager scan is neither claimed nor true, and the flip test on
pinned states is the decision-parity bar the registration actually set.

### Diagnostics per arm

Measured on the trained model (eval regime, first 4 full calibration windows, all 24
layers, via `whole_scan_record` — the eager walk behind the graph):

* **op-firing rates per token** (lane 0: the one decision every token gets) — the
  op-mix-vs-chunk-size interaction is part of the answer, not noise: at small chunks
  the router sees fresher state, so the append/correct/pop/merge mix and the tau
  thresholds interact differently;
* **live-slot occupancy** at chunk entries (the scan's own liveness rule, walked over
  the recorded schedule);
* **read-weight entropy** (nats over the live-slot weight distribution, per token).

## The gradient table

| chunk | decisions every | eval NLL (nats/byte) | exposure (MB) | steps | wall (s) | tok/s | append | correct | pop | merge | none | occupancy (live/128) | read entropy (nats) |
|------:|----------------:|---------------------:|--------------:|------:|---------:|------:|-------:|--------:|----:|------:|-----:|---------------------:|--------------------:|
| 128   | 128 tokens | *pending* | | | | | | | | | | | |
| 64    | 64 tokens  | *pending* | | | | | | | | | | | |
| 32    | 32 tokens  | *pending* | | | | | | | | | | | |
| 8     | 8 tokens   | *pending* | | | | | | | | | | | |

Receipts: `receipts/sweep_chunk_c{128,64,32,8}.json` (config hash, exposure, eval
NLL, op rates, occupancy, entropy, walls, dispatch ledger), running table in
`receipts/sweep_chunk_gradient.json`.

## Verdict

*Pending — the four points.*

The three readings the sweep distinguishes, decided by the numbers:

* **(a) monotone improvement as chunks shrink** → sequential disruption confirmed;
  next step is the per-token-read variant (the §7 diagnostic on the runtime path)
  and/or re-registering at small chunks.
* **(b) optimum in the middle** → the registered 64/128 sits on the wrong side and
  the sweep finds the right side.
* **(c) flat** → the hypothesis is dead at this scale and the LR/architecture lines
  own the gap.

Honesty notes carried into the verdict: single seed (the standing one), one
transferred LR (stated above), ~20 MB exposure (a probe, not a headline number),
warmup 69 of ~165 steps (shared across arms).
