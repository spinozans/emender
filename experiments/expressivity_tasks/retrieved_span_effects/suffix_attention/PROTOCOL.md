# Suffix-attention ladder at language scale (SAL-1)

Operator directive 2026-11-27: earlier small-archive pilots were too small to test the
theory; pursue attention over XSA-type/suffix-structured memory at **reasonable language
modeling scale**, autonomously, with any experiments needed. This campaign registers that
program. It follows [XSA_ATTENTION_LAW.md](../XSA_ATTENTION_LAW.md) exactly: the count
skeleton (order 0) is the mandatory exact control, and the learned residual (order 2 with an
exact matched core) must earn its keep **on top of the skeleton**, not instead of it.

## Arms (separately co-trained, paired seeds 5751/5752)

Same causal byte host for every arm: width 128, 4 pre-norm blocks, 64-target reset windows,
FP32, no TF32, AdamW 2e-3 (.9/.98, decay .01, clip 1), 32-step warmup then cosine to 2e-4,
matched starts and identical target order within seeds. Reader width 64.

* **LOCAL**: host only; local rows + no-read. No archive.
* **SKELETON** (order 0, inside the softmax): the exact count law over the whole archive —
  per-query exact-depth classes (m_ℓ, h_ℓ) for ℓ = 0..8 from static suffix counts, values =
  literal successor histograms (pointer kind), one learned corpus log-prior and the nine
  learned depth weights g(ℓ). No learned keys, no bank to rebuild; its per-update index cost
  is zero because all counts are static and prebound.
* **ORDER2** (order 2 + exact core): learned keys/values, current-weight bank rebuilt every
  update; the deepest matched suffix interval (leaf-aligned, cap 256 rows per query, larger
  intervals fall back to moment nodes) scored exactly per row; the complement covered by a
  complete dyadic tree of 128-row leaves carrying Gaussian moment statistics (mean key,
  covariance, mean value, value–key tilt) with per-node radius certificates; learned corpus
  log-prior. No stale summaries, ever.

Primary comparisons: **SKELETON vs LOCAL** (does the exact skeleton pay at a 4:1 archive
ratio?) and **ORDER2 vs SKELETON** (does the learned residual pay at matched access?).
Secondary: fidelity ladder and achieved certificate ε at a reduced archive where dense exact
attention is computable.

## Data (fresh, document-disjoint, TRAIN roles only)

Fresh contiguous suffix after **all** prior reservations (the closed panel and the raw-byte
pilot included), pinned manifest, checked C14 4,097-byte windows, within-excerpt context
only. No reuse of any previous BUILD/FIT/PROBE role; no reopening of any closed panel.

* **BUILD**: 2,048 documents → 8,388,608 archive rows (a power-of-two multiple of the
  128-row leaf: a complete dyadic tree). 512× the previous pilot's archive.
* **FIT**: 512 documents → 2,097,152 shifted targets, each seen **once** per fit; 2,048
  updates at 16×64 targets.
* **PROBE**: 128 documents → 524,288 scored targets plus first bytes, opened once after all
  fits are frozen and independently witnessed.

All counts, covers, core row ids and histograms are prebound per batch before training and
verified against brute force on invented data; they never depend on weights.

## Gates, in order

1. **Phase A (this code, CPU, invented bytes only)**: brute-force oracles — skeleton equals
   depth-scored dense attention exactly; telescoped classes sum to all rows; order-2 beats
   order-1 under dispersion and equals dense at zero dispersion (gradients included);
   radius certificates hold and merge conservatively; moment merges match direct statistics;
   complement covers partition the archive; skeleton gradients reach its parameters.
   Publish the passing source **before** any new natural membership is opened.
2. **Phase B (data + binding)**: reservation scripts qualify on invented inputs; verify the
   fresh suffix, document disjointness, counts/coverage oracles at full scale, batch memory
   (per-batch stats, never whole-split histograms), and the pointer/vector LM integration
   parity before FIT extraction.
3. **Phase C (hardware)**: staged single-use no-wait leases through the standard gate
   (detached owner, adopted broker lease, renewed heartbeat + advancing witnessed units,
   timeout, verified release, numeric charge). Campaign cap **15,000 card-seconds** total;
   each stage registers its own cap first. Device parity of any ported kernel before use.
   No automatic retry; a failed stage is a result, not a rerun.
4. **Diagnostics** at a reduced BUILD (≤ 262,144 rows, where dense exact fits on the card):
   per-rung log-Z error, read TV, next-token KL, achieved certificate ε per budget, core
   coverage, and the **transverse mass fraction** (top-mass rows outside the matched suffix
   interval) — the ceiling of any suffix-aligned approximation.

## Decision rules (registered in advance)

* SKELETON ≤ LOCAL: exact count statistics do not help a co-trained byte host at a 4:1
  archive ratio — the next lever is ratio (bigger BUILD), not more learned machinery.
* SKELETON > LOCAL, ORDER2 ≤ SKELETON: the exact skeleton is the law worth shipping; park
  the learned residual.
* ORDER2 > SKELETON with small achieved ε: the learned residual pays; scale the ladder.
* ORDER2 ≈ exact at small ε but both ≤ LOCAL at the big archive: the deficit is the law or
  the prior, not the approximation.

## Reporting

Ideal bits/original byte over **all** 524,416 probe bytes per fit (same formula as the
pilot), per-seed values, paired whole-document percentile bootstrap (10,000 resamples, RNG
5750), both seed directions; matched measured training time alongside equal dose; per-arm
update clocks, parameter counts and peak allocation; actual zstd level-3 frame bytes with
and without the raw BUILD dictionary (decoded and verified) as descriptive coding baselines;
core coverage and certificate statistics for ORDER2. No domain claims (no labels are bound).
No weights, per-document scores, memberships or text in public artifacts.
