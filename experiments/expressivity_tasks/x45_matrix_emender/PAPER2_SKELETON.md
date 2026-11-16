# PAPER 2 SKELETON — the claim→evidence map for the two-paper arc

**Date:** 2026-11-16 · **Status:** skeleton, not a draft. This document maps what
Paper 2 (the decision space over sequences — the fork-resolved program's second paper,
[`../x43_fork_resolved/PROGRAM.md`](../x43_fork_resolved/PROGRAM.md) opening) can
defensibly claim from the matrix-emender arc, what receipts evidence each claim, which
falsifiers have already fired, and what is still missing. Companion synthesis:
[PROGRAM_SYNTHESIS.md](PROGRAM_SYNTHESIS.md). Paper 1's precedent is
[`../x41_unified_report/PAPER.md`](../x41_unified_report/PAPER.md) — receipt-backed,
nulls retained.

**Scale vocabulary, used consistently below:**

* **proven** — machine-checked in Lean (`formal/lean/OPERATOR_ALGEBRA_STATUS.md`,
  `lake build` green, no sorries, no new axioms; audit census in
  `build_logs/attention_algebra_validation.txt`).
* **board-scale** — a 2700 active-second arm on the registered 150 M flagship-stream
  board, single seed 3401, the honest clock, the document-disjoint calibration eval
  (7 windows / 14,336 targets).
* **probe-scale** — either 20,022,866-target matched-exposure LM arms at 150 M
  parameters, or ~30k-parameter synthetic-task sweeps (3000×256 steps). Probe-scale
  results are evidence for mechanisms and capability boundaries, never board claims.
* **pending** — pre-registered, sweep in flight or sign-off outstanding, no committed
  result.

---

## Claim 1 — Theory boundary results (the algebra and its deliberate edge)

**Claim.** The five-op store algebra is a monoid of affine operators; under a *fixed*
decision and numerical-gate schedule the chunked fold equals the sequential fold and
any ordered parenthesization is legal; the read is linear in values under a fixed
schedule; per-operator truncation normalization is *not* harmless (the obstruction),
with full-population normalization the exact resolution; zero-injection growth is
scan-legal; state-dependent management is provably non-affine. Decision
**generation** is outside every one of these theorems — the boundary is the paper's
structural point, and the empirical separators (Claim 3) witness it.

| sub-claim | evidence | status |
|---|---|---|
| monoid + scan laws + exact rank-1 decomposition | `MemoryAlgebra.AffineOperator.thenOp_assoc`, `chunked_fold_equiv`, `ordered_tree_equiv`, `EmenderMemory.step_decomposition` ([`formal/lean/OPERATOR_ALGEBRA_STATUS.md`](../../../formal/lean/OPERATOR_ALGEBRA_STATUS.md)) | **proven** |
| fixed-schedule readout linear/affine; combined scan compatibility | `AttentionOperator.readout_linear`, `fixed_schedule_scan`, `Event.combined_chunked_scan` | **proven** |
| truncation law | `Obstructions.truncation_sharpens_survivor` + `full_coefficients_truncation_invariant`, `full_truncation_mass_monotone` | **proven**; load-bearing in K7/K8 kernels ([K7_HIGH_RANK.md](K7_HIGH_RANK.md) §1.1) |
| zero-injection growth legality | `LiftedMemory.arbitrary_family_scan`, `injection_propagation` | **proven**; wired in K8 |
| non-affinity of state-dependent weights/death/support/alpha | `Obstructions.state_dependent_*_not_affine` | **proven** |
| decision generation NOT covered | `TraceMatches` is a hypothesis; OPERATOR_ALGEBRA_STATUS "Phase one assumed, estimated, and open" 3–4 | **proven absence** (the boundary statement) |

**Falsifier status.** N/A for the theorems; the *absence* claim is falsifiable by a
Lean generation-level theorem — that programme is open
([RESULT_GRANULARITY_SEPARATOR.md](RESULT_GRANULARITY_SEPARATOR.md) §7: channel
lemma machine-checked as a fixture test now; no-recovery lemma and the depth-≥2
strategy claim conjectural).

**Missing for the paper.** A written generation-level statement (the separator's §7
shape) is the natural theory section; without it the paper reports the boundary
honestly rather than closing it.

## Claim 2 — The algebra instantiation (the registered machine's semantics)

**Claim.** The five ops on an orthonormal-key matrix state are exact
(CORRECT/POP touch one basis direction), the certified read carries strengths without
renormalization, and the whole thing is implementable at the registered shape with
bit-exact decisions and floor-held reads across four independently-built paths
(eager, chunked-fused, persistent, distributed) and two granularities (chunk, token).

| sub-claim | evidence | status |
|---|---|---|
| exactness with orthonormal keys | [`../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md`](../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md) gates 2–3, held in every stage suite | receipted (engineering) |
| bit-exact decisions, 0 flips on real fixtures | K5A 8,650,752 lanes; K6 17,039,360; K7 279,040 at rank 2048; K9 0 at {256,1024,2048} (`receipts/k5a_routing_diff.json`, `k6_gates.json`, `k7_high_rank.json`, `k9_gates.json`) | receipted |
| reduction-order discipline (measured floors, never fitted) | every stage receipt; the defined-order spec resolving the K5A reproducibility obstacle ([PREREGISTRATION_SEQUENTIAL_MATRIX.md](PREREGISTRATION_SEQUENTIAL_MATRIX.md)) | receipted |

**Falsifier status.** The parity gates are the falsifiers and they are standing tests;
the K5A probe (~48 % bit-equal similarity from a Triton rewrite) is the receipted
*reason* decisions stay torch-computed in the chunked arm, and the K6 defined-order
spec is the receipted *resolution* for the new registration.

**Missing for the paper.** Nothing essential; a table of the paths and their
parity instruments is a section, not a gap.

## Claim 3 — The expressivity separators

**Claim (3a, granularity).** For this algebra's chunked form (entry-pinned reads,
entry-pinned free-slot sets), chunk-granular decision generation is an expressivity
weakening: a task family exists that the chunked machine cannot solve at any
parameters while the per-token machine — same code, operators, thresholds, only the
granularity dial moved — solves it at 99.9 %. Two-part signature: learnability side
(monotone, accelerating LM-loss gradient into the per-token limit) + expressivity side
(the chance-level mechanical witness with exactly-zero read weight on live answer
rows).

| sub-claim | evidence | status |
|---|---|---|
| SEP-AR: per-token 0.856–0.999; C=8 readable 0.984–0.989 / blind 0.289–0.322; C=32/128 at chance (0.028–0.033) across every LR and seed | [RESULT_GRANULARITY_SEPARATOR.md](RESULT_GRANULARITY_SEPARATOR.md), 48 cells `receipts/separator_ar_*.json`, aggregate `separator_ar_sweep.json` | **probe-scale, receipted** |
| P-4 mechanism: attention weight exactly 0.0 on live blind answer rows, in the scan's own recorded channel | `receipts/separator_mechanism.json` | receipted |
| refutation clause not triggered | pre-registered in commit `469bfa71` | receipted |
| learnability side: 0.390 nats/byte over the full dial (2.4173→2.0278) | `receipts/sweep_chunk_*.json`, `receipts/k6_probe.json` | **probe-scale, receipted** |
| SEP-EVICT corroboration | per-token cells 0.93–0.97; chunked cells **not run** (lane ceiling) | **partial, honestly marked** |

**Falsifier status.** Pre-registered and passed (P-1..P-4 MET, P-5 partial); the
refutation clause did not fire.

**Missing for the paper.** (i) The SEP-EVICT chunked cells (one short follow-up lane).
(ii) A generation-level Lean statement of at least the channel lemma
([RESULT_GRANULARITY_SEPARATOR.md](RESULT_GRANULARITY_SEPARATOR.md) §7) so the
empirical witness sits against a proved boundary. (iii) Honest scoping language: the
claim is about *this algebra's* chunked form, not all chunked attention.

**Claim (3b, key separation — PENDING).** Key-value separation is necessary at the
first store layer (fused key=value is mechanically role-blind: the collision lemma is
machine-checked in `test_key_separation_probe.py`), with a derived 0.5 set-retrieval
ceiling at depth 1 and a calibrated depth-substitution escape at depth 2 (composite
payloads synthesize context addressing one layer deep, at a full layer's cost).

| sub-claim | evidence | status |
|---|---|---|
| collision lemma (bit-identical payloads; mechanical identity rewrite; exact equal mass; M2 order-blindness) | [RESULT_KEY_SEPARATION.md](RESULT_KEY_SEPARATION.md) §2, machine-checked; predictions committed `63d2b681` | **machine-checked fixtures; sweep PENDING** |
| 0.5 ceiling at depth 1; depth-2 composite escape | §5 (the linear-coefficient argument + disclosed calibration cells 0.514/0.516 fused vs 1.000 separated) | **pending the 48-cell sweep** |

**Falsifier status.** Committed verbatim in the pre-registration (any fused depth-1
cell at 0.85 refutes; separated below 0.85 is inconclusive).

**Missing for the paper.** The sweep itself (running), the key-dim curve (P-5), and
the depth-2 corroboration receipts (P-6). Nothing may be claimed until they land.

**Claim (3c, Wheeler-order capacity — PENDING).** An ordered, run-coalescing store
holds Θ(r × run) of context mass where the pointwise store evicts (derived collapse
point S=8; predicted eviction-onset decay vs dilution curves with committed
quantitative bands and a refutation clause).

| evidence | status |
|---|---|
| [DESIGN_WHEELER_ORDER.md](DESIGN_WHEELER_ORDER.md) (`4acac6da`), [RESULT_WHEELER_CAPACITY.md](RESULT_WHEELER_CAPACITY.md) predictions (`abbdd903`), 15 machine-checks in `test_wheeler_capacity_probe.py` | **design + pre-registration committed; sweep PENDING** |

**Missing for the paper.** The sweep (running), and the stated dependency: both arms
sit on the SEP-KV separated-key base and reclassify if separation is refuted.

## Claim 4 — The frontier characterization (the 150 M board and its attribution)

**Claim.** At matched parameters, matched time and exposure matched to 0.13 %, the
five-op matrix arm **loses** to plain GDN2 by 0.5600 nats/byte at the 150 M board;
the registered C1 ordering is not met in the matrix form and the tape-era ops result
is scoped tape-specific; management's demonstrated contribution at this configuration
is *trainability* (the append-only control deterministically diverges at step 56,
bit-identically across two runs, while the ops arm completes 832 steps); the gap
attributions are measured, and the residual is diagnosed mechanistically as
identity-cache degeneration with read-back sharpness — not capacity — the binding
constraint.

| sub-claim | evidence | status |
|---|---|---|
| 1.7576 vs 1.1976 same-session (1.1407 recorded); exposure matched 0.13 % | [`../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md`](../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md), `runs_150m_matched/` (`b9d6402f`/`b52d4472`/`7284d7e4`) | **board-scale, receipted** |
| append-only DNF at step 56, bit-identical ×2, fail-closed upheld | same, §3, `runs_150m_matched/matrix_append/dnf.json` | **board-scale, receipted** |
| tuning closes 0.076 at board scale (13.6 %); probe-scale search asymptotes an order of magnitude short | [CMAES_TUNING.md](CMAES_TUNING.md), `receipts/cmaes_result*.json`, `runs_150m_matched/matrix_ops_cmaes_confirm/` | **board-scale + probe-scale, receipted** |
| the chunk dial closes 0.024 at board scale (~4.3 %); the probe-scale −0.199 does not transfer | [RESULT_BOARD_CHUNK8.md](RESULT_BOARD_CHUNK8.md), `receipts/board_chunk8.json` | **board-scale, receipted** |
| identity-cache diagnosis: CORRECT 60–71 % identity rewriting, MERGE dedup, POP marginal, occupancy 66–92 %, tuned gain entirely read-sharpening, query gradient 1–2 orders small | [CMAES_TUNING.md](CMAES_TUNING.md) diagnosis, `receipts/cmaes_diagnosis_*.json` | receipted (single-seed instrumentation) |
| capacity three-way tie 2.0278 / 2.0186 / 2.0240 → content diversity, not size | `receipts/k8_probe_{grow,fixed2048}.json` (`3004f950`/`ca576f41`), `receipts/k6_probe.json` | **probe-scale, receipted** |
| per-token beats the dial's best at probe scale (−0.1513 beyond chunk 4) | [K6_SEQUENTIAL_MATRIX.md](K6_SEQUENTIAL_MATRIX.md), `receipts/k6_probe.json` | **probe-scale; the board-scale arm is gated on the registration sign-off** |

**Falsifier status.** C1's own registered falsifier **fired** and was reported per its
scoping clause; C2 is untested (rank axis untouched at board scale); C3 is untested
(the saturation regime, §Open). The cross-scale falsifier also fired honestly: the
probe-scale dial gradient did not transfer to the board.

**Missing for the paper.** (i) A second seed for the headline gap (named follow-up,
not run). (ii) The sequential arm's board-scale row — gated on the operator's
sign-off of [PREREGISTRATION_SEQUENTIAL_MATRIX.md](PREREGISTRATION_SEQUENTIAL_MATRIX.md)
— without which the per-token story stays probe-scale. (iii) C2's rank sweep at board
scale, now affordable on the K5B path. (iv) The C3 saturation probe (the one regime
where capacity is testable in value). (v) The tuned-LR confirmation for the small-chunk
reading.

## Claim 5 — The growth capability

**Claim.** Growth by the registered rule is wired, legal (Lean zero-injection),
gated (0 flips in 2.16 M lanes; gradients within floors *across* rank changes;
zero-injection, prefix and determinism receipts), and demonstrated: the model climbs
128→2048 in every window, trains through every rank change, and — the honest null —
buys nothing at probe scale because the store never fills (the three-way tie of
Claim 4).

| sub-claim | evidence | status |
|---|---|---|
| capability (fires, re-dispatches, zero rows exact, backward survives, trains) | [K8_GROWTH.md](K8_GROWTH.md) §4, `receipts/k8_smoke.json`, `receipts/k8_gates.json` | **probe-scale, receipted** |
| legality + parity gates | `receipts/k8_gates.json` (0 flips in 2,162,688 lanes; grads 0.89–1.67× floors across boundaries) | receipted |
| the registered trigger fires on an idle store (identity-rewrite pressure, not occupancy) | [K8_GROWTH.md](K8_GROWTH.md) §5 | receipted, honestly retained |
| utility null (the tie) | Claim 4 row | **probe-scale null, receipted** |
| the high-rank machine that makes growth runnable at 2048 | [K7_HIGH_RANK.md](K7_HIGH_RANK.md), [K9_DISTRIBUTED_WALK.md](K9_DISTRIBUTED_WALK.md) (6.7× at rank 2048; 0 flips) | receipted (engineering) |

**Falsifier status.** The capability claim's falsifiers are the gates and they hold;
the *utility* claim's pre-stated tie-branch landed as predicted
([K8_GROWTH.md](K8_GROWTH.md) §7's verdict form).

**Missing for the paper.** The saturation-regime value test (C3) — until it runs,
growth is a capability + legality result with a probe-scale utility null, and the
paper must say exactly that.

## Claim 6 — The machine (the engineering contribution)

**Claim.** The decision-algebra machine is buildable at GDN2's throughput class with
bit-exact semantics: 434,357 → 125,033 → 384 kernel-launch-equivalents per step;
31,046 tok/s steady in the matched four-arm session (33,300 same-day draw) against
the registered arm's ~10–13 k, with the step's host gap at 24 ms; per-token decisions
at chunk-4-class throughput; any rank to 2048; growth composed. The lessons are
themselves results: the step was launch-bound, not arithmetic-bound; component wins
do not compose (the scale proof's three-scale verdict); per-chunk Python dispatch
beats launch count as the residual owner; two impossibility ceilings (register tile,
smem/grid barrier) are receipted with their measured fallbacks.

| sub-claim | evidence | status |
|---|---|---|
| launch-bound diagnosis | [PROFILE_STEP_VS_GDN2.md](PROFILE_STEP_VS_GDN2.md), `receipts/profile_step_vs_gdn2.json` | receipted (diagnosis) |
| three-scale honest verdict | [RESULT_SCALE_PROOF.md](RESULT_SCALE_PROOF.md), `receipts/scale_proof_*.json` | receipted |
| K5B: host 42.0→1.0 ms/scan; 3.25 s step; bit-equal whole-training-step loss+grads | [K5B_WHOLE_SCAN.md](K5B_WHOLE_SCAN.md), `receipts/k5b_*.json` | receipted |
| K6: per-token at 11.6 k tok/s probe arm; 32 MB vs the dial's 1 GB | [K6_SEQUENTIAL_MATRIX.md](K6_SEQUENTIAL_MATRIX.md) gate D | receipted |
| K9: 374.5→56.2 ms at rank 2048; gate-0 impossibility receipt | [K9_DISTRIBUTED_WALK.md](K9_DISTRIBUTED_WALK.md), `receipts/k9_barrier_economics.json` | receipted |
| K5A's honest negative (fusion that bought launches, not wall) | [K5A_ROUTING_DIFF.md](K5A_ROUTING_DIFF.md) | receipted |

**Falsifier status.** Each stage's gates are the falsifiers; the negatives
(K5A wall, scale-proof model step, the 32 k target not reached by K5/K5A) are retained
as results, not footnotes.

**Missing for the paper.** (i) The 32 k tok/s line was crossed only in a single-arm
draw (31,046 in the matched session) — one more matched session would pin it. (ii)
The board arms currently run the K5B chunk-granular path; the K6/K9 stack's whole-arm
throughput at board scale is projected ([K9_DISTRIBUTED_WALK.md](K9_DISTRIBUTED_WALK.md)
§3), not measured. (iii) The hi-rank backward remains the dominant scan cost after K9
(named follow-up).

---

## The paper's shape, honestly

**Can be written today (defensible):** Claims 1, 2, 3a, 4 (as a characterization with
its nulls), 5 (as capability + legality + utility null), 6. The paper's spine —
*the theorems cover replay, the experiments separate generation* — is fully receipted.

**Blocked, with the blocking item named:** 3b and 3c (the running sweeps); the
board-scale per-token row (the registration sign-off); C2/C3 (unmeasured axes); the
second seed.

**Must-not-claim list (from the arc's own receipts):** no matrix-beats-GDN2 claim at
150 M; no growth-pays claim; no transfer of probe-scale gradients to board scale
without the board arm; no theorem-status language for any separator; no "fast at all
scales" claim for any fused path (the scale proof forbids it); nothing from an
in-flight lane until its RESULT file lands.

**Positioning note.** Per the program ledger's standing discipline, no novelty
priority or comparative-performance claim follows from the positioning; the exact
structural decomposition into walks and decisions is the proposed distinction
([`../x43_fork_resolved/PROGRAM.md`](../x43_fork_resolved/PROGRAM.md), opening;
[`../x43_fork_resolved/POSITIONING.md`](../x43_fork_resolved/POSITIONING.md) for the
sweep, full-claim verification pending). Related-work verification for the
expressivity claims (against the structured-memory and separator literature) has not
been re-run for this arc and belongs in the paper's related-work pass.
