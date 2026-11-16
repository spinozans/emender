# PROGRAM SYNTHESIS — the matrix emender arc, connected into the research thread

**Date:** 2026-11-16 · **Status:** synthesis (no new measurements) · **Scope:** the
~40 commits from [`PREREGISTRATION_MATRIX_EMENDER.md`](../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md)
(2026-11-13) through the SEP-KV/Wheeler pre-registrations (2026-11-16), read back as one
research thread. Every claim below cites its receipt; nothing here is a new result, and
no receipt, registration or RESULT file was modified to produce it. Nulls are stated as
nulls, pending work as pending.

The program this arc belongs to is the fork-resolved LM's Paper 2 — the decision space
over sequences — whose ledger is [`../x43_fork_resolved/PROGRAM.md`](../x43_fork_resolved/PROGRAM.md).
Paper 1 ([`../x41_unified_report/PAPER.md`](../x41_unified_report/PAPER.md)) established
the memory's scoped boundary properties; this arc builds the machine that runs the
decision algebra, measures it honestly at the 150 M board, and separates what the
architecture *can express* from what it merely *learns slowly*.

---

## 1. The thesis, and its two degenerate points

The interval-operator framing ([`../x43_fork_resolved/THEORY_INTERVAL_OPERATORS.md`](../x43_fork_resolved/THEORY_INTERVAL_OPERATORS.md))
puts every sequence machine on one axis: what the state carries, and who manages it.

* **The transformer is recompute-everything.** Its state is the full prefix; every
  token re-reads all of it. No management exists because nothing is ever discarded or
  revised. On our own board it is the strong-but-expensive reference: dense
  transformer 1.2807 nats/byte on the recorded 150 M board
  ([`../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md`](../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md),
  "Why this exists").
* **GDN2 is fixed-state, no-management.** A rank-1 (per-head) matrix state with a
  chunked parallel scan, learned decay, and no decision machinery: nothing is
  selected, nothing is discarded deliberately. It beat the transformer on the same
  board (1.1407 recorded; 1.1976 same-session,
  [`../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md`](../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md)).
* **The emender is the generalization:** a matrix state `S ∈ ℝ^{r×d_v}` per head —
  capacity, not parameters — written by five rank-1 management ops (APPEND / CORRECT /
  POP / DECAY / MERGE) under a detached architectural decision schedule, plus
  capacity growth by zero-injection
  ([`../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md`](../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md)).
  GDN2 is the degenerate point of this family where the op set collapses to
  DECAY-plus-blind-write and `r` is fixed at 1; the transformer is the degenerate
  point where `r` grows by one per token and no op ever fires.

The tape-era motivation is retained, not forgotten: the tape store's ops earned
+0.66 nats/byte at 2.13 M parameters, but spent 78 % of its step in
gradient-checkpoint replay because its state was a per-token trajectory
([`../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md`](../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md)).
The matrix move keeps the algebra and drops the tape.

## 2. The theory layer: what Lean proves, and what it deliberately does not

All theorem names are in
[`formal/lean/OPERATOR_ALGEBRA_STATUS.md`](../../../formal/lean/OPERATOR_ALGEBRA_STATUS.md)
(phase one: monoid and scan; phase two, 2026-11-13: the attention/retrieval algebra).
`lake build` passes with no sorries and no new axioms; the audit census is retained
in `build_logs/attention_algebra_validation.txt`.

**Proven:**

* **The monoid and the scan laws.** `AffineOperator.thenOp_assoc`, `apply_one`,
  `apply_mul`; `foldl_eq_composeAll`, `composeAll_flatten`, `chunked_fold_equiv`,
  `prefix_scan_equiv`, `ordered_tree_equiv` — the chunked fold equals the sequential
  fold **for a fixed decision and numerical-gate schedule**, and any ordered
  parenthesization is legal. `injection_propagation` and
  `prefix_injection_propagation` carry in-chunk writes without inverses.
* **The exact decomposition.** `EmenderMemory.step_decomposition`,
  `token_step_eq_fold`: every step is `D ⊙ s + Σ e_j ⊗ b_j` — a diagonal decay plus
  rank-1 injections — which is what makes the five ops *scannable* at all
  ([`../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md`](../x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md),
  P1 closed).
* **The read is linear in values under a fixed schedule.**
  `AttentionOperator.readout_linear`/`readout_affine`,
  `AttentionOperator.fixed_schedule_scan`, `Event.combined_chunked_scan` — the
  fixed-schedule readout, including scheduled index reads and the store writes,
  shares one affine scan.
* **The truncation law.** `Obstructions.truncation_sharpens_survivor` (a dropped
  atom with retained normalization doubles a survivor's coefficient — the
  obstruction, with `full_coefficients_truncation_invariant`,
  `full_truncation_mass_monotone`, `full_truncation_decomposition` as the exact
  resolution: normalise once by the **full-population** mass). This law is
  load-bearing engineering: K7's split softmax and K8's cross-block combines
  implement it
  ([K7_HIGH_RANK.md](K7_HIGH_RANK.md) §1.1, [K8_GROWTH.md](K8_GROWTH.md) §1.3).
* **Zero-injection growth is legal.** `LiftedMemory.arbitrary_family_scan` /
  `injection_propagation`: new rows enter as exact zeros and the monoid and scan
  theorems hold — the legality K8 wires into a runtime capability
  ([K8_GROWTH.md](K8_GROWTH.md) §1.1, gate A receipts `receipts/k8_gates.json`).
* **State-dependent management is genuinely non-affine.**
  `Obstructions.state_dependent_weights_not_affine`,
  `state_dependent_death_not_affine`, `state_dependent_support_not_affine`,
  `state_dependent_alpha_not_affine` — the obstruction side is proved too: the
  decision channels cannot be smuggled into the affine scan.

**Deliberately NOT covered — and that gap is the arc's scientific engine:**
decision **generation**. `TraceMatches` is a hypothesis; the theorems quantify over
supplied decisions and "do not prove that decision generation itself is parallel or
state independent"
([`formal/lean/OPERATOR_ALGEBRA_STATUS.md`](../../../formal/lean/OPERATOR_ALGEBRA_STATUS.md),
"Phase one assumed, estimated, and open" items 3–4; the same statement opens
[RESULT_GRANULARITY_SEPARATOR.md](RESULT_GRANULARITY_SEPARATOR.md) §1). Nothing in
Lean says *when* the decisions may be taken. That is exactly the question the
granularity separator answers empirically (§5 below): chunk-granular decision
generation is a strictly weaker machine, and the theorem layer's conditional shape
(fixed schedules only) is the precise boundary the experiment witnesses.

## 3. The machine, as one engineering arc — and its honest lessons

The kernel lane (K1–K9) took the registered matrix arm from a 24-kernel-per-chunk
eager pipeline to a trainable, any-rank, growth-capable machine, one gated stage at a
time. Stage documents and receipts:

| stage | what it did | headline | receipt |
|---|---|---|---|
| [K1](K1_SCHEDULE_SCAN.md) | the state scan fused to one Triton kernel | 87→1 kernels per chunk; state/liveness bit-exact; reads 0.0–1.66× the measured floor | `receipts/k1_gates.json`, `receipts/k1_schedule_scan.json` |
| [K2+K3](K23_READ.md) | the read as two `tl.dot` kernels | 24→3 kernels per chunk; 1 392→48 per block call; read parity at every token ≤1.65× floor | `receipts/k23_gates.json`, `receipts/k23_read.json` |
| [no-host-loop](NO_HOST_LOOP.md) | one launch for the whole 2048-token forward (persistent kernel A; graph B; wide-chunk C) | 48→1 launches; wall/occupancy 4.60×→1.12×; rank 2048 refused **by name** | `receipts/hostloop_free.json` |
| [K4](K4_FUSED_BACKWARD.md) | the fused backward: one reverse-walk kernel | 2 027→3 kernels; block step 44.8→6.0 ms; dstate/dpayload bit-exact grads; trajectory 2.00 GiB stated | `receipts/k4_fused_backward.json` |
| [scale proof](RESULT_SCALE_PROOF.md) | honest three-scale verdict | block call fast; **model step not fast** (OOM at mb8; 2.65× slower at mb2); whole arm 16× below GDN2's recorded class | `receipts/scale_proof_{blocks,model_step,training}.json` |
| [profile](PROFILE_STEP_VS_GDN2.md) | same-session step decomposition vs GDN2 | the step is **launch-bound, not arithmetic-bound**: 434 357 kernels, 64 % host gap | `receipts/profile_step_vs_gdn2.json` |
| [K5](K5_LAUNCH_COUNT.md) | compile hosting + recompute-over-store + decision-chain fusion | mb8 parity with the registered twin at −52 % launches | `receipts/k5_*.json` |
| [K5A](K5A_ROUTING_DIFF.md) | the routing region's differentiable channels fused | 125 033 kernels/step; **wall did not improve** — per-chunk Python op dispatch owns the residual | `receipts/k5a_routing_diff.json`, `receipts/k5a_step_metric.json` |
| [K5B](K5B_WHOLE_SCAN.md) | one op per (layer, window), CUDA-graph replayed | host 42.0→1.0 ms/scan; 24 576→384 dispatches/step; **3.25 s step, 31 046 tok/s** matched session; bit-equal to the per-chunk arm, loss+grads of a whole training step included | `receipts/k5b_whole_scan.json`, `receipts/k5b_step_metric.json` |
| [K6](K6_SEQUENTIAL_MATRIX.md) | the per-token sequential arm (see §4.3) | 0 flips in 17.0 M lanes; 32 MB checkpoints where the dial OOMs at 1 GB | `receipts/k6_gates.json`, `receipts/k6_probe.json` |
| [K7](K7_HIGH_RANK.md) | the rank ceiling removed | rank 2048 executes on both sides of the scan; split softmax under the Lean truncation law; 0 flips in 279 040 lanes | `receipts/k7_high_rank.json` |
| [K8](K8_GROWTH.md) | growth wired (see §4.4) | the machine grows and trains through it | `receipts/k8_gates.json`, `receipts/k8_smoke.json` |
| [K9](K9_DISTRIBUTED_WALK.md) | the high-rank walk distributed onto the L2 | 374.5→56.2 ms at rank 2048 (6.7×); 0 flips at every gated rank | `receipts/k9_barrier_economics.json`, `receipts/k9_gates.json` |

Four lessons, each receipted, none of them the lesson the lane expected:

1. **The step was launch-bound, not arithmetic-bound.** The registered arm issued
   434 357 kernels/step whose device work summed to 3.32 s while the wall was 9.29 s;
   GDN2 issues 59 294 and stays 90 % device-busy
   ([PROFILE_STEP_VS_GDN2.md](PROFILE_STEP_VS_GDN2.md) §1). Fusing arithmetic first
   (K1–K4) optimized a component that was never the wall; the scale proof's honest
   three-scale verdict caught it
   ([RESULT_SCALE_PROOF.md](RESULT_SCALE_PROOF.md): fast block, slow step, slow arm).
   The wall only moved when the *host* was removed — K5B's one-op-per-scan CUDA-graph
   replay (host gap 0.024 s of a 3.25 s step,
   [K5B_WHOLE_SCAN.md](K5B_WHOLE_SCAN.md) gate D).
2. **Component-vs-model measurement traps are real and reproducible.** A kernel that
   wins its block boundary can lose the model step twice over: K4's trajectory store
   summed to 51.5 GB over 24 layers (OOM at the registered microbatch), and its
   autograd node could not be hosted by `torch.compile` on this stack, forcing eager
   routing and multiplying launches 4×
   ([RESULT_SCALE_PROOF.md](RESULT_SCALE_PROOF.md) §2). Both levers were then built and
   gated (K5 gates B and C). The inverse trap also fired: fusing the discrete decision
   chain was bit-identical and moved the step metric by ~0, because inductor had
   already fused it — the launches lived in the *differentiable* channels
   ([K5_LAUNCH_COUNT.md](K5_LAUNCH_COUNT.md) gate A). Measure the step, not the
   component; and when a profile attributes a cost, re-measure the attribution
   ([K5A_ROUTING_DIFF.md](K5A_ROUTING_DIFF.md): the fused routing bought the launch
   count and *lost* wall time to per-chunk Python dispatch).
3. **The untangling: sequential decisions + parallel values.** The chunk-size dial's
   memory law (peak GB = 21.54 + 0.0265 × chunks/window — exact over five arms;
   [RESULT_CHUNK_SEQUENTIALITY.md](RESULT_CHUNK_SEQUENTIALITY.md)) says a whole-scan
   graph's checkpoint memory scales with *decisions per window*, so the dial can never
   reach per-token by memory. K6 made decision granularity and checkpoint granularity
   **independent knobs**: per-token decisions with 64-token segment checkpoints, 32 MB
   per scan where the dial's chunk-2 point demands 1 GB
   ([K6_SEQUENTIAL_MATRIX.md](K6_SEQUENTIAL_MATRIX.md) §1.3). The split-scan direction
   the sweep named — a small sequential decision layer over a large-chunk parallel
   value scan — is the architectural consequence
   ([RESULT_CHUNK_SEQUENTIALITY.md](RESULT_CHUNK_SEQUENTIALITY.md), next steps 3–4).
4. **The impossibility receipts.** Two ceilings were *measured, not engineered
   around*: (a) the register tile — every one-program-owns-the-store kernel refuses
   beyond its budget by name (rank 2048 = 32 768 register floats vs a budget of 8 192;
   [NO_HOST_LOOP.md](NO_HOST_LOOP.md) §4, [K7_HIGH_RANK.md](K7_HIGH_RANK.md) §0), which
   K7 removed by moving the store to global memory under the Lean one-tree law;
   (b) the smem ceiling — Triton 3.5.1 has no grid-wide barrier and no cooperative
   launch, and the card fits 142 co-resident 64 KB CTAs = 9.3 MB of addressable
   on-chip state against the 16 MB the probe shape needs: the distributed smem design
   is **physically impossible on this stack**, receipted as a first-class deliverable
   ([K9_DISTRIBUTED_WALK.md](K9_DISTRIBUTED_WALK.md), gate 0,
   `receipts/k9_barrier_economics.json`, commits `9a316ce9`/`d319f898`). The K9 walk
   runs on the on-chip substrate the card does have — the L2 — with the graph-captured
   launch boundary as the per-token barrier.

The parity discipline held at every stage: state and liveness bit-exact against the
per-token oracle wherever the semantics are exact
([K1](K1_SCHEDULE_SCAN.md) §2, [K23](K23_READ.md) §2,
[no-host-loop](NO_HOST_LOOP.md) §2), decisions bit-identical by construction with
mandatory 0-flip tests on real routed fixtures (K5A: 0 flips in 8 650 752 lanes; K6:
17 039 360; K7: 279 040 at rank 2048; K9: 0 at every gated rank), and reductions
held to *measured* floors, never fitted constants (`NOISE_FACTOR = 8×` of
`sequential_noise_floor`). Every fusion that could not be bit-exact said so and
measured its floor instead.

## 4. The empirical results, with their honest readings

### 4.1 The 150 M matched board: C1 not met, and the cleanest stability receipt of the night

At matched parameters, matched time (2700 active seconds) and **exposure matched to
0.13 %** (91.87 M vs 91.75 M bytes — the speed program's doing, not a coincidence), the
five-op matrix arm **loses to plain GDN2 by 0.5600 nats/byte**: matrix_ops 1.7576 vs
gdn2 1.1976 same-session (1.1407 recorded)
([`../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md`](../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md)).
The registered C1 ordering *five-ops > append-only > no-management* is **not met at
150 M in this configuration**, and the falsifier's own scoping clause applies: the
tape-era 0.66-nat ops result is scoped as **tape-specific**.

The append-only control could not be trained to the boundary at all: deterministic
divergence (non-finite gradient norm) at **step 56 of 2700 s, bit-identically across
two independent runs**, under the same optimizer with which the ops arm completed 832
steps — receipted as a DNF, fail-closed rule upheld. The stability contrast is real
evidence that management stabilizes training in the matrix form; the operator's ruling
is retained that the control is whole-vs-crippled (ops-off removes death itself), so it
is not a clean ablation either ([RESULT_150M_MATCHED_ARMS.md](../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md)
§3, §5.3). Caveats stated by the board itself: single seed; the matrix arms inherit the
gdn2 nomination's optimizer **unsearched**; first 2700 s run of the architecture.

**The gap attribution, as the follow-up lanes measured it** (all against the
0.5600-nat same-session board gap):

| component | closed | evidence |
|---|---|---|
| optimizer tuning | **0.076 nats at board scale** (13.6 %) | the CMA-ES confirmation: tuned config 1.6813 vs transferred 1.7576 at the full 2700 s clock ([CMAES_TUNING.md](CMAES_TUNING.md), `receipts/cmaes_confirm.console.log`, `runs_150m_matched/matrix_ops_cmaes_confirm/`) |
| the chunk-granularity dial | **0.024 nats at board scale** (~4.3 %) | the chunk-8 board arm: 1.7336, matched-time at 57.3 % exposure — a stated lower bound; the probe-scale −0.199 does **not** transfer ([RESULT_BOARD_CHUNK8.md](RESULT_BOARD_CHUNK8.md)) |
| decision granularity, probe scale | **0.39 nats at probe scale** (K6: 2.0278 vs 2.4173 over the full dial, 20 M targets) | [K6_SEQUENTIAL_MATRIX.md](K6_SEQUENTIAL_MATRIX.md) gate C, `receipts/k6_probe.json` — a probe-scale number, **not** yet a board claim (the sequential registration's sign-off gates it) |
| the architectural residual | what remains | the diagnosis below |

**The identity-cache diagnosis — what the residual is.** The CMA-ES search (32
candidates over optimizer *and* policy thresholds, every one scoring 2.2564–2.4468 at
probe exposure) closed only 12 % of the probe gap, and the diagnosis explains the
plateau mechanistically ([CMAES_TUNING.md](CMAES_TUNING.md)): at 150 M the five ops
degenerate into a **content-addressable cache** — CORRECT 60–71 % of decisions, firing
mostly as exact-match identity rewriting over repeated byte-level payload directions;
MERGE mostly dedupes near-duplicates; POP marginal; occupancy 66–92 % — **capacity is
never the binding constraint**. The tuned config's entire gain is read-back sharpening
(read_scale 2.0→3.33, read entropy 3.4→0.7 nats), and the query channel — the only path
that could learn *what to store* — receives 1–2 orders of magnitude less gradient than
the value/head channels. This is the honest 150 M implication for the C14-era
data-efficiency claim the program defends
([`../x43_fork_resolved/RESULT_C14_DATA_EFFICIENCY.md`](../x43_fork_resolved/RESULT_C14_DATA_EFFICIENCY.md)):
that result was earned by a tape store whose write policy was budgeted per token; the
matrix store's decisions at 150 M are dominated by byte-level identity matching — a
different, weaker mechanism.

### 4.2 The capacity three-way tie, and what it localizes

The K8 three-arm probe (per-token semantics, the sweep's matched-exposure protocol:
20 022 866 targets, seed 3401, one transferred LR, registered constants) landed as the
informative null K8's own verdict form predicted
([K8_GROWTH.md](K8_GROWTH.md) §7):

| arm | eval NLL (nats/byte) | receipt |
|---|---:|---|
| fixed-128 (the K6 per-token probe) | 2.0278 | `receipts/k6_probe.json` |
| grow (start 128, registered rule, cap 2048) | **2.0186** | `receipts/k8_probe_grow.json` (commit `3004f950`) |
| fixed-2048 (rank 2048 from token 0) | **2.0240** | `receipts/k8_probe_fixed2048.json` (commit `ca576f41`) |

A three-way tie within ~0.01 nats at single-seed, 14 336-target resolution. What it
localizes: on natural text at probe scale, the store **never fills** — occupancy sits
at ~84 mean live slots whether the physical rank is 128 or 2048 (4 % of the larger),
the policy runs ~93 % CORRECT in the identity-rewrite regime, and the grown rows are
legal, exact-zero, and never read. The binding constraint is **content diversity —
the number of distinct payload directions the data produces — not slot count**, which
is the CMA-ES diagnosis seen from the capacity side. The operator's transformer-parity
state target (2048 addresses) is therefore untestable in *value* in this regime; it
needs the saturation regime (the C3 claim's own probe, §6 below). Caveats verbatim from
the probe protocol: single seed, one transferred LR, probe-scale exposure,
pre-any-2700s-claim.

### 4.3 The growth capability: fires, trains, and costs

The capability question is answered **yes, as capability** ([K8_GROWTH.md](K8_GROWTH.md)
+ `receipts/k8_smoke.json`): given the registered rule (the append-collision statistic
verbatim, 64-token boundaries, geometric 2.0, snap set {64…2048}, cap 2048), wired with
its Lean-proven legality and held to the lane's bit-exact/floor gates (0 flips in
2 162 688 lanes; zero-injection receipts; prefix property; determinism; gradients
within floors **across** growth boundaries, 0.89–1.67×), the model climbs 128→2048 in
every window, the new rows enter exact zeros, the old rows' evolution is bit-untouched,
and the backward survives every rank change — six real training steps, finite
gradients, loss falling. The smoke's honest finding is retained: the registered
trigger fires on an **idle** store (the identity-rewrite regime makes the collision
rate ~1.0 inside the statistic's window), inherited verbatim per the operator's
instruction — growth is armed by write-reuse pressure, not by a full store
([K8_GROWTH.md](K8_GROWTH.md) §5). The utility question is the §4.2 tie: at probe scale
the model has no use for transformer-parity state, at the measured price of the probe
pass plus 16×-state walls (144 s/step vs the fixed-128 arm's 9.5 s at this shape).

### 4.4 The sequentiality dial, end to end

The dial's full curve, one LR, one seed, matched 20 M exposure
([RESULT_CHUNK_SEQUENTIALITY.md](RESULT_CHUNK_SEQUENTIALITY.md),
`receipts/sweep_chunk_c{128,64,32,8,4}.json`): 2.4173 → 2.3287 → 2.2462 → 2.2186 →
**2.1791** nats/byte at chunk 128→4, still falling at chunk 4, with the mechanism
receipted (MERGE collapses 62× — the staleness artifact; reads sharpen monotonically).
The chunk-2 point is **not run, with reason**: the memory law OOMs the card
mid-capture (`receipts/sweep_chunk_c2_oom.json`) — the dial cannot reach per-token by
memory, not by time. K6 then built the limit itself and beat the dial's best row:
**2.0278, −0.1513 beyond chunk 4**, the gradient accelerating into the limit
([K6_SEQUENTIAL_MATRIX.md](K6_SEQUENTIAL_MATRIX.md) §4). At board scale the dial
delivers an order of magnitude less than the probe promised (§4.1) — the honest
cross-scale lesson, stated in both receipts. The K6 arm's own falsified sub-prediction
is retained: MERGE does not recover at the limit, it completes its collapse to
0.0002/token — confirmed as the dial's staleness artifact, not a needed operator at
per-token granularity.

## 5. The expressivity results

### 5.1 The granularity separator: the two-part signature

The operator's hypothesis — with irreversible, state-dependent management ops, a
decision delayed past the information it acts on is permanent loss; therefore
chunk-granular decision *generation* is an expressivity weakening — is now receipted
in both halves
([RESULT_GRANULARITY_SEPARATOR.md](RESULT_GRANULARITY_SEPARATOR.md), predictions
committed before any separator training run, commit `469bfa71`; harvest `d1d62f69`):

* **The learnability side (natural text):** every granularity is solvable and the
  gradient is monotone, accelerating into the per-token limit — 0.390 nats/byte over
  the full dial at probe scale ([K6_SEQUENTIAL_MATRIX.md](K6_SEQUENTIAL_MATRIX.md)).
* **The expressivity side (the mechanical witness):** SEP-AR, 48 cells, chance =
  1/32. Per-token 0.856–0.999 across every cell; C=8 readable probes 0.984–0.989 but
  **blind probes 0.289–0.322**; C=32 and C=128 **at chance (0.028–0.033) across every
  LR and seed** — a capability ceiling, not a learning failure. The P-4 mechanism
  receipt is the sharp form of the thesis: the scan's own recorded read channel puts
  **exactly 0.0 attention weight on rows the occupancy log proves live and holding
  the answer** (`receipts/separator_mechanism.json`) — the row exists, the generated
  schedule cannot see it, at any parameters. The refutation clause (any chunked cell
  at 0.85 on blind probes) was **not triggered**.

Honest edges, kept: P-5 (the SEP-EVICT corroboration) is **partial** — the per-token
evict cells ran (0.93–0.97), the chunked evict cells did not (the lane died at its
ceiling mid-sweep); it is not load-bearing for the headline. The claim is scoped to
*this* algebra's chunked form (entry-pinned reads, entry-pinned free-slot sets), with
the Lean theorem shape stated as the generation-level programme
([RESULT_GRANULARITY_SEPARATOR.md](RESULT_GRANULARITY_SEPARATOR.md) §7) — the channel
lemma is machine-checked now; the no-recovery lemma and the strategy-level depth-≥2
claim remain the formalization's open work.

Together the two halves sign the claim the theory layer could not: decision
granularity is both a quality dial and a semantic axis — chunk-granular generation is
a weaker machine, and the LM-scale loss gap is the learnability shadow of that
weakening.

### 5.2 The depth-substitution finding (in flight)

The key-separation probe's pre-registration
([RESULT_KEY_SEPARATION.md](RESULT_KEY_SEPARATION.md), predictions committed before
any protocol-budget run, commit `63d2b681`) sharpened the operator's
compression-theoretic frame — a store whose keys ARE its contents has compression
ratio 1 and degenerates to identity caching, exactly the 150 M diagnosis — into a
**per-layer** necessity claim with a machine-checked collision lemma (bit-identical
payloads at layer 1; identity rewrite mechanical at every decay; equal mass on
identical rows exact; the derived 0.5 set-retrieval ceiling at depth 1). Its
pre-disclosed calibration finding is the depth-substitution result the sweep will
receipt: at depth 2 the **fused** (key=value) arm escapes the ceiling through
**composite payloads** — the layer-2 binding row carries window-invariant context
components injected by the layer-1 read, so content matching synthesizes
context-dependent addressing one layer deep, at the cost of a full layer of parameters
and serial indirection. The registration case for a dedicated key channel is therefore
the **layer-1 boundary and the per-layer indirection cost**, not a claim that deep
fused nets are context-blind everywhere ([RESULT_KEY_SEPARATION.md](RESULT_KEY_SEPARATION.md)
§5, §11). The necessity sweep itself (48 cells, fused vs separated at depth 1, plus
the key-dim curve and the depth-2 corroboration with composite-payload receipts) is
**running; no results are cited**.

### 5.3 The in-flight lanes, cited as pending

* **SEP-KV (key-separation necessity):** pre-registration committed (`63d2b681`);
  the sweep is in flight on its own lane; no committed results yet. Its open
  registration questions — keys-and-decay, run-boundary semantics, orthonormality vs
  a statistical margin bound, growth interaction, key-dim scaling — are listed in
  [RESULT_KEY_SEPARATION.md](RESULT_KEY_SEPARATION.md) §10.
* **Wheeler-order capacity (SPAN-RUN):** design committed (`4acac6da`,
  [DESIGN_WHEELER_ORDER.md](DESIGN_WHEELER_ORDER.md)); pre-registration committed
  (`abbdd903`, [RESULT_WHEELER_CAPACITY.md](RESULT_WHEELER_CAPACITY.md)): the
  ordered, run-coalescing store's Θ(r × run) capacity-multiplier claim against the
  pointwise arm's derived eviction-onset decay (collapse point S=8), with the
  dependency stated — both arms sit on the SEP-KV separated-key base, and reclassify
  if separation is refuted. The sweep is in flight; no committed results yet.

If either lane commits while this document stands, its RESULT file supersedes the
"pending" here; this synthesis does not block on them.

## 6. The open questions, ranked, with owners

1. **The key-family registration (address vs content).** The Wheeler-frame question:
   what a slot is *addressed by*. The 150 M identity-cache diagnosis, the SEP-KV
   collision lemma and the depth-substitution finding all point at the same fork — a
   dedicated small key channel at the first layer vs composite layers bought with
   depth, and flat content-free keys vs ordered, run-coalescing (Wheeler-style) keys.
   **Owner:** the SEP-KV and Wheeler lanes (running), then the operator's registration
   decision over [RESULT_KEY_SEPARATION.md](RESULT_KEY_SEPARATION.md) §10's list. The
   falsifiers are already committed in both pre-registrations.
2. **C3 saturation — the one regime where the 2048 target is testable in value.** The
   three-way tie (§4.2) localizes the constraint to content diversity; the capacity
   claim needs a stream or horizon that forces the store past 128 live rows. K9 made
   it affordable (~3.3 h per 180-step arm at rank 2048 against the banked ~7.4 h,
   [K9_DISTRIBUTED_WALK.md](K9_DISTRIBUTED_WALK.md) §3). **Owner:** a probe lane,
   once the operator prioritizes it; C3 remains otherwise untested
   ([`../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md`](../x43_fork_resolved/RESULT_150M_MATCHED_ARMS.md)
   §4 for C2/C3 status).
3. **The sequential registration sign-off.**
   [PREREGISTRATION_SEQUENTIAL_MATRIX.md](PREREGISTRATION_SEQUENTIAL_MATRIX.md) is a
   **DRAFT; operator sign-off pending** — its own rule gates any 2700 s claim for the
   per-token arm, whose probe-scale promise (−0.39 nats over the dial) is the
   board-scale question now standing. The tuned-LR confirmation is its named twin.
   **Owner:** the operator.
4. **The 1.3B flagship.** The byte flagship registration remains a DRAFT with explicit
   gates (all three byte nominations plus operator sign-off of the completed scale
   registration; no launch authorization)
   ([`../x43_fork_resolved/PREREGISTRATION_1P3B.md`](../x43_fork_resolved/PREREGISTRATION_1P3B.md)).
   The arc's results feed it as evidence, not as gates: the head-asymmetry reversal at
   Pile scale (transformer the better array-reader,
   [`../x43_fork_resolved/PROGRAM.md`](../x43_fork_resolved/PROGRAM.md) "C14 transformer
   stacking"), the matrix arm's 150 M placement, and the 150 M mini-lattice's
   registered arms ([`../x43_fork_resolved/PREREGISTRATION_150M_LATTICE.md`](../x43_fork_resolved/PREREGISTRATION_150M_LATTICE.md)).
   **Owner:** the operator.

## 7. What this arc does and does not license

Stated plainly, because the arc's own receipts demand it:

* **Proven (Lean):** the fixed-schedule scan laws, the exact rank-1 decomposition, the
  truncation law, zero-injection growth legality, and the non-affinity of
  state-dependent management. Nothing about decision generation.
* **Receipted at board scale (2700 s, single seed):** the C1 null with its stability
  contrast; the tuning (0.076) and dial (0.024) attributions; the machine's 31 k
  tok/s class with bit-equal semantics.
* **Receipted at probe scale (20 M targets / synthetic tasks):** the sequentiality
  gradient and its limit; the growth capability and the capacity tie; the granularity
  separator's two-part signature.
* **Pending:** SEP-KV, Wheeler, the sequential registration's sign-off, C3 saturation,
  the flagship.
* **Not claimed anywhere in this arc:** that the five-op matrix arm beats GDN2 at 150 M
  (it loses by 0.56 nats, attribution above); that the ops' value is refuted (the
  stability contrast and the tape scoping stand; the *matrix-form* C1 ordering is what
  failed); that growth pays (it ties at probe scale); that any expressivity claim is a
  theorem (the separator is an empirical witness for this algebra's chunked form; the
  Lean generation-level programme is open).

The research thread, in one sentence: the theory layer proves the algebra scannable
under fixed schedules and locates the true boundary at decision generation; the
machine arc made the algebra runnable at every rank with growth, honestly paying for
each lesson with a receipt; the measurements found the registered chunk-granular form
losing at the board and degenerate into identity caching; and the expressivity
programme turned that loss into a separation result — chunk-granular generation is a
weaker machine — with the key-family question (what a slot is addressed by) now the
registered next decision.
