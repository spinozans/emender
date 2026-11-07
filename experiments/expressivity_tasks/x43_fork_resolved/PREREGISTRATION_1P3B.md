# X42/X43 — terminal 1.3B comparison registration DRAFT

**DRAFT; operator amendment and sign-off required. No launch authorization.**
Recorded 2026-11-07. The pure mimetic search's nomination is pending (operator
planning estimate November 8–9, not an observed completion date). Both that
committed nomination AND explicit operator approval of the completed registration
are mandatory before any training. A date passing does not release either gate.
This work creates no lease, training, extraction, or corpus-read request.

## Evidence and interpretation

- [Transformer nomination](../x42_pile_scale/receipts/velocity_1p3b_nominations/transformer.json):
  proxy NLL **3.190717468659083**; [GDN2 nomination](../x42_pile_scale/receipts/velocity_1p3b_nominations/gdn2.json):
  **3.1674960454305015**. These are proxy results, not 1.3B training results.
  Each receipt contains its proposed scale architecture, optimizer and provenance;
  preserve those configs verbatim except explicitly approved scale-transfer changes.
- [Velocity protocol](../x42_pile_scale/VELOCITY_PROTOCOL.md) and
  [three-family protocol](../x42_pile_scale/VELOCITY_THREE_FAMILY.md) select under
  synchronized active time, including data/transfer/update. Baseline score is raw
  NLL; pure score includes occupancy/degeneracy terms. Compare raw NLL separately.
- [Corpus attestation](../x42_pile_scale/receipts/velocity_corpus_attestation.json):
  **12,756,052,869 training tokens**, **128,849,018 validation tokens**, GPT-2
  vocabulary; uncopyrighted Pile subset, not the entire original Pile.
  Its manifest binds tokenizer, source revision, document index and split hashes.
  This registration reads that receipt, not the corpus.
- X40's [capacity/compute doctrine and best-practice baselines](../x40_emender_lm/REPORT_PART8.md)
  match total capacity within **0.1%**, including frozen features, and distinguish
  total from trainable parameters. Baselines use BF16 dense/projection compute,
  FP32 master parameters/AdamW; pure retains the certified FP32 store.
  The historical [surface audit](../x40_emender_lm/receipts/part9/decision.md)
  gives **1.5854x** pure/transformer step cost at T512/B64, not a universal pure
  speedup. X42's [lineage audit](../x42_pile_scale/receipts/lineage_suspects/LINEAGE.md)
  retains failed overall surface statuses. These are historical references.
- [C14 first-pass ladder](RESULT_C14_FIRST_PASS.md): final-test **214,097,022
  real byte targets**, byte vocabulary **256**. Dense transformer **1.404428829**,
  passive consulted **1.266063829**, consulted+soft **1.263566972** nats/byte;
  passive Emender **1.509260010**, consulted+soft **1.505908651**. The best measured
  system is transformer+XSA+soft. Literal walks and array-alone have infinite total
  loss; conditional array loss **0.663434002** is not a total-LM lower bound.
- [Protocol-shape receipt](RESULT_PROTOCOL_SHAPE_GAP.md): disposable
  **1,301,052,902**-parameter byte fixture, B16, S257 versus S48: unbounded/bounded
  full-step cost **1.474241x at T128**, **1.660072x at T256**. These qualify a
  feasible state-class completion, not trained token-model throughput. Physical
  capacity and observed occupancy differ; the oversized S2049 control is separate.
- [Matched-exposure resolver](RESULT_C14_B_MATCHED.md): unbounded minus bounded
  **−0.019429161** nats/byte, but **16,094.703 / 2700.032 active seconds** at equal
  exposure; passive remains better in absolute NLL. Lifecycle advantage over
  append-only at fixed time is **0.661320101** nats/byte. The
  [second-seed sign](RESULT_C14_B_SEED3402.md) is **−0.018504590**, with both arms
  at **35,419 updates / 71,677,042 targets**. Two seeds support a sign, not a law.
- [Transfer evaluation](RESULT_P100_OUT_OF_INDEX.md) and
  [paired decomposition](receipts/p100_heldout/decomposition.json): consulted
  Emender gain **1.073160034**, transformer **0.370784759** nats/byte outside the
  index; in-region gains **0.8006653652251972 / 0.19683378857940004**. Out-of-index
  gains exceed in-region. Index and documents change together (Wikipedia versus
  Pile-head100); this does not identify a pure coverage effect.

## Six-arm matrix — DRAFT DECISION: approve or amend each arm

| Arm | Configuration and scientific role | Required paired reporting |
|---|---|---|
| 1 Transformer | Nominated 3.1907 policy; unbounded per-window KV baseline | Raw NLL, actual exposure, KV memory versus length |
| 2 GDN2 | Nominated 3.1675 policy; fixed-state baseline | Raw NLL, actual exposure, fixed state size |
| 3 Pure Emender, bounded | Pending pure nomination; certified replay architecture | Nomination score versus raw NLL, budgets/horizons, certification scope |
| 4 Pure Emender, unbounded | Same searched pure configuration as arm 3, cap released | Matched-time primary; matched-compute and matched-exposure secondary |
| 5 Transformer + XSA | Arm 1 backbone; passive causal consultation and C14 all-feature dropout | Device on/off; calibration-only soft fusion; zero-feature control |
| 6 Emender + XSA, unbounded | Searched pure backbone; enabled store door, lifecycle ops active, cap nonbinding; XSA-derived memories can be amended | On/off, operation/ownership traces, matched-compute and exposure secondary |

Arm 6 uses the mechanism in [enabled store-door receipt](RESULT_C14_B1.md),
not merely passive features relabeled as memory operations. Porting its byte
mechanism to the searched token backbone remains unqualified. Arm 5's flagship
score includes soft fusion: report consultation alone and consultation+soft
separately. Do not silently credit its stacked score to passive consultation alone.
**DRAFT DECISION:** retain matched zero-feature projections in dense controls and
match total parameters within the X40 tolerance; disclose trainable/frozen counts,
embedding/projection overhead and any unavoidable mismatch before approval.

## Units and document boundaries — DRAFT DECISION / unresolved approval gate

Propose token-level modeling across all six arms on the attested velocity corpus.
Report full-vocabulary token NLL, summed NLL / actual raw-text byte count, actual
tokens, real bytes, windows and optimizer steps. A byte-derived feature distribution
is not a normalized GPT-2 token distribution. No next-byte probability may be used
as a token probability, and forced bytes cannot force tokens. Freeze a causal
byte-to-token boundary map and either validate normalized full-token fusion or
mark soft fusion unavailable; that would amend arm 5's flagship endpoint.

**Standing instruction conflict:** AGENTS.md requires C14 and subsequent training
to use natural raw bytes/vocabulary 256, never token IDs as raw text. Therefore
the proposed token-level terminal protocol requires an explicit operator amendment
of that rule before launch. Approval of a draft title is not that amendment.
Alternatively the operator can choose all-byte arms, but proxy policies then need
fresh transfer qualification and cannot be described as already matched token
nominations. No token/byte compromise is implemented by this document.

In either approved unit system, every input, shifted target, consultation prefix
and held-out sequence stays inside one document; truncate/mask tails and reset all
model state at each window. Never feed a target or future suffix into consultation.
Reconcile raw text to documents.jsonl hashes using the checked
[c14_documents.py](c14_documents.py) windowing/split discipline. Do not reuse packed
velocity windows across document boundaries. Re-windowing changes proxy exposure;
record it. Calibration and final test split by document, frozen before fitting.

## Budget and length columns — DRAFT DECISIONS

Propose **24 synchronized active training hours per arm per seed**, seeds **3401
and 3402** (seed precedent: the twin receipt above). These hours are a proposed
allocation, not a measured completion forecast. Include data assembly/transfer,
forward/backward/replay, clipping, optimizer and last completed step; report
overshoot. Separately retain elapsed wall time including setup, checkpointing,
extraction and evaluation. Use the same hardware envelope, document order and
initialization convention; unequal bytes/steps at matched time are results.

Primary column: loss after equal active wall-clock budget. Secondary for arms 4/6:
loss at the common cumulative measured compute envelope with their bounded/pure
reference, with compute accounting method frozen before launch. GPU-hours alone
are not FLOPs. Report measured FLOPs when available, otherwise explicitly label
card-time and leave matched-FLOP comparison unavailable. Also retain common-prefix
matched-exposure checkpoints; extra continuation costs additional card-hours and
must be separately approved. No monotone learning extrapolation substitutes for
an observed checkpoint.

**DRAFT DECISION:** token train context T2048 and effective B64 follow nomination
receipts; choose feasible microbatch/accumulation after scale-transfer preflight.
Eval reset sweeps T512/1024/2048/4096 follow velocity length accounting; include
T128/256 byte-window diagnostic columns to connect to C14/protocol-shape evidence.
These are different unit columns, not equivalent context lengths. Report raw-byte
context spans for token windows. Preflight every length for all arms, retaining
OOM/errors as failures. Cap released means no capacity eviction at any registered
length: bind physical allocation, per-head maximum reachable writes including
array writes, observed live peaks and R-CAP counts. S257 from the byte fixture
cannot be copied blindly to T2048 token training. No cross-window carried state.

## Array and extraction operations — DRAFT DECISIONS

Propose the immutable **Wikipedia 100GB spine** as primary, with its existing
[coverage/query receipt](RESULT_C14_PHASE1.md) and repaired C14 serving bindings.
The [C14 registration](PREREGISTRATION_C14.md) receipts support at byte lengths
16/64/256/1024 of **36.5900% / 2.3600% / 0.5900% / 0.0900%**; this is sampled
training-interior coverage, not a held-out content-component estimate. Freeze
executable/index/source hashes and fresh sampled independent branch checks before
use. Registered deployment follow-up: swap the same frozen heads to Pile-head100,
with no weight or gate refitting, separate inclusion/exclusion and repeated-content
checks as in the transfer receipt. The follow-up has a separate budget; it cannot
replace the primary Wikipedia endpoint post hoc.

The [union-service precedent](PROGRAM.md#c14-managed-union-extraction-2026-11-03-running-science-pending)
banks **5,884.18 raw targets/s**, **1,406,579,817 union positions**, about **66.4
host hours**. The [first-hours receipt](receipts/c14_union_first_hours/REPORT.md)
observed **5,612.91/s** clean and **4,814.02/s** including contention; host has
**256 logical CPUs**, recent cache density **97.91 bytes/target**. Those are host
throughputs, not per-core rates. Actual summed CPU time was not established by
these numbers: host-days and logical-core-day reservation ceilings are distinct.

Let R be actual queried raw-byte positions in the union of the two consultation
training prefixes plus evaluation. Extract common prefixes once and share immutable
caches across seeds/arms only when boundary and feature identities match. Cost
scenario: R/5,884.18/86,400 host-days, or R/4,814.02/86,400 under observed contention;
logical-core-day ceiling is host-days times 256, not measured consumed CPU-days.
**DRAFT DECISION:** reserve that full host envelope and require measured user+system
CPU seconds in the extraction completion receipt. No CPU-day precision is invented.

For the full attested training stream, **DRAFT DECISION planning assumption** of
4 raw bytes/token gives R=51,024,211,476 before evaluation: **100.36–122.67 host-days**,
up to **25,693–31,405 logical-core-days** at full-host reservation, and about
**5.00 TB** cache at the observed density (derived arithmetic, not corpus measurement).
At one queried byte per token the floor scenario is already **25.09–30.67 host-days**.
Actual bytes/token, token-boundary query reuse, new feature geometry and content
can change all estimates; no full-stream cache is approved. A 24-hour training
prefix may be shorter, but cannot be costed until throughput and byte spans are
measured. Validation/eval queries add cost. The historical **150GB reserve floor**
(first-hours receipt) is not permission to allocate terabytes. Freeze a measured
R, storage reservation, CPU-day ceiling and extraction stop line before launch.
Feature extraction is prepaid in the training-time column and fully charged in
the end-to-end column. Pile-head100 follow-up extraction is additional.

## Cards, phase order and precision — DRAFT DECISIONS

Propose physical card **4**, serially, for qualification and then every arm, with
no preemption; leave search card **1** untouched (protocol-shape precedent).
Use card UUID/model and measured memory envelope, not index alone, in final approval.
If that card is unavailable, wait or seek an amended allocation. No lease now.
Proposed training reservation: **24 card-hours per arm/seed**, **48 per arm over
both seeds**, **288 total** for the six-arm matrix. Propose an additional **24
card-hours total** for qualification/evaluation, thus **312 card-hours** before
any matched-exposure extensions or follow-up. This is a spending proposal, not
an assurance all six arms fit or finish; report failure if the ceiling is exhausted.
CPU extraction above is additional and may dominate calendar time.

Phase order: (a) receive pure nomination; (b) operator resolves units, precision,
configs, budget and signs; (c) approve a concrete hash-bound launch amendment;
(d) serial scale-transfer, arithmetic/gradient/operation and memory qualification;
(e) determine exposure envelope and extract/verify immutable union caches;
(f) train arms in a frozen order with paired seed/config receipts; (g) freeze
calibration gates before final scoring, then length sweeps and secondary columns;
(h) operator-approved deployment swap. No scientific result may select the order,
checkpoint or precision after seeing held-out outcomes. Qualification failure blocks
training and requires an amended receipt rather than an invented score.

- [ ] **Operator checkbox — BF16 store decision:** retain certified FP32 store
  (proposed default), or explicitly approve a BF16 store implementation and parity,
  first-order adjoint, finite-loss and matched-curve gates before use.
  Existing baseline BF16 compute remains best practice. The protocol-shape receipt's
  conditional store-BF16 full-step ratios **1.405064x / 1.406169x** versus measured
  **1.474241x / 1.660072x** are insurance scenarios, not a delivered speedup or
  residency proof. Do not halve reported total memory or assume affordability.
- [ ] Pure nomination committed and scale config/source hashes bound.
- [ ] Units rule explicitly resolved; byte/token feature bridge certified.
- [ ] Exact per-arm configs, physical slots, lengths, dropout/fusion and capacity approved.
- [ ] Active/elapsed/compute budgets, R, storage, CPU-days and card-hours approved.
- [ ] **Operator signs final launch amendment:** name/date/commit pending.

Future long jobs must use the standing detached broker-wrapper playbook with
PPID=1, owner/starttime, heartbeat renewal and advancing steps proved; recover
only from a complete atomic checkpoint with hash and unsaved-span receipt. This
registration changes no training launcher, DiLoCo behavior or scheduler policy.

## Predictions and reporting — DRAFT DECISIONS

Bind directional predictions rather than borrowed absolute losses:

1. Asymmetry replicates: arm 5 consulted+soft beats arm 6 in absolute held-out
   NLL at matched time; arm 5 improves over arm 1. Report consultation-alone too.
2. Lifecycle ops retain value at scale: arm 6 improves over a separately qualified
   append-only intervention at common exposure. That diagnostic is **not a seventh
   full training arm**; its additional card budget/config must be signed before
   running. Without it, operation telemetry is evidence of use, not causal value.
3. Unbounded state improves loss per real training byte at common exposure
   (arm 4 versus 3, and store-door diagnostics), while consuming more time/memory.
   Predict a negative matched-exposure sign consistent with both twin seeds;
   make no universal fixed-time superiority prediction. The fixture cost ratio
   qualifies feasibility only; measured scale costs can refute affordability.
4. Frozen-index swap retains positive consultation gain; do not demand the exact
   small-model transfer magnitude or infer pure coverage causality.

All outcomes, including reversals, infinities, failed precision checks, OOM,
unsupported token fusion and missing secondary columns, remain results. Report
both seeds, raw NLL sums/denominators and paired deltas; two seeds are not a
significance test. Include supported/zero-target/empty/fork strata and reset versus
interior loss, device-off harm, actual bytes/steps, operations, peak state/memory,
active/elapsed time, extraction CPU-days, cache bytes and card-hours per arm.
Never discard impossible targets or reinterpret the conditional array as a bound.

Historical X40-era 1.3B arms are context only: different corpus, optimizer/kernel,
precision, lengths and exposure prohibit calling them matched controls. This
DRAFT's numerical priors are evidence for decisions, not terminal acceptance.
