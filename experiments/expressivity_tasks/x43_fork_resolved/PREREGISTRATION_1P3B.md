# X42/X43 — terminal 1.3B comparison registration DRAFT

**DRAFT; operator amendment and sign-off required. No launch authorization.**
Recorded 2026-11-07. The pure mimetic search's nomination is pending (operator
planning estimate November 8–9, not an observed completion date). All three committed BYTE nominations AND explicit operator approval of the completed
scale registration are mandatory before flagship training. Byte search launch is
separately authorized by the operator work order after protocol commitment. A date passing does not release either gate.
This scale registration itself creates no 1.3B training or extraction request.

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

## Two boards — operator amendment 2026-11-07: do byte level

The BYTE flagship is the primary 1.3B experiment. The TOKEN velocity board
completes separately as a secondary comparison; its nominations stay on that
board. Token nominations do not transfer as tuned byte baselines. Approximate
transfer violates the matched-best-practice standard: fresh byte-level CMA-ES
searches at the 150M class for transformer, GDN2 and bounded pure Emender are the
honest route. Their committed byte nominations gate the flagship scale configs.
The prior token proposal and token-fusion qualification gate are superseded.
This amendment authorizes registration and search preparation, not 1.3B training.

| BYTE flagship arm | Configuration | Reporting |
|---|---|---|
| byte-transformer | Byte CMA-ES transformer winner, scaled and qualified | nats/byte, actual bytes, KV memory |
| byte-GDN2 | Byte CMA-ES GDN2 winner, scaled and qualified | nats/byte, actual bytes, fixed state |
| byte-pure-bounded | Byte CMA-ES pure winner, certified FP32 store | raw nats/byte and search penalty components |
| byte-pure-unbounded | Same pure winner, cap released and proven nonbinding | matched time primary; matched compute/exposure secondary |
| byte-transformer+XSA | Byte transformer backbone, pile-head100 spine, C14 dropout | consultation alone and calibration-only soft fusion; on/off |
| byte-emender-unbounded+XSA | Pure winner with nonbinding cap and enabled store door, pile-head100 spine | ownership/operation traces; on/off; matched compute/exposure |

Arm 6 follows RESULT_C14_B1.md: array-derived memories can be amended by
lifecycle operations. Passive features alone do not qualify this mechanism.
Unbounded means bounded with the cap nonbinding; no independent unbounded search.
Total capacity including frozen geometry must match within 0.1%; disclose total,
trainable and frozen counts and embedding/projection overhead before scale launch.

The TOKEN secondary board retains the transformer/GDN2 nomination receipts linked
above verbatim and receives the pure velocity nomination when it arrives (planning
November 8–9, not an observed completion). It reports token NLL, actual tokens and
raw-byte denominators where attested. It cannot nominate BYTE flagship baselines.
Existing token search services and results remain separate. Any new token training
requires its own explicit exception to the standing C14+ byte constraint.

## Units, document boundaries and matched budgets

All six flagship arms use natural raw text bytes, vocabulary 256, scored as summed
held-out NLL / actual real byte targets (nats/byte). Every input, shifted target,
consultation context and evaluation sequence stays inside one document. Reset
state per window, truncate tails or mask all padding; split calibration/final-test
by document. Hash-match documents.jsonl via the checked c14_documents.py discipline.
Use pile.txt read-only by offset. **ALL SIX arms use the same registered source-order
stream of whole documents fully beyond byte 100,000,000,000**, including the
unconsulted controls. All evaluation documents are also fully beyond that mark
and document-disjoint from training; calibration and final-test are document-disjoint
from one another. Freeze document identities, source/index hashes, split manifests
and byte windows before launch; reject content-hash duplicates across splits.
The old C14 binding (7,831,211 documents / 43,494,394,389 selected bytes) lies wholly
below 59,187,954,937 and cannot serve as the new stream manifest. The
[geometry receipt](RESULT_SPINE_DECISION.md) qualifies a fresh beyond-boundary
sample and records this limitation. No corpus materialization or token IDs.

This span-disjoint paired design satisfies the C11 copy-law/train-disjoint doctrine
by geometry: the array contains exactly [0,100,000,000,000), while every training
and evaluation document starts beyond it. Repeated substrings may still occur;
span exclusion is not a claim of global substring uniqueness. Consulted and
unconsulted arms share stream, held-out windows and backbone nominations; on/off
interventions retain frozen weights/gates. This fixes E3's joint index/document
change for the flagship consultation comparison. It does not retroactively identify
E3's pure coverage effect or identify one without a matched inclusion intervention.
The dense byte searches retain their original below-boundary stream; their
hyperparameters require transfer to this new stream as well as to 1.3B capacity.

Proposed flagship allocation remains 24 synchronized active hours per arm/seed,
seeds 3401 and 3402. Include data assembly/transfer, forward/backward/replay,
clipping and optimizer; record complete-step overshoot. Separately report elapsed
time, extraction and evaluation costs, actual bytes/windows/steps and byte rates.
Unequal byte exposure at equal time is a result. Same hardware envelope and
source document order apply across arms.

Matched-compute secondary columns apply to every arm: measured FLOPs when
available, otherwise explicitly labeled card-time with matched-FLOP loss
unavailable. GPU-hours are not FLOPs. Retain common-prefix exposure checkpoints;
additional continuation requires a separately approved budget. No extrapolated
loss substitutes for an observed checkpoint.

Proposed flagship train context is T2048 bytes, effective B64; qualification must
bind feasible accumulation and all exact scale shapes. Eval resets at byte lengths
128/256/512/1024/2048/4096. Register physical allocation, maximum reachable writes
including XSA writes, live peaks and R-CAP counts: released cap must never bind.
No cross-window or cross-document state. Search protocol is separately registered
in ../x42_pile_scale/BYTE_VELOCITY_PROTOCOL.md.

## Array and extraction operations — DRAFT DECISIONS

**Operator spine decision, 2026-11-07: immutable pile-head100 is the base.**
Serving path `/mnt/nvme2n1/erikg/P100c/pile-head100.sxi2`, exactly the first
100,000,000,000 pile.txt bytes, container 63,163,489,696 bytes (63.16 GB),
χ=24.19G and χ/n=0.242; build certificate sxgc ee2561c. Our
[17/17 independent full-region brute-force query verification](RESULT_P100_QUERY_VERIFICATION.md)
and [E3 transfer/decomposition](RESULT_P100_OUT_OF_INDEX.md) motivate this choice:
out-of-index gains exceed in-region gains, survival **1.34–1.99×**. E3 remains
honestly not causally identified because corpus/index and documents changed together.
The paired protocol above removes that confound for its consultation contrast.

**Wikipedia remains the registered cross-corpus generalization follow-up**:
swap the same frozen heads from Pile to Wikipedia with no weight/gate refitting,
same held-out beyond-100GB windows, separate inclusion/exclusion and repeated-content
checks. Its [historical coverage receipt](RESULT_C14_PHASE1.md) is historical,
not Pile coverage. Separate follow-up budget; it cannot replace the primary endpoint.
Freeze executable/index/source hashes and verify bindings before either use.

[Readiness and geometry receipt](RESULT_SPINE_DECISION.md): our verification,
span geometry and exercised serving path are receipted for training use. The
sxgc-side `RESULTS_PILE_100GB.md` search certificate (sampled stream checks,
closure-0, T3 and brute-force query positions **on Pile data**) is a standing
external gate for deployment claims. Our sampled 17/17 branch/EOF verification
does not discharge that exhaustive upstream gate. Exact launch manifests,
scale qualification, measured extraction envelope and final approval remain open.

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

Historical token-based extraction cost scenario (not the byte allocation):
R=51,024,211,476 queried positions before evaluation: **100.36–122.67 host-days**,
up to **25,693–31,405 logical-core-days** at full-host reservation, and about
**5.00 TB** cache at the observed density (derived arithmetic, not corpus measurement).
At one queried byte per token the floor scenario is already **25.09–30.67 host-days**.
Actual bytes/token, token-boundary query reuse, new feature geometry and content
can change all estimates; no full-stream cache is approved. A 24-hour training
prefix may be shorter, but cannot be costed until throughput and byte spans are
measured. Validation/eval queries add cost. The historical **150GB reserve floor**
(first-hours receipt) is not permission to allocate terabytes. Freeze a measured
R, storage reservation, CPU-day ceiling and extraction stop line before launch.
Both consultation arms now query pile-head100 through the verified serving
path, retaining causal within-document contexts and shared immutable prefix caches.
The informal ~8–16 CPU-day stream-scoped estimate is an unqualified planning prior,
not measured consumption: index composition, support/suffix reuse, longer T2048
resets and the new stream can change query counts, throughput and cache density.
Neither χ/n nor container-size ratios justify scaling that estimate. Benchmark the
new geometry and bind R plus measured user+system seconds before extraction.
Feature extraction is prepaid in the training-time column and fully charged in
the end-to-end column. Wikipedia follow-up extraction is additional.

## Cards, phase order and precision — DRAFT DECISIONS

Byte search cards are **2/3/4**, one family each; card **1** stays untouched,
and **0/5/6** are reserved for fork/eval. After searches, propose card **4**
serially for flagship qualification and every arm, with
no preemption; leave search card **1** untouched (protocol-shape precedent).
Use card UUID/model and measured memory envelope, not index alone, in final approval.
If that card is unavailable, wait or seek an amended allocation. No lease now.
Proposed training reservation: **24 card-hours per arm/seed**, **48 per arm over
both seeds**, **288 total** for the six-arm matrix. Propose an additional **24
card-hours total** for qualification/evaluation, thus **312 card-hours** before
any matched-exposure extensions or follow-up. This is a spending proposal, not
an assurance all six arms fit or finish; report failure if the ceiling is exhausted.
CPU extraction above is additional and may dominate calendar time.

Phase order: (a) receive all three byte nominations; (b) operator resolves units, precision,
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
- [ ] All three byte nominations committed and scale config/source hashes bound.
- [x] Operator selected pile-head100 base, beyond-100GB paired stream and Wikipedia follow-up.
- [x] Operator selected BYTE flagship; token board is secondary and separate.
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

## Operator work order — flagship stream and CPU extraction, 2026-11-07

The operator separately authorized nomination-independent stream registration and
CPU extraction now. [Frozen registration](receipts/flagship_stream/registration.json)
and [stopping rule](receipts/flagship_stream/STOPPING_RULE.md) bind all six arms to
the same beyond-100GB source-order training prefix, document-disjoint calibration
and test, and primary T2048 windows. This authorization supersedes the extraction
phase-order dependency above for this finite stream only. Training nomination,
qualification, config and approval gates remain pending. The index/executable,
manifest, ordering, split and source identities are frozen in that registration.
The 1,499,511,583-position extraction has a 200GB storage reservation with a 150GB
free floor; historical end-to-end rates project 2.95–3.61 host days. These are
planning estimates, not measurements of this new stream. The managed service is
`emender-flagship-union.service`; heartbeat paths and failure policy are in the
stopping rule. Alternative evaluation reset lengths require distinct extraction.
