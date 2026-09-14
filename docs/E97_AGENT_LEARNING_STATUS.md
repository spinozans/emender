# E97 agent learning: current status and decision reset

Status after the grounded-expansion training and evaluation. This is a
programme-level interpretation and priority decision, not a replacement for any
frozen experimental report, failed gate or source authority. The complete
[numerical exploration index](validation/e97-numerical-exploration-index.md)
records the sequence, evidence and limits.

The operator subsequently authorized a return to execution. The new
[grounded expansion tranche](validation/e97-grounded-expansion-v1.md) freezes
2,048 paired authored demonstrations,18 deduplicated real correction/success
records,32 new evaluation cases and a separate32-update BF16-SR SFT budget.
The first data attempt hit its30-minute limit after1,333 verified records; its
artifacts and clean teardown are retained. The separately authorized completion
pass finished the remaining715 in1,000s, preserving the audited prefix. The
complete derivative has2,583 records and926,091 supervised targets. Full audit and
packing passed after fixing and retaining an auditor replay-order failure. The
frozen schedule consumes every record, totaling1,588,936 target exposures.
The new32-update BF16-SR run completed with exact schedule/clock matching and a
finite complete checkpoint (`17aa2672…`); that budget is now closed. Audited
before/after evaluation shows **fresh same-format3/16→16/16**, but old regression
cases7/16→6/16 and structural-transfer0/16→1/16. Both x/y retention gates passed.
**The overall gate failed; there is no promotion.** All192 episodes were regraded
against actual reader snapshots, with192 agent and192 reader cleanups confirmed.

## Goal and actual capability

Build a conversational, reasoning, observation-dependent tool agent, measured by
**autonomous completion of fresh tasks**, with conversation/tool retention—not by
loss alone, valid first calls, teacher success or numerical agreement.

| Work completed | What it establishes | What it does not establish |
|---|---|---|
|880-update native/conversation/retention SFT|166,885,738 supervised target exposures, including50,069,457 native targets; better likelihood/retention panels|Reliable execution: subsequent autonomous diagnostics failed|
|1,024 executor-verified authored grounding trajectories; separate32-update correction|First requested calls0/16→16/16; autonomous completions0/16→7/16|General agency; fresh-value gate3/8 missed4/8; editing/recovery still failed|
|32 stochastic training-only rollouts on16 new tasks|16 autonomous successes and16 same-state verified teacher repairs; collection/correction pipeline works|A new evaluation score or model improvement: zero updates occurred|
|Grounded expansion:2,048 authored examples,18 deduplicated canary records and a new32-update SFT tranche|Fresh same-format completion3/16→16/16, including4/4 edits and4/4 recovery; both x/y retention passed|Generalization/acceptance: old7/16→6/16 and structural-transfer0/16→1/16; joint gate failed|
|Faithful native source/runtime pipeline|Actual observations/errors, full trajectories and routing/masks can be preserved|That more tokens alone will solve remaining behavior or that every candidate is admitted|

The latest measured gain is **16/16 fresh same-format autonomous completions**,
not a replacement score for the older panel, which fell to6/16. The model/trainer
can learn useful behavior, but changed representations still largely fail. It is
not yet a reliably generalizing agent. The data problem is **coverage, fidelity,
curriculum and feedback**, not
simply a shortage of raw token volume. The large mixture already contained4,216
unique native trajectories and106,637 unique conversation records.

Sources: [closed880 programme](validation/e97-native-training-50m-agent-v1.md),
[grounding correction](validation/e97-grounding-correction-v1.md),
[on-policy canary](validation/e97-native-onpolicy-canary-v1.md),
[grounded expansion](validation/e97-grounded-expansion-v1.md).

## Numerical conclusion: remove the blanket SFT veto

**Cross-shape bitwise actor/trainer agreement is not a general prerequisite for
ordinary mixed-precision SFT or inference.** A failed diagnostic agreement gate
must not be interpreted as proof that the model cannot learn, BF16 has erased its
memory, or data development must stop.

We established, with actual operands and exact native-result binding:

1. An earlier recurrence discrepancy depended on checkpoint-workspace selection;
   uniform FP32 workspace repaired that measured dependence.
2. A subsequent early-prompt discrepancy starts at QKV projection output, before
   recurrence. At the same numeric input-row/weight addresses and strides, changing
   matrix height reproduces different **pre-store FP32** results exactly. BF16
   conversion faithfully stores each result.
3. Selected nearly cancelling dots amplify the relative significance of tiny
   absolute arithmetic errors. Exact rational checks of the captured BF16 operands
   corroborate the selected FP64 references. This does not prove semantic memory
   loss, vanishing gradients, or the cause of every task failure.
4. One fixed-row IEEE FP32 kernel removes the measured height/row-placement
   dependence, but isolated QKV GEMMs cost1.92× at one row and3.77× at512 rows.
   It is a **non-dispatched reference/repair candidate**, not a chosen production
   policy. No model integration or backward qualification has occurred.

This is sufficient to understand the observed mechanism at the operational level
and stop treating ordinary floating-point variation as a blanket training veto.
It is **not100% universal numerical qualification**: backend/instruction details,
long-context behavior of new policies, full4B backward and restart contracts have
not all been established. No current result demonstrates catastrophic BF16
information loss; absence of such evidence is not a universal guarantee.

Preserve all original verdicts: full57 current-policy forward agreement passed
under v2; the separate120-position prerequisite failed; no64K forwards ran in that
attempt. Historical recorded-probability binding and earlier exact-continuation
failure remain failed. Do not rewrite scores, remove failed prompt probes, loosen
those frozen limits, or describe either failed experiment as passed.

Evidence: [workspace repair](validation/e97-uniform-workspace-v2.md),
[early-prompt trace](validation/e97-early-prompt-trace-v1.md),
[actual Linear replay](validation/e97-linear-operands-v1.md),
[fixed-row candidate](validation/e97-fixed-linear-micro-v1.md),
[unrun packed stage](validation/e97-packed-forward-v1.md).

## Production baseline and real safety requirements

Use the **previously exercised BF16-SR/checkpointed-CE training recipe** as the
starting point for the next separately frozen SFT plan, rather than automatically
adopting the newest experimental precision policy. The completed training and
behavioral improvement are evidence for that baseline—not evidence that every
possible BF16 implementation is safe. This is a custom recurrent/Schedule-Free
stack, not just an unmodified standard Transformer recipe.

Do not spend a2–4× arithmetic premium merely to obtain bitwise equality where the
learning contract does not require it. Retain the fixed-row kernel for diagnosis
and possible selective use; any later production adoption must earn its cost in
end-to-end correctness, performance and behavioral measurements.

The following remain non-negotiable:

- **Weight identity:** use the intended Schedule-Free x/y representation, exact
  live-y restoration and explicit load mode; preserve slots/clocks and atomic
  checkpoint/latest publication. A fresh optimizer stage and a resumed stage are
  different experiments. Failed exact restart evidence is not silently waived.
- **Updates:** BF16 small updates really can round away. Preserve the exercised
  stochastic-store optimizer implementation and finite-gradient/update checks;
  do not casually replace it with ordinary BF16 stores or a hidden FP32 master.
- **Objective and packing:** supervise the intended assistant spans, preserve
  authentic observations and failed actions as appropriate context, reset every
  recurrent layer between independent records—not between turns—and exclude
  cross-record targets and invalid padding. Keep checkpointing distinct from TBPTT.
- **Data identity and leakage:** immutable source/serialization/mask audits,
  explicit internal-use authority, held-out task separation, no protected-data
  training or unsupported licence/admission claims. Earlier conversion defects
  were identified and rebuilt; that does not erase their historical effects.
- **Outcomes:** independent executor/oracle grading, no teacher-to-autonomous
  credit transfer, fixed fresh-task and conversation/tool-retention evaluation.
- **Execution:** bounded, owned sandboxes and GPU leases, no overlapping jobs,
  finite state/gradients/memory, source/checkpoint fingerprints and cleanup.

## Actual next bottleneck and work queue

Prioritize the **verified data → SFT/DAgger → fresh execution feedback loop**.
The current evidence points to specific deficits: exact observed-value fidelity,
source-versus-destination handling, edits with read-back verification, recovery
from genuine failed actions, and truthful completion. It does not prove that data
quantity is the sole cause or that another broad epoch is sufficient.

1. Inventory those failure modes and expand executor-verifiable trajectories and
   same-state corrections, with intermediate difficulty and varied instructions,
   schemas, paths and multi-step compositions. Use observation counterfactuals to
   test dependency on actual content rather than prompt/template memorization.
2. Keep failed accepted model actions unsupervised in corrective context; supervise
   only verified teacher suffixes. Verify successful self-trajectories separately.
   Maintain source preservation, real errors and native tool semantics.
3. Freeze fresh evaluation before training. Separate same-family fresh values
   from genuinely held-out task families/repositories. Measure the distinction.
4. Prepare one small, separately authorized learning tranche from the known
   checkpoint/optimizer baseline, with exact unique/repeated target accounting,
   autonomous completion and retention gates. Dataset preparation need not wait
   for experimental fixed-kernel or cross-layout parity work.
5. Use SFT/DAgger before outcome RL for this material. The collected batch has
   **zero mixed-reward task groups**, hence zero group-centered advantages; better
   numerical agreement cannot create a useful group-relative learning signal.
   RL also needs its own faithful probability bookkeeping and optimizer/DDP
   qualification. Those are not blanket dependencies of supervised correction.

Existing candidates:16 successful trajectories with2,718 supervised targets and
16 verified teacher suffixes with3,012 targets. They are useful seeds, not a
sufficiently broad curriculum or automatically admitted training data. Deduplication
finds18 distinct records (10 repairs,8 successes), with3,447 distinct targets.
Original candidate eligibility and the production registry remain unchanged;
internal use is separately bounded by the new tranche authority.

## Authorization and reporting

The880-update programme, original32-update correction and new32-update grounded
expansion are all **closed**. No further training, data admission, rollout expansion
or checkpoint promotion is authorized by this status document. The grounded
expansion's failed overall gate is retained despite its same-format learning gain.
A follow-up requires a separate bounded plan and fresh frozen evaluation.

Continue routine bounded diagnosis and preparation without repeated operator
handoffs. Escalate new learning budgets, consequential trade-offs or authorization
boundaries—not every numerical result. Report progress primarily as data coverage,
verified examples, autonomous fresh-task outcomes and retained capabilities.
Keep numerical qualification in its proper supporting role.
