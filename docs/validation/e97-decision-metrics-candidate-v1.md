# Decision-balanced loss reducer: candidate v1

Date: 2026-09-10 UTC. Implemented locally; **not wired into training or a frozen LR
sweep**. This is an implementation candidate for the approved learning/validation
addendum, not approval of a new protocol, source, or annotation policy.

Files:

- `ndm/e97_decision_metrics.py`
- `tests/test_e97_decision_metrics.py`

CPU validation: **13 passed** using the local approved Python environment, with
CUDA hidden. No training data or checkpoint was read for these tests.

## Proposed scoring contract

For each of opening, choice, arguments, and termination:

1. Average target NLL within each annotated field span.
2. Average field values within each applicable assistant turn.
3. Average applicable turns within each trajectory.
4. Average trajectories with that component.

Then `D = (opening + choice + arguments + termination) / 4`.

Reasoning and auxiliary text are measured separately and never enter D. All
supervised tokens must nevertheless be classified exactly once. Auxiliary fields
may identify answer text or remaining syntax; their semantic annotation is the
adapter's responsibility, not something inferred by this reducer.

Long reasoning cannot overwhelm decision errors, long arguments cannot dominate
short fields, and long trajectories do not automatically outweigh short ones.
Every component reports its token, field, applicable-turn, absent-turn, and
trajectory counts.

## Missing components and alignment

Every turn declares applicability for all six categories. An absent component is
**not assigned zero loss**. Every decision component must occur somewhere in the
panel, or D is undefined and scoring fails. Missing generated actions are not
teacher-forced missing-component cases: they remain failures on the separate
free-generation/execution axes.

The reducer rejects overlapping, unclassified, unsupervised, nonfinite, negative,
or ambiguously typed annotations/scores. Each field has one uniquely named
contiguous span. Duplicate trajectory/turn identities are rejected. A complete
stream's BPE bytes must exactly reproduce the serialization; a field boundary
inside a BPE token fails closed rather than silently assigning the token twice.
The adapter must explicitly perform the input-to-prediction-target index shift.

A score-independent annotation SHA binds causal probe identities, applicability,
masks, and spans, allowing a caller to reject comparisons across changed panels.
The causal probe digest must cover tokenizer/input/reset/valid identity, not just
displayed text. The reducer cannot establish the authenticity of a supplied
probe digest; building and verifying that authority is a separate gate.

## Tested and still open

Tests include a candidate with much better token-average text loss but worse
opening loss: D correctly worsens. Field and trajectory normalization, explicit
absence, malformed likelihoods, coverage/overlap failures, and BPE boundary
checks also pass.

Still open: protocol-specific audited span construction, independent annotation
review, complete causal probe receipts, source/split admission, per-token scorer
integration, primary weight-mode/replication freeze, and the generation/transfer/
execution panels. An action-only interface may combine opening and choice in a
single token; this implementation deliberately does not invent disjoint labels
for that case. Representation comparisons require a separately reviewed common
metric policy.
