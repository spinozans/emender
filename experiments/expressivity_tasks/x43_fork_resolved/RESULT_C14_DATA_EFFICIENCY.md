# C14-B data-efficiency receipt

Single-seed endpoint comparison from existing receipts; no new training or evaluation.
Loss is final-test nats per raw byte. Exposure counts real training targets,
excluding padding. These are endpoints, not measured learning curves or a
matched-loss sample-complexity estimate.

| Arm | Final-test NLL | Training targets | Dense-anchor NLL improvement per million targets |
|---|---:|---:|---:|
| dense | 2.276133549 | 225,880,702 | 0.000000 |
| passive v2 | 1.509260010 | 179,162,804 | 0.004280 |
| budgeted B1-v2 | 1.584700014 | 71,677,042 | 0.009647 |
| unbudgeted | 1.716496444 | 14,455,380 | 0.038715 |
| append-only | 2.246020115 | 75,394,452 | 0.000399 |
| first attempt | 2.206163868 | 479,328 | 0.145975 |

The descriptive efficiency statistic is (frozen dense NLL minus arm NLL) /
training exposure. It is **9.045x** larger for unbudgeted than passive.
Passive saw 12.3942x more targets, and still has the better
absolute loss (1.509 versus 1.716). Thus “9x” means dense-anchor improvement
per exposure in this single-seed endpoint comparison; it does not mean 9x
less data to reach the same loss. The dense anchor itself consumed 225.9M
targets, and this normalization is not a causal attribution or a learning slope.
The first attempt's large normalized gain is especially unstable at only
479,328 targets; retain it as engineering evidence, not an efficiency winner.

Unbudgeted trained 7,142 steps in 2700.288256 active seconds (378.086 ms/step)
versus budgeted's 35,419 in 2700.032370 (76.231 ms/step): roughly 4.96x
active compute time per position under the same document-local batch protocol.
This is a trained-heavy-compute counterpoint, an accidental pondering model
in the operational sense of spending more compute per training position.
It is not proof of greater serial reasoning depth, useful FLOPs, or intrinsic
architectural cost: allocated slots, masked-slot tensor work, recomputation,
implementation overhead and trajectory remain confounded. Peak live 110 is
not a stationary mean; early sampled occupancies are nearly equal. A 2.3x
peak comparison cannot establish worse-than-linear live-slot scaling.

[Operator rounds](RESULT_OPERATOR_ROUNDS.md) added inference rounds to weights
not trained for those extra rounds and worsened fork NLL (B1 k=1/2/4:
1.912022/2.012623/2.165866). Here weights were trained throughout the expensive
unbudgeted computation. The results distinguish training with heavier compute
from adding rounds to a frozen policy; they do not establish that trained
extra rounds would help. Single seed, different exposure and capacity prevent
that causal conclusion.

Budget pressure remains unresolved. The named matched-exposure unbudgeted twin
at exactly 71,677,042 targets is the resolving comparison, not an extrapolated
score. This receipt neither runs it nor changes any frozen arm.

Sources: [B1](RESULT_C14_B1.md), [triangle](RESULT_C14_ABLATION_TRIANGLE.md),
[occupancy](RESULT_C14_OCCUPANCY_SCALING.md),
[dense](RESULT_C14_DENSE_TAKE4.md), and the passive exposure retained in PROGRAM.md.
Existing step-composition receipts show full configured-slot replay reductions
and a verified 17.4% wrapper improvement at budgeted capacity; they do not
certify an intrinsic unbudgeted ceiling.

Manual notes: supervisor/intercom transport is absent from the available tool
catalog. No supervisor contact is claimed. No GPU acquired for this receipt;
no model, checkpoint, service, protected directory or production kernel changed.
No resilient DiLoCo work; gap-matrix IDs are inapplicable.
