# X42 velocity protocol

This protocol replaces the legacy fixed-token adapter search. The executable
registration JSON binds this document, the exact score source verbatim, corpus
attestations, dependencies, validation evidence, and all search dimensions.
No scientific candidate may start before that JSON is committed.

The proxy is pure EmenderLM: 12 memory/FFN blocks, width 768, untied GPT-2
50257-vocabulary embedding/projection, approximately 150M trainable parameters.
Frozen per-head geometry/embeddings add parameters reported separately. Every
block has N heads; there is no attention. Horizons are h_min * ratio**head,
with decay exp(log(0.001)/horizon). Budgets are rounded
budget_min * budget_ratio**head. These are log-spaced spectra. Constants are
shared across layers, with separate per-head gate constants. Inactive head
coordinates remain in the CMA vector so changing N preserves population state.
The W_in width is a learned content bottleneck, not management capacity.

Match-wire multiplies match evidence before the gate sigmoid, independently
for each head: match_coef * (match_wire[head] * evidence). Write-gate bias is
a separate additive search coordinate. Management drives, including the
write projection, match coefficient and append/pop drives, are frozen.
The learned query/value/content paths retain the certified replay spine.

At the gate's existing operation point (after POP, before WRITE), each head
uses its own live-slot count divided by its own budget:
bias_eff = bias_base + slope[head] * max(0, occ_head - onset[head]).
Match-wire is in [0.25,4.0], onset [0.3,0.9], slope [0.1,2.0]. No management
parameter receives learned gradients. Local kernel/forward copies isolate
these policy additions from certified X40 source files and receipt gates.
Independent reconstruction from integral traces checks live occupancy; the
unchanged X40 scan checks state, routing, outputs, and first-order adjoints.

Each candidate starts from seed 3401 and the identical contiguous training
prefix, physical/effective batch 1, context 512. Packed token 0 is content.
Train-loop time includes mmap data loading, transfer, forward, backward,
clipping, optimizer, and synchronized step completion. Model construction,
optimizer setup, receipt writes and evaluation are excluded. Complete the
last update crossing 2700 seconds and report the overshoot. Never score a
shortened, killed, nonfinite, or failed candidate. A killed candidate restarts
from scratch with the same seed and prefix; no model checkpoint is used.

Use the certified memory-bounded FP32 vocabulary loss (ChunkedLoss, 1024
positions per projection chunk), FP32 stores, BF16 dense autocast, AdamW
betas .9/.95, weight decay .1, clip norm 1, searched LR and linear warmup.
The gate-cost coordinate multiplies detached policy-gate means in the train
objective; it introduces no management gradient. Its lack of influence on
trainable gradients is an explicit limitation of this frozen-management recipe.

Held-out NLL is the equal mean of 24 consecutive reset windows, context 2048,
from the attested valid prefix. Evaluation is outside the training budget.
For each layer/head report gate means, event density (non-NO-WRITE fraction),
live occupancy normalized by that head's budget, ever-touched slot fraction,
and R-CAP rate. Aggregate occupancy equally across layer/head observations
and held-out windows. Ever-touched is the mean per-window fraction of rows
allocated or matched, with state reset each window. Sum head R-CAP events per
token, average across layers/windows. Measure peak live occupancy at reset
lengths 512/1024/2048/4096 and report each log-log slope; the score uses the
maximum per-head slope. An empty head has slope zero and independently
receives the dead-store flag. Finite capacity remains visible in all receipts.

Score = NLL_heldout + 0.01 * occ_frac + degeneracy_penalties, MINIMIZED.
Penalties add 0.05 if any head's ever-touched fraction <0.01, 0.05 if
R-CAP events/token >0.05, and 0.02 if occupancy slope >=0.9.
The committed score utility is the arithmetic authority, copied verbatim into
registration. Tokens consumed and tokens/active-training-second are reported.

Run one full 2700-second smoke before generation 1. A direct detached parent
then owns one population of 12, full covariance CMA-ES, seed 4201, sigma .2,
reflected normalized coordinates and rounded integer dimensions. The parent
owns subprocesses, lease wrappers and failure signals. It fsyncs population,
means, covariance, evolution paths, RNG and completed endpoints atomically.
A resumed generation reuses that exact population, retaining completed scores
and rerunning incomplete candidates from scratch. Nonzero child exit stops
the parent and terminates its other candidate groups; failures are receipts,
never substitute scores. Persisted PID start ticks protect restart cleanup.

Only the broker pool 1/5/6 is enabled, with at most three candidate subprocesses.
Check PID 634691 first; this implementation conservatively excludes card 0 for
the entire run, including after the depth probe exits. Cards 2/3/4 and 7 remain
excluded. No supervisor service, WorksGood machinery, Slurm or DiLoCo is used.

Promotion rule: rank complete eligible candidates by exact minimized score,
then stable candidate index on an exact tie. Report the best full-generation
policy and comparison with the full-budget smoke; claim improvement only if
its score is strictly lower. Any degeneracy flags remain prominently reported.
This selection authorizes no automatic 1.3B training or scientific quality
claim. Scaling requires a separate measured scale-transfer validation.
