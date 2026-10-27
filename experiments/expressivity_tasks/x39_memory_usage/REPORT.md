# X39 result — control trained; usage arms incomplete

## Part 6 — B64 control completed; four usage arms incomplete

The newly authorized B64 exact-exposure control **completed** 3052 updates,
100,007,936 target tokens, on broker card 1 in **831.07 seconds**. Its final
fixed-window validation perplexity is **6.8564175640**. At update 611 /
20,021,248 tokens its perplexity is **14.6701488443**; this is the denominator
for the registered 2% lambda decision. No X39 lambda pilot has been run, so
there is neither a selected lambda nor evidence for a no-qualifier null.

The amendment and its operator provenance are in RECEIPTS.md. Configuration
and data/source hashes are in `runs/control_b64/registration.json` and
`receipts/part6_control_training.json`. X36's optimized model preserves the
X34 skeleton, initialized weights, policies, seed, data order and optimizer;
only the registered batch change changes update grouping. Parameter totals
are 21,913,872 (21,847,052 trainable), versus base_lm's 21,914,119. Every one
of the 3052 loss graphs asserted ReplayBackward, and every gradient norm and
loss passed finite checks. The raw training curve is `runs/control_b64/curve.jsonl`;
validation checkpoints and `receipts/part6_control_curve.svg` retain its curve.
Checkpoint weights remain local and ignored by git.
`runs/control_b64/registered_protocol.md` preserves the exact pre-training
registration bytes and verified hash before the final dispositions were appended.

Before training, the fresh production-shape B64/H4/K64/V128/S2944/T512 X36
versus X31 forward/backward test passed in 16.25 seconds, leased on card 6.
Its XML and console are `receipts/part6_control_parity.{xml,txt}`. This test
compares the optimized implementation with the certified X31 implementation;
it is not an assertion that the entire adversarial suite was rerun. The
existing CPU X39 suite also passed all 11 tests.

`receipts/part6_control_evaluation.json` contains two windows at each registered
length, full per-position NLL, occupancy profiles, successful allocations,
requests, write-enable fractions, sigmoid strength, per-head pressure and
R-CAP/W64 statistics, the first 16 validation stories, and seven depth probes.
Evaluation verified that all training-recorded source hashes still matched.
The measured per-head occupancy slopes are **1, 1, 1, .9997886161**. Rich-budget
pressure is zero throughout the four packed-context evaluations. Allocation
is .5/token/head in most windows, with small reductions in head 4 at longer
contexts. Control context perplexities at T=512/1024/2048/4096 are
7.164688392/8.817347908/12.383386623/18.992085553; these use different windows
from the fixed short-context quality validation above.

### Completion limit and remaining ordered work

**The four usage trainings, X39 lambda sweep and all trained-arm adjudications
remain unrun.** The requested part-5 directive and blocker receipt are not in
the checkout or recent X39 history. Tool discovery by name and description
found no contact_supervisor/Intercom capability, so the required decision
channel cannot supply those inputs. No alternative service was started.
The part-6 instruction supplies enough detail to complete the control; it
does not supply the missing directive to execute the subsequent arms exactly.
In particular, the part-4 report leaves the optimized match wire and placement
of the error term unspecified. No architecture was silently substituted.

An additional source-level discrepancy should be resolved with that directive:
`part4_match.py` computes match evidence with scale 1/sqrt(K), while actual
certified READ uses the sharpened 2/sqrt(K). Its comment equating these scales
is inaccurate. The constructive repeat-suppression witness remains evidence
for its recorded circuit; it does not certify the trained match wire. The
historical receipt and its hashes are retained unchanged.

X37/X38 remain read-only. `receipts/part6_family_comparability.json` rechecks
recorded batch, seed, context, tokenizer, packed train/validation data hashes
and exact evaluation exposure. X37 b/c at 100M have perplexity ratios
.9844033740/.9761630030 to the new B64 control, both within the no-more-than-2%
worse band. X37 d lacks a completed endpoint. X38 pilots at 20M have ratios
.9973254272/.9989068527/1.0102315072/1.0222863909 for costs
0/.001/.01/.1: the first three pass 2%, .1 fails. Its .01 full run has ratio
.9821494932 at 100M and passes. These are family quality comparisons, **not
X39 gate-cost lambda selection**. Archived model implementations were not
recertified and upstream conclusions were not rewritten.

The part-4 code, tests, receipts and documentation are included with this
partial result. A success title claiming four trained usage arms would be
false; the commit records the completed control and incomplete arms instead.
No compression claim follows from this control. The ordered X39 arm runs and
their registered poor/rich, decay, quality, story/depth/context and slope
receipts are still required.

## Part 4 partial result — two gates passed; training incomplete

`receipts/part4_decay.json` records a broker lease on physical card 6 and the
optimized X36 lane at fp32 decay **0.9474635124206543**. Against the CPU oracle,
the entire `freed`, `alloc_target`, and `case` tensors and final live mask agree
exactly, with nonzero decay deaths. This covers isolated births at B1, H4,
K64, V128, S2944, T512. It does **not** certify B64 training configurations,
gradients, ReplayBackward, or the full adversarial parity suite.

`receipts/part4_match.json` establishes the requested constructive CPU circuit.
With semantic keys and detached pre-read maximum scaled match evidence wired
to write-enable, novel enable/allocation fractions are **1.0/1.0** and repeat
fractions are **0.0/0.0**. Identity keys with the wire, and semantic keys without
the wire, both retain repeat enable/allocation **1.0/1.0**. Empty-store evidence
is zero; the hand-set construction uses write bias +2, match coefficient -1,
zero error coefficient, no POP, and an unchanged sequential CPU oracle. No
optimized read-to-gate implementation or learned behavior is claimed.

The CPU suite passes **11 tests**. Initial receipt execution caught a Python
scalar in the unwired ablation; replacing it with a shape-matched zero tensor
fixed the error before the passing receipt was generated.

**The requested training and evaluations remain unrun.** There is no lambda
operating point, per-arm parameter receipt, B64 oracle certification,
ReplayBackward attestation, trained slope, perplexity band, decay NLL curve,
or poor/rich deployment result. The B8 100M X34 checkpoint cannot supply the
registered exact-exposure control for B64 20M lambda selection. No supervisor
tool was exposed after discovery by name and description. No alternative
orchestration service was started. No success-titled commit or push was made
because that would claim that the cleared arms were trained.

Residual risks: hand-set score margins may not survive learned feature scales;
the architectural term is not yet implemented in an optimized training lane;
the existing error-magnitude coefficient belongs to the append-amplitude gate,
so a future implementation must explicitly specify its placement in the new
write-enable gate. Capacity-capped slopes can still conceal lost information.
All upstream edits already present in the shared tree remain untouched. No
`.pt` files were added. The sections below preserve earlier historical stops;
the part-4 instruction supersedes their blanket semantic stop for arms 1–3.

**The four-arm experiment is incomplete.** Pre-registration, CPU mechanism
counterexamples, and analysis of the existing X34 control are complete. No X39
training, lambda selection, parameter-match certification, GPU parity, or
trained-arm evaluation has run. This is not a scientific null from trained
regularized models.

## Measured control

The immutable X34 receipt already includes the requested longer contexts:

| T | Peak slots/head (all four heads) | Allocations/token/head | Write enabled | Pressure |
|---|---:|---:|---:|---:|
| 512 | 256 | .5 | 1 | 0 |
| 1024 | 512 | .5 | 1 | 0 |
| 2048 | 1024 | .5 | 1 | 0 |
| 4096 | 2048 | .5 | 1 | 0 |

OLS log-log peak slope is **1.0000000000000004** for every head. Story peak
median is 102, maximum 264. This supports the transcript diagnosis without
new training or GPU use. Mean occupancy and occupancy profiles are unavailable
in the original receipt and remain explicitly missing, rather than inferred
as fresh measurements. Source hashes are retained in `receipts/preflight.json`.

The actual imported T2 defaults give a **1,381,547.60-token** untouched-slot
fade horizon, longer than the task's approximate 200k estimate. The 128-token
arm's derived decay is **0.9474635256553754**, float32
**0.9474635124206543**, implied horizon **127.99996687** tokens. Exact GPU
death timing still needs configuration-specific parity; this calculation alone
does not certify the optimized lane.

## Blocking mechanism findings

1. `x30_tape_cell/naive_reference.py` selects append requests before computing
   the sigmoid append gate. The gate multiplies the stored value; allocation
   initializes strength to 1 regardless of its magnitude. X36 `kernels.py`
   uses the same order. CPU gate-logit extremes -20/+20 yield identical
   allocation traces with fixed nonzero values, while the read loss has a
   nonzero gate-logit gradient. Thus the differentiable path exists for value
   strength, not allocation rate. Indirect changes in residuals or POP may
   change future occupancy; that does not turn this gate into an allocation
   probability or supply a derivative of the hard allocation count.
2. Write correction is addressed by the age-1 slot. Random versus repeated
   keys produce identical routing/allocation traces when POP is disabled and
   content is fixed, although reads change. X37 W_in therefore cannot alone
   turn novelty routing into semantic matching. In addition,
   **R-ZERO-RESIDUAL appends**: zero-content repeated-concept counterexamples
   allocate on every token. Making this a no-op would change certified
   semantics. X36 explicitly treats cases 1, 2, and 3 as append requests.
3. X34 registration, CLI, and fingerprints specify effective batch **8**;
   the requested **64** changes optimizer updates. The control's existing
   receipt is at 100,003,840 tokens, not exactly the proposed 20M pilot
   exposures. Lambda selection needs an exact matched-exposure control
   validation receipt and a resolved batch policy.

These are scope decisions for the operator: define an allocating gate and its
gradient, define semantic match/no-op routing and recertify replay, and resolve
batch matching while preserving the instruction not to retrain control.
The requested Intercom contact_supervisor capability is unavailable: discovery
found no supervisor/intercom tool. No substitute service was started.

## Validation and remaining work

`/home/erikg/emender/.venv/bin/python -m
experiments.expressivity_tasks.x39_memory_usage.audit` passed and wrote the
source-hashed machine receipt. `/home/erikg/emender/.venv/bin/python -m pytest
experiments/expressivity_tasks/x39_memory_usage/tests -q` passed **5 tests**.
These are CPU oracle counterexamples and receipt-analysis checks; they are
explicitly not GPU oracle parity or ReplayBackward acceptance.

All upstream experiment directories were read-only in this task. No GPU was
allocated, no existing checkpoint was modified, and no .pt file was added.
The shared tree already contained extensive unrelated changes, including X36
source modifications, and those remain untouched. Source hashes identify the
working-tree implementations inspected; they must be checked again before any
future GPU certification.

After operator resolution, the implementation, broker runner, exact per-arm
parameter receipts, fresh oracle parity, ReplayBackward assertions, sequential
20M sweeps and 100M runs, poor/rich evaluation, occupancy profiles and decay
curves are all still required. Do not infer compression or acceptable quality
from this preflight. Rich S=2944 control does not saturate by T=4096, so a
zero-pressure result there would not demonstrate an improvement over control.

## Part 2 — redesigned CPU mechanisms (incomplete)

The operator superseded the earlier scope questions: use B64, penalize the
continuous **write-enable** sigmoid (not append amplitude), keep certified ISA
semantics, and combine semantic keys with that cost. The original preflight
above remains historical evidence, not a veto of these redesigned paths.

`redesigned.py` optimizes a learned scalar write bias initialized at +2, on
fixed features in [-1,1]. The objective is exactly lambda times mean sigmoid;
SGD uses learning rate 10 for 100 steps. The resulting sigmoid is thresholded
and detached as in the imported cell before entering the unmodified CPU oracle.
Costs 0/.001/.01/.1 yield physical allocation rates
.5/.5/.3359375/0. This proves indirect control across a hard threshold, not a
derivative of allocation count or language-model quality. The first nonzero
cost leaves allocation unchanged; monotonicity is non-increasing, not strict.
These artificial optimizer settings do not select a training lambda.

The derived decay arm is exercised for 512 tokens with S=512, showing decay
deaths and occupancy equilibrium without R-CAP. Full per-token/head occupancy
is retained in `receipts/redesigned_mechanisms.json`. The tests also verify the
strict .5 write-enable boundary. Eight CPU tests pass, including all five
original binding counterexamples.

**The requested training experiment is still incomplete.** No GPU lease, GPU
parity, ReplayBackward assertion, parameter certification, lambda sweep, 100M
training or trained-arm evaluation was performed. Poverty and semantic-repeat
mechanism receipts remain unimplemented. Pressure is returned as telemetry in
X34; its objective is cross entropy only. A poverty mechanism receipt must
therefore demonstrate an effect through capacity-limited reads on loss, rather
than treating pressure telemetry as an existing explicit loss penalty.
No claim is made that this absence alone prevents scarcity learning.

B64 rounds pilot/full runs to 611/3052 updates, 20,021,248/100,007,936 tokens.
This resolves the requested training batch but does not magically supply a B64
matched-exposure control for the 2% lambda decision: the original X34 checkpoint
is B8 at 100,003,840 tokens. That comparison remains unperformed.

Residual risks: synthetic gate closure can erase useful writes; the semantic
read-confirms circuit has not been learned or demonstrated; budget saturation
can produce misleading low occupancy slopes. X37's per-head W_in is not the
requested shared W_in and must not be reused unchanged. No upstream source or
ISA semantics was edited. No .pt files were added. Intercom supervisor tools
remain unavailable in this session. This is partial mechanism evidence only.

## Part 3 — poverty receipted; semantic gate failed, stopped before GPU

The ordered CPU preflight ran in `part3_mechanisms.py`, with source hashes and
full poverty occupancy traces in `receipts/part3_mechanisms.json`. **The semantic
mechanism did not pass. The fail-closed instruction stops this experiment before
GPU parity, training, or evaluation of any arm.** No GPU broker was invoked.

Poverty uses 24 orthogonal bindings, one enabled write every other token, then
24 read-only recall queries, identically across four heads. Only budget changes.
The real X30 CPU cell's 128-token cap/request pressure register is recorded;
its read outputs agree with the differentiable sequential oracle within 2e-5.
Cross entropy is computed from 20 times the recalled value with the binding
identity as target. Values receive nonzero loss gradients. There is no added
pressure penalty and these are synthetic recall losses, not LM perplexities.

| S/head | Peak live/head | R-CAP (all heads) | Pressure max | Recall CE | Recall accuracy |
|---|---|---|---|---|---|
| 8 | 8 | 64 | .666667 | 3.882920 | .333333 |
| 16 | 16 | 32 | .333333 | 2.166516 | .666667 |
| 32 (rich counterfactual) | 24 | 0 | 0 | 0 (fp32) | 1 |

This receipts a scarcity teaching signal through capacity-limited reads and
loss, not successful scarcity learning or transfer to S=2944.

Semantic repeats use eight concepts, each presented twice with different alias
identities. A single hand-set shared W_in folds the alias pairs to common
content keys before identical per-head identity geometry. Mean target read on
repeat tokens improves from .090664 (identity keys) to .999920 (semantic keys).
However, repeat write-enable and allocation fractions are **1.0 in both cases**;
allocation traces are exactly identical. Retrieval succeeds; gate closure fails.

The inspected X36/X37 path computes `W_write(x)` before the tape scan and has no
current tape-read input to that gate. Changing shared key/query features does
not itself implement read-confirms-stored gating. This witness does not prove
that transformer context cannot learn repetition indirectly; it records that
the requested CPU circuit was not established by the hand-set construction.
No read-to-gate architectural redesign was invented, and no trained semantic
null is claimed. The arm requires redesign or disposition before GPU work.

Validation: the full X39 CPU test suite passes **10 tests**, including two new
mechanism regression tests. A passing regression asserts the semantic failure
and stop condition; it does not turn the scientific gate green. The Intercom
supervisor tool remains unavailable after discovery by name and description.
Upstream directories and unrelated working-tree changes were left untouched.
No .pt files were added. The requested success commit title would misstate the
outcome; the part-3 commit records the mechanism failure instead.

Residual work: decay GPU death-schedule parity, per-arm parameter receipts,
ReplayBackward assertions, lambda operating-point selection with a valid
matched-exposure control, all training, and every trained-arm evaluation remain
unrun. The prior B8-control/B64-training comparison limitation remains.
