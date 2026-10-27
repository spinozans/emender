# X39 part 7 — provisional three-arm measurements; source gate failed

## Part 7 measured result — three trainings complete; provenance failed; fourth unrun

**All new scientific measurements are provisional.** Final integrity checking
detected external edits to X36 `backward.py` and `replay_setup.py` during the
single Python training process. Initial decay registration matched the B64
control's hashes for these modules, but later registrations saw changed disk
bytes. Cached Python imports suggest that much loaded code stayed unchanged;
this is not a complete loaded-source attestation. The unchanged-source gate
**failed**. `receipts/part7_integrity.json` retains each registered/current hash
mismatch. No training is claimed fully certified from this evidence. Upstream
files were read-only for this task and were not restored or modified.

Arms 1–3 completed sequentially at B64, seed 3401, context 512, each with
100,007,936 target tokens. The three lambda pilots each completed 20,021,248
tokens. Corrected runs held broker card 0 exclusively; certified cards 2/3/4
and forbidden card 7 were excluded by the unchanged X39 wrapper.
**The four-arm task is incomplete:** arm 4' was not trained. This is an
implementation blocker, not a scientific null from a trained semantic model.

| Arm | Pre-registered | Measured PPL / control ratio | Per-head slope | Disposition |
|---|---|---|---|---|
| 1 decay, horizon 128 | 100M; PPL ratio <= 1.05; slope <= .5 | 6.92328267 / 1.00975219 | 0.0000/0.0000/0.0000/0.0000 | provisional compression screen passes; source attestation fails |
| 2 gate cost, lambda 0.01 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.91114998 / 1.00798265 | 0.9575/0.9602/0.9294/0.9402 | quality passes; strong compression fails |
| 3 poverty, S128 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.89398492 / 1.00547915 | 0.0000/0.0000/0.0000/0.0000 | capacity-capped; rich transfer is the disposition test |
| 3 same weights, S2944 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.87446791 / 1.00263262 | 1.0000/1.0000/1.0000/1.0000 | quality passes; strong compression fails |
| 4' semantic keys + match wire | Operating lambda; 100M; parity + ReplayBackward | Not run | Unmeasured | Optimized recurrent wire absent; incomplete, not a trained null |


The control is 6.8564175640 at 100M and 14.6701488443 at the exact pilot
exposure. The selected lambda is **0.01**, the largest swept value passing
the registered threshold 14.96355182. The full run starts
independently; it is not a continuation of its pilot.

| Lambda | PPL at 20,021,248 tokens | Ratio to control | Within 2% |
|---|---|---|---|
| 0.001 | 14.67135383 | 1.00008214 | True |
| 0.01 | 14.78951122 | 1.00813641 | True |
| 0.1 | 15.21379435 | 1.03705794 | False |


`receipts/part7_lambda_interpolation.json` records descriptive log-lambda/PPL
boundary interpolation between bracketing pilots, with no extrapolation and
no invented unswept operating point. Validation checkpoint receipts and every
training loss/gate mean are retained as curves. The repeated .01 full-run
20M validation is 14.7947889612, versus the pilot's 14.7895112197; both pass.
Same-seed CUDA training is not claimed to be bit-identical across executions.

### Synthesis and structural-lessons ledger

The provisional decay measurements show non-capacity-limited sublinear retained memory at an acceptable
short-context quality cost. Its unchanged .5 allocation rate distinguishes
expiration from selective writing. Gate cost reduces writes substantially,
but does not produce the registered strong sublinear scaling. No strong compression transfers to rich deployment.
The poor/rich evaluations share the exact checkpoint hash, changing only the
deployment budget. Pressure zero in a rich budget is not itself an improvement:
the control also has zero pressure over these lengths.

| Structural lesson | Trained evidence and limit |
|---|---|
| Decay bounds retained memory without reducing writes | Every registered length peaks at 64 slots/head; slope about zero, allocation .5/token/head, zero rich pressure; short-context PPL costs .975%. This is recent retention, not semantic deduplication. |
| Write-enable cost reduces the allocation coefficient | Selected lambda .01 gives slopes .9294–.9602: fewer writes, but the <=.5 strong-compression criterion fails in all heads. PPL costs .798%. |
| Capacity caps can mimic compression | S128 occupancy must be interpreted alongside pressure and the same trained weights at S2944; low capped slope alone is not learned compression. |
| Semantic retrieval is not trained deduplication evidence | Prior constructive CPU wire passes its two ablations, but no optimized recurrent wire or trained semantic result exists. |


| Arm | T | Allocation/token/head (two-window mean) | Pressure fraction/head | Context PPL / control |
|---|---|---|---|---|
| decay | 512 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 1.009874 |
| decay | 1024 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.961139 |
| decay | 2048 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.910427 |
| decay | 4096 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.928291 |
| gate_0.01_full | 512 | 0.3105/0.2236/0.1689/0.2314 | 0.0000/0.0000/0.0000/0.0000 | 1.033041 |
| gate_0.01_full | 1024 | 0.3047/0.2134/0.1621/0.2280 | 0.0000/0.0000/0.0000/0.0000 | 1.008674 |
| gate_0.01_full | 2048 | 0.3022/0.2131/0.1577/0.2229 | 0.0000/0.0000/0.0000/0.0000 | 0.975756 |
| gate_0.01_full | 4096 | 0.2979/0.2124/0.1404/0.2119 | 0.0000/0.0000/0.0000/0.0000 | 0.990245 |
| poverty | 512 | 0.2500/0.2500/0.2500/0.2500 | 0.5000/0.5000/0.5000/0.5000 | 1.013493 |
| poverty | 1024 | 0.1250/0.1250/0.1250/0.1250 | 0.7500/0.7500/0.7500/0.7500 | 1.004890 |
| poverty | 2048 | 0.0625/0.0625/0.0625/0.0625 | 0.8750/0.8750/0.8750/0.8750 | 0.991460 |
| poverty | 4096 | 0.0312/0.0312/0.0312/0.0312 | 0.9375/0.9375/0.9375/0.9375 | 0.994263 |
| poverty_rich | 512 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 1.010187 |
| poverty_rich | 1024 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 1.001422 |
| poverty_rich | 2048 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.980498 |
| poverty_rich | 4096 | 0.5000/0.5000/0.5000/0.5000 | 0.0000/0.0000/0.0000/0.0000 | 0.983249 |


Full occupancy profiles (mean, peak, final), allocation/request/enable/sigmoid
rates, per-head pressure/R-CAP/W64 statistics, first 16 story profiles with
truncation flags, seven depth probes and all per-position context NLL are in
the per-arm `part7_*_evaluation.json` receipts. `part7_decay_shape.json` gives
128-token NLL bins and matched context ratios; raw position curves remain in
the evaluation receipts. The decay context ratios are about 1.0099/.9611/
.9104/.9283 at T=512/1024/2048/4096: longer-context degradation relative to
control is not observed on these registered windows.

### Validation, provenance and remaining work

Every invocation passed a fresh exact-configuration B64/H4/K64/V128/T512
comparison against certified X31: forward values, final state, routing traces
and gradients bitwise. This is the targeted production-shape gate, not a new
claim that the entire adversarial suite was rerun. Parameter receipts match
the B64 control's 21,913,872 total / 21,847,052 trainable tape parameters
against base_lm's 21,914,119 total. Every accepted update asserts ReplayBackward
and finite loss/gradient norm. All 11 existing X39 CPU tests pass.
`part7_integrity.json` verifies uninterrupted step/exposure sequences but
**fails unchanged-source provenance**. Each run retains the exact
`registered_protocol.md` bytes used for its fingerprint.

An initial implementation attempt was stopped after 460 updates / 15,073,280
tokens because its RNG-consuming model-construction order differed from the
control. It is archived under `receipts/part7_aborted_initialization`, excluded
from all results, and replaced by fresh runs matching control construction
order. Every gate run's first-step CE exactly equals the control's first CE.
No checkpoint files are committed. Pre-existing upstream working-tree edits
remain untouched; their source hashes identify the implementations used.

Arm 4' still requires an optimized recurrent pre-read-match-to-write-enable
lane and its exact oracle/replay certification. `part4_match.py` is explicitly
a sequential CPU witness; its 1/sqrt(K) evidence scale differs from certified
READ's 2/sqrt(K). The archived report also leaves error-term placement
unresolved. No architecture was silently substituted. Discovery by tool name
and description exposed no contact_supervisor/Intercom capability; no alternate
orchestration service was started. `part7_semantic_blocker.json` records this
remaining work. A title claiming four trained arms would misstate the result.

Residual risks: failed source provenance from concurrent external changes,
one training seed, small registered validation/probe samples,
same-seed numerical variation, and no learned semantic-arm evidence. Capacity
limits can destroy information while flattening slopes. Reported bands are
fixed-window comparisons and do not establish general language quality or
longer-than-4096 retention.

## Historical reports (superseded where arms 1–3 are described as unrun)

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
