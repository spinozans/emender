# X39 result — blocked before training

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
