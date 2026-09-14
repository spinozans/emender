# E97 fixed-row FP32 Linear: one non-dispatched repair candidate

## Why this change

[Actual-operand replay](e97-linear-operands-v1.md) establishes matrix-height
dependence of pre-store FP32 QKV results at shared numeric input/weight addresses
and strides. BF16 conversion is faithful. Instead of increasing precision again,
try one fixed arithmetic schedule whose row count does not alter its compiled
arithmetic. This is a targeted candidate for removing that dependence, not a
claim of correctly rounded GEMM or a policy/model change already in use.

`ndm/triton/fixed_fp32_linear.py` introduces only a **non-dispatched prototype**:
`e97-row-fixed-fp32-m16-n64-k32-w4-s2-v1`. Fixed16-row×64-column tiles,32-term K
blocks, four warps,two stages; `tl.dot(..., input_precision='ieee')`, FP32
accumulator/output, no split-K/atomics/autotuning. Optional FP32 bias. M is a runtime
argument explicitly excluded from value and alignment specialization. A single
compiled variant is required across the actual one-/512-row and rotated calls.

No existing numerical policy imports or selects it. No weights, gradients,
optimizer, recurrence or checkpoint format changes. Standalone autograd is rejected
when requested; eventual model integration/backward tests are separate gates.

## Frozen micro qualification

Actual capsule authority: `linear-operands-v1` under
`/mnt/nvme2n1/erikg/e97_systematic_posttraining`.

- Capture summary `f53513d58ef4b6b12ee7b8f39dcbf950753feff7334fe46c695a1b68bd8cb47c`.
- Replay summary `80e2ff8a7a9ed5eb915f77c0951bf4122e370019ff7e1bedd828d4a0e2e9eb5d`.
- Capsule/output files must match their hash-bound capture manifest.

Before evaluating the candidate, the current native FP32 GEMM must reproduce both
complete original captured results exactly on the common teacher-backed input and
weight allocation. Then require:

1. Candidate one-row and512-row repeats exact; **one-row output equals the first
   row of the full output byte-for-byte**.
2. Rotate the actual512 input rows by3 and12 (the frozen packed anchor residues).
   Entire output matrices must equal the corresponding output permutations exactly,
   twice each. No fabricated or changed numeric rows.
3. All results finite and within `atol=1e-5, rtol=1e-5` of ordinary highest-FP32
   native GEMM. Twenty-five fixed FP64 reference dots (rows0,1,5,31,511; columns
   0,36,1023,5759,11519) satisfy the same bounds. This operation-level tolerance is
   not a replacement for the existing model log-probability limits.
4. A small synthetic N65/K33/M17 case tests bias and N/K/M tails, numerical
   agreement and exact standalone rows0,3,12,16. Synthetic cases remain distinct
   from captured production QKV evidence.
5. No input mutation; allocated HBM≤2GiB. Exactly one compiled actual-geometry
   kernel variant before the separate synthetic cases; retain its PTX.
6. Median isolated-GEMM slowdown **≤4× native** at each height. Three warmup pairs
   then ten timed pairs, alternating order to reduce clock/order bias. This is a
   micro feasibility cap, not production throughput qualification: casts and the
   complete model are not timed.

One candidate only; no automatic tile/warp/precision sweep. Maximum100 GEMMs,
one leased GPU,worker300s,outer600s,teardown30s. Immutable export/inventory, source
audits and owned cleanup; local CUDA/NUMA, isolated cache and expandable allocator
variables. No model loaded. Input fingerprints/HBM retained in a terminal guard
on ordinary exits; forced termination cannot guarantee in-memory receipts.

**111 CPU tests pass,15 CUDA skips**, including runtime-M specialization guards.
The GPU worker itself executes the new real/synthetic cases. Old numerical policies
and historical failures are untouched.

Plan SHA `9e4502cd7b37df43a3a0160667b0c57097b2462b6238613118e18a2eb6418159`.
Preflight `fixed-linear-micro-v1-preflight`; results `fixed-linear-micro-v1`.
Worker `scripts/qualify_e97_fixed_linear.py`, runner
`scripts/run_e97_fixed_linear.sh`.

A pass would justify bounded model-policy integration and the unchanged full57 /
120-position / packed qualification sequence—not admission or training. A failure
is retained and investigated at this small operation, without silently changing
the model or relaxing its numerical gates. Existing880+32 budgets remain closed.
## Completed micro result and subsequent priority decision

Source **`1ce5696a`**, process `proc_9439`,49s,exit0. All eight checks passed:
exact height and full-row permutation invariance, one actual compiled variant,
FP32/selected-FP64 accuracy, synthetic bias/tails, unchanged inputs, memory and
micro performance. Peak allocated HBM249,823,232 bytes.

| Matrix height | Native median | Fixed median | Slowdown |
|---|---:|---:|---:|
|1|.180736ms|.346944ms|1.91962×|
|512|1.281936ms|4.833168ms|3.77021×|

Paired raw timings, native/candidate receipts, candidate output tensors and PTX
are retained. Independent audit recomputed the exact height/permutation hashes,
repeat hashes, recorded FP64-dot comparisons, medians/ratios and input fingerprints;
17,707 source files/inventory verified before and after. Summary SHA:
`fe0cc9e2d010078ea827c1acd10616f4a8eca7d9ca4240143da41a581bc27617`.
PTX SHA `2c6f58cd0eb99aabd7f990b5be70e914ce391f7d678500eee5bd12501adf0c36`.

**This kernel was not integrated into a model policy.** Following the operator's
programme-level review, do not make its exact arithmetic a blanket prerequisite
for SFT or accept its throughput cost automatically. Retain it as a bounded
reference/repair candidate and prioritize verified data and learning experiments.
See [current learning status](../E97_AGENT_LEARNING_STATUS.md) and the
[numerical exploration index](e97-numerical-exploration-index.md). No updates,
training readiness or behavioral improvement resulted from this micro test.
