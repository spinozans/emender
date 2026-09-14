# E97 actual Linear operands and model-free replay

## Purpose in the learning programme

The goal remains observed conversational/tool competence, not perpetual numerical
instrumentation. The [early-prompt trace](e97-early-prompt-trace-v1.md) narrowed the
blocked packed-stage prerequisite to the first QKV Linear operation, before
recurrence. This experiment supplies the small, reference-bound reproducer needed
to choose a targeted repair; it is not another precision sweep or learning run.
Routine bounded diagnosis/repair/qualification continues without requesting a new
operator handoff at every result. A failed gate still blocks advancement through
that gate; the880+32 training budgets remain closed.

## Immutable references and question

Same correction-y parameters, same `fp32-linear-v2`, same original record0 native
executions. Actor uses one-row QKV calls, teacher's first call uses512 rows, with
3,840 input and11,520 output features. The six-row trace bound original native
scores but observed only BF16 projection outputs. This experiment separates the
actual FP32 GEMM result from the subsequent BF16 conversion.

Reference `early-prompt-trace-v1` under
`/mnt/nvme2n1/erikg/e97_systematic_posttraining`:

| Artifact | SHA256 |
|---|---|
|Summary|`406b8896763c337572ba2681b09c693a3466236adbf97623696a99f2d27eee02`|
|Fine actor tensor bank|`441d3800b2e7f72960507bf5bf7e82d937442843684c07ec6eca61f11d9c284a`|
|Fine teacher tensor bank|`51e0d67f21bf9671d10e725004c55c5845c3480186bc61c5c4e868da9d5a1040`|

The original packed baseline's plan, inputs and repeated actor/teacher receipts
are also verified by the existing reference loader.

## Capture and controls

Scope a functional observer to the **first invocation of layer0 QKV only**. Capture
the actual transient FP32 input, weight and bias with the existing storage/stride/
offset/alias-preserving capsule. Invoke the original `F.linear` on its original
objects, retain its actual FP32 result and return that same object. The module's
passive post-hook records its native BF16 output and returns `None`. Restore the
owned functional symbol and hooks on success/failure; break the observer's
bound-method reference cycle promptly.

Actor twice and complete original teacher twice: four full-model evaluations.
Each must preserve exact original scores/full teacher receipt and the appropriate
one or six BF16 QKV trace rows. Actual FP32 input rows must bind to the trace's
BF16 inputs; transient weights must bind exactly to the BF16 model parameter's
cast. Operands must remain unchanged during GEMM. All captured FP32 results and
BF16 outputs must repeat exactly. CPU FP32-to-BF16 conversion must reproduce the
native BF16 stores exactly. Otherwise stop, retain failure and do not interpret.

After model teardown, a separate model-free process performs exactly eight GEMMs:

1. Restore each original capsule, preserving its shape/stride/offset/aliases;
   replay twice and require exact **full** FP32 and BF16 captured outputs.
2. Require equal actor/teacher weights/bias and equal first input row. Use one
   shared teacher-backed GPU input/weight allocation. Compute with row counts1
   and512, twice each, requiring the corresponding exact captured outputs.
   The input-row/weight addresses and input strides must remain the same.

This tests dependence on matrix height at fixed numeric entries/addresses/strides.
It includes resulting backend selection and output allocation; it does not claim
an instruction-level mechanism or independently vary those mechanisms. Original
GPU addresses are recorded, not reconstructed. No changed rows, synthetic filler,
alternate matmul precision, output substitution or model-free repair is included.

## Resource and safety contract

Capsules and captured FP32 results each≤256MiB; planned transient CPU capture
allowance2GiB excluding checkpoint mappings. Production QKV cast is176,947,200
bytes, an actual bounded transient arithmetic operand—not an FP32 master model.
Model parameters stay BF16; no gradients or optimizer state are created.

One lease across sequential capture/replay workers; no overlap/retry. Capture≤900s,
16GiB allocated HBM; replay≤300s,2GiB HBM; outer≤1,500s,teardown30s. Reuse terminal
parameter/buffer/gradient/tuner/HBM audit for capture; replay publishes its memory
receipt even on ordinary failure. SIGKILL cannot guarantee in-memory receipts.
Immutable export/inventory, source audits and owned lease cleanup, local CUDA/NUMA,
isolated caches, both expandable allocator variables, highest FP32/no TF32 and
fixed recurrence remain required.

CPU tests cover actual transient operands, faithful BF16 conversion, unchanged
original return-object identity, first-call selection, mutation rejection and
restoration after an original-operation exception. Existing capsule/trace/policy
regressions also run before a lease is acquired. **109 CPU tests passed,14 CUDA
skips** before export; no production policy code changed.

Plan SHA `01fc9ac66a5bee2d183a45b4187f55b2a344208dda07e4a28320e5ab96b52d6e`.
Preflight root `linear-operands-v1-preflight`; results `linear-operands-v1`.
Worker `scripts/diagnose_e97_linear_operands.py`, driver
`scripts/run_e97_linear_operands.sh`.

No64K forward, backward, optimizer update, admission or historical-probability
replacement. Existing failed gates stand. After bound replay, proceed toward a
targeted implementation repair and the unchanged numerical/packed qualifications;
do not call this diagnostic success a capability gain.

## Result: exact model-free replay; pre-store matrix-height dependence

Source **`0bb82822`**, process `proc_2e02`,253s,exit0. All four complete native
captures preserve original scores/teacher receipts and sampled trace bytes.
Operands, full FP32 outputs and BF16 stores repeat exactly. All eight model-free
GEMMs reproduce their corresponding full captured FP32 and BF16 outputs exactly,
including both matrix heights on the shared input/weight allocations and strides.

The first FP32 output row differs at **10,245/11,520 elements**, maximum absolute
gap **1.9073486328125e-6**. At the first previously identified BF16 discrepancy,
feature36, the actual pre-store FP32 difference is **5.960464477539063e-8**.
Every BF16 store matches conversion of its actual FP32 result. Thus the observed
cross-path difference is already present in the matrix result, not introduced by
an inconsistent BF16 conversion. Matrix height controls the difference at common
numeric input/weight addresses and strides; backend and output-allocation choices
remain coupled, and no instruction-level mechanism is claimed.

Independent CPU audit verifies input/weight equality, full output hashes and
conversion, all native/replay bindings, shared addresses/strides, source inventory
and17,702 source files. Capture parameter/buffer fingerprints are unchanged, no
gradients/timing-tuner entries. Allocated HBM peaks **9,004,484,608 bytes** for
capture and **253,755,392 bytes** for replay. Eight GPUs idle, no compute processes
or lease files after cleanup. No updates, policy changes or64K work.

| Artifact | SHA256 |
|---|---|
|Capture summary|`f53513d58ef4b6b12ee7b8f39dcbf950753feff7334fe46c695a1b68bd8cb47c`|
|Replay summary|`80e2ff8a7a9ed5eb915f77c0951bf4122e370019ff7e1bedd828d4a0e2e9eb5d`|
|Independent audit|`e71e22885658c3a0abe0b4aa40453e8142c36badd09dc9c785c4705f93cad43f`|
|Actor capsule|`256b4d7b2d9876dda81a8fa60c0a531d2f84b5dabf6a1236a58fb8d084a121f7`|
|Teacher capsule|`1150994cf12c22dbc74d55334e4bff1e61445c23ff7b443ccd6042a6b098bc8e`|

Next is one **non-dispatched fixed-row FP32 kernel candidate**, tested on these
same operands for numerical accuracy, exact height/row-placement invariance and
bounded slowdown before model-policy integration. No precision sweep or claim of
training readiness follows from this reproducer.
