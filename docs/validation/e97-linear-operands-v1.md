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
do not call this diagnostic success a capability gain. Results pending.
