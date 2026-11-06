# Round7 fused op-cycle: compiler-rejected prototype; INCOMPLETE / NO-GO

Design a346ffe9 was committed and pushed before implementation. The prototype
and two compiler failures were committed/pushed in efe195a2. No production
kernel or registration changed. The requested shared-memory op-cycle is NOT
implemented: the retained prototype is the explicitly designated global-scratch
fusion milestone, using existing arithmetic helpers inside one owning CTA.

## Launch headline and failure

Baseline: round6 CUPTI measured 6145 launches at T1024/BS2048. The persistent
prototype schedules initialization plus one owning-CTA launch (two intended
launches). **Measured successful fused launch count: unavailable.** It fails
LLVM lowering even at B1/T32/S65/BS128, before candidate execution. Two must
not be presented as a measured 3072.5x launch reduction.

quick_terminal.json = exit1: R5 tiled READ inside persistent cycle.
retry_terminal.json = exit1: simpler R3 pair READ inside persistent cycle.
step_terminal.json = exit1: remove the outer context loop, dispatch one fused
cycle per position instead (T+1 intended launches). This isolates away the
persistent context loop as a sufficient explanation; it is not a new design
promotion. All fail ConvertTritonGPUToLLVM with SmallVector<long> idx<size()
assertion / PassManager::run failed. Logs retain the generated MLIR reproducer.
The exact cause is unresolved. No compiler repair is claimed.

No successful candidate fingerprint exists. Ten native pairs, compositional
carry gates, long T512/T1024/T2048 masked/reset gates, 109 X31 and 108 operator
tests, launch profiling and ten-shape timing are **NOT RUN** for round7.
Round5 passing tests do not qualify this failed candidate.

## Analytic source accounting, not executed physical traffic

The prototype preserves the full-slot global update_value scans and READ;
launch fusion does not implement the intended page residency. At the round6
T1024/BS2048 fixture, its inherited control value scans request
2,686,451,712 bytes (1024 pre +257 enabled post scans, each 2048*128*8 bytes).
Scoring requests 536,870,912 key bytes; READ requests 1,073,741,824 value bytes.
These three terms sum to 4,297,064,448 bytes, 23.863x the conditional
180,075,776-byte/head payload. Score/product scratch adds six FP32 BS-vector
accesses per valid position: 50,331,648 bytes (score store, normalize load,
product store, READ load, post-control score and product loads). The subtotal
is 4,347,396,096 bytes, 24.142x. Metadata, carries, initialization, outputs,
inputs and dirty key writes are additional; this is an incomplete source
accounting subtotal, not a complete pass traffic measurement or HBM bound.
Compiler optimization could eliminate scratch traffic if compilation succeeds.

Per 128-slot score page: 32,768 key bytes plus 512 score-store bytes, excluding
query. Per 128-slot READ page: 65,536 value bytes +512 product bytes +1024 carry
store bytes, plus1024 carry-load bytes except the first page; final output512
bytes only on the last page. Full-store value update scans consume
2,097,152 bytes per enabled call at BS2048. The aspirational shared-page design
has no implemented-pass accounting or achieved gap yet. Hardware counters
remain unavailable; no permission-changing operation was attempted.

## 1.3B comparison: historical fallback only, no round7 projection

Using the independent round5 ten-shape receipt (five wrapper-inclusive samples),
the unchanged scenario is 2B bytes *88 native passes *batch-wall seconds /
(B*T) /86400. These are forward native-only extrapolations, with a nominal
second pass; actual dense/backward/optimizer/data/checkpoint work is excluded.
They are not new measurements or a certified training throughput.

|B|T|historical R5 batch wall ms|native-only projected days|
|---:|---:|---:|---:|
|1|32|6.228769|396.507286|
|1|64|11.824157|376.347590|
|1|96|16.539764|350.959498|
|1|128|22.081602|351.414384|
|1|129|22.203361|350.612936|
|1|257|43.636692|345.873766|
|1|512|86.837874|345.492120|
|1|1024|197.998854|393.877929|
|16|128|22.684822|22.563390|
|16|512|88.593347|22.029776|

B16 timings are whole batch timings, divided by16 in the projection. Even
B16/T512 is22.03 native-only days against the one-day budget. It is the
applicable historical batched comparison, not the B1 number. A round7 batched
registration number remains unavailable. The separate round6 B1/T1024 repeat
211.404172ms projects420.545days; do not mix that run with the R5 sweep.

## Resume and manual notes

This is an unresolved compiler failure, not a hardware-counter prerequisite or
a claimed runtime ceiling. Resume by minimizing the retained MLIR in
retry_console.txt and step_console.txt, preserving exact arithmetic; obtain a
compiling nonzero B1/T32 pair before larger compilation. Then implement actual
separate key/value shared-page staging and update source traffic accounting.
Run all standing gates before timing and promotion. Keep the design's explicit
normalization tree and two parity chains; no online-softmax reassociation.
The existing quick/retry/step scripts refuse receipt overwrite. Copy to a new
label/output path for subsequent attempts. Only approved venv Python is used.

All work serial/attended, exclusive broker physical4 only. Final physical4 idle
(2MiB/0%); lease release on wrapper EXIT. No training/checkpoint, no .pt commit,
no WG service/workers, no subagents, no protected-card workload, no sxgc or
/mnt/nvme2n1 access. Existing unrelated workspace changes preserved;
AGENTS.md/CLAUDE.md unchanged. No resilient training behavior changed, so
DiLoCo gap-matrix requirements are inapplicable. Tool discovery found no
contact_supervisor tool; no supervisor contact or approval is claimed.

Residual risks: compiler failure blocks all byte-exact and timing claims;
shared-page staging absent; source arithmetic is not physical traffic; full
trained-model and >=2M paired optimizer/RNG/checkpoint gates remain outstanding.
