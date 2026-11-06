# Phase5 round5: imported attention tiling, real1.3B NO-GO

**T1024 improves from 255.924206 to 191.888684ms/head
(25.02% less wall time, 1.333712x).
The native-only1.3B projection is 381.723005days: NO-GO.**
No production kernel changed and no training launched.

## Profile and the attention import

Design/algebra a8fb1ac5 committed/pushed before implementation; implementation
4b9aa02d committed/pushed before the complete validation battery. Read algebra
is scaled dot-product SOFTMAX over currently live slots, then strength-weighted
values. match_coefs/match_wire affect post-read write enable, not addressing.
The exact algebra and the correct online m,d,U equations are in DESIGN.md.
[FlashAttention](https://arxiv.org/html/2205.14135v2) supplies register-sized tiling
and streaming accumulators. Direct online-softmax rescaling changes native FP32
parenthesization/division placement, so it cannot simply replace the required
byte semantics. R5 imports tiles while preserving normalization's denominator
tree and the exact two parity accumulation chains. No full context x slots
matrix is materialized. Normalization's BS-vector scratch remains.

The early CUDA-event profile correctly selected READ, but included enqueue gaps.
Subsequent CUPTI profiling successfully measures actual kernel durations,
without claiming Nsight hardware spill counters or register-lifetime tracing:

|Stage|R4 kernel ms|R5 kernel ms|
|---|---:|---:|
|initialization|0.002336|0.002368|
|pre/post control,2048 calls|56.561090|56.662388|
|address scores,1024 calls|3.729571|3.728535|
|normalization,1024 calls|1.855219|1.856075|
|READ including gather/carries,2048 calls|189.907448|27.859426|
|sum of kernel durations|252.055664|90.108792|

READ fell6.82x in this instrumented comparison, from75.34% of device time
to30.92%; control is now62.88%. Uninstrumented matched wall time remains
191.888684ms. These are separate profiled and unprofiled iterations: their
difference is NOT a certified additive measurement of launch overhead. It does
show that further work must measure host enqueue/launch gaps and control rather
than assuming addressing matrix arithmetic dominates. Score/normalization are
only5.58ms of the R4 device profile. Carry/gather costs within READ are not
individually resolved. The profile receipts retain exact samples and counts.

R5 loads32x128 products into a register tile, exposes16 pairs by bit-preserving
reshape/permute/split, then appends pairs in exact increasing order. No tile
subtotal. PAGE1024 boundary stores/loads preserve FP32 carries; first pair
initializes without adding zero. T1024 READ42 registers/0 compiler spills versus
R4's28/0; increased tile registers fit without observed compiler spilling.
See COMPOSITION_PROOF.md for pair/tile/page/token induction and scope.

R4 partitions launches but does not implement PagedAttention page tables.
[Its source literature](https://arxiv.org/abs/2309.06180) addresses physical KV
allocation/sharing; no measured layout/fragmentation cost here justifies a table.
Context transitions cannot run as independent immutable KV queries: CORRECT,
expiry, array writes, POP and MERGE mutate earlier slots. The existing ordered
state machine is retained. No O(1) total metadata claim: scalar control and
normalization remain BS-wide.

## Sparse read decision

PB3 reports all sampled array effective weights nonzero (findability1.0), not
their mass. Its existing counters cannot establish top16/32 concentration.
Fresh synthetic snapshots at t256/512/1023 have top16 masses
.687718/.667267/.661266 and top32 .790909/.839760/.810231, with256/257/257
nonzero products. These are RANDOM-FIXTURE measurements, not trained PB3
statistics. There is no byte-exact pruning result. Trained top-k mass remains
unmeasured. Top-k is not chosen or implemented; any future variant requires
registration and quality predictions before training. Exact top-k selection
itself still scans occupancy unless an index is introduced.

## Ten independent monolithic pairs

Broker physical5, five synchronized allocation/wrapper-inclusive wall samples
per arm on identical nonzero inputs. All ten complete read/final physical state/
raw trace fingerprints and repeats match. Small shapes still regress strongly;
no unconditional selector is added. Timing varies across cards/runs; independent
sweep T1024 is197.998854ms versus the separate matched R4 comparison191.888684ms.
Compiler resource columns describe READ and post-control only.

|B|T|monolithic ms/head|R5 ms/head|READ regs/spills|post regs/spills|
|---:|---:|---:|---:|---:|---:|
|1|32|0.617261|6.228769|40 / 0|255 / 6|
|1|64|1.446592|11.824157|44 / 0|255 / 0|
|1|96|3.525755|16.539764|44 / 0|255 / 0|
|1|128|4.845097|22.081602|44 / 0|255 / 0|
|16|128|5.711568|22.684822|44 / 0|255 / 0|
|1|129|23.275513|22.203361|44 / 0|254 / 0|
|1|257|79.213763|43.636692|44 / 0|255 / 0|
|1|512|147.407901|86.837874|44 / 0|254 / 0|
|16|512|189.284201|88.593347|44 / 0|254 / 0|
|1|1024|714.436272|197.998854|42 / 0|255 / 26|

## Gates, failures and projection

- 109 X31 tests pass in15.57s;108 C14/operator tests pass in37.71s; no skips.
- Eight operation-rich short state pairs and four native/replay/gradient pairs
  pass full byte fingerprints and repeats.
- All24 independent matrix READ/previous-value adversarial pairs pass through
  BS4096: random, cancellation, signed zeros and NaN payloads.
- T512/T1024 unchanged monolithic long gates and T2048 compositional gate pass
  complete native/state/read/trace/replay/gradient fingerprints. All have peak257,
  seven masked-tail positions and distinct-document fresh-state reset equality.
- T2048 reference is PAGE256 decomposition plus the T1024 monolithic anchor and
  carry proof. No repaired unchanged T2048 LLVM compilation is claimed.
- Two profile harness failures (wrong positional index, then missing proxy
  attribute) and one candidate SliceLayout compilation failure are retained
  as exit1 with logs/source snapshot. Corrected final profile/reduction and all
  final gates exit0. No failed receipt qualifies a result.

At22 layers*2heads*2native passes, T1024 and2B raw-byte scenario targets,
0.016490434s/target, 60.641218targets/s,
381.723005native-only days. Required23148.148148targets/s,
<=.502691ms/head with every other cost free. Dense/replay/optimizer/source/
checkpoint cost is excluded: this is a lower-bound scenario, not training.
No150M/1.3B throughput or memory certification or eventual speed promise.

Residual risks: maximum-live operation-rich long cases; full trained-model
fingerprints; >=2M paired real-target optimizer/RNG/cursor/checkpoint-resume
identity; sustained full-width throughput/memory; BS-wide control; enqueue gaps;
trained top-k concentration. Diagnostic wrapper temporarily patches a shared
READ function under try/finally and assumes serial execution; not thread safe
or suitable for production installation. All are explicit non-promotion limits.

Manual notes: attended single-thread work, broker cards4/5 only, approved venv
Python only. No training/checkpoint, WG, subagents, protected-card workload,
corpus change, sxgc or /mnt/nvme2n1 access, or .pt commit. AGENTS.md/CLAUDE.md
unchanged. No resilient DiLoCo change; gap-matrix IDs inapplicable. Supervisor/
intercom discovery exposed no callable transport, so no contact is claimed.
Unrelated workspace changes preserved. Historical phase2/phase5/R2 source guards
match; source/battery hashes, terminal exits and exact resume are retained here.
