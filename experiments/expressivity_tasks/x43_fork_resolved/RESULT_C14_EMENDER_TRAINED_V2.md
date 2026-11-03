# C14 Emender consultation v2: feature bottleneck removed

| Arm | Steps | Actual target bytes | Active seconds | Steps/s |
| --- | ---: | ---: | ---: | ---: |
| Dense (frozen) | 111596 | 225880702 | 2700.022528 | 41.3 |
| Consultation v1 (historical) | 27401 | 55447993 | 2700.021958 | 10.148436 |
| Consultation v2 (primary) | 88520 | 179162804 | 2700.019260 | 32.784951 |

V2 is 3.231x faster than v1, within
1.260x of frozen dense. It consumed
79.32% of dense's targets. The primary-arm
amendment was committed before training in `e3149dd3`; v1 remains the honest first
attempt. The governing stopping-rule receipt remains unchanged. Actual steps and
targets remain first-class paired columns; no paired quality delta is claimed.

The exact 304-step v1-loop profile reproduced 10.206 steps/s: sparse decode
9.895s, Python assembly 7.248s, sampling 0.023s, source transfer 0.110s, feature
transfer/dropout 0.250s, training 12.252s. Native read/hash/decode took 0.303s;
training took 12.312s. Sequential file reads were negligible in the separate CPU
profile (0.016s). Dropout was already vectorized. The dominant cost was Python
sparse dictionary/entropy reconstruction followed by Python scalar assembly.

The shared native decoder preserves original branch order, double arithmetic,
and one final float32 conversion; fast-math and FP contraction are disabled.
All 4,864 prefix windows and 2,176 spread-check windows were float32 bit-identical
to the original Python path, including padding, with sparse and dense hashes.
After 304 optimizer updates, all 215 compared model/optimizer/RNG tensors and
next-window/target state match exactly. Existing extraction oracle attestations
remain hash-bound; no fresh native extractor oracle rerun is claimed. The early
shard-buffered exploratory profile retains its source instrumentation limitation
note; the exact v1-loop profile supersedes it for conclusions.

Final review also checked CPython 3.12 compensated float summation. An exhaustive
native audit of all 1,785,536 cached windows / 225,880,702 positions found zero
float32 entropy differences between the trained decoder and compensated sum
(76.773s, 22,962,125,375 sparse bytes rehashed/read). Production now uses the
compensated algorithm for unseen future shards and fails closed outside Python
3.12. This does not change any cached training input. Exact trained runner/C
sources are retained as training_runner.py and training_decoder.c, and their
hashes match the checkpoint config. The runner additionally supports a profile
label to isolate benchmark receipts/checkpoints; default protocol is unchanged.
See full_sum_audit.json and decoder_promotion.json. The algorithm follows
[CPython 3.12.3](https://raw.githubusercontent.com/python/cpython/v3.12.3/Python/bltinmodule.c).

The full v2 run charged native read/hash/decode 58.241s,
sampling/transfer 28.219s,
feature transfer/dropout 42.752s,
and training 2569.479s to active time.
Unexplained transient training delays remain charged; see training_delays.json.
Only checkpoint save/hash stalls (1.901667s) were excluded and itemized.
Recorded runner wall was 2722.966710s, starting inside main after imports
and decoder loading, including preflight. Lease acquisition, earlier profiling,
compiler setup and historical extraction are outside that wall measurement.

Protocol is unchanged: seed/dropout seed 3401, batch16/context128, BF16,
AdamW 3e-4/decay .01, clip1, 2,133,700 parameters, independent document/window
resets, masked tails, keep=.5 jointly on all 260 scalars without rescaling.
Card 2 was acquired using gpu_lease.sh with --no-wait. Every consumed shard was
rehashed and every source window identity/length checked. Launch rehashed all
13,950 immutable shards. No dense corpus copy was written: dense buffers are
17,039,360 bytes per shard (at most two during replacement). The 150 GB free
stop guard remained satisfied; resource_receipt.json retains byte counts.

Checkpoint outside git: `/home/erikg/c14_dense_bytes/emender_consultation_v2_resume.pt`; SHA256
`6b12a443249f0464808ba3d52274d373902ad43ea40913ce347cdf1c1af9811b`. Exact model, optimizer, CPU/CUDA/dropout RNG,
next window 1416320, steps, targets and active time are saved;
no automatic resume interface is claimed. Independent validation confirms finite
model/optimizer tensors, accounting, source attributes, retained training-source hashes, unchanged
v1 checkpoint and unchanged v1/dense receipts. All 35 C14 tests pass. The initial
new-test import failure and its correction remain in validation_history.json.

Transformer exposure is 972,375,733 versus 225,880,702 targets (4.305x).
transformer_exposure.json binds its future training requirement to this same
shared decoder for the Emender prefix and transformer continuation. Its actual
launch integration and throughput remain unverified: no transformer training
runner exists yet. The running union extraction service, ~/sxgc, and /mnt/nvme2n1
were not modified. Paired quality evaluation remains pending. No resilient DiLoCo
behavior was changed, tested or run; gap-matrix requirements do not apply.

The hardened-path 1,024-step benchmark measured 25.391 steps/s including startup,
and 29.940 steps/s over steps 100–1000. Both rates are retained; the full-budget
primary run above used the archived decoder with identical cached input bytes.
The bounded 4,096-step hardened-path benchmark completed in 144.611 active
seconds at 28.324 steps/s including startup and all delays, within 1.458x of
dense. Its reported losses/targets match the primary run at every logged step.
The exact audited C source is retained in audited_decoder_source.json; the final
production source differs only by removal of trailing whitespace. This
source-equivalence check is included in validation.

Release proof also confirms the normalized and audited compiled shared libraries
are byte-identical. Final validation and all 35 C14 tests pass.
