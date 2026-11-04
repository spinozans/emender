# End-to-end follow-up: redundant training oracle

The first registered v2 launch exposed another wrapper bottleneck that the
model-only profiles had excluded: oracle_shard rebuilt and rechecked every
260-scalar Python row on every shard. The run was intentionally stopped at
684 steps / 1,383,272 targets / 104.875 active seconds (6.522 steps/s), before
using it in any PB evaluation. Its unchanged raw receipts are retained under
../c14_b_v2_slow_decode_attempt/, with a checkpoint relocation manifest and SHA.
That attempt is not resumed or mixed into the final registered B1-v2 run.

A 32-shard decode profile covers 4,096 windows / 517,605 targets / 256
batch-16 step equivalents. Every identity, length, feature and context byte
matches the independent oracle. The instrumented Python oracle took 34.308
seconds; the native path took 0.150 seconds without cProfile. These asymmetric
instrumentation timings are diagnosis, not a claimed training speedup.

The final training adapter reuses the unchanged standing native feature decoder
and the existing raw-context decoder. Only unused Python row materialization
and redundant recomputation are removed. All shard hashes, source/document
hashes, identity checks, bounds checks, valid lengths, float32 feature bits,
raw contexts, dropout, optimizer and active-time rules remain. The independent
Python oracle is retained in the launch gates, including first/middle/last
full-shard bit identity. 82 tests passed on card 2 in 23.73 seconds after this
change. The gate hashes bind both native decoder sources and their adapters.

The final end-to-end 200-step profile reads 25 consecutive shards, matching
cache identities to verified document windows. It includes cache read/hash,
decode, assembly/transfer, forward, backward, update and telemetry aggregation.
Graph capture is included. Full timing breakdown and all 200 rows are in
pipeline_profile.json; pipeline_summary.json gives warm medians. This corrects
the model-only scope of the earlier profiles. No profiling model is saved or
selected for a verdict. A fresh seed-3401, 2700-active-second B1-v2 starts only
after this profile and the complete gates pass.

The first 237-step run remains untouched and excluded as the throughput-
confounded first attempt; first_attempt_manifest.json binds its receipt bytes.
There are no production store kernel changes, no frozen-arm changes, no
protocol changes and no supervisor decision needed at the kernel boundary.
The absent contact_supervisor tool prevents the requested progress notification;
no delivery or reply is claimed. Actual registered throughput and residual
exposure differences will be reported after completion, without extrapolating
from profile timings.
