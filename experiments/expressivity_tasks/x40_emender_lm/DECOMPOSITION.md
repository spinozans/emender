# Part 3: unchanged v1, measured before optimization

Broker card 1, RTX6000 Ada, B64/T512, two warmups, five updates, FP32,
ignition-v1. Median **8.030112 s**, 4080.64 tokens/s, peak allocated
30,957,720,576 bytes. X31 oracle adapter: 109 passed; all four production
X40 policy/wire/state/trace/gradient certifications pass.

| Profiled device component | ms/update | Activities |
|---|---:|---:|
| Persistent forward scan | 5240.911 | 24 |
| Journal-driven reverse | 2651.728 | 24 |
| Other forward activities | 8.943 | 672 |
| Other replay activities | 9.732 | 144 |
| Other device work (dense LM, CE, clipping, AdamW) | see full ledger | retained |

The 7.893 s scan+reverse device busy time explains 98.3% of the separate
8.030 s unprofiled wall sample numerically; profiling and wall samples are
separate, so this is attribution, not an additive wall identity. The gap is
inside recurrent kernels, not telemetry or fills. Synchronization runtime
3.610 s waits for already queued work; adding it to GPU time double counts.
9 inclusive sync runtime calls / 8 DtoH transfers include profiler teardown;
trace retains timestamps for boundary attribution. Six gate scalars, loss and
finite-norm checking cause steady host observation. No training trace_metrics.

Per-policy scan milliseconds in the first block: 30.878 / 93.874 / 203.666 /
543.958 (budgets64/256/736/1472). Same sequence recurs across all six blocks.
Reverse first block (execution reversed): 204.038 /169.489 /66.375 /27.178;
subsequent long-head reverse is ~174.3ms. The X36 43ms scan is a different
homogeneous decay regime; it cannot be multiplied by six as an equivalent
measurement of this four-policy composition.

The source already uses the X36 persistent per-chunk core, batched tiled
reads and all-field-journal replay. The named expensive mechanisms are
`advance_exact` from the last mutation for each live row/read/token and
`death_exact(1)` recomputed on each allocation; horizons are128/512/2048/8192.
The former repeats historical decay; the latter repeats a constant horizon.
Neither is a missing persistent-core port. Both can preserve exact FP32
semantics while avoiding repeated work. This diagnosis motivates those
specific changes; causal savings must be established by before/after parity
and measurements, not attributed to source counts alone.

Separate initial-bottom-block forward journal ablation (five CUDA-event
samples after two warmups):

| Budget | No journal ms | Journal ms | Increment estimate ms | Journal regs/spills |
|---|---:|---:|---:|---:|
|64|27.779|31.502|3.723|168/46|
|256|95.294|94.651|-0.643|244/0|
|736|204.490|204.496|0.006|230/0|
|1472|545.673|544.566|-1.107|230/0|

Negative estimates remain negative: different compiler specializations and
noise, not negative journal cost. Journal capture is not the multi-second
residual. Each block reserves 1,339,033,600 journal bytes across its heads;
actual per-store counts ~86,145–98,567, capacities233,472–282,624.
Full evidence: `receipts/part3_baseline/{wall,profile,ablations}.json`, lossless
`trace.json.gz`, oracle log/XML and parity receipt. Source hashes bind the
unchanged implementation. No performance fix preceded these receipts.
