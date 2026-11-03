# C14 Emender cache closure — training blocked by unresolved protocol conflict

The Emender boundary at 2026-11-03 17:25:16 UTC is authentic. Independent
streaming SHA256 checks of every committed shard pass: **13,950 shards,
1,785,536 windows (0..1,785,535), 225,880,702 targets, 22,962,125,375 bytes**.
The journal SHA256 is
`3ae1aadac6688019d7c7195ce80e7163a7483ca456a8b67adda006f4c79895cb`,
matching the boundary receipt exactly. All shard identities, contiguous indices,
file sizes and hashes, full compact-row parity attestations, and three exhaustive
suffix-sample attestations per shard pass. These are audited extraction
attestations, not a fresh native-oracle rerun. Frozen dense receipt hashes pass.

The work order's 13,699 shards / 1,752,576 windows does **not** match this cache.
The actual totals agree with the frozen stream audit and phase-complete receipt;
the cache must not be truncated to the work order's smaller cardinality.

Training has **not** started. The hash-bound
`receipts/c14_registered_scale/stream_audit.json` still says
`blocked_scope_conflict`: the dense Emender used 111,596 steps in 2700.022528
seconds. Exact replay of that prefix and stopping after exactly 2700 active
seconds are separate stopping rules; feature assembly/dropout changes throughput.
This task asks for both without resolving the recorded conflict. No alternate
budget, timer exclusion, shorter prefix, or changed dense control is authorized
here. `contact_supervisor` is absent from the enabled tool catalog (searched
names and descriptions); a supervisor decision could not be requested. No GPU
lease was taken, no checkpoint was created, and no training completion is claimed.
The requested trained commit title is reserved for actual completed training.

Resume after the stopping-rule decision: implement consultation training from
seed 3401, batch 16, context 128, BF16, AdamW lr 3e-4 / decay .01, clip 1,
2,133,700 trainable parameters, all real targets and independent document/window
resets; attach the immutable cache in source order and apply independent
per-position keep=.5 to **all** 260 feature scalars, without rescaling. Acquire
only card 2 or 4 through `scripts/gpu_lease.sh`. Dense controls remain frozen.
Commit curves/config/seeds/cache/checkpoint hash immediately after actual
training. Checkpoints remain outside git. No consultation training implementation
or throughput measurement exists in this receipt.

Paired evaluation is pending the live `registered_eval/` phase. Evaluate only the
frozen 3,431,242 windows / 434,204,084 targets, windows SHA256
`2ab51ce59ad94010d78f6817f29fa06fc5ee64feea766376ff0e35347f196d4d`.
At 17:33:05 UTC the journal contained 21,760 committed windows / 2,754,837
targets. Initial observed rate 6503.91 targets/s projects November 4 11:58 UTC;
the accepted 5884.18/s baseline projects November 4 13:55:54 UTC from the actual
17:26:01 phase start. Both are scenarios, not bounds; the earlier ~13:30
projection is superseded by this explicit start-time calculation. No evaluation
was run, and the service was neither stopped, restarted, nor reconfigured.

## OOM diagnosis and monitoring plan

Systemd confirms OOM kills at 15:17:07 and 15:57:46, followed by automatic
restarts. Failed-instance memory peaks were 273.3 GiB and 308.5 GiB. Current
instance PID 3822378 is active, NRestarts=2; current native child PID 4129373
serves with `-j 128`, 258 native threads. The Python parent is ~350 MiB RSS,
while native RSS is ~289 million KiB (~276 GiB). A smaps snapshot attributes
~260 million KiB to anonymous memory and ~29 million KiB to file-backed pages.
Service cgroup memory is ~310 GB, including ~266 GB anonymous / ~43 GB file.
This is predominantly native resident storage, not Python journal growth alone.

Native source evidence shows shared index run columns materialized at open
(13 bytes/run plus a row directory), a small counter-only per-worker context,
and bounded queues (jobs*4 permits including the reorder buffer). Thus 128
workers amplify concurrent request/scratch demand, but the code does not support
a claim that every worker owns a full index or that an unbounded request queue
caused these kills. The two observed failed peaks and short current snapshots
are insufficient to prove a leak or establish a per-hour growth rate. Lowering
-j may reduce scratch/thread costs; it does not eliminate the large shared index.
It requires restart and is **not applied** to the ongoing eval phase.

Concurrent scatter_sort/xsa_finish processes each held roughly 11–15 GB RSS.
Host available memory was ~630 GB at the initial snapshot, but swap was nearly
full (8.59 GB of 8.59 GB). Concurrent build I/O and index mmap pages can add
page-cache/reclaim pressure. Available-memory and current samples cannot
reconstruct pressure at the two kills. Kernel OOM victim-selection logs are
not readable by this account; the exact triggering allocation/global-versus-
ancestor limit is unproven. Current service and inspected ancestor memory.max
and memory.high are unlimited. Ancestor oom_kill counters include historical
and unrelated events and must not be attributed solely to this service.

StartLimitBurst=3 / StartLimitInterval=1h is a live availability risk: a future
cluster of starts within an hour can suppress automatic restart. NRestarts=2 is
lifetime state, not evidence that two start-limit slots remain consumed now.
Do not reset counters or reload/restart this service to mask the risk. No cheap
memory mutation is justified by current evidence; no cache dropping, resource
limit, OOM-score change, or concurrent-workload intervention was applied.

Monitoring plan (documented, not an unattended monitor installed): every five
minutes retain UTC time, systemctl ActiveState/MainPID/NRestarts/MemoryCurrent/
MemoryPeak, cgroup memory.stat and memory.events, host MemAvailable/SwapFree,
top RSS processes, and the last committed eval shard. Compare anonymous versus
file growth and native versus Python RSS. Escalate immediately on a restart,
inactive/failed service, increasing oom_kill, no shard advance over 15 minutes,
or MemAvailable below 150 GB; preserve user journal and crash remnants before
any authorized recovery. Track recent start timestamps over the rolling hour.
Reproject completion from committed targets periodically. Do not kill or restart
the service as a monitoring action. Post-eval, benchmark lower -j with fixed
windows before proposing a future setting.

Evidence is in `receipts/c14_emender_closed/`. The cache audit is green;
**training and full task acceptance remain blocked**, with exact data resume
state bound above. No resilient DiLoCo behavior was changed, tested or run;
DiLoCo gap-matrix requirements do not apply to this C14 cache-only audit.
