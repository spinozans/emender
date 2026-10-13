# F8 staged persistent corrections — operator-gated, OFF

Sources are staged at
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/F8-staged/scripts`.
The live `scripts`, bank configuration, GPU leases and collection lane are NOT
modified. F7 and F9 staging remain separate. Do not deploy during collection.

At a separately approved relaunch deploy the complete F8 bundle, including
`rl_bank.py` (delegated-claim sweep protection), `rl_bank_lane.py`,
`rl_correction_service.py`, `rl_correction_pool.py`, `rl_correction_config.py`
and the unchanged F7 companion scripts/deadline adapter. Deploy the coordinator
against the same bank helper. A mixed old sweeper/new lane is unsafe: the old
sweeper could recycle a task already owned by the correction spool.

The existing bank `correction-config.json` is the only durable enablement gate.
Missing config means live-flash teacher, async OFF, one worker, 64 outstanding
corrections per lane. Existing three-key F7 files remain valid. Example only:

```json
{"teacher_model":"deepseek-4.1-flash-background","async_enabled":false,"pool_width":1,"backlog_capacity":64}
```

`pool_width` (1..12) means persistent correction workers PER LANE, not a global
semaphore. `backlog_capacity` (1..4096) bounds all admitted-but-unpublished jobs,
including queued, executing and ready results. Exhaustion immediately produces
an explicit failed correction; it never delays policy generation or silently
drops a task. Review ALL callers against the operator's 12 endpoint lanes before
an OFF-to-ON trial. No model flip or sustained throughput is qualified here.

## Ownership and recovery

* The CPU service survives collect/cycle/daemon exits. A lane owner starts or
  resumes it without waiting; a nonblocking lane-local flock prevents duplicate
  managers. Worker width is reread from the durable configuration.
* Each correction gets a copied, fsynced workspace/attempt snapshot and an
  immutable spec under `lanes/lane-NN/corrections/job-*` (use `lane_paths` for
  the exact spelling). A durable rename is admission. Attempt/grade/routing/
  served-checkpoint identities and owned-file hashes remain bound. Crashed-cycle
  archival cannot rename the worker's workspace. Unadmitted staging left by a
  dead admission owner is removed on service restart; live staging is untouched.
* **The approved 10s gate is at correction-EPISODE start, not enqueue.** The
  service persists start times across cycles/restarts and spaces episode starts;
  the collect path never sleeps for pacing or waits for a worker/result.
* A durable job owns the original task claim until publication/explicit failure.
  Pool TTL sweeping and failed-collect requeue skip that exact delegated task
  incarnation. They do not create a second correction for an admitted splice.
* Every episode retains a separate Python/Pi group, F7's 750s wall bound, and
  Linux parent-death containment. Ordinary worker crashes finalize explicitly
  and release worker slots. Service loss fails interrupted execution claims on
  restart rather than rerunning a possibly mutated splice. Queued admissions
  resume; partial API logs remain explicit. No background role loads a GPU.
* Only the lane owner publishes: its collect child polls ready work without
  waiting; the daemon also publishes before new cycles/while idle. The daemon
  never publishes concurrently with its synchronous collect child. The service
  itself cannot append receipts, change lineage or retire tasks.
* A durable publication intent precedes receipt append. Replay repairs the
  immutable-copy-only crash, recognizes the exact already-appended receipt,
  and finishes retirement once. New receipts use the CURRENT validated stream
  tail while retaining the ORIGINAL served checkpoint and originating cycle.
  Partial/corrupt stream tails fail closed and require attended repair; there
  is no truncation or guessed-tail recovery.
* Cycle summaries stay immutable and explicitly list pending job IDs; terminal
  GPU correction drain is zero. Final outcomes/API metrics/publication intents
  live in the job directories. Late receipts feed the unchanged receipt
  inventory/exposure rules on later collection/training opportunities. State
  tracks accounted publication IDs so owner restart does not lose late receipt
  counters. Published workspaces are retained evidence, not abandoned workers.

Turning async OFF prevents new durable admissions; already admitted work remains
owned and may finish/publish after OFF. OFF does not discard the backlog. The
old synchronous correction path remains for new work while OFF, as in F7.
Stopping the lane does not cancel the persistent correction service. At an
attended shutdown, stop that service separately; restart will explicitly fail
interrupted jobs, retain completed results and resume queued jobs. Never remove
its spool or recycle delegated claims as a shortcut.

**Conversation-async support is deferred to F9 deployment.** F8's pre-F9
runtime marks scripted conversation tasks explicitly ineligible/uncorrectable
and retires them failed for the existing retry policy; it never generates a
fake scripted attempt or splices a conversation. F9's driver/bridge are not
folded into this bundle. Requalify this seam after the real F9 runtime lands.

## Metrics / rollout gate

Cycle summary and cycle metrics expose backlog depth, queue/in-flight/ready
counts, per-correction wait times, oldest pending age, admitted/started/finalized/
native-closed counts, published receipt/target counts, generation attempts/h,
correction admissions/outcomes/h and backlog growth/h. Cycle rates share the
actual generation observation wall; lifetime rates explicitly name their
separate observation wall. Finalized failures are NOT native completions or
eligible receipts. Original summary pending IDs are historical snapshots: use
job publication records for current completion state.

The CPU tests prove no wait for a slow correction or an exhausted backlog, not
zero admission/publication filesystem cost or a sustained live GPU/API rate.
Disk persistence, receipt encoding, policy grading, unchanged GPU lease/training
boundaries and historical-spool metric scans still have costs. Retention and
mixed-workload/provider-cancellation qualification remain operator work. Leave
async OFF until independent review plus a controlled endpoint-budgeted trial.
