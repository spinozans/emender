# X43 C14 union extraction first-hours verification

Attended early snapshot of commit `36689fa5`, on lambda01. Final snapshot:
**2026-11-03T06:48:12.039973+00:00**, **57,600 windows /
7,284,742 positions**. The service remains enabled, active and
running with PID 1821314 and zero automatic restarts. No extraction check failed;
no stop, restart, tuning, GPU lease or training was performed by this verification.
This covers the first approximately 18 minutes of the current invocation, not
multiple hours of continuous monitoring.

## Health and storage

`before_*` and `after_*` retain systemd state, service journal, console, extraction
journal, host processes/pressure, stderr, UTC timestamps, and hash/byte manifests.
The console advanced from 3,656,110 to 7,284,742 positions between
these snapshots. System journal has only the current start; both service/error
journals show no failures, stderr is empty, NRestarts=0. Earlier service journal
starts are known manual installation restarts, not a current restart loop.

The host has 256 CPUs. Before the extra oracle probe, the load was below that
count but a separate native index benchmark and substantial memory pressure were
present. After the probe and external benchmark exited, short-term CPU and memory
pressure returned close to zero. `during_oracle_host.txt` and
`operational_notes.txt` receipt the diagnostic's own temporary load. The service
continued advancing throughout; no resource or integrity failure was observed.

Free disk is **802,693,865,472 bytes** (802.69 GB), versus the hard
**150,000,000,000-byte STOP line**. Committed cache bytes total
**856,216,169**. Growth since the original 19,968-window
launch snapshot is **466,045,698 bytes** for
4,760,046 targets (97.91 bytes/target).
The corresponding recent-tail union scenario is **137.86 GB**;
the cumulative-prefix scenario is **165.32 GB**.
Both are consistent with an approximate 159 GB planning estimate and ample floor
headroom; 159 GB is not an enforced cap or a proven full-stream upper bound.
Future content and unrelated writers remain risks. No files were deleted.

## Rate diagnosis and revised projection

**The 4,473.92/s launch slice is startup-biased, not evidence of a persistent
24% rate regression.** Its 46.8316-second denominator includes the first completed
shard at 17.4945 seconds. On the exact same journal slice, using completed endpoints
(window 18,432 to 19,968) gives **6,609.01 targets/s**. This removes startup plus the
first shard from both numerator and denominator; it does not subtract time while
keeping that shard's targets.

Across the longest interval free of this verifier's extra native process,
**1,701,834 targets / 303.200027 seconds =
5,612.91 targets/s**, windows 44,160 through
57,600. This is **4.61% below**
the accepted 5,884.18/s baseline. The earlier clean endpoint interval was 4,824.29/s
over 274.7146 seconds, with slower blocks overlapping external native index
benchmarking/memory pressure. `accounting.json` retains every nonoverlapping
2,048-window block, both clean intervals, and the full invocation endpoint rate.
"Clean" here means uninterrupted and outside this verifier's native load; it does
not mean controlled host isolation or identical document content.

The full invocation including startup and diagnostic contention achieved
**4,814.02/s**.
That observed loss is real and is retained rather than discarded. The recovery,
unchanged executable/selector bindings, and strong journal variation support
startup/contention/content variation rather than a demonstrated implementation
regression. Attribution to content versus contention is observational; no controlled
A/B or new optimization was authorized or performed. Baseline matching on the
remaining stream is not certified.

At the snapshot, 218,595,960 Emender targets remain.
Linear boundary scenarios on **2026-11-03 UTC** are:

- Longest clean observed rate: **17:37:17** (10.82 hours remaining).
- Full invocation including diagnostic load: **19:25:00**.
- Accepted baseline, as a comparison only: **17:07:21**.

The previous 17:04 projection is superseded by these explicitly conditional
scenarios. No one of them is a completion promise.

## Hash-bound parity and validation

`final_integrity.json` rehashes **all 450 committed shards** in the final
snapshot, checks ordered windows/counts/bytes, verifies both snapshot manifests,
and confirms each journal row's full compact round-trip and three exhaustive
suffix oracle receipts. A live pending write is excluded rather than mistaken
for an orphan or adopted as committed.

`parity.json` checks newly extracted shards at indices 19,968, 24,448 and 28,800
against an exhaustive native suffix oracle independent of production binary search
and query reuse: **three document-bounded windows, nine sampled positions, 579
native suffix queries**, exact sparse rows and all 260 float feature bits.
Shard, document, context, native executable, offset receipt and feature hashes are
retained. Samples use original hash-verified Pile text bytes, reset at their window
start and truncate at their document tail. `bindings.json` checks frozen stream
receipt hashes and source-index stat identity, and binds the native executable to
the frozen fixture. This verifier did not repeat the full 42 GB index hash; the
unchanged hash-bound service code performs it before each phase/start, as receipted
at launch. This is spot verification, not an independent full-corpus oracle proof.

Eight targeted service/document boundary tests passed. The program validator is
**green: 17 claims, 264 bindings**. No science claim, dense receipt, frozen split,
cache format, service implementation or rate policy changed.

## Reproduction and residual risks

From the repository root on lambda01:

```sh
.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_union_first_hours/verify.py before
.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_union_first_hours/verify.py parity
# After a clean observation interval:
.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_union_first_hours/verify.py after
.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_union_first_hours/analyze.py
.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_union_first_hours/verify.py integrity
.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.validate_program
```

The scripts write snapshots into this receipt directory; preserve the committed
receipts before re-running. A detected health/hash/parity failure stops the unit
before recording `FAILURE.json`. An initial one-worker oracle load was canceled
before queries, then retried with the established 128-worker setting; both loads
are excluded from primary clean accounting. See operational notes for exact PIDs,
times, and the temporary diagnostic contention.

Residual risks: this early observation is shorter than an hour; rates and byte
sizes remain document/host dependent; no persistent-regression claim is proved or
ruled out for future content; parity is sampled beyond journal self-checks; all
three caches remain incomplete and no phase boundary has arrived. Future health
or disk-floor failures must still fail closed. `contact_supervisor` is unavailable
in this session's enabled tool catalog; no supervisor response is implied.
Unrelated pre-existing working-tree changes were left untouched.
