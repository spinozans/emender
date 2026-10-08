# F4 — Parallel collection for the single learner

## Implemented and default-off safety

Implemented the audit's "one learner with PARALLEL collection" change (task F4,
operator pre-approvals on record) on **lambda01**. Defaults preserve current
behavior exactly: no collector lanes unless `COLLECTOR_LANES` is set, the
learner's selection pool stays its own lane stream unless `BLOCK_MODE=on`
(whose new streams-root default is the bank's own lanes dir, so the single-lane
bank's union is just its own stream until collector lanes exist), and ledger
lines written without the streams flag are byte-identical to the pre-F4 format
(`source_lane` appears only on cross-stream lines). The running single-learner
bank was not restarted, signalled, or reconfigured; no live bank state, stream,
historical ledger, checkpoint, held-out evidence, coordinator, or supply
implementation was edited. Changed: the three owned deployed scripts plus
`run_bank.sh` (deployed and repository copies byte-identical), and this report.

A prior interrupted attempt had already drafted the `rl_build_pack.py` /
`rl_bank_lane.py` mechanics uncommitted in the repository; this task audited
them against the pre-approved design, fixed one patch-seam regression
(`load_selection_receipts` must keep the module-global `walk_stream` binding),
completed the collector/block accounting, added the tests, `run_bank.sh`
support, and the deployment.

## Cross-stream selection (rl_build_pack.py)

New optional `--streams-root <bank>/lanes`. When set (and window mode is
active; the flag is ignored otherwise), the selection pool is the UNION of the
`walk_stream`-verified receipts of every existing `lanes/lane-0N/receipts/
stream.jsonl`, each lane chain verified fail-closed before union. Non-numeric
or non-directory `lane-*` entries and lanes without a stream file are skipped;
a missing streams root fails closed. The union dedupes by `receipt_sha256`
(identical receipts across streams are impossible in practice but dedupe
anyway). Cross-stream FIFO orders by the verified `created_unix`, with
deterministic (numeric lane index, stream position) tiebreaks — the
supervisor-approved rule. The teacher-corrected-first priority of era-11 is
unchanged and applied across streams. Without the flag: exactly current
behavior. The consumed ledger stays at the LEARNER lane (`--workspace` root),
as today. `--streams-root` is bound by `rl_bank_lane.py` only in block mode
(`BLOCK_STREAMS_ROOT` env, default `$BANK/lanes` when `BLOCK_MODE=on`, `off` =
single-stream behavior unchanged).

## Ledger source accounting

With the streams flag active, every selected receipt carries a `source_lane`
annotation naming its originating stream. It is recorded in the packed
inventory ledger lines (`packed-receipts.jsonl`), the authority record
metadata (`records.jsonl`), and — through `_adopted_exposure_rows` — in the
E1 consumed ledger entries (`consumed-receipts.jsonl`), so consumption
accounting still records which stream/lane each adopted, sampled receipt came
from, and only sampled+adopted receipts advance. Without the flag, the
annotation key is absent and every ledger line is byte-identical to the
pre-F4 format (asserted by exact-string comparison in the tests).

## Collector lanes (rl_bank_lane.py)

`--collector-only` (env `COLLECTOR_ONLY`, default 0) makes a lane collect
only: an explicit early-out in the train stage skips the policy-gradient
channel, corpus anchors, and the receipts/block channel — no training,
probing, or adoption of any kind (not a threshold hack). The lane's lineage
stays at its init seed; no checkpoints are written by it; merge adoption and
block accounting are also skipped in the run loop. Collect/teacher
correction/receipt chaining are unchanged, so collector receipts are
receipt-eligible as usual (teacher-corrected and on-policy-success both feed
the learner's blocks through the union). Greppable lane event:
`LANE_COLLECTOR_ONLY lane=N cycle=C receipts=R (seed lineage retained;
training disabled)`.

## run_bank.sh (COLLECTOR_LANES)

New `COLLECTOR_LANES=N` env (default 0; unset = byte-identical current
behavior). With `N>0`: the script requires `LANES=1` (the single-learner bank;
it fails closed otherwise, since collector lanes are numbered 1..N by the
pre-approved single-learner geometry), seeds and launches lanes 1..N with
`COLLECTOR_ONLY=1` alongside the one learner lane (lane 0) exactly as today,
and the coordinator keeps `--lanes "$LANES"` = 1 so it watches only the
learner lane — collector lanes never enter the K-merge geometry or the
per-lane session reports (stale-claim sweeping is lane-count independent and
still covers collector claims). The preflight literal tracks the suite total:
now **138** (120 pre-existing + 18 F4 regressions).

## Validation

Final CPU suite from the deployed nvme copy: **138/138 checks** — all 120
pre-existing checks pass unchanged (one F3 patch seam was re-pointed from
`rl_receipts.walk_stream` to `rl_build_pack.walk_stream`; assertions and
coverage unchanged), plus 18 F4 checks in `test_parallel_collection`. Tests
build REAL digest-chained receipt streams for a learner lane and three
collector lanes (full `verify_receipt` re-encode), and cover: union
verification + decoy skipping, cross-stream FIFO with created_unix and
(lane, position) tiebreaks, sha dedupe, `source_lane` annotations,
teacher-first window ordering, byte-identical unflagged behavior (exact ledger
line strings, metadata and exposure key sets), missing-root fail-closed, block
inventory spanning the lanes root, a real end-to-end streams-root pack build
with the ledger at the learner lane only, adopted-exposure consumption with
source lanes exactly once per receipt, collector channel early-out (no
anchor/sft/block/pg/probe invocation, lineage and counters unchanged,
greppable event), collector metrics with no train row, and lane_run gates
(learner keeps merge polling + block accounting; collector never adopts merges
and never initializes block accounting). Exact command:

```bash
/home/erikg/emender/.venv/bin/python \
  /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts/rl_bank_unit_test.py \
  --scratch /home/erikg/emender-scratch/fix-fanout/F4-unit \
  > /home/erikg/emender-scratch/fix-fanout/F4-tests.log 2>&1
```

Retained log: `/home/erikg/emender-scratch/fix-fanout/F4-tests.log` (clean
138/138). All three Python scripts compile; `bash -n` accepts run_bank.sh;
`git diff --check` passes; deployed and repository copies compare byte for
byte. The E97 GPU subprocess and behavioral probe are stubbed in the unit
tests; this is not a GPU qualification or a wall-clock accumulation-rate
claim.

## Scope, risks, and next step

- No live-bank state was touched. The operator opts in by relaunching the bank
  with `COLLECTOR_LANES=N` (e.g. `LANES=1 BLOCK_MODE=on COLLECTOR_LANES=3
  bash scripts/run_bank.sh`); environment variables cannot mutate running
  processes, so the change is effective only on the next explicitly
  authorized launch.
- Collector lanes hold GPU leases around claimed cycles exactly like today's
  lanes (attempts need the policy model); on the shared box N collectors
  increase lease contention, not training. `--max-tasks` and cycle bounds are
  unchanged.
- Cross-lane consumption is still recorded by the learner's single-writer
  ledger; a learner crash mid-block retains F1's reconciliation limitation
  (checkpoint/sample-log reconciliation, not a new crash transaction).
- The wash-scale accumulation rate itself is not asserted here; the
  mechanism removes the single-lane collection bottleneck only. Recommended
  next step: operator-authorized relaunch with `COLLECTOR_LANES` and a
  separately approved accumulation-rate smoke check.
