# Frontier Allocation and Experiment Ledger

Task: `frontier-allocation-ledger`  
Created: 2026-06-21  
Allocation window: through 2026-09-01  
Total allocation: 20,000 Frontier node-hours

This ledger is the source of truth for planned and actual Frontier node-hour
spend for the ROCm, e97, GDN-2, and DiLoCo scaleout workstream. Later WG
workers should update this file before and after every Frontier submission.

## Accounting Rules

- Requested node-hours = `nodes * requested walltime hours`.
- Consumed node-hours should use scheduler-reported charged time when
  available. Until then, enter `TBD` and keep requested node-hours in the plan.
- Allocation remaining before/after must use consumed node-hours when known;
  otherwise use requested node-hours as the conservative debit.
- Extended-queue work is not approved by this document. Every extended-queue
  candidate must show requested spend, remaining allocation before/after, the
  current reserve, and an explicit approval record before submission.
- Keep a conservative failed-job and rerun reserve until the final launch
  window. The initial reserve is 4,928 node-hours, or 24.6% of the allocation.
- Debug, scaleout, and extended production spends must remain distinct in the
  `Spend class` field.

## Initial Budget Proposal

This plan accounts for the full 20,000 node-hours and keeps substantial reserve
for failed jobs, reruns, scheduler loss, and late-stage validation before the
2026-09-01 deadline.

| Budget bucket | Spend class | Planned node-hours | Notes |
| --- | --- | ---: | --- |
| ROCm/debug validation | `debug` | 1,000 | Cheap debug-queue and small-node correctness, throughput, and launch validation for e97, e97-linear, GDN-2, kernels, data, logging, and checkpoint paths. |
| DiLoCo scaleout pilots | `scaleout` | 2,320 | 8/16/32-node pilots and one partial retry pool before any 64-node DiLoCo production run. |
| e97 64x24h production candidates | `extended-production` | 6,144 | Capacity for four 64-node x 24-hour e97/e97-linear runs after debug validation and approval. |
| DiLoCo 64x24h production candidates | `extended-production` | 4,608 | Capacity for three 64-node x 24-hour hierarchical local-SGD/DiLoCo runs after scaleout validation and approval. |
| Integration, I/O, checkpoint, short rerun overhead | `scaleout` | 1,000 | Data staging verification, restart tests, failed-node recovery tests, short resume checks, and checkpoint inspection. |
| Failed-job and rerun reserve | `reserve` | 4,928 | Held back unless explicitly released. This covers about three 64x24h jobs plus 320 node-hours. |
| **Total** |  | **20,000** | Must not exceed allocation through 2026-09-01. |

Initial spend posture:

- Treat the first 1,000 node-hours as the debug-validation ceiling.
- Do not start extended production until debug validation records job ids,
  artifacts, rough throughput, loss sanity, checkpoint path, and known issues.
- Keep at least 4,928 node-hours reserved until a human explicitly releases
  reserve in a WG message, task description, or committed approval artifact.

## Experiment Unit Costs

| Experiment shape | Spend class | Nodes | Walltime | Requested node-hours | Use |
| --- | --- | ---: | ---: | ---: | --- |
| Debug smoke | `debug` | 1 | 2h | 2 | Launch/env smoke, import tests, short train step. |
| Debug model matrix entry | `debug` | 1 | 6h | 6 | Per model/kernel variant sanity run. |
| Small scaleout smoke | `scaleout` | 8 | 4h | 32 | First distributed launch and NCCL/RCCL wiring check. |
| 8-node DiLoCo pilot | `scaleout` | 8 | 8h | 64 | Cheap island behavior and checkpoint/merge validation. |
| 16-node DiLoCo pilot | `scaleout` | 16 | 12h | 192 | Production-shaped pilot from `SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`. |
| 32-node DiLoCo pilot | `scaleout` | 32 | 12h | 384 | Intermediate scaling point before 64 nodes. |
| 64-node e97 production run | `extended-production` | 64 | 24h | 1,536 | Full e97/e97-linear candidate run. Approval required. |
| 64-node DiLoCo production run | `extended-production` | 64 | 24h | 1,536 | Full hierarchical local-SGD/DiLoCo candidate. Approval required. |

## Capacity View After Debug Validation

Assumption for this view: the full 1,000 node-hour debug ceiling has been
spent, leaving 19,000 node-hours. If actual debug spend is lower, replace
19,000 with the current `Allocation remaining before` from the run ledger.

With a 4,928 node-hour reserve held back, the post-debug spendable balance is
14,072 node-hours. At 1,536 node-hours each, that can fit nine 64-node x
24-hour jobs while still preserving the reserve:

```text
floor((19,000 - 4,928) / 1,536) = 9
```

The initial budget assigns seven of those nine slots: four e97 production
candidates and three DiLoCo production candidates. The remaining spendable
headroom is 3,320 node-hours for scaleout pilots, short reruns, and measured
overhead before reserve release.

If a later worker is comparing only one experiment shape at a time after debug
validation, use this fit table. It assumes 19,000 node-hours remain and the
4,928 node-hour reserve is still protected.

| Shape after debug validation | Nodes | Walltime | Requested node-hours each | Max count while preserving reserve | Initial proposed count |
| --- | ---: | ---: | ---: | ---: | ---: |
| e97/e97-linear production | 64 | 24h | 1,536 | 9 | 4 |
| DiLoCo scaleout | 8 | 24h | 192 | 73 | 0-2 pilots, then reassess |
| DiLoCo scaleout | 16 | 24h | 384 | 36 | 1-2 pilots, then reassess |
| DiLoCo scaleout | 32 | 24h | 768 | 18 | 1 pilot, then reassess |
| DiLoCo production | 64 | 24h | 1,536 | 9 | 3 |

The max counts are not recommendations to run that many jobs. They are the
upper bound for a single-shape plan before accounting for already-submitted
jobs, failed-job reserve releases, and deadline risk.

## Extended-Queue Candidate Gate

These are capacity candidates, not launch approvals. A worker preparing any
extended-queue submission must update `Approval status`, `Approval artifact`,
and the run ledger row before submission.

| Candidate ID | Spend class | Queue | Nodes | Walltime | Requested node-hours | Remaining before | Reserve held | Remaining after | Approval status | Approval artifact | Preconditions |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `E97-64x24-01` | `extended-production` | extended | 64 | 24h | 1,536 | 19,000 | 4,928 | 17,464 | `BLOCKED_PENDING_APPROVAL` | `TBD` | Debug validation complete; e97/e97-linear chosen with artifact links. |
| `E97-64x24-02` | `extended-production` | extended | 64 | 24h | 1,536 | 17,464 | 4,928 | 15,928 | `BLOCKED_PENDING_APPROVAL` | `TBD` | Candidate 01 outcome reviewed; no unresolved correctness issue. |
| `E97-64x24-03` | `extended-production` | extended | 64 | 24h | 1,536 | 15,928 | 4,928 | 14,392 | `BLOCKED_PENDING_APPROVAL` | `TBD` | Updated budget confirms reserve remains intact. |
| `E97-64x24-04` | `extended-production` | extended | 64 | 24h | 1,536 | 14,392 | 4,928 | 12,856 | `BLOCKED_PENDING_APPROVAL` | `TBD` | Prior e97 result justifies another seed, variant, or rerun. |
| `DILOCO-64x24-01` | `extended-production` | extended | 64 | 24h | 1,536 | 12,856 | 4,928 | 11,320 | `BLOCKED_PENDING_APPROVAL` | `TBD` | 8/16/32-node scaleout evidence complete. |
| `DILOCO-64x24-02` | `extended-production` | extended | 64 | 24h | 1,536 | 11,320 | 4,928 | 9,784 | `BLOCKED_PENDING_APPROVAL` | `TBD` | Candidate 01 reviewed; merge/checkpoint behavior acceptable. |
| `DILOCO-64x24-03` | `extended-production` | extended | 64 | 24h | 1,536 | 9,784 | 4,928 | 8,248 | `BLOCKED_PENDING_APPROVAL` | `TBD` | Updated ledger confirms no higher-priority rerun needs the slot. |

## Run Ledger

Use one row per submitted job or coherent job group. For job arrays or a WG
task that launches multiple jobs, add one row per scheduler job id unless the
jobs are inseparable for accounting.

| Run ID | WG task | Spend class | Status | Approval status | Approval artifact | Job ID(s) | Queue | Submit date UTC | Start UTC | End UTC | Nodes | Walltime requested | Node-hours requested | Node-hours consumed | Allocation remaining before | Reserve held | Allocation remaining after | Model/kernel variant | Seed/run ID | Data/context | Outcome | Artifact links | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `FRONTIER-LEDGER-000` | `frontier-allocation-ledger` | `reserve` | `planned` | `N/A` | `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` | `N/A` | `N/A` | 2026-06-21 | `N/A` | `N/A` | 0 | 0h | 0 | 0 | 20,000 | 4,928 | 20,000 | ledger | `N/A` | `N/A` | Ledger initialized. | `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` | Later workers add actual job rows before submitting. |
| `DEBUG-MATRIX-20260621-PREFLIGHT` | `frontier-debug-benchmark-matrix` | `debug` | `cancelled` | `N/A` | `docs/FRONTIER_DEBUG_BENCHMARK_MATRIX_20260621.md` | `not submitted` | debug | 2026-06-21 | `N/A` | `N/A` | 1 planned | 00:30:00 planned per variant | 1.500000 planned retry / 0 submitted | 0.000000 | 20,000 | 4,928 | 20,000 | e97-MLP, e97-linear-MLP, gdn2-MLP | `preflight` | canonical commapile present at `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`; smoke DATA/VAL_DATA env paths absent | Observed: scheduler available and no current user debug jobs, but ROCm Python runtime, smoke data paths, and GDN2 checkout were missing; no jobs submitted, no throughput/memory/loss evidence. Hypothesis: variants remain untested rather than failed. | `docs/FRONTIER_DEBUG_BENCHMARK_MATRIX_20260621.md`; `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/env.txt`; `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/manifest.json` | Retry debug after staging ROCm Python env, smoke data or explicit data plan, and `GDN2_PATH`; do not prepare extended launch from this evidence. |
| `DEBUG-MATRIX-PLACEHOLDER` | `frontier-debug-benchmark-matrix` | `debug` | `planned` | `N/A` | `TBD` | `TBD` | debug | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | up to 1,000 | `TBD` | 20,000 | 4,928 | at least 19,000 | e97/e97-linear/gdn2 ROCm debug variants | `TBD` | ctx/model matrix from task | Pending downstream task. | `TBD` | Record actual debug rows and artifacts before extended candidates. |
| `SCALEOUT-DILOCO-PLACEHOLDER` | `frontier-diloco-scaleout-readiness` | `scaleout` | `planned` | `N/A` | `TBD` | `TBD` | regular/extended | `TBD` | `TBD` | `TBD` | 8/16/32 | `TBD` | up to 2,320 | `TBD` | `TBD` | 4,928 | `TBD` | ScheduleFree local-SGD/DiLoCo | `TBD` | ctx2k, one-node islands | Pending downstream task. | `TBD` | Fill concrete pilot rows before 64-node DiLoCo approval. |
| `E97-64x24-01` | `frontier-extended-e97-readiness` | `extended-production` | `candidate` | `BLOCKED_PENDING_APPROVAL` | `TBD` | `TBD` | extended | `TBD` | `TBD` | `TBD` | 64 | 24h | 1,536 | `TBD` | 19,000 | 4,928 | 17,464 | e97 or e97-linear ROCm kernel variant | `TBD` | `TBD` | Not submitted. | `TBD` | Require debug evidence, exact launch package, and explicit approval. |
| `DILOCO-64x24-01` | `frontier-diloco-scaleout-readiness` | `extended-production` | `candidate` | `BLOCKED_PENDING_APPROVAL` | `TBD` | `TBD` | extended | `TBD` | `TBD` | `TBD` | 64 | 24h | 1,536 | `TBD` | 12,856 | 4,928 | 11,320 | ScheduleFree local-SGD/DiLoCo | `TBD` | ctx2k, 64 one-node islands | Not submitted. | `TBD` | Require 8/16/32-node scaleout evidence and explicit approval. |

Allowed values:

- `Spend class`: `debug`, `scaleout`, `extended-production`, `reserve`.
- `Status`: `planned`, `candidate`, `submitted`, `running`, `completed`,
  `failed`, `cancelled`, `superseded`.
- `Approval status`: `N/A`, `BLOCKED_PENDING_APPROVAL`, `APPROVED`,
  `REJECTED`, `SUPERSEDED`.
- `Outcome`: short neutral statement separating observation from hypothesis,
  for example `Observed: loss finite through 2h; Hypothesis: suitable for
  16-node pilot; Risk: checkpoint restore not tested`.

## Usage Note for Later Workers

Before submitting a Frontier job:

1. Add a new row to `Run Ledger`.
2. Fill `WG task`, `Spend class`, `Queue`, `Nodes`, `Walltime requested`,
   `Node-hours requested`, `Allocation remaining before`, `Reserve held`,
   `Allocation remaining after`, `Model/kernel variant`, `Seed/run ID`, and
   planned `Artifact links`.
3. If `Queue` is `extended` or `Spend class` is `extended-production`, leave
   `Status=submitted` blocked until `Approval status=APPROVED` and
   `Approval artifact` points to a WG message, task artifact, or committed
   launch approval.
4. Commit the ledger update or record it as a WG artifact before launch.

After the job finishes or fails:

1. Replace `Job ID(s)`, `Start UTC`, `End UTC`, and `Node-hours consumed` with
   scheduler evidence.
2. Recompute `Allocation remaining after` using consumed node-hours if known;
   otherwise keep requested node-hours as the conservative debit.
3. Link artifacts directly in `Artifact links`: launch script, stdout/stderr,
   scheduler accounting output, environment capture, git commit, checkpoints,
   throughput/loss metrics, and any generated report.
4. Set `Outcome` to a concise observed result and set `Next action` to the
   exact downstream decision, such as `rerun debug with fixed RCCL env`,
   `prepare 16-node pilot`, or `do not proceed to extended queue`.
5. Run `wg artifact <task-id> docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` for
   the task that updated the ledger.

## Validation Checklist

- Ledger fields include job id, queue, nodes, walltime, node-hours requested,
  node-hours consumed, model variant, seed/run id, outcome, and artifact links.
- The initial plan accounts for all 20,000 node-hours through 2026-09-01.
- Debug, scaleout, extended production, and reserve spends are distinct.
- Extended-queue candidates show requested spend, remaining allocation
  before/after, reserve, and approval status.
- Later workers can add rows by following the schema and usage note without
  inferring hidden fields.
