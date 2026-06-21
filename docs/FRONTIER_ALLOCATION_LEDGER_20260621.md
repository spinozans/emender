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
| `DEBUG-MATRIX-20260621-RETRY` | `retry-frontier-debug` | `debug` | `failed` | `N/A` | `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md` | `4880725`, `4880730`, `4880747`, `4880875` | debug | 2026-06-21 | 2026-06-21T09:47:03 | 2026-06-21T10:06:06 | 1 per job | 00:30:00 per job | 2.000000 accepted / 1.500000 matrix rerun | 0.197224 elapsed | 20,000 | 4,928 | 19,999.802776 | e97-MLP, e97-linear-MLP, gdn2-MLP | `debug-smoke-20260621` | smoke DATA `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt`; smoke VAL_DATA `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt`; GDN2 checkout `/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2` commit `95709fc`; runtime `EMENDER_CONDA_ENV=base` with ROCm torch/triton/schedulefree/tiktoken/numpy importable | Observed: runtime/data/GDN2 were staged and jobs submitted sequentially under debug QOS. e97-MLP passed kernel smoke then failed on compute-node `p50k_base.tiktoken` download; e97-linear-MLP failed chunked-E97 parity/finiteness smoke; gdn2-MLP passed external GDN2 bf16 fwd/bwd preflight then failed on the same tokenizer download. No training throughput or loss curve measured. | `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`; `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z`; `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z`; `frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z`; `logs/frontier/debug/emender-smoke-4880875.out`; `logs/frontier/debug/emender-smoke-4880730.out`; `logs/frontier/debug/emender-smoke-4880747.out` | Fix shared tiktoken cache and rerun e97-MLP/gdn2-MLP to first loss/throughput; fix or disable e97-linear chunked ROCm path before extended readiness. |
| `CHECKPOINT-RESUME-20260621-GDN2-CKPT` | `verify-selected-frontier` with post-cache job from `stage-p50k-cache` | `debug` | `completed` | `N/A` | `docs/FRONTIER_CHECKPOINT_RESUME_VERIFY_20260621.md` | `4881374` | batch/debug | 2026-06-21 | 2026-06-21T15:20:27Z | 2026-06-21T15:25:59Z | 1 | 00:30:00 | 0.500000 | 0.092222 | 19,999.802776 | 4,928 | 19,999.710554 | gdn2-MLP | `debug-smoke-20260621-gdn2-ckpt` | smoke DATA `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt`; shared tokenizer cache `/lustre/orion/bif148/proj-shared/tiktoken_cache`; GDN2 checkout `/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2` | Observed: corrected post-cache GDN2 one-node run completed with fused GDN2 guard, finite training loss through step 262, global throughput around 34k-35k tok/s after warmup, peak memory 16,847 MB, checkpoint `checkpoint_step_000262_loss_0.0766.pt`, and `latest.pt`. Evidence only: this selects GDN2 for resume verification, not as a model-quality verdict. | `docs/FRONTIER_CHECKPOINT_RESUME_VERIFY_20260621.md`; `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z`; `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/logs/train.log`; `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/latest.pt` | Resume from `latest.pt` in `CHECKPOINT-RESUME-20260621-GDN2-RESUME`; do not extrapolate to scaleout without separate approval. |
| `CHECKPOINT-RESUME-20260621-GDN2-RESUME` | `verify-selected-frontier` | `debug` | `completed` | `N/A` | `docs/FRONTIER_CHECKPOINT_RESUME_VERIFY_20260621.md` | `4881380` | batch/debug | 2026-06-21 | 2026-06-21T15:27:07Z | 2026-06-21T15:28:28Z | 1 | 00:20:00 | 0.333333 | 0.022500 | 19,999.710554 | 4,928 | 19,999.688054 | gdn2-MLP | `debug-smoke-20260621-gdn2-resume` | resumed from `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/latest.pt` | Observed: one-node resume loaded checkpoint at step 262 and reached later finite losses at steps 263-267, including `step 263 | loss 0.0383` and final `FINAL_LOSS_LAST100: 0.0415`; resume job also wrote `checkpoint_step_000267_loss_0.0415.pt` and `latest.pt`. Hypothesis: this is sufficient to unblock a separate small scaleout checkpoint/restart design, subject to approval. | `docs/FRONTIER_CHECKPOINT_RESUME_VERIFY_20260621.md`; `frontier_runs/debug/20260621/gdn2-MLP/resume-20260621T152703Z/logs/train.log`; `frontier_runs/debug/20260621/gdn2-MLP/resume-20260621T152703Z/train/levelgdn2-mlp_100m_20260621_112753/latest.pt`; `logs/frontier/debug/gdn2-resume-4881380.out` | Record post-cache stage report separately; use this as checkpoint/resume evidence only. |
| `DEBUG-MATRIX-PLACEHOLDER` | `frontier-debug-benchmark-matrix` | `debug` | `planned` | `N/A` | `TBD` | `TBD` | debug | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | up to 1,000 | `TBD` | 20,000 | 4,928 | at least 19,000 | e97/e97-linear/gdn2 ROCm debug variants | `TBD` | ctx/model matrix from task | Pending downstream task. | `TBD` | Record actual debug rows and artifacts before extended candidates. |
| `DILOCO-8x4H-CANARY` | `frontier-diloco-scaleout-readiness` | `scaleout` | `candidate` | `BLOCKED_PENDING_APPROVAL` | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md` plus required WG human approval before submission | `TBD` | regular | `TBD` | `TBD` | `TBD` | 8 | 04:00:00 | 32 | `TBD` | 19,999.802776 | 4,928 | 19,967.802776 | ScheduleFree local-SGD/DiLoCo, `--diloco_island_size 8`, selected arm after debug retry | `canary-k250-beta0` | ctx2k, one-node islands, commapile, shared tokenizer cache required | Not submitted. Observed debug evidence currently blocks this row until tokenizer cache fix reaches first loss/throughput. Hypothesis: 8 one-node islands can expose launch/RCCL/merge behavior cheaply. | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`; `scripts/frontier/diloco_scaleout_readiness.sbatch` | Require debug gates, ledger confirmation, and explicit human approval before `sbatch`. |
| `DILOCO-8x8H-PILOT` | `frontier-diloco-scaleout-readiness` | `scaleout` | `planned` | `BLOCKED_PENDING_APPROVAL` | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md` plus required WG human approval before submission | `TBD` | regular | `TBD` | `TBD` | `TBD` | 8 | 08:00:00 | 64 | `TBD` | 19,967.802776 after canary | 4,928 | 19,903.802776 | ScheduleFree local-SGD/DiLoCo, `--diloco_island_size 8`, selected arm | `pilot-8n-k250-beta0` | ctx2k, one-node islands, commapile | Not submitted. Hypothesis: longer 8-node run tests checkpoint/restart and post-merge loss recovery after the 4h canary. | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`; `scripts/frontier/diloco_scaleout_readiness.sbatch` | Run only if 8x4h canary passes stopping rules and approval is refreshed. |
| `DILOCO-16x12H-PILOT` | `frontier-diloco-scaleout-readiness` | `scaleout` | `planned` | `BLOCKED_PENDING_APPROVAL` | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md` plus required WG human approval before submission | `TBD` | regular | `TBD` | `TBD` | `TBD` | 16 | 12:00:00 | 192 | `TBD` | 19,903.802776 after 8-node pilots | 4,928 | 19,711.802776 | ScheduleFree local-SGD/DiLoCo, `--diloco_island_size 8`, selected arm | `pilot-16n-k250-beta0` | ctx2k, 16 one-node islands, commapile | Not submitted. Hypothesis: 16 islands provide production-shaped launch/merge evidence without entering 64-node cost. | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`; `scripts/frontier/diloco_scaleout_readiness.sbatch` | Run only after 8-node evidence shows stable launch, finite loss, usable throughput, merge metrics, and checkpoint artifacts. |
| `DILOCO-32x12H-PILOT` | `frontier-diloco-scaleout-readiness` | `scaleout` | `planned` | `BLOCKED_PENDING_APPROVAL` | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md` plus required WG human approval before submission | `TBD` | regular/extended | `TBD` | `TBD` | `TBD` | 32 | 12:00:00 | 384 | `TBD` | 19,711.802776 after 16-node pilot | 4,928 | 19,327.802776 | ScheduleFree local-SGD/DiLoCo, `--diloco_island_size 8`, selected arm | `pilot-32n-k250-beta0` | ctx2k, 32 one-node islands, commapile | Not submitted. Hypothesis: 32 islands tests merge-latency growth and loss shock before any 64-node candidate. | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`; `scripts/frontier/diloco_scaleout_readiness.sbatch` | Run only after 16-node pilot passes and approval is refreshed. |
| `E97-64x24-01` | `frontier-extended-e97-readiness` | `extended-production` | `candidate` | `BLOCKED_PENDING_APPROVAL` | `TBD` | `TBD` | extended | `TBD` | `TBD` | `TBD` | 64 | 24h | 1,536 | `TBD` | 19,000 | 4,928 | 17,464 | e97 or e97-linear ROCm kernel variant | `TBD` | `TBD` | Not submitted. | `TBD` | Require debug evidence, exact launch package, and explicit approval. |
| `DILOCO-64x24-01` | `frontier-diloco-scaleout-readiness` | `extended-production` | `candidate` | `BLOCKED_PENDING_APPROVAL` | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md` plus explicit 64-node human approval artifact | `TBD` | extended | `TBD` | `TBD` | `TBD` | 64 | 24h | 1,536 | `TBD` | 19,327.802776 after 8/16/32-node ladder in this plan | 4,928 | 17,791.802776 | ScheduleFree local-SGD/DiLoCo, `--diloco_island_size 8`, selected arm | `candidate-64n-k250-beta0` | ctx2k, 64 one-node islands, commapile | Not submitted. Observed current debug evidence is insufficient; this row is blocked until 8/16/32-node scaleout evidence separates launch stability, throughput, optimizer/loss sanity, communication overhead, and cost. | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`; `scripts/frontier/diloco_scaleout_readiness.sbatch` | Require 8/16/32-node scaleout evidence, updated ledger accounting immediately before submission, and explicit human approval. |

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
