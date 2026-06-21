# Frontier DiLoCo 8x4h Canary Review - 2026-06-21

Task: `review-frontier-evidence`

## Decision

Decision: **REJECT `DILOCO-8x4H-CANARY` for submission now**.

This is a rejection of exactly one proposed Frontier canary:

| Candidate | Queue | Nodes | Walltime | Expected node-hours | Remaining before | Reserve held | Remaining after if later approved | Approval artifact |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `DILOCO-8x4H-CANARY` | `regular` or eligible short non-debug production queue | 8 | 04:00:00 | 32 | 19,999.802776 | 4,928 | 19,967.802776 | `docs/FRONTIER_DILOCO_8X4H_CANARY_REVIEW_20260621.md` |

The ledger arithmetic is unchanged from the DiLoCo readiness plan:

```text
8 nodes * 4 hours = 32 requested node-hours
19,999.802776 remaining before - 32 requested = 19,967.802776 remaining after
4,928 node-hours remain protected as reserve
```

No `sbatch` should be submitted from this review. The canary can be reconsidered
only after the follow-up evidence tasks below land artifacts and a new approval
artifact explicitly changes the status.

## Evidence Reviewed

Pre-cache one-node debug evidence exists from `retry-frontier-debug`:

| Variant | Job ID | Artifact links | Observed result |
| --- | --- | --- | --- |
| `e97-MLP` | `4880875` | `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`; `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z`; `logs/frontier/debug/emender-smoke-4880875.out` | Fused split-edit Triton smoke passed, but training failed before first loss/throughput because the tokenizer attempted a compute-node `p50k_base.tiktoken` download. |
| `gdn2-MLP` | `4880747` | `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`; `frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z`; `logs/frontier/debug/emender-smoke-4880747.out` | External GDN2 bf16 forward/backward preflight passed, but training hit the same tokenizer-cache failure before first loss/throughput. |
| `e97-linear-MLP` | `4880730` | `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`; `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z`; `logs/frontier/debug/emender-smoke-4880730.out` | Chunked-E97 ROCm parity/finiteness smoke failed before training and is not eligible for this canary. |
| Retry setup / sequencing | `4880725` | `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`; `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` | Recorded as part of the debug retry evidence set; the ledger group reports accepted debug jobs and consumed elapsed node-hours. |

Required post-cache evidence does **not** exist in the current repo state:

- No completed post-cache `p50k_base` cache report is present.
- No post-cache `e97-MLP` or `gdn2-MLP` one-node debug job id is recorded.
- No post-cache first finite training loss is recorded.
- No post-cache global tokens/sec or memory sample is recorded.
- No selected arm is justified from observed post-cache metrics.
- No selected-arm checkpoint write and `latest.pt` evidence is recorded.
- No selected-arm resume job reaches a later finite loss.

The upstream readiness docs require these gates before an 8+ node canary:

- `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md` requires shared
  tokenizer cache proof, finite one-node loss/throughput for the candidate arm,
  no silent eager fallback, checkpoint/validation evidence where enabled, ledger
  confirmation, and human approval.
- `docs/FRONTIER_EXTENDED_E97_READINESS_20260621.md` requires post-cache debug
  evidence and a debug or canary-scale checkpoint/resume test before larger
  allocation commitments.

Because the required post-cache and resume artifacts are absent, this review
does not select an arm for the canary. `gdn2-MLP` and `e97-MLP` remain
candidates for the post-cache debug retry; `e97-linear-MLP` remains blocked by
its ROCm parity/finiteness failure unless fixed or deliberately disabled.

## Rejection Rationale

The rejected canary would spend 32 node-hours and test the GPU-island DiLoCo
hypothesis, but the current evidence has not yet proven that the intended
training command can pass the tokenizer gate, reach one finite training metric,
emit usable throughput, or write and resume a checkpoint on Frontier.

Submitting 8 nodes now would therefore test too many unresolved conditions at
once: tokenizer cache, selected arm viability, fused/no-eager guard, checkpoint
I/O, resume semantics, rank mapping, DiLoCo merge behavior, RCCL behavior, and
throughput scaling. The readiness plan intentionally requires the cheap
one-node gates first.

## Follow-Up Evidence Links

The missing evidence is already represented by WG follow-up tasks:

| Missing evidence | WG task | Required outcome before reconsideration |
| --- | --- | --- |
| Shared `TIKTOKEN_CACHE_DIR` with `p50k_base.tiktoken` readable in compute-node context; post-cache `e97-MLP` and `gdn2-MLP` one-node debug job ids; first finite loss; throughput; memory; fused/no-eager guard; ledger rows. | `stage-p50k-cache` | Task reaches `done` with a committed report and ledger rows. |
| Selected arm justification from post-cache evidence; one-node checkpoint write; `latest.pt`; resume from checkpoint to later finite loss; queue/nodes/walltime/node-hour accounting. | `verify-selected-frontier` | Task reaches `done` with a committed checkpoint/resume evidence report and ledger rows. |
| Current synthesis of observed facts, risks, recommendations, blockers, and next-job table. | `frontier-results-synthesis` | Task reaches `done` and any blockers remain linked to concrete WG tasks. |
| `e97-linear-MLP` chunked ROCm parity/finiteness failure. | `fix-or-quarantine` | Either fix and requalify the variant or mark it out of scope for canary selection. |

## Future Approval Conditions

A later approval artifact may approve `DILOCO-8x4H-CANARY` only if it cites
completed artifacts for the two direct evidence gaps above and keeps the
success criteria and stopping rules from
`docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`:

- chosen arm has post-cache finite one-node loss and global tokens/sec;
- shared tokenizer cache and data paths are proven from compute-node context;
- fused/no-eager guard is recorded for the chosen arm;
- checkpoint write and resume-to-later-finite-loss evidence exists;
- ledger row shows queue, nodes, walltime, requested node-hours, remaining
  allocation before/after, reserve held, approval status, and artifact links;
- stop if launch fails before first training step, any rank has nonfinite
  loss/grad, no DiLoCo merge occurs by `K=250`, global tok/s is below 4x the
  one-node baseline after warmup, RCCL/NCCL errors occur, checkpoint/manifest is
  missing, or the requested ledger row is absent.

Until those conditions are met, `DILOCO-8x4H-CANARY` remains rejected for
submission.
