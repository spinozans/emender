# E97/Emender DiLoCo Checkpoint Disk Usage Audit - 2026-06-23 17:03 UTC

Task: `audit-e97-diloco`

Scope inspected, read-only: `/mnt/nvme1n1/erikg/diloco_8gpu/emender`

Commands used only filesystem metadata and sizes: `find`, `du -sb`, `df -B1`, and `date`.
No live training process was inspected or touched, and no files under the live run tree were
modified, deleted, moved, or renamed.

## Summary

At the time sampled, the specified E97/Emender DiLoCo tree contained **42 checkpoint files**,
not approximately 183 checkpoint files. Every checkpoint file found by the checkpoint-like
search was a `checkpoint_step_*.pt` file under `runs/`.

The 42 checkpoints consume **324,226,286,244 bytes** (**324.23 GB**, **301.96 GiB**).
Each checkpoint is exactly **7,719,673,482 bytes** (**7.72 GB**, **7.19 GiB**), so average,
minimum, and maximum checkpoint size are the same.

The filesystem hosting the run, `/dev/nvme1n1` mounted at `/mnt/nvme1n1`, had
**839,528,574,976 bytes available** (**839.53 GB**, **781.95 GiB**) and was **95% used**.

This is not an immediate disk-full condition from the currently visible 42 checkpoints alone,
but it is operationally risky if dense 500-step checkpointing continues unattended. The active
segment is saving about one checkpoint every **8.6 minutes**, or about **50 GiB/hour**. With
about 782 GiB free, the filesystem has roughly **108 additional checkpoints** of headroom,
or about **15-16 hours** at the current save cadence, before checkpoint growth alone could
consume the remaining free space. Other users or jobs on the same filesystem would shorten
that window.

If there really were 183 checkpoints of this exact size in the target tree, they would consume
about **1.41 TB** (**1.29 TiB**) and would be too many for a filesystem already at 95% usage.
The current live target path does not show that count.

## Counts By Run Segment

| Run segment | Checkpoints | Step range | Checkpoint bytes | Directory bytes |
|---|---:|---:|---:|---:|
| `levelE97_100m_20260621_131958` | 1 | 500-500 | 7,719,673,482 | 7,719,676,276 |
| `levelE97_100m_20260621_133317` | 20 | 62,500-72,000 | 154,393,469,640 | 154,393,472,543 |
| `levelE97_100m_20260622_101547` | 1 | 72,500-72,500 | 7,719,673,482 | 7,719,676,385 |
| `levelE97_100m_20260623_103742` | 20 | 85,000-94,500 | 154,393,469,640 | 154,393,472,656 |
| **Total** | **42** | **500-94,500** | **324,226,286,244** | **324,226,297,860** |

Largest checkpoint directories:

1. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742`:
   154,393,472,656 bytes, 20 checkpoints.
2. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317`:
   154,393,472,543 bytes, 20 checkpoints.
3. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547`:
   7,719,676,385 bytes, 1 checkpoint.
4. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_131958`:
   7,719,676,276 bytes, 1 checkpoint.

The whole inspected E97/Emender tree was **324,227,377,644 bytes**, so nearly all of its
disk usage is checkpoint files.

## Checkpoint Size Statistics

| Metric | Bytes | Decimal GB | Binary GiB |
|---|---:|---:|---:|
| Total checkpoint bytes | 324,226,286,244 | 324.23 | 301.96 |
| Average checkpoint size | 7,719,673,482 | 7.72 | 7.19 |
| Minimum checkpoint size | 7,719,673,482 | 7.72 | 7.19 |
| Maximum checkpoint size | 7,719,673,482 | 7.72 | 7.19 |

## Filesystem Free Space

`df -B1 /mnt/nvme1n1/erikg/diloco_8gpu/emender` reported:

| Filesystem | Size bytes | Used bytes | Available bytes | Use |
|---|---:|---:|---:|---:|
| `/dev/nvme1n1` mounted on `/mnt/nvme1n1` | 15,360,854,417,408 | 14,521,325,842,432 | 839,528,574,976 | 95% |

Available space in checkpoint equivalents:

- `839,528,574,976 / 7,719,673,482 = 108.75` checkpoint files.
- Conservatively, that is **108 more full checkpoints** before this run alone would exhaust
  the currently reported free bytes.

## Save Cadence And Risk

Observed checkpoint mtimes in the active segment
`levelE97_100m_20260623_103742` run from step 85,000 at `2026-06-23 14:13:10 UTC`
through step 94,500 at `2026-06-23 16:56:36 UTC`.

That segment has 20 checkpoints over 19 intervals:

- Average interval: about **516 seconds** (**8.6 minutes**) per checkpoint.
- Growth rate: about **53.9 GB/hour decimal**, or **50.2 GiB/hour binary**.
- Daily rate if unchanged: about **1.29 TB/day decimal**, or **1.18 TiB/day binary**.

Risk assessment:

- **Current visible checkpoint count is not 183** in the specified live path; it is 42.
- **Current count is manageable but not benign** because the filesystem is already 95% full.
- **The cadence is the risk**. Dense saves every 500 steps create about 7.72 GB each.
- If dense saving continues without pruning or cadence changes, the free space window is
  roughly **15-16 hours** at the observed active-segment cadence, before allowing for any
  unrelated writes on `/mnt/nvme1n1`.
- Treat this as a near-term cleanup/cadence-management issue, not as a reason to interrupt
  the live training process abruptly.

## Safe Retention Guidance

Do not delete anything without explicit approval and a fresh re-scan immediately before
cleanup. Before any pruning, verify the training code's exact resume semantics, including
whether it resumes only from the latest file in the active segment or may need a specific
merge-aligned step.

Recommended preservation set:

- Keep all currently resume-critical recent checkpoints in the active segment, at minimum
  the latest several saves. At this sample time that means preserving the latest checkpoint
  `checkpoint_step_094500_loss_2.9042.pt` and preferably the preceding few checkpoints
  (`093000`, `093500`, `094000`) until a successful newer checkpoint is confirmed.
- Keep early proof / transition checkpoints explicitly called out by the task:
  `checkpoint_step_000500_loss_5.5811.pt`,
  `checkpoint_step_072000_loss_2.9556.pt`, and
  `checkpoint_step_072500_loss_2.9730.pt`.
- Keep periodic long-run milestones for traceability and rollback. A conservative pattern
  is every 5,000 or 10,000 steps, plus segment boundaries and the latest checkpoint from
  each run directory.
- In dense 500-step ranges, prune only redundant middle checkpoints after approval. For
  example, once newer checkpoints exist and resume safety is confirmed, the dense ranges
  `062500-071500` and `085000-092500` are the main candidates for thinning while keeping
  milestones and recent files.

Practical policy proposal:

1. Near-term, cap the active run to **latest 3-5 checkpoints plus every 5,000-step milestone**.
2. Long-term, keep **segment boundary checkpoints, the latest checkpoint per segment, early
   proof checkpoints, and every 10,000-step milestone**.
3. If disk pressure remains high, reduce save cadence or add a separate retention job that
   only runs after a checkpoint has completed and after a fresh `df`/`find` audit.

This policy would preserve resume safety and useful historical coverage while avoiding
unbounded 500-step checkpoint accumulation.

## Validation Checklist

- Checkpoint count and total size reported: **yes**.
- Filesystem free space reported: **yes**.
- Risk assessment provided: **yes**.
- Safe retention recommendation provided: **yes**.
- No files modified or deleted under `/mnt/nvme1n1/erikg/diloco_8gpu/emender`: **yes**.
