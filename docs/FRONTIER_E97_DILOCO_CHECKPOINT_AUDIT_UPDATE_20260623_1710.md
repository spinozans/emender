# E97/Emender DiLoCo Checkpoint Audit Update - 2026-06-23 17:10 UTC

Task: `audit-e97-diloco`

Scope inspected, read-only: `/mnt/nvme1n1/erikg/diloco_8gpu/emender`

This update refreshes the earlier `2026-06-23 17:03 UTC` audit after the task was
reopened for a numeric WG log line. The scan used only filesystem metadata/size
commands: `find`, `du -sb`, `df -B1`, and `date`. No files under the live run
tree were modified, deleted, moved, or renamed, and the live training process
was not touched.

## Summary

The refreshed live E97/Emender DiLoCo tree contains **42 checkpoint files** under
`runs/`, all matching `checkpoint_step_*.pt`. The visible count is still not
approximately 183 checkpoints.

The 42 checkpoint files consume **324,226,286,244 bytes** (**324.23 GB**,
**301.96 GiB**). Every checkpoint file is exactly **7,719,673,482 bytes**
(**7.72 GB**, **7.19 GiB**), so the average, minimum, and maximum checkpoint
sizes are identical.

The relevant filesystem is `/dev/nvme1n1` mounted at `/mnt/nvme1n1`. At the
sample time it had **839,595,028,480 bytes available** (**839.60 GB**,
**782.00 GiB**) and was **95% used**.

At the observed active-segment cadence, checkpoint accumulation is a near-term
disk-pressure risk even though the current visible count is only 42. The active
segment is saving roughly every **8.6 minutes**, adding about **53.9 GB/hour**
decimal (**50.2 GiB/hour**) if the cadence continues. Current free space is
about **108.76 checkpoint equivalents**, or roughly **15-16 hours** of checkpoint
growth before checkpoint writes alone could consume the reported free bytes.
Other writes on `/mnt/nvme1n1` would shorten that window.

## Counts By Run Segment

| Run segment | Checkpoints | Step range | Checkpoint bytes | Directory bytes |
|---|---:|---:|---:|---:|
| `levelE97_100m_20260621_131958` | 1 | 500-500 | 7,719,673,482 | 7,719,676,276 |
| `levelE97_100m_20260621_133317` | 20 | 62,500-72,000 | 154,393,469,640 | 154,393,472,543 |
| `levelE97_100m_20260622_101547` | 1 | 72,500-72,500 | 7,719,673,482 | 7,719,676,385 |
| `levelE97_100m_20260623_103742` | 20 | 85,500-95,000 | 154,393,469,640 | 154,393,472,656 |
| **Total** | **42** | **500-95,000** | **324,226,286,244** | **324,226,297,860** |

Largest checkpoint directories:

1. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742`:
   154,393,472,656 bytes, 20 checkpoints.
2. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317`:
   154,393,472,543 bytes, 20 checkpoints.
3. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547`:
   7,719,676,385 bytes, 1 checkpoint.
4. `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_131958`:
   7,719,676,276 bytes, 1 checkpoint.

The whole inspected E97/Emender tree was **324,227,380,267 bytes**, so nearly
all usage in the scoped tree is checkpoint payload.

## Checkpoint Size Statistics

| Metric | Bytes | Decimal GB | Binary GiB |
|---|---:|---:|---:|
| Total checkpoint bytes | 324,226,286,244 | 324.23 | 301.96 |
| Average checkpoint size | 7,719,673,482 | 7.72 | 7.19 |
| Minimum checkpoint size | 7,719,673,482 | 7.72 | 7.19 |
| Maximum checkpoint size | 7,719,673,482 | 7.72 | 7.19 |

## Filesystem Free Space

`df -B1 /mnt/nvme1n1/erikg/diloco_8gpu/emender` reported:

| Filesystem | Size bytes | Used bytes | Available bytes | Use | Mount |
|---|---:|---:|---:|---:|---|
| `/dev/nvme1n1` | 15,360,854,417,408 | 14,521,259,388,928 | 839,595,028,480 | 95% | `/mnt/nvme1n1` |

Available space in checkpoint equivalents:

- `839,595,028,480 / 7,719,673,482 = 108.76` checkpoint files.
- Conservatively, that is **108 more full checkpoints** before this run's
  checkpoint files alone consume the currently reported free bytes.

## Save Cadence And Risk

In the active segment `levelE97_100m_20260623_103742`, the first currently
visible checkpoint is `checkpoint_step_085500_loss_2.9886.pt` with mtime
`2026-06-23 14:21:49 UTC`; the latest visible checkpoint is
`checkpoint_step_095000_loss_3.0385.pt` with mtime `2026-06-23 17:05:12 UTC`.

That gives 20 checkpoints over 19 intervals:

- Average interval: about **516 seconds** (**8.6 minutes**) per checkpoint.
- Growth rate: about **53.9 GB/hour decimal**, or **50.2 GiB/hour binary**.
- Daily rate if unchanged: about **1.29 TB/day decimal**, or **1.18 TiB/day binary**.

Risk assessment:

- **Current visible count is 42, not 183**, in the specified live path.
- **42 checkpoints is not itself excessive for rollback coverage**, but each one
  is 7.72 GB and the filesystem is already 95% full.
- **The save cadence is the main risk**. Dense 500-step checkpointing can consume
  the remaining filesystem headroom within the same day if left unattended.
- If there were actually 183 checkpoints of this same size in this tree, they
  would occupy about **1.41 TB** decimal (**1.29 TiB**) and would be too many
  for a filesystem already at 95% usage.

## Safe Retention Guidance

Do not prune anything without explicit approval and a fresh re-scan immediately
before cleanup. Also verify the training code's resume semantics before deleting
anything, especially whether it expects the latest checkpoint only or a specific
merge-aligned step.

Recommended preservation set:

- Keep resume-critical latest checkpoints in the active segment. At this sample
  time, preserve at least `checkpoint_step_095000_loss_3.0385.pt` plus the
  preceding few checkpoints such as `093500`, `094000`, and `094500` until newer
  checkpoints are confirmed good.
- Keep early proof / transition checkpoints explicitly called out by the task:
  `checkpoint_step_000500_loss_5.5811.pt`,
  `checkpoint_step_072000_loss_2.9556.pt`, and
  `checkpoint_step_072500_loss_2.9730.pt`.
- Keep periodic long-run milestones for traceability and rollback, such as every
  5,000 or 10,000 steps, plus segment boundaries and the latest checkpoint from
  each run directory.
- Treat dense middle saves in 500-step ranges as pruning candidates only after
  approval. The dense ranges `062500-071500` and `085500-093000` are the obvious
  candidates once newer checkpoints exist and resume safety is confirmed.

Practical policy proposal:

1. Near term: cap the active run to **latest 3-5 checkpoints plus every
   5,000-step milestone**.
2. Long term: keep **segment boundary checkpoints, latest checkpoint per segment,
   early proof checkpoints, and every 10,000-step milestone**.
3. If disk pressure remains high: reduce save cadence or add an approved
   retention job that runs only after a checkpoint completes and after a fresh
   `df`/`find` audit.

## Validation Checklist

- Checkpoint count and total size reported: **yes**.
- Filesystem free space reported: **yes**.
- Risk assessment provided: **yes**.
- Safe retention recommendation provided: **yes**.
- No files modified or deleted under `/mnt/nvme1n1/erikg/diloco_8gpu/emender`: **yes**.
