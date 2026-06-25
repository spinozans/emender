# E97 Checkpoint Retention Guard Report

Generated: 2026-06-25T16:25:04.440950+00:00
Root: `/mnt/nvme1n1/erikg/diloco_8gpu/emender`

## Filesystem

- Before free: 840,948,092,928 bytes (840.95 GB / 783.19 GiB)
- After free: 848,667,766,784 bytes (848.67 GB / 790.38 GiB)
- Before used: 14,519,906,324,480 bytes (14519.91 GB / 13522.72 GiB)
- After used: 14,512,186,650,624 bytes (14512.19 GB / 13515.53 GiB)
- Planned delete bytes: 15,439,346,964 bytes (15.44 GB / 14.38 GiB)
- Deleted bytes: 15,439,346,964 bytes (15.44 GB / 14.38 GiB)

## Inventory

- Checkpoint count before: 25
- Keep count: 23
- Planned delete count: 2
- Deleted count: 2
- Complete checkpoint size: 7,719,673,482 bytes (7.72 GB / 7.19 GiB)
- Newest checkpoint: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_260000_loss_2.7599.pt` mtime=2026-06-25T16:20:54.380067+00:00
- Observed active checkpoint interval: 7056.7 seconds
- Dense growth rate: 3.94 GB/hour
- Time to full at dense cadence after pruning: 215.5 hours

## Keep List

- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_131958/checkpoint_step_000500_loss_5.5811.pt` - latest-symlink-target, resume-config, resume-proof-critical, segment-first, segment-latest
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_062500_loss_2.9238.pt` - segment-first
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_070000_loss_3.0757.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_072000_loss_2.9556.pt` - latest-symlink-target, resume-config, resume-proof-critical, segment-latest
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547/checkpoint_step_072500_loss_2.9730.pt` - latest-symlink-target, resume-config, resume-proof-critical, segment-first, segment-latest
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_130000_loss_2.7040.pt` - 10000-step-milestone, segment-first
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_140000_loss_2.6782.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_150000_loss_2.9570.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_160000_loss_2.7092.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_170000_loss_2.8106.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_180000_loss_2.7579.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_190000_loss_2.7826.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_200000_loss_2.6192.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_210000_loss_2.8159.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_220000_loss_2.7872.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_230000_loss_2.7524.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_240000_loss_2.7610.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_250000_loss_2.7520.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_258000_loss_2.6889.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_258500_loss_2.6234.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_259000_loss_2.6734.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_259500_loss_2.4780.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_260000_loss_2.7599.pt` - 10000-step-milestone, global-latest, latest-5-active, latest-symlink-target, segment-latest

## Delete List

- deleted: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_257000_loss_2.7264.pt` size=7719673482 mtime=2026-06-25T15:29:25.978638+00:00
- deleted: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_257500_loss_2.7237.pt` size=7719673482 mtime=2026-06-25T15:38:01.339189+00:00

## Safety

- The training process was not stopped, killed, restarted, or reconfigured.
- Only `checkpoint_step_*.pt` files from the computed redundant set were eligible.
- The newest checkpoint and `latest.pt` target were kept.
- Files with non-modal size, too-new mtime, or not older than newest were skipped.
