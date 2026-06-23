# E97 Checkpoint Retention Guard Report

Generated: 2026-06-23T20:08:10.744112+00:00
Root: `/mnt/nvme1n1/erikg/diloco_8gpu/emender`

## Filesystem

- Before free: 839,491,207,168 bytes (839.49 GB / 781.84 GiB)
- After free: 839,491,207,168 bytes (839.49 GB / 781.84 GiB)
- Before used: 14,521,363,210,240 bytes (14521.36 GB / 13524.07 GiB)
- After used: 14,521,363,210,240 bytes (14521.36 GB / 13524.07 GiB)
- Planned delete bytes: 231,590,204,460 bytes (231.59 GB / 215.69 GiB)
- Deleted bytes: 0 bytes (0.00 GB / 0.00 GiB)

## Inventory

- Checkpoint count before: 42
- Keep count: 12
- Planned delete count: 30
- Deleted count: 0
- Complete checkpoint size: 7,719,673,482 bytes (7.72 GB / 7.19 GiB)
- Newest checkpoint: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_105500_loss_2.7685.pt` mtime=2026-06-23T20:05:37.780576+00:00
- Observed active checkpoint interval: 515.5 seconds
- Dense growth rate: 53.91 GB/hour
- Time to full at dense cadence after pruning: 15.6 hours

## Keep List

- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_131958/checkpoint_step_000500_loss_5.5811.pt` - latest-symlink-target, resume-config, resume-proof-critical, segment-first, segment-latest
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_062500_loss_2.9238.pt` - segment-first
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_070000_loss_3.0757.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_072000_loss_2.9556.pt` - latest-symlink-target, resume-config, resume-proof-critical, segment-latest
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547/checkpoint_step_072500_loss_2.9730.pt` - latest-symlink-target, resume-config, resume-proof-critical, segment-first, segment-latest
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_096000_loss_2.9628.pt` - segment-first
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_100000_loss_2.8582.pt` - 10000-step-milestone
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_103500_loss_2.8982.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_104000_loss_2.8354.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_104500_loss_2.9981.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_105000_loss_2.9437.pt` - latest-5-active
- `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_105500_loss_2.7685.pt` - global-latest, latest-5-active, latest-symlink-target, segment-latest

## Delete List

- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_063000_loss_2.9200.pt` size=7719673482 mtime=2026-06-22T07:28:24.232744+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_063500_loss_2.9859.pt` size=7719673482 mtime=2026-06-22T07:36:59.162080+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_064000_loss_3.0097.pt` size=7719673482 mtime=2026-06-22T07:45:33.209975+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_064500_loss_2.8266.pt` size=7719673482 mtime=2026-06-22T07:54:07.643064+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_065000_loss_2.7802.pt` size=7719673482 mtime=2026-06-22T08:02:43.705966+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_065500_loss_2.9109.pt` size=7719673482 mtime=2026-06-22T08:11:18.396183+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_066000_loss_2.8795.pt` size=7719673482 mtime=2026-06-22T08:19:53.533623+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_066500_loss_2.8240.pt` size=7719673482 mtime=2026-06-22T08:28:29.838646+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_067000_loss_2.6378.pt` size=7719673482 mtime=2026-06-22T08:37:05.734465+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_067500_loss_2.9196.pt` size=7719673482 mtime=2026-06-22T08:45:39.928434+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_068000_loss_2.8659.pt` size=7719673482 mtime=2026-06-22T08:54:14.880781+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_068500_loss_2.9674.pt` size=7719673482 mtime=2026-06-22T09:02:49.440933+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_069000_loss_2.8546.pt` size=7719673482 mtime=2026-06-22T09:11:25.079623+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_069500_loss_2.7933.pt` size=7719673482 mtime=2026-06-22T09:19:59.554732+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_070500_loss_2.9315.pt` size=7719673482 mtime=2026-06-22T09:37:09.335365+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_071000_loss_2.9467.pt` size=7719673482 mtime=2026-06-22T09:45:45.005071+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260621_133317/checkpoint_step_071500_loss_3.0169.pt` size=7719673482 mtime=2026-06-22T09:54:19.890385+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_096500_loss_2.9209.pt` size=7719673482 mtime=2026-06-23T17:30:58.065037+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_097000_loss_2.9447.pt` size=7719673482 mtime=2026-06-23T17:39:31.862808+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_097500_loss_2.9895.pt` size=7719673482 mtime=2026-06-23T17:48:05.984740+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_098000_loss_2.8920.pt` size=7719673482 mtime=2026-06-23T17:56:40.820029+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_098500_loss_2.9358.pt` size=7719673482 mtime=2026-06-23T18:05:15.997489+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_099000_loss_2.8848.pt` size=7719673482 mtime=2026-06-23T18:13:51.480102+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_099500_loss_2.9075.pt` size=7719673482 mtime=2026-06-23T18:22:26.658562+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_100500_loss_2.9048.pt` size=7719673482 mtime=2026-06-23T18:39:36.658304+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_101000_loss_2.9545.pt` size=7719673482 mtime=2026-06-23T18:48:12.507100+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_101500_loss_2.9315.pt` size=7719673482 mtime=2026-06-23T18:56:49.673554+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_102000_loss_2.9349.pt` size=7719673482 mtime=2026-06-23T19:05:24.009593+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_102500_loss_2.6907.pt` size=7719673482 mtime=2026-06-23T19:14:00.240580+00:00
- planned: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_103000_loss_2.8098.pt` size=7719673482 mtime=2026-06-23T19:22:36.771717+00:00

## Safety

- The training process was not stopped, killed, restarted, or reconfigured.
- Only `checkpoint_step_*.pt` files from the computed redundant set were eligible.
- The newest checkpoint and `latest.pt` target were kept.
- Files with non-modal size, too-new mtime, or not older than newest were skipped.
