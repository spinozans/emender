# monitor-ref-gdn2 interim evaluation

Checked at: 2026-06-15T18:00:14Z

## Verdict

Endpoint validation is not yet complete. The monitored detached run is live and matches the `run-ref-gdn2` source-of-truth PID, but `heldout_curve.csv` has not reached 2,000,000,000 tokens and no endpoint/final checkpoint exists yet.

This report is an interim monitor artifact. The task should remain waiting until either the run exits or `heldout_curve.csv` includes a row with `tokens >= 2000000000`.

## Source-of-truth files

- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.pid`
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.log`
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/heldout_curve.csv`
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/launch_manifest.json`
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/recipe_manifest.json`

## Run identity

- `run.pid` contains `3139509`.
- `ps -fp 3139509` shows `/usr/bin/python3 /home/erikg/ndm/.wg-worktrees/agent-1452/train.py ... --output /mnt/nvme1n1/erikg/ref_gdn2_mlp/runs ... --heldout_curve_path /mnt/nvme1n1/erikg/ref_gdn2_mlp/heldout_curve.csv`.
- `launch_manifest.json` records `"pid": 3139509`, `"gpus_requested": 1`, and `"leased_gpu_ids": "1"`.
- `scripts/gpu_lease.sh status` shows GPU 1 leased by PID `3139509`; `nvidia-smi` shows PID `3139509` using GPU memory on the corresponding device.
- Process search found one `ref_gdn2_mlp` training process for this output root. Other active training processes use unrelated output roots.

Conclusion: this is the same detached run launched by `run-ref-gdn2`, not a duplicate launch.

## Current curve state

`heldout_curve.csv` currently contains two measured rows:

| step | tokens | train_loss | heldout_ce | heldout_bpb | mode | wall_time_utc |
| ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 500 | 4,098,000 | 5.717713 | 5.667046 | 2.072381 | y | 2026-06-15T17:45:34+00:00 |
| 1000 | 8,196,000 | 5.235021 | 5.269750 | 1.927094 | y | 2026-06-15T17:53:59+00:00 |

Latest measured row:

- step: `1000`
- tokens: `8196000`
- train_loss: `5.235021`
- heldout_bpb: `1.927094`
- monotonicity: heldout_bpb is monotone non-increasing over the measured rows (`2.072381 -> 1.927094`)
- rollover: no rollover observed in the measured rows

The endpoint criterion is not yet met because the latest measured token count is below `2,000,000,000`.

## Checkpoint state

`find /mnt/nvme1n1/erikg/ref_gdn2_mlp/runs -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name 'checkpoint*' \)` returned no checkpoint files.

The run configuration has `--save_every 25000`; the current measured curve state is step `1000`, so the absence of an endpoint/final checkpoint is expected at this interim point but does not satisfy the task validation requirement.

## Recipe verification

`launch_manifest.json` and `recipe_manifest.json` still show:

- single GPU: `gpus_requested=1`, `leased_gpu_ids=1`, `recipe.single_gpu=true`
- optimizer: `schedulefree`
- learning rate: `0.000474` (`4.74e-4`)
- warmup: `warmup_steps=0`
- no cosine/decay: `cosine=false`, `decay=false`
- data: `/home/erikg/elman/data/pile.txt`
- tokenizer: `p50k_base`
- heldout mode: `y`
- heldout tensor: `/mnt/nvme1n1/erikg/ref_gdn2_mlp/heldout_pile_tail_p50k_2048_1m.pt`

## Validation checklist status

- Confirm monitored PID/run is same detached run from `run-ref-gdn2`, not a duplicate launch: PASS
- `heldout_curve.csv` contains row with `tokens >= 2,000,000,000`: PENDING
- Checkpoint exists under `/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/` for endpoint/final step: PENDING
- Report measured final step, tokens, train_loss, heldout_bpb, monotonicity/rollover: INTERIM ONLY, latest measured row reported above
- Verify manifests show required recipe: PASS
