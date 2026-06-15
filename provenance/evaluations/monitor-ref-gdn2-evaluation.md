# monitor-ref-gdn2 endpoint monitor report

Checked at: 2026-06-15T21:31:14Z

## Verdict

Endpoint validation is not satisfied for the current source-of-truth run at
`/mnt/nvme1n1/erikg/ref_gdn2_mlp`.

The current detached run is the same clean run launched by `run-ref-gdn2` in its
latest retry: `run.pid` contains PID `584124`, `launch_manifest.json` records
PID `584124`, and the live process command matches the wrapper-launched
`gdn2-mlp` training command. The run is alive on broker-leased GPU 1.

However, the current source-of-truth directory does not contain
`heldout_curve.csv` or `recipe_manifest.json`, and no checkpoint file exists
under `runs/`. The current launch command also omits held-out curve flags; the
run's `args.json` records `heldout_curve_every: 0`, `heldout_curve_path: null`,
`heldout_eval_mode: "x"`, and `final_heldout_eval: false`. Therefore there is
no measured held-out BPB curve row at or above 2,000,000,000 tokens to report.

This monitor task must not launch a second training process, so this report
records the validation failure rather than attempting to repair or relaunch the
run.

## Source-of-truth files checked

- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.pid`: present; contains `584124`
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/run.log`: present; advancing
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/launch_manifest.json`: present
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/heldout_curve.csv`: missing
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/recipe_manifest.json`: missing
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627/args.json`: present

## Run identity

- `run.pid` contains `584124`.
- `launch_manifest.json` records `"pid": 584124`, `"gpus_requested": 1`, and
  `"leased_gpu_ids": "1"`.
- `ps -p 584124 -o pid,ppid,stat,etime,cmd` showed a live detached process:
  `python -u train.py --level gdn2-mlp --dim 2176 --depth 12 --n_heads 30 ... --output /mnt/nvme1n1/erikg/ref_gdn2_mlp/runs`.
- `scripts/gpu_lease.sh status` showed GPU 1 leased by PID `584124` with a
  fresh heartbeat.
- `nvidia-smi --query-compute-apps=pid,...` showed PID `584124` using GPU
  memory.
- Process search found one current `ref_gdn2_mlp` training command for the
  source-of-truth output root, plus the wrapper shell that launched it.

Conclusion: this monitor inspected the same latest detached run from
`run-ref-gdn2`, not a duplicate training launch.

## Current measured state

The current run has no held-out curve file, so the requested final
`heldout_bpb` state is not available.

The latest training-log row observed at check time was:

| step | tokens inferred from batch/chunk | train_loss | heldout_bpb | log time |
| ---: | ---: | ---: | ---: | :--- |
| 287 | 2,351,104 | 6.0562 | unavailable | 2026-06-15T21:31:14+00:00 |

Token inference uses `batch_size=4` and `chunk_size=2048`, so each step is
8,192 tokens. This inferred training-token count is far below
2,000,000,000. It is not a substitute for a measured `heldout_curve.csv` row.

## Held-out monotonicity / rollover

No monotonicity or rollover conclusion can be made for the current clean run:
there is no current `heldout_curve.csv`, and `args.json` shows held-out curve
generation disabled.

A stale renamed directory,
`/mnt/nvme1n1/erikg/ref_gdn2_mlp.contaminated_2057`, contains an older
`heldout_curve.csv`, but it is not the source-of-truth path for this task and is
not evidence for the current PID `584124` run.

## Checkpoint state

No checkpoint or model-weight file was found under
`/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/` at max depth 8. The only file under the
current run directory was:

- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627/args.json`

The current launch uses `--save_every 25000`, so at the observed step 287 it has
not reached the first configured checkpoint cadence.

## Recipe verification

Evidence that matches the required recipe:

- single GPU: `launch_manifest.json` has `gpus_requested=1`,
  `leased_gpu_ids="1"`, and GPU lease status shows PID `584124` on GPU 1
- optimizer: `schedulefree`
- learning rate: `0.000474` (`4.74e-4`)
- warmup: `warmup_steps=0`
- corpus: `/home/erikg/elman/data/pile.txt`
- tokenizer: `p50k_base`
- model geometry: `level=gdn2-mlp`, `dim=2176`, `depth=12`, `n_heads=30`,
  `gdn2_mlp_ratio=3.258732449079677`
- fused path: `run.log` reports `FLA chunked GDN-2 fused kernel, NO eager fallback`

Evidence that fails or is unavailable:

- `recipe_manifest.json` is missing from the current source-of-truth directory.
- No manifest field verifies y-mode heldout for the current run.
- `args.json` records `heldout_eval_mode: "x"`, `heldout_curve_every: 0`,
  `heldout_curve_path: null`, `heldout_tensor: null`, and
  `final_heldout_eval: false`.
- The command uses `--val_every 999999999` and contains no
  `--heldout_curve_path`, no `--heldout_curve_every`, and no y-mode heldout
  argument.

## Validation checklist status

- Confirms monitored PID/run is the same detached run from `run-ref-gdn2`, not a
  duplicate launch: PASS for latest clean run PID `584124`.
- `heldout_curve.csv` contains a row with tokens `>= 2,000,000,000`: FAIL;
  current source-of-truth file is missing.
- A checkpoint exists under `/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/` for the
  endpoint or final step: FAIL; no checkpoint exists under the current run
  directory.
- Reports measured final step, tokens, train_loss, `heldout_bpb`, and whether
  `heldout_bpb` is monotone or rolls over: FAIL as an endpoint report; the
  latest log row is step 287 with inferred 2,351,104 training tokens and
  `train_loss=6.0562`, but `heldout_bpb` is unavailable because the held-out
  curve is absent.
- Verifies launch/recipe manifests still show single GPU, schedulefree
  `lr=4.74e-4`, warmup 0, no cosine/decay, `p50k_base` `pile.txt`, y-mode
  heldout: PARTIAL/FAIL; launch and args confirm single GPU, schedulefree,
  `lr=4.74e-4`, warmup 0, `p50k_base`, and `pile.txt`, but
  `recipe_manifest.json` is missing and current args show x-mode/no heldout
  curve rather than y-mode heldout.

## Commands used for validation

- `cat /mnt/nvme1n1/erikg/ref_gdn2_mlp/run.pid`
- `cat /mnt/nvme1n1/erikg/ref_gdn2_mlp/launch_manifest.json`
- `cat /mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627/args.json`
- `ps -p 584124 -o pid,ppid,stat,etime,cmd`
- `pgrep -af 'ref_gdn2_mlp|level gdn2-mlp|--level gdn2-mlp'`
- `scripts/gpu_lease.sh status`
- `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader`
- `tail -n 5 /mnt/nvme1n1/erikg/ref_gdn2_mlp/run.log`
- `find /mnt/nvme1n1/erikg/ref_gdn2_mlp -maxdepth 8 -iname '*heldout*' -o -iname '*curve*' -o -iname '*checkpoint*' -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt'`
- `find /mnt/nvme1n1/erikg -maxdepth 4 -path '*ref_gdn2_mlp*' -type f -printf ...`
