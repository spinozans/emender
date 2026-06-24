# E97 8-GPU DiLoCo Versus Prior Single-GPU Run

Generated: 2026-06-24T07:52Z

Scope: read-only comparison of the current E97/Emender 8-GPU DiLoCo run against the best available prior E97 single-GPU run. I did not modify the live run, launch training, launch eval, stop or restart processes, prune checkpoints, or alter `/mnt` run artifacts. The only repository change is this Markdown report.

## Bottom Line

The current 8-GPU E97/Emender DiLoCo run appears to be essentially at ideal wall-clock token-throughput scaling relative to the prior single-GPU E97 baseline:

- Baseline single-GPU recent sustained throughput: about `8.16k global tok/s` over the last 100 logged rows.
- Current 8-GPU recent sustained throughput: about `65.8k global tok/s` after excluding merge/save slowdown rows, and about `65.8k` median over the last 100 logged rows including normal merge/save effects.
- Speedup estimate: `65.76k / 8.16k = 8.06x` by recent median global throughput; `65.85k / 8.16k = 8.07x` when using the recent non-merge steady-state mean; `64.07k / 8.08k = 7.93x` over the full parsed active logs.

This is close enough to call "near 8x" on wall-clock token throughput. The caveat is that this is a log-derived comparison, not a controlled benchmark, and the 8-GPU run is a resumed DiLoCo continuation rather than a from-scratch single-GPU-matched run. Quality is less directly comparable: training loss is healthy and lower than the single-GPU baseline at similar step numbers, but the current 8-GPU run has no heldout/BPB/racer eval artifact yet.

## Baseline Candidate Selection

Best comparable single-GPU baseline:

`/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750`

Why this is the best candidate:

- It is explicitly E97, `params=100m`, with the same geometry recorded in `args.json`: `dim=1792`, `depth=11`, `n_heads=216`, `n_state=32`, `mlp_ratio=2.2623`, `batch_size=4`, `chunk_size=2048`, `lr=0.001007`, `optimizer=schedulefree`, `bf16=true`, `use_triton=1`, `tokenizer=p50k_base`, `data=/home/erikg/elman/data/pile.txt`, `seed=42`.
- Its launch manifest requests exactly one GPU and runs plain `python train.py`, not `torchrun`.
- Existing project artifacts already use it as the E97 single-GPU reference. `experiments/diloco_seed_race_i4/README.md` names its step-150500 checkpoint as the seed for a DiLoCo dress rehearsal and describes it as the single-GPU reference. `experiments/diloco_seed_race_i4/REPORT.md` calls it the "single-GPU emender reference" and uses its heldout curve for matched-token comparisons.
- It has the needed artifacts: raw `run.log`, `args.json`, `launch_manifest.json`, 12 checkpoint files through final step `244141`, and committed heldout/BPB curve data in `experiments/diloco_seed_race_i4/reference_heldout_bpb.csv`.

Rejected or weaker candidates:

- `/mnt/nvme1n1/erikg/ref_emender_mlp.contaminated_2057` and archived subdirectories are explicitly marked contaminated or failed OOM and are not the clean baseline.
- Other DiLoCo sweep directories are multi-island runs, not single-GPU baselines.
- Paper pinned E88/GDN/M2RNN checkpoints are different run families and not a direct E97 single-GPU baseline for this task.

## Current 8-GPU Run

Primary live run:

`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742`

Primary log and metadata:

- Log: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- Args: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/args.json`
- Launch manifest: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/launch_manifest.json`
- Fresh health snapshot: `docs/FRESH_E97_RUN_STATUS_20260624.md`

Configuration and topology:

| Field | Value |
| --- | --- |
| Model | Level `E97`, `1,286,589,072` parameters |
| Geometry | `dim=1792`, `depth=11`, `n_heads=216`, `n_state=32`, `mlp_ratio=2.2623`, `mlp_multiple=64`, `embed_dim=1024` |
| Data/tokenizer | `/home/erikg/elman/data/pile.txt`, `p50k_base` |
| Precision/kernel | `bf16`, `use_triton=1`, fused split-edit Triton guard reports "NO eager fallback" |
| Optimizer | schedule-free AdamW, `lr=0.001007`, `weight_decay=0.01`, `grad_clip=1.0` |
| Per-rank batch | `batch_size=4`, `chunk_size=2048`, `grad_accum=1`, effective per-rank batch `4` |
| Launch | `torchrun --standalone --nproc_per_node=8` |
| GPUs/islands | 8 ranks, one per RTX 6000 Ada GPU; `world_size=8`; DiLoCo `K=250`, `outer_lr=1.0`, `outer_beta=0.0`, `outer_optimizer=avg` |
| Resume | Resumed from `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260622_101547/checkpoint_step_072500_loss_2.9730.pt` |
| Active-log start | step `72525`, loss `2.6353`, elapsed `0.018h`, time `2026-06-23T10:38:37Z` |
| Fresh snapshot | step `146050`, loss `2.7956`, per-rank `tok/s=8299`, `global_tok/s=66390`, time `2026-06-24T07:41:53Z` |
| Latest parsed while comparing | step `146400`, loss `2.7792`, elapsed `21.171h`, time `2026-06-24T07:47:47Z` |

Checkpoint state:

- Fresh snapshot reported latest checkpoint `checkpoint_step_146000_loss_2.8710.pt`, `latest.pt` pointing to it, and 17 checkpoint files under the broader E97 runs tree at `2026-06-24T07:42Z`.
- The active run directory had 12 retained checkpoint files when parsed for this comparison, from `checkpoint_step_096000_loss_2.9628.pt` through `checkpoint_step_146000_loss_2.8710.pt`; older active-run checkpoints had been pruned by the live retention guard.
- The raw active `run.log` records 147 saves for this active continuation, from step `73000` through step `146000`, but not all are retained on disk.

Training-loss summary from parsed active log rows:

| Window | Step range | Mean loss | Mean per-rank tok/s | Mean global tok/s | Median per-rank tok/s | Median global tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full parsed active log | `72525-146400` | `2.8519` | `8008.4` | `64067.0` | `8207.0` | `65657.0` |
| Last 100 logged rows | `143925-146400` | `2.7941` | `8020.3` | `64162.4` | `8219.5` | `65756.0` |
| Recent non-merge steady rows | `143475-146400` sampled rows with `tok/s>7500` | `2.7933` | `8231.0` | `65848.5` | `8221.0` | `65764.5` |

Current heldout/BPB/racer eval:

- I found no current-run heldout, BPB, racer, eval, or curve artifact under `/mnt/nvme1n1/erikg/diloco_8gpu/emender` at max depth 3.
- Prior diagnosis output (`.wg/output/diagnose-e97-diloco`) also recommended continuing while scheduling a real heldout/racer eval, explicitly noting that training loss is not a substitute for heldout/racer evaluation.
- Therefore current quality must be reported from training loss only, with high uncertainty.

## Baseline Single-GPU Run

Baseline run:

`/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750`

Primary log and metadata:

- Log: `/mnt/nvme1n1/erikg/ref_emender_mlp/run.log`
- Args: `/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750/args.json`
- Launch manifest: `/mnt/nvme1n1/erikg/ref_emender_mlp/launch_manifest.json`
- Heldout/BPB curve: `experiments/diloco_seed_race_i4/reference_heldout_bpb.csv`

Configuration and topology:

| Field | Value |
| --- | --- |
| Model | Level `E97`, `1,286,589,072` parameters |
| Geometry | Same as current: `dim=1792`, `depth=11`, `n_heads=216`, `n_state=32`, `mlp_ratio=2.2623`, `mlp_multiple=64`, `embed_dim=1024` |
| Data/tokenizer | `/home/erikg/elman/data/pile.txt`, `p50k_base` |
| Precision/kernel | `bf16`, `use_triton=1`, fused split-edit Triton guard reports "NO eager fallback" |
| Optimizer | schedule-free AdamW, `lr=0.001007`, `weight_decay=0.01`, `grad_clip=1.0` |
| Batch | `batch_size=4`, `chunk_size=2048`, `grad_accum=1`, effective batch `4` |
| Launch | `python train.py` |
| GPUs/islands | 1 GPU, no DiLoCo, `world_size=1` |
| Start/end | started from step `0`; completed at step `244141` |
| Wall-clock duration | last logged row at step `244100` had `elapsed_h=68.845`; final completion immediately after that |
| Final training summary | `FINAL_LOSS_LAST100: 3.1168`; peak memory `26533 MB`; reserved `31164 MB` |

Checkpoint state:

- 12 retained checkpoint files on disk, from `checkpoint_step_021500_loss_3.7778.pt` through `checkpoint_step_244141_loss_3.1168.pt`.
- Raw log records 11 periodic saves through `236500`; the final step `244141` checkpoint exists on disk even though the save line was not part of the sampled save-line regex output.

Training-loss summary from parsed baseline log rows:

| Window | Step range | Mean loss | Mean per-GPU tok/s | Mean global tok/s | Median per-GPU tok/s | Median global tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full parsed log | `100-244100` | `3.2694` | `8080.4` | `8080.4` | `8089.0` | `8089.0` |
| Last 100 logged rows | `234200-244100` | `3.1168` | `8146.6` | `8146.6` | `8161.0` | `8161.0` |
| Recent steady rows | `234100-244100` | `3.1161` | `8158.2` | `8158.2` | `8161.0` | `8161.0` |

Heldout/BPB already available for baseline:

| Step | Tokens | Heldout BPB |
| ---: | ---: | ---: |
| `21500` | `176,128,000` | `1.40351040` |
| `43000` | `352,256,000` | `1.28385275` |
| `64500` | `528,384,000` | `1.36944774` |
| `86000` | `704,512,000` | `1.24722992` |
| `107500` | `880,640,000` | `1.28705787` |
| `129000` | `1,056,768,000` | `1.22339529` |
| `150500` | `1,232,896,000` | `1.22907375` |
| `172000` | `1,409,024,000` | `1.23477140` |
| `193500` | `1,585,152,000` | `1.23780440` |
| `215000` | `1,761,280,000` | `1.24083433` |
| `236500` | `1,937,408,000` | `1.17021503` |
| `244141` | `2,000,003,072` | `1.17262861` |

The BPB values come from the committed `experiments/diloco_seed_race_i4/reference_heldout_bpb.csv`; that artifact says the curve was scored on a shared pile-tail held-out tensor with the same scoring basis used by the seed-race comparison.

## Normalized Throughput Comparison

The relevant distinctions:

- `tok/s` in the current 8-GPU log is per rank/per GPU. It is not the whole-job token rate.
- `global_tok/s` in the current 8-GPU log is approximately `tok/s * 8` and is the right wall-clock token-throughput number for the whole job.
- In the single-GPU baseline, `tok/s` and `global_tok/s` are identical because `world_size=1`.
- DiLoCo merge/save rows temporarily depress instantaneous throughput, so both all-row and steady-row windows are useful.

| Metric | Single-GPU baseline | Current 8-GPU DiLoCo | Ratio |
| --- | ---: | ---: | ---: |
| Recent median per-GPU/rank tok/s | `8161` | `8219.5` | `1.01x` |
| Recent median global tok/s | `8161` | `65756` | `8.06x` |
| Recent steady mean per-GPU/rank tok/s | `8158.2` | `8231.0` | `1.01x` |
| Recent steady mean global tok/s | `8158.2` | `65848.5` | `8.07x` |
| Full-log mean global tok/s | `8080.4` | `64067.0` | `7.93x` |

Conclusion: by log-reported global token throughput, the 8-GPU run is within measurement noise of ideal 8x scaling. Per-GPU throughput is not worse than baseline; it is slightly higher in recent windows, likely within normal run-to-run/kernel/noise variation.

Step-rate cross-check:

- Baseline final logged step: `244100` at `68.845h`, about `3546` optimizer steps/hour over the whole run.
- Current active continuation: from step `72525` at `0.018h` to step `146400` at `21.171h`, about `3492` optimizer steps/hour for the active continuation.
- The step rate is similar because each rank still processes the same per-step local batch. The 8x wall-clock gain comes from eight independent ranks processing about eight times as many aggregate tokens per elapsed second.

## Quality And Loss Progress

Training loss:

- At roughly comparable optimizer step `146000`, the single-GPU baseline log had loss `2.8456` at elapsed `41.302h`; the current 8-GPU active log had loss `2.8710` on the checkpoint row at elapsed `21.057h`, and the recent current-window mean was `2.7941`.
- The current run's recent training-loss band is therefore not worse by training loss, and appears healthy. Earlier health/diagnosis artifacts reached the same operational conclusion.
- The baseline final last-100 training loss was `3.1168`, but direct comparison to the current live tail is not a final quality statement because the runs differ in continuation history and DiLoCo aggregation.

Heldout/BPB:

- Baseline single-GPU heldout/BPB exists and ends around `1.1726 BPB` at `2.000B` tokens.
- Current 8-GPU live run has no discovered heldout/BPB/racer artifact yet.
- A related but not identical 4-island DiLoCo seed-race artifact showed DiLoCo beating the single-GPU reference by about `0.109 BPB` over matched-token overlap, but that was a different run (`seed_race_i4`) and should not be treated as a measured eval of the current 8-GPU live run.

Quality-at-step versus quality-at-token:

- Quality-at-step: the current 8-GPU run is at similar or slightly better training-loss territory than the baseline near step `146k`, but because it resumed from a previous 8-GPU checkpoint at step `72500`, this is a continuation comparison, not a from-scratch controlled run.
- Quality-at-wall-clock: the current active run reached the `146k` step neighborhood in about `21.1h` of the active continuation after resuming from `72.5k`; the baseline reached step `146k` from scratch at about `41.3h` and needed `68.8h` to finish. This supports the throughput read, but the resume point prevents a clean from-scratch wall-clock-to-quality comparison.
- Quality-at-token: if "tokens" means aggregate hardware tokens consumed by all DiLoCo ranks, the current run has consumed many more aggregate tokens per step than the single-GPU run. If "tokens" means the historical per-step token accounting used by some existing E97 artifacts (`step * 8192` before seed-aware corrections), the current and baseline steps can be compared directly, but that hides the actual wall-clock throughput scaling. Any quality-at-token claim should define which token convention it uses.

## Uncertainty

Confidence in throughput scaling: high. The run logs directly report per-rank and global tok/s, the same batch geometry is recorded in both args files, and the observed ratio is stable across recent and full-log windows.

Confidence in quality comparison: low to moderate. Training loss looks healthy, but current-run heldout/BPB/racer eval is absent. The single-GPU baseline has a proper heldout curve; the current 8-GPU run does not yet.

Main caveats:

- Current active run is resumed from `checkpoint_step_072500_loss_2.9730.pt`; it is not a from-scratch comparison against the baseline.
- DiLoCo changes optimization dynamics and checkpoint semantics; training loss is not a substitute for heldout/BPB/racer eval.
- Retention pruning means not all current active-run checkpoints remain on disk, though the raw log records saves and the latest retained checkpoint is present.
- The live run kept advancing while this report was written; numeric "latest" values are tied to the cited observation times.

## Validation

- Prior E97 single-GPU candidate identified: yes, `/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750`, with rationale and rejected alternatives.
- Current 8-GPU metrics summarized: yes, config, GPUs/ranks, DiLoCo settings, steps, throughput, elapsed time, loss windows, checkpoints, and eval absence.
- Baseline metrics summarized: yes, config, one-GPU topology, steps, throughput, wall-clock duration, loss windows, checkpoints, and heldout/BPB curve.
- Speedup estimate reported with caveats: yes, about `8.06x` recent median global throughput and `7.93x-8.07x` depending on window.
- No live run modified and no eval/training launched: yes. Commands used for run state were read-only inspections (`sed`, `rg`, `find`, `wc`, and Python log parsing). The only file written was this report.
