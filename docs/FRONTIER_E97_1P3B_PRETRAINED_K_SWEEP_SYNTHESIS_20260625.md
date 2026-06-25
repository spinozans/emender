# E97 1.3B Pretrained 8-Node K-Sweep Synthesis - 2026-06-25

WG task: `synthesize-e97-1-3b`

## Decision

Proceed to exactly one bounded 16-node pretrained GPU-island/no-DDP K40 probe.

K40 is the only green member of the 8-node bracket. It used the same staged
pretrained source checkpoint and fixed eval baseline as K160 and K320, completed
cleanly, produced finite/improving train-loss windows, wrote a final consensus
checkpoint with valid `latest.pt`, and passed the fixed eval non-regression
gate:

- CE delta: `+0.00803614`, within the `<= +0.025` gate.
- BPB delta: `+0.00334350`, within the `<= +0.010` gate.

K160 and K320 are not scale candidates from this bracket. Both were
operationally clean and had finite/improving train loss, but both regressed
well beyond the fixed eval thresholds:

- K160: CE `+0.23717952`, BPB `+0.09868055`.
- K320: CE `+0.30940366`, BPB `+0.12873002`.

Created downstream WG task `run-e97-1-3b-4` for the one selected 16-node K40
probe. No 16-node Slurm job was submitted from this synthesis task.

`run-64-node-e97` was checked after synthesis and remains `open (PAUSED)`.

## Shared Source And Baseline

Validation source: `docs/FRONTIER_E97_1P3B_PRETRAINED_VALIDATION_20260625.md`.

The validation task authorized only the bounded 8-node pretrained
GPU-island/no-DDP K40/K160/K320 bracket and explicitly excluded 16/32/64-node
continuation, GDN2/CMAES, and schedule-free outer jobs at that stage.

All rows below used the same staged source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt
```

All fixed eval comparisons used the same scoring tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

The source fixed eval baseline was:

| Row | Step | CE | BPB |
| --- | ---: | ---: | ---: |
| Source pretrained checkpoint | `260500` | `26.93483090` | `11.20646537` |

The absolute CE/BPB values on this smoke tensor are poor; the decision signal is
the row-matched candidate-minus-source delta under the same invocation.

## Bracket Summary

| Row | Job ids | Nodes / topology | K / outer | Source -> final step | Train loss trend | Throughput | Merges | Node-hours | Fixed eval CE / BPB delta | Finalization behavior | Verdict |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- |
| Source baseline | eval `4900725` | 1-node eval only | n/a | `260500` | Baseline checkpoint loss metadata `2.7480917453765867`; no continuation | n/a | n/a | eval elapsed `00:00:49` | `+0.00000000` / `+0.00000000` | Strict load OK; baseline CSV written | Baseline |
| 8-node K40 | train `4900838`; eval `4901316` | 8 Frontier nodes, 64 singleton GPU islands, no DDP | `DILOCO_K=40`, avg outer, export basis `x` | `260500 -> 263082` | Rank-0 first-20 mean `2.7153`, last-20 mean `2.6672`, `FINAL_LOSS_LAST100=2.6645`; finite and mildly improving | mean `139802` global tok/s; median after first 10 rows `164697` global tok/s | `66` | train requested/actual `8.0` / `6.704444`; eval requested/actual `1.0` / `0.023333` | `+0.00803614` / `+0.00334350` | Clean exit; periodic K-aligned saves through `263080`; final consensus merge at `263082`; final checkpoint and `latest.pt` valid | Green: select for one 16-node probe |
| 8-node K160 | train `4900869`; eval `4901464` | 8 Frontier nodes, 64 singleton GPU islands, no DDP | `DILOCO_K=160`, avg outer, export basis `x` | `260500 -> 263840` | First-100 mean `2.71014`, last-100 mean `2.636824`, `FINAL_LOSS_LAST100=2.6368`; finite and improving | filtered mean `167306`, filtered median `168217` global tok/s | `21` | train requested/actual `8.0` / `6.700`; eval actual `0.023056` from `00:01:23` elapsed | `+0.23717952` / `+0.09868055` | Clean exit; final step exactly on K boundary, final merge skipped because already consensus; final checkpoint and `latest.pt` valid | Red: fixed eval fails |
| 8-node K320 | train `4901367`; eval `4901744` | 8 Frontier nodes, 64 singleton GPU islands, no DDP | `DILOCO_K=320`, avg outer, export basis `x` | `260500 -> 263947` | First-20 mean `2.7035`, last-20 mean `2.5451`, `FINAL_LOSS_LAST100=2.6770`; finite and mildly improving | mean `161693`, median `165734`, last-20 mean `166255` global tok/s | `11` | train requested/actual `8.000000` / `6.704444`; eval actual `0.023333` | `+0.30940366` / `+0.12873002` | Clean exit; final consensus merge/checkpoint at `263947`; `latest.pt` valid; 64/64 final-ready markers present | Red: fixed eval fails |

## Systems Interpretation

The systems result alone does not explain the quality split. All three 8-node
arms launched and finalized successfully, used the intended singleton-island
no-DDP path, wrote final checkpoints, and had no observed OOM, Python traceback,
non-finite loss, RCCL/NCCL watchdog timeout, or collective mismatch.

The cadence difference mostly changed merge frequency and synchronization cost:

- K40 merged `66` times, with total sync time `273.306` seconds and average
  sync `4141.0` ms.
- K160 merged `21` times, with total sync time `85.970` seconds and average
  sync `4093.8` ms.
- K320 merged `11` times, with total sync time `49.402` seconds and average
  sync `4491.1` ms.

Throughput for K160/K320 was slightly higher than K40 because there were fewer
save/merge dips, but that operational gain did not preserve fixed-eval quality.
For this staged pretrained checkpoint and this short continuation envelope,
lower merge cadence was not quality-preserving.

## Quality Interpretation

Train loss alone would have been misleading. K160 and K320 both showed finite
and improving logged train-loss windows, and K160 even had the lowest reported
`FINAL_LOSS_LAST100` among the three arms. The fixed eval row, however, showed
large candidate-minus-source regressions for both K160 and K320 under the same
scoring tensor and same invocation.

K40 is not proven optimal; it is only the cleanest bracket row. It stayed within
the fixed eval gate while maintaining the same operational reliability as the
higher-K arms. That is enough evidence for one bounded 16-node selected-recipe
probe, but not enough to skip directly to 32 or 64 nodes.

## Downstream Task

Created exactly one downstream WG task:

```text
run-e97-1-3b-4 - Run E97 1.3B pretrained 16-node K40 probe
```

Required envelope for that task:

- Resume from the same staged source checkpoint used here.
- Keep the same fixed eval scoring tensor and source baseline.
- Use singleton GPU islands/no DDP.
- Use `DILOCO_K=40`, `SAVE_EVERY=40`, `DILOCO_OUTER_OPTIMIZER=avg`,
  `DILOCO_OUTER_LR=1.0`, `DILOCO_OUTER_BETA=0.0`,
  `DILOCO_EXPORT_BASIS=x`, `BATCH_SIZE=1`, and `CHUNK_SIZE=2048`.
- Submit exactly one bounded 16-node training job and exactly one fixed eval.
- Do not submit 32-node, 64-node, GDN2, CMAES, schedule-free outer, LR sweep,
  beta sweep, extra K sweep, or a second 16-node job.
- Confirm `run-64-node-e97` remains paused.

## Scope Confirmation

- Read and compared `docs/FRONTIER_E97_1P3B_PRETRAINED_VALIDATION_20260625.md`.
- Read and compared all completed 8-node K reports:
  - `docs/FRONTIER_E97_1P3B_PRETRAINED_8NODE_K40_20260625.md`
  - `docs/FRONTIER_E97_1P3B_PRETRAINED_K160_8NODE_20260625.md`
  - `docs/FRONTIER_E97_1P3B_PRETRAINED_8N_K320_20260625.md`
- Created exactly one downstream WG scaleout task because K40 was green:
  `run-e97-1-3b-4`.
- Did not run `sbatch` and submitted no Slurm job from this synthesis task.
- Confirmed `run-64-node-e97` remains `open (PAUSED)`.

