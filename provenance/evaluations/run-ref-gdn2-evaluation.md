# Evaluation: run-ref-gdn2

Task: `run-ref-gdn2`
Evaluator: `agent-1485`; continuation check by `agent-1486`
Date: 2026-06-15

## Grade

Overall score: **0.58 / 1.00**

Confidence: **0.86**

Rubric underspecified: **no**

The validation checklist is explicit: the deliverable is a verified live,
detached GDN-2 MLP reference run, launched through the wrapper, alone on exactly
one broker-leased GPU, with the specified schedule-free recipe and directly
comparable held-out curve artifacts. The actor did produce a real detached run
with strong recipe evidence and measured curve rows, but that run is no longer
live, the source-of-truth directory `/mnt/nvme1n1/erikg/ref_gdn2_mlp` is absent,
and the existing artifacts have been moved aside as
`/mnt/nvme1n1/erikg/ref_gdn2_mlp.contaminated_2057`. The current task state
therefore does not satisfy the hard gate: a confirmed live clean replacement run.

## Dimension Scores

| Dimension | Weight | Score | Rationale |
| --- | ---: | ---: | --- |
| Wrapper launch, detachment, and exclusive GPU | 0.25 | 0.35 | Stale artifacts and logs show wrapper-style launch, PID `3139509`, `gpus_requested=1`, leased GPU `1`, and live validation at handoff. Current verification fails the hard gate: PID `3139509` is dead, no active lease exists for GPU 1, and `/mnt/nvme1n1/erikg/ref_gdn2_mlp` no longer exists. |
| Recipe fidelity | 0.25 | 0.88 | Manifest/log evidence matches the main recipe: `dim=2176`, `n_heads=30`, `depth=12`, GDN-2 MLP ratio `3.2587`, `batch_size=4`, `chunk_size=2048`, `schedulefree`, `lr=0.000474`, `bf16`, `pile.txt`, and `p50k_base`; logs confirm warmup `0` and fused GDN-2 path with no eager fallback. Main miss: held-out cadence was `--heldout_curve_every 500`, while the task asks for 2000-step cadence. |
| Durable run artifacts and curve growth | 0.20 | 0.50 | `run.pid`, `launch_manifest.json`, `recipe_manifest.json`, `run.log`, held-out tensor, and `heldout_curve.csv` exist in the renamed contaminated directory, and the curve reached step `12000` / `98,352,000` tokens with decreasing BPB. However, the required source directory is gone, no current `train.log`/`run.log` is advancing, and no live process remains. |
| Direct comparability to ref-emender | 0.15 | 0.35 | Data path, tokenizer, y-mode held-out tensor size, and native single-GPU batch are aligned. Comparability is materially weakened because this run was moved aside as contaminated and used a 500-step held-out cadence despite the current spec requiring cadence matching at 2000. |
| Reporting of verified PID/GPU/first metrics | 0.10 | 0.86 | WG logs and `provenance/run-ref-gdn2-launch.md` report PID `3139509`, GPU `1`, first measured row `step=500`, `heldout_bpb=2.072381`, and log lines include tok/s around `8701` at that step. This was good at handoff but is stale after reset. |
| WG delivery hygiene | 0.05 | 0.80 | The actor committed relevant launcher/training support and a launch evidence artifact. Follow-on monitoring was set up. Hygiene is reduced because the task was later reset, the run was invalidated/renamed, and downstream source paths now point at missing files. |

Weighted total:

`0.25*0.35 + 0.25*0.88 + 0.20*0.50 + 0.15*0.35 + 0.10*0.86 + 0.05*0.80 = 0.586`.

Rounded calibrated score: **0.58**. The task's gate-scored deliverable is not
just a plausible launch artifact; it is a verified live clean run. The current
hard gate fails.

## Evidence Checked

- `wg show run-ref-gdn2` shows the task is currently `in-progress`, reset after
  the prior completion, with the current validation still requiring a live clean
  detached run.
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp` is absent.
- `ps -p 3139509` shows the previously reported detached PID is no longer alive.
- `scripts/gpu_lease.sh status` shows active leases only for unrelated GPUs
  `2,3,4,5`; GPU `1` is idle and not held by the prior run.
- `pgrep -af 'ref_gdn2_mlp|launch_ref_gdn2_mlp|train.py|gdn2'` finds unrelated
  DiLoCo training but no live `ref_gdn2_mlp` process.
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp.contaminated_2057` contains the stale run
  artifacts: `run.pid`, `launch_manifest.json`, `recipe_manifest.json`,
  `run.log`, held-out tensor, and `heldout_curve.csv`.
- `launch_manifest.json` records one requested GPU, leased GPU `1`, PID
  `3139509`, the correct command geometry, and `--heldout_curve_every 500`.
- `recipe_manifest.json` records `schedulefree`, `lr=0.000474`,
  `warmup_steps=0`, `cosine=false`, `decay=false`, `bf16=true`,
  `single_gpu=true`, `heldout_eval_mode=y`, and the external GDN-2 FLA chunked
  fused path.
- `run.log` contains the fused guard and schedule-free startup line:
  GDN-2 fused kernel, no eager fallback, schedule-free AdamW with
  `lr=0.000474`, `warmup_steps=0`.
- `heldout_curve.csv` grew from step `500` to step `12000`; the last row is
  `tokens=98352000`, `train_loss=3.837054`, `heldout_bpb=1.412094`.

## Validation Checklist Assessment

- Launched via wrapper, detached, on exactly one exclusive GPU: **partially met**.
  This was evidenced at handoff, but no live process or lease remains now.
- Recipe fidelity: **mostly met**, except held-out cadence mismatch
  (`500` observed versus `2000` requested).
- `run.pid` + manifest written; log/curve advancing: **partially met**. Artifacts
  exist only in the renamed contaminated directory and are no longer advancing.
- Same corpus/tokenizer/held-out/cadence as `run-ref-emender`: **partially met**.
  Corpus/tokenizer/y-mode are aligned; cadence and contamination invalidate full
  comparability.
- Reported verified PID, GPU id, first `(step, tok/s, heldout_bpb)`: **mostly
  met** from WG logs and launch evidence, but stale after reset.

## Final Assessment

This is not a successful completion of the current gate. The prior actor did
substantial useful work and launched a real GDN-2 reference process with the
right core recipe, so a zero would be too harsh. But the live-run deliverable is
currently absent, the original run was moved aside as contaminated, and the
current source-of-truth path has no advancing log or curve. The calibrated grade
is therefore **0.58 / 1.00**, and the task should be treated as incomplete until
a clean live detached replacement is verified.

## Continuation Check: 2026-06-15T21:07:25Z

The retry assigned to `agent-1486` was a continuation of the interrupted
evaluator closure, not a new actor launch. I rechecked the live system state
before closing the retry:

- `/mnt/nvme1n1/erikg/ref_gdn2_mlp` is still absent.
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp.contaminated_2057` still contains only the
  stale artifacts (`run.pid`, `launch_manifest.json`, `recipe_manifest.json`,
  `run.log`, `heldout_curve.csv`, and the held-out tensor).
- `pgrep -af 'ref_gdn2_mlp|train.py|gdn2'` finds the unrelated live
  `ref_emender_mlp` and DiLoCo training jobs, but no live `ref_gdn2_mlp`
  process.
- `scripts/gpu_lease.sh status` shows active leases on unrelated GPUs `0` and
  `2,3,4,5`; GPU `1`, the stale GDN-2 launch GPU, has no active lease.

This continuation evidence does not change the calibrated score. The grade
remains **0.58 / 1.00** with rubric underspecification flag **no**, and the
task remains incomplete until a clean live wrapper-launched replacement run is
verified on exactly one exclusive GPU.
