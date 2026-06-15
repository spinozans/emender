# Evaluation: catchup-parallel-diloco

Task ID: `catchup-parallel-diloco`
Evaluator: `agent-1451`
Date: 2026-06-15

## Summary

Score: **0.00 / 1.00**
Confidence: **0.98**
Rubric underspecified: **No**

The task had a concrete validation checklist requiring multi-GPU experiment runs, durable held-out BPB curves, throughput measurements, comparison against the `run-ref-emender-mlp` reference, and explicit conclusions about the DDP batch/LR confound. No evidence was submitted that any of this work was attempted. The WG task record contains no artifacts, no commits, no validation logs, and no files in the required output location (`/mnt/nvme1n1/erikg/catchup_parallel/`) were visible during evaluation.

## Dimension Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| GPU leasing and execution discipline | 0.00 | No logs or artifacts indicate that `scripts/gpu_lease.sh` was used, that GPUs were leased, or that any training job was started. |
| Configuration fidelity | 0.00 | No run configs were provided for DiLoCo-native, DDP-naive, or DDP-LR-rescaled. There is no evidence of schedule-free CMA LR `1.007e-3`, fused bf16, matching corpus/tokenizer/held-out, or native per-replica `bs4`. |
| Measured BPB curves | 0.00 | No held-out BPB-vs-tokens logs were produced for any requested config. The required durable directory was absent or empty. |
| Reference comparison | 0.00 | No overlay or comparison against the `run-ref-emender-mlp` answer-key curve was produced. |
| Throughput reporting | 0.00 | No measured tok/s or projected days-to-100B were reported. |
| Confound conclusion | 0.00 | The task did not state whether DDP-naive reproduced rollover or whether LR-rescaling and/or DiLoCo-native removed it. |
| Completion behavior | 0.00 | There was no evidence of staying to completion; the task contained only assignment lifecycle logs before evaluator inspection. |

## Evidence Reviewed

- `wg show catchup-parallel-diloco`: task was in progress with zero commits ahead, no artifacts, and only assignment/evaluator logs.
- `wg context catchup-parallel-diloco`: reported no dependency artifacts.
- `wg show .assign-catchup-parallel-diloco`: only the bare assignment task had completed.
- `find /mnt/nvme1n1/erikg/catchup_parallel -maxdepth 3 -type f`: produced no result.
- `git log --oneline main..HEAD`: no prior actor commits were present.

## Calibration

This is a near-complete miss rather than a partial attempt. The task was not merely missing a final report; it lacked the primary experimental outputs required for every validation item. Because the rubric was explicit and all core deliverables were absent, the calibrated score is 0.00.
