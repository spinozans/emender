# X4 — released 1.3B pair, matched lr 1e-3

Status: **incomplete_fail_closed**.

Named reference throughout: **marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874%**. Task-spec shorthand `8.224` was corrected before launch; it is not ln(3721).
Loss below the marginal-optimal reference indicates predictive signal, not alone learned deep computation. Full-vocabulary argmax exact joint class61*x+y; invalid classes fail both coordinates. All table accuracy units are percent; CE is nats.

## Amendment #5 registration (before launch)

```json
{
  "amendment": 5,
  "registered_at": "2026-10-16T22:28:34.355156+00:00",
  "base_commit": "8ea0c761f234e0a4d535bf463d7ef55ddcbda024",
  "authority": "/home/erikg/emender-scratch/fix-fanout/taskX4-13b-confirm.md; operator matched1e-3 ruling and live storage/concurrency/reference correction approvals",
  "matrix": [
    {
      "name": "headline_e97_seed0",
      "arm": "e97",
      "seed": 0,
      "lr": 0.001,
      "drivers": 128
    },
    {
      "name": "headline_e97_seed1",
      "arm": "e97",
      "seed": 1,
      "lr": 0.001,
      "drivers": 128
    },
    {
      "name": "headline_e97_seed2",
      "arm": "e97",
      "seed": 2,
      "lr": 0.001,
      "drivers": 128
    },
    {
      "name": "headline_gdn2_seed0",
      "arm": "gdn2",
      "seed": 0,
      "lr": 0.001,
      "drivers": 128
    },
    {
      "name": "headline_gdn2_seed1",
      "arm": "gdn2",
      "seed": 1,
      "lr": 0.001,
      "drivers": 128
    },
    {
      "name": "headline_gdn2_seed2",
      "arm": "gdn2",
      "seed": 2,
      "lr": 0.001,
      "drivers": 128
    }
  ],
  "released_parameters": {
    "e97": 1286589072,
    "gdn2": 1286713448
  },
  "recipe": "scripts/expressivity_1p3b.py released strict loaders with Schedule-Free train/y weight recovery; full bf16 fine-tune, fresh AdamW(0.9,0.95), batch32, wd0.01, clip0.5, 300-update warmup then constant matched1e-3; no prior optimizer state loaded",
  "training": "Fresh procedural generate_batch on every update with seed0/1/2 streams paired across arms; archived128 pool unused; no pilot continuation",
  "evaluation": "Same X-series frozen512val/512test at128/512/1024 drivers plus2 initialization tokens; joint class61*x+y over full pretrained vocab, causal right-pad-to16 sliced before loss/scoring",
  "frozen_data_receipts": {
    "0": {
      "sha256": "93146340c49620188803292d5df1416f2f5bbf6a9bb64f895d4ce4f101b677a7",
      "seed": 0,
      "pools": {
        "train": {
          "shape": [
            128,
            130
          ],
          "input_sha256": "f47e08c37df6f71b62d5b76dd0f352e00df95fcf82e71d0b5d7e9dc8ccd1f3ad"
        },
        "val": {
          "shape": [
            512,
            130
          ],
          "input_sha256": "3a96282a28ead06cfc3e77818368b4a7d2e368d1ffe543a41f35f143216ae2a9"
        },
        "test_1x": {
          "shape": [
            512,
            130
          ],
          "input_sha256": "acabe2cea91c288e54cbe67b4ec8a15bf32fd05c0147905e291fa031ba11fb94"
        },
        "test_4x": {
          "shape": [
            512,
            514
          ],
          "input_sha256": "40e0eb958cf63b07e05d92c97732eae8ae05e6832b7f327e3a124a130dd74943"
        },
        "test_8x": {
          "shape": [
            512,
            1026
          ],
          "input_sha256": "961e353b6cae1269364cc943545a8cf869b3ff8a320a2142cd0be0cd7de06347"
        }
      }
    },
    "1": {
      "sha256": "3987144cfe60a2b3cc788eb66e7dc3a483e2fbe91dc85ec5abb4532ddbebbd4a",
      "seed": 1,
      "pools": {
        "train": {
          "shape": [
            128,
            130
          ],
          "input_sha256": "e89f6dbb1980e22ed525e51054e9710f68263bff1d11dda161c4369abc0f8423"
        },
        "val": {
          "shape": [
            512,
            130
          ],
          "input_sha256": "5af6c6d14bc689d851ec181d3e45beeb565d5e5bd90fc69d3fa0355a836331a1"
        },
        "test_1x": {
          "shape": [
            512,
            130
          ],
          "input_sha256": "08beb2caf6e127b97fcd20b0c2ca465e25b02867caa791bc378aea18810893c0"
        },
        "test_4x": {
          "shape": [
            512,
            514
          ],
          "input_sha256": "6421447e71a306e6c3c1d8a8c7d9be4e62d2a77a52c51fb5b3f4a9dfa521f280"
        },
        "test_8x": {
          "shape": [
            512,
            1026
          ],
          "input_sha256": "fcfd153db359416785dba99e7494a6d0300536d0f753e7cce671001a04834ed2"
        }
      }
    },
    "2": {
      "sha256": "03bf507a9cafdbca9bd4e574b83d1f663f72d7504c193fb6214d0f505ace257c",
      "seed": 2,
      "pools": {
        "train": {
          "shape": [
            128,
            130
          ],
          "input_sha256": "1cbc01c64613a23566cb198c5902473ddbc5fa44ed3d3c6ace1373a84c0f689f"
        },
        "val": {
          "shape": [
            512,
            130
          ],
          "input_sha256": "c6074d797d25034a799f4d0003e7ae56d93c01ae70ccb894b3252c2a7e5a7b04"
        },
        "test_1x": {
          "shape": [
            512,
            130
          ],
          "input_sha256": "66ad8e6f6182349993c49479291fbcb43d712868e643033c8b572f963603ceef"
        },
        "test_4x": {
          "shape": [
            512,
            514
          ],
          "input_sha256": "463af3ca8f73f047edd93dcfd2e822512334bd53514ab7f633284b16bc5b6f1b"
        },
        "test_8x": {
          "shape": [
            512,
            1026
          ],
          "input_sha256": "290c9a5301bb98ba6909e86fc295ef7df5c9c4de94d6567a12668c909af3ef12"
        }
      }
    }
  },
  "selection": "Max exact joint-state validation accuracy, then lower validation CE, then earlier update; selection_frozen persisted before test/x4/x8; all3 seeds, never best-of-seed",
  "plateau": "Eligibility>=10000; every250 completed updates; trailing2000 best-val gain<0.001 (0.1pp) AND first250/final250 relative train-loss mean improvement<0.001 (0.1%); strict thresholds; cap60000; calibrated rule from amendment#4, not a proof of convergence",
  "marginal_optimal_reference": {
    "name": "marginal-optimal loss ln(3721)",
    "nats": 8.221747728346623,
    "display": "marginal-optimal loss ln(3721) = 8.221748 nats"
  },
  "reference_correction": "Task shorthand8.224 corrected before launch by live supervisor: ln(3721)=8.221747728346623, displayed8.221748 nats; 8.224 is only the task-spec shorthand, not an equality. Below-reference loss indicates learned predictive signal, not by itself deep computation.",
  "supersession": "Matched1e-3 for BOTH arms supersedes pilot per-architecture E97@2e-4/GDN2@1e-3 reading, not archived evidence. X2b5e-5, X2c failed pilots, X3 symmetric8M grid/difficulty/one-step and ablation results all stand unchanged. No extra ablations/diagnostics launched by X4.",
  "capacity": "If free capacity <400GiB at a checkpoint check, retain best-val chain, every10K grid, latest250 rolling, own stop, legacy fixed {1000,4000,8000,12000,16000,18269}, and all known headline stop steps; scan published frozen selections each prune. Each prune records removed steps and retained hashes. Any cross-arm panel cell whose checkpoint was already pruned before its step became required is reported as unavailable/pruned - never imputed, never re-created by interpolation, never silently dropped. Stop steps broadcast via selection_frozen.json; running arms scan each250 check.",
  "storage": "Binary weights only off-repo at /mnt/nvme1n1/erikg/invertible_stage4_13b/<run>/; receipts bind absolute paths, bytes and SHA256. Capacity checks target actual weights filesystem. JSON/logs/pools remain this results root. No older roots moved or pruned; X3 retention unchanged.",
  "resources": "Six concurrent single-GPU processes on six exclusive scripts/gpu_lease.sh leases explicitly approved; single-writer implementation/debugging; local lambda01 only",
  "basis": {
    "task_sha256": "45e4ad754407166b001b75b3975d8bf6b50daf7734c6faa49499964c32d1993c",
    "X2b-report.md_sha256": "1fb298497e173d9693f0bf9344f69bc0d9ddbb3e98d6af6b24d1c38f2785910e",
    "X2c-report.md_sha256": "a031ca68d54f15bdcbff1b010bb1e8e023414075cb0ee80117a7aecb78c8af47",
    "X3-report.md_sha256": "6f80f5e4f57bfebca5a2ed5b6c8156af8ba7d57b05a73a26d43eccdabe156533"
  },
  "resilience_requirements": "Not applicable: no DiLoCo behavior changed, no Frontier/Slurm or WorksGood service"
}
```

Fresh released weights and cleared AdamW states, not pilot continuation. Both arms matched1e-3, warmup300 then constant, batch32, wd.01, clip.5. Fresh procedural batches; unchanged frozen evaluation. Causal right-pad-to16; pad logits never scored. Validation-only selection frozen before test. Calibrated plateau eligibility10K, trailing2000 val<0.1pp AND train-window<0.1%, cap60K. No new arithmetic controls or ablations in this stage.

## Three-seed final comparison (mean ± sample SD; SEM in JSON)

| Arm / LR | Best val % | Val CE | Test1x % / CE | x4 % / CE | x8 % / CE | Stop updates | Named references |
|---|---:|---:|---:|---:|---:|---|---|

## Per-seed selected results

| Run | Best val % / CE | Selected | Stop / reason | Test1x % / CE | x4 % / CE | x8 % / CE | Named references |
|---|---:|---:|---|---:|---:|---:|---|
| headline_e97_seed0 | not_completed | — | — | — | — | — | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | not_completed | — | — | — | — | — | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | not_completed | — | — | — | — | — | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | not_completed | — | — | — | — | — | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | not_completed | — | — | — | — | — | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | not_completed | — | — | — | — | — | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |

## Matched-step panels and union of stopping steps

Never train beyond a stop to fill a cell; pruned-before-required cells are unavailable, never imputed. Selection panels above are not equal-step comparisons.

| Run | Step | Test1x % / CE | x4 % / CE | x8 % / CE | Named references |
|---|---:|---:|---:|---:|---|

## Coordinates, depth-relevant post-suffix7, and cost

| Run | x1x % | y1x % | Terminal1x % | Post7 1x % | Post7 x4 % | Post7 x8 % | Parameters | Real tokens | Train tokens/s | Wall seconds | Named references |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

Real tokens include initialization, exclude pad overhead. Per-layer matrix-state geometry and flags are in configs. Seed-paired contrasts/counts and all full-vocabulary CE are retained in JSON.

## Interpretation across the X-series

Incomplete/fail-closed: no final architecture conclusion is supported.

## Receipts, storage, and validation

Each run binds command, GPU lease/UUID, source checkpoint/args/runner hashes, optimizer freshness, paired frozen NPZ/input identities, every actual update/validation boundary, selected-before-test receipt, binary SHA256 and required-set pruning ledgers. NVMe weights are local off-repo artifacts; textual evidence and pools stay in the tracked results root. No old result roots modified. See receipts/cpu_tests.txt, launch_commit.json, environment.txt, final_audit.json and validation_commands.txt. Independent acceptance review remains required.

Local lambda01; six exclusive GPU leases explicitly approved; no Slurm, Frontier, or WorksGood service. Resilient DiLoCo requirement IDs: not applicable (no resilient behavior changed).
