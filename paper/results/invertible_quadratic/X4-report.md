# X4 — released 1.3B pair, matched lr 1e-3

Status: **complete**.

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
| e97 / 0.001 | 0.290934% ± 0.081750% | 8.187417 ± 0.004465 | 0.268046% ± 0.090547% / 8.188895 ± 0.005370 | 0.088755% ± 0.022277% / 8.222450 ± 0.000761 | 0.056330% ± 0.010493% / 8.227873 ± 0.000715 | [10000, 10750, 10500] | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| gdn2 / 0.001 | 0.237528% ± 0.286241% | 8.222555 ± 0.050120 | 0.227356% ± 0.290923% / 8.223262 ± 0.049952 | 0.078710% ± 0.074176% / 8.267019 ± 0.012408 | 0.055313% ± 0.038397% / 8.274224 ± 0.006709 | [10000, 10000, 12000] | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |

## Per-seed selected results

| Run | Best val % / CE | Selected | Stop / reason | Test1x % / CE | x4 % / CE | x8 % / CE | Named references |
|---|---:|---:|---|---:|---:|---:|---|
| headline_e97_seed0 | 0.254822% / 8.190513 | 9250 | 10000 / plateau | 0.233459% / 8.192425 | 0.069809% / 8.223181 | 0.048637% / 8.228573 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 0.233459% / 8.189439 | 10500 | 10750 / plateau | 0.199890% / 8.191545 | 0.083160% / 8.222506 | 0.052071% / 8.227902 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 0.384521% / 8.182298 | 10000 | 10500 / plateau | 0.370789% / 8.182716 | 0.113297% / 8.221662 | 0.068283% / 8.227144 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 0.086975% / 8.246042 | 10000 | 10000 / plateau | 0.090027% / 8.245719 | 0.045776% / 8.273456 | 0.037575% / 8.278212 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 0.057983% / 8.256618 | 3250 | 10000 / plateau | 0.030518% / 8.258045 | 0.026703% / 8.274886 | 0.028992% / 8.277981 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 0.567627% / 8.165004 | 12000 | 12000 / plateau | 0.561523% / 8.166023 | 0.163651% / 8.252715 | 0.099373% / 8.266478 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |

## Matched-step panels and union of stopping steps

Never train beyond a stop to fill a cell; pruned-before-required cells are unavailable, never imputed. Fixed steps16000/18269 and20K–60K are not reached by any run. Selection panels above are not equal-step comparisons.

| Run | Step | Test1x % / CE | x4 % / CE | x8 % / CE | Named references |
|---|---:|---:|---:|---:|---|
| headline_e97_seed0 | 1000 | 0.038147% / 8.238213 | 0.026321% / 8.237921 | 0.026131% / 8.238349 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed0 | 4000 | 0.032043% / 8.227817 | 0.028229% / 8.232145 | 0.022697% / 8.232610 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed0 | 8000 | 0.144958% / 8.196345 | 0.063705% / 8.223934 | 0.044823% / 8.228891 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed0 | 10000 | 0.244141% / 8.187838 | 0.083542% / 8.221796 | 0.053978% / 8.227838 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed0 | 10500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed0 | 10750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed0 | 12000 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 1000 | 0.045776% / 8.235784 | 0.024033% / 8.237443 | 0.024796% / 8.237838 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 4000 | 0.028992% / 8.229621 | 0.026703% / 8.231875 | 0.029755% / 8.232651 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 8000 | 0.093079% / 8.203170 | 0.040054% / 8.226018 | 0.033188% / 8.229651 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 10000 | 0.195312% / 8.191131 | 0.067520% / 8.223450 | 0.047302% / 8.228463 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 10500 | 0.199890% / 8.191545 | 0.083160% / 8.222506 | 0.052071% / 8.227902 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 10750 | 0.239563% / 8.190419 | 0.082397% / 8.222485 | 0.057411% / 8.228327 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 12000 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 1000 | 0.030518% / 8.236677 | 0.026321% / 8.238790 | 0.024796% / 8.239426 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 4000 | 0.039673% / 8.210585 | 0.040054% / 8.227917 | 0.029182% / 8.231318 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 8000 | 0.285339% / 8.186957 | 0.086594% / 8.223213 | 0.060654% / 8.228416 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 10000 | 0.370789% / 8.182716 | 0.113297% / 8.221662 | 0.068283% / 8.227144 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 10500 | 0.392151% / 8.182751 | 0.107574% / 8.221289 | 0.070763% / 8.227613 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 10750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 12000 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 1000 | 0.025940% / 8.283833 | 0.024033% / 8.284862 | 0.022507% / 8.285512 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 4000 | 0.038147% / 8.256615 | 0.025177% / 8.274883 | 0.031662% / 8.277300 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 8000 | 0.042725% / 8.254222 | 0.035095% / 8.273309 | 0.030327% / 8.277201 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 10000 | 0.090027% / 8.245719 | 0.045776% / 8.273456 | 0.037575% / 8.278212 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 10500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 10750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 12000 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 1000 | 0.028992% / 8.282714 | 0.037003% / 8.286677 | 0.024605% / 8.286473 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 4000 | 0.039673% / 8.256803 | 0.029373% / 8.274266 | 0.025940% / 8.278119 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 8000 | 0.028992% / 8.251113 | 0.033951% / 8.275012 | 0.028801% / 8.278400 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 10000 | 0.033569% / 8.252317 | 0.032043% / 8.272528 | 0.032234% / 8.275964 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 10500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 10750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 12000 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 1000 | 0.030518% / 8.292104 | 0.024414% / 8.289998 | 0.026321% / 8.290459 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 4000 | 0.090027% / 8.246507 | 0.043869% / 8.273535 | 0.035095% / 8.276771 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 8000 | 0.363159% / 8.191889 | 0.100708% / 8.260084 | 0.068665% / 8.269830 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 10000 | 0.454712% / 8.175762 | 0.125122% / 8.257415 | 0.083160% / 8.270125 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 10500 | 0.477600% / 8.175132 | 0.138092% / 8.252799 | 0.084877% / 8.266014 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 10750 | 0.526428% / 8.171138 | 0.138092% / 8.253031 | 0.084686% / 8.266835 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 12000 | 0.561523% / 8.166023 | 0.163651% / 8.252715 | 0.099373% / 8.266478 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |

## Coordinates, depth-relevant post-suffix7, and cost

| Run | x1x % | y1x % | Terminal1x % | Post7 1x % | Post7 x4 % | Post7 x8 % | Parameters | Real tokens | Train tokens/s | Wall seconds | Named references |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| headline_e97_seed0 | 2.438354% | 1.828003% | 0.000000% | 0.030924% | 0.025964% | 0.024991% | 1286589072 | 41600000 | 7556.1 | 5979.9 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed1 | 2.362061% | 1.794434% | 0.000000% | 0.029297% | 0.026739% | 0.025952% | 1286589072 | 44720000 | 7463.2 | 6533.9 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_e97_seed2 | 2.484131% | 2.015686% | 0.000000% | 0.027669% | 0.031002% | 0.028067% | 1286589072 | 43680000 | 7736.0 | 6183.1 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed0 | 2.432251% | 1.699829% | 0.000000% | 0.026042% | 0.024414% | 0.029028% | 1286713448 | 41600000 | 7911.5 | 5708.7 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed1 | 2.464294% | 1.649475% | 0.000000% | 0.019531% | 0.024414% | 0.026721% | 1286713448 | 41600000 | 8175.5 | 5544.8 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |
| headline_gdn2_seed2 | 2.728271% | 2.096558% | 0.000000% | 0.021159% | 0.031390% | 0.026144% | 1286713448 | 49920000 | 7582.1 | 7145.2 | marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874% |

Real tokens include initialization, exclude pad overhead. Per-layer matrix-state geometry and flags are in configs. Seed-paired contrasts/counts and all full-vocabulary CE are retained in JSON.

## Observed matched-lr reading

All-three-seed selected test1x point estimate favors E97: E97 0.268046%, GDN2 0.227356%. Paired E97−GDN2 mean difference 0.040690 ± 0.200839 pp (sample SD): not an established architecture advantage. Their selected test1x CE means are 8.188895/8.223262 nats; post7 joint1x means are 0.029297%/0.022244%. Named references: marginal-optimal loss ln(3721) = 8.221748 nats; suffix floor 0.026874%. Read these with x4/x8 and matched-step panels, not as a best-seed or equal-step selection claim.

Both arms' post7 scores remain near the suffix floor and terminal joint accuracy is zero for every seed. E97's below-marginal1x CE is predictive signal concentrated in early drivers, not evidence of successful deep recurrence; mean extrapolation CE is at/above the marginal reference for both arms.

## Interpretation across the X-series

The released-pair matched1e-3 measurements above must be read alongside X2b's matched5e-5 fine-tunes, not as an LR-invariant architecture verdict. X2b favored GDN2, whereas X4's small reversed point-estimate ordering is dominated by seed uncertainty; neither establishes practical deep-map computation. X2b used a1% train-window plateau threshold versus X4's calibrated0.1%, so the cross-stage change is not a controlled LR-only effect. X2's frozen128 training pool memorization is superseded as primary evidence by fresh-data X2b; X2c's failed3K rescue gates remain failed gates, not completed full runs. X3's symmetric nominal8M (measured9.90M/9.887M) tuned grid selects1e-3 for both architectures on p61; both best-grid and matched-LR readings remain near the suffix reference on post7/x4/x8. Its p13/p31 ladder shows easier-task signal without establishing deep invertible-map computation. Late one-step arithmetic successes do not certify arithmetic mastery at an earlier long-task plateau, and failed arithmetic controls block a depth-specific interpretation. X2b's single-seed tanh-removal contrast is exploratory, not a causal recurrent-nonlinearity result. Full pretrained vocabulary/AdamW fine-tunes and task-sized-vocabulary/ScheduleFree scratch runs are different experiment classes; raw scale comparisons are not architecture-only effects. Differences in recurrence state size, pretraining and optimization remain confounds. Loss departures below the named marginal-optimal reference show predictive signal but do not by themselves establish robust deep computation, universal expressivity, impossibility, or CMA causality. The calibrated task-observed stopping rule is not proof of eventual convergence.

## Receipts, storage, and validation

Each run binds command, GPU lease/UUID, source checkpoint/args/runner hashes, optimizer freshness, paired frozen NPZ/input identities, every actual update/validation boundary, selected-before-test receipt, binary SHA256 and required-set pruning ledgers. NVMe weights are local off-repo artifacts; textual evidence and pools stay in the tracked results root. No old result roots modified. See receipts/cpu_tests.txt, receipts/cpu_tests_final.txt, receipts/launch_commit.json, receipts/environment.txt, receipts/final_audit.json and receipts/validation_commands.txt. Training/evaluation sources remained identical to the prelaunch commit throughout; report-generation wording alone was refined after launch. Final audit replays all six stops, window means, selected-before-test chronology, exact observation counts and required checkpoint hashes; all six training and six cross-evaluation processes exited0. No capacity prune was needed (NVMe stayed above400GiB free). Independent acceptance review remains required.

Local lambda01; six exclusive GPU leases explicitly approved; no Slurm, Frontier, or WorksGood service. Resilient DiLoCo requirement IDs: not applicable (no resilient behavior changed).
