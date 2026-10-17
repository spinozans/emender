# X3 — from-scratch architecture sweep on the invertible map

Status: **complete_audited**; 36/36 runs available. Exact p61 floor **0.0269%** (1/3721).

## Fairness (verbatim)
the grid is symmetric across architectures; any statement about an architecture's performance cites its best grid point AND the matched-lr reading.

## Amendment #4 / prelaunch registration
```json
{
  "amendment": 4,
  "registered_at": "2026-10-16T20:58:18.630812+00:00",
  "base_commit": "f8225a95f6f875902444de6678784b9c4c2398be",
  "authority": "/home/erikg/emender-scratch/fix-fanout/taskX3-8m-sweep.md; live supervisor geometry/recipe/matrix rulings",
  "task_sha256": "21868018a88ca8c1b33b8a098f3430d0ecedd0e22a4686d210627af33b20b17f",
  "matrix": [
    {
      "name": "primary_p61_e97_lr0.001_seed0",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 0,
      "lr": 0.001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.001_seed0",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 0,
      "lr": 0.001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.001_seed1",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 1,
      "lr": 0.001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.001_seed1",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 1,
      "lr": 0.001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.001_seed2",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 2,
      "lr": 0.001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.001_seed2",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 2,
      "lr": 0.001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.0003_seed0",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0003,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.0003_seed0",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0003,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.0003_seed1",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 1,
      "lr": 0.0003,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.0003_seed1",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 1,
      "lr": 0.0003,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.0003_seed2",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 2,
      "lr": 0.0003,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.0003_seed2",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 2,
      "lr": 0.0003,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.0001_seed0",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.0001_seed0",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.0001_seed1",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 1,
      "lr": 0.0001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.0001_seed1",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 1,
      "lr": 0.0001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_e97_lr0.0001_seed2",
      "arm": "e97",
      "p": 61,
      "drivers": 128,
      "seed": 2,
      "lr": 0.0001,
      "kind": "primary"
    },
    {
      "name": "primary_p61_gdn2_lr0.0001_seed2",
      "arm": "gdn2",
      "p": 61,
      "drivers": 128,
      "seed": 2,
      "lr": 0.0001,
      "kind": "primary"
    },
    {
      "name": "difficulty_p13_e97_lr0.001_seed0",
      "arm": "e97",
      "p": 13,
      "drivers": 128,
      "seed": 0,
      "lr": 0.001,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p13_gdn2_lr0.001_seed0",
      "arm": "gdn2",
      "p": 13,
      "drivers": 128,
      "seed": 0,
      "lr": 0.001,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p13_e97_lr0.0003_seed0",
      "arm": "e97",
      "p": 13,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0003,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p13_gdn2_lr0.0003_seed0",
      "arm": "gdn2",
      "p": 13,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0003,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p13_e97_lr0.0001_seed0",
      "arm": "e97",
      "p": 13,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0001,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p13_gdn2_lr0.0001_seed0",
      "arm": "gdn2",
      "p": 13,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0001,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p31_e97_lr0.001_seed0",
      "arm": "e97",
      "p": 31,
      "drivers": 128,
      "seed": 0,
      "lr": 0.001,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p31_gdn2_lr0.001_seed0",
      "arm": "gdn2",
      "p": 31,
      "drivers": 128,
      "seed": 0,
      "lr": 0.001,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p31_e97_lr0.0003_seed0",
      "arm": "e97",
      "p": 31,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0003,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p31_gdn2_lr0.0003_seed0",
      "arm": "gdn2",
      "p": 31,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0003,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p31_e97_lr0.0001_seed0",
      "arm": "e97",
      "p": 31,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0001,
      "kind": "difficulty"
    },
    {
      "name": "difficulty_p31_gdn2_lr0.0001_seed0",
      "arm": "gdn2",
      "p": 31,
      "drivers": 128,
      "seed": 0,
      "lr": 0.0001,
      "kind": "difficulty"
    },
    {
      "name": "one_step_p61_e97_lr0.001_seed0",
      "arm": "e97",
      "p": 61,
      "drivers": 1,
      "seed": 0,
      "lr": 0.001,
      "kind": "one_step"
    },
    {
      "name": "one_step_p61_gdn2_lr0.001_seed0",
      "arm": "gdn2",
      "p": 61,
      "drivers": 1,
      "seed": 0,
      "lr": 0.001,
      "kind": "one_step"
    },
    {
      "name": "one_step_p61_e97_lr0.0003_seed0",
      "arm": "e97",
      "p": 61,
      "drivers": 1,
      "seed": 0,
      "lr": 0.0003,
      "kind": "one_step"
    },
    {
      "name": "one_step_p61_gdn2_lr0.0003_seed0",
      "arm": "gdn2",
      "p": 61,
      "drivers": 1,
      "seed": 0,
      "lr": 0.0003,
      "kind": "one_step"
    },
    {
      "name": "one_step_p61_e97_lr0.0001_seed0",
      "arm": "e97",
      "p": 61,
      "drivers": 1,
      "seed": 0,
      "lr": 0.0001,
      "kind": "one_step"
    },
    {
      "name": "one_step_p61_gdn2_lr0.0001_seed0",
      "arm": "gdn2",
      "p": 61,
      "drivers": 1,
      "seed": 0,
      "lr": 0.0001,
      "kind": "one_step"
    }
  ],
  "models": {
    "e97": {
      "parameters": 9900800,
      "mixer_class": "E97SplitEditLayer",
      "dim": 256,
      "depth": 4,
      "mlp_ratio": 2,
      "expansion": 1,
      "state_per_layer": [
        32,
        32,
        32
      ],
      "total_state_elements": 131072,
      "allow_neg_eigval": null,
      "state_activation": "tanh"
    },
    "gdn2": {
      "parameters": 9887056,
      "mixer_class": "FLAGatedDeltaNetLayer",
      "dim": 256,
      "depth": 4,
      "mlp_ratio": 2,
      "expansion": 8.625,
      "state_per_layer": [
        2,
        128,
        1104
      ],
      "total_state_elements": 1130496,
      "allow_neg_eigval": true,
      "state_activation": "linear"
    }
  },
  "geometry_correction": "Prelaunch inspection rejected legacy gdn-8m: it wraps fla.DeltaNet and silently ignores allow_neg_eigval. Approved replacement is actual FLAGatedDeltaNetLayer/fla.GatedDeltaNet, expansion8.625 chosen by instantiated parameter counts at dim256/depth4; wrapper default2 heads, key128, value1104. No legacy-gdn training performed. E97 retains approved H32/N32/expansion1. Parameter matching is not state-size matching or a one-factor mechanism ablation.",
  "parameter_gap_gdn2_relative_e97": -0.0013881706528765525,
  "recipe": "Prior scratch8M suite: AdamWScheduleFree betas(.9,.95), wd.01, clip1, bf16 autocast/fp32 parameters, batch32, no warmup; every evaluation and checkpoint uses ScheduleFree eval/x-mode, training resumes y-mode; all trainable tied task-sized heads; no pretrained weights or optimizer state loaded",
  "instrument_difference": "Dedicated trainable3721-class scratch vocab vs full pretrained vocab in X2 1.3B fine-tunes. Separate experiment classes, not directly numerically comparable; 1/p^2 remains reference. Small-prime diagnostics use p^2 vocab with unchanged backbone geometry.",
  "plateau": "Eligibility>=10000; every250 boundaries; over trailing2000 best-val gain<0.001 (0.1pp) AND first250/final250 relative train-loss mean improvement<0.001 (0.1%); hard cap60000. Strict thresholds; no change to archived X2 stops.",
  "calibration_basis": {
    "headline_e97_identity_seed0": {
      "step": 10000,
      "first250": 8.682504413604736,
      "final250": 8.677132766723632,
      "relative_loss_improvement": 0.0006186748229792638,
      "best_val_gain_2000": 0.0005035400390625,
      "validation_sha256": "d266be8b3e0ec6d2b23e3e5d68c1d063673f9da04e9a19f59792cac81ec9a46f"
    },
    "headline_e97_seed0": {
      "step": 10000,
      "first250": 8.681537109375,
      "final250": 8.676050342559815,
      "relative_loss_improvement": 0.0006320040732487301,
      "best_val_gain_2000": 0.0009613037109375,
      "validation_sha256": "c4d2a7570c253625c0b7969ad9948fab6b6ed99ce9f1d8b09cf2d495b6de7632"
    },
    "headline_e97_seed1": {
      "step": 10000,
      "first250": 8.682783504486084,
      "final250": 8.67702156829834,
      "relative_loss_improvement": 0.0006636047282264856,
      "best_val_gain_2000": 0.0009918212890625,
      "validation_sha256": "39ed84fa5d16fce1098df0151307f9d3b46af6c08efabf60fbf1f89fc9048960"
    },
    "headline_e97_seed2": {
      "step": 10000,
      "first250": 8.6810029296875,
      "final250": 8.675533992767335,
      "relative_loss_improvement": 0.0006299890651415673,
      "best_val_gain_2000": 0.0006256103515625,
      "validation_sha256": "637c97120dda4941fed07e7ad712bd58325faed179406d364de1565204a30b03"
    },
    "headline_gdn2_seed0": {
      "step": 21500,
      "first250": 8.061312046051025,
      "final250": 8.054002422332763,
      "relative_loss_improvement": 0.0009067535999729606,
      "best_val_gain_2000": 0.0009613037109375,
      "validation_sha256": "c840703b10d67c00ebee9b424170fe10cb4b42e8003ea2881cffe1f7e8ba688e"
    },
    "headline_gdn2_seed1": {
      "step": 23750,
      "first250": 8.039898818969727,
      "final250": 8.033143711090087,
      "relative_loss_improvement": 0.0008401981208645538,
      "best_val_gain_2000": 0.00048828125,
      "validation_sha256": "040e3c9c396b9d44976fe919acddb16c5014c253f76335935d43891bd9a6477a"
    },
    "headline_gdn2_seed2": {
      "step": 23500,
      "first250": 8.043895748138429,
      "final250": 8.035010208129883,
      "relative_loss_improvement": 0.0011046314232257472,
      "best_val_gain_2000": 0.0009307861328125,
      "validation_sha256": "7f2204472130d34adaa6c7551e9f05affba53bc14d3b449963451b3489da1fb6"
    },
    "pilot_1e-3_e97_seed0": {
      "passed": false,
      "first250_mean": 8.238054298400879,
      "final250_mean": 8.231817073822022,
      "relative_improvement": 0.0007571235091358973,
      "ols_slope_nats_per_update": -3.3295951135953712e-06,
      "window_start_update": 1001,
      "window_end_update": 3000,
      "receipt_sha256": "386d4a7edb5f8090a655bed1b36e7563baf2e8663d1cb55e2820c55468905b02"
    },
    "pilot_1e-3_gdn2_seed0": {
      "passed": false,
      "first250_mean": 8.285579166412354,
      "final250_mean": 8.258979866027833,
      "relative_improvement": 0.003210312743416729,
      "ols_slope_nats_per_update": -1.626006655834092e-05,
      "window_start_update": 1001,
      "window_end_update": 3000,
      "receipt_sha256": "0ccf70ed65089429f2e384e38d742eb7b1495a5fe07254036d065d7f485f2924"
    },
    "pilot_2e-4_e97_seed0": {
      "passed": false,
      "first250_mean": 8.2153642578125,
      "final250_mean": 8.203115364074707,
      "relative_improvement": 0.001490973906135016,
      "ols_slope_nats_per_update": -6.395560714538451e-06,
      "window_start_update": 1001,
      "window_end_update": 3000,
      "receipt_sha256": "79dad2cc8d80aa1d888e45cf40cea049a3778c95cd63433c4d058350c0e2c03d"
    },
    "pilot_2e-4_gdn2_seed0": {
      "passed": false,
      "first250_mean": 8.229950958251953,
      "final250_mean": 8.220161853790284,
      "relative_improvement": 0.0011894486991874313,
      "ols_slope_nats_per_update": -5.2645366974513125e-06,
      "window_start_update": 1001,
      "window_end_update": 3000,
      "receipt_sha256": "88a8c6a4e4034fc8d136ea1f0f1769dcdc9f7005fc480e68561eb1513104ac67"
    }
  },
  "calibration_rationale": "X2c E97 loss-window movement0.1491% at2e-4 and0.0757% at1e-3; GDN2 0.1189% and0.3210%. Prior1% threshold called all stagnant despite negative OLS slopes; calibrated0.1% resolves movement of this order. X2b terminal windows retained above. Task-observed calibration on released fine-tunes is disclosed, not an assertion of scratch convergence.",
  "data": "Fresh procedural batch/update, seed-paired training streams across architectures/LRs; frozen128 archived train(unused)/512val/512test1x/512test4x/512test8x; p61 exact X2 input and NPZ hashes. One-step has only test1x; difficulty seed0 allLRs.",
  "selection": "Per run maximum validation exact-state accuracy, lower validation CE, earlier update. Per architecture LR chosen by mean three-seed best validation accuracy, lower mean valCE, then smaller LR. No test-based LR/step/seed selection. Full matched-LR grid and best-LR views share the same data.",
  "fairness_verbatim": "the grid is symmetric across architectures; any statement about an architecture's performance cites its best grid point AND the matched-lr reading.",
  "capacity": "Every1000 plus best-val chain and stop; if free<400GiB prune to step0/current/current best/every10000, retaining checkpoint identities/hashes in ledger; never fabricate pruned cells. Fail closed below8GiB. Three co-tenants/card under exclusive leases, no preemption.",
  "resilience_requirements": "Not applicable: standalone local lambda01 single-GPU training runs, no DiLoCo behavior, Frontier, Slurm, or WorksGood service used."
}
```

## Symmetric primary LR grid — three seeds, mean ± sample SD
Validation-only LR choice; all test panels use each run's best-validation checkpoint. SEM and paired differences are in JSON. Percent units below; no best-of-seed claims.

| Arm | LR | Best val % | Test1x % | Test4x % | Test8x % | Post-suffix7 1x % | Stop updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| e97 | 0.001 | 0.0488 ± 0.0100 | 0.0356 ± 0.0023 | 0.0275 ± 0.0042 | 0.0275 ± 0.0010 | 0.0250 ± 0.0009 | 10000.0000 ± 0.0000 |
| e97 | 0.0003 | 0.0392 ± 0.0009 | 0.0219 ± 0.0023 | 0.0224 ± 0.0023 | 0.0266 ± 0.0023 | 0.0217 ± 0.0025 | 10000.0000 ± 0.0000 |
| e97 | 0.0001 | 0.0427 ± 0.0040 | 0.0290 ± 0.0026 | 0.0249 ± 0.0033 | 0.0266 ± 0.0036 | 0.0304 ± 0.0034 | 13083.3333 ± 520.4165 |
| gdn2 | 0.001 | 0.0595 ± 0.0055 | 0.0493 ± 0.0112 | 0.0323 ± 0.0039 | 0.0298 ± 0.0038 | 0.0304 ± 0.0095 | 10000.0000 ± 0.0000 |
| gdn2 | 0.0003 | 0.0544 ± 0.0032 | 0.0331 ± 0.0092 | 0.0312 ± 0.0017 | 0.0280 ± 0.0023 | 0.0190 ± 0.0034 | 10000.0000 ± 0.0000 |
| gdn2 | 0.0001 | 0.0529 ± 0.0038 | 0.0381 ± 0.0067 | 0.0313 ± 0.0049 | 0.0268 ± 0.0021 | 0.0260 ± 0.0049 | 10166.6667 ± 288.6751 |

## Both grid readings

- **e97 best validation LR=0.001**: test1x 0.0356 ± 0.0023%; x4 0.0275 ± 0.0042%; x8 0.0275 ± 0.0010%. The full matched-LR reading follows, not an alternative experiment.
- **gdn2 best validation LR=0.001**: test1x 0.0493 ± 0.0112%; x4 0.0323 ± 0.0039%; x8 0.0298 ± 0.0038%. The full matched-LR reading follows, not an alternative experiment.
- Matched LR=0.001: paired E97−GDN2 test1x -0.0137 ± 0.0106 pp (all3 paired seeds; positive favors E97); both arms' absolute values are in the full grid above.
- Matched LR=0.0003: paired E97−GDN2 test1x -0.0112 ± 0.0115 pp (all3 paired seeds; positive favors E97); both arms' absolute values are in the full grid above.
- Matched LR=0.0001: paired E97−GDN2 test1x -0.0092 ± 0.0081 pp (all3 paired seeds; positive favors E97); both arms' absolute values are in the full grid above.

## All primary seeds / selected steps / extrapolation / coordinates / suffix calibration

| Run | Val % | Selected | Stop/reason | Test1x % | x4 % | x8 % | x1x % | y1x % | Post7 1x % | Suffix7-only 1x % |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| primary_p61_e97_lr0.001_seed0 | 0.0473% | 9500 | 10000/plateau | 0.0351% | 0.0233% | 0.0265% | 2.3483% | 1.5991% | 0.0244% | 0.0244% |
| primary_p61_gdn2_lr0.001_seed0 | 0.0610% | 2000 | 10000/plateau | 0.0366% | 0.0290% | 0.0286% | 2.4033% | 1.7319% | 0.0195% | 0.0244% |
| primary_p61_e97_lr0.001_seed1 | 0.0397% | 8250 | 10000/plateau | 0.0336% | 0.0317% | 0.0275% | 2.0508% | 1.6449% | 0.0244% | 0.0179% |
| primary_p61_gdn2_lr0.001_seed1 | 0.0534% | 9750 | 10000/plateau | 0.0534% | 0.0366% | 0.0340% | 2.3422% | 1.6953% | 0.0374% | 0.0179% |
| primary_p61_e97_lr0.001_seed2 | 0.0595% | 9750 | 10000/plateau | 0.0381% | 0.0275% | 0.0284% | 2.4155% | 1.6525% | 0.0260% | 0.0260% |
| primary_p61_gdn2_lr0.001_seed2 | 0.0641% | 10000 | 10000/plateau | 0.0580% | 0.0313% | 0.0267% | 2.3666% | 1.6769% | 0.0342% | 0.0260% |
| primary_p61_e97_lr0.0003_seed0 | 0.0397% | 1000 | 10000/plateau | 0.0214% | 0.0248% | 0.0277% | 1.6754% | 1.5717% | 0.0212% | 0.0244% |
| primary_p61_gdn2_lr0.0003_seed0 | 0.0580% | 8250 | 10000/plateau | 0.0320% | 0.0313% | 0.0273% | 2.4033% | 1.5854% | 0.0163% | 0.0244% |
| primary_p61_e97_lr0.0003_seed1 | 0.0397% | 3000 | 10000/plateau | 0.0244% | 0.0202% | 0.0282% | 1.5701% | 1.6281% | 0.0244% | 0.0179% |
| primary_p61_gdn2_lr0.0003_seed1 | 0.0519% | 4750 | 10000/plateau | 0.0244% | 0.0328% | 0.0261% | 2.2903% | 1.5427% | 0.0179% | 0.0179% |
| primary_p61_e97_lr0.0003_seed2 | 0.0381% | 9750 | 10000/plateau | 0.0198% | 0.0221% | 0.0240% | 1.5503% | 1.6449% | 0.0195% | 0.0260% |
| primary_p61_gdn2_lr0.0003_seed2 | 0.0534% | 8000 | 10000/plateau | 0.0427% | 0.0294% | 0.0305% | 2.4384% | 1.6571% | 0.0228% | 0.0260% |
| primary_p61_e97_lr0.0001_seed0 | 0.0458% | 6750 | 12500/plateau | 0.0275% | 0.0221% | 0.0301% | 1.5198% | 1.5747% | 0.0277% | 0.0244% |
| primary_p61_gdn2_lr0.0001_seed0 | 0.0534% | 8750 | 10000/plateau | 0.0458% | 0.0324% | 0.0246% | 2.1683% | 1.7471% | 0.0309% | 0.0244% |
| primary_p61_e97_lr0.0001_seed1 | 0.0443% | 1500 | 13500/plateau | 0.0320% | 0.0286% | 0.0229% | 1.6312% | 1.7761% | 0.0342% | 0.0179% |
| primary_p61_gdn2_lr0.0001_seed1 | 0.0488% | 10000 | 10500/plateau | 0.0351% | 0.0355% | 0.0288% | 2.3102% | 1.6235% | 0.0212% | 0.0179% |
| primary_p61_e97_lr0.0001_seed2 | 0.0381% | 5250 | 13250/plateau | 0.0275% | 0.0240% | 0.0269% | 1.6739% | 1.7105% | 0.0293% | 0.0260% |
| primary_p61_gdn2_lr0.0001_seed2 | 0.0565% | 8500 | 10000/plateau | 0.0336% | 0.0259% | 0.0269% | 2.2614% | 1.5671% | 0.0260% | 0.0260% |

## Difficulty ladder and one-step arithmetic diagnostics (all LRs, seed0)

One-step has no depth or suffix7 panel. Small-prime arms retain backbone geometry but train task-sized p² embeddings/heads. Floors: p13=0.5917%, p31=0.1041%, p61=0.0269%.

| Run | Best val % | Selected | Stop/reason | Test1x % | x4 % | x8 % | x1x % | y1x % | Post7 1x % | Suffix7-only % |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| difficulty_p13_e97_lr0.001_seed0 | 3.0273% | 11250 | 11250/plateau | 3.0060% | 1.2009% | 0.9041% | 10.5560% | 9.9655% | 0.6022% | 0.5876% |
| difficulty_p13_gdn2_lr0.001_seed0 | 4.5639% | 10250 | 10750/plateau | 4.5380% | 1.5720% | 1.0733% | 11.8668% | 11.2457% | 0.6055% | 0.5876% |
| difficulty_p13_e97_lr0.0003_seed0 | 2.9404% | 19750 | 19750/plateau | 2.9144% | 1.1761% | 0.8965% | 10.1410% | 9.8312% | 0.5811% | 0.5876% |
| difficulty_p13_gdn2_lr0.0003_seed0 | 6.0928% | 26500 | 26750/plateau | 6.0379% | 1.9741% | 1.2678% | 12.7472% | 12.6740% | 0.5794% | 0.5876% |
| difficulty_p13_e97_lr0.0001_seed0 | 1.4771% | 14500 | 15000/plateau | 1.4038% | 0.7725% | 0.6842% | 8.2596% | 8.4625% | 0.6185% | 0.5876% |
| difficulty_p13_gdn2_lr0.0001_seed0 | 5.1300% | 41500 | 42000/plateau | 5.1636% | 1.6994% | 1.1435% | 11.9492% | 12.0728% | 0.6266% | 0.5876% |
| difficulty_p31_e97_lr0.001_seed0 | 0.8133% | 14750 | 15000/plateau | 0.7858% | 0.2808% | 0.1888% | 3.9246% | 3.9871% | 0.0814% | 0.1188% |
| difficulty_p31_gdn2_lr0.001_seed0 | 1.6998% | 8500 | 10000/plateau | 1.6663% | 0.5028% | 0.3033% | 4.8737% | 4.7974% | 0.1090% | 0.1188% |
| difficulty_p31_e97_lr0.0003_seed0 | 0.1495% | 6750 | 10000/plateau | 0.1266% | 0.1102% | 0.0980% | 3.9627% | 3.1693% | 0.1090% | 0.1188% |
| difficulty_p31_gdn2_lr0.0003_seed0 | 0.9140% | 10250 | 10250/plateau | 0.8850% | 0.2945% | 0.1987% | 4.0421% | 3.9719% | 0.0993% | 0.1188% |
| difficulty_p31_e97_lr0.0001_seed0 | 0.1328% | 1250 | 10000/plateau | 0.1083% | 0.1022% | 0.1080% | 3.1403% | 3.1555% | 0.1090% | 0.1188% |
| difficulty_p31_gdn2_lr0.0001_seed0 | 0.1877% | 8250 | 10000/plateau | 0.1480% | 0.1183% | 0.1036% | 3.9261% | 3.2455% | 0.1123% | 0.1188% |
| one_step_p61_e97_lr0.001_seed0 | 98.6328% | 59500 | 60000/hard_cap | 97.6562% | N/A | N/A | 100.0000% | 97.6562% | N/A | N/A |
| one_step_p61_gdn2_lr0.001_seed0 | 100.0000% | 18500 | 18500/plateau | 100.0000% | N/A | N/A | 100.0000% | 100.0000% | N/A | N/A |
| one_step_p61_e97_lr0.0003_seed0 | 2.1484% | 10250 | 13750/plateau | 1.7578% | N/A | N/A | 96.6797% | 1.7578% | N/A | N/A |
| one_step_p61_gdn2_lr0.0003_seed0 | 3.3203% | 9750 | 10250/plateau | 3.1250% | N/A | N/A | 99.8047% | 3.1250% | N/A | N/A |
| one_step_p61_e97_lr0.0001_seed0 | 2.5391% | 35000 | 36500/plateau | 1.3672% | N/A | N/A | 98.6328% | 1.3672% | N/A | N/A |
| one_step_p61_gdn2_lr0.0001_seed0 | 3.7109% | 16250 | 21000/plateau | 1.5625% | N/A | N/A | 96.8750% | 1.5625% | N/A | N/A |

## Suffix7-only calibration at every reported depth

Nontrained canonical(0,0) propagated using only the last7 drivers; scored after two or more omitted drivers (driver>=9). Identical frozen inputs give identical calibrations across architectures/LRs. Sequence-level SEM is in JSON; overlapping positions are not independent trials. One-step is N/A. Fresh procedural training is not filtered against frozen inputs: at one step the finite61³ input domain can be revisited, so this is an arithmetic/encoding control, not a novel-input generalization certificate.

| Field / seed | Floor % | Suffix7 1x % | Suffix7 x4 % | Suffix7 x8 % |
|---|---:|---:|---:|---:|
| p61 / seed0 | 0.0269% | 0.0244% | 0.0264% | 0.0229% |
| p61 / seed1 | 0.0269% | 0.0179% | 0.0287% | 0.0227% |
| p61 / seed2 | 0.0269% | 0.0260% | 0.0314% | 0.0300% |
| p13 / seed0 | 0.5917% | 0.5876% | 0.5685% | 0.5909% |
| p31 / seed0 | 0.1041% | 0.1188% | 0.0996% | 0.1046% |

## Cost and evidence

| Run | Parameters | Real train tokens | Train tokens/s | Wall seconds |
|---|---:|---:|---:|---:|
| primary_p61_e97_lr0.001_seed0 | 9900800 | 41600000 | 82579.4 | 751.7 |
| primary_p61_gdn2_lr0.001_seed0 | 9887056 | 41600000 | 98053.5 | 442.3 |
| primary_p61_e97_lr0.001_seed1 | 9900800 | 41600000 | 84120.2 | 741.8 |
| primary_p61_gdn2_lr0.001_seed1 | 9887056 | 41600000 | 102478.1 | 424.5 |
| primary_p61_e97_lr0.001_seed2 | 9900800 | 41600000 | 83497.5 | 751.3 |
| primary_p61_gdn2_lr0.001_seed2 | 9887056 | 41600000 | 106867.7 | 406.8 |
| primary_p61_e97_lr0.0003_seed0 | 9900800 | 41600000 | 79102.0 | 767.2 |
| primary_p61_gdn2_lr0.0003_seed0 | 9887056 | 41600000 | 104333.9 | 416.9 |
| primary_p61_e97_lr0.0003_seed1 | 9900800 | 41600000 | 80633.3 | 762.3 |
| primary_p61_gdn2_lr0.0003_seed1 | 9887056 | 41600000 | 99052.5 | 438.4 |
| primary_p61_e97_lr0.0003_seed2 | 9900800 | 41600000 | 84030.3 | 739.3 |
| primary_p61_gdn2_lr0.0003_seed2 | 9887056 | 41600000 | 101001.4 | 429.5 |
| primary_p61_e97_lr0.0001_seed0 | 9900800 | 52000000 | 82561.7 | 923.6 |
| primary_p61_gdn2_lr0.0001_seed0 | 9887056 | 41600000 | 100709.8 | 431.9 |
| primary_p61_e97_lr0.0001_seed1 | 9900800 | 56160000 | 87294.6 | 941.9 |
| primary_p61_gdn2_lr0.0001_seed1 | 9887056 | 43680000 | 102247.0 | 446.6 |
| primary_p61_e97_lr0.0001_seed2 | 9900800 | 55120000 | 86325.3 | 939.6 |
| primary_p61_gdn2_lr0.0001_seed2 | 9887056 | 41600000 | 100318.9 | 434.0 |
| difficulty_p13_e97_lr0.001_seed0 | 8991488 | 46800000 | 88614.9 | 796.2 |
| difficulty_p13_gdn2_lr0.001_seed0 | 8977744 | 44720000 | 104045.3 | 450.6 |
| difficulty_p13_e97_lr0.0003_seed0 | 8991488 | 82160000 | 97406.0 | 1268.8 |
| difficulty_p13_gdn2_lr0.0003_seed0 | 8977744 | 111280000 | 113328.0 | 1025.7 |
| difficulty_p13_e97_lr0.0001_seed0 | 8991488 | 62400000 | 90182.3 | 1021.5 |
| difficulty_p13_gdn2_lr0.0001_seed0 | 8977744 | 174720000 | 116415.0 | 1561.7 |
| difficulty_p31_e97_lr0.001_seed0 | 9194240 | 62400000 | 149314.5 | 752.8 |
| difficulty_p31_gdn2_lr0.001_seed0 | 9180496 | 41600000 | 124569.9 | 349.0 |
| difficulty_p31_e97_lr0.0003_seed0 | 9194240 | 41600000 | 155377.4 | 504.3 |
| difficulty_p31_gdn2_lr0.0003_seed0 | 9180496 | 42640000 | 122499.3 | 363.6 |
| difficulty_p31_e97_lr0.0001_seed0 | 9194240 | 41600000 | 124500.8 | 570.8 |
| difficulty_p31_gdn2_lr0.0001_seed0 | 9180496 | 41600000 | 120776.6 | 359.1 |
| one_step_p61_e97_lr0.001_seed0 | 9900800 | 5760000 | 5496.2 | 1215.5 |
| one_step_p61_gdn2_lr0.001_seed0 | 9887056 | 1776000 | 2884.8 | 639.7 |
| one_step_p61_e97_lr0.0003_seed0 | 9900800 | 1320000 | 4076.3 | 365.6 |
| one_step_p61_gdn2_lr0.0003_seed0 | 9887056 | 984000 | 2838.4 | 359.4 |
| one_step_p61_e97_lr0.0001_seed0 | 9900800 | 3504000 | 4918.9 | 815.9 |
| one_step_p61_gdn2_lr0.0001_seed0 | 9887056 | 2016000 | 2944.4 | 707.4 |

Full source, environment, lease/GPU UUID snapshots, exact commands, exit codes, fresh training losses, frozen pools/hash receipts, all validation boundaries, selected-before-test receipts, checkpoint hashes, and strict stopping-rule replay accompany the JSON. Binary checkpoints remain local ignored artifacts, not git payloads. Every1000 retention plus best-val chain/stop; pruning activates below400GiB and records hashes/removals. Smoke runs are instrument checks, excluded from the36-run experiment.

Resilient DiLoCo requirement IDs: not applicable (standalone local lambda01, no resilient behavior or Frontier/Slurm submission). No WorksGood service, workers, or dispatcher used. Independent acceptance review remains required.

## Architecture question and single-1.3B recommendation

This is a learning-efficiency comparison of two approximately parameter-matched scratch backbones with fixed-depth MLPs, not a representability impossibility result or a controlled state-nonlinearity ablation: recurrent state sizes and mixer plumbing differ. The full LR grid and validation-selected best LR must be read together. Full-sequence accuracy can benefit from early positions; the post-suffix7 panel is the depth-relevant check against1/p². A failed one-step arithmetic control blocks a depth-specific interpretation of the corresponding long-task failure. The calibrated plateau rule is a disclosed task-observed stopping criterion, not proof of eventual convergence.
The scratch task-sized output vocabulary and ScheduleFree recipe differ from X2's full-pretrained-vocabulary fine-tunes and AdamW warmup. Do not compare raw numeric accuracy across those experiment classes as an architecture-only effect. A single1.3B follow-up would be a separately authorized scale-transfer diagnostic, not a two-arm architecture comparison. Numerical diagnostic qualification and the conditional recommendation follow; no1.3B job is launched by X3.

### Observed diagnostic qualification (best-LR AND matched-LR views)

The primary best-LR and matched-LR results above do not demonstrate learned deep invertible-map computation: inspect their post7/extrapolation values beside the printed floor, not merely the elevated full-sequence score that includes early drivers. No controlled causal advantage of recurrent nonlinearity is established at this nominal8M budget (measured totals9.90M/9.887M). A null under these geometry/optimizer/stopping rules is not a proof of equal ultimate capabilities. In particular, one-step training may require far more updates than a primary run's plateau; a later arithmetic success cannot certify that the primary model already mastered arithmetic at its earlier stop.
**Single1.3B recommendation:** the receipt/selection/stopping plumbing is validated, but this sweep has not validated a successful architecture-discriminating protocol. Do not label one further large run an architecture confirmation of a scratch separation that was not observed. If a separately approved scale-transfer calibration is pursued, the validation-best LR below is the symmetric-grid candidate, not a proven transferable optimum. First require arithmetic calibration at the target instrument/precision, and preregister an eligibility/budget decision that addresses late one-step learning before spending the single deep-task run; changing amendment#4 or launching that run is outside X3. Retain fresh training, frozen pools, explicit vocabulary/optimizer class, validation-only selection, both grid readings, x4/x8 and suffix calibration. No1.3B job was launched.

- e97, validation-best LR=0.001: one-step test 97.6562%; long-task post7 joint1x 0.0250 ± 0.0009%, x4 0.0257 ± 0.0035%, x8 0.0263 ± 0.0013%, against floor 0.0269%. Best-LR headline and all three matched-LR contrasts are reported above; neither reading can be omitted.
- gdn2, validation-best LR=0.001: one-step test 100.0000%; long-task post7 joint1x 0.0304 ± 0.0095%, x4 0.0260 ± 0.0036%, x8 0.0274 ± 0.0030%, against floor 0.0269%. Best-LR headline and all three matched-LR contrasts are reported above; neither reading can be omitted.

| Matched LR | E97 one-step % | GDN2 one-step % | E97 post7 1x % | GDN2 post7 1x % |
|---|---:|---:|---:|---:|
| 0.001 | 97.6562% | 100.0000% | 0.0250 ± 0.0009 | 0.0304 ± 0.0095 |
| 0.0003 | 1.7578% | 3.1250% | 0.0217 ± 0.0025 | 0.0190 ± 0.0034 |
| 0.0001 | 1.3672% | 1.5625% | 0.0304 ± 0.0034 | 0.0260 ± 0.0049 |
