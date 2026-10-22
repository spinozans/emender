# X21E first-tranche results

Native baseline uses bf16 autocast but fp32 recurrent state accumulators.
No training. Full position metrics and every minibatch receipt are retained.

|Champion|State precision|Surviving columns at 1/2/4/8/16/32x|Failure/censoring|
|---|---|---|---|
|X10-adamw|native|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X10-adamw|wrapper-fp32|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X10-adamw|bf16|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X10-adamw|fp8|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X10-adamw|int8|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X10-adamw|int4|[1, 1, 1, 1, 1, 1]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 16, 'depth': 16}|
|X11-GDN2|native|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X11-GDN2|wrapper-fp32|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X11-GDN2|bf16|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X11-GDN2|fp8|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 16}|
|X11-GDN2|int8|[3, 3, 3, 3, 3, 3]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 16, 'depth': 16}|
|X11-GDN2|int4|[0, 0, 0, 0, 0, 0]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 16, 'depth': 16}|
|X12-GDN2|native|[6, 5, 6, 5, 5, 6]|{'status': 'observed-failure', 'multiplier': 2, 'training_horizon': 64, 'depth': 128}|
|X12-GDN2|wrapper-fp32|[5, 5, 6, 5, 5, 6]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|
|X12-GDN2|bf16|[5, 5, 6, 5, 5, 6]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|
|X12-GDN2|fp8|[5, 5, 5, 5, 5, 5]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|
|X12-GDN2|int8|[3, 3, 3, 3, 3, 3]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|
|X12-GDN2|int4|[0, 0, 0, 0, 0, 0]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|
|X13-seed2|native|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X13-seed2|wrapper-fp32|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X13-seed2|bf16|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X13-seed2|fp8|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X13-seed2|int8|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X13-seed2|int4|[0, 0, 0, 0, 0, 0]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|
|X16E-curriculum|native|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X16E-curriculum|wrapper-fp32|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X16E-curriculum|bf16|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X16E-curriculum|fp8|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X16E-curriculum|int8|[4, 4, 4, 4, 4, 4]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X16E-curriculum|int4|[0, 0, 0, 0, 0, 0]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|
|X17-wean|native|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X17-wean|wrapper-fp32|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X17-wean|bf16|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X17-wean|fp8|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X17-wean|int8|[2, 2, 2, 2, 2, 2]|{'status': 'right-censored', 'beyond_multiplier': 32, 'training_horizon': 64}|
|X17-wean|int4|[1, 1, 1, 1, 1, 1]|{'status': 'already-failed-1x', 'multiplier': 1, 'training_horizon': 64, 'depth': 64}|

## Alignment
See summary.json for every surviving column and both start/suffix offsets.
No designated events: structure-relative is not identifiable. Ties ambiguous.

## Wrapper controls
{
  "X10-adamw": {
    "integer_counts_exact": false,
    "joint_count_deltas": [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "loss_delta": -0.00021282577764925037
  },
  "X11-GDN2": {
    "integer_counts_exact": false,
    "joint_count_deltas": [
      0,
      0,
      0,
      0,
      -2,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "loss_delta": 6.724092644017077e-05
  },
  "X12-GDN2": {
    "integer_counts_exact": false,
    "joint_count_deltas": [
      0,
      0,
      0,
      0,
      0,
      -7,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      -1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ],
    "loss_delta": 2.2277233939327346e-05
  },
  "X13-seed2": {
    "integer_counts_exact": false,
    "joint_count_deltas": [
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "loss_delta": 2.1162668706331544e-05
  },
  "X16E-curriculum": {
    "integer_counts_exact": false,
    "joint_count_deltas": [
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      -1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      -1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      -1,
      0,
      0,
      -1,
      0,
      1,
      1,
      0,
      0
    ],
    "loss_delta": 0.00027175443636107133
  },
  "X17-wean": {
    "integer_counts_exact": false,
    "joint_count_deltas": [
      0,
      0,
      4,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      -1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      -1,
      0,
      0,
      0,
      0,
      1,
      0,
      -1,
      0,
      0,
      0,
      0,
      0,
      -2,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      -1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      -1
    ],
    "loss_delta": -0.000137892695414088
  }
}

Architecture comparisons are descriptive: T16 cold AdamW pairs; T64 cold
AdamW with seed2 vs seed0 confound. Staged overlays are not matched recipes.
Failure is loss of a native1x >=480/512 column; >32x is right-censored,
never proof of unbounded computation. Quantization includes integer scale metadata.
Resilient DiLoCo R/NDP/V21S/ISP requirements: not applicable; no conformance claim.
