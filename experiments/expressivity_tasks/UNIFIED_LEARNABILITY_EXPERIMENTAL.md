<!-- AUTO-GENERATED (aggregate_unified_learnability.py). -->

_Spread-arm runs found: 60 / 60._

## 1. Accuracy — does spread-init + knob-LR close the learnability gap?

### Accuracy @ T=128  (mean±std over seeds [42, 123, 456]; train T=128)

| probe (corner) | best preset | generic learned | LSTM | spread klr1 | spread klr5 | spread klr10 | spread klr20 |
|---|---|---|---|---|---|---|---|
| s5_permutation (track) | track 0.834 | 0.061 | 0.979 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |
| anbncn_viability (count) | track 1.000 | 0.993 | 1.000 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |
| iterated_nonlinear_map (nonlin) | track 0.903 | 0.680 | 0.985 | 0.832±0.00 | 0.844±0.01 | 0.846±0.00 | 0.850±0.01 |
| flag_hold_recall (latch) | track 1.000 | 1.000 | 0.512 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |
| mixed_probe (ALL) | track 0.840 | 0.538 | 0.674 | 0.907±0.00 | 0.918±0.00 | 0.913±0.00 | 0.922±0.01 |

### Accuracy @ T=1024  (mean±std over seeds [42, 123, 456]; train T=128)

| probe (corner) | best preset | generic learned | LSTM | spread klr1 | spread klr5 | spread klr10 | spread klr20 |
|---|---|---|---|---|---|---|---|
| s5_permutation (track) | track 0.111 | 0.015 | 0.563 | 0.224±0.00 | 0.347±0.00 | 0.448±0.02 | 0.541±0.02 |
| anbncn_viability (count) | count 0.920 | 0.836 | 0.992 | 0.843±0.01 | 0.875±0.00 | 0.898±0.01 | 0.896±0.01 |
| iterated_nonlinear_map (nonlin) | track 0.893 | 0.671 | 0.983 | 0.823±0.00 | 0.836±0.01 | 0.838±0.00 | 0.840±0.01 |
| flag_hold_recall (latch) | latch 1.000 | 0.531 | 0.562 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |
| mixed_probe (ALL) | track 0.569 | 0.478 | 0.634 | 0.587±0.04 | 0.623±0.03 | 0.668±0.04 | 0.719±0.03 |

### Gap closure @ T=128: does spread+knob-LR match the best preset?

| probe (corner) | best preset | generic learned (matches?) | best spread arm (matches?) |
|---|---|---|---|
| s5_permutation (track) | track 0.834 | 0.061 (no) | spread-klr1 1.000 (YES) |
| anbncn_viability (count) | track 1.000 | 0.993 (YES) | spread-klr1 1.000 (YES) |
| iterated_nonlinear_map (nonlin) | track 0.903 | 0.680 (no) | spread-klr20 0.850 (no) |
| flag_hold_recall (latch) | track 1.000 | 1.000 (YES) | spread-klr1 1.000 (YES) |
| mixed_probe (ALL) | track 0.840 | 0.538 (no) | spread-klr20 0.922 (YES) |

### Gap closure @ T=1024: does spread+knob-LR match the best preset?

| probe (corner) | best preset | generic learned (matches?) | best spread arm (matches?) |
|---|---|---|---|
| s5_permutation (track) | track 0.111 | 0.015 (no) | spread-klr20 0.541 (YES) |
| anbncn_viability (count) | count 0.920 | 0.836 (no) | spread-klr10 0.898 (YES) |
| iterated_nonlinear_map (nonlin) | track 0.893 | 0.671 (no) | spread-klr20 0.840 (no) |
| flag_hold_recall (latch) | latch 1.000 | 0.531 (no) | spread-klr1 1.000 (YES) |
| mixed_probe (ALL) | track 0.569 | 0.478 (no) | spread-klr20 0.719 (YES) |

## 2. Knob drift — do heads HOLD their corner or drift to center?

### Knob drift — arm `unified-learned-spread-klr1`  (per-head, pooled over layers+seeds)

Each head starts on one of the four corners (round-robin). After training we classify each head to its NEAREST corner centroid (track/count/latch/nonlin/center) in normalized (lambda,beta,gamma) space. "held" = nearest corner is still a real corner (not center); "to center" = drifted to the generic compromise regime.

| probe | n heads | mean lambda | mean beta | mean gamma | %eig<0 | %held corner | %to center | mean |dlambda| | mean |dbeta| | mean |dgamma| |
|---|---|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 1.056 | 0.589 | 0.497 | 25% | 100% | 0% | 0.034 | 0.014 | 0.003 |
| anbncn_viability | 384 | 1.025 | 0.577 | 0.499 | 25% | 100% | 0% | 0.005 | 0.003 | 0.001 |
| iterated_nonlinear_map | 384 | 1.000 | 0.569 | 0.503 | 25% | 100% | 0% | 0.025 | 0.012 | 0.004 |
| flag_hold_recall | 384 | 1.023 | 0.575 | 0.500 | 25% | 100% | 0% | 0.002 | 0.002 | 0.001 |
| mixed_probe | 384 | 1.041 | 0.583 | 0.499 | 25% | 100% | 0% | 0.025 | 0.010 | 0.002 |

### Knob drift — arm `unified-learned-spread-klr5`  (per-head, pooled over layers+seeds)

Each head starts on one of the four corners (round-robin). After training we classify each head to its NEAREST corner centroid (track/count/latch/nonlin/center) in normalized (lambda,beta,gamma) space. "held" = nearest corner is still a real corner (not center); "to center" = drifted to the generic compromise regime.

| probe | n heads | mean lambda | mean beta | mean gamma | %eig<0 | %held corner | %to center | mean |dlambda| | mean |dbeta| | mean |dgamma| |
|---|---|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 1.062 | 0.616 | 0.495 | 25% | 100% | 0% | 0.049 | 0.041 | 0.007 |
| anbncn_viability | 384 | 1.030 | 0.580 | 0.497 | 25% | 100% | 0% | 0.017 | 0.012 | 0.006 |
| iterated_nonlinear_map | 384 | 0.935 | 0.540 | 0.516 | 25% | 100% | 0% | 0.091 | 0.050 | 0.021 |
| flag_hold_recall | 384 | 1.017 | 0.575 | 0.500 | 25% | 100% | 0% | 0.008 | 0.008 | 0.005 |
| mixed_probe | 384 | 1.016 | 0.598 | 0.503 | 25% | 100% | 0% | 0.048 | 0.041 | 0.011 |

### Knob drift — arm `unified-learned-spread-klr10`  (per-head, pooled over layers+seeds)

Each head starts on one of the four corners (round-robin). After training we classify each head to its NEAREST corner centroid (track/count/latch/nonlin/center) in normalized (lambda,beta,gamma) space. "held" = nearest corner is still a real corner (not center); "to center" = drifted to the generic compromise regime.

| probe | n heads | mean lambda | mean beta | mean gamma | %eig<0 | %held corner | %to center | mean |dlambda| | mean |dbeta| | mean |dgamma| |
|---|---|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 1.062 | 0.632 | 0.492 | 25% | 100% | 0% | 0.063 | 0.059 | 0.013 |
| anbncn_viability | 384 | 1.025 | 0.584 | 0.495 | 25% | 100% | 0% | 0.026 | 0.027 | 0.011 |
| iterated_nonlinear_map | 384 | 0.884 | 0.508 | 0.528 | 27% | 100% | 0% | 0.142 | 0.087 | 0.037 |
| flag_hold_recall | 384 | 1.008 | 0.576 | 0.500 | 25% | 100% | 0% | 0.017 | 0.016 | 0.011 |
| mixed_probe | 384 | 0.994 | 0.600 | 0.506 | 27% | 100% | 0% | 0.068 | 0.059 | 0.021 |

### Knob drift — arm `unified-learned-spread-klr20`  (per-head, pooled over layers+seeds)

Each head starts on one of the four corners (round-robin). After training we classify each head to its NEAREST corner centroid (track/count/latch/nonlin/center) in normalized (lambda,beta,gamma) space. "held" = nearest corner is still a real corner (not center); "to center" = drifted to the generic compromise regime.

| probe | n heads | mean lambda | mean beta | mean gamma | %eig<0 | %held corner | %to center | mean |dlambda| | mean |dbeta| | mean |dgamma| |
|---|---|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 1.050 | 0.654 | 0.488 | 25% | 100% | 0% | 0.074 | 0.085 | 0.024 |
| anbncn_viability | 384 | 1.019 | 0.593 | 0.490 | 25% | 100% | 0% | 0.042 | 0.053 | 0.025 |
| iterated_nonlinear_map | 384 | 0.824 | 0.489 | 0.543 | 33% | 100% | 0% | 0.203 | 0.130 | 0.063 |
| flag_hold_recall | 384 | 0.994 | 0.575 | 0.500 | 25% | 100% | 0% | 0.032 | 0.033 | 0.024 |
| mixed_probe | 384 | 0.954 | 0.588 | 0.514 | 30% | 100% | 0% | 0.106 | 0.086 | 0.040 |

## 3. Final corner occupancy (best-separating arms)

### Final corner occupancy — arm `unified-learned-spread-klr1` (nearest-centroid head counts)

| probe | track | count | latch | nonlin | center |
|---|---|---|---|---|---|
| s5_permutation | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| anbncn_viability | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| iterated_nonlinear_map | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| flag_hold_recall | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| mixed_probe | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |

### Final corner occupancy — arm `unified-learned-spread-klr5` (nearest-centroid head counts)

| probe | track | count | latch | nonlin | center |
|---|---|---|---|---|---|
| s5_permutation | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| anbncn_viability | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| iterated_nonlinear_map | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| flag_hold_recall | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| mixed_probe | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |

### Final corner occupancy — arm `unified-learned-spread-klr10` (nearest-centroid head counts)

| probe | track | count | latch | nonlin | center |
|---|---|---|---|---|---|
| s5_permutation | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| anbncn_viability | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| iterated_nonlinear_map | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| flag_hold_recall | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| mixed_probe | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |

### Final corner occupancy — arm `unified-learned-spread-klr20` (nearest-centroid head counts)

| probe | track | count | latch | nonlin | center |
|---|---|---|---|---|---|
| s5_permutation | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| anbncn_viability | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| iterated_nonlinear_map | 95 (25%) | 96 (25%) | 95 (25%) | 98 (26%) | 0 (0%) |
| flag_hold_recall | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) |
| mixed_probe | 94 (24%) | 96 (25%) | 96 (25%) | 98 (26%) | 0 (0%) |
