<!-- AUTO-GENERATED (aggregate_specialization_study.py). -->

_Specialization-study runs found: 270 / 270._

## 1. Accuracy — does the approach specialize WITHOUT hurting accuracy?

### Accuracy @ T=128 (mean±std over seeds [42, 123, 456]; train T=128)

Reference columns reused from prior sweeps. Regularizer arms are on a GENERIC-init learned cell; dict/fixedpop are structural.

| probe (corner) | best preset | generic | spread-klr20 | LSTM | reg-pull-w0p1 | reg-pull-w0p3 | reg-pull-w1 | reg-anticenter-w0p1 | reg-anticenter-w0p3 | reg-anticenter-w1 | reg-coverage-w0p1 | reg-coverage-w0p3 | reg-coverage-w1 | reg-pull_cov-w0p1 | reg-pull_cov-w0p3 | reg-pull_cov-w1 | reg-anticenter_cov-w0p1 | reg-anticenter_cov-w0p3 | reg-anticenter_cov-w1 | unified-dict4 | unified-dict8 | unified-fixedpop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s5_permutation (track) | track 0.834 | 0.061 | 1.000 | 0.979 | 0.061±0.00 | 0.061±0.00 | 0.061±0.00 | 0.061±0.00 | 0.062±0.00 | 0.062±0.00 | 0.059±0.00 | 0.060±0.00 | 0.060±0.00 | 0.060±0.00 | 0.060±0.00 | 0.060±0.00 | 0.061±0.00 | 0.061±0.00 | 0.059±0.00 | 0.058±0.00 | 0.058±0.00 | 0.995±0.00 |
| anbncn_viability (count) | track 1.000 | 0.993 | 1.000 | 1.000 | 0.995±0.00 | 0.995±0.00 | 0.988±0.01 | 0.993±0.01 | 0.992±0.00 | 0.990±0.01 | 0.994±0.00 | 0.993±0.00 | 0.994±0.00 | 0.997±0.00 | 0.994±0.00 | 0.995±0.00 | 0.995±0.00 | 0.992±0.00 | 0.994±0.00 | 0.986±0.00 | 0.987±0.00 | 0.989±0.02 |
| iterated_nonlinear_map (nonlin) | track 0.903 | 0.680 | 0.850 | 0.985 | 0.683±0.01 | 0.687±0.01 | 0.687±0.01 | 0.685±0.00 | 0.688±0.01 | 0.693±0.01 | 0.682±0.01 | 0.675±0.00 | 0.681±0.01 | 0.675±0.00 | 0.677±0.01 | 0.677±0.00 | 0.677±0.01 | 0.683±0.01 | 0.687±0.01 | 0.645±0.01 | 0.652±0.01 | 0.833±0.00 |
| flag_hold_recall (latch) | track 1.000 | 1.000 | 1.000 | 0.512 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |
| mixed_probe (ALL) | track 0.840 | 0.538 | 0.922 | 0.674 | 0.537±0.01 | 0.536±0.01 | 0.539±0.01 | 0.538±0.01 | 0.540±0.01 | 0.540±0.01 | 0.536±0.01 | 0.538±0.02 | 0.537±0.01 | 0.538±0.01 | 0.537±0.01 | 0.535±0.01 | 0.537±0.02 | 0.537±0.01 | 0.537±0.01 | 0.522±0.02 | 0.525±0.01 | 0.784±0.01 |

### Accuracy @ T=1024 (mean±std over seeds [42, 123, 456]; train T=128)

Reference columns reused from prior sweeps. Regularizer arms are on a GENERIC-init learned cell; dict/fixedpop are structural.

| probe (corner) | best preset | generic | spread-klr20 | LSTM | reg-pull-w0p1 | reg-pull-w0p3 | reg-pull-w1 | reg-anticenter-w0p1 | reg-anticenter-w0p3 | reg-anticenter-w1 | reg-coverage-w0p1 | reg-coverage-w0p3 | reg-coverage-w1 | reg-pull_cov-w0p1 | reg-pull_cov-w0p3 | reg-pull_cov-w1 | reg-anticenter_cov-w0p1 | reg-anticenter_cov-w0p3 | reg-anticenter_cov-w1 | unified-dict4 | unified-dict8 | unified-fixedpop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s5_permutation (track) | track 0.111 | 0.015 | 0.541 | 0.563 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.015±0.00 | 0.142±0.00 |
| anbncn_viability (count) | count 0.920 | 0.836 | 0.896 | 0.992 | 0.839±0.01 | 0.837±0.01 | 0.865±0.04 | 0.839±0.01 | 0.850±0.01 | 0.835±0.02 | 0.816±0.04 | 0.838±0.02 | 0.840±0.01 | 0.840±0.01 | 0.831±0.01 | 0.838±0.02 | 0.837±0.01 | 0.840±0.01 | 0.838±0.01 | 0.834±0.02 | 0.835±0.01 | 0.851±0.02 |
| iterated_nonlinear_map (nonlin) | track 0.893 | 0.671 | 0.840 | 0.983 | 0.674±0.01 | 0.678±0.01 | 0.676±0.01 | 0.676±0.01 | 0.678±0.01 | 0.685±0.01 | 0.672±0.01 | 0.666±0.01 | 0.672±0.01 | 0.667±0.01 | 0.669±0.01 | 0.670±0.01 | 0.670±0.01 | 0.674±0.01 | 0.677±0.01 | 0.633±0.01 | 0.642±0.00 | 0.822±0.00 |
| flag_hold_recall (latch) | latch 1.000 | 0.531 | 1.000 | 0.562 | 0.555±0.03 | 0.516±0.02 | 0.477±0.05 | 0.680±0.23 | 0.680±0.23 | 0.641±0.26 | 1.000±0.00 | 1.000±0.00 | 0.901±0.14 | 0.966±0.05 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |
| mixed_probe (ALL) | track 0.569 | 0.478 | 0.719 | 0.634 | 0.475±0.05 | 0.477±0.05 | 0.480±0.05 | 0.478±0.05 | 0.492±0.07 | 0.517±0.05 | 0.475±0.05 | 0.478±0.05 | 0.479±0.05 | 0.480±0.05 | 0.477±0.05 | 0.476±0.05 | 0.475±0.05 | 0.477±0.05 | 0.495±0.06 | 0.466±0.05 | 0.477±0.06 | 0.547±0.04 |

## 2. Specialization — coverage of the four corners (MIXED probe)

### Specialization — do heads cover all four corners or collapse to center?

On the MIXED probe (all four capabilities trained at once). "%held" = heads whose nearest centroid is a REAL corner (not center); "covered" = distinct corners with >=5% of heads (4 = full coverage). Occupancy = head counts (track/count/latch/nonlin/center).

| approach | n | %held | %center | %eig<0 | covered | track | count | latch | nonlin | center |
|---|---|---|---|---|---|---|---|---|---|---|
| generic (ref) | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| spread-klr20 (ref) | 384 | 100% | 0% | 30% | 4/4 | 94 | 96 | 96 | 98 | 0 |
| reg-pull-w0p1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-pull-w0p3 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-pull-w1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-anticenter-w0p1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-anticenter-w0p3 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-anticenter-w1 | 384 | 67% | 33% | 0% | 1/4 | 0 | 0 | 0 | 257 | 127 |
| reg-coverage-w0p1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-coverage-w0p3 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-coverage-w1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-pull_cov-w0p1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-pull_cov-w0p3 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-pull_cov-w1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-anticenter_cov-w0p1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-anticenter_cov-w0p3 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| reg-anticenter_cov-w1 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| unified-dict4 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| unified-dict8 | 384 | 0% | 100% | 0% | 0/4 | 0 | 0 | 0 | 0 | 384 |
| unified-fixedpop | 384 | 100% | 0% | 25% | 4/4 | 96 | 96 | 96 | 96 | 0 |

## 3. Specialization-vs-accuracy trade-off

### Specialization-vs-accuracy trade-off (MIXED probe, accuracy @ T=1024)

mixed_held/cov from final knobs; mixed_acc is mixed_probe accuracy. A good approach has high %held AND high coverage AND high accuracy.

| approach | mixed %held | covered | mixed acc@128 | mixed acc@1024 |
|---|---|---|---|---|
| generic (ref) | 0% | 0/4 | 0.538 | 0.478 |
| spread-klr20 (ref) | 100% | 4/4 | 0.922 | 0.719 |
| reg-pull-w0p1 | 0% | 0/4 | 0.537 | 0.475 |
| reg-pull-w0p3 | 0% | 0/4 | 0.536 | 0.477 |
| reg-pull-w1 | 0% | 0/4 | 0.539 | 0.480 |
| reg-anticenter-w0p1 | 0% | 0/4 | 0.538 | 0.478 |
| reg-anticenter-w0p3 | 0% | 0/4 | 0.540 | 0.492 |
| reg-anticenter-w1 | 67% | 1/4 | 0.540 | 0.517 |
| reg-coverage-w0p1 | 0% | 0/4 | 0.536 | 0.475 |
| reg-coverage-w0p3 | 0% | 0/4 | 0.538 | 0.478 |
| reg-coverage-w1 | 0% | 0/4 | 0.537 | 0.479 |
| reg-pull_cov-w0p1 | 0% | 0/4 | 0.538 | 0.480 |
| reg-pull_cov-w0p3 | 0% | 0/4 | 0.537 | 0.477 |
| reg-pull_cov-w1 | 0% | 0/4 | 0.535 | 0.476 |
| reg-anticenter_cov-w0p1 | 0% | 0/4 | 0.537 | 0.475 |
| reg-anticenter_cov-w0p3 | 0% | 0/4 | 0.537 | 0.477 |
| reg-anticenter_cov-w1 | 0% | 0/4 | 0.537 | 0.495 |
| unified-dict4 | 0% | 0/4 | 0.522 | 0.466 |
| unified-dict8 | 0% | 0/4 | 0.525 | 0.477 |
| unified-fixedpop | 100% | 4/4 | 0.784 | 0.547 |

## 4. Emergent head-type mixture per task

### Emergent head-type mixture per task (corner occupancy)

For each selected approach: how many heads of each type emerge per probe. Occupancy = (track / count / latch / nonlin / center).

**reg-pull_cov-w0p3**

| probe | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|
| s5_permutation | 0 | 0 | 0 | 0 | 384 | 0/4 |
| anbncn_viability | 0 | 0 | 0 | 0 | 384 | 0/4 |
| iterated_nonlinear_map | 0 | 0 | 0 | 0 | 384 | 0/4 |
| flag_hold_recall | 0 | 0 | 0 | 0 | 384 | 0/4 |
| mixed_probe | 0 | 0 | 0 | 0 | 384 | 0/4 |

**reg-anticenter_cov-w0p3**

| probe | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|
| s5_permutation | 0 | 0 | 0 | 0 | 384 | 0/4 |
| anbncn_viability | 0 | 0 | 0 | 0 | 384 | 0/4 |
| iterated_nonlinear_map | 0 | 0 | 0 | 0 | 384 | 0/4 |
| flag_hold_recall | 0 | 51 | 0 | 0 | 333 | 1/4 |
| mixed_probe | 0 | 0 | 0 | 0 | 384 | 0/4 |

**unified-dict4**

| probe | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|
| s5_permutation | 0 | 0 | 0 | 0 | 384 | 0/4 |
| anbncn_viability | 0 | 0 | 0 | 0 | 384 | 0/4 |
| iterated_nonlinear_map | 0 | 0 | 0 | 0 | 384 | 0/4 |
| flag_hold_recall | 0 | 0 | 0 | 0 | 384 | 0/4 |
| mixed_probe | 0 | 0 | 0 | 0 | 384 | 0/4 |

**unified-dict8**

| probe | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|
| s5_permutation | 0 | 0 | 0 | 0 | 384 | 0/4 |
| anbncn_viability | 0 | 0 | 0 | 0 | 384 | 0/4 |
| iterated_nonlinear_map | 0 | 0 | 0 | 0 | 384 | 0/4 |
| flag_hold_recall | 0 | 0 | 0 | 0 | 384 | 0/4 |
| mixed_probe | 0 | 0 | 0 | 0 | 384 | 0/4 |

**unified-fixedpop**

| probe | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|
| s5_permutation | 96 | 96 | 96 | 96 | 0 | 4/4 |
| anbncn_viability | 96 | 96 | 96 | 96 | 0 | 4/4 |
| iterated_nonlinear_map | 96 | 96 | 96 | 96 | 0 | 4/4 |
| flag_hold_recall | 96 | 96 | 96 | 96 | 0 | 4/4 |
| mixed_probe | 96 | 96 | 96 | 96 | 0 | 4/4 |

**unified-learned-spread-klr20**

| probe | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|
| s5_permutation | 96 | 96 | 96 | 96 | 0 | 4/4 |
| anbncn_viability | 96 | 96 | 96 | 96 | 0 | 4/4 |
| iterated_nonlinear_map | 95 | 96 | 95 | 98 | 0 | 4/4 |
| flag_hold_recall | 96 | 96 | 96 | 96 | 0 | 4/4 |
| mixed_probe | 94 | 96 | 96 | 98 | 0 | 4/4 |

## 5. Parameter / compute overhead

### Parameter / compute overhead

Params and wall-clock from a representative mixed_probe seed-42 run (depth 4, 32 heads, dim 384). Regularizer arms add NO parameters (train-time penalty only); dict adds K shared prototypes + per-head soft weights; fixedpop REMOVES the learnable knobs.

| approach | params | Δparams vs generic | train wall (s) |
|---|---|---|---|
| generic (ref) | 7,926,912 | +0 | 104 |
| reg-pull_cov-w1 | 7,926,912 | +0 | 122 |
| unified-dict4 | 7,926,976 | +64 | 106 |
| unified-dict8 | 7,927,552 | +640 | 108 |
| unified-fixedpop | 7,926,400 | +-512 | 113 |
