<!-- AUTO-GENERATED (aggregate_e98_on_e97.py). -->

_E98 runs found: 33._

## 1. Corner re-confirmation on the split-gated cell

### Corner re-confirmation @ T=128 — does each split-gated preset win its corner?

| probe (corner) | E98 preset (split) | E88 preset | E98 spread | E88 spread |
|---|---|---|---|---|
| s5_permutation (track) | 1.000±0.00 | 0.834±0.01 | 1.000±0.00 | 1.000±0.00 |
| anbncn_viability (count) | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |
| iterated_nonlinear_map (nonlin) | 0.686±0.01 | 0.695±0.01 | 0.848±0.01 | 0.850±0.01 |
| flag_hold_recall (latch) | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |

### Corner re-confirmation @ T=1024 — does each split-gated preset win its corner?

| probe (corner) | E98 preset (split) | E88 preset | E98 spread | E88 spread |
|---|---|---|---|---|
| s5_permutation (track) | 0.156±0.00 | 0.111±0.00 | 0.846±0.11 | 0.541±0.02 |
| anbncn_viability (count) | 0.926±0.00 | 0.920±0.01 | 0.879±0.00 | 0.896±0.01 |
| iterated_nonlinear_map (nonlin) | 0.678±0.01 | 0.686±0.01 | 0.838±0.01 | 0.840±0.01 |
| flag_hold_recall (latch) | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 |

## 2. Learnability (winning form: spread-init + knob-LR 20x) on the split-gated cell

### Learnability @ T=128 — E98 spread (split) on every probe

| probe (corner) | E98 spread (split) | E88 spread | E98 generic | E98 fixedpop |
|---|---|---|---|---|
| s5_permutation (track) | 1.000±0.00 | 1.000±0.00 |   --   |   --   |
| anbncn_viability (count) | 1.000±0.00 | 1.000±0.00 |   --   |   --   |
| iterated_nonlinear_map (nonlin) | 0.848±0.01 | 0.850±0.01 |   --   |   --   |
| flag_hold_recall (latch) | 1.000±0.00 | 1.000±0.00 |   --   |   --   |
| mixed_probe (ALL) | 0.903±0.01 | 0.922±0.01 | 0.538±0.01 | 0.712±0.01 |

### Learnability @ T=1024 — E98 spread (split) on every probe

| probe (corner) | E98 spread (split) | E88 spread | E98 generic | E98 fixedpop |
|---|---|---|---|---|
| s5_permutation (track) | 0.846±0.11 | 0.541±0.02 |   --   |   --   |
| anbncn_viability (count) | 0.879±0.00 | 0.896±0.01 |   --   |   --   |
| iterated_nonlinear_map (nonlin) | 0.838±0.01 | 0.840±0.01 |   --   |   --   |
| flag_hold_recall (latch) | 1.000±0.00 | 1.000±0.00 |   --   |   --   |
| mixed_probe (ALL) | 0.854±0.02 | 0.719±0.03 | 0.475±0.05 | 0.551±0.04 |

### Final corner occupancy — E98 spread (split) (nearest-centroid head counts, pooled layers+seeds)

| probe | n heads | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 96 (25%) | 96 (25%) | 95 (25%) | 97 (25%) | 0 (0%) | 4/4 |
| anbncn_viability | 384 | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) | 4/4 |
| iterated_nonlinear_map | 384 | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) | 4/4 |
| flag_hold_recall | 384 | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) | 4/4 |
| mixed_probe | 384 | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) | 4/4 |

### Final corner occupancy — E88 spread (no split) (nearest-centroid head counts, pooled layers+seeds)

| probe | n heads | track | count | latch | nonlin | center | covered |
|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) | 4/4 |
| anbncn_viability | 384 | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) | 4/4 |
| iterated_nonlinear_map | 384 | 95 (25%) | 96 (25%) | 95 (25%) | 98 (26%) | 0 (0%) | 4/4 |
| flag_hold_recall | 384 | 96 (25%) | 96 (25%) | 96 (25%) | 96 (25%) | 0 (0%) | 4/4 |
| mixed_probe | 384 | 94 (24%) | 96 (25%) | 96 (25%) | 98 (26%) | 0 (0%) | 4/4 |

## 3. E97-split vs E88-coupled — does the split gate help?

### E97-split vs E88-coupled head-to-head @ T=128 (E98 − E88, +=split helps)

| probe (corner) | spread E98 | spread E88 | Δ spread | preset E98 | preset E88 | Δ preset |
|---|---|---|---|---|---|---|
| s5_permutation (track) | 1.000 | 1.000 | +0.000 | 1.000 | 0.834 | +0.166 |
| anbncn_viability (count) | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| iterated_nonlinear_map (nonlin) | 0.848 | 0.850 | -0.002 | 0.686 | 0.695 | -0.010 |
| flag_hold_recall (latch) | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| mixed_probe (ALL) | 0.903 | 0.922 | -0.019 | -- | -- | -- |

_Spread head-to-head @ T=128: split-gate WINS 0, LOSES 1, TIES 4 (|Δ|>0.01)._

### E97-split vs E88-coupled head-to-head @ T=1024 (E98 − E88, +=split helps)

| probe (corner) | spread E98 | spread E88 | Δ spread | preset E98 | preset E88 | Δ preset |
|---|---|---|---|---|---|---|
| s5_permutation (track) | 0.846 | 0.541 | +0.305 | 0.156 | 0.111 | +0.045 |
| anbncn_viability (count) | 0.879 | 0.896 | -0.017 | 0.926 | 0.920 | +0.006 |
| iterated_nonlinear_map (nonlin) | 0.838 | 0.840 | -0.002 | 0.678 | 0.686 | -0.008 |
| flag_hold_recall (latch) | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| mixed_probe (ALL) | 0.854 | 0.719 | +0.135 | -- | -- | -- |

_Spread head-to-head @ T=1024: split-gate WINS 2, LOSES 1, TIES 2 (|Δ|>0.01)._

### Parameter cost (depth 4, dim 384, 32 heads, N=V=32)

| arm | params | note |
|---|---|---|
| E98 spread (split) | 11,072,640 | unified + 2 split-gate projections (erase b*k, value w*v) |
| E88 spread (no split) | 7,926,912 | unified cell, coupled erase=write=k |
| E98 fixedpop (split) | 11,072,128 |  |
| E88 fixedpop (no split) | 7,926,400 |  |
