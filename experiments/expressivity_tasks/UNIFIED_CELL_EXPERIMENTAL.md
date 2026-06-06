<!-- AUTO-GENERATED experimental sections (aggregate_unified.py). -->
<!-- Re-run: python experiments/expressivity_tasks/aggregate_unified.py -->

_Runs found: 96 / 96._

## Expressivity table (Run A)

### Expressivity @ T=128  (mean±std over seeds [42, 123, 456]; train T=128)

| arm | s5_permutation<br>(track) | anbncn_viability<br>(count) | iterated_nonlinear_map<br>(nonlinear) | flag_hold_recall<br>(latch) |
|---|---|---|---|---|
| _random baseline_ | 0.008 | 0.500 | 0.100 | 0.500 |
| unified-track | 0.834±0.01 | 1.000±0.00 | 0.903±0.00 | 1.000±0.00 |
| unified-count | 0.049±0.00 | 1.000±0.00 | 0.436±0.00 | 1.000±0.00 |
| unified-latch | 0.034±0.00 | 0.948±0.01 | 0.495±0.01 | 1.000±0.00 |
| unified-nonlin | 0.066±0.00 | 0.994±0.01 | 0.695±0.01 | 0.496±0.00 |
| unified-learned-free | 0.061±0.00 | 0.993±0.00 | 0.680±0.01 | 1.000±0.00 |
| unified-learned-clamp | 0.059±0.00 | 0.991±0.01 | 0.672±0.00 | 1.000±0.00 |
| unified-e88base | 0.064±0.00 | 0.978±0.01 | 0.708±0.01 | 0.509±0.04 |
| lstm | 0.979±0.01 | 1.000±0.00 | 0.985±0.00 | 0.512±0.07 |

### Expressivity @ T=512  (mean±std over seeds [42, 123, 456]; train T=128)

| arm | s5_permutation<br>(track) | anbncn_viability<br>(count) | iterated_nonlinear_map<br>(nonlinear) | flag_hold_recall<br>(latch) |
|---|---|---|---|---|
| _random baseline_ | 0.008 | 0.500 | 0.100 | 0.500 |
| unified-track | 0.215±0.00 | 0.798±0.02 | 0.894±0.00 | 0.500±0.03 |
| unified-count | 0.019±0.00 | 0.926±0.00 | 0.423±0.00 | 0.909±0.06 |
| unified-latch | 0.015±0.00 | 0.855±0.01 | 0.493±0.01 | 1.000±0.00 |
| unified-nonlin | 0.023±0.00 | 0.845±0.01 | 0.689±0.01 | 0.492±0.02 |
| unified-learned-free | 0.021±0.00 | 0.858±0.01 | 0.675±0.01 | 0.512±0.02 |
| unified-learned-clamp | 0.021±0.00 | 0.834±0.04 | 0.667±0.01 | 0.513±0.02 |
| unified-e88base | 0.023±0.00 | 0.846±0.00 | 0.703±0.01 | 0.538±0.04 |
| lstm | 0.767±0.08 | 0.997±0.00 | 0.984±0.00 | 0.579±0.06 |

### Expressivity @ T=1024  (mean±std over seeds [42, 123, 456]; train T=128)

| arm | s5_permutation<br>(track) | anbncn_viability<br>(count) | iterated_nonlinear_map<br>(nonlinear) | flag_hold_recall<br>(latch) |
|---|---|---|---|---|
| _random baseline_ | 0.008 | 0.500 | 0.100 | 0.500 |
| unified-track | 0.111±0.00 | 0.754±0.01 | 0.893±0.00 | 0.432±0.02 |
| unified-count | 0.013±0.00 | 0.920±0.01 | 0.421±0.00 | 0.797±0.11 |
| unified-latch | 0.012±0.00 | 0.839±0.02 | 0.490±0.01 | 1.000±0.00 |
| unified-nonlin | 0.016±0.00 | 0.829±0.02 | 0.686±0.01 | 0.568±0.02 |
| unified-learned-free | 0.015±0.00 | 0.836±0.01 | 0.671±0.01 | 0.531±0.03 |
| unified-learned-clamp | 0.015±0.00 | 0.803±0.05 | 0.664±0.00 | 0.477±0.04 |
| unified-e88base | 0.015±0.00 | 0.829±0.01 | 0.700±0.01 | 0.508±0.05 |
| lstm | 0.563±0.09 | 0.992±0.01 | 0.983±0.00 | 0.562±0.14 |

### Does LEARNED match the best preset on each probe? @ T=128

| probe (corner) | best preset (acc) | LEARNED-free (acc) | E88-base (acc) | LSTM (acc) | LEARNED matches best? |
|---|---|---|---|---|---|
| s5_permutation (track) | track (0.834) | 0.061 | 0.064 | 0.979 | no |
| anbncn_viability (count) | track (1.000) | 0.993 | 0.978 | 1.000 | YES |
| iterated_nonlinear_map (nonlinear) | track (0.903) | 0.680 | 0.708 | 0.985 | no |
| flag_hold_recall (latch) | track (1.000) | 1.000 | 0.509 | 0.512 | YES |

### Does LEARNED match the best preset on each probe? @ T=512

| probe (corner) | best preset (acc) | LEARNED-free (acc) | E88-base (acc) | LSTM (acc) | LEARNED matches best? |
|---|---|---|---|---|---|
| s5_permutation (track) | track (0.215) | 0.021 | 0.023 | 0.767 | no |
| anbncn_viability (count) | count (0.926) | 0.858 | 0.846 | 0.997 | no |
| iterated_nonlinear_map (nonlinear) | track (0.894) | 0.675 | 0.703 | 0.984 | no |
| flag_hold_recall (latch) | latch (1.000) | 0.512 | 0.538 | 0.579 | no |

## Un-cribbing demo (controlled headline): lambda FREE vs CLAMPED to (0,1)

Same LEARNED cell, identical recipe; the ONLY difference is whether the gain lambda may reach/exceed 1 (`lam_max=1.5`, FREE) or is clamped to (0,1) (`lam_max=1.0`, the cribbed E88 regime). Counting and latching require eigenvalue magnitude >=1; clamping should kill them.

| probe (corner) | T | LEARNED-free | LEARNED-clamp | delta (free-clamp) |
|---|---|---|---|---|
| anbncn_viability (count) | 128 | 0.993±0.00 | 0.991±0.01 | +0.002 |
| anbncn_viability (count) | 512 | 0.858±0.01 | 0.834±0.04 | +0.023 |
| anbncn_viability (count) | 1024 | 0.836±0.01 | 0.803±0.05 | +0.033 |
| flag_hold_recall (latch) | 128 | 1.000±0.00 | 1.000±0.00 | +0.000 |
| flag_hold_recall (latch) | 512 | 0.512±0.02 | 0.513±0.02 | -0.001 |
| flag_hold_recall (latch) | 1024 | 0.531±0.03 | 0.477±0.04 | +0.055 |
| s5_permutation (track) | 128 | 0.061±0.00 | 0.059±0.00 | +0.002 |
| s5_permutation (track) | 512 | 0.021±0.00 | 0.021±0.00 | +0.000 |
| s5_permutation (track) | 1024 | 0.015±0.00 | 0.015±0.00 | +0.000 |
| iterated_nonlinear_map (nonlinear) | 128 | 0.680±0.01 | 0.672±0.00 | +0.008 |
| iterated_nonlinear_map (nonlinear) | 512 | 0.675±0.01 | 0.667±0.01 | +0.007 |
| iterated_nonlinear_map (nonlinear) | 1024 | 0.671±0.01 | 0.664±0.00 | +0.007 |

## Emergent specialization (Run C): per-head learned knobs

Per-head (lambda, beta, gamma, eig_along=lambda-beta) after training the LEARNED-free cell, pooled over all heads/layers/seeds, reported per probe. Corner signatures: track=eig_along<0 (reflection); count=lambda~1 & beta~0; latch=lambda>1 & gamma high (tanh); nonlinear=gamma high & lambda<1.

### s5_permutation (track) — 384 heads pooled
- lambda : mean=0.966 min=0.944 max=1.009
- beta   : mean=0.530 min=0.486 max=0.612
- gamma  : mean=0.501 min=0.489 max=0.516
- eig_along (lambda-beta): mean=0.436 min=0.348 max=0.496  (0% heads reflecting, eig<0)
- heads in count-corner (lambda>0.9 & beta<0.3): 0 (0%); latch-corner (lambda>1 & gamma>0.6): 0 (0%)

### anbncn_viability (count) — 384 heads pooled
- lambda : mean=0.949 min=0.933 max=0.961
- beta   : mean=0.503 min=0.481 max=0.532
- gamma  : mean=0.498 min=0.487 max=0.504
- eig_along (lambda-beta): mean=0.447 min=0.409 max=0.467  (0% heads reflecting, eig<0)
- heads in count-corner (lambda>0.9 & beta<0.3): 0 (0%); latch-corner (lambda>1 & gamma>0.6): 0 (0%)

### iterated_nonlinear_map (nonlinear) — 384 heads pooled
- lambda : mean=0.910 min=0.825 max=0.944
- beta   : mean=0.539 min=0.490 max=0.684
- gamma  : mean=0.511 min=0.494 max=0.537
- eig_along (lambda-beta): mean=0.371 min=0.141 max=0.439  (0% heads reflecting, eig<0)
- heads in count-corner (lambda>0.9 & beta<0.3): 0 (0%); latch-corner (lambda>1 & gamma>0.6): 0 (0%)

### flag_hold_recall (latch) — 384 heads pooled
- lambda : mean=0.958 min=0.944 max=0.993
- beta   : mean=0.505 min=0.500 max=0.528
- gamma  : mean=0.498 min=0.484 max=0.502
- eig_along (lambda-beta): mean=0.452 min=0.436 max=0.474  (0% heads reflecting, eig<0)
- heads in count-corner (lambda>0.9 & beta<0.3): 0 (0%); latch-corner (lambda>1 & gamma>0.6): 0 (0%)
