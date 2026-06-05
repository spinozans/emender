## [auto] E98 five-corner aggregated tables

Params (≈): spread-4 7376640, spread-5 7376640, leaky 7362048, gdn 1612544.

### 1. RECALL (MQAR) — does the leaky-linear workhorse solve it where the exotic corners fail?

Random baseline ≈ 0.0156 (1/vocab). MQAR pairs scale with length (T//16): 8 pairs @128 → 64 @1024.

| arm | @128 | @256 | @512 | @1024 |
|---|---|---|---|---|
| leaky | 0.111±0.01 | 0.068±0.00 | 0.027±0.00 | 0.019±0.00 |
| spread-5 | 0.293±0.10 | 0.157±0.04 | 0.075±0.02 | 0.044±0.01 |
| gdn | 0.951±0.00 | 0.829±0.01 | 0.531±0.02 | 0.259±0.01 |
| spread-4 | 0.277±0.12 | 0.138±0.04 | 0.065±0.01 | 0.036±0.00 |
| track | 0.165±0.01 | 0.083±0.01 | 0.035±0.00 | 0.022±0.00 |
| count | 0.136±0.01 | 0.054±0.00 | 0.044±0.00 | 0.038±0.00 |
| latch | 0.215±0.01 | 0.130±0.00 | 0.070±0.00 | 0.042±0.00 |
| nonlin | 0.143±0.01 | 0.073±0.00 | 0.028±0.00 | 0.018±0.00 |

### 2. 4-vs-5 HEAD-TO-HEAD @ T=128 — spread-5 − spread-4 (+ = 5-type wins; − on an exotic probe = regression)

| probe (corner) | baseline | spread-4 | spread-5 | Δ(5−4) | verdict |
|---|---|---|---|---|---|
| s5_permutation (track) | 0.008 | 1.000±0.00 | 1.000±0.00 | +0.000 | tie |
| anbncn_viability (count) | 0.500 | 1.000±0.00 | 1.000±0.00 | +0.000 | tie |
| iterated_nonlinear_map (nonlin) | 0.100 | 0.849±0.02 | 0.859±0.01 | +0.010 | tie |
| flag_hold_recall (latch) | 0.500 | 1.000±0.00 | 1.000±0.00 | +0.000 | tie |
| mqar_recall (leaky (recall)) | 0.016 | 0.277±0.12 | 0.293±0.10 | +0.016 | 5 stronger |
| mixed_probe (ALL-5) | 0.189 | 0.867±0.01 | 0.876±0.01 | +0.009 | tie |

**@T=128: spread-5 wins 1 / regresses 0 / ties 5 of 6 probes.**

### 2. 4-vs-5 HEAD-TO-HEAD @ T=1024 — spread-5 − spread-4 (+ = 5-type wins; − on an exotic probe = regression)

| probe (corner) | baseline | spread-4 | spread-5 | Δ(5−4) | verdict |
|---|---|---|---|---|---|
| s5_permutation (track) | 0.008 | 0.825±0.21 | 0.736±0.18 | -0.089 | REGRESSION |
| anbncn_viability (count) | 0.500 | 0.856±0.02 | 0.867±0.01 | +0.012 | 5 stronger |
| iterated_nonlinear_map (nonlin) | 0.100 | 0.841±0.02 | 0.850±0.00 | +0.009 | tie |
| flag_hold_recall (latch) | 0.500 | 1.000±0.00 | 1.000±0.00 | +0.000 | tie |
| mqar_recall (leaky (recall)) | 0.016 | 0.036±0.00 | 0.044±0.01 | +0.008 | tie |
| mixed_probe (ALL-5) | 0.189 | 0.808±0.03 | 0.784±0.10 | -0.024 | REGRESSION |

**@T=1024: spread-5 wins 1 / regresses 2 / ties 3 of 6 probes.**

### 3. PER-HEAD corner occupancy — spread-5 (nearest-centroid over 5 corners, pooled layers+seeds)

| probe | n heads | track | count | latch | nonlin | leaky | center | covered |
|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 84 (22%) | 131 (34%) | 72 (19%) | 72 (19%) | 25 (7%) | 0 (0%) | 5/5 |
| anbncn_viability | 384 | 84 (22%) | 87 (23%) | 72 (19%) | 72 (19%) | 69 (18%) | 0 (0%) | 5/5 |
| iterated_nonlinear_map | 384 | 84 (22%) | 0 (0%) | 72 (19%) | 72 (19%) | 156 (41%) | 0 (0%) | 4/5 |
| flag_hold_recall | 384 | 84 (22%) | 82 (21%) | 72 (19%) | 72 (19%) | 74 (19%) | 0 (0%) | 5/5 |
| mqar_recall | 384 | 85 (22%) | 53 (14%) | 1 (0%) | 119 (31%) | 99 (26%) | 27 (7%) | 4/5 |
| mixed_probe | 384 | 84 (22%) | 94 (24%) | 71 (18%) | 73 (19%) | 62 (16%) | 0 (0%) | 5/5 |

### 3. PER-HEAD corner occupancy — spread-4 (nearest-centroid over 5 corners, pooled layers+seeds)

| probe | n heads | track | count | latch | nonlin | leaky | center | covered |
|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 96 (25%) | 83 (22%) | 96 (25%) | 96 (25%) | 13 (3%) | 0 (0%) | 4/4 |
| anbncn_viability | 384 | 96 (25%) | 87 (23%) | 96 (25%) | 96 (25%) | 9 (2%) | 0 (0%) | 4/4 |
| iterated_nonlinear_map | 384 | 96 (25%) | 1 (0%) | 96 (25%) | 96 (25%) | 95 (25%) | 0 (0%) | 3/4 |
| flag_hold_recall | 384 | 96 (25%) | 94 (24%) | 96 (25%) | 96 (25%) | 2 (1%) | 0 (0%) | 4/4 |
| mqar_recall | 384 | 96 (25%) | 42 (11%) | 3 (1%) | 160 (42%) | 52 (14%) | 31 (8%) | 3/4 |
| mixed_probe | 384 | 96 (25%) | 68 (18%) | 94 (24%) | 97 (25%) | 28 (7%) | 1 (0%) | 4/4 |

### 3b. spread-5 leaky-group knobs on MQAR (heads classified to the leaky corner)

| stat | n | mean λ | mean β | mean γ | mean eig(λ−β) |
|---|---|---|---|---|---|
| mqar_recall | 99 | 0.55 | 0.22 | 0.11 | +0.33 |
| mixed_probe | 62 | 0.83 | 0.25 | 0.08 | +0.58 |
