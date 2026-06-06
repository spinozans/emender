## [auto] E98 sixth-corner aggregated tables

Params (≈): spread-4 7376640, spread-5 7376640, spread-6 7376640, gated-delta 7394944, gdn 1612544, gdn-neg 1359632.

### 1. RECALL (MQAR) — does the gated-delta PRESET close the gap to GDN where leaky could not?

Random baseline ≈ 0.0156 (1/vocab). MQAR pairs scale with length. GDN is the workhorse bar; leaky was the 5-corner recall corner (the WORST).

| arm | @128 | @256 | @512 | @1024 | params |
|---|---|---|---|---|---|
| gdn | 0.951±0.00 | 0.828±0.01 | 0.531±0.02 | 0.260±0.01 | 1.61M |
| gated-delta | 0.171±0.00 | 0.070±0.00 | 0.029±0.00 | 0.018±0.00 | 7.39M |
| leaky | 0.111±0.00 | 0.069±0.00 | 0.028±0.00 | 0.018±0.00 | 7.36M |
| spread-6 | 0.261±0.03 | 0.131±0.01 | 0.055±0.00 | 0.026±0.00 | 7.36M |
| spread-5 | 0.271±0.07 | 0.141±0.02 | 0.068±0.01 | 0.040±0.01 | 7.36M |
| spread-4 | 0.440±0.21 | 0.229±0.12 | 0.097±0.05 | 0.046±0.02 | 7.36M |

**@128: gated-delta preset 0.171 vs GDN 0.951 (gap -0.780); leaky preset 0.111. gated-delta − leaky = +0.060.**

### 2. S5 (track) — does gated-delta ALSO track (neg-eig)? -> the dual recall+track backbone

Random baseline ≈ 0.0083. GDN+neg-eig is the length-robust track bar.

| arm | @128 | @256 | @512 | @1024 |
|---|---|---|---|---|
| gdn-neg | 0.998±0.00 | 0.988±0.02 | 0.949±0.06 | 0.824±0.10 |
| gated-delta | 0.179±0.01 | 0.094±0.00 | 0.051±0.00 | 0.030±0.00 |
| spread-6 | 1.000±0.00 | 1.000±0.00 | 0.985±0.02 | 0.762±0.14 |
| spread-5 | 1.000±0.00 | 1.000±0.00 | 0.978±0.02 | 0.721±0.17 |
| spread-4 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | 0.983±0.02 |

### 3. 4-vs-5-vs-6 HEAD-TO-HEAD @ T=128 — spread-6 vs spread-5 vs spread-4 (+Δ on recall = real gain; −Δ on an exotic probe = regression)

| probe (corner) | baseline | spread-4 | spread-5 | spread-6 | Δ(6−5) | Δ(6−4) | verdict |
|---|---|---|---|---|---|---|---|
| s5_permutation (track) | 0.008 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | +0.000 | +0.000 | tie |
| anbncn_viability (count) | 0.500 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | +0.000 | +0.000 | tie |
| iterated_nonlinear_map (nonlin) | 0.100 | 0.833±0.01 | 0.864±0.01 | 0.868±0.00 | +0.005 | +0.035 | tie |
| flag_hold_recall (latch) | 0.500 | 1.000±0.00 | 1.000±0.00 | 1.000±0.00 | +0.000 | +0.000 | tie |
| mqar_recall (recall) | 0.016 | 0.440±0.21 | 0.271±0.07 | 0.261±0.03 | -0.010 | -0.179 | tie |
| mixed_probe (ALL) | 0.189 | 0.867±0.01 | 0.876±0.01 | 0.878±0.00 | +0.001 | +0.011 | tie |

**@T=128: spread-6 beats spread-5 on 0 / regresses 0 / ties 6 of 6 probes.**

### 3. 4-vs-5-vs-6 HEAD-TO-HEAD @ T=1024 — spread-6 vs spread-5 vs spread-4 (+Δ on recall = real gain; −Δ on an exotic probe = regression)

| probe (corner) | baseline | spread-4 | spread-5 | spread-6 | Δ(6−5) | Δ(6−4) | verdict |
|---|---|---|---|---|---|---|---|
| s5_permutation (track) | 0.008 | 0.983±0.02 | 0.721±0.17 | 0.762±0.14 | +0.041 | -0.220 | 6 stronger |
| anbncn_viability (count) | 0.500 | 0.858±0.02 | 0.864±0.02 | 0.865±0.01 | +0.001 | +0.007 | tie |
| iterated_nonlinear_map (nonlin) | 0.100 | 0.824±0.01 | 0.852±0.00 | 0.861±0.00 | +0.009 | +0.037 | tie |
| flag_hold_recall (latch) | 0.500 | 1.000±0.00 | 1.000±0.00 | 0.992±0.01 | -0.008 | -0.008 | tie |
| mqar_recall (recall) | 0.016 | 0.046±0.02 | 0.040±0.01 | 0.026±0.00 | -0.014 | -0.020 | REGRESSION vs5 |
| mixed_probe (ALL) | 0.189 | 0.808±0.03 | 0.784±0.10 | 0.800±0.08 | +0.015 | -0.008 | 6 stronger |

**@T=1024: spread-6 beats spread-5 on 2 / regresses 1 / ties 3 of 6 probes.**

### 4. PER-HEAD corner occupancy — spread-6 (nearest-centroid over 6 corners, pooled layers+seeds)

| probe | n heads | track | count | latch | nonlin | leaky | gated-delta | center | covered |
|---|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 112 (29%) | 117 (30%) | 60 (16%) | 60 (16%) | 15 (4%) | 20 (5%) | 0 (0%) | 5/6 |
| anbncn_viability | 384 | 75 (20%) | 68 (18%) | 60 (16%) | 60 (16%) | 64 (17%) | 57 (15%) | 0 (0%) | 6/6 |
| iterated_nonlinear_map | 384 | 104 (27%) | 0 (0%) | 60 (16%) | 60 (16%) | 133 (35%) | 27 (7%) | 0 (0%) | 5/6 |
| flag_hold_recall | 384 | 72 (19%) | 72 (19%) | 60 (16%) | 60 (16%) | 60 (16%) | 60 (16%) | 0 (0%) | 6/6 |
| mqar_recall | 384 | 96 (25%) | 39 (10%) | 2 (1%) | 97 (25%) | 105 (27%) | 24 (6%) | 21 (5%) | 5/6 |
| mixed_probe | 384 | 100 (26%) | 59 (15%) | 60 (16%) | 60 (16%) | 63 (16%) | 42 (11%) | 0 (0%) | 6/6 |

### 4. PER-HEAD corner occupancy — spread-5 (nearest-centroid over 5 corners, pooled layers+seeds)

| probe | n heads | track | count | latch | nonlin | leaky | center | covered |
|---|---|---|---|---|---|---|---|---|
| s5_permutation | 384 | 84 (22%) | 131 (34%) | 72 (19%) | 72 (19%) | 24 (6%) | 0 (0%) | 5/5 |
| anbncn_viability | 384 | 84 (22%) | 93 (24%) | 72 (19%) | 72 (19%) | 63 (16%) | 0 (0%) | 5/5 |
| iterated_nonlinear_map | 384 | 84 (22%) | 0 (0%) | 72 (19%) | 72 (19%) | 156 (41%) | 0 (0%) | 4/5 |
| flag_hold_recall | 384 | 84 (22%) | 81 (21%) | 72 (19%) | 72 (19%) | 75 (20%) | 0 (0%) | 5/5 |
| mqar_recall | 384 | 81 (21%) | 55 (14%) | 1 (0%) | 123 (32%) | 95 (25%) | 23 (6%) | 4/5 |
| mixed_probe | 384 | 84 (22%) | 82 (21%) | 71 (18%) | 73 (19%) | 60 (16%) | 0 (0%) | 5/5 |

### 4b. spread-6 gated-delta-group knobs (heads classified to the gated-delta corner)

| probe | n | mean λ | mean β | mean γ | mean eig(λ−β) |
|---|---|---|---|---|---|
| mqar_recall | 24 | 1.00 | 1.23 | 0.09 | -0.24 |
| s5_permutation | 20 | 0.98 | 1.38 | 0.07 | -0.40 |
| mixed_probe | 42 | 1.00 | 1.20 | 0.07 | -0.20 |
