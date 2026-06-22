# SF-DiLoCo P2 matched-gain summary

All momentum configs are matched effective gain (`outer_lr ~= 1 - outer_beta`); `beta=0.9, outer_lr=1.0` is not run.

## Verdict

Post-fix matched-gain outer momentum is stable in this 2-GPU, K=250, 1100-step
real-data smoke: all three momentum configs completed with finite loss, finite
held-out BPB, fused E97 bf16 NO-eager guards on both ranks, and bounded
`|z-x|/|x|` geometry ratios. This is qualitatively different from the pre-fix
`beta=0.9, outer_lr=1.0` blow-up class.

Momentum did **not** beat plain periodic averaging at matched tokens on the
fixed held-out tensor. The plain-avg comparator ended at held-out BPB 2.0290.
Matched-gain momentum ended at 2.0983 (`beta=0.5, lr=0.5`), 2.1669
(`beta=0.8, lr=0.2`), and 2.2028 (`beta=0.9, lr=0.1`). Post-sync shock is
present but recovers within the observed window for periodic merges; the final
merge has no following local-step window, so its recovery is intentionally blank.

Artifacts:
- Raw logs and held-out curves: `/mnt/nvme1n1/erikg/sf_diloco_p2_matched_gain/`
- Machine summary: `/mnt/nvme1n1/erikg/sf_diloco_p2_matched_gain/summary.json`
- Held-out tensor: `experiments/lb_compare_20260613/heldout_p50k_2048.pt`

| run | outer_beta | outer_lr | final heldout BPB | final train loss | max shock | max recovery steps | geom land_frac range | gap_health range | fused guard ranks | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| pavg_beta00_lr10 | 0.0 | 1.0 | 2.029 | 5.087 | 0.17159999999999975 | 75 | n/a | n/a | 2 | stable |
| beta05_lr05 | 0.5 | 0.5 | 2.0983 | 5.3287 | 0.33340000000000014 | 100 | 0.5000-1.4569 | 4.593e-01-1.338e+00 | 2 | stable |
| beta08_lr02 | 0.8 | 0.2 | 2.1669 | 5.1818 | 0.2625000000000002 | 150 | 0.2000-1.4536 | 4.914e-01-1.380e+00 | 2 | stable |
| beta09_lr01 | 0.9 | 0.1 | 2.2028 | 5.4277 | 0.05959999999999965 | 175 | 0.1000-0.9429 | 4.675e-01-1.427e+00 | 2 | stable |

## Per-merge shock and geometry

### pavg_beta00_lr10

| merge | step | loss pre -> post | jump | recovery steps | land_frac | disp_mag | gap_health |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 250 | 6.2238 -> 6.1856 | -0.0382 | 25 | n/a | n/a | n/a |
| 2 | 500 | 5.5214 -> 5.5461 | +0.0247 | 50 | n/a | n/a | n/a |
| 3 | 750 | 5.3384 -> 5.2804 | -0.0580 | 25 | n/a | n/a | n/a |
| 4 | 1000 | 5.0988 -> 4.8836 | -0.2152 | 75 | n/a | n/a | n/a |
| 5 | 1100 | 4.9154 -> 5.0870 | +0.1716 |  | n/a | n/a | n/a |

### beta05_lr05

| merge | step | loss pre -> post | jump | recovery steps | land_frac | disp_mag | gap_health |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 250 | 6.2767 -> 6.3151 | +0.0384 | 75 | 0.5000 | 4.089e-01 | 4.593e-01 |
| 2 | 500 | 5.5497 -> 5.4803 | -0.0694 | 100 | 0.7308 | 2.773e-01 | 7.720e-01 |
| 3 | 750 | 5.5272 -> 5.5385 | +0.0113 | 25 | 0.8724 | 1.876e-01 | 1.021e+00 |
| 4 | 1000 | 5.2913 -> 5.1226 | -0.1687 | 25 | 0.9033 | 1.385e-01 | 1.256e+00 |
| 5 | 1100 | 4.9953 -> 5.3287 | +0.3334 |  | 1.4569 | 5.316e-02 | 1.338e+00 |

### beta08_lr02

| merge | step | loss pre -> post | jump | recovery steps | land_frac | disp_mag | gap_health |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 250 | 6.1192 -> 5.8348 | -0.2844 | 150 | 0.2000 | 4.395e-01 | 4.914e-01 |
| 2 | 500 | 5.7709 -> 5.6842 | -0.0867 | 75 | 0.3839 | 2.904e-01 | 7.847e-01 |
| 3 | 750 | 5.7040 -> 5.4598 | -0.2442 | 25 | 0.5427 | 2.176e-01 | 1.064e+00 |
| 4 | 1000 | 5.2195 -> 5.4820 | +0.2625 | 25 | 0.6844 | 1.641e-01 | 1.303e+00 |
| 5 | 1100 | 5.2346 -> 5.1818 | -0.0528 |  | 1.4536 | 6.490e-02 | 1.380e+00 |

### beta09_lr01

| merge | step | loss pre -> post | jump | recovery steps | land_frac | disp_mag | gap_health |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 250 | 6.2209 -> 6.1389 | -0.0820 | 175 | 0.1000 | 4.370e-01 | 4.675e-01 |
| 2 | 500 | 5.9488 -> 5.9089 | -0.0399 | 75 | 0.2052 | 2.906e-01 | 7.920e-01 |
| 3 | 750 | 5.7717 -> 5.5419 | -0.2298 | 25 | 0.2998 | 2.338e-01 | 1.074e+00 |
| 4 | 1000 | 5.4248 -> 5.2651 | -0.1597 | 75 | 0.4075 | 1.840e-01 | 1.333e+00 |
| 5 | 1100 | 5.3681 -> 5.4277 | +0.0596 |  | 0.9429 | 7.580e-02 | 1.427e+00 |
