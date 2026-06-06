# redo-e99-1-3b — aggregated tables (real run outputs)

## LM screen — 32 evals, 15.0-min bf16, GPUs [0, 1, 2, 3, 4, 5, 6, 7], 651.5 GPU-min

Ranked by AvgLoss (CMA fitness, lower=better). BPB on canonical Pile held-out slice.

**arm**: SEARCHED = a 5-type fused CMA candidate (selection-eligible). CONTROL = fixed dense-GDN-2 (M0) or gdn2_nonlin_shell (S*) arm (capability/accuracy only, NEVER wallclock/tok-min ranked). ANCHOR = fixed comparability mixture (M1/C*), not selection-ranked.

| rank | name | arm | gdn/nonlin/shell | AvgLoss | Final | held-out BPB | tok/s | steps | params_B | NaN | RT |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 1 | cma_g3_e1 | SEARCHED | 102/0/0 | 6.3224 | 5.5607 | 2.0405 | 9656 | 2086 | 1.272 | n | — |
| 2 | cma_g2_e1 | SEARCHED | 102/0/0 | 6.3314 | 5.5591 | 2.0382 | 9654 | 2087 | 1.272 | n | — |
| 3 | cma_g1_e3 | SEARCHED | 102/0/0 | 6.3364 | 5.5364 | 2.0395 | 9395 | 2031 | 1.272 | n | — |
| 4 | cma_g3_e5 | SEARCHED | 102/0/0 | 6.3477 | 5.5587 | 2.0492 | 9297 | 2007 | 1.272 | n | — |
| 5 | cma_g3_e3 | SEARCHED | 102/0/0 | 6.3559 | 5.5394 | 2.0405 | 9378 | 2026 | 1.272 | n | — |
| 6 | cma_g2_e3 | SEARCHED | 102/0/0 | 6.3651 | 5.5454 | 2.0418 | 9374 | 2026 | 1.272 | n | — |
| 7 | cma_g2_e4 | SEARCHED | 102/0/0 | 6.3703 | 5.5807 | 2.0520 | 9112 | 1968 | 1.272 | n | — |
| 8 | M0_dense_gdn2 | CONTROL | 102/0/0 | 6.3786 | 5.6128 | 2.0450 | 8912 | 1928 | 1.272 | n | ok |
| 9 | cma_g3_e0 | SEARCHED | 102/0/0 | 6.3837 | 5.5846 | 2.0513 | 8848 | 1914 | 1.272 | n | — |
| 10 | cma_g3_e2 | SEARCHED | 102/0/0 | 6.3944 | 5.6404 | 2.0481 | 8890 | 1923 | 1.272 | n | — |
| 11 | cma_g3_e4 | SEARCHED | 102/0/0 | 6.3997 | 5.6024 | 2.0629 | 9113 | 1968 | 1.272 | n | — |
| 12 | cma_g2_e2 | SEARCHED | 102/0/0 | 6.4206 | 5.6393 | 2.0761 | 8896 | 1920 | 1.272 | n | — |
| 13 | cma_g2_e5 | SEARCHED | 100/0/0 | 6.4268 | 5.6887 | 2.0514 | 8292 | 1792 | 1.281 | n | — |
| 14 | S3_gdn_shell_f50 | CONTROL | 51/0/51 | 6.4459 | 5.6952 | 2.0642 | 8168 | 1765 | 1.272 | n | ok |
| 15 | cma_g0_e4 | SEARCHED | 100/0/0 | 6.4563 | 5.7089 | 2.0697 | 8167 | 1765 | 1.281 | n | — |
| 16 | cma_g1_e2 | SEARCHED | 101/0/0 | 6.4711 | 5.6632 | 2.0705 | 8034 | 1735 | 1.276 | n | — |
| 17 | S2_gdn_shell_f33 | CONTROL | 68/0/34 | 6.4746 | 5.6293 | 2.0792 | 7991 | 1725 | 1.272 | n | ok |
| 18 | cma_g1_e0 | SEARCHED | 98/2/0 | 6.4833 | 5.6065 | 2.0898 | 7697 | 1667 | 1.267 | n | — |
| 19 | S1_gdn_shell_f17 | CONTROL | 85/0/17 | 6.5035 | 5.7186 | 2.0923 | 8095 | 1747 | 1.272 | n | ok |
| 20 | cma_g2_e0 | SEARCHED | 98/0/0 | 6.5339 | 5.6393 | 2.0886 | 7692 | 1664 | 1.267 | n | — |
| 21 | M1_priorE99_5to1 | ANCHOR | 84/17/0 | 6.5872 | 5.7561 | 2.0962 | 8094 | 1751 | 1.278 | n | ok |
| 22 | C1_gdn_nonlin_f17 | ANCHOR | 85/17/0 | 6.6226 | 5.7388 | 2.1103 | 7550 | 1633 | 1.274 | n | ok |
| 23 | cma_g0_e5 | SEARCHED | 80/5/0 | 6.6594 | 5.7381 | 2.1229 | 7952 | 1718 | 1.271 | n | — |
| 24 | C2_gdn_nonlin_f33 | ANCHOR | 68/34/0 | 6.7291 | 5.7942 | 2.1319 | 7881 | 1701 | 1.269 | n | ok |
| 25 | cma_g1_e1 | SEARCHED | 64/15/0 | 6.8847 | 5.8788 | 2.1586 | 7949 | 1552 | 1.260 | n | — |
| 26 | C3_gdn_nonlin_f50 | ANCHOR | 51/51/0 | 6.8904 | 5.8969 | 2.1552 | 7422 | 1606 | 1.282 | n | ok |
| 27 | cma_g1_e4 | SEARCHED | 64/15/0 | 6.8921 | 5.9228 | 2.1755 | 7533 | 1496 | 1.260 | n | — |
| 28 | cma_g0_e0 | SEARCHED | 80/3/0 | 7.0103 | 5.8881 | 2.1847 | 7670 | 1661 | 1.271 | n | — |
| 29 | cma_g1_e5 | SEARCHED | 48/8/0 | 7.0132 | 5.8842 | 2.1517 | 7510 | 1626 | 1.267 | n | — |
| 30 | cma_g0_e2 | SEARCHED | 61/1/0 | 7.0600 | 5.9870 | 2.1734 | 7329 | 1436 | 1.270 | n | — |
| 31 | cma_g0_e3 | SEARCHED | 57/1/0 | 7.0965 | 5.9059 | 2.1583 | 7716 | 1506 | 1.260 | n | — |

## Three-way fairness control — matched-fraction triples

(a) native GDN-2 linear = M0_dense (same at every f). (b) GDN-2-shell nonlinear = S*. (c) legacy UnifiedCell nonlin = C*. (a)vs(b)=nonlinearity itself; (b)vs(c)=external plumbing.

| f (nonlin slot) | (a) dense AvgLoss/BPB | (b) shell AvgLoss/BPB | (c) nonlin-corner AvgLoss/BPB | (b)−(a) BPB | (b)−(c) BPB |
|---|---|---|---|---:|---:|
| 1/6 | 6.3786 / 2.0450 | 6.5035 / 2.0923 | 6.6226 / 2.1103 | 0.0473 | -0.0180 |
| 1/3 | 6.3786 / 2.0450 | 6.4746 / 2.0792 | 6.7291 / 2.1319 | 0.0342 | -0.0527 |
| 1/2 | 6.3786 / 2.0450 | 6.4459 / 2.0642 | 6.8904 / 2.1552 | 0.0192 | -0.0909 |

## Capability axis — 4000 steps, seeds [42], depth4/h48/n32, shell=tanh

| mixture | gdn/nonlin/shell | mean | min | mqar | s5 | anbncn | flag | iterated | mixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0_dense_gdn2 | 48/0/0 | 0.8282 | 0.6399 | 0.690 | 0.903 | 0.845 | 0.959 | 0.932 | 0.640 |
| M1_priorE99_5to1 | 40/8/0 | 0.8392 | 0.5791 | 0.681 | 0.995 | 0.905 | 0.955 | 0.920 | 0.579 |
| C1_gdn_nonlin_f17 | 40/8/0 | 0.8369 | 0.5787 | 0.687 | 0.995 | 0.905 | 0.936 | 0.919 | 0.579 |
| C2_gdn_nonlin_f33 | 32/16/0 | 0.8323 | 0.5826 | 0.663 | 0.956 | 0.901 | 0.975 | 0.916 | 0.583 |
| C3_gdn_nonlin_f50 | 24/24/0 | 0.8058 | 0.5756 | 0.686 | 0.827 | 0.846 | 0.994 | 0.907 | 0.576 |
| S1_gdn_shell_f17 | 40/0/8 | 0.8155 | 0.5932 | 0.609 | 0.970 | 0.855 | 0.931 | 0.935 | 0.593 |
| S2_gdn_shell_f33 | 32/0/16 | 0.8286 | 0.5803 | 0.606 | 0.958 | 0.894 | 0.998 | 0.935 | 0.580 |
| S3_gdn_shell_f50 | 24/0/24 | 0.8265 | 0.5464 | 0.546 | 0.939 | 0.924 | 0.988 | 0.930 | 0.630 |
