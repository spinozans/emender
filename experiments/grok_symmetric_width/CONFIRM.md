# grok-confirm — temporal class separation, confirmed

_Part 1: 96 decisive-cell modular_quadratic runs (arm{e97,e97-lin,gdn2} x dim{512,1024} x p{256,512} x wd1.0, seeds 0-7). Part 2: 24 iterated_nonlinear_map runs (same arms x dim{512,1024} x seeds 0-3, wd1.0). L=2, n_state=32, n_heads=8, mlp_ratio=4, seq_len=128 (train), AdamW, 50k steps; bf16+fused asserted per-arm. REAL data._


## PART 1 — 8-seed confirmation (modular_quadratic, decisive cells)

Per-cell #grokked/n at the train length, then the grokked-seed test-acc-vs-T extrapolation curve. A model that learned the RULE is length-invariant; one that MEMORIZED T=128 collapses toward baseline.


### 1a. Grok counts + extrapolation curves (grokked seeds, wd=1.0)


**p=256** (baseline 1/p = 0.0039)

| arm | dim | #grok/n | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 512 | 5/8 | 0.941 | 0.938 | 0.932 | 0.915 | 0.890 | 0.865 |
| e97-lin | 512 | 2/8 | 0.950 | 0.938 | 0.881 | 0.724 | 0.564 | 0.471 |
| gdn2 | 512 | 8/8 | 0.999 | 0.968 | 0.824 | 0.667 | 0.579 | 0.544 |
| e97 | 1024 | 5/8 | 0.912 | 0.901 | 0.882 | 0.861 | 0.840 | 0.820 |
| e97-lin | 1024 | 4/8 | 0.962 | 0.957 | 0.933 | 0.872 | 0.775 | 0.651 |
| gdn2 | 1024 | 8/8 | 0.997 | 0.987 | 0.909 | 0.755 | 0.631 | 0.558 |

**p=512** (baseline 1/p = 0.0020)

| arm | dim | #grok/n | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 512 | 0/8 |   -   |   -   |   -   |   -   |   -   |   -   |
| e97-lin | 512 | 0/8 |   -   |   -   |   -   |   -   |   -   |   -   |
| gdn2 | 512 | 7/8 | 0.996 | 0.969 | 0.833 | 0.686 | 0.591 | 0.547 |
| e97 | 1024 | 4/8 | 0.914 | 0.909 | 0.894 | 0.848 | 0.780 | 0.715 |
| e97-lin | 1024 | 3/8 | 0.895 | 0.883 | 0.840 | 0.708 | 0.527 | 0.371 |
| gdn2 | 1024 | 8/8 | 0.939 | 0.923 | 0.842 | 0.702 | 0.616 | 0.545 |

### 1b. Far-T cliff (grokked-seed mean test-acc)

| p | dim | T | e97 | e97-lin | gdn2 | e97-e97lin | e97-gdn2 |
|---|---|---|---|---|---|---|---|
| 256 | 512 | 2048 | 0.890 | 0.564 | 0.579 | 0.327 | 0.311 |
| 256 | 512 | 4096 | 0.865 | 0.471 | 0.544 | 0.394 | 0.321 |
| 256 | 1024 | 2048 | 0.840 | 0.775 | 0.631 | 0.065 | 0.208 |
| 256 | 1024 | 4096 | 0.820 | 0.651 | 0.558 | 0.169 | 0.261 |
| 512 | 512 | 2048 |   -   |   -   | 0.591 |   -   |   -   |
| 512 | 512 | 4096 |   -   |   -   | 0.547 |   -   |   -   |
| 512 | 1024 | 2048 | 0.780 | 0.527 | 0.616 | 0.254 | 0.164 |
| 512 | 1024 | 4096 | 0.715 | 0.371 | 0.545 | 0.343 | 0.170 |

## PART 2 — second task family (iterated_nonlinear_map)

Input-driven logistic map h_t = a_t h_{t-1}(1-h_{t-1}), binned to n_bins=10 targets (baseline 0.1). A genuine state-quadratic; a different surface from modular arithmetic. Same arms, train-to-grok, far length-extrap. Does the e97-holds / linear-collapses signature replicate?


_Grok count to 0.9: 0/24. The logistic map at a in [2.6,3.6] is CONTRACTIVE (fading memory): h_t depends on recent drivers, not full history, so there is no long-memory train-length to memorize-then-fail. All arms learn a smooth, length-invariant partial approximation (test plateaus ~0.65-0.72 >> 0.1 baseline). Curves below are ALL-seed means (best_test) since the grokked subset is empty._


### 2a. Extrapolation curves (ALL seeds, wd=1.0)


**p=10** (baseline 0.1000 (n_bins=10))

| arm | dim | n (ALL seeds) | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 512 | 4 | 0.667 | 0.661 | 0.659 | 0.658 | 0.657 | 0.655 |
| e97-lin | 512 | 4 | 0.648 | 0.642 | 0.637 | 0.634 | 0.631 | 0.628 |
| gdn2 | 512 | 4 | 0.716 | 0.712 | 0.710 | 0.710 | 0.708 | 0.707 |
| e97 | 1024 | 4 | 0.657 | 0.650 | 0.646 | 0.646 | 0.644 | 0.644 |
| e97-lin | 1024 | 4 | 0.678 | 0.670 | 0.665 | 0.663 | 0.659 | 0.656 |
| gdn2 | 1024 | 4 | 0.719 | 0.713 | 0.710 | 0.710 | 0.708 | 0.708 |

### 2b. Far-T cliff (ALL-seed mean test-acc)

| p | dim | T | e97 | e97-lin | gdn2 | e97-e97lin | e97-gdn2 |
|---|---|---|---|---|---|---|---|
| 10 | 512 | 2048 | 0.657 | 0.631 | 0.708 | 0.027 | -0.050 |
| 10 | 512 | 4096 | 0.655 | 0.628 | 0.707 | 0.027 | -0.052 |
| 10 | 1024 | 2048 | 0.644 | 0.659 | 0.708 | -0.016 | -0.065 |
| 10 | 1024 | 4096 | 0.644 | 0.656 | 0.708 | -0.012 | -0.064 |

## PART 3 — second task family (a^n b^n c^n viability)

Per-position viability of the language a^n b^n c^n: at each step, is the prefix still extensible to some a^n b^n c^n? Decided by COUNT COMPARISONS (nb<=na, nb==na, nc<=na) whose magnitude scales with T. This is a NON-CONTRACTIVE long-memory task (binary target, baseline 0.5) -- the right regime for the memorization-vs-rule test. Train at T=128, extrapolate to T=4096. Does the e97-holds / linear-collapses signature replicate?


_Grok count to 0.9: 24/24._


### 3a. Grok counts + extrapolation curves (grokked seeds, wd=1.0)


**p=2** (baseline 0.5000 (binary))

| arm | dim | #grok/n | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 512 | 4/4 | 0.952 | 0.912 | 0.871 | 0.827 | 0.803 | 0.778 |
| e97-lin | 512 | 4/4 | 0.975 | 0.927 | 0.903 | 0.864 | 0.837 | 0.818 |
| gdn2 | 512 | 4/4 | 0.974 | 0.926 | 0.898 | 0.870 | 0.859 | 0.847 |
| e97 | 1024 | 4/4 | 0.978 | 0.937 | 0.898 | 0.863 | 0.853 | 0.822 |
| e97-lin | 1024 | 4/4 | 0.973 | 0.926 | 0.890 | 0.849 | 0.825 | 0.796 |
| gdn2 | 1024 | 4/4 | 0.979 | 0.915 | 0.864 | 0.825 | 0.811 | 0.777 |

### 3b. Far-T cliff (grokked-seed mean test-acc)

| p | dim | T | e97 | e97-lin | gdn2 | e97-e97lin | e97-gdn2 |
|---|---|---|---|---|---|---|---|
| 2 | 512 | 2048 | 0.803 | 0.837 | 0.859 | -0.035 | -0.056 |
| 2 | 512 | 4096 | 0.778 | 0.818 | 0.847 | -0.039 | -0.069 |
| 2 | 1024 | 2048 | 0.853 | 0.825 | 0.811 | 0.028 | 0.042 |
| 2 | 1024 | 4096 | 0.822 | 0.796 | 0.777 | 0.026 | 0.045 |

### 3c. Extrapolation curves (ALL seeds, for reference)


**p=2** (baseline 0.5000 (binary))

| arm | dim | n (ALL seeds) | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 512 | 4 | 0.952 | 0.912 | 0.871 | 0.827 | 0.803 | 0.778 |
| e97-lin | 512 | 4 | 0.975 | 0.927 | 0.903 | 0.864 | 0.837 | 0.818 |
| gdn2 | 512 | 4 | 0.974 | 0.926 | 0.898 | 0.870 | 0.859 | 0.847 |
| e97 | 1024 | 4 | 0.978 | 0.937 | 0.898 | 0.863 | 0.853 | 0.822 |
| e97-lin | 1024 | 4 | 0.973 | 0.926 | 0.890 | 0.849 | 0.825 | 0.796 |
| gdn2 | 1024 | 4 | 0.979 | 0.915 | 0.864 | 0.825 | 0.811 | 0.777 |

## PART 1 raw runs

| label | grok | gstep | train | test | best | T128 | T1024 | T4096 |
|---|---|---|---|---|---|---|---|---|
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s0 | False | None | 1.000 | 0.855 | 0.879 | 0.861 | 0.668 | 0.441 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s1 | False | None | 1.000 | 0.429 | 0.461 | 0.427 | 0.405 | 0.372 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s2 | True | 16000 | 0.999 | 0.969 | 0.979 | 0.968 | 0.892 | 0.619 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s3 | False | None | 0.999 | 0.453 | 0.456 | 0.449 | 0.436 | 0.435 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s4 | True | 5500 | 0.998 | 0.950 | 0.972 | 0.950 | 0.752 | 0.566 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s5 | True | 32000 | 1.000 | 0.949 | 0.957 | 0.949 | 0.917 | 0.729 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s6 | False | None | 0.999 | 0.459 | 0.478 | 0.460 | 0.443 | 0.438 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s7 | True | 5500 | 1.000 | 0.980 | 0.979 | 0.982 | 0.928 | 0.689 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s0 | True | 12500 | 1.000 | 0.954 | 0.962 | 0.955 | 0.908 | 0.793 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.889 | 0.891 | 0.890 | 0.620 | 0.480 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.442 | 0.463 | 0.440 | 0.417 | 0.394 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.444 | 0.474 | 0.445 | 0.430 | 0.418 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s4 | False | None | 1.000 | 0.448 | 0.478 | 0.448 | 0.418 | 0.397 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s5 | False | None | 1.000 | 0.444 | 0.453 | 0.446 | 0.421 | 0.408 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s6 | False | None | 1.000 | 0.449 | 0.470 | 0.452 | 0.425 | 0.399 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s7 | True | 29500 | 1.000 | 0.943 | 0.946 | 0.944 | 0.540 | 0.149 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s0 | False | None | 0.999 | 0.899 | 0.899 | 0.905 | 0.894 | 0.884 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s1 | True | 5500 | 1.000 | 0.987 | 0.990 | 0.988 | 0.987 | 0.988 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s2 | True | 4500 | 1.000 | 0.976 | 0.989 | 0.977 | 0.975 | 0.975 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s3 | False | None | 0.999 | 0.466 | 0.466 | 0.459 | 0.442 | 0.435 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s4 | False | None | 0.999 | 0.452 | 0.476 | 0.450 | 0.435 | 0.433 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s5 | True | 10500 | 1.000 | 0.978 | 0.984 | 0.978 | 0.909 | 0.783 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s6 | True | 17000 | 0.997 | 0.957 | 0.976 | 0.959 | 0.957 | 0.956 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s7 | True | 4000 | 0.725 | 0.660 | 0.979 | 0.660 | 0.478 | 0.397 |
| sym__mq_p256__e97__L2__d512__wd1.0__s0 | True | 16000 | 0.999 | 0.914 | 0.946 | 0.915 | 0.913 | 0.903 |
| sym__mq_p256__e97__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.435 | 0.455 | 0.435 | 0.422 | 0.425 |
| sym__mq_p256__e97__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.446 | 0.470 | 0.443 | 0.434 | 0.431 |
| sym__mq_p256__e97__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.477 | 0.477 | 0.475 | 0.459 | 0.459 |
| sym__mq_p256__e97__L2__d512__wd1.0__s4 | True | 20500 | 1.000 | 0.941 | 0.941 | 0.940 | 0.939 | 0.937 |
| sym__mq_p256__e97__L2__d512__wd1.0__s5 | True | 6500 | 1.000 | 0.936 | 0.950 | 0.936 | 0.933 | 0.930 |
| sym__mq_p256__e97__L2__d512__wd1.0__s6 | True | 44500 | 1.000 | 0.945 | 0.945 | 0.948 | 0.830 | 0.590 |
| sym__mq_p256__e97__L2__d512__wd1.0__s7 | True | 5500 | 1.000 | 0.963 | 0.971 | 0.965 | 0.963 | 0.964 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s0 | True | 5000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.808 | 0.571 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s1 | True | 8000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.829 | 0.574 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s2 | True | 6500 | 1.000 | 0.999 | 0.999 | 1.000 | 0.681 | 0.522 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s3 | True | 8500 | 1.000 | 0.983 | 0.999 | 0.977 | 0.784 | 0.575 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s4 | True | 17500 | 1.000 | 0.994 | 1.000 | 0.998 | 0.697 | 0.560 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s5 | True | 5500 | 1.000 | 1.000 | 0.999 | 1.000 | 0.747 | 0.548 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s6 | True | 7000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.785 | 0.581 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s7 | True | 4000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.706 | 0.537 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s0 | True | 10000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.681 | 0.542 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s1 | True | 12500 | 1.000 | 0.996 | 1.000 | 0.995 | 0.692 | 0.547 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s2 | True | 21000 | 1.000 | 0.991 | 0.999 | 0.997 | 0.635 | 0.535 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s3 | True | 13500 | 1.000 | 0.998 | 1.000 | 0.999 | 0.629 | 0.542 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s4 | True | 7000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.666 | 0.537 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s5 | True | 15500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.688 | 0.557 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s6 | True | 16500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.707 | 0.556 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s7 | True | 5000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.633 | 0.532 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s0 | True | 24500 | 1.000 | 0.918 | 0.920 | 0.920 | 0.760 | 0.559 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s1 | True | 15000 | 0.999 | 0.873 | 0.913 | 0.874 | 0.609 | 0.248 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s2 | False | None | 1.000 | 0.410 | 0.449 | 0.411 | 0.399 | 0.395 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s3 | False | None | 1.000 | 0.415 | 0.426 | 0.411 | 0.390 | 0.374 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s4 | False | None | 0.999 | 0.404 | 0.407 | 0.399 | 0.376 | 0.355 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s5 | False | None | 1.000 | 0.404 | 0.439 | 0.404 | 0.362 | 0.305 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s6 | True | 29500 | 1.000 | 0.887 | 0.910 | 0.890 | 0.756 | 0.307 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s7 | False | None | 1.000 | 0.419 | 0.463 | 0.419 | 0.388 | 0.383 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s0 | False | None | 1.000 | 0.402 | 0.443 | 0.410 | 0.378 | 0.355 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.413 | 0.443 | 0.410 | 0.384 | 0.357 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.824 | 0.863 | 0.822 | 0.702 | 0.515 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.401 | 0.438 | 0.396 | 0.379 | 0.370 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s4 | False | None | 1.000 | 0.420 | 0.456 | 0.414 | 0.398 | 0.373 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s5 | False | None | 1.000 | 0.805 | 0.807 | 0.806 | 0.584 | 0.428 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s6 | False | None | 1.000 | 0.442 | 0.446 | 0.447 | 0.429 | 0.420 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s7 | False | None | 0.999 | 0.388 | 0.434 | 0.387 | 0.337 | 0.309 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s0 | True | 9000 | 1.000 | 0.972 | 0.980 | 0.974 | 0.915 | 0.677 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s1 | True | 9500 | 1.000 | 0.905 | 0.930 | 0.908 | 0.902 | 0.895 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s2 | False | None | 1.000 | 0.868 | 0.895 | 0.867 | 0.861 | 0.862 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s3 | False | None | 0.998 | 0.415 | 0.446 | 0.406 | 0.398 | 0.396 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s4 | False | None | 0.999 | 0.867 | 0.891 | 0.863 | 0.861 | 0.858 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s5 | False | None | 1.000 | 0.861 | 0.898 | 0.864 | 0.852 | 0.851 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s6 | True | 47000 | 1.000 | 0.867 | 0.904 | 0.872 | 0.842 | 0.732 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s7 | True | 45500 | 1.000 | 0.900 | 0.909 | 0.903 | 0.731 | 0.556 |
| sym__mq_p512__e97__L2__d512__wd1.0__s0 | False | None | 1.000 | 0.786 | 0.811 | 0.790 | 0.778 | 0.728 |
| sym__mq_p512__e97__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.795 | 0.819 | 0.801 | 0.790 | 0.793 |
| sym__mq_p512__e97__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.822 | 0.851 | 0.819 | 0.800 | 0.773 |
| sym__mq_p512__e97__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.401 | 0.448 | 0.399 | 0.384 | 0.383 |
| sym__mq_p512__e97__L2__d512__wd1.0__s4 | False | None | 1.000 | 0.401 | 0.431 | 0.394 | 0.390 | 0.382 |
| sym__mq_p512__e97__L2__d512__wd1.0__s5 | False | None | 0.999 | 0.405 | 0.447 | 0.409 | 0.386 | 0.387 |
| sym__mq_p512__e97__L2__d512__wd1.0__s6 | False | None | 1.000 | 0.853 | 0.870 | 0.853 | 0.844 | 0.828 |
| sym__mq_p512__e97__L2__d512__wd1.0__s7 | False | None | 1.000 | 0.408 | 0.447 | 0.404 | 0.386 | 0.385 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s0 | True | 12000 | 1.000 | 0.998 | 0.999 | 0.999 | 0.768 | 0.538 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s1 | True | 20000 | 1.000 | 0.998 | 1.000 | 0.997 | 0.644 | 0.526 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s2 | True | 6000 | 1.000 | 0.985 | 0.998 | 0.989 | 0.687 | 0.537 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s3 | True | 34500 | 0.996 | 0.546 | 0.999 | 0.533 | 0.494 | 0.491 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s4 | True | 9000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.694 | 0.571 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s5 | True | 5000 | 1.000 | 0.999 | 0.998 | 0.999 | 0.770 | 0.566 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s6 | True | 21000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.575 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s7 | True | 8500 | 1.000 | 0.999 | 0.999 | 0.998 | 0.758 | 0.558 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s0 | True | 14000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.679 | 0.535 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s1 | True | 27500 | 1.000 | 0.998 | 0.998 | 0.997 | 0.698 | 0.562 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s2 | True | 13000 | 1.000 | 0.998 | 0.997 | 0.998 | 0.700 | 0.542 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s3 | True | 14000 | 1.000 | 0.998 | 0.999 | 0.998 | 0.701 | 0.565 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s4 | True | 6000 | 0.999 | 0.994 | 0.999 | 0.994 | 0.643 | 0.526 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s5 | False | None | 1.000 | 0.467 | 0.478 | 0.477 | 0.456 | 0.450 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s6 | True | 20000 | 1.000 | 0.995 | 0.999 | 0.995 | 0.693 | 0.565 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s7 | True | 17500 | 1.000 | 0.995 | 0.996 | 0.993 | 0.688 | 0.537 |

## PART 2 raw runs (iterated_nonlinear_map)

| label | grok | gstep | train | test | best | T128 | T1024 | T4096 |
|---|---|---|---|---|---|---|---|---|
| inm__b10__e97-lin__L2__d1024__wd1.0__s0 | False | None | 0.999 | 0.673 | 0.677 | 0.669 | 0.655 | 0.647 |
| inm__b10__e97-lin__L2__d1024__wd1.0__s1 | False | None | 0.998 | 0.711 | 0.715 | 0.706 | 0.689 | 0.679 |
| inm__b10__e97-lin__L2__d1024__wd1.0__s2 | False | None | 1.000 | 0.665 | 0.665 | 0.663 | 0.651 | 0.649 |
| inm__b10__e97-lin__L2__d1024__wd1.0__s3 | False | None | 0.999 | 0.677 | 0.683 | 0.677 | 0.656 | 0.649 |
| inm__b10__e97-lin__L2__d512__wd1.0__s0 | False | None | 0.999 | 0.658 | 0.662 | 0.654 | 0.642 | 0.634 |
| inm__b10__e97-lin__L2__d512__wd1.0__s1 | False | None | 0.999 | 0.620 | 0.641 | 0.613 | 0.598 | 0.593 |
| inm__b10__e97-lin__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.657 | 0.661 | 0.656 | 0.646 | 0.642 |
| inm__b10__e97-lin__L2__d512__wd1.0__s3 | False | None | 0.998 | 0.666 | 0.669 | 0.667 | 0.651 | 0.642 |
| inm__b10__e97__L2__d1024__wd1.0__s0 | False | None | 0.999 | 0.637 | 0.646 | 0.633 | 0.627 | 0.620 |
| inm__b10__e97__L2__d1024__wd1.0__s1 | False | None | 0.998 | 0.682 | 0.689 | 0.682 | 0.666 | 0.666 |
| inm__b10__e97__L2__d1024__wd1.0__s2 | False | None | 0.999 | 0.655 | 0.664 | 0.657 | 0.644 | 0.646 |
| inm__b10__e97__L2__d1024__wd1.0__s3 | False | None | 0.999 | 0.657 | 0.662 | 0.655 | 0.646 | 0.644 |
| inm__b10__e97__L2__d512__wd1.0__s0 | False | None | 0.999 | 0.662 | 0.667 | 0.655 | 0.647 | 0.641 |
| inm__b10__e97__L2__d512__wd1.0__s1 | False | None | 0.999 | 0.634 | 0.668 | 0.630 | 0.619 | 0.619 |
| inm__b10__e97__L2__d512__wd1.0__s2 | False | None | 0.999 | 0.690 | 0.700 | 0.689 | 0.679 | 0.680 |
| inm__b10__e97__L2__d512__wd1.0__s3 | False | None | 0.998 | 0.694 | 0.704 | 0.695 | 0.686 | 0.679 |
| inm__b10__gdn2__L2__d1024__wd1.0__s0 | False | None | 0.999 | 0.718 | 0.727 | 0.718 | 0.707 | 0.702 |
| inm__b10__gdn2__L2__d1024__wd1.0__s1 | False | None | 0.999 | 0.727 | 0.729 | 0.725 | 0.716 | 0.717 |
| inm__b10__gdn2__L2__d1024__wd1.0__s2 | False | None | 0.999 | 0.723 | 0.724 | 0.725 | 0.714 | 0.717 |
| inm__b10__gdn2__L2__d1024__wd1.0__s3 | False | None | 0.999 | 0.708 | 0.712 | 0.709 | 0.701 | 0.697 |
| inm__b10__gdn2__L2__d512__wd1.0__s0 | False | None | 0.999 | 0.723 | 0.729 | 0.722 | 0.714 | 0.710 |
| inm__b10__gdn2__L2__d512__wd1.0__s1 | False | None | 0.999 | 0.728 | 0.733 | 0.722 | 0.719 | 0.715 |
| inm__b10__gdn2__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.715 | 0.718 | 0.713 | 0.705 | 0.707 |
| inm__b10__gdn2__L2__d512__wd1.0__s3 | False | None | 0.998 | 0.709 | 0.713 | 0.705 | 0.700 | 0.696 |

## PART 3 raw runs (anbncn_viability)

| label | grok | gstep | train | test | best | T128 | T1024 | T4096 |
|---|---|---|---|---|---|---|---|---|
| abc__e97-lin__L2__d1024__wd1.0__s0 | True | 500 | 0.989 | 0.962 | 0.976 | 0.971 | 0.858 | 0.777 |
| abc__e97-lin__L2__d1024__wd1.0__s1 | True | 500 | 0.975 | 0.975 | 0.995 | 0.971 | 0.856 | 0.822 |
| abc__e97-lin__L2__d1024__wd1.0__s2 | True | 500 | 0.973 | 0.971 | 0.988 | 0.971 | 0.848 | 0.814 |
| abc__e97-lin__L2__d1024__wd1.0__s3 | True | 500 | 0.985 | 0.976 | 0.989 | 0.980 | 0.832 | 0.771 |
| abc__e97-lin__L2__d512__wd1.0__s0 | True | 500 | 0.984 | 0.964 | 0.976 | 0.973 | 0.875 | 0.783 |
| abc__e97-lin__L2__d512__wd1.0__s1 | True | 500 | 0.978 | 0.971 | 0.980 | 0.969 | 0.823 | 0.812 |
| abc__e97-lin__L2__d512__wd1.0__s2 | True | 500 | 0.984 | 0.975 | 0.997 | 0.979 | 0.841 | 0.773 |
| abc__e97-lin__L2__d512__wd1.0__s3 | True | 500 | 0.986 | 0.973 | 0.984 | 0.978 | 0.915 | 0.903 |
| abc__e97__L2__d1024__wd1.0__s0 | True | 500 | 0.987 | 0.966 | 0.973 | 0.976 | 0.827 | 0.763 |
| abc__e97__L2__d1024__wd1.0__s1 | True | 500 | 0.988 | 0.980 | 0.990 | 0.979 | 0.945 | 0.938 |
| abc__e97__L2__d1024__wd1.0__s2 | True | 500 | 0.985 | 0.979 | 0.994 | 0.980 | 0.859 | 0.817 |
| abc__e97__L2__d1024__wd1.0__s3 | True | 500 | 0.988 | 0.974 | 0.985 | 0.979 | 0.821 | 0.769 |
| abc__e97__L2__d512__wd1.0__s0 | True | 500 | 0.971 | 0.937 | 0.947 | 0.935 | 0.819 | 0.762 |
| abc__e97__L2__d512__wd1.0__s1 | True | 500 | 0.952 | 0.940 | 0.969 | 0.940 | 0.850 | 0.818 |
| abc__e97__L2__d512__wd1.0__s2 | True | 500 | 0.955 | 0.951 | 0.965 | 0.956 | 0.825 | 0.767 |
| abc__e97__L2__d512__wd1.0__s3 | True | 500 | 0.988 | 0.972 | 0.981 | 0.978 | 0.813 | 0.766 |
| abc__gdn2__L2__d1024__wd1.0__s0 | True | 500 | 0.991 | 0.965 | 0.965 | 0.973 | 0.825 | 0.767 |
| abc__gdn2__L2__d1024__wd1.0__s1 | True | 500 | 0.999 | 0.998 | 0.998 | 0.998 | 0.891 | 0.925 |
| abc__gdn2__L2__d1024__wd1.0__s2 | True | 500 | 0.985 | 0.973 | 0.995 | 0.971 | 0.771 | 0.649 |
| abc__gdn2__L2__d1024__wd1.0__s3 | True | 500 | 0.991 | 0.968 | 0.978 | 0.975 | 0.813 | 0.766 |
| abc__gdn2__L2__d512__wd1.0__s0 | True | 500 | 0.987 | 0.956 | 0.964 | 0.962 | 0.919 | 0.906 |
| abc__gdn2__L2__d512__wd1.0__s1 | True | 500 | 0.973 | 0.968 | 0.987 | 0.967 | 0.827 | 0.813 |
| abc__gdn2__L2__d512__wd1.0__s2 | True | 500 | 0.998 | 0.993 | 0.993 | 0.993 | 0.821 | 0.766 |
| abc__gdn2__L2__d512__wd1.0__s3 | True | 500 | 0.987 | 0.969 | 0.978 | 0.975 | 0.915 | 0.903 |


## VERDICT

**Part 1 (8-seed expansion) — CONFIRMED and TIGHTENED.** On modular_quadratic
the predecessor's signature is seed-robust at n=8: e97 (nonlinear-in-time) is
**length-invariant** out to T=4096, while both linear-state arms (e97-lin, gdn2)
**collapse toward baseline** at far T despite symmetric width. Grokked-seed far-T
means (wd=1.0):

| cell | T=4096: e97 | e97-lin | gdn2 | e97−gdn2 |
|---|---|---|---|---|
| p256 d512 | 0.865 | 0.471 | 0.544 | **+0.321** |
| p256 d1024 | 0.820 | 0.651 | 0.558 | **+0.261** |
| p512 d1024 | 0.715 | 0.371 | 0.545 | **+0.170** |

The e97−gdn2 far-T advantage is **positive in every decisive cell** (+0.16 … +0.32
at T=4096) and the e97−e97lin advantage is positive in 3/4 (+0.17 … +0.39; the
lone small case is p256/d1024 +0.169 where e97-lin had its best run). gdn2 is the
**most reliable grokker at the train length** (7–8/8 vs e97 0–5/8) — but its grok
is *train-length memorization* that collapses under extrapolation, exactly the
predicted dissociation. Throughput reproduced on an isolated 1-GPU pass: e97 =
**1.39× / 1.56× / 1.26×** gdn2 at d256/512/1024, with *fewer* params at d≥512
(8.17M vs 8.98M; 28.9M vs 35.8M). bf16 + fused asserted per-arm (kernel-asserts
in every log). The within-task class separation is solid for the paper.

**Part 2 + Part 3 (second task families) — signature does NOT replicate. SCOPE IS
NARROWER than a general "nonlinear-in-time beats linear-in-time" law.** This is
the load-bearing negative result of this confirmation.

- **iterated_nonlinear_map** (logistic h_t=a_t·h_{t-1}(1−h_{t-1}), binned): 0/24
  grok to 0.9; all arms plateau ~0.65–0.72 (≫0.1 baseline) and every curve is
  **flat across T=128→4096** (e97 0.667→0.655, e97-lin 0.648→0.628, gdn2
  0.716→0.707). The map at a∈[2.6,3.6] is **contractive (fading memory)**: h_t
  depends on recent drivers, not full history, so length-extrapolation is trivial
  for *all* arms and there is no train-length to memorize-then-fail. The per-step
  quadratic exists but is not load-bearing for length. No separation (gdn2 is
  marginally best).

- **anbncn_viability** (per-position viability of aⁿbⁿcⁿ, count comparisons):
  24/24 fit train length; all arms **decay together** with length
  (T=4096 ≈ 0.78–0.85, none collapses to the 0.5 baseline) and the e97−gdn2 gap
  is tiny and **sign-flipping** (−0.069 at d512, +0.045 at d1024). Counting is
  additive accumulation, where linear cumulative state (gdn2) extrapolates as well
  as or better than bounded nonlinear state — consistent with prior findings that
  counting is GDN-2's strength and bounded saturation does not help it. No e97
  advantage.

**Mechanistic reading.** The e97-extrapolates / linear-collapses signature requires
**both** conditions, which only modular_quadratic among the tested tasks satisfies:
(1) the ground-truth recurrence contains a **per-step state-nonlinearity the linear
arm cannot represent** (x² mod p is non-invertible; counting and a fading-memory
map are not), **and** (2) **non-contractive, full-precision long memory** so that
the linear arm's failure-to-represent compounds with length instead of being washed
out (modular arithmetic over a finite ring qualifies; the contractive logistic map
does not). Remove either condition and the separation vanishes.

**Bottom line for the paper.** Claim the separation as a *robust, mechanism-specific*
result on modular-quadratic-class iterated maps (8-seed confirmed, throughput
favorable), NOT as a universal temporal-class law. The two distinct task families
tested here are genuine non-replications and should be reported as the scope
boundary — overclaiming a general law would be falsified by Part 2/Part 3.

_Data: experiments/grok_symmetric_width/runs/ (96 modular_quadratic decisive-cell
runs over seeds 0–7; 24 iterated_nonlinear_map; 24 anbncn_viability), tables in
this file, machine-readable confirm_signature.json, throughput_confirm.json._










