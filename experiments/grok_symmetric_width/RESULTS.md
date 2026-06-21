# Symmetric Width + Far Length-Extrapolation — Results

_216 runs. modular_quadratic x_t=(x_{t-1}^2+c_t) mod p; per-position supervision; L=2; bf16+fused. WIDTH applied symmetrically to ALL arms (e97 nonlinear, e97-lin linear, gdn2 linear) at dim{256,512,1024}; extrap pushed to T=4096._


## 1. Final held-out accuracy (seed-pooled over {0,1,2,3}; mean acc / grok-rate / n)

acc = mean_seeds max(final_test,best_test); gr = frac seeds test>=0.9.


### wd=1.0

| p | dim | e97 (gr,n) | e97-lin (gr,n) | gdn2 (gr,n) | baseline |
|---|---|---|---|---|---|
| 64 | 256 | 0.752 (0.50,4) | 0.874 (0.75,4) | 1.000 (1.00,4) | 0.0156 |
| 64 | 512 | 0.999 (1.00,4) | 1.000 (1.00,4) | 1.000 (1.00,4) | 0.0156 |
| 64 | 1024 | 0.996 (1.00,4) | 0.999 (1.00,4) | 1.000 (1.00,4) | 0.0156 |
| 256 | 256 | 0.940 (1.00,4) | 0.580 (0.25,4) | 1.000 (1.00,4) | 0.0039 |
| 256 | 512 | 0.587 (0.25,4) | 0.697 (0.25,4) | 1.000 (1.00,4) | 0.0039 |
| 256 | 1024 | 0.836 (0.50,4) | 0.694 (0.25,4) | 1.000 (1.00,4) | 0.0039 |
| 512 | 256 | 0.537 (0.00,4) | 0.434 (0.00,4) | 0.995 (1.00,4) | 0.0020 |
| 512 | 512 | 0.732 (0.00,4) | 0.547 (0.00,4) | 0.998 (1.00,4) | 0.0020 |
| 512 | 1024 | 0.813 (0.50,4) | 0.677 (0.50,4) | 0.999 (1.00,4) | 0.0020 |

### wd=0.1

| p | dim | e97 (gr,n) | e97-lin (gr,n) | gdn2 (gr,n) | baseline |
|---|---|---|---|---|---|
| 64 | 256 | 0.753 (0.50,4) | 0.631 (0.25,4) | 0.997 (1.00,4) | 0.0156 |
| 64 | 512 | 0.893 (0.75,4) | 0.631 (0.25,4) | 1.000 (1.00,4) | 0.0156 |
| 64 | 1024 | 0.977 (1.00,4) | 0.879 (0.75,4) | 1.000 (1.00,4) | 0.0156 |
| 256 | 256 | 0.824 (0.75,4) | 0.600 (0.25,4) | 0.497 (0.00,4) | 0.0039 |
| 256 | 512 | 0.833 (0.75,4) | 0.601 (0.25,4) | 0.610 (0.25,4) | 0.0039 |
| 256 | 1024 | 0.722 (0.50,4) | 0.586 (0.25,4) | 0.998 (1.00,4) | 0.0039 |
| 512 | 256 | 0.449 (0.00,4) | 0.449 (0.00,4) | 0.468 (0.00,4) | 0.0020 |
| 512 | 512 | 0.477 (0.00,4) | 0.557 (0.00,4) | 0.507 (0.00,4) | 0.0020 |
| 512 | 1024 | 0.521 (0.00,4) | 0.446 (0.00,4) | 0.486 (0.00,4) | 0.0020 |

## 2. Length-extrapolation: test-acc vs T per (arm x dim) — HOLDS vs COLLAPSES

Curve over seeds that GROKKED the train length (final_test>=0.9), so the memorization-vs-rule question is clean (n = #grokked seeds). A model that learned the RULE holds as T grows; one that MEMORIZED the train length (T=128) collapses toward baseline at far T. wd=1.0.


**p=64** (baseline 1/p = 0.0156)

| arm | dim | n | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 256 | 2 | 0.998 | 0.998 | 0.996 | 0.993 | 0.998 | 0.994 |
| e97-lin | 256 | 3 | 0.996 | 0.988 | 0.968 | 0.892 | 0.774 | 0.659 |
| gdn2 | 256 | 4 | 0.990 | 0.930 | 0.764 | 0.629 | 0.558 | 0.529 |
| e97 | 512 | 4 | 0.996 | 0.993 | 0.992 | 0.985 | 0.977 | 0.970 |
| e97-lin | 512 | 4 | 0.912 | 0.892 | 0.869 | 0.810 | 0.730 | 0.641 |
| gdn2 | 512 | 4 | 1.000 | 0.990 | 0.870 | 0.704 | 0.602 | 0.558 |
| e97 | 1024 | 4 | 0.986 | 0.982 | 0.960 | 0.948 | 0.914 | 0.880 |
| e97-lin | 1024 | 4 | 0.999 | 0.997 | 0.989 | 0.953 | 0.845 | 0.685 |
| gdn2 | 1024 | 4 | 0.999 | 0.982 | 0.888 | 0.746 | 0.655 | 0.555 |

**p=256** (baseline 1/p = 0.0039)

| arm | dim | n | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 256 | 4 | 0.931 | 0.930 | 0.928 | 0.927 | 0.927 | 0.919 |
| e97-lin | 256 | 1 | 0.899 | 0.873 | 0.829 | 0.756 | 0.621 | 0.596 |
| gdn2 | 256 | 4 | 0.997 | 0.962 | 0.833 | 0.683 | 0.604 | 0.551 |
| e97 | 512 | 1 | 0.915 | 0.913 | 0.914 | 0.913 | 0.913 | 0.903 |
| e97-lin | 512 | 1 | 0.955 | 0.950 | 0.943 | 0.908 | 0.842 | 0.793 |
| gdn2 | 512 | 4 | 0.998 | 0.962 | 0.817 | 0.659 | 0.580 | 0.542 |
| e97 | 1024 | 2 | 0.982 | 0.981 | 0.981 | 0.981 | 0.981 | 0.981 |
| e97-lin | 1024 | 1 | 0.968 | 0.968 | 0.957 | 0.892 | 0.787 | 0.619 |
| gdn2 | 1024 | 4 | 0.994 | 0.989 | 0.927 | 0.776 | 0.658 | 0.560 |

**p=512** (baseline 1/p = 0.0020)

| arm | dim | n | T=128 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|---|---|---|
| e97 | 256 | 0 |   -   |   -   |   -   |   -   |   -   |   -   |
| e97-lin | 256 | 0 |   -   |   -   |   -   |   -   |   -   |   -   |
| gdn2 | 256 | 4 | 0.991 | 0.941 | 0.791 | 0.645 | 0.574 | 0.529 |
| e97 | 512 | 0 |   -   |   -   |   -   |   -   |   -   |   -   |
| e97-lin | 512 | 0 |   -   |   -   |   -   |   -   |   -   |   -   |
| gdn2 | 512 | 4 | 0.998 | 0.980 | 0.852 | 0.694 | 0.593 | 0.551 |
| e97 | 1024 | 2 | 0.941 | 0.938 | 0.933 | 0.909 | 0.844 | 0.786 |
| e97-lin | 1024 | 2 | 0.897 | 0.885 | 0.838 | 0.684 | 0.515 | 0.404 |
| gdn2 | 1024 | 4 | 0.879 | 0.851 | 0.759 | 0.648 | 0.587 | 0.523 |

## 3. The cliff — extrapolation at FAR T (2048, 4096), high p, WIDE (grokked seeds, wd=1.0)

Does symmetric width let the LINEAR arms EXTRAPOLATE, or only memorize the train length? Compare e97 vs the linear arms at the widest dim and farthest T.

| p | dim | T | e97 | e97-lin | gdn2 | e97 - e97lin | e97 - gdn2 |
|---|---|---|---|---|---|---|---|
| 256 | 512 | 2048 | 0.913 | 0.842 | 0.580 | 0.071 | 0.333 |
| 256 | 512 | 4096 | 0.903 | 0.793 | 0.542 | 0.109 | 0.361 |
| 256 | 1024 | 2048 | 0.981 | 0.787 | 0.658 | 0.194 | 0.323 |
| 256 | 1024 | 4096 | 0.981 | 0.619 | 0.560 | 0.362 | 0.421 |
| 512 | 512 | 2048 |   -   |   -   | 0.593 |   -   |   -   |
| 512 | 512 | 4096 |   -   |   -   | 0.551 |   -   |   -   |
| 512 | 1024 | 2048 | 0.844 | 0.515 | 0.587 | 0.329 | 0.257 |
| 512 | 1024 | 4096 | 0.786 | 0.404 | 0.523 | 0.382 | 0.263 |

## 4. Throughput (tok/s, fwd+bwd+step at T=128 bs=64) — isolated 1-GPU pass (clean_throughput.py, max of 3 reps)

| dim | e97 | e97-lin | gdn2 | e97/gdn2 | e97/e97-lin | params(e97/gdn2) |
|---|---|---|---|---|---|---|
| 256 | 811497 | 798562 | 508811 | 1.59 | 1.02 | 2.51M / 2.26M |
| 512 | 775352 | 780901 | 492786 | 1.57 | 0.99 | 8.17M / 8.97M |
| 1024 | 410982 | 409323 | 328010 | 1.25 | 1.00 | 28.92M / 35.77M |

## 5. All runs (raw)

| label | grok | gstep | train | test | best | T128 | T1024 | T4096 | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| sym__mq_p256__e97-lin__L2__d1024__wd0.1__s0 | True | 44500 | 1.000 | 0.891 | 0.923 | 0.896 | 0.859 | 0.583 | 405289 |
| sym__mq_p256__e97-lin__L2__d1024__wd0.1__s1 | False | None | 1.000 | 0.435 | 0.470 | 0.434 | 0.279 | 0.124 | 409784 |
| sym__mq_p256__e97-lin__L2__d1024__wd0.1__s2 | False | None | 1.000 | 0.473 | 0.477 | 0.470 | 0.454 | 0.453 | 223493 |
| sym__mq_p256__e97-lin__L2__d1024__wd0.1__s3 | False | None | 1.000 | 0.437 | 0.473 | 0.435 | 0.418 | 0.415 | 427710 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s0 | False | None | 1.000 | 0.855 | 0.879 | 0.861 | 0.668 | 0.441 | 405203 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s1 | False | None | 1.000 | 0.429 | 0.461 | 0.427 | 0.405 | 0.372 | 412470 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s2 | True | 16000 | 0.999 | 0.969 | 0.979 | 0.968 | 0.892 | 0.619 | 279939 |
| sym__mq_p256__e97-lin__L2__d1024__wd1.0__s3 | False | None | 0.999 | 0.453 | 0.456 | 0.449 | 0.436 | 0.435 | 425549 |
| sym__mq_p256__e97-lin__L2__d256__wd0.1__s0 | True | 30000 | 1.000 | 0.949 | 0.949 | 0.952 | 0.930 | 0.745 | 872077 |
| sym__mq_p256__e97-lin__L2__d256__wd0.1__s1 | False | None | 1.000 | 0.485 | 0.485 | 0.487 | 0.458 | 0.451 | 883009 |
| sym__mq_p256__e97-lin__L2__d256__wd0.1__s2 | False | None | 1.000 | 0.485 | 0.487 | 0.486 | 0.463 | 0.457 | 869183 |
| sym__mq_p256__e97-lin__L2__d256__wd0.1__s3 | False | None | 1.000 | 0.480 | 0.479 | 0.477 | 0.460 | 0.452 | 871106 |
| sym__mq_p256__e97-lin__L2__d256__wd1.0__s0 | False | None | 0.999 | 0.436 | 0.448 | 0.441 | 0.402 | 0.379 | 855580 |
| sym__mq_p256__e97-lin__L2__d256__wd1.0__s1 | False | None | 0.999 | 0.452 | 0.469 | 0.455 | 0.431 | 0.418 | 901481 |
| sym__mq_p256__e97-lin__L2__d256__wd1.0__s2 | True | 41000 | 0.996 | 0.894 | 0.925 | 0.899 | 0.756 | 0.596 | 818340 |
| sym__mq_p256__e97-lin__L2__d256__wd1.0__s3 | False | None | 1.000 | 0.458 | 0.480 | 0.453 | 0.423 | 0.403 | 898780 |
| sym__mq_p256__e97-lin__L2__d512__wd0.1__s0 | False | None | 0.999 | 0.440 | 0.487 | 0.442 | 0.408 | 0.395 | 868401 |
| sym__mq_p256__e97-lin__L2__d512__wd0.1__s1 | True | 27000 | 1.000 | 0.967 | 0.966 | 0.967 | 0.952 | 0.798 | 904195 |
| sym__mq_p256__e97-lin__L2__d512__wd0.1__s2 | False | None | 1.000 | 0.467 | 0.468 | 0.467 | 0.449 | 0.440 | 609714 |
| sym__mq_p256__e97-lin__L2__d512__wd0.1__s3 | False | None | 1.000 | 0.442 | 0.481 | 0.435 | 0.419 | 0.413 | 888947 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s0 | True | 12500 | 1.000 | 0.954 | 0.962 | 0.955 | 0.908 | 0.793 | 860718 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.889 | 0.891 | 0.890 | 0.620 | 0.480 | 586403 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.442 | 0.463 | 0.440 | 0.417 | 0.394 | 889619 |
| sym__mq_p256__e97-lin__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.444 | 0.474 | 0.445 | 0.430 | 0.418 | 893903 |
| sym__mq_p256__e97__L2__d1024__wd0.1__s0 | False | None | 1.000 | 0.486 | 0.499 | 0.492 | 0.463 | 0.453 | 194037 |
| sym__mq_p256__e97__L2__d1024__wd0.1__s1 | True | 7000 | 1.000 | 0.950 | 0.980 | 0.952 | 0.950 | 0.951 | 423949 |
| sym__mq_p256__e97__L2__d1024__wd0.1__s2 | True | 41500 | 1.000 | 0.920 | 0.939 | 0.918 | 0.917 | 0.918 | 182057 |
| sym__mq_p256__e97__L2__d1024__wd0.1__s3 | False | None | 1.000 | 0.470 | 0.470 | 0.462 | 0.446 | 0.448 | 417653 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s0 | False | None | 0.999 | 0.899 | 0.899 | 0.905 | 0.894 | 0.884 | 390436 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s1 | True | 5500 | 1.000 | 0.987 | 0.990 | 0.988 | 0.987 | 0.988 | 422823 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s2 | True | 4500 | 1.000 | 0.976 | 0.989 | 0.977 | 0.975 | 0.975 | 162518 |
| sym__mq_p256__e97__L2__d1024__wd1.0__s3 | False | None | 0.999 | 0.466 | 0.466 | 0.459 | 0.442 | 0.435 | 416329 |
| sym__mq_p256__e97__L2__d256__wd0.1__s0 | True | 18000 | 1.000 | 0.968 | 0.972 | 0.969 | 0.968 | 0.968 | 885422 |
| sym__mq_p256__e97__L2__d256__wd0.1__s1 | False | None | 1.000 | 0.444 | 0.479 | 0.447 | 0.430 | 0.430 | 892397 |
| sym__mq_p256__e97__L2__d256__wd0.1__s2 | True | 47000 | 1.000 | 0.892 | 0.907 | 0.891 | 0.887 | 0.887 | 890120 |
| sym__mq_p256__e97__L2__d256__wd0.1__s3 | True | 40500 | 1.000 | 0.937 | 0.937 | 0.936 | 0.935 | 0.935 | 903334 |
| sym__mq_p256__e97__L2__d256__wd1.0__s0 | True | 6500 | 1.000 | 0.944 | 0.957 | 0.947 | 0.944 | 0.944 | 812220 |
| sym__mq_p256__e97__L2__d256__wd1.0__s1 | True | 19500 | 1.000 | 0.932 | 0.940 | 0.933 | 0.929 | 0.929 | 902384 |
| sym__mq_p256__e97__L2__d256__wd1.0__s2 | True | 11000 | 1.000 | 0.927 | 0.930 | 0.926 | 0.918 | 0.887 | 882701 |
| sym__mq_p256__e97__L2__d256__wd1.0__s3 | True | 17500 | 1.000 | 0.922 | 0.934 | 0.920 | 0.918 | 0.917 | 887252 |
| sym__mq_p256__e97__L2__d512__wd0.1__s0 | True | 32500 | 1.000 | 0.946 | 0.967 | 0.947 | 0.946 | 0.946 | 744593 |
| sym__mq_p256__e97__L2__d512__wd0.1__s1 | True | 43000 | 1.000 | 0.947 | 0.948 | 0.948 | 0.938 | 0.937 | 829405 |
| sym__mq_p256__e97__L2__d512__wd0.1__s2 | False | None | 1.000 | 0.437 | 0.479 | 0.436 | 0.421 | 0.419 | 489467 |
| sym__mq_p256__e97__L2__d512__wd0.1__s3 | True | 40000 | 1.000 | 0.938 | 0.937 | 0.937 | 0.937 | 0.937 | 886701 |
| sym__mq_p256__e97__L2__d512__wd1.0__s0 | True | 16000 | 0.999 | 0.914 | 0.946 | 0.915 | 0.913 | 0.903 | 889681 |
| sym__mq_p256__e97__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.435 | 0.455 | 0.435 | 0.422 | 0.425 | 876509 |
| sym__mq_p256__e97__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.446 | 0.470 | 0.443 | 0.434 | 0.431 | 833851 |
| sym__mq_p256__e97__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.477 | 0.477 | 0.475 | 0.459 | 0.459 | 885349 |
| sym__mq_p256__gdn2__L2__d1024__wd0.1__s0 | True | 41500 | 1.000 | 0.999 | 0.998 | 0.998 | 0.841 | 0.602 | 183693 |
| sym__mq_p256__gdn2__L2__d1024__wd0.1__s1 | True | 34000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.932 | 0.641 | 274094 |
| sym__mq_p256__gdn2__L2__d1024__wd0.1__s2 | True | 40000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.883 | 0.663 | 187417 |
| sym__mq_p256__gdn2__L2__d1024__wd0.1__s3 | True | 41500 | 1.000 | 0.992 | 0.993 | 0.988 | 0.782 | 0.582 | 211875 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s0 | True | 5000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.808 | 0.571 | 284864 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s1 | True | 8000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.829 | 0.574 | 308199 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s2 | True | 6500 | 1.000 | 0.999 | 0.999 | 1.000 | 0.681 | 0.522 | 294730 |
| sym__mq_p256__gdn2__L2__d1024__wd1.0__s3 | True | 8500 | 1.000 | 0.983 | 0.999 | 0.977 | 0.784 | 0.575 | 198683 |
| sym__mq_p256__gdn2__L2__d256__wd0.1__s0 | False | None | 0.999 | 0.464 | 0.492 | 0.472 | 0.460 | 0.455 | 519112 |
| sym__mq_p256__gdn2__L2__d256__wd0.1__s1 | False | None | 0.999 | 0.469 | 0.496 | 0.468 | 0.458 | 0.458 | 524466 |
| sym__mq_p256__gdn2__L2__d256__wd0.1__s2 | False | None | 1.000 | 0.485 | 0.500 | 0.486 | 0.472 | 0.470 | 508299 |
| sym__mq_p256__gdn2__L2__d256__wd0.1__s3 | False | None | 1.000 | 0.474 | 0.501 | 0.467 | 0.458 | 0.458 | 524722 |
| sym__mq_p256__gdn2__L2__d256__wd1.0__s0 | True | 20500 | 1.000 | 0.993 | 0.999 | 0.991 | 0.617 | 0.512 | 524580 |
| sym__mq_p256__gdn2__L2__d256__wd1.0__s1 | True | 27000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.689 | 0.558 | 529133 |
| sym__mq_p256__gdn2__L2__d256__wd1.0__s2 | True | 11500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.755 | 0.587 | 524420 |
| sym__mq_p256__gdn2__L2__d256__wd1.0__s3 | True | 16000 | 1.000 | 0.998 | 1.000 | 0.999 | 0.670 | 0.549 | 544966 |
| sym__mq_p256__gdn2__L2__d512__wd0.1__s0 | False | None | 1.000 | 0.496 | 0.500 | 0.500 | 0.485 | 0.479 | 526229 |
| sym__mq_p256__gdn2__L2__d512__wd0.1__s1 | True | 49500 | 1.000 | 0.909 | 0.925 | 0.891 | 0.589 | 0.530 | 492929 |
| sym__mq_p256__gdn2__L2__d512__wd0.1__s2 | False | None | 1.000 | 0.470 | 0.499 | 0.471 | 0.461 | 0.462 | 523338 |
| sym__mq_p256__gdn2__L2__d512__wd0.1__s3 | False | None | 1.000 | 0.517 | 0.517 | 0.508 | 0.491 | 0.488 | 528122 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s0 | True | 10000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.681 | 0.542 | 527180 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s1 | True | 12500 | 1.000 | 0.996 | 1.000 | 0.995 | 0.692 | 0.547 | 522080 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s2 | True | 21000 | 1.000 | 0.991 | 0.999 | 0.997 | 0.635 | 0.535 | 523041 |
| sym__mq_p256__gdn2__L2__d512__wd1.0__s3 | True | 13500 | 1.000 | 0.998 | 1.000 | 0.999 | 0.629 | 0.542 | 516985 |
| sym__mq_p512__e97-lin__L2__d1024__wd0.1__s0 | False | None | 1.000 | 0.382 | 0.434 | 0.391 | 0.370 | 0.364 | 404751 |
| sym__mq_p512__e97-lin__L2__d1024__wd0.1__s1 | False | None | 1.000 | 0.424 | 0.436 | 0.431 | 0.412 | 0.412 | 409242 |
| sym__mq_p512__e97-lin__L2__d1024__wd0.1__s2 | False | None | 1.000 | 0.431 | 0.434 | 0.430 | 0.417 | 0.414 | 222130 |
| sym__mq_p512__e97-lin__L2__d1024__wd0.1__s3 | False | None | 1.000 | 0.446 | 0.478 | 0.435 | 0.383 | 0.371 | 422860 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s0 | True | 24500 | 1.000 | 0.918 | 0.920 | 0.920 | 0.760 | 0.559 | 406498 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s1 | True | 15000 | 0.999 | 0.873 | 0.913 | 0.874 | 0.609 | 0.248 | 407441 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s2 | False | None | 1.000 | 0.410 | 0.449 | 0.411 | 0.399 | 0.395 | 195283 |
| sym__mq_p512__e97-lin__L2__d1024__wd1.0__s3 | False | None | 1.000 | 0.415 | 0.426 | 0.411 | 0.390 | 0.374 | 424648 |
| sym__mq_p512__e97-lin__L2__d256__wd0.1__s0 | False | None | 1.000 | 0.444 | 0.447 | 0.450 | 0.435 | 0.434 | 901845 |
| sym__mq_p512__e97-lin__L2__d256__wd0.1__s1 | False | None | 1.000 | 0.388 | 0.448 | 0.392 | 0.360 | 0.346 | 880760 |
| sym__mq_p512__e97-lin__L2__d256__wd0.1__s2 | False | None | 1.000 | 0.388 | 0.450 | 0.389 | 0.372 | 0.367 | 883460 |
| sym__mq_p512__e97-lin__L2__d256__wd0.1__s3 | False | None | 1.000 | 0.395 | 0.451 | 0.387 | 0.368 | 0.361 | 898154 |
| sym__mq_p512__e97-lin__L2__d256__wd1.0__s0 | False | None | 1.000 | 0.407 | 0.429 | 0.411 | 0.378 | 0.315 | 915722 |
| sym__mq_p512__e97-lin__L2__d256__wd1.0__s1 | False | None | 1.000 | 0.410 | 0.441 | 0.412 | 0.377 | 0.344 | 898443 |
| sym__mq_p512__e97-lin__L2__d256__wd1.0__s2 | False | None | 1.000 | 0.417 | 0.441 | 0.419 | 0.400 | 0.395 | 841730 |
| sym__mq_p512__e97-lin__L2__d256__wd1.0__s3 | False | None | 1.000 | 0.396 | 0.427 | 0.391 | 0.362 | 0.304 | 897960 |
| sym__mq_p512__e97-lin__L2__d512__wd0.1__s0 | False | None | 1.000 | 0.435 | 0.463 | 0.433 | 0.408 | 0.402 | 879832 |
| sym__mq_p512__e97-lin__L2__d512__wd0.1__s1 | False | None | 1.000 | 0.445 | 0.446 | 0.452 | 0.423 | 0.400 | 883123 |
| sym__mq_p512__e97-lin__L2__d512__wd0.1__s2 | False | None | 0.999 | 0.806 | 0.864 | 0.803 | 0.707 | 0.519 | 546965 |
| sym__mq_p512__e97-lin__L2__d512__wd0.1__s3 | False | None | 1.000 | 0.388 | 0.454 | 0.384 | 0.364 | 0.349 | 899099 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s0 | False | None | 1.000 | 0.402 | 0.443 | 0.410 | 0.378 | 0.355 | 806132 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.413 | 0.443 | 0.410 | 0.384 | 0.357 | 873725 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.824 | 0.863 | 0.822 | 0.702 | 0.515 | 857285 |
| sym__mq_p512__e97-lin__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.401 | 0.438 | 0.396 | 0.379 | 0.370 | 763135 |
| sym__mq_p512__e97__L2__d1024__wd0.1__s0 | False | None | 1.000 | 0.424 | 0.424 | 0.428 | 0.412 | 0.408 | 169741 |
| sym__mq_p512__e97__L2__d1024__wd0.1__s1 | False | None | 1.000 | 0.782 | 0.782 | 0.785 | 0.782 | 0.780 | 423399 |
| sym__mq_p512__e97__L2__d1024__wd0.1__s2 | False | None | 0.998 | 0.420 | 0.455 | 0.416 | 0.389 | 0.390 | 392892 |
| sym__mq_p512__e97__L2__d1024__wd0.1__s3 | False | None | 1.000 | 0.373 | 0.424 | 0.371 | 0.357 | 0.355 | 414651 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s0 | True | 9000 | 1.000 | 0.972 | 0.980 | 0.974 | 0.915 | 0.677 | 188875 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s1 | True | 9500 | 1.000 | 0.905 | 0.930 | 0.908 | 0.902 | 0.895 | 421620 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s2 | False | None | 1.000 | 0.868 | 0.895 | 0.867 | 0.861 | 0.862 | 181235 |
| sym__mq_p512__e97__L2__d1024__wd1.0__s3 | False | None | 0.998 | 0.415 | 0.446 | 0.406 | 0.398 | 0.396 | 413357 |
| sym__mq_p512__e97__L2__d256__wd0.1__s0 | False | None | 0.998 | 0.412 | 0.462 | 0.410 | 0.381 | 0.372 | 903606 |
| sym__mq_p512__e97__L2__d256__wd0.1__s1 | False | None | 1.000 | 0.441 | 0.442 | 0.439 | 0.426 | 0.424 | 886038 |
| sym__mq_p512__e97__L2__d256__wd0.1__s2 | False | None | 1.000 | 0.441 | 0.444 | 0.436 | 0.430 | 0.428 | 887053 |
| sym__mq_p512__e97__L2__d256__wd0.1__s3 | False | None | 0.999 | 0.388 | 0.449 | 0.384 | 0.371 | 0.366 | 805145 |
| sym__mq_p512__e97__L2__d256__wd1.0__s0 | False | None | 1.000 | 0.840 | 0.844 | 0.844 | 0.811 | 0.739 | 880746 |
| sym__mq_p512__e97__L2__d256__wd1.0__s1 | False | None | 1.000 | 0.405 | 0.433 | 0.406 | 0.386 | 0.385 | 882429 |
| sym__mq_p512__e97__L2__d256__wd1.0__s2 | False | None | 0.999 | 0.407 | 0.436 | 0.406 | 0.393 | 0.395 | 897996 |
| sym__mq_p512__e97__L2__d256__wd1.0__s3 | False | None | 0.999 | 0.403 | 0.436 | 0.396 | 0.383 | 0.378 | 870634 |
| sym__mq_p512__e97__L2__d512__wd0.1__s0 | False | None | 0.999 | 0.393 | 0.450 | 0.396 | 0.371 | 0.364 | 860537 |
| sym__mq_p512__e97__L2__d512__wd0.1__s1 | False | None | 1.000 | 0.550 | 0.566 | 0.557 | 0.449 | 0.444 | 817855 |
| sym__mq_p512__e97__L2__d512__wd0.1__s2 | False | None | 1.000 | 0.446 | 0.449 | 0.445 | 0.432 | 0.429 | 879775 |
| sym__mq_p512__e97__L2__d512__wd0.1__s3 | False | None | 1.000 | 0.440 | 0.443 | 0.440 | 0.426 | 0.427 | 886414 |
| sym__mq_p512__e97__L2__d512__wd1.0__s0 | False | None | 1.000 | 0.786 | 0.811 | 0.790 | 0.778 | 0.728 | 895242 |
| sym__mq_p512__e97__L2__d512__wd1.0__s1 | False | None | 1.000 | 0.795 | 0.819 | 0.801 | 0.790 | 0.793 | 533495 |
| sym__mq_p512__e97__L2__d512__wd1.0__s2 | False | None | 1.000 | 0.822 | 0.851 | 0.819 | 0.800 | 0.773 | 481420 |
| sym__mq_p512__e97__L2__d512__wd1.0__s3 | False | None | 1.000 | 0.401 | 0.448 | 0.399 | 0.384 | 0.383 | 892329 |
| sym__mq_p512__gdn2__L2__d1024__wd0.1__s0 | False | None | 1.000 | 0.481 | 0.483 | 0.484 | 0.467 | 0.465 | 184955 |
| sym__mq_p512__gdn2__L2__d1024__wd0.1__s1 | False | None | 1.000 | 0.467 | 0.489 | 0.470 | 0.432 | 0.433 | 307077 |
| sym__mq_p512__gdn2__L2__d1024__wd0.1__s2 | False | None | 1.000 | 0.473 | 0.480 | 0.473 | 0.461 | 0.460 | 297288 |
| sym__mq_p512__gdn2__L2__d1024__wd0.1__s3 | False | None | 1.000 | 0.485 | 0.493 | 0.484 | 0.469 | 0.467 | 303634 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s0 | True | 12000 | 1.000 | 0.998 | 0.999 | 0.999 | 0.768 | 0.538 | 284617 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s1 | True | 20000 | 1.000 | 0.998 | 1.000 | 0.997 | 0.644 | 0.526 | 305830 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s2 | True | 6000 | 1.000 | 0.985 | 0.998 | 0.989 | 0.687 | 0.537 | 299852 |
| sym__mq_p512__gdn2__L2__d1024__wd1.0__s3 | True | 34500 | 0.996 | 0.546 | 0.999 | 0.533 | 0.494 | 0.491 | 170279 |
| sym__mq_p512__gdn2__L2__d256__wd0.1__s0 | False | None | 1.000 | 0.421 | 0.466 | 0.426 | 0.415 | 0.410 | 527161 |
| sym__mq_p512__gdn2__L2__d256__wd0.1__s1 | False | None | 1.000 | 0.448 | 0.460 | 0.451 | 0.439 | 0.438 | 515773 |
| sym__mq_p512__gdn2__L2__d256__wd0.1__s2 | False | None | 1.000 | 0.430 | 0.472 | 0.423 | 0.420 | 0.418 | 522048 |
| sym__mq_p512__gdn2__L2__d256__wd0.1__s3 | False | None | 1.000 | 0.434 | 0.475 | 0.423 | 0.418 | 0.415 | 521086 |
| sym__mq_p512__gdn2__L2__d256__wd1.0__s0 | True | 30500 | 0.999 | 0.991 | 0.997 | 0.991 | 0.635 | 0.523 | 517816 |
| sym__mq_p512__gdn2__L2__d256__wd1.0__s1 | True | 37500 | 1.000 | 0.997 | 0.997 | 0.998 | 0.644 | 0.539 | 522496 |
| sym__mq_p512__gdn2__L2__d256__wd1.0__s2 | True | 45000 | 1.000 | 0.992 | 0.991 | 0.991 | 0.663 | 0.511 | 527938 |
| sym__mq_p512__gdn2__L2__d256__wd1.0__s3 | True | 29000 | 1.000 | 0.987 | 0.997 | 0.986 | 0.640 | 0.545 | 522678 |
| sym__mq_p512__gdn2__L2__d512__wd0.1__s0 | False | None | 0.999 | 0.493 | 0.583 | 0.495 | 0.448 | 0.436 | 521240 |
| sym__mq_p512__gdn2__L2__d512__wd0.1__s1 | False | None | 1.000 | 0.463 | 0.471 | 0.470 | 0.457 | 0.454 | 511786 |
| sym__mq_p512__gdn2__L2__d512__wd0.1__s2 | False | None | 1.000 | 0.479 | 0.482 | 0.480 | 0.472 | 0.466 | 518424 |
| sym__mq_p512__gdn2__L2__d512__wd0.1__s3 | False | None | 1.000 | 0.439 | 0.491 | 0.434 | 0.430 | 0.426 | 520919 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s0 | True | 14000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.679 | 0.535 | 525842 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s1 | True | 27500 | 1.000 | 0.998 | 0.998 | 0.997 | 0.698 | 0.562 | 527230 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s2 | True | 13000 | 1.000 | 0.998 | 0.997 | 0.998 | 0.700 | 0.542 | 532656 |
| sym__mq_p512__gdn2__L2__d512__wd1.0__s3 | True | 14000 | 1.000 | 0.998 | 0.999 | 0.998 | 0.701 | 0.565 | 525819 |
| sym__mq_p64__e97-lin__L2__d1024__wd0.1__s0 | True | 1000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.998 | 0.722 | 409083 |
| sym__mq_p64__e97-lin__L2__d1024__wd0.1__s1 | False | None | 1.000 | 0.517 | 0.517 | 0.515 | 0.497 | 0.495 | 412437 |
| sym__mq_p64__e97-lin__L2__d1024__wd0.1__s2 | True | 38500 | 1.000 | 0.998 | 1.000 | 0.999 | 0.941 | 0.665 | 197670 |
| sym__mq_p64__e97-lin__L2__d1024__wd0.1__s3 | True | 4500 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 | 425160 |
| sym__mq_p64__e97-lin__L2__d1024__wd1.0__s0 | True | 4000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.980 | 0.884 | 409183 |
| sym__mq_p64__e97-lin__L2__d1024__wd1.0__s1 | True | 6000 | 0.999 | 0.999 | 0.999 | 0.999 | 0.959 | 0.516 | 414819 |
| sym__mq_p64__e97-lin__L2__d1024__wd1.0__s2 | True | 8500 | 1.000 | 0.999 | 0.999 | 0.998 | 0.892 | 0.659 | 397227 |
| sym__mq_p64__e97-lin__L2__d1024__wd1.0__s3 | True | 5500 | 1.000 | 0.999 | 0.999 | 0.999 | 0.980 | 0.680 | 428684 |
| sym__mq_p64__e97-lin__L2__d256__wd0.1__s0 | False | None | 1.000 | 0.483 | 0.505 | 0.490 | 0.468 | 0.452 | 871481 |
| sym__mq_p64__e97-lin__L2__d256__wd0.1__s1 | False | None | 0.999 | 0.490 | 0.509 | 0.492 | 0.474 | 0.471 | 854982 |
| sym__mq_p64__e97-lin__L2__d256__wd0.1__s2 | False | None | 1.000 | 0.510 | 0.513 | 0.508 | 0.490 | 0.489 | 872014 |
| sym__mq_p64__e97-lin__L2__d256__wd0.1__s3 | True | 39000 | 1.000 | 0.999 | 1.000 | 0.999 | 0.965 | 0.468 | 887019 |
| sym__mq_p64__e97-lin__L2__d256__wd1.0__s0 | False | None | 0.999 | 0.492 | 0.501 | 0.498 | 0.478 | 0.461 | 868844 |
| sym__mq_p64__e97-lin__L2__d256__wd1.0__s1 | True | 15500 | 1.000 | 0.998 | 0.998 | 0.998 | 0.944 | 0.789 | 867013 |
| sym__mq_p64__e97-lin__L2__d256__wd1.0__s2 | True | 23000 | 0.996 | 0.990 | 0.999 | 0.991 | 0.807 | 0.480 | 696484 |
| sym__mq_p64__e97-lin__L2__d256__wd1.0__s3 | True | 5000 | 1.000 | 0.998 | 0.999 | 0.999 | 0.924 | 0.709 | 863027 |
| sym__mq_p64__e97-lin__L2__d512__wd0.1__s0 | False | None | 1.000 | 0.488 | 0.507 | 0.493 | 0.473 | 0.468 | 482267 |
| sym__mq_p64__e97-lin__L2__d512__wd0.1__s1 | True | 18500 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.978 | 894237 |
| sym__mq_p64__e97-lin__L2__d512__wd0.1__s2 | False | None | 1.000 | 0.508 | 0.508 | 0.507 | 0.482 | 0.483 | 886572 |
| sym__mq_p64__e97-lin__L2__d512__wd0.1__s3 | False | None | 1.000 | 0.504 | 0.509 | 0.505 | 0.494 | 0.493 | 873785 |
| sym__mq_p64__e97-lin__L2__d512__wd1.0__s0 | True | 38000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.903 | 0.676 | 490612 |
| sym__mq_p64__e97-lin__L2__d512__wd1.0__s1 | True | 6500 | 1.000 | 1.000 | 0.999 | 1.000 | 0.990 | 0.807 | 871615 |
| sym__mq_p64__e97-lin__L2__d512__wd1.0__s2 | True | 6500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.835 | 0.577 | 899707 |
| sym__mq_p64__e97-lin__L2__d512__wd1.0__s3 | True | 12500 | 0.712 | 0.645 | 0.999 | 0.647 | 0.513 | 0.503 | 763936 |
| sym__mq_p64__e97__L2__d1024__wd0.1__s0 | True | 15500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 395234 |
| sym__mq_p64__e97__L2__d1024__wd0.1__s1 | True | 49999 | 1.000 | 0.909 | 0.909 | 0.913 | 0.756 | 0.545 | 425382 |
| sym__mq_p64__e97__L2__d1024__wd0.1__s2 | True | 5000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 178310 |
| sym__mq_p64__e97__L2__d1024__wd0.1__s3 | True | 43000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 191928 |
| sym__mq_p64__e97__L2__d1024__wd1.0__s0 | True | 2500 | 1.000 | 0.999 | 0.999 | 0.999 | 0.998 | 0.999 | 398822 |
| sym__mq_p64__e97__L2__d1024__wd1.0__s1 | True | 4500 | 1.000 | 0.999 | 1.000 | 1.000 | 0.992 | 0.940 | 425478 |
| sym__mq_p64__e97__L2__d1024__wd1.0__s2 | True | 6000 | 0.990 | 0.984 | 1.000 | 0.983 | 0.971 | 0.970 | 193992 |
| sym__mq_p64__e97__L2__d1024__wd1.0__s3 | True | 1500 | 1.000 | 0.975 | 0.987 | 0.962 | 0.832 | 0.611 | 421348 |
| sym__mq_p64__e97__L2__d256__wd0.1__s0 | False | None | 1.000 | 0.503 | 0.503 | 0.506 | 0.493 | 0.490 | 826824 |
| sym__mq_p64__e97__L2__d256__wd0.1__s1 | True | 1000 | 0.999 | 0.998 | 1.000 | 0.998 | 0.998 | 0.998 | 793193 |
| sym__mq_p64__e97__L2__d256__wd0.1__s2 | False | None | 1.000 | 0.490 | 0.509 | 0.497 | 0.478 | 0.479 | 784830 |
| sym__mq_p64__e97__L2__d256__wd0.1__s3 | True | 46000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 874568 |
| sym__mq_p64__e97__L2__d256__wd1.0__s0 | False | None | 1.000 | 0.492 | 0.504 | 0.493 | 0.480 | 0.476 | 877632 |
| sym__mq_p64__e97__L2__d256__wd1.0__s1 | True | 10000 | 1.000 | 0.998 | 1.000 | 0.999 | 0.998 | 0.997 | 874196 |
| sym__mq_p64__e97__L2__d256__wd1.0__s2 | False | None | 0.999 | 0.494 | 0.505 | 0.501 | 0.484 | 0.480 | 864822 |
| sym__mq_p64__e97__L2__d256__wd1.0__s3 | True | 9000 | 1.000 | 0.999 | 0.998 | 0.998 | 0.988 | 0.991 | 814597 |
| sym__mq_p64__e97__L2__d512__wd0.1__s0 | True | 19000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 878694 |
| sym__mq_p64__e97__L2__d512__wd0.1__s1 | True | 18500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 | 0.821 | 885274 |
| sym__mq_p64__e97__L2__d512__wd0.1__s2 | False | None | 1.000 | 0.571 | 0.572 | 0.583 | 0.508 | 0.510 | 892560 |
| sym__mq_p64__e97__L2__d512__wd0.1__s3 | True | 17500 | 1.000 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 883362 |
| sym__mq_p64__e97__L2__d512__wd1.0__s0 | True | 3000 | 1.000 | 0.999 | 0.999 | 0.999 | 0.995 | 0.991 | 838147 |
| sym__mq_p64__e97__L2__d512__wd1.0__s1 | True | 10500 | 0.997 | 0.996 | 1.000 | 0.996 | 0.989 | 0.987 | 892414 |
| sym__mq_p64__e97__L2__d512__wd1.0__s2 | True | 8500 | 0.999 | 0.995 | 0.998 | 0.994 | 0.973 | 0.956 | 568846 |
| sym__mq_p64__e97__L2__d512__wd1.0__s3 | True | 3000 | 0.997 | 0.994 | 0.999 | 0.994 | 0.984 | 0.948 | 874355 |
| sym__mq_p64__gdn2__L2__d1024__wd0.1__s0 | True | 12000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.710 | 0.542 | 288475 |
| sym__mq_p64__gdn2__L2__d1024__wd0.1__s1 | True | 29500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.851 | 198578 |
| sym__mq_p64__gdn2__L2__d1024__wd0.1__s2 | True | 29000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.775 | 0.521 | 165615 |
| sym__mq_p64__gdn2__L2__d1024__wd0.1__s3 | True | 15500 | 1.000 | 1.000 | 0.999 | 0.999 | 0.685 | 0.543 | 196342 |
| sym__mq_p64__gdn2__L2__d1024__wd1.0__s0 | True | 2000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.878 | 0.567 | 170357 |
| sym__mq_p64__gdn2__L2__d1024__wd1.0__s1 | True | 2000 | 1.000 | 0.998 | 1.000 | 0.998 | 0.693 | 0.530 | 312799 |
| sym__mq_p64__gdn2__L2__d1024__wd1.0__s2 | True | 3500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.686 | 0.545 | 302009 |
| sym__mq_p64__gdn2__L2__d1024__wd1.0__s3 | True | 2500 | 0.999 | 0.998 | 1.000 | 0.999 | 0.729 | 0.578 | 165295 |
| sym__mq_p64__gdn2__L2__d256__wd0.1__s0 | True | 33000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.719 | 0.545 | 524793 |
| sym__mq_p64__gdn2__L2__d256__wd0.1__s1 | True | 33500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.770 | 0.569 | 522230 |
| sym__mq_p64__gdn2__L2__d256__wd0.1__s2 | True | 48000 | 1.000 | 0.990 | 0.990 | 0.992 | 0.716 | 0.550 | 522608 |
| sym__mq_p64__gdn2__L2__d256__wd0.1__s3 | True | 30500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.836 | 0.576 | 512336 |
| sym__mq_p64__gdn2__L2__d256__wd1.0__s0 | True | 7000 | 1.000 | 0.996 | 1.000 | 0.996 | 0.620 | 0.519 | 545717 |
| sym__mq_p64__gdn2__L2__d256__wd1.0__s1 | True | 12000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.659 | 0.547 | 522572 |
| sym__mq_p64__gdn2__L2__d256__wd1.0__s2 | True | 9500 | 1.000 | 0.998 | 1.000 | 0.996 | 0.637 | 0.526 | 529808 |
| sym__mq_p64__gdn2__L2__d256__wd1.0__s3 | True | 7500 | 1.000 | 0.962 | 0.999 | 0.967 | 0.601 | 0.525 | 521172 |
| sym__mq_p64__gdn2__L2__d512__wd0.1__s0 | True | 35500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.763 | 0.557 | 520261 |
| sym__mq_p64__gdn2__L2__d512__wd0.1__s1 | True | 18500 | 1.000 | 1.000 | 0.999 | 1.000 | 0.996 | 0.916 | 524754 |
| sym__mq_p64__gdn2__L2__d512__wd0.1__s2 | True | 23000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.992 | 0.874 | 524310 |
| sym__mq_p64__gdn2__L2__d512__wd0.1__s3 | True | 19000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.840 | 0.583 | 485043 |
| sym__mq_p64__gdn2__L2__d512__wd1.0__s0 | True | 9500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.727 | 0.580 | 497397 |
| sym__mq_p64__gdn2__L2__d512__wd1.0__s1 | True | 5000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.699 | 0.551 | 524862 |
| sym__mq_p64__gdn2__L2__d512__wd1.0__s2 | True | 6000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.695 | 0.544 | 517044 |
| sym__mq_p64__gdn2__L2__d512__wd1.0__s3 | True | 4000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.696 | 0.556 | 517193 |


## VERDICT — does symmetric width let the LINEAR arms extrapolate?

**NO — symmetric width does NOT rescue the linear arms at length; it only buys
train-length memorization. The prediction under test HOLDS: wide-e97 extrapolates,
wide-LINEAR collapses at far T despite width.** 216 runs, 0 failures, bf16+fused
asserted on every run (e97/e97-lin: "no eager" Triton split-edit; gdn2: fla-gdn
fused chunk kernel).

This is the control the predecessor (grok-highp-temporal) never ran. That run
concluded "width closes the gap → capacity, not a temporal class" — but it (a)
widened ONLY the linear arms, never e97, and (b) measured final accuracy at the
TRAIN length (T=128). Both gaps are filled here, and the conclusion flips **in the
extrapolation domain**:

1. **At the train length, width DOES help the linear arms grok — the predecessor
   was right about that.** Wide e97-lin/gdn2 reach high test-acc at T=128 (e.g.
   p256: e97-lin d512 0.955@T128; gdn2 groks ~1.0 at essentially every p×dim).
   So a linear cell has the *capacity* to fit the iterated nonlinear map at the
   trained length. That is NOT in dispute.

2. **But that grok is MEMORIZATION of the train length, not the rule — it
   collapses at far T even at dim 1024 (the cliff, §3).** Among seeds that grokked
   T=128, accuracy at T=4096:
   - p256, d1024: **e97 0.981 (flat) vs e97-lin 0.619 vs gdn2 0.560** — e97 holds,
     both linear arms collapse ~0.36–0.42 below it AT THE WIDEST WIDTH.
   - p512, d1024: **e97 0.786 vs e97-lin 0.404 vs gdn2 0.523**.
   - p64, d1024 (clean n=4 vs n=4): **e97 0.880 vs e97-lin 0.685 vs gdn2 0.555**.
   The e97 curve is length-INVARIANT (p256 d1024: 0.982→0.981 across T=128→4096),
   the linear curves decay monotonically toward baseline. Width raises the linear
   arms' *starting* (train-length) accuracy but does not change the slope: they
   still collapse. Width = memorization headroom, not extrapolation.

3. **The cleanest control isolates the per-step nonlinearity as causal.** e97 vs
   e97-lin is byte-identical code on the SAME fused kernel — ONLY the per-step
   state map differs (tanh vs identity). e97-lin collapses essentially identically
   to the architecturally-different gdn2 (both linear-in-time), while e97 holds.
   So it is the nonlinearity-in-time, not capacity, depth, or the specific linear
   architecture, that produces length-extrapolation of the nested-squaring
   composition.

4. **e97 runs well — in fact it is FASTER than the gdn2 baseline (§4).** Clean,
   isolated 1-GPU throughput: e97 is **1.25× (d1024) to 1.59× (d256)** faster than
   gdn2 at matched geometry, with FEWER params at d512/d1024 (28.9M vs 35.8M at
   d1024). The per-step nonlinearity is throughput-free (e97 ≈ e97-lin, ratio
   0.99–1.02) because the fused bf16 split-edit kernel already runs the sequential
   recurrence; tanh is one extra op per step. The PI's "e97 (fused kernel) runs
   well" is satisfied with margin.

**Honest scope and caveats (this is an EXTRAPOLATION/robustness separation, not a
train-length expressivity-class separation):**

- The advantage e97 holds is **length-extrapolation** (algorithm-vs-memorization),
  NOT train-length sample efficiency. gdn2 is the most *reliable* train-length
  grokker (4/4 seeds nearly everywhere); e97's train-length grok-rate is bimodal
  and not uniformly higher than the linear arms (e.g. p64 d256: e97 2/4 vs e97-lin
  3/4). If you only score final acc at the trained length — as the predecessor did
  — e97 looks equal-or-worse. The separation lives entirely in T > train length.
- gdn2 collapses at far T too (it is linear-in-state); it collapses *hardest* of
  the three (p256 d1024 0.994→0.560). "A linear cell already groks p=256" (the
  predecessor's point #3) is true at T=128 and FALSE at T=4096. The only arm that
  extrapolates is the one with the nonlinear per-step state map.
- Some far-T cells average few grokked seeds (n=1 for several p256/p512 linear
  cells, since the linear arms grok the train length less reliably at high p). The
  pattern is nonetheless robust: every cell with n≥2 on both arms — and the clean
  n=4-vs-n=4 cells at p64 (d512, d1024) — shows the same e97-holds / linear-
  collapses signature. Small-n rows are flagged with their n in §2.
- wd is grokking-critical: wd=0.1 mostly fails to grok at high p (§1, wd=0.1
  block), wd=1.0 is where grokking occurs; the extrapolation analysis (§2–3) uses
  wd=1.0. p=512 is hard — only d1024 groks any seeds, and only ~2/4.

**Bottom line.** The predecessor's verdict was a measurement artifact of scoring at
the train length: width buys the linear arms train-length memorization, which
*looks* like closing the gap. Pushed to far T, symmetric width does NOT let the
linear arms extrapolate — wide-e97 extrapolates and "kicks ass" (length-invariant,
~0.78–0.98 at T=4096 across p), wide-e97-lin and wide-gdn2 collapse toward baseline
despite dim 1024, and e97 does it faster than gdn2. Per-step nonlinearity-in-time
is the causal lever for length-extrapolation of the iterated nonlinear map.




