# High-p Temporal-Composition Separation — Results

> **CORRECTION — this run's "NO-GO" verdict is SUPERSEDED by `experiments/grok_symmetric_width/`.**
> The "width closes the gap → not a class separation" conclusion was a **capacity confound**: the
> width-control was applied **only to the linear arms** (e97-lin, gdn2), never to e97, and
> extrapolation was measured only to T=1024. A high-capacity linear-state+MLP model can **memorize
> any finite instance**, so "more width → groks the finite test set at train length" is capacity
> buying memorization, not the linear model acquiring the capability. The discriminator capacity
> cannot fake — **length-extrapolation** — was already present here (e97 holds flat while gdn2
> collapses 0.997→0.685 @T1024) and was read past. The symmetric-width follow-up (width on ALL
> arms incl. e97, extrap to T=4096, 216 runs) confirms: **e97 holds test-acc flat T=128→4096 at
> every width; the linear arms memorize train-length and collapse toward baseline even at d1024
> (gdn2 0.994→0.560, e97-lin 0.999→0.685 @T4096); width does NOT let them extrapolate. The
> temporal class separation is REAL, and e97 runs 1.25–1.59× faster than gdn2.** See
> `experiments/grok_symmetric_width/RESULTS.md`.

_58 runs aggregated. modular_quadratic x_t=(x_{t-1}^2+c_t) mod p; per-position supervision; bf16+fused._


## 1. Final held-out accuracy (L=2, d256, wd=1.0, seed-POOLED over {0,1,2,3})

baseline acc = 1/p. acc = mean over seeds of max(final_test,best_test); gr = grok-rate (frac of seeds reaching test>=0.9). n = #seeds.

| p | baseline | n | e97 (gr) | e97-lin (gr) | gdn2 (gr) | sep mean | sep grok-rate |
|---|---|---|---|---|---|---|---|
| 32 | 0.0312 | 2 | 1.000 (1.00) | 1.000 (1.00) | 1.000 (1.00) | -0.000 | 0.00 |
| 64 | 0.0156 | 2 | 0.752 (0.50) | 0.749 (0.50) | 1.000 (1.00) | 0.003 | 0.00 |
| 128 | 0.0078 | 4 | 0.860 (0.75) | 0.737 (0.50) | 1.000 (1.00) | 0.122 | 0.25 |
| 256 | 0.0039 | 4 | 0.943 (1.00) | 0.579 (0.25) | 1.000 (1.00) | 0.363 | 0.75 |

## 2. Separation vs eval-T (length-extrapolation, MAIN L=2 wd=1.0, mean seeds)

sep(p,T) = acc_e97(T) - acc_e97lin(T) on FRESH sequences of length T.


**p=32**

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97 | 0.995 | 0.994 | 0.994 | 0.989 |
| e97-lin | 0.999 | 0.995 | 0.972 | 0.873 |
| gdn2 | 0.997 | 0.949 | 0.762 | 0.648 |
| **sep** | -0.004 | -0.002 | 0.022 | 0.116 |

**p=64**

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97 | 0.748 | 0.743 | 0.741 | 0.737 |
| e97-lin | 0.747 | 0.742 | 0.734 | 0.719 |
| gdn2 | 0.998 | 0.946 | 0.768 | 0.638 |
| **sep** | 0.001 | 0.001 | 0.007 | 0.018 |

**p=128**

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97 | 0.859 | 0.855 | 0.853 | 0.845 |
| e97-lin | 0.735 | 0.729 | 0.721 | 0.699 |
| gdn2 | 0.994 | 0.949 | 0.796 | 0.645 |
| **sep** | 0.124 | 0.127 | 0.131 | 0.147 |

**p=256**

| arm | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| e97 | 0.924 | 0.921 | 0.917 | 0.916 |
| e97-lin | 0.568 | 0.556 | 0.543 | 0.528 |
| gdn2 | 0.997 | 0.962 | 0.833 | 0.685 |
| **sep** | 0.356 | 0.365 | 0.375 | 0.388 |

## 3. Width control — does more width close the gap?

sep here = (e97 @ d256, seed-pooled) - (linear arm @ dim). If sep stays HIGH as dim grows -> depth/temporal, not capacity. If width rescues the linear arm -> capacity.


**p=128** (L=2, wd=1.0, width seed0)

| dim | e97 (d256 ref) | e97-lin | gdn2 | sep=e97-e97lin |
|---|---|---|---|---|
| 256 | 0.860 | 0.737 | 1.000 | 0.122 |
| 512 | (pooled d256) | 0.967 | 1.000 | -0.107 |
| 1024 | (pooled d256) | 0.996 | 1.000 | -0.136 |

**p=256** (L=2, wd=1.0, width seed0)

| dim | e97 (d256 ref) | e97-lin | gdn2 | sep=e97-e97lin |
|---|---|---|---|---|
| 256 | 0.943 | 0.579 | 1.000 | 0.363 |
| 512 | (pooled d256) | 0.961 | 0.999 | -0.018 |
| 1024 | (pooled d256) | 0.969 | 1.000 | -0.026 |

## 4. Depth control — does the gap shrink as L grows? (wd=1.0, seed0)

| p | L | e97 | e97-lin | gdn2 | sep |
|---|---|---|---|---|---|
| 64 | 2 | 0.505 | 0.499 | 1.000 | 0.006 |
| 64 | 4 | 1.000 | 0.999 | 1.000 | 0.000 |
| 256 | 2 | 0.955 | 0.475 | 0.999 | 0.480 |
| 256 | 4 | 0.989 | 0.991 | 0.999 | -0.003 |

## 5. Weight-decay sweep (p=64, L=2, seed0)

| wd | e97 | e97-lin | sep |
|---|---|---|---|
| 0.01 | 0.502 | 0.499 | 0.003 |
| 0.1 | 0.506 | 0.501 | 0.004 |
| 0.3 | 0.525 | 0.505 | 0.021 |
| 1.0 | 0.505 | 0.499 | 0.006 |

## 6. All runs (raw)

| label | grok | gstep | train | test | best |
|---|---|---|---|---|---|
| confirm__mq_p128__e97-lin__L2__d256__wd1.0__s2 | False | None | 0.999 | 0.481 | 0.483 |
| confirm__mq_p128__e97-lin__L2__d256__wd1.0__s3 | True | 15000 | 1.000 | 0.991 | 0.991 |
| confirm__mq_p128__e97__L2__d256__wd1.0__s2 | True | 10000 | 1.000 | 0.992 | 0.992 |
| confirm__mq_p128__e97__L2__d256__wd1.0__s3 | True | 4000 | 1.000 | 0.997 | 0.998 |
| confirm__mq_p128__gdn2__L2__d256__wd1.0__s2 | True | 23000 | 1.000 | 0.994 | 1.000 |
| confirm__mq_p128__gdn2__L2__d256__wd1.0__s3 | True | 18500 | 1.000 | 0.996 | 1.000 |
| confirm__mq_p256__e97-lin__L2__d1024__wd1.0__s0 | True | 8500 | 1.000 | 0.949 | 0.969 |
| confirm__mq_p256__e97-lin__L2__d256__wd1.0__s2 | True | 34000 | 1.000 | 0.924 | 0.930 |
| confirm__mq_p256__e97-lin__L2__d256__wd1.0__s3 | False | None | 1.000 | 0.450 | 0.458 |
| confirm__mq_p256__e97-lin__L2__d512__wd1.0__s0 | True | 14500 | 0.999 | 0.940 | 0.961 |
| confirm__mq_p256__e97__L2__d1024__wd1.0__s0 | True | 28500 | 1.000 | 0.996 | 0.996 |
| confirm__mq_p256__e97__L2__d256__wd1.0__s2 | True | 11500 | 0.997 | 0.905 | 0.939 |
| confirm__mq_p256__e97__L2__d256__wd1.0__s3 | True | 14500 | 0.999 | 0.923 | 0.940 |
| confirm__mq_p256__e97__L2__d512__wd1.0__s0 | True | 33500 | 1.000 | 0.924 | 0.945 |
| confirm__mq_p256__gdn2__L2__d1024__wd1.0__s0 | True | 5000 | 1.000 | 1.000 | 0.999 |
| confirm__mq_p256__gdn2__L2__d256__wd1.0__s2 | True | 11500 | 1.000 | 1.000 | 1.000 |
| confirm__mq_p256__gdn2__L2__d256__wd1.0__s3 | True | 16000 | 1.000 | 0.998 | 1.000 |
| confirm__mq_p256__gdn2__L2__d512__wd1.0__s0 | True | 14000 | 1.000 | 0.987 | 0.999 |
| ldepth__mq_p256__e97-lin__L4__d256__wd1.0__s0 | True | 4000 | 0.998 | 0.971 | 0.991 |
| ldepth__mq_p256__e97__L4__d256__wd1.0__s0 | True | 6000 | 1.000 | 0.982 | 0.989 |
| ldepth__mq_p256__gdn2__L4__d256__wd1.0__s0 | True | 4500 | 1.000 | 0.997 | 0.999 |
| ldepth__mq_p64__e97-lin__L4__d256__wd1.0__s0 | True | 2000 | 1.000 | 0.999 | 0.999 |
| ldepth__mq_p64__e97__L4__d256__wd1.0__s0 | True | 2500 | 1.000 | 1.000 | 1.000 |
| ldepth__mq_p64__gdn2__L4__d256__wd1.0__s0 | True | 2500 | 1.000 | 1.000 | 1.000 |
| main__mq_p128__e97-lin__L2__d256__wd1.0__s0 | True | 3500 | 1.000 | 0.986 | 0.993 |
| main__mq_p128__e97-lin__L2__d256__wd1.0__s1 | False | None | 1.000 | 0.480 | 0.482 |
| main__mq_p128__e97__L2__d256__wd1.0__s0 | False | None | 1.000 | 0.480 | 0.486 |
| main__mq_p128__e97__L2__d256__wd1.0__s1 | True | 45500 | 1.000 | 0.964 | 0.964 |
| main__mq_p128__gdn2__L2__d256__wd1.0__s0 | True | 29500 | 1.000 | 1.000 | 0.999 |
| main__mq_p128__gdn2__L2__d256__wd1.0__s1 | True | 13000 | 0.995 | 0.983 | 1.000 |
| main__mq_p256__e97-lin__L2__d256__wd1.0__s0 | False | None | 1.000 | 0.451 | 0.475 |
| main__mq_p256__e97-lin__L2__d256__wd1.0__s1 | False | None | 1.000 | 0.445 | 0.453 |
| main__mq_p256__e97__L2__d256__wd1.0__s0 | True | 6500 | 0.999 | 0.937 | 0.955 |
| main__mq_p256__e97__L2__d256__wd1.0__s1 | True | 14500 | 1.000 | 0.935 | 0.937 |
| main__mq_p256__gdn2__L2__d256__wd1.0__s0 | True | 20500 | 1.000 | 0.993 | 0.999 |
| main__mq_p256__gdn2__L2__d256__wd1.0__s1 | True | 27000 | 1.000 | 1.000 | 0.999 |
| main__mq_p32__e97-lin__L2__d256__wd1.0__s0 | True | 46000 | 1.000 | 1.000 | 1.000 |
| main__mq_p32__e97-lin__L2__d256__wd1.0__s1 | True | 9000 | 1.000 | 0.999 | 1.000 |
| main__mq_p32__e97__L2__d256__wd1.0__s0 | True | 6000 | 0.993 | 0.992 | 0.999 |
| main__mq_p32__e97__L2__d256__wd1.0__s1 | True | 6500 | 1.000 | 1.000 | 1.000 |
| main__mq_p32__gdn2__L2__d256__wd1.0__s0 | True | 8500 | 1.000 | 1.000 | 1.000 |
| main__mq_p32__gdn2__L2__d256__wd1.0__s1 | True | 11500 | 1.000 | 0.994 | 1.000 |
| main__mq_p64__e97-lin__L2__d256__wd1.0__s0 | False | None | 0.999 | 0.493 | 0.499 |
| main__mq_p64__e97-lin__L2__d256__wd1.0__s1 | True | 15000 | 1.000 | 0.999 | 0.999 |
| main__mq_p64__e97__L2__d256__wd1.0__s0 | False | None | 1.000 | 0.493 | 0.505 |
| main__mq_p64__e97__L2__d256__wd1.0__s1 | True | 10000 | 1.000 | 0.999 | 0.999 |
| main__mq_p64__gdn2__L2__d256__wd1.0__s0 | True | 7000 | 1.000 | 0.996 | 1.000 |
| main__mq_p64__gdn2__L2__d256__wd1.0__s1 | True | 9500 | 1.000 | 0.999 | 1.000 |
| wdsweep__mq_p64__e97-lin__L2__d256__wd0.01__s0 | False | None | 1.000 | 0.497 | 0.499 |
| wdsweep__mq_p64__e97-lin__L2__d256__wd0.1__s0 | False | None | 1.000 | 0.485 | 0.501 |
| wdsweep__mq_p64__e97-lin__L2__d256__wd0.3__s0 | False | None | 1.000 | 0.488 | 0.505 |
| wdsweep__mq_p64__e97__L2__d256__wd0.01__s0 | False | None | 1.000 | 0.501 | 0.502 |
| wdsweep__mq_p64__e97__L2__d256__wd0.1__s0 | False | None | 1.000 | 0.502 | 0.506 |
| wdsweep__mq_p64__e97__L2__d256__wd0.3__s0 | False | None | 1.000 | 0.525 | 0.525 |
| width__mq_p128__e97-lin__L2__d1024__wd1.0__s0 | True | 16000 | 1.000 | 0.995 | 0.996 |
| width__mq_p128__e97-lin__L2__d512__wd1.0__s0 | True | 35000 | 0.999 | 0.961 | 0.967 |
| width__mq_p128__gdn2__L2__d1024__wd1.0__s0 | True | 3500 | 1.000 | 1.000 | 0.999 |
| width__mq_p128__gdn2__L2__d512__wd1.0__s0 | True | 4500 | 1.000 | 1.000 | 1.000 |

## 7. VERDICT — does per-step nonlinearity buy a temporal class?

**NO. The claim does NOT hold.** The large e97 vs e97-lin gap at p=256/L=2 is a CAPACITY + grokking-reliability effect, not an O(T)-vs-O(L) temporal-composition class separation. Three controls each dissolve it:

1. **Width closes it (the deciding control).** At p=256 the narrow e97-lin (d256) groks 1/4 seeds (acc 0.579), but widening the SAME linear cell to d512/d1024 makes it grok reliably (0.961 / 0.969); sep collapses +0.363 -> -0.018 / -0.026. Same at p=128 (sep +0.122 -> -0.107 / -0.136). A capacity-starved narrow cell, not an unreachable capability.

2. **Depth closes it.** At p=256 the gap is +0.480 at L=2 but -0.003 at L=4 (e97-lin groks 0.991). Both width and depth supply the missing realizable composition budget -> the deficit is budget, not per-step nonlinearity.

3. **A linear cell already solves the task.** gdn2 (linear-state gated-delta) groks 4/4 seeds at EVERY p incl. 256 at d256 (acc ~1.0). If per-step nonlinearity were required for the nested-squaring composition, no linear recurrence could grok p=256 -- but gdn2 does, reliably. So linearity-in-time is not the barrier; e97-lin is simply a weaker/narrower grokker than both e97 and gdn2.

**What per-step nonlinearity DOES buy (secondary, real but modest):** at fixed small width (d256) e97 is a more RELIABLE grokker than e97-lin (p=256: 4/4 vs 1/4; p=128: 3/4 vs 2/4) -- a sample/parameter-efficiency edge that width or depth erases -- plus a small length-extrapolation robustness edge at p=32 (T=1024: e97 0.989 vs e97-lin 0.873 vs gdn2 0.648). These are efficiency/robustness signals, NOT a capability class linear cells lack.

**On the pre-registered signature:** sep(e97-e97lin) DOES grow with p in the narrow d256 regime (p32 ~0, p64 ~0, p128 +0.12, p256 +0.36) and is roughly flat-to-slightly-growing in T -- which in isolation looks like the temporal thesis. But the width-control (the task's own tie-breaker: 'if more width does NOT close the gap it is depth/temporal, not capacity') is decisive and NEGATIVE: more width DOES close it. Per the task's stated criterion, this is the 'stays ~0 / claim dead' branch once capacity is controlled. bf16+fused asserted on all 58 runs (40 e97-type 'no eager', 18 gdn2 fused; 0 eager fallbacks).

