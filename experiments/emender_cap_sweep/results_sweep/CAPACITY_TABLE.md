# emender-cap-sweep — capacity boundary (measured)

Dims: [256, 384, 512, 768, 1024]  Arms: ['emender_cma', 'gdn2', 'gdn2typed']  Seeds: 3  (T=512 length-extrap)


## modular_counter — accuracy@T512 (mean over seeds)

| dim | emender_cma | gdn2 | gdn2typed | sep(emender_fix−gdn2typed) |
|---|---|---|---|---|
| 256 | 0.433 | 0.457 | 0.458 | — |
| 384 | 0.403 | 0.530 | 0.529 | — |
| 512 | 0.459 | 0.557 | 0.666 | — |
| 768 | 0.514 | 0.559 | 0.602 | — |
| 1024 | 0.463 | 0.510 | 0.525 | — |

## modular_quadratic — accuracy@T512 (mean over seeds)

| dim | emender_cma | gdn2 | gdn2typed | sep(emender_fix−gdn2typed) |
|---|---|---|---|---|
| 256 | 0.990 | 0.675 | 0.996 | — |
| 384 | 1.000 | 0.995 | 0.995 | — |
| 512 | 1.000 | 0.996 | 0.995 | — |
| 768 | 1.000 | 0.995 | 0.983 | — |
| 1024 | 1.000 | 0.998 | 0.889 | — |

## s5_permutation — accuracy@T512 (mean over seeds)

| dim | emender_cma | gdn2 | gdn2typed | sep(emender_fix−gdn2typed) |
|---|---|---|---|---|
| 256 | 0.774 | 0.999 | 0.998 | — |
| 384 | 0.565 | 0.935 | 0.995 | — |
| 512 | 0.315 | 0.998 | 0.996 | — |
| 768 | 0.541 | 0.999 | 0.989 | — |
| 1024 | 0.312 | 0.893 | 0.990 | — |

## Capacity boundary (smallest dim where emender−gdn2typed ≤ 0.05 at T512)

- modular_counter__emender_cma: closed at dim 256
- modular_quadratic__emender_cma: closed at dim 256
- s5_permutation__emender_cma: closed at dim 256
