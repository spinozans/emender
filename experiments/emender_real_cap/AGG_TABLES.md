## Expressivity — small (dim256 nh32, 2/32 & 4/32 = documented regime)
Mean accuracy over seeds (train T=128, eval-length extrapolation). `sep` = emender − gdn2.

### modular_quadratic
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.714 | 0.693 | 0.676 |
| gdn2typed | 1.000 | 1.000 | 0.997 |
| emender4 | 1.000 | 1.000 | 0.990 |
| emender8 | 1.000 | 1.000 | 0.994 |

*separation (arm − gdn2):* `emender4@128=+0.286`, `emender4@256=+0.307`, `emender4@512=+0.314`, `emender8@128=+0.286`, `emender8@256=+0.307`, `emender8@512=+0.317`

### iterated_nonlinear_map
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.941 | 0.934 | 0.934 |
| gdn2typed | 0.959 | 0.954 | 0.954 |
| emender4 | 0.959 | 0.952 | 0.953 |
| emender8 | 0.961 | 0.955 | 0.955 |

*separation (arm − gdn2):* `emender4@128=+0.018`, `emender4@256=+0.018`, `emender4@512=+0.019`, `emender8@128=+0.019`, `emender8@256=+0.021`, `emender8@512=+0.021`

### s5_permutation
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 1.000 | 1.000 | 0.999 |
| gdn2typed | 1.000 | 1.000 | 0.996 |
| emender4 | 1.000 | 0.999 | 0.919 |

*separation (arm − gdn2):* `emender4@128=+0.000`, `emender4@256=-0.001`, `emender4@512=-0.080`

### modular_counter
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.701 | 0.496 | 0.348 |
| gdn2typed | 0.872 | 0.676 | 0.456 |
| emender4 | 0.943 | 0.780 | 0.508 |

*separation (arm − gdn2):* `emender4@128=+0.242`, `emender4@256=+0.285`, `emender4@512=+0.160`

### mqar_recall
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.995 | 0.969 | 0.826 |
| gdn2typed | 0.996 | 0.947 | 0.734 |
| emender4 | 0.993 | 0.952 | 0.746 |

*separation (arm − gdn2):* `emender4@128=-0.002`, `emender4@256=-0.018`, `emender4@512=-0.080`

## Expressivity — large (dim512 nh64, 4/64 & 8/64 = literal fractions)
Mean accuracy over seeds (train T=128, eval-length extrapolation). `sep` = emender − gdn2.

### modular_quadratic
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 1.000 | 1.000 | 0.996 |
| gdn2typed | 1.000 | 1.000 | 0.996 |
| emender4 | 1.000 | 1.000 | 0.999 |
| emender8 | 1.000 | 1.000 | 0.997 |
| shell4 | 1.000 | 0.999 | 0.980 |

*separation (arm − gdn2):* `emender4@128=+0.000`, `emender4@256=+0.000`, `emender4@512=+0.003`, `emender8@128=+0.000`, `emender8@256=+0.000`, `emender8@512=+0.001`

### iterated_nonlinear_map
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.957 | 0.952 | 0.951 |
| gdn2typed | 0.969 | 0.965 | 0.965 |
| emender4 | 0.969 | 0.963 | 0.963 |
| emender8 | 0.969 | 0.964 | 0.965 |
| shell4 | 0.967 | 0.962 | 0.962 |

*separation (arm − gdn2):* `emender4@128=+0.012`, `emender4@256=+0.012`, `emender4@512=+0.012`, `emender8@128=+0.011`, `emender8@256=+0.012`, `emender8@512=+0.013`

### s5_permutation
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 1.000 | 1.000 | 0.914 |
| gdn2typed | 1.000 | 1.000 | 0.994 |
| emender4 | 1.000 | 1.000 | 0.962 |
| emender8 | 1.000 | 1.000 | 0.941 |
| shell4 | 1.000 | 1.000 | 0.998 |

*separation (arm − gdn2):* `emender4@128=+0.000`, `emender4@256=+0.000`, `emender4@512=+0.048`, `emender8@128=+0.000`, `emender8@256=+0.000`, `emender8@512=+0.027`

### modular_counter
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.985 | 0.863 | 0.585 |
| gdn2typed | 0.984 | 0.898 | 0.630 |
| emender4 | 0.928 | 0.736 | 0.483 |
| emender8 | 0.981 | 0.833 | 0.546 |
| shell4 | 0.738 | 0.612 | 0.414 |

*separation (arm − gdn2):* `emender4@128=-0.057`, `emender4@256=-0.127`, `emender4@512=-0.102`, `emender8@128=-0.004`, `emender8@256=-0.030`, `emender8@512=-0.039`

### mqar_recall
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.998 | 0.978 | 0.851 |
| gdn2typed | 0.996 | 0.958 | 0.766 |
| emender4 | 0.996 | 0.954 | 0.737 |
| emender8 | 0.997 | 0.959 | 0.772 |
| shell4 | 0.994 | 0.955 | 0.752 |

*separation (arm − gdn2):* `emender4@128=-0.002`, `emender4@256=-0.024`, `emender4@512=-0.114`, `emender8@128=-0.001`, `emender8@256=-0.018`, `emender8@512=-0.079`

## Throughput @ 1.3B head shape (fwd+bwd bf16, ratio vs GDN-2)

| config | tok/s | ratio |
|---|---|---|
| gdn_pure (GDN-2 ceiling) | 12256 | 1.000 |
| emender4 e97d-tanh ov=0 | 8879 | 0.724 |
| emender4 e97d-tanh ov=1 | 9232 | 0.753 |
| emender8 e97d-tanh ov=0 | 8628 | 0.704 |
| emender8 e97d-tanh ov=1 | 8884 | 0.725 |
| shell4 ov=1 | 6754 | 0.551 |
| shell8 ov=1 | 8237 | 0.672 |
