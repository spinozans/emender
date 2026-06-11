# emender-real-cap — RESULTS

**The REAL Emender, at matched precision, with the mixture DISCOVERED by CMA-ES (not hand-picked).**

## 0. Precision finding (flagged, verified on real GPU)

The fused E97 split-edit Triton kernel (the Emender's nonlinear EMENDMENT head, `e97_delta` with per-step bounded `tanh`) is **bf16-ONLY**: pure fp16 raises `RuntimeError` (it refuses rather than silently running the eager T-scan), and fp16-autocast silently casts the emendment head to bf16 (a hidden dtype mismatch — the very thing the task forbids). So **fp16-uniform-fused is impossible** for the Emender. Per the task's escape clause, all arms run **bf16 UNIFORM** (half precision, uniform, fused, no fp32, no mismatch) — the matched-precision fix to the opt-1p3b fp32 strawman (which ran the Emender in fp32 = 4.3x token-starved vs bf16 controls).

## 1. CMA-ES search — the FOUND mixture (core deliverable)

- **Search space**: CMA-ES over [a_delta,a_track] -> {gdn2_recall sea, e97_delta, e97_track}
- **Fitness**: held-out BPB (real-Pile slice), bf16 UNIFORM, fused asserted
- **Param/FLOP-locked**: target 40000000 params, dim derived per candidate
- **popsize/gens**: 5/4; proxy {'depth': 8, 'n_heads': 32, 'n_state': 32, 'chunk': 512, 'max_tokens': 6000000}

**FOUND mixture** = `cma_g3_e4`: held-out bpb **2.37287**, e97_delta frac **0.0625**, e97_track frac **0.0312**, total nonlinear fraction **0.0938**.

**VERDICT (loss-CMA)**: EMENDER (CMA kept a nonlinear emendment fraction on loss)

Leaderboard (lowest held-out bpb first):

| rank | name | bpb | f_delta | f_track | counts |
|---|---|---|---|---|---|
| 1 | cma_g3_e4 | 2.37287 | 0.062 | 0.031 | {'gdn2_recall': 29, 'e97_track': 1, 'e97_delta': 2} |
| 2 | cma_g3_e3 | 2.37614 | 0.156 | 0.000 | {'gdn2_recall': 27, 'e97_delta': 5} |
| 3 | cma_g0_e2 | 2.37801 | 0.031 | 0.031 | {'gdn2_recall': 30, 'e97_track': 1, 'e97_delta': 1} |
| 4 | cma_g2_e2 | 2.38269 | 0.094 | 0.000 | {'gdn2_recall': 29, 'e97_delta': 3} |
| 5 | anchor_emender_seed | 2.38316 | 0.062 | 0.031 | {'gdn2_recall': 29, 'e97_track': 1, 'e97_delta': 2} |
| 6 | cma_g2_e0 | 2.38596 | 0.031 | 0.031 | {'gdn2_recall': 30, 'e97_track': 1, 'e97_delta': 1} |
| 7 | cma_g2_e3 | 2.38784 | 0.125 | 0.031 | {'gdn2_recall': 27, 'e97_track': 1, 'e97_delta': 4} |
| 8 | cma_g0_e0 | 2.38971 | 0.188 | 0.000 | {'gdn2_recall': 26, 'e97_delta': 6} |

CMA generation trajectory (gen-best):

| gen | best bpb | f_delta | f_track |
|---|---|---|---|
| 0 | 2.37801 | 0.031 | 0.031 |
| 1 | 2.39485 | 0.062 | 0.000 |
| 2 | 2.38269 | 0.094 | 0.000 |
| 3 | 2.37287 | 0.062 | 0.031 |

## 2. Expressivity battery — separation vs BOTH controls (3 seeds, eval-length extrap)

`sepF` = emender − fla-gdn (incumbent); `sepT` = emender − gdn2typed (CLEAN same-path control = isolates the emendment HEADS).

### modular_quadratic
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 1.000 | 1.000 | 0.996 |
| gdn2typed | 1.000 | 1.000 | 0.996 |
| emender4 | 1.000 | 1.000 | 0.999 |
| emender8 | 1.000 | 1.000 | 0.997 |
| shell4 | 1.000 | 0.999 | 0.980 |

*`emender4 sepF@512=+0.003`, `emender4 sepT@512=+0.003`, `emender8 sepF@512=+0.001`, `emender8 sepT@512=+0.001`*

### iterated_nonlinear_map
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.957 | 0.952 | 0.951 |
| gdn2typed | 0.969 | 0.965 | 0.965 |
| emender4 | 0.969 | 0.963 | 0.963 |
| emender8 | 0.969 | 0.964 | 0.965 |
| shell4 | 0.967 | 0.962 | 0.962 |

*`emender4 sepF@512=+0.012`, `emender4 sepT@512=-0.002`, `emender8 sepF@512=+0.013`, `emender8 sepT@512=+0.000`*

### s5_permutation
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 1.000 | 1.000 | 0.914 |
| gdn2typed | 1.000 | 1.000 | 0.994 |
| emender4 | 1.000 | 1.000 | 0.962 |
| emender8 | 1.000 | 1.000 | 0.941 |
| shell4 | 1.000 | 1.000 | 0.998 |

*`emender4 sepF@512=+0.048`, `emender4 sepT@512=-0.031`, `emender8 sepF@512=+0.027`, `emender8 sepT@512=-0.053`*

### modular_counter
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.985 | 0.863 | 0.585 |
| gdn2typed | 0.984 | 0.898 | 0.630 |
| emender4 | 0.928 | 0.736 | 0.483 |
| emender8 | 0.981 | 0.833 | 0.546 |
| shell4 | 0.738 | 0.612 | 0.414 |

*`emender4 sepF@512=-0.102`, `emender4 sepT@512=-0.147`, `emender8 sepF@512=-0.039`, `emender8 sepT@512=-0.083`*

### mqar_recall
| arm | T=128 | T=256 | T=512 (cliff) |
|---|---|---|---|
| gdn2 | 0.998 | 0.978 | 0.851 |
| gdn2typed | 0.996 | 0.958 | 0.766 |
| emender4 | 0.996 | 0.954 | 0.737 |
| emender8 | 0.997 | 0.959 | 0.772 |
| shell4 | 0.994 | 0.955 | 0.752 |

*`emender4 sepF@512=-0.114`, `emender4 sepT@512=-0.029`, `emender8 sepF@512=-0.079`, `emender8 sepT@512=+0.006`*

## 3. Convergent-loss tie — held-out BPB (real-Pile, bf16 uniform, fused)

| budget | gdn2 bpb | emender(found) bpb | Δ (emender−gdn2) | gdn2 tok | emender tok |
|---|---|---|---|---|---|
| token-matched | 3.13158 | 3.33743 | +0.20585 | 12001280 | 12001280 |
| wall-matched | 3.43830 | 3.31637 | -0.12193 | 22126592 | 12281856 |

## 4. Throughput @ 1.3B head shape (fwd+bwd bf16, ratio vs GDN-2)

| config | tok/s | ratio |
|---|---|---|
| gdn_pure (GDN-2 ceiling) | 12256 | 1.000 |
| emender4 e97d-tanh ov=0 | 8879 | 0.724 |
| emender4 e97d-tanh ov=1 | 9232 | 0.753 |
| emender8 e97d-tanh ov=0 | 8628 | 0.704 |
| emender8 e97d-tanh ov=1 | 8884 | 0.725 |
| shell4 ov=1 | 6754 | 0.551 |
| shell8 ov=1 | 8237 | 0.672 |

*(base shape: dim=2240, {'depth': 18, 'n_heads': 64, 'n_state': 32, 'expansion': 1.0, 'mlp_ratio': 2.6944444444444446})*
