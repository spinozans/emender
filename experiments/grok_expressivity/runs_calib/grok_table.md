# grok-expressivity — grok-status table

Source: `experiments/grok_expressivity/runs_calib` (18 runs)

Arms: **e97** = E97 split-edit fused Triton, tanh state (nonlinear-in-time); **e97-lin** = same fused kernel, linear state (matched control); **gdn2** = FLA GatedDeltaNet (linear); **e97-ht** = phi-shell split-edit hardtanh.

`grokked` = test-acc reached --grok_acc; `grok_step` = first step it did; `mem_step` = first step train-acc>=--train_sat_acc.


## modular_quadratic  (baseline acc 0.143)

| arm | wd | n_train | seed | params | mem_step | grok_step | grokked | final_train | final_test | best_test | extrap@128 | extrap@256 | extrap@512 | extrap@1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e97 | 0.1 | 128 | 0 | 4,993,088 | 500 | 750 | Y | 1.000 | 0.958 | 0.964 | 0.952 | 0.955 | 0.950 | None |
| e97 | 1.0 | 128 | 0 | 4,993,088 | 500 | 500 | Y | 0.998 | 0.969 | 0.977 | 0.967 | 0.966 | 0.966 | None |
| e97 | 0.1 | 512 | 0 | 4,993,088 | 500 | 500 | Y | 1.000 | 0.992 | 0.991 | 0.990 | 0.990 | 0.989 | None |
| e97 | 1.0 | 512 | 0 | 4,993,088 | 500 | 500 | Y | 0.999 | 0.989 | 0.990 | 0.987 | 0.988 | 0.987 | None |
| e97 | 0.1 | 2048 | 0 | 4,993,088 | 500 | 500 | Y | 0.999 | 0.996 | 0.998 | 0.997 | 0.996 | 0.997 | None |
| e97 | 1.0 | 2048 | 0 | 4,993,088 | 500 | 500 | Y | 0.997 | 0.995 | 0.996 | 0.995 | 0.995 | 0.994 | None |
| e97-lin | 0.1 | 128 | 0 | 4,993,088 | 500 | 1000 | Y | 1.000 | 0.956 | 0.964 | 0.956 | 0.958 | 0.955 | None |
| e97-lin | 1.0 | 128 | 0 | 4,993,088 | 500 | 750 | Y | 0.999 | 0.971 | 0.981 | 0.969 | 0.970 | 0.970 | None |
| e97-lin | 0.1 | 512 | 0 | 4,993,088 | 500 | 500 | Y | 1.000 | 0.992 | 0.992 | 0.990 | 0.990 | 0.990 | None |
| e97-lin | 1.0 | 512 | 0 | 4,993,088 | 500 | 500 | Y | 0.998 | 0.985 | 0.990 | 0.987 | 0.986 | 0.986 | None |
| e97-lin | 0.1 | 2048 | 0 | 4,993,088 | 750 | 500 | Y | 1.000 | 0.998 | 0.998 | 0.998 | 0.997 | 0.998 | None |
| e97-lin | 1.0 | 2048 | 0 | 4,993,088 | 500 | 500 | Y | 0.999 | 0.997 | 0.998 | 0.998 | 0.998 | 0.997 | None |
| gdn2 | 0.1 | 128 | 0 | 4,489,408 | 250 | 10500 | Y | 1.000 | 0.929 | 0.929 | 0.924 | 0.924 | 0.921 | None |
| gdn2 | 1.0 | 128 | 0 | 4,489,408 | 250 | 1500 | Y | 1.000 | 0.977 | 0.977 | 0.976 | 0.975 | 0.972 | None |
| gdn2 | 0.1 | 512 | 0 | 4,489,408 | 250 | 500 | Y | 1.000 | 0.993 | 0.995 | 0.993 | 0.993 | 0.993 | None |
| gdn2 | 1.0 | 512 | 0 | 4,489,408 | 250 | 500 | Y | 1.000 | 0.994 | 0.996 | 0.992 | 0.992 | 0.992 | None |
| gdn2 | 0.1 | 2048 | 0 | 4,489,408 | 500 | 250 | Y | 1.000 | 0.999 | 1.000 | 0.999 | 0.999 | 0.999 | None |
| gdn2 | 1.0 | 2048 | 0 | 4,489,408 | 500 | 250 | Y | 0.998 | 0.998 | 0.999 | 0.997 | 0.998 | 0.997 | None |

## Per-arm rollup (best over wd x seed)

| task | arm | any_grokked | min_grok_step | best_test_acc |
|---|---|---|---|---|
| modular_quadratic | e97 | Y | 500 | 0.998 |
| modular_quadratic | e97-lin | Y | 500 | 0.998 |
| modular_quadratic | gdn2 | Y | 250 | 1.000 |
