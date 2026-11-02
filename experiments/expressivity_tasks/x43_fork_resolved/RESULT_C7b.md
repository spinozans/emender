# X43 C7b — constrained routing residual training

**Primary Emender fixed-blend fails all three dense comparisons; secondary transformer fixed-blend wins all three. Gate-floor training loses for both heads.** This is an architecture-dependent result, so it does not settle every training integration at inference or constitute a fourth uniformly negative training-side receipt.

All four sequential 180-second arms completed. Fixed-blend training has no learned routing: covered positions receive .5 true-target CE plus .5 exact-count teacher CE, and empty support receives pure true-target CE. The secondary mixture gate has neural probability weight at least .3. The latter prevents probability-weight collapse, but does not guarantee a .3 neural gradient responsibility.

## Decisive comparison: fixed minus reused dense

Signs are judged separately; negative favors fixed. No final-test selection of training arms or calibration recipe.

| Head | Total delta | Fork delta | Empty-support/OOD delta |
|---|---:|---:|---:|
| emender | +0.038320 | +0.051740 | +0.079243 |
| transformer | -0.091439 | -0.122941 | -0.178565 |

## Same C3 inference, all four recipes

Natural-log nats/byte. 55,769 calibration targets; 55,770 final targets, 32,415 forks. Calibrate on covered calibration forks; freeze coefficients before final loss; walk covered forced positions; abstain to the neural head at every empty-support position. Dense and C7a-starved predictions are reused without training.

| Head | Training | Fused total | Fused fork | OOD all (9,049) | OOD fork (2,709) | OOD forced (6,340) |
|---|---|---:|---:|---:|---:|---:|
| emender | dense | 1.673012 | 2.424886 | 2.444723 | 2.739430 | 2.318798 |
| emender | starved | 3.718402 | 4.902387 | 7.269978 | 6.394095 | 7.644231 |
| emender | fixed | 1.711331 | 2.476626 | 2.523966 | 2.834355 | 2.391341 |
| emender | floor | 4.552704 | 5.956247 | 9.051871 | 7.780673 | 9.595037 |
| transformer | dense | 1.746996 | 2.558332 | 2.466627 | 2.886269 | 2.287320 |
| transformer | starved | 14.647364 | 19.057854 | 30.744151 | 29.191829 | 31.407439 |
| transformer | fixed | 1.655556 | 2.435391 | 2.288062 | 2.701182 | 2.111541 |
| transformer | floor | 5.281165 | 7.443949 | 8.962798 | 10.287702 | 8.396684 |

Empty support is the first-class no-index regime: fused probability is exactly the neural probability. It is distinct from nonempty support excluding the target (17,066 final forks), retained in the JSON zero_target stratum. Full neural and fused strata, including every empty-support target, are retained.

## Calibration trust

Weights below are standardized-feature coefficients for INDEX weight sigmoid(intercept + confidence + log1p(context length) + log1p(branch count)); mean neural weights per interval-count, support and context bin are retained in results.json. Empty support is overridden to neural weight 1; forced weights are extrapolated before walk.

| Head/recipe | Index-gate coefficients |
|---|---|
| emender_dense | -1.542144, 0.351627, -0.384106, 1.141230 |
| emender_starved | -1.023763, 0.665450, -0.292712, 1.287800 |
| emender_fixed | -1.383980, 0.351574, -0.407724, 1.161982 |
| emender_floor | -1.098163, 0.601966, -0.284082, 1.236041 |
| transformer_dense | -2.820065, 0.716283, -0.375837, 1.247525 |
| transformer_starved | -0.747822, 0.042729, -0.437246, 1.164342 |
| transformer_fixed | -3.835806, 0.879421, -0.500204, 1.375103 |
| transformer_floor | -1.420192, 0.484722, -0.246331, 1.179076 |

## Execution and checks

Registered source and decisive comparison in commit 00f39a22 before any training; full commit and dependency hashes in dependencies.json. Each arm: seed 4301, batch 16, context 128, AdamW .0003/.01, clipping 1, BF16 with float32 normalized logits; same corpus/split/backbone parameter targeting as C7a. Teacher reconstructed from exact train-prefix occurrence counts and checked at every target against C6 q. Training setup, index reconstruction, initialization and inference are outside each loop budget; compilation and synchronization inside are charged.

| Arm | Configured parameters | Steps | Tokens | Train seconds | Inference seconds |
|---|---:|---:|---:|---:|---:|
| emender_fixed | 2067140 | 2708 | 5545984 | 180.011287 | 1.263607 |
| emender_floor | 2067144 | 6020 | 12328960 | 180.020069 | 1.271734 |
| transformer_fixed | 2067188 | 27283 | 55875584 | 180.006005 | 0.232080 |
| transformer_floor | 2067192 | 26956 | 55205888 | 180.004054 | 0.260791 |

GPU broker leased physical card 2 only (2/4 allowlist). No card 7 GPU work, Slurm, Frontier, WorksGood services, subagents, checkpoints or .pt files. Only /home/erikg/emender/.venv/bin/python used. Six CPU tests cover actual fixed routing without gate parameters, the gate probability floor and empty-support override, surviving fixed true-target gradient, and fail-closed ledger checks. Independent CPU validator replays all prediction losses and denominators, hashes, protocol/source registration, budgets and historical dense C3 reproduction.

## Interpretation and residual risks

emender: fixed beats dense in no reported columns; fails to beat dense in overall, fork, empty. This is a scoped short-budget recipe verdict, not a universal residual-learning result.
transformer: fixed beats dense in overall, fork, empty; fails to beat dense in no reported columns. This is a scoped short-budget recipe verdict, not a universal residual-learning result.

One seed and corpus; corpus-wide interval/context selectors remain transductive despite train-only counts. Training self-counts have no zero-target exposure and do not model held-out support failures. OOD here means empty index support, not proof of semantic/domain distribution shift. Equal wall clocks do not equal optimizer steps: first-arm cold kernels/compilation and later cached execution can alter effective progress; curves retain startup costs. Dense historical precision/initialization are not exact matched retrains, as controls were explicitly reused. Allocated unused conditioner/OOD parameters remain in configured counts; floor adds four parameters. No causal selector, leave-one-out exposure, significance, scale-up or architecture-wide conclusion. Routing constraints remove the identified collapse pathway but cannot by themselves establish residual superiority. The inference integration receipt C3 remains the established positive result unless these column-specific comparisons improve it. No resilient DiLoCo behavior was changed or run; its requirement IDs are inapplicable.

Initial pytest collection failed on a package-relative import; corrected to absolute import, then six tests passed. The requested contact_supervisor tool was absent from the complete active tool catalog; no orchestration service or normal-text supervisor handoff was substituted.
