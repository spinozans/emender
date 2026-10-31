# X42 suspect ablations

| Variant | Status | Score | NLL | NLL improvement | Occupancy | R-CAP rate | Mean gate |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | completed | 4.802025 | 4.742690 | 0.000000 | 0.933582 | 0.397870 | 0.968164 |
| budgets4 | completed | 4.768561 | 4.765585 | -0.022896 | 0.297600 | 0.000000 | 0.966685 |
| read_width256 | completed | 4.890189 | 4.830853 | -0.088164 | 0.933557 | 0.397917 | 0.972040 |

Material improvement thresholds were fixed before endpoints: score decrease >=0.1 and NLL decrease >=0.1, reported separately. The NLL threshold was fixed at launch; the score rule is additionally receipted in comparison_rule.json. Per-head telemetry, timing, token counts, source hashes and state attestation are retained in summary.json and the individual result/amendment receipts.

Pure search restored the exact paused gen-4 state with unchanged registration. Its attended launcher requests only card 1 through the broker and queues behind existing holders. Check pure_search.status/training_started in summary.json: parent restart alone does not prove a new training update.

Residual risks: One seed and one 2700-second wall-clock candidate per ablation; no significance estimate. Budget and width changes can alter throughput; width uses existing capacity solver and changes MLP hidden size. Part25 existing budget-transfer verdict is incomplete and was not modified. Event-density information-flow cost remains static analysis only. Historical v1 source differs from current source; exact registered historical Block was verified. Supervisor contact tool unavailable in this runtime. Existing baseline launchers also request card 1 and can reacquire it; pure parent is resumed but may remain queued without new training progress.
