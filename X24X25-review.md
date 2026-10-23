## Review

**Verdict: ACCEPT — registration/launch machinery only, not training outcomes.**  
**Merge verdict: OK.** No issues found.

Paths below use `X24` = `experiments/expressivity_tasks/results_stage24_softplus` and `X25` = `experiments/expressivity_tasks/results_stage25_recipe_controlled`.

### Correct

- **Kernel extension implements plain softplus consistently.** Forward and checkpoint replay use the stable softplus formula; backward uses the matching derivative. Defaults remain off. See `ndm/triton/e88_triton_forward.py:255–265`, `ndm/triton/e88_triton_backward.py:270–278,413–420`, and `ndm/triton/e97_sequential.py:35–78`.

- **Gate numbers match the receipts:** 87 tests passed, exit zero; 90,904-token zero-update forward/backward/evaluation smoke; worst parameter-gradient relative error **0.010769722869089489**, below the registered `.08` threshold. Output/state/evaluation errors also satisfy their `.03` budgets. Evidence: `X24/receipts/tests.txt:6–17`, `tests_exit_code.txt:1`, `kernel_match.json:1–45`, and `smoke_e97_graph.json:1–12`. These are qualification results, not trained-weight stability guarantees.

- **Concession #18 and precision ruling are explicit.** Plain softplus—not `softplus_c`—is selected; counting requires unbounded-above capacity. Legacy FP32 live accumulation/bf16 replay is retained, rather than introducing FP32-everywhere. See `X24/REGISTRATION.md:7–39` and `ndm/triton/e88_triton_forward.py:636–648`.

- **The E97 graph comparison changes activation only.** The softplus builder constructs the stock model before changing activation/binding flags. X25 applies the same constant LR and weight decay to every arm. Initial-weight equivalence and legacy precision have regression tests. See `experiments/expressivity_tasks/run_e97_softplus_graph.py:12–21`, `run_recipe_controlled_separators.py:14–32`, and `tests/test_recipe_controlled_separators.py:14–44`. Both graph configs record identical optimizer settings and dataset identities.

- **The amendment preserves scientific originals and control evidence.** X25 copies prior predictions and separately records the superseded recipe; historical registrations are hash-bound. Interrupted result/terminal records are retained separately, and checkpoint hashes match the before/after preservation receipts. See `scripts/register_recipe_controlled_separators.py:42–69` and both historical `receipts/e97_graph_seed0/{superseded_before,recipe_superseded,interruption_result,interruption_terminal}.json`.

- **Kills do not masquerade as scientific FAILs.** Both superseded graph runs have `status: censored`, exit 143, explicit “not a training verdict” wording, and CENSORED markers—not FAIL markers. Their results remain scientifically unadjudicated.

- **Predictions precede relaunches.** Supplied history artifacts and the parent’s ancestry audit establish registration-before-launch ordering. Launch receipts identify the registration commits themselves: X24 `fca01ad8`, X25 `36bd9d91`; X25’s five launch records retain commands, cards, trainer identities, optimizer settings, and runtime dispatch (`X25/receipts/launch_verified.json:1–239`). This history conclusion relies on the supplied artifacts/parent Git audit, not `watchdog_diff`.

- **GDN2 handling is appropriately bounded.** The retained traceback identifies nonfinite logits before update 5340’s backward. No state norms or kernel telemetry were available; the diagnosis explicitly declines to infer state overflow. See `X25/receipts/gdn2_nonfinite_diagnosis.json:1–24` and `gdn2_crash_traceback.txt:1–11`.

### Eventual-verdict boundaries

No additional registration conditions. Preserve the registered interpretation:

- X25 is evidence-informed; LR, schedule, and weight decay changed together, so improvement cannot isolate LR causality.
- Superseded hot controls remain censored—not architectural failures.
- Complete selected-checkpoint ID/OOD counts are required; ID failure and infrastructure interruption must remain separate.
- Single-seed activation results cannot establish exact counting realization, asymptotic expressivity, or directly diagnose GDN2 state overflow.

Tests were inspected through receipts, not rerun. No repository files were modified.