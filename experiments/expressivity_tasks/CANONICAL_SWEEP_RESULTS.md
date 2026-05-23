# Canonical paper sweep results

**Config:** `dim=384, depth=4, n_heads=32, n_state=32, sf-AdamW, 10K steps, batch_size=32`
6 tasks × 3 patterns × 3 seeds = **54 runs**.

## Summary table (mean ± std over 3 seeds)

| Task              | pure_E88           | pure_FLA           | hybrid_AABB        |
|-------------------|--------------------|--------------------|--------------------|
| parity            | **1.000 ± 0.000**  | 0.857 ± 0.022      | 1.000 ± 0.000      |
| modular_counter   | **0.903 ± 0.033**  | 0.648 ± 0.118      | 0.536 ± 0.238      |
| fsm_tracking      | **1.000 ± 0.000**  | 0.830 ± 0.040      | 0.713 ± 0.021      |
| dyck              | 1.000 ± 0.000      | 1.000 ± 0.000      | 1.000 ± 0.000      |
| assoc_recall      | 0.881 ± 0.025      | **0.997 ± 0.003**  | 0.947 ± 0.006      |
| selective_copy    | 1.000 ± 0.000      | 1.000 ± 0.000      | 1.000 ± 0.000      |

## Read

**Pure E88 wins or ties on 5 of 6 tasks.** It dominates the canonical state-tracking benchmarks (parity, modular counter K=5, FSM tracking) where parallel-scan SSMs are theoretically constrained. It matches FLA-GDN on dyck-1 and selective copy. Only on associative recall does FLA-GDN have a meaningful edge (+0.12), reflecting parallel-attention's natural fit for that task.

**Hybrid is an ablation, not a result.** Stacking E88-E88-FLA-FLA underperforms pure E88 on modular_counter (0.54 vs 0.90) and fsm_tracking (0.71 vs 1.00) — the FLA layers degrade what E88 can do alone. This argues against framing E88 as a "component for hybrid stacks"; the value is in pure E88.

## Reproducibility

```bash
cd experiments/expressivity_tasks
python run_canonical_sweep.py --gpus <gpus> --seeds 42 123 456
```

Results are written to `results/canon_<pattern>__<task>__seed<seed>.json` with `final_acc` field.

## Caveats

- 10K steps is enough for grokking on these tasks at dim=384, but is short; longer runs would tighten the ± bands.
- assoc_recall is tested at sequence length 64, K=8 keys. E88 may close the gap with longer training or a wider model.
- Hybrid_AABB is one specific stacking; other arrangements (BABA, ABAB) give similar conclusions per the earlier hybrid sweep.
