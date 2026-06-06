# LR x Shape Warm Pilot

Date: 2026-06-06T04:33:19.238534+00:00

## Scope

Two dense GDN-2 bf16 shapes were trained on real Pile tokens with the E99 matched-controls harness. Each run used the task-local wrapper `experiments/lr_x_shape/lr_x_shape_pilot.py`, which reuses held-out BPB, sustained tok/s, NaN detection, and fresh-process checkpoint round-trip validation from `experiments/e99_1p3b_controls/e99_lm_controls.py`.

No full/long run was launched. No `paper/main.typ` edit, push, HF publish, or checkpoint publish was performed; the harness deletes round-trip checkpoints after reload.

## Results

| config | shape | train min | steps | tokens | held-out BPB ↓ | late loss ↓ | final loss ↓ | tok/s ↑ | RT | stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---|
| fla-gdn-controls-shape | dim2688/depth21/44h/ns64/lr0.000863 | 60.0 | 6530 | 26759940 | 1.764601 | 4.694623 | 3.892880 | 7526.4 | pass | budget_reached |
| fla-gdn-handoff-shape | dim3456/depth12/38h/ns64/lr0.0008627 | 60.0 | 7582 | 31071036 | 1.730988 | 4.628097 | 4.943975 | 8744.5 | pass | budget_reached |

## Recommendation

Recommend `fla-gdn-handoff-shape` (dim3456/depth12/38h/ns64/lr0.0008627) for the dense GDN-2 full-run shape.
It has the best gated held-out BPB among round-trip-clean, NaN-free runs (1.730988) while sustaining 8744.5 tok/s.
The BPB margin over `fla-gdn-controls-shape` is 0.033613; throughput is treated as secondary after the BPB and round-trip gates.

## Validation

- [x] 2 configs x ~1h, idle-GPU-only, real Pile training.
- [x] Held-out BPB and tok/s reported per config.
- [x] One shape recommended with rationale.
- [x] Round-trip gate result reported per config.
- [x] No full/long run, `paper/main.typ` edit, push, HF publish, or checkpoint publish.
