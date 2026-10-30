# X42 velocity baseline anchors

45 minutes of active training per arm; evaluation and setup excluded. Whole updates may overshoot.
Baselines match c06 trainable capacity (150,003,984); c06 also has 77,313,120 frozen parameters.
Same GPT-2 vocabulary 50,257, fixed pile24 prefix, B64/T2048, seed 3401, c06 LR/warmup, 24 held-out reset windows.
Baselines use part8 bf16 compute and fused vocabulary CE; emender retains certified FP32 store/loss.
Baseline score = NLL; emender adds 0.01*occ_frac plus degeneracy penalties. Raw NLL is the direct comparison.

| arm | status | score | raw NLL | tokens seen | tok/s |
|---|---|---:|---:|---:|---:|
| pure emender c06 (fresh anchor) | queued | pending | pending | pending | pending |
| transformer | queued | pending | pending | pending | pending |
| gdn2 | queued | pending | pending | pending | pending |

Queued/failed rows are not measurements. Results are not yet evidence that 5.37 is strong.
No resilient DiLoCo behavior is changed; resilient requirement IDs are not applicable.
