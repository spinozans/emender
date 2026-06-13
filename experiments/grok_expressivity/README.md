# grok-expressivity

Does the e97 nonlinear-in-time cell GROK the hard algorithmic separators (S5
state-tracking, modular_quadratic, a^n b^n c^n counting) under canonical grokking
conditions — testing whether the prior capability nulls/ties were PRE-GROK artifacts
of the three grok-suppressors the earlier runs used (too few steps, wd 0/0.01,
schedule-free optimizer)?

**Verdict: NO. See [VERDICT.md](VERDICT.md).** The per-step nonlinearity unlocks no
grokked capability; on S5 the LINEAR GDN-2 groks to test-acc 1.000 while e97 never
leaves baseline. The nulls are real, not pre-grok artifacts for e97.

## Files

- `train_grok.py` — grokking trainer: fixed train/test split, vanilla AdamW with
  explicit `--weight_decay`, train+test accuracy logged every `--eval_interval`,
  grok/memorize-step detection, length-extrapolation eval. Arms: `e97` (E97 split-edit
  fused bf16 Triton, tanh state — asserted no eager), `e97-lin` (same fused kernel,
  linear state — the within-substrate control), `gdn2` (FLA GatedDeltaNet, fused),
  `e97-ht` (phi-shell hardtanh, per-step scan; optional/slow).
- `orchestrate.py` — fan a `task × arm × wd × n_train × seed` grid across GPUs.
- `aggregate.py` — build the grok-status table + summary from `runs/*.json`.
- `runs/` — full sweep: 3 tasks × 3 arms × wd{0.01,0.1,0.3,1.0} × 2 seeds, nt=128,
  100k steps. Per-run JSON holds the complete per-step train+test curve.
- `runs/grok_table.md`, `runs/grok_summary.json` — aggregated results.
- `runs_calib/` — n_train ∈ {128,512,2048} calibration on modular_quadratic that
  located the grok regime (nt=128).

## Reproduce

```bash
eval "$(scripts/gpu_lease.sh 8)"
python experiments/grok_expressivity/orchestrate.py --gpus "$CUDA_VISIBLE_DEVICES" \
  --tasks s5_permutation,modular_quadratic,anbncn_viability \
  --arms e97,e97-lin,gdn2 --wds 0.01,0.1,0.3,1.0 --n_trains 128 --seeds 0,1 \
  --steps 100000 --eval_interval 500 --patience_evals 50
python experiments/grok_expressivity/aggregate.py experiments/grok_expressivity/runs
```

REAL data, REAL algorithms, REAL fused Triton kernels. No mocks.
