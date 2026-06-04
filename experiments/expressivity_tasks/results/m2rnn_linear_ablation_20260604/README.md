# M2RNN raw-write state-nonlinearity ablation (2026-06-04)

Task `m2rnn-linear-ablation`. Symmetric counterpart of the E88 `linear_state`
knob (s5sym-eval), restricted to the M2RNN-CMA winner.

## Arms
- `m2rnn-nonlinear` — as built: `Z = tanh(h W + k v^T)`  (`--linear_state 0`)
- `m2rnn-linear`    — tanh removed: `Z = h W + k v^T`     (`--linear_state 1`)

Both use the M2RNN-CMA winner shape/lr verbatim
(`results/s5_symmetric_20260603/winners/m2rnn.args.json`):
dim=512, depth=5, n_heads=19, n_state=32, lr=0.0016057620284487264 (~8.0M params).

## Protocol (identical to s5sym-eval)
3 seeds {42,123,456}; S5 `s5_permutation` train T=128 20000 steps; S3
`s3_permutation` train T=128 10000 steps; eval grid {128,256,512,1024};
schedule-free AdamW, batch 32, seq_len 128; GPUs 6,7 only.

## What "linear M2RNN" means in code
`ndm/models/m2rnn_baseline.py` — the per-step state update in the PyTorch loop:

```python
pre = torch.matmul(h, W) + outer            # h W + k v^T
candidate = pre if self.linear_state else torch.tanh(pre)
h = f_t * h + (1.0 - f_t) * candidate
```

`linear_state` is the exact raw-write analogue of E88's `linear_state`
(`ndm/models/e88_fla_hybrid.py:1709`, `S = decay*S + outer` vs
`tanh(decay*S + outer)`). The forget gate `f_t`, the read-out `q^T h`, and the
`D·v` residual are unchanged; only the candidate state-nonlinearity is removed.

The XMA Triton kernel hardcodes the tanh and cannot express the linear variant,
so when `linear_state=True` the layer always uses the PyTorch loop. `XMA_PATH`
is unset in this environment, so the nonlinear arm also runs the PyTorch loop —
both arms share the identical code path (matched comparison).

## Files
- `m2rnn-{nonlinear,linear}.args.json` — per-arm config + semantics note.
- `eval/{arm}_{S5,S3}_seed{42,123,456}.json` — real `train_hybrid` output
  (per-step `eval_acc`, `final_acc`, `length_extrap`).
- `eval/summary.json` — mean/std over seeds (written by `aggregate_m2rnn_linear_ablation.py`).
- `TABLE.md` — comparison table incl. the E88 knob reference rows.

## Nonlinear-arm provenance
The `m2rnn-nonlinear_*` JSONs are copied from the s5sym-eval M2RNN winner runs
(`results/s5_symmetric_20260603/eval/m2rnn_*`). Those runs used `linear_state=None`,
which defaults to `False` in the layer (tanh ON) — behaviorally byte-identical to
`--linear_state 0` under the new code (the edit only adds an `if self.linear_state`
branch; the `False` path computes the same `tanh(matmul(h,W)+outer)` as before).
They are the same winner config, seeds, step counts, and eval grid, so they are
the genuine as-built nonlinear arm — reused rather than recomputed to respect the
shared GPUs. Re-train them with `eval_m2rnn_linear_ablation.py --force-nonlinear`.
