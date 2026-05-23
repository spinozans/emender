# Modular Counter Follow-Up

This follow-up targets the ambiguous `modular_counter` row from the matched 8M
suite. At 10K steps, tied/CMAES-shaped M2RNN slightly outruns E88 on
`K=5`, while paper-shaped M2RNN stays near random. The key question is whether
this is a real architectural edge or a finite-training / precision artifact.

Runner:

```bash
python experiments/expressivity_tasks/run_modular_counter_followup.py \
  --gpus 0,5,6,7 \
  --output_dir experiments/expressivity_tasks/results/modular_counter_followup_20260511
```

Default variants:

- `K5_T128_long`: 30K steps at `K=5`, `T=128`, with length extrapolation to
  `128,256,512,1024,2048`.
- `K20_T256_hard`: 20K steps at `K=20`, `T=256`, with extrapolation to
  `256,512,1024,2048`.
- `K50_T256_hard`: 20K steps at `K=50`, `T=256`, with extrapolation to
  `256,512,1024,2048`.

Default models:

- `E88_H32N32_bf16`: current matched-suite E88 shape.
- `E88_H32N32_fp32`: same shape with autocast disabled.
- `E88_H64N16_fp32`: more address programs, smaller per-head state.
- `FLA_H32N32_fp32`: linear delta baseline with matched precision.
- `M2RNN_tied`: tied/CMAES-shaped M2RNN.
- `M2RNN_paper`: paper-shaped grouped-head M2RNN.

Interpretation:

- If E88 reaches parity with M2RNN tied on `K=5` after 30K steps, the 10K result
  was probably a grokking-speed artifact.
- If E88 fp32 materially beats E88 bf16, exact algorithmic tasks are precision
  sensitive and the matched suite should report precision explicitly.
- If tied M2RNN holds `K=5` but fails `K=20/K=50` or length extrapolation, it is
  learning an easy finite counter rather than a durable algorithm.
- If paper-shaped M2RNN remains near baseline, the negative result is specific
  to the released/paper geometry, not to every matrix-state RNN.
