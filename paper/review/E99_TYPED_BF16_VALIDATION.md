# E99 typed-gdn2-lm bf16 validation pilot

**Date:** 2026-06-06
**Task:** `bf16-validation-pilot`
**Scope:** non-gating bf16 validation for the CMA-searched typed/unified-cell +
FLA-GDN-2 candidate region from `E99_1P3B_LM_DECISION.md` section 7.

## Result

bf16 **works** for the searched `typed-gdn2-lm` eval87/eval95 shape. Both
token-matched real-training pilots were NaN-free, kept the fp32 held-out BPB
within about 0.007 BPB, and passed the fresh-process checkpoint round-trip gate
with 0 missing / 0 unexpected keys.

The original decision document's main uncertainty was whether typed had a real
~3x throughput handicap or only an unvalidated dtype handicap. This pilot shows
the handicap was dtype-path, not intrinsic to the searched typed shape:
bf16 reaches **8.35k-8.59k tok/s**, versus the fp32 eval87 pilot's 2.69k tok/s.
The typed candidate still does not beat dense GDN-2's previously measured short
held-out BPB (2.047), but the "typed is 3x slower" reason no longer holds for
this shape.

## Protocol

- Real training, no mock: production `LadderLM` path, real Pile text
  `/home/erikg/elman/data/pile.txt`, `p50k_base`, context 2048, schedule-free
  AdamW, batch size 2.
- Dtype path: model parameters cast to `bfloat16`, CUDA autocast enabled for
  train, held-out eval, and reload-loss forward, matching the production
  `train.py --bf16` convention.
- Runs: eval87 and eval95 were token-matched to their fp32 pilot step counts.
- Idle-GPU gate: all 8 GPUs were idle at the first gate. A first launch on GPUs
  0/1 was interrupted when another task started 60-minute jobs there after the
  gate and before producing results. The valid measurements below were then run
  on idle GPUs 2/3 while 0/1 were occupied by the other task.
- Bound: each valid run completed in under 15 minutes, well below the 1-hour
  per-run cap.
- Held-out slice: canonical E99 pilot Pile slice
  `/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt`,
  9,999,511 bytes, sha256
  `3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a`
  (verified). BPB uses the same 40-batch slice method as the fp32 CMA pilot so
  the deltas are like-for-like.
- Checkpoints: temporary `.pt` files were created only for the fresh-process
  reload gate and deleted by the harness. No checkpoint, safetensors, HF artifact,
  or generated PDF was staged or published. `paper/main.typ` was not edited.

## Results

| candidate | dtype | steps | tokens | wall min | tok/s | final loss | held-out BPB | fp32 BPB | BPB delta vs fp32 | finite? | round-trip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| eval87, dim3328 h102 ns32 depth17 lr1.471e-3 | bf16 | 1,796 | 7,360,008 | 14.94 | 8,348.4 | 5.79527 | 2.09672 | 2.08987 | +0.00685 | yes | pass |
| eval95, dim3328 h101 ns32 depth17 lr1.497e-3 | bf16 | 1,643 | 6,733,014 | 13.29 | 8,587.8 | 5.63665 | 2.09475 | 2.09990 | -0.00515 | yes | pass |

## Round-Trip Gate

| candidate | l_pre | l_post | delta | missing keys | unexpected keys | gate |
|---|---:|---:|---:|---:|---:|:--:|
| eval87 bf16 | 10.032095 | 10.031895 | 0.00020027 | 0 | 0 | pass |
| eval95 bf16 | 8.357517 | 8.357538 | 0.00002098 | 0 | 0 | pass |

Gate criterion: fresh-process reload, loss delta <= 1e-2, 0 missing keys, 0
unexpected keys.

## NaN / Finite-State Check

| candidate | nan_seen | nonfinite_grad_seen | finite_losses | finite_grad_norms | fp32 baseline stability |
|---|:--:|:--:|:--:|:--:|:--:|
| eval87 bf16 | false | false | true | true | fp32 pilot was NaN-free and RT pass |
| eval95 bf16 | false | false | true | true | fp32 pilot was NaN-free and RT pass |

## Artifacts

- bf16 eval87 raw result:
  `experiments/e99_1p3b_bf16_validation/results/eval87_bf16/pilot_results.json`
- bf16 eval95 raw result:
  `experiments/e99_1p3b_bf16_validation/results/eval95_bf16/pilot_results.json`
- config and fp32 baseline pointers:
  `experiments/e99_1p3b_bf16_validation/configs_eval87_eval95.json`
- harness change:
  `experiments/e99_1p3b_cma/pilot.py` now supports the production bf16 path and
  records tok/s, finite-loss/finite-gradient flags, peak memory, verified held-out
  slice metadata, and dtype in the result JSON.

## Interpretation

The bf16 typed pilot satisfies the non-gating discriminator: bf16 is stable, the
checkpoint gate passes, and per-token quality is preserved relative to the fp32
eval87/eval95 baselines at matched token counts. Throughput is now in the dense
GDN-2 range: eval87 is 8,348 tok/s and eval95 is 8,588 tok/s, compared with the
previous fp32 typed reference of about 2,686 tok/s.

This means the typed candidate loses the prior cost disqualification. It does
not, by itself, prove typed should replace dense GDN-2: the dense short-run
control remains lower on held-out BPB (2.047), while bf16 typed lands at
2.095-2.097 on the same E99 pilot method used for its fp32 comparison. The next
scale-choice review should therefore compare dense and typed on equal bf16
throughput/cost rather than treating typed as a 3x slower fp32-only option.

## Validation Checklist

- [x] Short runs only: eval87 14.94 min, eval95 13.29 min; no full/long run.
- [x] Idle-GPU-only valid measurements: rerun on GPUs 2/3 after aborting the
      conflicted 0/1 launch; no reported result comes from the conflicted launch.
- [x] Real Pile, `p50k_base`, ctx2048, real training; no mock data.
- [x] Reported bf16 tok/s, held-out BPB, and NaN-freeness vs fp32 baseline.
- [x] Checkpoint round-trip gate passed in a fresh process with delta <= 1e-2
      and 0/0 keys for both candidates.
- [x] No `paper/main.typ` edit; no push; no HF/checkpoint publish; no checkpoint
      artifact left in the repo.
