# X2c — fresh both-arm fine-tunes at an E97-gated common LR

Status: **gate_failed_no_full_runs**. Exact suffix floor **0.026874%** (printed rounded: **0.0269%**, 1/3721).

Neither registered LR rescued E97 by the approved gate. No gated LR exists: the seven full runs and three one-step diagnostics were NOT launched. Three-seed means, steps-to-plateau, x4/x8 and ablation effects are unavailable, not zero or imputed. Stopping after both gates fail is the binding supervisor ruling.

## Amendment #3 registration

```json
{
  "amendment": 3,
  "registered_at": "2026-10-16T17:26:53.341683+00:00",
  "base_commit": "fb2c1b6e2cdc196403332e1ac86d9390be33c6e0",
  "authority": "/home/erikg/emender-scratch/fix-fanout/taskX2c-eq-lr.md and live supervisor gate/recipe ruling",
  "recipe": "X2b unchanged except LR: fresh AdamW (0.9,0.95), batch32, wd0.01, clip0.5, bf16, 300 warmup then constant LR; Schedule-Free y-mode released-weight recovery, no optimizer state loaded",
  "pilot": "Both released arms seed0, drivers128, 3000 fresh updates at2e-4; E97 gate: >1% first250/final250 improvement AND negative OLS slope, both over trailing2000 updates. If fail, one fresh paired repeat at1e-3. If both fail, no full runs.",
  "full_runs": "Seven headline runs: E97/GDN2 seeds0,1,2; E97 identity seed0. Three seed0 one-step diagnostics, preserving X2b contract. All start fresh from released weights at gated LR.",
  "fairness_rule": "whichever lr rescues E97 is the lr BOTH arms run; no per-arm tuning; if 1e-3 is needed to move E97, GDN2 runs 1e-3 too.",
  "evaluation": "X2/X2b frozen val/test pools unchanged, same seeds/input hashes; validation-only selection, joint-class full-vocab argmax; causal right-padding to16; x4/x8; matched and union-of-stop panels",
  "plateau": "Eligibility>=10000; trailing2000 first250-vs-final250 mean train-loss improvement<1% AND best-val gain<0.001; cap60000; all arms identical",
  "capacity": "If free capacity <400GiB at a checkpoint check, retain best-val chain, every10K grid, latest250 rolling, own stop, legacy fixed {1000,4000,8000,12000,16000,18269}, and all known headline stop steps; scan published frozen selections each prune. Each prune records removed steps and retained hashes. Any cross-arm panel cell whose checkpoint was already pruned before its step became required is reported as unavailable/pruned - never imputed, never re-created by interpolation, never silently dropped. Stop steps broadcast via selection_frozen.json; running arms scan each250 check.",
  "x2b_status": "X2b's 5e-5 results are retained as the lr-sensitivity data point, superseded as the primary comparison by the gated-lr runs. Active X2b GDN2 runs finish naturally and are archived, never cancelled or modified by X2c.",
  "basis": {
    "X2b_report_sha256": "aad117acc97fbbc7816e01a9b228d9b2009e612c66e1e72a582d14d264f6a53d",
    "X2b_amendment_sha256": "1abd96758d0222b7b0a5f3f34d91e573a0553fc9b0990d9d22c4312e92471230",
    "task_sha256": "5e98214227aa6f5852d61cbdb997cdcc1d134656eaab12092085e1b87e36221f"
  },
  "resilience_requirements": "Not applicable: standalone single-GPU expressivity fine-tunes, no DiLoCo behavior changed; local lambda01, no Slurm or WorksGood service"
}
```

## Fairness rule (verbatim)

whichever lr rescues E97 is the lr BOTH arms run; no per-arm tuning; if 1e-3 is needed to move E97, GDN2 runs 1e-3 too.

X2b's 5e-5 results are retained as the lr-sensitivity data point, superseded as the primary comparison by the gated-lr runs. Active X2b GDN2 runs finish naturally and are archived, never cancelled or modified by X2c.

That is the registration's intended primary-comparison designation, not a claim of completed replacement evidence: the gate failed, so no gated-LR primary comparison exists and no X2b measurement has been replaced.

## Pilot stall/rescue evidence

Gate preregistered before launch: 3000 fresh updates, trailing2000 first250/final250 means, >1% improvement AND negative OLS slope. Both arms use fresh released weights/cleared AdamW states for every attempt. No test-pool selection.

| Pilot | LR | First250 loss | Final250 loss | Improvement | OLS nats/update | Gate | Floor |
|---|---:|---:|---:|---:|---:|---|---:|
| pilot_2e-4_e97_seed0 | 0.0002 | 8.215364 | 8.203115 | 0.1491% | -6.39556071e-06 | False | 0.026874% |
| pilot_2e-4_gdn2_seed0 | 0.0002 | 8.229951 | 8.220162 | 0.1189% | -5.2645367e-06 | False | 0.026874% |
| pilot_1e-3_e97_seed0 | 0.001 | 8.238054 | 8.231817 | 0.0757% | -3.32959511e-06 | False | 0.026874% |
| pilot_1e-3_gdn2_seed0 | 0.001 | 8.285579 | 8.258980 | 0.3210% | -1.62600666e-05 | False | 0.026874% |

Common LR decision: `None`; decision timestamp: `2026-10-16T18:30:03.291305+00:00`.

## Honest per-seed comparison

Full-vocabulary argmax exact joint class `61*x+y`; invalid classes are wrong. Best validation accuracy, then CE, then earlier update selects every arm. Test panels are evaluated only after selection_frozen.json. Fresh procedural batch32; unchanged frozen512 val/test pools; right-pad to16 and slice before loss/scoring.

| Run | Best val | Selected update | Stop update/reason | Test1x | Test4x | Test8x | Floor |
|---|---:|---:|---|---:|---:|---:|---:|
| headline_e97_seed0 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| headline_e97_seed1 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| headline_e97_seed2 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| headline_gdn2_seed0 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| headline_gdn2_seed1 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| headline_gdn2_seed2 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| headline_e97_identity_seed0 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| calibration_e97_seed0 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| calibration_gdn2_seed0 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |
| calibration_e97_identity_seed0 | not_launched_gate_failed | — | — | — | — | — | 0.026874% |

## Three-seed means ± sample SD (SEM in JSON)

| Arm | Best val | Steps to plateau/cap | Test1x | Test4x | Test8x | Floor |
|---|---:|---:|---:|---:|---:|---:|
| e97 | unavailable: gate failed | unavailable | unavailable | unavailable | unavailable | 0.026874% |
| gdn2 | unavailable: gate failed | unavailable | unavailable | unavailable | unavailable | 0.026874% |

## Coordinates and cost

| Run | x1x | y1x | Terminal1x | Parameters | Matrix state elements | Real train tokens | Tokens/s | Wall seconds | Floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Real train tokens include initialization, exclude pad overhead. State counts exclude short-conv caches; per-layer shapes/flags in configs. Released full-vocabulary heads and all parameters are fine-tuned; not scratch-model or parameter-matched arms.

## Matched-step and union-of-stop panels

No extra training beyond stop; pruned-before-required cells are unavailable, never imputed.

| Run | Update | Test1x | Test4x | Test8x | Floor |
|---|---:|---:|---:|---:|---:|

## Interpretation and receipts

The observed pilot loss slopes are negative but the required >1% window movement is absent at both registered LRs. This is a failed rescue gate, not a seven-run architecture comparison, arithmetic diagnostic, extrapolation result, or impossibility claim. The requested full comparison remains unperformed; any further training requires a new operator decision.

This comparison tests learning efficiency of this released pair under one common LR, not universal representability or CMA causality. A one-step arithmetic failure blocks a depth-specific interpretation; a single-seed tanh-removal contrast is exploratory. Near-floor results are not learned map computation. All seeds and incomplete/failing arms remain visible.

Pilot receipts bind source checkpoint/args/runner hashes, exact commands, leased GPU UUID/index snapshots, training/validation logs, fresh optimizer declaration, exit0 for all four pilots, unchanged frozen-pool hashes, replayed gate windows/slopes and final checkpoint hashes. No full-run selected/test/matched/cross-stop measurements exist.

Binary checkpoints are local ignored artifacts, identified by hashes, not git payloads. See receipts/cpu_tests.txt, receipts/environment.txt, receipts/validation_commands.txt and receipts/final_audit.json. Independent acceptance review remains required.

X2 and X2b evidence is preserved. Local lambda01, exclusive scripts/gpu_lease.sh leases; no Slurm or WorksGood service. Resilient DiLoCo gap-matrix IDs: not applicable (no resilient behavior changed).

