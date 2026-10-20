# X12: matched T64 depth separation, registered before launch

## Scheduling takes precedence

W3's frozen eight-GPU gate read has absolute priority. No X12 GPU lease,
probe, model transfer or trainer may run before its completion. CPU preparation
and a detached post-marker launcher are approved by the supervisor. A free card,
arm-training DONE, gate launcher or supervision directory is NOT admission.

The launcher fetches origin/main every 300 seconds. Admission requires a committed
`W3-report.md` Gate verdict/final-verdict heading AND a committed JSON under
`receipts/w3-matcheddose/` named `gate-read*` or containing `gate` and `verdict`,
with the frozen reader's `emender-w3-verdict-v1` schema and
`status=measurements-complete`. Unknown formats fail closed and need attended
clarification. Remote/report commits and receipt hashes are recorded. Only then
are two broker-grantable cards chosen, one leased per independent trainer.
W3 need not pass its optimizer gate: completion of the frozen read is the condition.

## Fixed matched recipe

- Scratch seed0 E97 and real, unmodified `fla-gdn` GDN2; stock bf16 autocast,
  fp32 parameters. No fp32 recurrence adapter. E97 uses its established X10
  legacy-bf16 training recurrence for both training and evaluation.
- **Four layers each**, dim256. E97: **9,900,800** parameters; GDN2: **9,887,056**.
  These are the existing nominal 8M configurations, not a new parameter match.
- Deep-first T64 from update1: 64 driver positions, p61/vocab3721, batch32,
  66 tokens right-padded to80 by the unchanged X3 forward path.
- Uniform masked-target CE, ordinary AdamW live parameters, betas(.9,.999),
  eps1e-8, clip1. All parameters decay .01, including GDN2 A_log/dt_bias;
  `_no_weight_decay` tags intentionally ignored, exactly the X10 AdamW policy.
- Updates1..5000 warm linearly to1e-3; cosine to1e-5 at **1,000,000**.
  The cosine denominator is extended to1M, not inherited from X10's500K schedule.
  Actual LR logged every update. Identical numpy.default_rng(0) procedural
  batches, independently reset for each architecture; torch.manual_seed(0).
- Frozen X3 seed0 T64 validation: 512 examples, P1-P64 correct/512 and CE at
  update0 and every500 updates through1M. Arrays hashed before launch and
  verified/archived by each trainer; never sampled for training. No test scores.
- No stagnation rule, threshold stop, all-perfect stop, curriculum or carry loss.
  Fixed1M ceiling only. Failures are infrastructure censoring, not architecture
  findings. No automatic retries/restarts. No wall-based early termination.
- Bounded retention: entry, aggregate-best, latest/final (at most3 checkpoints
  plus an atomic staging file per run), with deletion hashes. Aggregate-best
  retention is not primary-readout selection. This is standalone single-card
  research, not resilient training or a Frontier submission.

## Predictions registered verbatim before launch

(a) SEPARATION HYPOTHESIS (the operator's framing + the mechanism expectation): GDN2's affine carry is depth-robust by construction; E97's bounded tanh state must re-encode exact state per step. Prediction: as T grows past the depth-4 parity point, GDN2's perfect-prefix depth EXCEEDS E97's materially under the matched working recipe (e.g., GDN2 >=2 positions beyond E97's perfect prefix at T64's ceiling). Report each run's perfect-prefix depth and per-position maxima as the primary readout.

(b) PARITY-AT-DEPTH alternative: both reach the same prefix depth at T64/1M - then the architectures are equivalent on this task family at 8M under working recipes, and the composition-depth framing was always a training-stack artifact; register this reading plainly if it holds.

(c) The P5-first acquisition timing at T64 (before deeper positions can matter) - if neither learns P5 at the longer ceiling, register that the ceiling is still the binding constraint and the separation question needs either longer ceilings or curriculum staging under the working recipe.

(d) **GDN2 should not be able to do more nonlinear steps than it has layers**

Supervisor/operator addition: GDN2's recurrent carry is linear in state
(delta-rule update linear-in-S with token-dependent coefficients); state-dependent
nonlinearity (squaring the carried coordinate) is available only via the
readout->MLP->write-back path, once per layer per timestep; E97's tanh acts directly
on the state each step. Prediction(d): GDN2's perfect-prefix depth at T64 caps at
or near its **four layers**, while E97's prefix extends beyond its own **four
layers** if depth-robust.

**(a) and (d) are competing, contradictory predictions, not a reconciled story.**
Let the T64 columns decide. GDN2's observed four-perfect plus budding P5 must be
reconciled with whichever prediction survives. These are predictions, not
established mechanism conclusions. Report all other outcomes plainly.

## Primary instrument and native wake procedure

Report each architecture's contiguous perfect prefix at fixed-final1M and every
position's maximum correct count, side by side, through P64. Report maximum
prefix achieved at a single validation separately from independent position
maxima; maxima from different updates are not a jointly perfect prefix.
First>=50/512 (material) and first512/512 (perfect) timestamps are recorded from
the validation history, especially P5. Missing acquisition is right-censored;
partial runs cannot decide the fixed-final hypotheses.

Native supervisor watcher (no WorksGood service):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage12_separation --interval 900
# On wake, including a partial growth report:
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.watch_invertible_stage12_separation --once --report
```

Exit0 means both hash-bound successful1M terminal receipts, exit1 failure,
exit2 pending for `--once`. `GROWTH_REPORT.md` contains all64 positional rows,
latest counts, maxima and acquisition timing. Launcher/trainer identities include
host and /proc starttime, and workers heartbeat/release only their own leases.
