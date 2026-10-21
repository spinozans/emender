# X14 — no-freebie composition test (registered before launch)

Operator ruling 2026-10-21: P1 is a visible-input function, not map composition.
Only protocol variable: P1 has **zero training-loss weight at every update**.
Inputs/targets remain unchanged; uniform mean CE on P2..P16. P2 is the first
learnable position: one-step nonlinear composition through internal state.
P1 stays in frozen-pool evaluation as **diagnostic-only NON-COMPOSITION**;
it cannot inflate the primary aggregate or select the best checkpoint.

## Four confirmatory arms

Deep-first T16 from scratch: E97 seed0, stock-bf16 GDN2 seed0, E97 seed1,
stock-bf16 GDN2 seed1. Both seeds launch irrespective of seed0 scores.
Model/datum initialization is paired by seed, frozen pools match X10 exactly.
No resumed weights, ladder, curriculum, auxiliary carry targets, or tuning.

Exact X10-AdamW+cosine: actual stage10 update function, AdamW betas(.9,.999),
eps1e-8, decay .01 on ALL parameters (including tagged A_log/dt_bias), clip1,
batch32, p61/vocab3721, dim256/four layers, fp32 parameters/bf16 autocast.
E97 uses legacy recurrent precision and sequential Triton evaluation; GDN2
uses stock unmodified bf16 chunk recurrence, no fp32 adapter.
Linear 5K warmup to1e-3 then cosine to1e-5 at500K; LR logged every update.
Fresh numpy.default_rng(seed) procedural batches; torch.manual_seed(seed).
Frozen 512-sequence evaluation every500 updates; safety ceiling500K only.
No stagnation, threshold, or all-perfect early stop and no automatic retry.

## Predictions registered pre-launch

(a) **NO-FREEBIE CARRY:** if E97 acquires P2 materially (>=50/512), carry is
learnable without the visible subsidy; the foothold story re-scopes to
“P1 scaffolds learning but is not necessary for composition.”

(b) **FREEBIE-DEPENDENT:** if E97 stays at zero on P2+ while GDN2 acquires,
E97's composition learning depends on visible-position scaffolding: a genuinely
different learning-mode claim, receipted at **both seeds**.

(c) **GDN2 columns either way:** does the linear-state bootstrap survive
carry-only signal?

Report literal zero versus nonzero sub-material counts, and mixed outcomes;
do not force them into (a)/(b). Acquisition first50 and first512 are distinct.
Any threshold crossing is descriptive; fixed-final500K is primary, replicated
versus seed-specific outcomes are explicit. No post-result adjustment.

## Audit / standing discipline

This experiment shares X13's open audit question about the stage12 runner but
uses **stage10-pattern code paths only**. No conclusion from X12 E97-T64 is
imported. This is standalone single-card expressivity, not resilient DiLoCo or
Frontier; no distributed design requirements or Slurm binding are claimed.

P1..P16 joint counts and CE, first50/first512, latest/fixed-final, maximum-anywhere,
and composition perfect-prefix length starting at P2 are reported separately.
P1 is always labeled diagnostic-only non-composition. Primary aggregate and
best checkpoint use P2..P16 exclusively. Infrastructure/time/memory failures are
censored, not architecture failures; absent crossings are right-censored at the
observed step, never filled as zero. No score-selected restart or seed selection.
Retain entry + composition-best + latest/final (<=3 checkpoints plus staging),
with deletion hashes. Register code/dependency/pool hashes in registration.json,
commit before any confirmatory launch, verify committed bytes at launch.

## Operations / wake

Bind four distinct free physical cards using existing atomic GPU leases,
heartbeat throughout and fail closed if unavailable; never disturb live jobs.
Disposable seed99 probes exercise masked updates, all16 eval positions and
checkpoint-next-update reload parity without confirmatory frozen-pool scores.
Probe-derived wall/memory bounds are in registration.json.

Native watcher command (blocking; 0 complete, 1 failed):
`.venv/bin/python -m scripts.watch_invertible_stage14_nofreebie --interval 900`

Snapshot/wake command (2 pending is expected):
`.venv/bin/python -m scripts.watch_invertible_stage14_nofreebie --once --report`

Watcher writes GROWTH_REPORT.md on terminal wake; --report also writes a current
censored snapshot. Native arming is owned by the attended supervisor, not a
WorksGood daemon. Launch receipt and initial growth report follow separately.
