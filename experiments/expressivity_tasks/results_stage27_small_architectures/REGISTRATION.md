# X27 — our architectures at their scale: size-hypothesis pre-registration

Operator-directed exploratory size test, registered and pushed before any X27
optimizer updates. X25 is the large-model comparator, X26 the small upstream-model
comparator. Prior observed near-chance trajectories motivate, not establish, the
hypothesis: overparameterization may favor shortcut fits rather than procedures.
This is not an outcome-blind study or a claim of a proven size mechanism.

## Corrections — Concession #19

The task directive's batch128 conflicts with its size-only comparison to X25,
whose registered AND actual batch is64. Supervisor explicitly approved batch64:
the single-variable comparison to X25 governs. X26 uses batch128, different
architectures/readouts, and task-specific adapters/curricula. That cross-experiment
recipe difference limits direct X27–X26 numerical attribution; qualitative pattern
comparisons remain licensed. No claim that architecture alone was isolated across
X26/X27, or that the batch128 directive was silently followed.

## Frozen depth-1 graph configurations

All are HybridLadderLM, vocab8, tied embedding/output, SwiGLU mlp_ratio2,
no dropout, fresh initialization seed0. Counts are actual trainable tensor totals,
not estimates; `receipts/parameter_counts.json` gives module breakdowns and shapes.
- e97_graph_seed0: dim192, depth1, heads16, state16, expansion1,
  MLP hidden384; **570464 parameters**. Tanh, raw_write=False, legacy recurrent
  precision, fused Triton split-edit training and fused inference, as X25.
- e97softplus_graph_seed0: exactly the same tensors/count **570464**, but X24
  plain-softplus state using the same qualified fused engine, never centered
  softplus or eager fallback. Tanh/softplus initial tensors match at seed0.
- gdn2_graph_seed0: dim128, depth1, expansion8.625, MLP hidden256;
  **563218 parameters**. Stock fla-gdn, allow_neg_eigval=True, short-conv4,
  gated output, unchanged X25 wrapper defaults. The generic n_heads/n_state
  arguments are ignored by this wrapper: effective heads1, key head128, value
  head1104 (X25 effective heads2/key128/value1104). No false heads16 claim.

These lie in the requested200k–800k band and within1.3% of each other. Size is a
bundle (depth, width, E97 heads/state; GDN head count scales with width), not a
pure parameter-count intervention or an isolated depth-vs-width experiment.
Expansion ratios, architecture features, activations, precision, task and training
recipe remain unchanged relative to each corresponding X25 arm. No size search
based on task accuracy. No trained checkpoints restored.

## Pre-registered size / architecture / task grid

Use validation-selected complete ID/OOD test bins; learning/generalization means
>=99% ID and >=99% in BOTH OOD bins, consistent with X25's graph target.
ID failure and learned-ID/OOD degradation are reported separately, with exact
counts/losses per arm/bin; mixed outcomes are not forced into a single label.
- **SIZE-BINDING:** small-ours learn/generalize where their corresponding large
  X25 arms remain near chance or fail to generalize. Supports size as a practical
  obstruction in this setup, not a theorem or proof of shortcut-basin causality.
  Production implication to investigate: train small for algorithmic behavior,
  then transplant/distill; neither transplant nor production readiness is tested.
- **ARCHITECTURE-BINDING:** small-ours also fail while the X26 nonlinear RNN
  learns/generalizes (with DeltaNet's predicted OOD degradation as a separate
  control). Favors architecture-class differences over size alone (DPLR/nonlinear
  core vs ours), qualified by Concession19 and other X26 adaptations. No absolute
  architecture-incapacity conclusion from this one-seed recipe.
- **NOTHING-LEARNS:** small-ours and X26 small-theirs both fail on these reviewed
  generators. Task/encoding implementation becomes suspect, not proven wrong:
  the native-tokenization control and field-by-field source/generator audit fire.
  That control requires a NEW registration/approval, never an automatic launch.
- If large-ours also learn, small-only obstruction is unsupported. If only one
  activation or architecture succeeds, report that cell rather than generalizing
  across all ours. Software/nonfinite errors are not scientific failures; missing
  bins, infrastructure interruptions, wall/disk/OOM are right-censored.

Adjudication requires complete comparable trajectories, not current snapshots.
One seed is exploratory. No extra seeds/tasks/recipe retries are authorized.

## Unchanged X25 instrument

Use the actual X23E reviewed graph generator, Def11 typed unary serialization,
final-only binary BCE on two-logit margin, balanced immutable compact pools:
train70000, validation20000, test10000 in N1–100, OOD10000 each N101–200/201–300;
split seeds23001–23005. Link the actual shared datasets directory; no regeneration
or distribution changes. Bind existing pool receipts; missing test pools are
materialized only by the frozen generator at terminal evaluation, exactly X25.

Ordinary AdamW constant lr1e-4/wd1e-4, betas(.9,.999), eps1e-8, batch64,
clip1, no warmup/cosine. Identical X25 sample draws (rng23000), length-aware
microbatches and budget8192, bf16 autocast, initialization0, 60000-update cap.
Validation at0/every500; stop after three consecutive perfect ID validations.
Select best validation accuracy, tie-break validation BCE; tests only afterward.
No curriculum/auxiliary supervision added. No test-based choice or tuning.

## Smokes, launch order, standard chain

New zero-update max-length training forward/backward plus eval smokes for EACH
small arm (N300 worst unary path,90904 tokens); big-arm smokes are not reused.
X24 fused-kernel qualification remains SHA-bound, new shape smokes required.
Source/dependency/registration bytes must match worktree AND HEAD; registration
commit must match pushed upstream before launcher starts. Disk guard >=400GiB
free, <=40GiB/rung, wall<=7days; one validation-best checkpoint plus atomic staging.
Independent60s persistent watcher, identity-verified owned process groups only,
TERM then KILL after30s, hash-checked result/terminal/marker receipts.

Start tanh on physical card6. Softplus queued for freed X25 card0; GDN2 queued
for freed X25 card1. Only idle/unleased cards qualify (`gpu_lease.sh list-free`),
then atomic shell lease acquisition; no preemption or overlapping allocation.
All smokes run sequentially under a card6 shell lease before registration/launch.
Physical UUID snapshot, worker/trainer/watcher PID/starttime/host, exact commands,
lease, config/count/data/source hashes are receipted. Watcher wiring exposes
status/terminal paths and --once probe; native notification arming is supervisor
owned, not claimed here. IMM smalls are optional future work, NOT launched or
implicitly authorized in this graph-only registration.

Local lambda01 .venv Python3.12/CUDA research, not Frontier or resilient DiLoCo.
R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 do not apply; no Slurm Partition/QOS
or resilient conformance claimed. Independent acceptance review required.
