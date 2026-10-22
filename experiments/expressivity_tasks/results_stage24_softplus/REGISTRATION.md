# X24 — E97 plain-softplus state arm (pre-training amendment)

Operator-directed third arm of X23E; card 7 only; fresh initialization seed 0.
This registration and all gates are committed and pushed before any optimizer update.
Existing X23E tanh/GDN results were not used to choose a new recipe or task.

## Corrections — Concession #18

The supplied claim that `softplus_c(x)=0.5*softplus(2x)-0.5*log(2)` is
unbounded in both directions was wrong: it is bounded below at -log(2)/2.
Counting requires an unbounded-above state, not an unbounded-below state.
Plain softplus is bounded below at zero and unbounded above. Supervisor explicitly
selected **plain `softplus`**, not centered softplus or ReLU, after this audit.

## Supported activation audit and authorized engine amendment

`E88FLAHybrid` accepts tanh, identity (linear/affine aliases), ReLU, softplus;
softplus_c is NOT accepted. `ndm/triton/gdn2_nonlin_fused.py` PHI codes are
identity0/tanh1/relu2/softplus_c3/softplus4, but that is a DIFFERENT engine.
Before this amendment the E97 split-edit fused engine supported only tanh/linear;
E88 softplus/ReLU existed only in the eager FP32 reference and rejected Triton.
The supervisor rejected eager FP32 at 90,904 tokens as infeasible/confounded and
explicitly authorized extending the shared E88/E97 fused engine with plain softplus.
No claim that the pre-existing GDN PHI table made E97 softplus operational.

Forward and sparse-checkpoint replay use the stable form
`max(pre,0)+log1p(exp(-abs(pre)))`. Backward uses the matching identity
`sigmoid(pre)=-expm1(-softplus(pre))` on the replayed state. The switch is
compile-time, defaults off, and reaches direct, projection-chunk and CUDA-graph
split-edit calls. Tanh/linear defaults and model parameter schema are unchanged.
Runtime logs explicitly name softplus; no silent tanh fallback.

**Precision is exactly X23E `legacy`**, not a new fp32 policy: FP32 live state
accumulation, bf16 backward replay checkpoints/scratch, bf16 autocast compute,
FP32 parameters. Legacy training final states (and projection-chunk boundaries,
when enabled) are also bf16; evaluation carry is FP32. X23E/X24 production
projection_chunk_size=0, so no forward per-token state narrowing is introduced.
The supervisor clarified that FP32-everywhere would introduce
a second variable; it is NOT enabled. The full-layer eager FP32 oracle therefore
has a declared projection/replay numerical tolerance, not bitwise equality.
The finite-precision forward counting budget is not proof of an exact unbounded
realization; O10 realization remains open even after empirical success.

## Exact reused protocol, shape and selection

Normative task/recipe: `../results_stage23e_separators/REGISTRATION.md` and
`registration.json`; those identities are bound in this registration.
ONLY the sorted deterministic graph-connectivity Definition11 arm is run.
Unary encoding, vertices0..N, same bucket-path generator, final causal prediction,
same class balance, same bins/seeds/pools. `datasets` links to the ACTUAL X23E
shared dataset directory with its existing locks/hash receipts; no reimplementation
or independent generated pool. Existing pools necessarily predate this amendment;
no claim they were first generated after X24 registration.

`run_e97_softplus_graph.py` delegates model construction, smoke and the entire
training/evaluation loop to X23E, changing only the state activation/binding flag.
Stock dim256/depth4/MLP2/heads32/state32/expansion1, raw_write=False, fused training
and fused inference, 8,950,272 parameters. All weights at seed0 equal stock X23E.
Ordinary AdamW(.9,.999)/eps1e-8/wd.01, effective batch64, clip1, linear warmup5000
to .001 then cosine to .00001 at60000. Eval every500, same three-perfect-validation
stop; validation-only checkpoint choice, tests only after selection. Same 70000
train/20000 validation, 10000 each test N1–100/101–200/201–300. No truncation,
new architecture, curriculum, teacher forcing, post-outcome recipe choice or retry.

## Predictions locked before training

1. **activation-binding**: E97-softplus reaches >=99% ID and both OOD bins where
   E97-tanh degrades (>=5pp ID-to-far-OOD drop). This supports state nonlinearity
   as the practical binding constraint and the X22T candidate-host hypothesis;
   receipts motivate the mixed-ladder non-squashing-state production escape route,
   but do not constitute a production qualification or discharge O10.
2. **not-the-activation**: E97-softplus ALSO degrades OOD (same >=5pp criterion),
   despite learning ID. The binding constraint is elsewhere (wrapper/training/depth);
   O10 stays open; tanh boundedness was not the practical bottleneck in this setup.
3. **fails-to-train**: non-squashing training instability/nonfinite logits or
   gradients (Weiss caveat) is recorded honestly, not relabeled an expressivity
   theorem. X16E staging is the indicated remedy for a separately registered retry,
   not authorization to retry this arm. Raw software error receipts are retained;
   the scientific instability classification requires inspection of the exact error.

Every outcome is informative; ID failure is separate from OOD degradation. If tanh
also generalizes or outcomes lie between thresholds, report exact counts/losses,
not a forced category. One seed is exploratory. Incomplete bins, OOM, disk/wall or
external interruption are right-censored, NOT fails-to-train evidence. No bounded
experiment proves/disproves asymptotic expressivity or exact counter realization.

## Pre-update launch gates and operational chain

- Existing eager FP32 softplus full layer is the oracle, same weights and input;
  short-sequence fused output/final-state max-relative error <.03, input/parameter
  gradient max-relative error <.08, fused eval error <.03. These numerical budgets
  reflect bf16 compute/replay vs FP32 oracle; they do not permit a different formula.
  `receipts/kernel_match.json` binds measured values, seed and zero updates.
- Independent fused recurrence FP32 forward/backward oracle tolerances 1e-5/2e-4;
  legacy forward/backward tolerances .005/.06. Cover gate on/off, resets, invalid
  tokens, multiple checkpoint segments, stable extremes, unaligned forward-only
  evaluation and full-layer projection chunks. Existing tanh/linear regressions pass.
- Fresh max-bin smoke: N300/90,904 tokens forward/backward + fused evaluation,
  finite gradients, **zero optimizer updates**, receipted peak allocation.
- Registration, code, tests, gates, dependency identities bound by SHA256 and
  committed+pushed; trainer independently checks worktree AND HEAD blobs.
- Standalone shell GPU lease restricted to physical card7, no WorksGood service.
  Watcher persists separately, push-gates launch, writes PID/starttime/host, GPU
  snapshot, command/lease/trainer and hash-validated terminal/marker receipts.
- Prelaunch and training disk guard >=400GiB free; per-rung <=40GiB; max7days.
  Independent60s watcher enforces owned identity-verified process-group TERM/KILL
  (30s escalation). Retain only best checkpoint + atomic staging. Infra stops are
  censored; unexpected software errors have no automatic scientific verdict.
- `receipts/watcher_wiring.json` exposes exact status/terminal paths and --once
  probe for supervisor native arming. Persistent resource guard does not wait for
  native arming, and no native notification is claimed unless supervisor arms it.

This is single-host standalone research, not resilient DiLoCo. R01–R16,
NDP01–NDP17, V21S01–V21S17 and ISP01–ISP07 conformance is not claimed/applicable;
no resilient-training behavior, Frontier scheduler binding or scale gate changes.
