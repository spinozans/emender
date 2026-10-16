# Stage 2 proposal — NOT LAUNCHED, requires explicit operator approval

This is a designed experiment, not authorization to run it. Stage 1 performed
no training, GPU inspection, GPU allocation, or scheduler submission. The
production recommendation is p=61, with arithmetic pilots p=13 and p=31.
See `INVERTIBLE_QUADRATIC.md` for task semantics, the proof, and framing limits.

## Question and arms

Question: at practical state sizes, does nonlinear recurrence learn this
invertible computation more efficiently? Not: can a sufficiently large
linear-state automaton represent it (it can, via one-hot permutation matrices).
No CMA-causal claim about GDN2's previous failure is established or assumed.

Designed arms, with the SAME **18,269 optimizer-update budget** per trained arm,
per task, per seed (no validation-based early termination):

1. **E97, state tanh**, with the standard split-edit geometry.
2. **GDN2**, including its established negative-eigenvalue configuration.
3. **Mechanism ablation:** identical E97 geometry, initialization scheme,
   split-edit/write mechanism, readout, precision, optimizer, and data, with
   **only the recurrent-state tanh replaced by identity**. No compensating
   width, activation, or learning-rate changes in this contrast.
4. **Suffix baseline:** nontrained calibration, suffix widths 0/1/7, scored
   only where >=2 drivers are omitted. Also show the old parity+suffix transfer
   as a labeled empirical control, not as a theorem for all prefix summaries.
5. **One-step diagnostic:** train arms 1–3 on the corresponding `_one_step`
   task, also with 18,269 matched updates. This separates arithmetic/encoding
   failure from recurrent-depth failure. A model failing this control does not
   support a depth-specific conclusion.

Proposed common geometry follows the old decisive dim1024 study: dim1024,
depth2, 8 heads, n_state32, MLP ratio4.0, state-summary augmentation disabled.
Use the same geometry for E97 and its identity ablation. GDN2 retains its own
valid layer configuration; report actual trainable parameter counts and
recurrent-state sizes, including the identical 3,721-class input/output heads.
Equal width is **not** a claim of equal parameter count. Report wall time,
tokens/s and total training tokens as well as update count. Any extra
parameter-matched sweep needs a separately recorded configuration, not an
unreported architecture change. Proposed common seeds: 0,1,2, paired across arms.

## Data, depth, and metrics

- Training recurrence depth 128 drivers plus two initialization tokens
  (`seq_len=130`); full-field drivers and full-state running supervision.
- Primary held-out depth panels: 128 -> 512 -> 1024 drivers (1x/4x/8x),
  corresponding to 130/514/1026 total tokens. Optional secondary panels at
  256/2048/4096 drivers mirror the old benchmark's longer grid.
- One-step diagnostic: one driver, `seq_len=3`, only its joint-state target.
- Proposed fixed split protocol: 128 training sequences, 512 validation
  sequences, and 512 **separate test** sequences at training depth, with
  independent frozen test pools for each extrapolation length. Pair task
  streams/splits across arms; deduplicate by the full input including x0,y0.
  These are proposed sizes, not a claim that 128 one-step examples suffice.
  If arithmetic calibration fails, report that failure rather than attributing
  the long-task failure to depth or silently expanding the data budget.
- Proposed common AdamW settings from the old decisive protocol: lr=1e-3,
  weight_decay=1.0, batch64, gradient clipping1.0. Freeze these before launch;
  any tuning must have symmetric search budgets and an audit trail. None has
  been validated for the new task in Stage 1.
- Primary metric: exact full-state accuracy (argmax joint class p*x+y), not
  average coordinate accuracy. Also report loss, per-coordinate accuracy,
  and terminal-position accuracy to expose partial arithmetic progress.
- Show running full-state accuracy and post-burn-in suffix-calibration accuracy
  separately. For width seven, the suffix lemma first applies at driver t=9.
  Use independent trajectories for empirical-cap error bars, not overlapping
  windows treated as independent samples.

## Checkpoint selection and matched-step reporting (mandatory for BOTH arms)

- Validate at common steps every 250 updates and at update 18,269. Save a
  reconstructable best-validation checkpoint for **E97 AND GDN2**, and also
  the identity ablation. Never substitute a terminal GDN2 checkpoint for its
  best-validation checkpoint while selecting E97's best.
- Selection rule fixed before training: maximize exact joint-state validation
  accuracy at training depth; tie-break by lower validation CE, then earlier
  update. Do not select on held-out extrapolation test results. Freeze the
  selected checkpoint identifiers/steps before evaluating those test panels.
- Retain terminal checkpoints at exactly 18,269 updates for all trained arms.
  Record actual completed optimizer updates (not zero-based loop indices),
  hashes, seed, configuration, task/data version and selected step.
- Save and evaluate **matched-step** checkpoints at updates
  1,000 / 4,000 / 8,000 / 12,000 / 16,000 / 18,269, with the same held-out
  panels and observation counts for every arm. Report these curves alongside
  terminal and best-validation results, including selected-step differences.
- Publish all seeds, per-seed paired contrasts and aggregate uncertainty; do
  not hide nonlearners or label a best-of-seed result as typical performance.
  A selected-vs-terminal improvement alone is not a mechanism result.

## Prelaunch implementation/approval gate

The existing old `train_grok.py` is **not** an approved Stage-2 runner unchanged:
it uses its historical test split for selection and does not yet implement this
complete best-validation/terminal/matched-step artifact contract for both arms.
After explicit approval, first adapt and CPU-test checkpoint/data/metric
plumbing, verify the tanh/identity contrast really differs only at the state
activation, and audit the common update counters and validation selection for
both E97 and GDN2. Preserve task targets as labels only, never model inputs.
Then arrange any GPU use under the operator's resource policy. Do not launch
old orchestrators as a shortcut to this plan. Scheduler/resource details and
any Frontier queue binding must follow the repository's environment/Partition
AND QOS evidence rules if such a launch is subsequently approved.

Stage-2 entry requires operator review of `X1-report.md`, the exact inverse/
no-collision tests, known-initial-state suffix measurements, and this plan.
Small-prime learned pilots and one-step learned diagnostics come first; a
negative arithmetic control blocks a depth-specific interpretation, not its
publication. No result, checkpoint, GPU run, or learned pilot is claimed here.
