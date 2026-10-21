# X16E launch report

Preregistration **10bfb5fa** committed AND pushed before launch. Original X16,
replay/restart registrations and evidence unchanged. This separate arm explicitly
extends the three-perfect terminal teach state despite missing the original
four-position gate; it does not claim that gate was met.

## Hash-bound source and actual baseline

Checkpoint `../infra_restart/e97_curriculum_seed0/T16/step_500000.pt` SHA256
`870c2dcf5e863dcab817e3c6b0150d054e91cff8456d231378d3748676e06b1f` verified
against the terminal checkpoint/result receipt before registration and launch.
Model, AdamW moments and NumPy/torch/CUDA RNG restored and exactly compared.
Actual T16 terminal and restored step0 counts begin [512,512,512,5,0,1,0,0];
task shorthand P4=8 was not substituted for observed5. T64 zero-shot baseline
begins [512,512,512,9,0,0,0,0]; material baseline positions are1,2,3.
Full step0 counts/checkpoint hash: `receipts/baseline.json`.

Both readings of the teach restart divergence at252204 remain registered:
possible harm to late-budding P4 versus possible irrelevance/stalling. Neither
causal claim is established, including comparison to X10's same-seed four-perfect.

## Running recipe and bindings

Fresh T64 phase_step0 baseline, updates1..500000 T64 only, uniform CE, preserved
AdamW moments/RNG, fresh warmup-free cosine3e-4 to3e-6. Independent frozen
`retention_t16` at every500-update T64 validation, including baseline/final.
All three predictions TRANSFER / NO-TRANSFER / RETENTION DAMAGE registered before
launch in `AMENDMENT.md` and `registration.json`. Runtime zero-shot transfer must
not be relabeled as new LEARNING acquisition: new positions beyond baseline
reported separately. Exact trajectories and final versus transient retention
are retained; no early prediction call.

- Physical card6 UUID `GPU-6d8d6161-e8e8-0b81-9abb-1a8f7d15e65d`.
- Leased worker **2600934**, trainer **2601474**, native detached backup watcher
  **2601265**. Identity/starttime and argv receipts under `receipts/`.
- Parent CONFIRMED authoritative native watcher **proc_5d97**, PID **2602992**,
  `scripts.invertible_stage16_extension watch --interval 900`; first poll RUNNING.
  Its exit wakes the attended supervisor session natively. Confirmation and
  independently checked PID/starttime: `receipts/supervisor_arming.json`.
- Trainer/GPU UUID association: `receipts/compute_apps_after.txt`; GPU index/UUID
  map: `receipts/gpu_after.txt`. Other cards untouched by this arm; disposable
  GPU preflight also leased card6 only.
- Static launch snapshot reached extension update2000 validation; T64 first5
  [512,510,503,4,1], T16 retention first5 [512,510,508,3,1]. P3 retention was506
  at500 (below508), then512 at1000: transient damage already observed, NOT a
  fixed-final outcome. This is running evidence, not successful completion.

## Monitoring, receipts and censoring

`GROWTH_REPORT.md` regenerated each wake, reporting every position's latest/max
and first50/508/512, baseline/new positions, and all16 retention drops/minima.
Exact per-validation trajectories: `e97_extension_seed0/validation.json`.
One-shot: `.venv/bin/python -m scripts.invertible_stage16_extension watch --once`
(exit2 pending,0 hash-verified terminal,1 censored/error). Background interval900s.
Terminal contract: matching DONE/terminal receipts, final result hash,
500000 T64 updates and all64 plus16 retention positions. FAIL is infrastructure
censoring, not architecture failure. No automatic restart/retry. Fixed-final500K
primary; transients descriptive, missing crossings right-censored at observed step.
14-day/16GiB bounds,200GiB root-free guard, atomic checkpoints, extension
entry+best+latest <=3 with hash deletion receipts; source checkpoints untouched.

## Validation and review

- CPU tests:27 passed,2 GPU deselected, including5 new X16E tests covering fresh
  cosine, full mocked lifecycle, preserved optimizer step/moments, T64-only
  inputs, fresh phase clock, retention every evaluation, checkpoint bounds,
  terminal tamper, exact-checkpoint refusal and no lease operation on bad hashes.
- Leased card6 seed99 disposable GPU test:1 passed,17 deselected; exercises actual
  E97 T16/T64/T16 updates and depth-safe evaluation, no confirmatory pool used.
- Full inherited and new worktree/committed-blob source/dependency hashes,
  source artifact hashes and terminal checkpoint reverified after launch.
- Worker, trainer, detached watcher and parent-native watcher identities live.
- Staged diff whitespace check passed before preregistration commit.
- No WorksGood service/dispatcher, Frontier/Slurm or resilient DiLoCo behavior.
  DiLoCo requirement IDs not applicable to this standalone expressivity arm.
- Parent-owned reviewer gate still required; launch evidence is not a completed
  scientific result or an independent review acceptance.

Single seed, no multi-seed/causal/system-scale claim. Continue monitoring to the
registered terminal gate; inspect (a)/(b)/(c), baseline versus new acquisition,
exact retention trajectories, and infrastructure receipts on wake.
