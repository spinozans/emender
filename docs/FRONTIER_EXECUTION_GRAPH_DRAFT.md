# Frontier Execution Graph Draft

Task: `draft-frontier-execution`  
Date: 2026-06-21  
Inputs:

- `docs/FRONTIER_KICKOFF_SYNTHESIS.md`
- `docs/FRONTIER_ROCM_READINESS_INVENTORY.md`

This is a proposed WG task graph for the next Frontier phase. It is intended to
be reviewed and then converted into `wg add` tasks. It deliberately does not
create the downstream implementation batch.

## Ground Rules

Every recurrence/state-dynamics experiment in this graph is fused-only:

- The recurrence must run through the fused Triton kernel.
- Each run must emit the per-rank `[fused-guard] ... NO eager fallback` line.
- Each train smoke must exercise the matching fused backward/VJP path, not only
  a forward import or compile path.
- Eager or pure-Python recurrence is rejected for parity checks, prototypes,
  debug shortcuts, sanity checks, fallbacks, and preliminary signal checks.
- If a desired probe cannot be run with fused Triton recurrence today, the graph
  marks it as an implementation prerequisite rather than an experiment.

The graph starts with ROCm debug-queue evidence before any extended allocation
spend. Prior no-go framing is not inherited as fact: E97-MLP, GDN2-MLP,
E97-linear-MLP, and DiLoCo each get explicit decision gates.

Default scaleout hypothesis: horizontal single-GPU or single-GCD islands first,
because local evidence favors independent islands over no-NVLink synchronous
DDP. DDP/RCCL remains a measured risk and comparison point, not an assumed
blocker.

## Graph Overview

```text
frontier-00-env-rocm-smoke-spec
  -> frontier-01-impl-frontier-slurm-smokes
      -> frontier-02-debug-import-compile-smoke
          -> frontier-03-debug-e97-mlp-fused-smoke \
          -> frontier-04-debug-gdn2-mlp-fused-smoke  > frontier-06-debug-checkpoint-resume-eval-smoke
          -> frontier-05-debug-e97-linear-mlp-fused-smoke /
              optional -> frontier-05a-impl-chunked-e97-linear-control
frontier-06-debug-checkpoint-resume-eval-smoke
  -> frontier-07-debug-two-rank-rccl-fused-smoke
      -> frontier-08-throughput-sanity-and-budget-refresh
          -> frontier-09-single-gcd-island-probes \
          -> frontier-10-ddp-risk-probes           > frontier-11-diloco-scaleout-probes
frontier-11-diloco-scaleout-probes
  -> frontier-12-extended-run-readiness-review
      -> frontier-13-extended-8-node-24h
          -> frontier-14-extended-16-node-24h
              -> frontier-15-extended-32-node-24h
                  -> frontier-16-extended-64-node-24h
                      -> frontier-17-synthesis-and-next-decision
```

Same-file work is serialized. Parallel tasks below are only parallel where file
scopes do not overlap, and expensive Frontier jobs are gated on earlier
debug-queue fused-guard artifacts.

## Proposed WG Tasks

### frontier-00-env-rocm-smoke-spec

Title: Draft Frontier ROCm smoke command matrix and artifact contract

Depends on: `draft-frontier-execution` after review.

File scope:

- `docs/frontier/ROCM_SMOKE_COMMAND_MATRIX.md`

Expected artifacts:

- Command matrix for import, E97-MLP, GDN2-MLP, E97-linear-MLP, checkpoint
  resume/eval, two-rank RCCL, and throughput sanity smokes.
- Required environment variables, Slurm flags, output directory convention, and
  log filenames.
- Artifact contract naming the exact files each smoke must produce.

Validation checklist:

- Includes debug queue Slurm shape: `-p batch`, `-q debug`, max `02:00:00`.
- Requires `NDM_PIN_TRITON_AUTOTUNE=1` and `NDM_PIN_TRITON_VERBOSE=1`.
- Requires `--bf16`, `--use_triton 1` where applicable, and fused-guard log
  capture for every recurrence/state-dynamics smoke.
- Explicitly rejects eager/Python recurrence validation.
- Does not modify training code or upstream synthesis/inventory artifacts.

Resource estimate: docs-only, 0 node-hours.

Decision gate:

- Pass opens implementation of launch wrappers.
- Fail means the command contract is underspecified; do not run Frontier jobs.

### frontier-01-impl-frontier-slurm-smokes

Title: Implement Frontier Slurm smoke launch wrappers

Depends on: `frontier-00-env-rocm-smoke-spec`.

File scope:

- `scripts/frontier/`
- Optional generated examples under `docs/frontier/examples/`

Expected artifacts:

- Slurm wrapper scripts for debug import/compile, E97-MLP, GDN2-MLP,
  E97-linear-MLP, checkpoint resume/eval, two-rank RCCL, and throughput sanity.
- Wrapper README explaining required placeholders: account, repo path, data
  path, output path, `GDN2_PATH`, and container/module setup.
- No changes to `train.py` unless a later implementation-prerequisite task owns
  that file explicitly.

Validation checklist:

- Shell syntax checks pass.
- Dry-run or render mode prints the final Slurm script without submitting.
- Every training wrapper includes fused-guard log grep in post-run validation.
- Every recurrence wrapper fails closed if the fused guard is absent.
- No wrapper offers eager/Python recurrence fallback.

Resource estimate: local only, 0 node-hours.

Decision gate:

- Pass allows the first Frontier debug import job.
- Fail blocks all debug jobs because artifacts would be inconsistent.

### frontier-02-debug-import-compile-smoke

Title: Frontier debug import/compile smoke for ROCm PyTorch, Triton, and GDN2

Depends on: `frontier-01-impl-frontier-slurm-smokes`.

File scope:

- `runs/frontier/debug/import_compile/<timestamp>/`
- `docs/frontier/debug/import_compile_<timestamp>.md`

Expected artifacts:

- Slurm stdout/stderr with `torch`, `triton`, ROCm availability, local E97
  Triton imports, chunked E97 imports, multiquery E97 imports, and external GDN2
  wrapper imports.
- Environment manifest including container/module names, `GDN2_PATH`, repo
  commit, and Slurm job ID.

Validation checklist:

- Import succeeds in the Frontier job environment.
- External GDN2 and FLA path are present.
- `NDM_PIN_TRITON_VERBOSE=1` output is captured if kernels compile.
- The result is labeled "environment/import only", not model correctness.
- No eager/Python recurrence validation is performed or proposed.

Resource estimate: 1 node x 0.33 h = 0.33 node-hours.

Decision gate:

- Pass unlocks single-rank train/backward smokes.
- Fail creates an environment repair prerequisite; no model smoke should run.

### frontier-03-debug-e97-mlp-fused-smoke

Title: Frontier debug E97-MLP fused train/backward smoke

Depends on: `frontier-02-debug-import-compile-smoke`.

File scope:

- `runs/frontier/debug/e97_mlp/<timestamp>/`
- `docs/frontier/debug/e97_mlp_<timestamp>.md`

Expected artifacts:

- Slurm logs.
- Training args/config.
- Fused-guard excerpt.
- Finite loss/grad evidence for at least one optimizer step and backward pass.
- Checkpoint if the smoke uses `--save_every`.

Validation checklist:

- Runs `--level E97`, `--mlp_ratio > 0`, `--bf16`, `--use_triton 1`.
- Log contains `[fused-guard] rank 0/1: level=E97 ... fused split-edit Triton
  kernel, NO eager fallback`.
- The fused backward/VJP completes; forward-only success is insufficient.
- Loss and gradients are finite.
- Any missing fused guard, fallback recurrence, NaN, Triton compile failure, or
  missing artifact is a fail.

Resource estimate: 1 node x 0.67 h = 0.67 node-hours.

Decision gate for E97-MLP:

- Pass: E97-MLP can enter checkpoint/resume and throughput sanity.
- Fail due to environment or ROCm codegen: create a fused-kernel port/fix task.
- Fail due to model nonfinite behavior: triage under fused ROCm evidence only;
  do not re-run in eager.

### frontier-04-debug-gdn2-mlp-fused-smoke

Title: Frontier debug GDN2-MLP fused external-kernel smoke

Depends on: `frontier-02-debug-import-compile-smoke`.

File scope:

- `runs/frontier/debug/gdn2_mlp/<timestamp>/`
- `docs/frontier/debug/gdn2_mlp_<timestamp>.md`

Expected artifacts:

- Slurm logs.
- Training args/config.
- Fused-guard excerpt.
- External GDN2/FLA import and kernel log excerpts.
- Finite loss/grad evidence and checkpoint if configured.

Validation checklist:

- Runs `--level gdn2-mlp`, `--bf16`, and the intended external `GDN2_PATH`.
- Log contains `[fused-guard] ... FLA chunked GDN-2 fused kernel, NO eager
  fallback`.
- At least one fused train/backward optimizer step completes with finite loss.
- No fallback recurrence is accepted.
- Third-party FLA import success alone is not enough; the train path must run.

Resource estimate: 1 node x 0.83 h = 0.83 node-hours.

Decision gate for GDN2-MLP:

- Pass: GDN2-MLP can enter checkpoint/resume and throughput sanity.
- Fail due to missing external stack: create an environment/GDN2 checkout task.
- Fail due to ROCm fused kernel failure: create an external GDN2/FLA port task.

### frontier-05-debug-e97-linear-mlp-fused-smoke

Title: Frontier debug E97-linear-MLP fused control smoke

Depends on: `frontier-03-debug-e97-mlp-fused-smoke`.

File scope:

- `runs/frontier/debug/e97_linear_mlp/<timestamp>/`
- `docs/frontier/debug/e97_linear_mlp_<timestamp>.md`

Expected artifacts:

- Slurm logs.
- Training args/config.
- Fused-guard excerpt.
- Evidence identifying whether the run used the sequential E88 Triton path or a
  chunked single-query linear-state path.
- Finite loss/grad evidence and checkpoint if configured.

Validation checklist:

- Runs `--level E97 --linear_state 1 --mlp_ratio > 0 --bf16 --use_triton 1`.
- Log contains the E97 fused guard and `NO eager fallback`.
- Matching fused backward/VJP completes.
- The report explicitly states that current plain `--linear_state 1` is a
  fused linear-state control, not proof of chunked E97 throughput, unless source
  and logs show `e97_delta_chunked_triton`.
- No eager/Python recurrence validation is accepted.

Resource estimate: 1 node x 0.67 h = 0.67 node-hours.

Decision gate for E97-linear-MLP:

- Pass as sequential fused control: keep it as the causal nonlinearity ablation.
- Need chunked fair-throughput control: create `frontier-05a` below before using
  E97-linear as a GDN2-class throughput comparison.
- Fail: create a fused linear-state E97 triage task; do not substitute eager.

### frontier-05a-impl-chunked-e97-linear-control

Title: Implement explicit chunked single-query E97-linear-MLP control

Depends on: `frontier-05-debug-e97-linear-mlp-fused-smoke`, only if the
decision gate requires a chunked control.

File scope:

- `train.py`
- `ndm/models/ladder_lm.py`
- `ndm/models/e88_fla_hybrid.py`
- Focused tests under `tests/`

Expected artifacts:

- A command surface for `E88FLAHybrid(use_split_edit=True, use_triton=True,
  use_chunked_e97=True, linear_state=True, multiquery_r=1)` or equivalent.
- Fused guard/logging that distinguishes sequential E97-linear from chunked
  single-query E97-linear.
- Focused tests proving the new route selects the fused chunked recurrence.

Validation checklist:

- A failing test or route-selection assertion is added before implementation.
- Tests pass for the new route.
- The new route rejects or fails closed on non-fused recurrence.
- Matching fused backward/VJP support is present; if not present, the task fails
  and the control is not an experiment.
- No pure-Python recurrence is added as a validation path.

Resource estimate: local implementation, 0 Frontier node-hours.

Decision gate:

- Pass unlocks a follow-up chunked E97-linear debug smoke.
- Fail keeps the existing sequential fused E97-linear control only.

### frontier-06-debug-checkpoint-resume-eval-smoke

Title: Frontier debug checkpoint, resume, and held-out eval smoke

Depends on:

- `frontier-03-debug-e97-mlp-fused-smoke`
- `frontier-04-debug-gdn2-mlp-fused-smoke`
- `frontier-05-debug-e97-linear-mlp-fused-smoke`

File scope:

- `runs/frontier/debug/checkpoint_resume/<timestamp>/`
- `docs/frontier/debug/checkpoint_resume_<timestamp>.md`

Expected artifacts:

- For each arm: initial short train log, checkpoint, resume log, eval log, and
  config.
- Filesystem note: Orion vs `/mnt/bb/$USER` staging choice and observed write
  behavior.

Validation checklist:

- Each resumed recurrence run again emits the fused guard.
- Checkpoint load resumes from the expected step.
- Eval does not change recurrence mode or suppress the fused guard where the
  model path executes recurrence.
- Held-out eval produces finite metrics.
- No run is accepted if checkpointing works only after disabling fused kernels.

Resource estimate: 3 arms x 1 node x 0.75 h = 2.25 node-hours.

Decision gate:

- Pass unlocks distributed and throughput probes.
- Fail creates a checkpoint/resume implementation task before any longer job.

### frontier-07-debug-two-rank-rccl-fused-smoke

Title: Frontier debug two-rank RCCL/DDP fused smoke

Depends on: `frontier-06-debug-checkpoint-resume-eval-smoke`.

File scope:

- `runs/frontier/debug/rccl_2rank/<timestamp>/`
- `docs/frontier/debug/rccl_2rank_<timestamp>.md`

Expected artifacts:

- Two-rank logs for E97-MLP and, if E97 passes cleanly, GDN2-MLP.
- Per-rank fused-guard excerpts.
- Backend, rank/device binding, and output-writer behavior notes.

Validation checklist:

- Every rank prints the fused guard with `NO eager fallback`.
- DDP/RCCL initializes and completes at least two optimizer steps.
- Rank 0 is the only checkpoint writer unless the implementation intentionally
  uses sharded checkpointing.
- Failure is reported as measured RCCL/DDP risk; it is not interpreted as an
  architecture no-go.

Resource estimate: 1 node x 0.75 h x 2 arms = 1.5 node-hours.

Decision gate:

- Pass: DDP can be included as a measured comparison.
- Fail: single-GCD island strategy remains the default, and a DDP/RCCL triage
  task is created before DDP-dependent scale jobs.

### frontier-08-throughput-sanity-and-budget-refresh

Title: Frontier fused throughput sanity and budget refresh

Depends on: `frontier-07-debug-two-rank-rccl-fused-smoke`.

File scope:

- `runs/frontier/debug/throughput_sanity/<timestamp>/`
- `docs/frontier/FRONTIER_BUDGET_REFRESH_<timestamp>.md`

Expected artifacts:

- 20-50 step logs for E97-MLP, GDN2-MLP, and E97-linear-MLP control.
- Tokens/sec, step time, memory, compile/autotune time, and checkpoint write
  time.
- Updated token-per-24h and node-hour burn estimates for the ladder.

Validation checklist:

- Every run emits fused guard and completes fused backward steps.
- The report separates first-compile/autotune time from steady-state step time.
- Any pathological throughput is treated as a systems finding, not a cue to use
  eager recurrence.
- Updated budget table keeps a conservative reserve through 2026-09-01.

Resource estimate: 3 arms x 1 node x 1 h = 3 node-hours.

Decision gate:

- Pass unlocks smallest informative scale probes.
- Fail creates ROCm performance/compile triage tasks before further spend.

### frontier-09-single-gcd-island-probes

Title: Frontier horizontal single-GCD island probes

Depends on: `frontier-08-throughput-sanity-and-budget-refresh`.

File scope:

- `runs/frontier/scale/islands/<timestamp>/`
- `docs/frontier/scale/single_gcd_islands_<timestamp>.md`

Expected artifacts:

- 1-node and 2-node island runs, one process per GCD where feasible.
- Per-island fused-guard logs.
- Throughput scaling, loss traces, checkpoint cadence, and failure notes.

Validation checklist:

- Each island uses fused Triton recurrence and emits its own fused guard.
- No global per-step DDP is required for the default island hypothesis.
- The report compares observed aggregate tokens/sec to single-GCD expectation.
- Nonfinite or missing island outputs are triaged and excluded explicitly, not
  silently averaged.

Resource estimate: 1 node x 2 h + 2 nodes x 2 h = 6 node-hours.

Decision gate:

- Pass: horizontal islands are eligible for DiLoCo probes.
- Fail: create island launcher/fault-isolation tasks before DiLoCo scaleout.

### frontier-10-ddp-risk-probes

Title: Frontier DDP/RCCL risk probes at smallest informative scale

Depends on: `frontier-08-throughput-sanity-and-budget-refresh`.

File scope:

- `runs/frontier/scale/ddp_risk/<timestamp>/`
- `docs/frontier/scale/ddp_risk_<timestamp>.md`

Expected artifacts:

- 1-node full-GCD DDP run if two-rank passed.
- Optional 2-node DDP run only if 1-node DDP is informative and stable.
- Per-rank fused-guard logs, communication timing, and memory telemetry.

Validation checklist:

- Every rank emits fused guard.
- Per-step global DDP is measured for throughput and stability.
- Failure is logged as a communications/system risk, not an architecture
  verdict.
- No extended DDP job is proposed unless this probe provides positive evidence.

Resource estimate: 1 node x 2 h + optional 2 nodes x 2 h = 2-6 node-hours.

Decision gate:

- Pass: DDP remains a comparison option.
- Fail: DDP-dependent ladder edges are blocked; island/DiLoCo remains primary.

### frontier-11-diloco-scaleout-probes

Title: Frontier DiLoCo scaleout probes with fused recurrence islands

Depends on:

- `frontier-09-single-gcd-island-probes`
- `frontier-10-ddp-risk-probes` if DDP is used inside islands

File scope:

- `runs/frontier/diloco/<timestamp>/`
- `docs/frontier/diloco/diloco_scaleout_<timestamp>.md`

What to vary:

- Architecture: E97-MLP, GDN2-MLP, and E97-linear-MLP control where the control
  has passed its fused decision gate.
- Island shape: 1 GCD per island first; 1 node per island only after local
  RCCL/DDP evidence supports it.
- Island count: 8, 16, then 32 islands before any 64-node production shape.
- Local sync interval `K`: start with 100 and 250; test 500 only if recovery is
  stable.
- Outer optimizer: beta 0.0/plain averaging first; beta 0.5 or 0.9 only as a
  new experiment after beta 0.0 is stable.
- Outer LR: 1.0 first; 0.5 only if loss shocks or drift require damping.
- Local batch: per-GCD batch 1 first; batch 2 only after memory and learning
  telemetry are stable.

Telemetry to capture:

- Per-island fused-guard lines.
- Local tokens/sec, aggregate tokens/sec, step time, and sync time.
- Loss before sync, after sync, and over the next recovery window.
- Weight delta norm, drift, gradient norm if available, and nonfinite counters.
- Tokens processed per island for token-weighted averaging.
- Checkpoint wall time, checkpoint size, resume success, and failed-island
  handling.
- Network/RCCL timing if any communication is used inside islands or between
  islands.

Validation checklist:

- Every island runs fused Triton recurrence and emits fused guard.
- Matching backward/VJP is exercised inside each island.
- The run rejects missing or nonfinite island results rather than averaging them
  silently.
- The report compares DiLoCo to matched-token local/DDP baselines where
  available.
- Eager/Python recurrence is never used to diagnose parity or rescue failures.

Resource estimate:

- 8 islands on 1 node x 2 h x 3 settings = 6 node-hours.
- 16 islands on 2 nodes x 3 h x 3 settings = 18 node-hours.
- 32 islands on 4 nodes x 4 h x 2 settings = 32 node-hours.
- Conservative DiLoCo probe envelope: 60 node-hours including reruns.

Decision gate for DiLoCo:

- Success means beta 0.0/plain averaging has stable loss recovery, acceptable
  drift, and aggregate throughput close enough to island scaling to justify
  8/16-node 24h runs.
- Failure means either the island launcher, outer merge, optimizer choice, or
  communication path is not ready. It does not imply E97-MLP or GDN2-MLP is a
  no-go unless the same fused architecture also fails its direct smokes.

### frontier-12-extended-run-readiness-review

Title: Frontier extended-run readiness review and go/no-go matrix

Depends on: `frontier-11-diloco-scaleout-probes`.

File scope:

- `docs/frontier/EXTENDED_RUN_READINESS_<timestamp>.md`

Expected artifacts:

- Gate matrix for E97-MLP, GDN2-MLP, E97-linear-MLP, and DiLoCo.
- Updated node-hour budget and reserve.
- Selected run ladder and stop conditions.
- Explicit statement of which arms are eligible for 8-node and larger runs.

Validation checklist:

- Cites debug fused-guard artifacts for every eligible recurrence arm.
- Includes throughput estimates from Frontier, not inherited planning numbers.
- Rejects arms that lack matching fused backward/VJP evidence.
- Documents unresolved DDP risk separately from island/DiLoCo readiness.
- Keeps a conservative reserve through 2026-09-01.

Resource estimate: docs-only, 0 node-hours.

Decision gate:

- Pass unlocks extended 24h ladder.
- Fail blocks extended jobs until the missing evidence is produced.

### frontier-13-extended-8-node-24h

Title: Frontier 8-node 24h fused extended run

Depends on: `frontier-12-extended-run-readiness-review`.

File scope:

- `runs/frontier/extended/8node_<timestamp>/`
- `docs/frontier/extended/8node_<timestamp>.md`

Expected artifacts:

- 24h logs, checkpoints, resume probe, held-out eval, telemetry summary, and
  allocation accounting.
- Per-rank or per-island fused-guard excerpts preserved in the summary.

Validation checklist:

- Eligible arms only; no arm lacking debug fused-guard evidence may run.
- Checkpoint heartbeat at least every 15-30 minutes to hot storage if enabled,
  and durable checkpoint at least every 2 hours.
- Sparse WG log updates during stable periods and intense updates during
  failures or decision gates.
- Stop if fused guard disappears, recurrence fallback appears, repeated
  nonfinite loss occurs, or checkpointing cannot resume.

Resource estimate: 8 nodes x 24 h = 192 node-hours.

Decision gate:

- Pass unlocks 16-node 24h.
- Fail creates focused triage before rerun or scale-up.

### frontier-14-extended-16-node-24h

Title: Frontier 16-node 24h fused extended run

Depends on: `frontier-13-extended-8-node-24h`.

File scope:

- `runs/frontier/extended/16node_<timestamp>/`
- `docs/frontier/extended/16node_<timestamp>.md`

Expected artifacts and validation checklist:

- Same as 8-node, plus scale efficiency relative to 8-node and DiLoCo sync
  stability if DiLoCo is active.

Resource estimate: 16 nodes x 24 h = 384 node-hours.

Decision gate:

- Pass unlocks 32-node 24h.
- Fail blocks larger jobs until the failure is explained under fused evidence.

### frontier-15-extended-32-node-24h

Title: Frontier 32-node 24h fused extended run

Depends on: `frontier-14-extended-16-node-24h`.

File scope:

- `runs/frontier/extended/32node_<timestamp>/`
- `docs/frontier/extended/32node_<timestamp>.md`

Expected artifacts and validation checklist:

- Same as 16-node, plus communication/network telemetry and allocation burn
  update before any 64-node submission.

Resource estimate: 32 nodes x 24 h = 768 node-hours.

Decision gate:

- Pass unlocks 64-node 24h.
- Fail blocks 64-node jobs unless the readiness review explicitly approves a
  lower-risk rerun shape.

### frontier-16-extended-64-node-24h

Title: Frontier 64-node 24h fused extended run

Depends on: `frontier-15-extended-32-node-24h`.

File scope:

- `runs/frontier/extended/64node_<timestamp>/`
- `docs/frontier/extended/64node_<timestamp>.md`

Expected artifacts:

- Full 24h logs, checkpoints, evals, throughput, DiLoCo or island telemetry,
  allocation accounting, and failure/restart notes.
- If comparing E97-MLP and GDN2-MLP, either pack both arms into one job where
  feasible or document why the scheduler/resource shape required separate
  submissions.

Validation checklist:

- Every recurrence arm has debug, scale-probe, and readiness evidence.
- Every rank/island emits fused guard.
- Checkpoint/resume is demonstrated from the 64-node artifact or an immediately
  preceding equivalent shape.
- The summary separates learning signal, systems throughput, and communication
  reliability.
- No inherited no-go or winner conclusion is accepted without this run's fused
  evidence.

Resource estimate: 64 nodes x 24 h = 1,536 node-hours.

Decision gate:

- Pass feeds architecture and DiLoCo synthesis.
- Fail triggers failure triage and may justify rerun only within the reserved
  node-hour envelope.

### frontier-17-synthesis-and-next-decision

Title: Synthesize Frontier fused evidence and recommend next jobs

Depends on: `frontier-16-extended-64-node-24h`, or the largest converged ladder
step if the review stops earlier.

File scope:

- `docs/frontier/FRONTIER_FUSED_EVIDENCE_SYNTHESIS_<timestamp>.md`

Expected artifacts:

- Evidence table for E97-MLP, GDN2-MLP, E97-linear-MLP, DiLoCo, DDP, and island
  strategy.
- Node-hour spend vs budget.
- Recommended next graph: rerun, scale, paper artifact, or stop.

Validation checklist:

- Every conclusion cites Frontier fused-guard artifacts.
- Negative conclusions distinguish environment, kernel, communication,
  optimizer, and architecture causes.
- DiLoCo conclusions state which `K`, island shape, outer LR, and beta were
  tested.
- The synthesis does not rewrite upstream kickoff synthesis or ROCm inventory.

Resource estimate: docs-only, 0 node-hours.

Decision gate:

- This is the join point for human review before any new fan-out.

## Allocation Budget

Total available budget through 2026-09-01: 20,000 node-hours.

Conservative burn envelope:

| Phase | Proposed spend | Notes |
|---|---:|---|
| Debug queue environment/import and three arm smokes | 6 node-hours | Includes import, E97-MLP, GDN2-MLP, E97-linear-MLP, and small rerun slack. |
| Checkpoint/resume/eval and two-rank RCCL smokes | 5 node-hours | Must pass before longer jobs. |
| Throughput sanity and budget refresh | 5 node-hours | Includes rerun allowance for first-compile/autotune noise. |
| Single-GCD island and DDP risk probes | 20 node-hours | Uses 1/2-node or smallest informative scale. |
| DiLoCo scaleout probes | 60 node-hours | 8/16/32-island probes plus failed-island rerun room. |
| First extended ladder, one selected configuration | 2,880 node-hours | 8 + 16 + 32 + 64 nodes, each 24h. |
| Matched-control extended ladder | 2,880 node-hours | Repeat ladder for the matched control if gates justify it. |
| Focused reruns/failure recovery | 4,000 node-hours | Reserved for failed checkpoints, scheduler interruptions, and one repeated 64-node job per arm. |
| Later scale or comparison reserve | 5,000 node-hours | Held for reviewed follow-up after synthesis. |
| Hard reserve through 2026-09-01 | 5,144 node-hours | Do not allocate without explicit review. |
| **Total** | **20,000 node-hours** | Conservative envelope. |

Rough extended job costs:

- 1 node x 2 h debug job = 2 node-hours.
- 2 nodes x 2 h scale probe = 4 node-hours.
- 8 nodes x 24 h = 192 node-hours.
- 16 nodes x 24 h = 384 node-hours.
- 32 nodes x 24 h = 768 node-hours.
- 64 nodes x 24 h = 1,536 node-hours.
- One 8/16/32/64 ladder = 2,880 node-hours.
- Two matched ladders = 5,760 node-hours.

The readiness review must refresh this table with measured Frontier throughput
before submitting any 24h jobs. The budget should count failed jobs at requested
node-hours unless OLCF accounting proves otherwise.

## Long-Running Coordination Conventions

WG task naming:

- Prefix implementation tasks with `frontier-impl-`.
- Prefix debug jobs with `frontier-debug-`.
- Prefix scale probes with `frontier-scale-`.
- Prefix extended jobs with `frontier-extended-`.
- Prefix synthesis/review tasks with `frontier-review-`.

Artifacts:

- Store run outputs under `runs/frontier/<phase>/<arm_or_probe>/<timestamp>/`.
- Store human-readable summaries under `docs/frontier/<phase>/`.
- Every summary must include commit, Slurm job ID, node count, walltime,
  environment, command, fused-guard excerpts, checkpoint paths, and validation
  result.

Logs and heartbeat cadence:

- Debug jobs: log start, fused-guard observed/missing, first optimizer step,
  checkpoint, and final status.
- Scale/extended jobs: WG heartbeat at job submission, first fused guard, first
  checkpoint, every 2-4 hours while healthy, and immediately on anomaly.
- Checkpoint cadence target: hot checkpoint every 15-30 minutes when available;
  durable checkpoint every 2 hours for long runs.
- Resume probe: every extended phase must demonstrate at least one resume path
  before scaling further.

Failure triage:

- Classify failures as environment, fused kernel compile, fused backward/VJP,
  numerics, data/checkpoint, DDP/RCCL, DiLoCo merge, scheduler/filesystem, or
  architecture signal.
- Do not convert a systems or environment failure into an architecture verdict.
- Do not retry by disabling fused recurrence or using eager/Python recurrence.
- Create repair tasks with file scopes that do not overlap active jobs.

Feedback loop:

- Sparse mode during stable 24h runs: heartbeat summaries only.
- Intense mode at gates/failures: immediate WG log, artifact capture, and a
  short triage task before resubmission.
- Each expensive job requires an explicit prior artifact proving the previous
  gate passed.

## Immediate Next Tasks After Review

Recommended first tasks to publish after human/chat-agent review:

1. `frontier-00-env-rocm-smoke-spec`
2. `frontier-01-impl-frontier-slurm-smokes`
3. `frontier-02-debug-import-compile-smoke`

Do not publish the full downstream batch until the smoke command matrix and
launch wrappers are reviewed. That keeps same-file implementation serialized
and prevents expensive Frontier jobs from starting without debug fused-guard
evidence.
