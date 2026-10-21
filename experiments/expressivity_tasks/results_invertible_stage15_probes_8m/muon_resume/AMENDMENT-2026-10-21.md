# X15b — attended Muon infrastructure-resume amendment, 2026-10-21 UTC

Authority: operator taskX15b-muonresume.md plus explicit supervisor corrections
and replay-evidence ruling in the attended session. This is a supersession for
**Muon only**. Original registration, sources, checkpoints, truncated logs,
GDN2 amendment and results remain immutable; all new outputs live here.

## Interruption and correction #9

Root volume reached 100% at 2026-10-21 20:59 UTC during external /tmp
bioinformatics churn, per operator incident attribution (not independently
established by this resume). Original Muon stream ends in a partial ENOSPC row
for update 433,296, with full rows through 433,295; watcher/worker terminal
publication also hit Errno28. This is infrastructure censoring, not an optimizer
failure, and the one attended continuation below is explicitly authorized.

**Concession #9:** supervisor's resume task text said 500K ceiling and 66,704
remaining updates from memory. The immutable registration is authoritative:
**1,000,000 ceiling, original cosine endpoint 1,000,000**, linear warmup5K,
peak3e-3/floor3e-5. Neither schedule nor ceiling changes. Resume requires 567,000
executed updates from checkpoint433000: 296 diagnostic-verified replay updates
433001–433296, then **566,704 net-new updates** 433297–1000000. The 500K read is
an intermediate descriptive snapshot, not a stop or replacement endpoint.

## Admission and exact restoration

Load only original step_433000.pt, SHA256
`373a7ece3a97790e81d9fa3d0888afdedd74e8eaf8e9b68bbc834c8cf2e882c0`,
checked against original checkpoints.json. Before launch verify nonempty Muon
momentum and auxiliary AdamW state, plus numpy PCG64, torch CPU and CUDA RNG.
At runtime strict-load model and optimizer; compare complete restored model,
optimizer, RNG state tensors/values exactly (storage device excluded). Missing
state/hash mismatch fails closed before any update. Parameter routing, trusted
stage10 loader, fresh procedural batches, T64 pool array hashes, all64 positional
evaluation every500, no metric stop, seed0, batch32, decay and optimizer unchanged.

Replay all296 updates deterministically from checkpoint RNG and compare **exactly**
loss, grad_norm, LR, full_update_rms and muon_update_rms against every readable
original row, including the complete diagnostic fields before the truncated
433296 timing field. Runtime timings are not deterministic diagnostics. **No
per-update RNG receipts or full-state hashes exist in the old stream.** Diagnostic
agreement is replay evidence, **NOT bitwise full-state proof**. The supervisor
explicitly approved this scoped claim. Any mismatch aborts fail closed with
INFRA-INTERRUPTED and original checkpoint preserved: no fresh updates. Never
label approximate agreement as exact, and never silently accept a tolerance.

Provenance: `checkpoint-433000 + verified-diagnostic replay + fresh updates`.
New training.jsonl replay rows say `verified_diagnostic_replay`; rows433297+
say `resumed`. Historical training.jsonl stays unchanged (including partial row).
New validation.json contains original history through433000 followed by new
500-step rows; validation does not run during replay. The original checkpoint
is retained permanently; pruning can remove only newly written resume files.

## New fail-closed root disk guard

Require root `shutil.disk_usage('/').free >= 200*1024**3` (conservative 200 GiB)
before **every update/log write**, before validation, and before each checkpoint,
validation/index and terminal-result write. Missing disk evidence raises, never
passes. On low disk, abort without starting a checkpoint/validation write and
retain the latest atomic, SHA-recorded resumable checkpoint (original433000
before any new checkpoint). Publish small interrupted.json and worker terminal
INFRA-INTERRUPTED/FAIL marker; lease is released. There is no automatic retry.
The guard abort uses the last committed checkpoint, not a new below-threshold
checkpoint write. Existing atomic checkpoint staging never replaces the last
committed checkpoint until serialization succeeds.

A root-free precheck is not a filesystem reservation: concurrent external writes
can exhaust the filesystem between check and write, and catastrophic exhaustion
can prevent even tiny terminal metadata publication. No impossible guarantee
against external consumption is claimed. Atomic staging plus immutable original
checkpoint protects resumability even then; dead-process watcher fails closed.

## Re-registration, launch, wake and verdict

Preserve original registration.json; this directory's registration.json binds
original code plus new amendment, supersession, runner, watcher, tests and
prelaunch receipts. Large immutable run artifacts are SHA-bound separately and
verified at runtime (not committed as source blobs). Both worktree and committed
source blobs must verify before launch. Commit/push admission chain before GPU
updates. Exclusive gpu_lease physical **card7**, unchanged heartbeat/wall/memory
bounds, detached worker, PID/starttime/boot identity watcher. No WorksGood service.
Standalone expressivity only: Frontier queue and resilient DiLoCo requirements
are not applicable.

Watcher: `.venv/bin/python -m scripts.watch_invertible_stage15_muon_resume --interval 900`.
It watches this Muon continuation and the completed GDN2 amendment; exit0 only
for validated terminal receipts, exit1 interruption/dead identity, --once exit2
pending. Restart detached watcher locally and request explicit supervisor native
re-arming; a running local process alone is not supervisor-side arming.

On completion produce X15G-growth-report-addendum.md and full all64
marginal_signature.json: any position >=50/512 at any observed step is the
registered escape; report fixed-final retention separately, full CE trajectory
vs ln3721=8.221747728, all64 counts/per-position CE and training loss trajectory.
Near-marginal CE alone is NOT proof of zero learning. No new numerical
equivalence tolerance or chance test. Report no escape honestly if none; do not
turn that alone into an exact-marginal/optimizer-robust-failure claim. Until
1M completion all missing crossings remain censored, not terminal verdicts.
