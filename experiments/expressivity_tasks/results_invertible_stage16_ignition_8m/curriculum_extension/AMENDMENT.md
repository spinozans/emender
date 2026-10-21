# X16E: authorized curriculum EXTENSION amendment

Append-only new arm `e97_extension_seed0`, physical card6 only. Original X16,
replay admission and restart registration/artifacts remain immutable. Original
criterion (>=4 positions >=508/512) was NOT met: T16 ended at500000 with three
perfect positions, and no original T64 phase ran. This amendment explicitly
authorizes a separate extension despite the missed criterion, not a retroactive
success or an alteration of the original criterion.

## Source and provenance

Load `infra_restart/e97_curriculum_seed0/T16/step_500000.pt` relative to X16 root,
SHA256 `870c2dcf5e863dcab817e3c6b0150d054e91cff8456d231378d3748676e06b1f`.
The hash-bound final teaching validation is [512,512,512,5,...], NOT the task
summary's P4=8. Preserve and report the artifact, not the shorthand.
Teaching carries restart divergence at252204, grad_norm
0.09198356419801712 versus0.09198416024446487. Two readings remain registered:
the checkpoint restart may have damaged late-budding P4 relative to the same-seed
four-perfect X10 AdamW trajectory, OR the divergence may be irrelevant and this
teaching trajectory simply stalled. Neither causal reading is claimed.

## Extension recipe (registered before launch)

Restore and exactly compare terminal model, AdamW moments, NumPy/torch/CUDA RNG.
No optimizer reset. Same X10 AdamW: batch32, fp32 parameters/bf16 autocast, legacy
E97 recurrence and sequential Triton evaluation, clip1, betas(.9,.999), eps1e-8,
all parameters wd.01. Only schedule/horizon change: FRESH warmup-free cosine
3e-4 peak to3e-6 floor over500000 extension updates. Fresh phase_step0 is T64
zero-shot baseline; all updates1..500000 use procedural T64 (sequence length66),
uniform full P1..P64 CE. No T16 replay/mixing and no frozen-pool training.
Use original frozen512-example T64 and independent T16 pools, hash-verified.
Evaluate T64 AND `retention_t16` at0 and every500 updates, including final500K.
Checkpoints use T64/fresh extension step numbering, separate from teaching.

## Registered predictions

(a) TRANSFER: any T64 P2+ >=50/512 with taught P1-P3 retention on T16
(>=508/512) materially beats cold-start zero. Distinguish zero-shot step0 runtime
transfer from NEW acquisition beyond baseline material positions: only the latter
supports teach-short-then-extend as LEARNING transfer consistent with X13/X15.
Report baseline, new transient/latest/final positions, exact counts and crossings;
do not label pre-existing step0 transfer as learned extension acquisition.

(b) NO-TRANSFER: T64 columns stay zero/chance while T16 retention holds;
extension has its own horizon-specific ignition barrier and curriculum is partial.
If existing transfer holds but no new acquisition occurs, report that separately
instead of forcing this outcome into a binary zero/transfer claim.

(c) RETENTION DAMAGE: taught T16 positions decay during T64 training; extension
destroys what was taught. Report exact per-validation trajectories either way,
all16 baseline/latest/min counts, drops, first below508, transient versus final.

## Standing rules, monitoring and interpretation

Fixed-final500K primary, transients descriptive. Missing crossings right-censored
at observed phase_step. No stagnation/ignition early stopping. No automatic
retry/restart. Disk/wall/memory/worker interruption is infrastructure-censored,
not architecture failure. Max14 days, peak16GiB, >=200GiB root-free before normal
writes; emergency interruption/terminal metadata is the below-floor exception.
Atomic checkpoint staging; bounded extension entry+best+latest <=3, deletion
hash receipts; source teach checkpoints never pruned. Atomic no-wait card6 lease,
heartbeat60s, release only if owned. Other cards untouched.

Native detached watcher every900s plus parent-owned native supervisor arming.
Watcher command: `.venv/bin/python -m scripts.invertible_stage16_extension watch --interval 900`.
One-shot adds `--once` (exit2 pending,0 hash-verified terminal,1 censored/error).
Growth report regenerated every poll; predictions unresolved before terminal.
Single seed0, standalone single-host expressivity only; no Frontier, Slurm,
resilient DiLoCo, multi-seed or system-scale claim. No WorksGood machinery.
