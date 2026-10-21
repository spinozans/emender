# X16 preregistration — ignition dose and teach-short-then-extend

Operator directive: `emender-scratch/fix-fanout/taskX16-sweep-curriculum.md`.
This registration and its executable source/hash manifest are committed BEFORE
any confirmatory training. No post-launch choice of recipe, threshold or seed.

## Shared recipe and scope

Standalone lambda01 single-card expressivity, not Frontier or resilient DiLoCo.
Exact X10 AdamW+cosine, stage10-pattern model/forward/task paths: seed0 scratch,
p61, vocab3721, dimension256, four layers, batch32, FP32 parameters/BF16 autocast,
clip1, AdamW betas(.9,.999), eps1e-8, weight_decay=.01 on ALL parameters (including
tagged A_log/dt_bias, intentionally matching X10). Warmup5000 to1e-3, cosine to
1e-5 at500000. E97 legacy state/sequential Triton evaluation; GDN2 stock BF16 FLA
chunk recurrence, no FP32 adapter. Fresh procedural numpy RNG(0); frozen512
validation examples never trained on. Evaluation every500 updates, entry included.
Depth-local evaluation derives its reshape from its own pool; no global patching.

## Measurement 1: explicit-P1-weight sweep

|Run|Physical card|P1 total share w|Other target weights|
|---|---:|---:|---|
|e97_w64_seed0|0|1/64 = .015625|(1-w)/15 per P2..P16|
|e97_w32_seed0|1|1/32 = .03125|(1-w)/15 per P2..P16|
|gdn2_w512_seed0|2|1/512 = .001953125|(1-w)/15 per P2..P16|
|gdn2_w2048_seed0|3|1/2048 = .00048828125|(1-w)/15 per P2..P16|

Fixed T16 from scratch, 500K updates. Only explicit loss weight changes within
architecture; same seed/init/pool/data stream/optimizer/schedule/precision.
Loss is w*mean_batch CE(P1)+(1-w)*mean_batch_and_position CE(P2..P16).
Inputs, targets and full loss mask remain unchanged. No ignition stopping.

Registered predictions:
(a) E97 ignition bracket lies between existing w=0 (X14 no ignition) and w=1/16
(X10 ignition); the two interior points measure, not assume, that bracket.
(b) GDN2 failure point, if in the ladder, is measured at1/512 or1/2048. X8's1/136
ignition is CONTEXT ONLY: its optimizer and non-P1 weighting differ; no matched
causal attribution to explicit w across that historical endpoint.
(c) Assemble ignition (ANY position >=50/512) versus explicit w by architecture,
including first crossing updates and per-position first50/508/512. Report max and
fixed-final counts separately, P1 explicitly identified; composition P2+ separately
visible. No assumed monotonicity. T64 E97 failure at1/64 is context, not a T16 point.
Any additional bracket-closing arms require a separate prelaunch registration;
this launch NEVER uses cards4/5/7, even if they become idle.

## Measurement 2: criterion-gated curriculum

`e97_curriculum_seed0` uses card6. Teach uniform-loss T16 with exact X10 update
function until the FIRST scheduled evaluation with >=4 positions >=508/512
(P1 included; no contiguous-prefix requirement). Maximum500K teaching updates;
if criterion is not reached, stop with criterion_not_reached, censor T64, do NOT
silently weaken the criterion or run a different curriculum.

ONE design choice, fixed now: same live model, AdamW moments and numpy/torch/CUDA
RNG streams continue; no checkpoint reload, optimizer reset, mixing or T16 replay.
At the attained teaching LR, begin a WARMUP-FREE cosine to1e-5 over500K T64 updates.
Formula: lr(s)=1e-5+0.5*(attained_lr-1e-5)*(1+cos(pi*s/500000)).
No LR jump; T64 loss is uniform P1..P64. Evaluate T64 BEFORE its first update
(zero-shot transfer baseline), then every500, plus the SAME frozen T16 retention
pool at every T64 evaluation. Preserve criterion checkpoint separately from T64
entry (distinct checkpoint directories, even at the same global update).

Registered predictions:
(a) TEACH-SHORT-THEN-EXTEND: any T64 position>=50/512 beats cold-start zero;
separate inherited zero-shot acquisition from NEW material positions during
extension. Report transition/maximum/final material frontier and position sets.
(b) If it stays at taught/zero-shot levels without new material positions, extension
has its own ignition barrier: practical fix is partial. Do not claim new learning
merely because transferred P1..P4 remain learned.
(c) Retention: report every T16 position throughout, taught positions tagged,
transition/latest/minimum counts, latest drop, first below508 for taught positions;
transient damage is not conflated with fixed-final damage.

## Evidence, censoring, bounds, monitoring

First crossings are sampled at500-update resolution. Final500K is primary; early
curves descriptive. Absent crossings are RIGHT-CENSORED at observed phase update,
not zero/failure. Infrastructure/time/memory error is censored, never architecture
failure; no automatic retries/restarts. A complete500K no-ignition arm establishes
only no observed ignition by that ceiling. Criterion-not-reached is NOT T64 failure.
One seed only; no multi-seed generality claim. Preserve entry+best+latest (<=3 sweep
checkpoints), curriculum entry+criterion+phase-best+latest (<=4), atomic staging,
SHA256 deletion receipts. Safety bounds:14days wall,16GiB allocated per run.

Preflight uses disposable seed99, no confirmatory scores; check loss gradients,
source model parity, exact optimizer/schedule, T16/T64/T16 shape-safe evaluation,
serialization/next-update parity, criterion and phase switching, retention and
terminal tamper checks. Source/dependency/pool hashes bound in registration.json;
launch refuses worktree/committed-blob mismatch. Leases restricted to0/1/2/3/6.
Native detached watcher checks PID+starttime, terminal result SHA256 and phase
ceilings, writes GROWTH_REPORT on EVERY wake and terminal.json once all arms
terminate. One arm's failure does not disarm the remaining monitoring.
Parent/supervisor should arm this native watcher (no wg service or machinery).
