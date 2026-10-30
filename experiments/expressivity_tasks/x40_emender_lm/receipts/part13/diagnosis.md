# X40 part 13 diagnosis and validation

The depth call chain is depth.evaluate -> Probe.eval / Probe.forward ->
LM.features -> Block.forward -> Cell.forward. Cell.forward calls the
model-module alias of X39 control_evaluate.profile whenever training is false.
It was inherited at the shared LM cell, not explicitly requested by depth.

X39 profile reconstructs live occupancy from an initially zero ledger:
cumsum(allocation minus all frees), then compares its last token with native
live.sum(-1). This assumes a complete event trace from an empty entry state and
matching allocation/free semantics. It cannot in general describe a continued
state without an entry occupancy. X40 depth currently invokes scan with state=None;
there is no evidence here that depth passed a continued state across that hook.
Removing the cross-family hook is a scope correction, not proof of a
chunk-boundary accounting defect.

Crucially, the original console contains Triton int32-overflow device assertions
BEFORE the stack reaches torch.equal. An asynchronous CUDA error surfacing there
is not a Python AssertionError and does not prove the live-accounting equality
failed. The original underlying device failure remains unlocalized. Production
parity and kernel debug assertions have not been disabled or changed.

Only depth.evaluate temporarily replaces the shared profile alias, restoring it
on success or failure. Depth telemetry keeps token allocation/write event density,
existing routing metrics and X40 gate/match/error means. It does not reconstruct
cumulative occupancy or introduce another live-accounting parity assertion.
X39 sources and other LM evaluation profiles are unchanged.

CPU regression tests: 2 passed, checking evaluation flags, profile selection,
event density without consulting final state, and restoration on exceptions.
Real GPU validation: broker-exclusive physical card 6, depth checkpoint 8992,
CUDA_LAUNCH_BLOCKING=1, complete 2000-example validation pool passed with balanced
accuracy 0.995. See eval_console.txt. This does not yet certify all later
checkpoints or the four final bins.

Parent now accepts only broker-bound cards 0/1/5/6 and records the physical card.
It captures original post-checkpoint traces before truncation and compares loss,
model, optimizer, all RNGs and sampler hashes after crash continuation, failing
closed on divergence. The existing deliberate SIGKILL test at updates 14977/14978
and final publication of depth.json remain required.

Resume evidence: emender-part13-resume acquired physical GPU 6 exclusively,
leaving the anchor on card 0 and searches on cards 1/5 untouched. Cards 2/3/4/7
were not used. The runner's preflight passed 109 oracle tests and production
parity. Crash continuation updates 8993/8994 matched the original traces exactly
across loss, weights, optimizer, RNG and sampler hashes; the stable receipt is
crash_resume_roundtrip_8992.json. Resumed validation at update 9000 passed:
2000 examples, balanced accuracy 0.997 (depth_validation_at_resume.json).

The run remains active. The 30000-update finish, four final bins and deliberate
mid-depth SIGKILL acceptance are pending. resume_status.json is a dated snapshot,
not a completion claim. The existing parent publishes ignition_v1/depth.json only
after successful training and verified deliberate kill roundtrip. No weights
are included in this commit.
