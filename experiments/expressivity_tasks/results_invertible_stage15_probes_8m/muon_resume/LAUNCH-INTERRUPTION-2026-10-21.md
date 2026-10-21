# X15b amendment-1 launch: terminal INFRA-INTERRUPTED

Admission commit 974320fc pushed to origin/main before launch. Committed source
and artifact hashes verified, root free 1,653,038,776,320 bytes. Physical card7
exclusive gpu_lease; worker790243, trainer798875. Exact model/optimizer/RNG
restoration checks passed. Updates433001–433043 matched every registered
non-timing diagnostic exactly. Update433044 failed closed:

- muon_update_rms actual0.11409317095890362
- original0.11409317090657782
- absolute difference approximately5.23e-11

No tolerance was applied. No update433297+ (fresh updates) occurred. Original
checkpoint433000 SHA373a7ece3a97790e81d9fa3d0888afdedd74e8eaf8e9b68bbc834c8cf2e882c0
is preserved. interrupted.json, terminal.json and FAIL agree. Lease released;
local watcher874375 observed RUNNING then one-shot watcher observed FAIL; the
sleeping local watcher was stopped with identity verification. No native arming
is claimed. Supervisor confirmed terminal failure, no automatic retry.

Supervisor separately approved amendment-2 restart-from-checkpoint: no exact
at-death-restoration claim; 567,000 fresh updates from433000, original1M cosine
and ceiling, provenance explicitly superseding the interrupted historical tail.
That new admission is separate; this failed admission and its evidence remain
immutable. CUDA floating-point nondeterminism is a plausible cause of the tiny
diagnostic divergence, not independently isolated by this one comparison.

Validation: 24 CPU/adjacent tests passed,1 opt-in GPU test skipped. The first
pytest+tee command could not persist its log because receipts/ was absent; after
creating it, the complete suite was rerun and its successful receipt committed.
Global git diff --check reports pre-existing unrelated trailing blank lines in
stage14/stage16 reports; scoped staged diff check passed.
Standalone expressivity: Frontier/resilient-DiLoCo requirement IDs not applicable.
