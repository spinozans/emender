# X40 part 25: budget-scaled safe-depth probe

The transfer verdict is pending. No part-25 accuracy result is claimed.

Prediction committed before evaluation in `97fb72be`; preregistration SHA256
`60d253d382c74f32287e0dddb284564e2756daa68b50d5db8544dee77e9c3c95`.
Budget4 must reach balanced accuracy >=0.85 at 2x, and budget16 >=0.85 at
4x. Either miss falsifies this conjunction. Infrastructure failures remain
incomplete rather than being interpreted as a scientific null.

The original v1 depth fine-tune overrides all four heads to 2944 slots;
therefore the matched budgets here are 2944, 11776, 47104 **per head**, not
multiples of the differentiated LM-pretraining budgets. All variants begin
from the same hash-verified 3052-step reproduction checkpoint. Parameter
shapes are unchanged. Each runs 30000 updates using the original optimizer,
sampler, B64, seed, validation selection, and replay requirements. Part25's
isolated depth module preserves the shared v1 code used by other sessions.
Only ID, 2x, and 4x final evaluations exist in this runner; 9x is excluded.

Telemetry retains every evaluation minibatch and block/head: native final
live occupancy (absolute and budget fraction), R-CAP rate per token and per
allocation/refusal request, and the existing gate, match, routing and pressure
metrics. Final occupancy is a terminal snapshot, not a time-average; its scope
is explicit. Validation telemetry is discarded from the training trajectory.

The X28E `x28e_completion_adjudication/phase_diagram.json` read during setup
marks B0x16 final evaluation as infrastructure-censored. The supplied 0.95
9x claim is motivational context, not independently attested by that receipt.
The existing v1 safe-bin results and transformer control were read first.

Run management: `part25_run.sh` obtains a broker lease restricted to cards 3/4.
One detached shell owns the lease/heartbeat and one parent processes all three
variants sequentially. Parent lock prevents duplicate launches. Parent/watch
receipts name PIDs and active scale. Workers checkpoint atomically every 32
updates, including optimizer/RNG/sampler and selected-validation weights.
Relaunching `bash experiments/expressivity_tasks/x40_emender_lm/part25_run.sh`
resumes each incomplete variant. Source manifest drift fails closed rather
than silently adopting concurrent kernel changes. No wg services are used.

An initial parent was deliberately stopped during parity preflight, before
training or evaluation, to bind the transitive source manifest. The replacement
parent retains the same pre-registration and checkpoint. Runtime receipts in
`receipts/part25_budget` show the current state. Check `watcher.json`,
`budget*/depth_checkpoint.json`, `budget*/terminal.json`, and `verdict.json`.
Do not interpret initial model allocation or parity as completed training.

Validation: telemetry unit test passed (1 test); Python compilation and shell
syntax checks passed. Full GPU parity is a mandatory worker preflight at each
budget and has not yet been attested complete. Large budgets may exhaust
memory or expose safe-depth kernel limitations; those failures remain receipts.
No `.pt` or `.ckpt` files are committed. The contact_supervisor tool was not
available in the active tool catalog, so no intercom update could be sent.
