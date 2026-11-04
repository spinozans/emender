# C14-B registration and exact resume state

Registration landed alone in `8196246b`, before any B1 head implementation,
training, GPU lease or checkpoint. Its protocol SHA256 and 28 committed source /
receipt bindings are in registration.json. Registration audit passes and the
standing CPU construction preserves 2,133,700 trainable parameters and all
registered parameter shapes. Actual C14 physical capacities are 32/64 per
layer, not the six-layer X40 capacities. No dense/C14 arm or receipt was changed.

This delivery is PARTIAL. Phase 1 is complete. Phase 2 contains registration
validation only: there is no enabled B1 head or training runner. No enabled-path
parity, gradient, oracle, lifecycle or full test-battery claim is made. Phase 3
has not started. PB1-PB4 and conditional ablations remain unmeasured. Do not
interpret the successful three registration-audit tests as acceptance of B1.

Resume at Phase 2 from registration commit 8196246b. Steps=0, training targets=0,
active seconds=0, checkpoint=None, optimizer=None, RNG state=None, data cursor=0,
lease=None. Initialize the registered seed only when the full implementation
and launch gates pass. No partial training state needs recovery.

Implementation seam: c11.Boosted currently inserts consultation via an embedding
forward hook. B1 must remove that enabled-channel dependency and insert into
velocity_candidate.Cell's existing per-head physical stores instead. Its scan
feeds x40 custom autograd/replay and checkpoint machinery. That interface
currently accepts one stream key/value event per position, with reads before
writes; it does not implement suffix ownership, pre-read external APPEND, or
post-observation array CORRECT. Adding an external Python side store or
mutating a returned FusedState would not certify shared-budget semantics or
correct replay gradients. Extend forward, decision journal and backward
consistently; verify delayed targets and suffix retirement, and preserve the
native dense route for force-disabled exact parity. These are outstanding work,
not proven impossibility or a request for a new design decision.

Initial pytest collection failed because the new test used a relative import
in a directory without package collection. Changed to the absolute project
import; rerun passes all three tests. Failed output retained. Registration
validator output is registration_validation.json. All Python commands used
/home/erikg/emender/.venv/bin/python. No GPU job, broker mutation, WorksGood
service, dispatcher, worker, Frontier command or subagent was started. The
transformer union service and P3 stacking files/processes were not touched.
/mnt/nvme2n1 was not written. No .pt file was produced or staged.

The complete enabled tool catalog has no contact_supervisor tool. Discovery
was performed first; no supervisor session resolution, progress notification or
reply is claimed. No substitute orchestration service was used. Unrelated
pre-existing workspace modifications/untracked files remain untouched.

Residual risks: no B1 implementation/training/evaluation; native autograd and
checkpoint integration remains unverified; exact enabled causal and cache
parity remain unverified; one-seed fixed-time exposure will differ from v2;
occupancy alone cannot establish useful management; dropout and independent
resets can reduce or mask pressure; suffix retirement may prevent MERGE from
occurring, which must be reported; blank-reset PB4 alone cannot attest decay.
No resilient DiLoCo behavior was changed or tested; gap-matrix IDs inapplicable.
