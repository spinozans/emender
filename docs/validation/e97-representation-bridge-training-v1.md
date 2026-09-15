# Representation bridge SFT: separately authorized32-update tranche

The operator approved the [bounded proposal](e97-representation-bridge-preparation-v1.md)
with **“run it”**. This is a new experiment; the880-update programme and both
completed32-update correction/expansion budgets stay closed. No automatic retries,
extra updates, threshold changes or checkpoint promotion are authorized.

## Frozen scope

-768 previously frozen bridge fixtures, all to be executed/verified anew. The48
  native-preflight examples are a distinct seed and are not substituted for them.
-512 byte-identical prior verified grounded **training** records:64 complete
  counterfactual pairs per family, selected deterministically by hash.
-Unchanged admitted replay: at least300,000 conversation,100,000 native and100,000
  retention targets. Whole-record overshoot is explicit, not silently trimmed.
-Replay source order is an explicit array, independent of JSON key serialization.
-Exact live-y of expansion-u32 (`17aa26720cc9f4f921cc547ae6599f964bcaa48b15f4b3b436876fba167c471b`),
  fresh BF16 stochastic-rounding Schedule-Free,32 updates, LR1e-5. Reuse the exercised
  `b8ee034f` trainer; no experimental linear/recurrent policies or new RL updates.
-Whole records in boundary-aware64K epoch-permutation packs, recurrent reset only
  between records, unchanged architecture/checkpointed-CE geometry. Persistent
  weights/gradients/optimizer remain BF16. No fresh bitwise continuation claim.
-Native data≤2,400s; training≤5,400s; execution/retention evaluation≤7,200s each.
  Source-audit/teardown allowances are separately bounded by the outer controller.

Original preparation candidates remain ineligible. Only this separately authorized
internal derivative becomes eligible after full generation; an independent audit
is required before packing/training. Production admission/allowlists and protected
repositories are unchanged. Actual observations and original source bytes are
preserved; authored failure prefixes are explicitly labeled and masked.

## Fixed evaluation and acceptance

Four matched models: `pre-y`, `pre-x` are expansion-u32; `bridge-y`, `bridge-x` are
the new endpoint. y means explicit `train` loading, x means `saved`.96 unassisted
cases per model (384 episodes):

|Cohort|Cases|Primary bridge-y requirement|
|---|---:|---|
|Prior regression|16|≥7 (restore previous best)|
|Prior same-format fresh|16|16 (preserve the expansion gain)|
|Prior structural variants|16|≥4|
|New fresh|32|`min(32,max(24,pre-y+4))`, each family≥4/8|
|New composition|16|`min(16,max(4,pre-y+2))`|

The first48 are regression/development diagnostics, not independent held-out
families for the new curriculum. New nested/composed cases are controlled
compositional probes, not independent repository-level generalization.

Both new x/y must retain tool accuracy≥.98, conversation macro NLL≤pre-y+.15 and
native-development macro NLL≤pre-y+.10 on unchanged panels. Missing/duplicate cases,
nonboolean outcomes, missing models, assisted cases and invalid/nonfinite metrics
fail closed. All15 checks must pass for positive bridge evidence; even a pass does
not promote the checkpoint or authorize further updates.

## Implementation and pre-launch checks

-`configs/pi/e97-representation-bridge-training-v1.json` freezes authorization,
  parent, data sources, quotas, budgets and acceptance language.
-`scripts/build_e97_representation_bridge.py` preserves source bytes, candidate
  journals and deterministic paired rehearsal provenance.
-`scripts/audit_e97_representation_bridge.py` independently reconstructs native
  masks/oracles and replay selections, checks every authority byte/index/metadata
  row and both sandbox cleanup receipts.
-`scripts/train_e97_representation_bridge.py` uses the unchanged stage preparer,
  audits exact32-step exposures and builds matched96-case evaluations.
-`scripts/run_e97_representation_bridge.sh` separates data/prepare/train/evaluate,
  requires an attended exposure-hash review before training, composes GPU lease
  and source-audit cleanup, and retains no-overwrite training-attempt evidence.

**88 CPU tests passed** before launch, including executed bridge teacher examples,
strict gate failure cases, full paired rehearsal selection, replay-order invariance,
exact source-copy preservation and refusal to overwrite a published authority.
Training remains contingent on complete768-record verification and independent
packing/exposure audits. No new capability result is claimed at launch.

## Phase status

Authorized for the full bounded sequence. **`proc_edfb` is running the768-record
native data verification**, from frozen source `5263ada2` (17,733 files). The same88
CPU tests also passed inside that immutable export; its input-binding preflight
verified512 paired rehearsal IDs and96 frozen unassisted evaluation cases.

Controller: `R/representation-bridge-v1-training-control/run-phase.sh`
(`R=/mnt/nvme2n1/erikg/e97_systematic_posttraining`).

|Identity|SHA256|
|---|---|
|Source inventory|`f7bb2c1250a8d7ab39f56d2e6b51d3a86a69920b3f525547c0c51385fb7e70fa`|
|Training recipe|`e363bcfda60df0e57f5466773cf57805286f64e4fb148e183410f696bba31cc9`|
|Outer phase launcher|`ef6b8ef778401ac78da3e867d3956b419126764eeb2cfc4e922daf29aac45673`|

On success, continue audit/packing, attended exposure review,32-update SFT and
matched task/retention evaluation without additional routine operator handoffs.
Stop and preserve evidence on failure; do not reopen a completed phase. No model
has been loaded or updated in this new experiment at data launch.
