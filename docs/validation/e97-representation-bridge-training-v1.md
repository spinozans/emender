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

Authorized for the full bounded sequence. **`proc_edfb` completed all768 native
training demonstrations** in1,179s, exit0, from frozen source `5263ada2` (17,733
files). The terminal source audit passed. The same88 CPU tests also passed inside
that immutable export; its input-binding preflight verified512 paired rehearsal
IDs and96 frozen unassisted evaluation cases.

Completed dataset:2,223 records,6,056,589 input tokens,848,660 supervised targets:

|Source|Dataset targets (not scheduled exposures)|
|---|---:|
|New representation bridges|218,384|
|Prior grounded rehearsal|126,193|
|Conversation replay|300,316|
|Native replay|103,665|
|Retention replay|100,102|

Authority SHA256:
`a21dba6f80e58e87ea838a98018b40650decebed3cf4b0f98dacdb750520469d`.
**`proc_b8f5` passed** the independent audit,64K packing and exact32-step exposure
preparation in84s. Both owned sandboxes were cleaned; all source audits passed.
The data and preparation phases performed zero model updates.

### Reviewed training exposures and launch

All2,223 authority records are scheduled two or three times:5,804 occurrences,
15,548,238 input exposures and2,148,282 supervised target exposures.

|Source|Scheduled target exposures|Share|
|---|---:|---:|
|New representation bridges|564,651|26.28%|
|Prior grounded rehearsal|330,736|15.40%|
|Conversation replay|772,162|35.94%|
|Native replay|215,648|10.04%|
|Retention replay|265,085|12.34%|

Independent byte-level counting finds **2,215 distinct token/mask records**, not
2,223: the619 retention records contain611 distinct sequences. Their eight
inherited duplicates cause some distinct retention records to appear four through
six times in the schedule. All other source records are byte-distinct within this
authority. This was explicitly accepted in `exposure-review-addendum.json`; no
source bytes, quotas, schedule or global target counts were changed, and duplicate
records are not extra unique coverage. Native replay covers eight distinct
trajectories, not eight new task families.

All768 bridge examples (192 per family) and512 paired rehearsal examples are
covered. `exposure-review.json` and `distinct-exposure-audit.json` retain the
attended review and independent occurrence accounting. GPUs were idle with no
active leases before launch.

**`proc_442e` is running the separately authorized32-update SFT tranche**, from
exact expansion live-y, using the unchanged exercised trainer. No post-training
behavioral result is available yet.

|Training input|SHA256|
|---|---|
|Pack manifest|`70ee78073715cf4d87b03579323de0efa0e0c71e676ca693c4034833f3529dc2`|
|Exact schedule|`cfd47a01b76d56f735211a5206d439e6a1bf08f40166f304d19a4ec7725eefef`|
|Reviewed exposure audit|`1d6c82e598d3be63b8cdcd0d05190a13ab3febac83fb64d6598145a240aaad80`|

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
