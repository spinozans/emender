# Grounded expansion v1: new data and32-update SFT tranche

## Authorization and purpose

Following the documented [return to data and tuning](../E97_AGENT_LEARNING_STATUS.md),
the operator directed: “Fantastic. Let's roll, baby. Go.” This freezes a **new
bounded internal experiment**, not an extension of the closed880- or32-update
programmes, not production source admission and not outcome RL.

Start from exact correction-u32 live-y, checkpoint SHA
`48dffaf72900419a6481bae05bfd149710627f4bf684a152d7d17804cf55b3e6`, explicit
`weight_mode=train`. Use the exercised immutable `b8ee034f` trainer, fresh BF16-SR
Schedule-Free state, **32 updates at LR1e-5**, original64K/group3/MLP4096/CE128/
eight-rank recipe. No experimental numerical policy or fixed-row kernel. LR is
previously exercised, not claimed optimal. No automatic expansion or promotion.

Question: does denser, more varied observation-dependent supervision and verified
failed-state correction improve editing/recovery and exact value use, while
retaining prior behavior? More data alone is not presumed to answer that question.

## Frozen data

**2,048 new authored, real-executor-verified trajectories:**

| Family | Records | Expanded coverage |
|---|---:|---|
|Lookup|256|Variable selector/table names, decoys,8/16/20/32-character values; actual-file extraction|
|Sum|256|Alternative field names and signed values; executed arithmetic|
|Edit|768|Variable counters, nested fields, positive/negative increments, explicit source/destination distinction and read-back verification|
|Recovery|768|Real initial missing-file errors, one-/two-hop observed pointers and varied value fields|

Pairs share **identical prompts and file paths but different environment contents**
and required outcomes. Each world is executed in its own bounded native sandbox;
training cases have unique namespaces within a world. Half use the minimal system,
half the source system. Training is still four controlled families—not repository
or general-agency coverage.

Half the edit/recovery examples include **authored failure prefixes**: a rejected
create on an existing source or repeated reads of a missing file. These are actual
executor errors, explicitly labelled authored, not model failures. Their assistant
units receive zero loss; only the verified corrective suffix is supervised. Normal
trajectories retain their required initial-action supervision. No observation is
rewritten and no record is truncated.

Each teacher reads real files, computes against their actual contents, verifies
its answer/output against host fixture truth, then finishes. A separate read-only
post-finish checksum command verifies original source bytes. That command is an
external integrity check, not a model action, observation or training target.
Autonomous evaluation continues to use the qualified paused-filesystem oracle.

### Real failed-state material already collected

The32 audited canary candidate occurrences are **18 distinct records**:
10 teacher repairs and8 autonomous successes, **3,447 unique assistant targets**.
All original candidate/episode hashes, immutable rewards and native token/mask
reconstruction are rechecked. Exact token/mask duplicates are included once, with
all original IDs retained. Do not claim32 unique examples or5,730 unique targets.

The raw candidates remain `training_eligible:false`. This operator-authorized
internal derivative is distinct; no original candidate, registry allowlist,
production admission or legal-clearance status changes. Old probabilities are
not used as supervised weights or replaced with newly computed scores.

Copy unchanged admitted replay records to at least100,000 native,200,000
conversation and50,000 retention targets, counting whole-record overshoot. Report
unique inventory separately from repeated runtime exposure. Boundary-aware packing
and the exact32-update schedule must be frozen and audited before model loading.

## Evaluation frozen before demonstrations/training

Retain the previous16 execution cases as regression checks. Add16 fresh same-family
cases and16 **structural-transfer variants**: array-record selection, list summation,
file-specified edit increments and indirect catalog selection. These32 new cases
are paired, have disjoint namespaces/answers from the checked correction/canary/
diagnostic material, and are never inserted into the training authority.
Structural-transfer is not an independent repository/family-generalization claim.

Primary post-y gates, with formulas fixed before observing the baseline:

- Old16 successes≥pre-y.
- Fresh16 successes≥`min(16,max(8,pre_y_fresh+2))`.
- Transfer16 successes≥`min(16,max(4,pre_y_transfer+2))`.
- Fresh+transfer edit/recovery16 successes≥`min(16,pre_y_edit_recovery+2)`.
- Both post x/y retain tool accuracy≥.98, conversation NLL≤pre-y+.15 and native
  development NLL≤pre-y+.10. Report x and y separately with matched weight modes.

A finish claim is not success. Failed gates remain failed; no automatic additional
updates, changed thresholds or candidate promotion. The32 new evaluation cases
are experimental tests, not a final independent holdout.

## Safety, bounds and implementation

Data builder≤1,800s; training≤5,400s; each evaluation phase≤7,200s; teardown30s.
The data phase loads no model and uses no GPU. Use immutable source inventories,
no-overwrite artifacts, owner-cleaned nonroot/networkless native sandboxes, and
checked GPU leases/NUMA/device mapping for subsequent model phases. Preserve exact
Schedule-Free representation, state/clocks and checkpoint publication.

The unrelated dirty dataset-loader changes are explicitly excluded: data build
requires committed loader SHA
`3d49215c5aad63186aecbc0fdf699c26776e8167bde2f3db4aa4e55f1e84889c`.

- Recipe: `configs/pi/e97-grounded-expansion-v1.json`.
- Fixtures/teacher: `scripts/e97_grounded_curriculum.py`.
- Derivative builder: `scripts/build_e97_grounded_expansion.py`.
- Data runner: `scripts/run_e97_grounded_expansion_data.sh`.
- Tests: `tests/test_e97_grounded_curriculum.py` plus native/canary regressions.

**45 CPU tests pass**, including real file/subprocess arithmetic in private test
fixtures, observation-counterfactual pairing, source preservation, suffix masks
and structural-transfer teacher/oracle checks. This is not the native container
verification or a model result.

CPU preflight `grounded-expansion-v1-preflight` has frozen identities:

| Item | SHA256 |
|---|---|
|Recipe|`12a6d13088dfdcfd1484e413ad13a402bc2ce99dfdb64651829f3c5775e5987f`|
|Training cases|`a21f68ad922f5d505980381ef17fa57bf624e1a179d986cf86d60ac8ee30540e`|
|New evaluation cases|`36921d4b718c3d8e0bf88c9fc1cc04ae15ada219229442eb420e5843a84c9331`|

Data run root: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/grounded-expansion-v1-data`.
The native build ran from frozen source `e11f8f6e` (`proc_8ce8`). It stopped at
the1,800-second data deadline (process elapsed1,814s, exit124), after **1,333
complete verified records**, leaving715 uncompleted. Both private journals have
1,333 records; both owned sandboxes were removed. The source audits passed while
the controller preserved exit124. There is no completed training authority,
no packing and no optimizer update. The failed attempt is retained—not relabelled
a pass, retried automatically or used as a smaller replacement dataset.

Source inventory:17,715 files, SHA
`23f899004d3ef6a941615cf8871c34f1b56d1153bf69818f79400c2749b40751`.

Matched controller/gate logic is implemented in
`scripts/prepare_e97_grounded_expansion.py`;44 CPU tests pass in its overlapping
controller/curriculum/execution regression panel. Tests explicitly prevent using
u880 as the new pre-y baseline, dropping outcomes, accepting duplicate/nonboolean
outcomes, or hiding failed editing/recovery behind sums. Previous16-case panel
SHA is `7f03244fca9437ebdfbc7044287c9ddeff61d1953c8a80f062032e46f097ad67`.

Independent audit code now reconstructs native token masks, binds every authored
call/observation, applies the host task oracle, validates source-file checksums and
sandbox cleanup, deduplicates the original candidates and verifies unchanged replay
bytes. A partial-audit mode does not create an eligible data authority.

The launcher separates preparation, training and evaluation. Preparation records
unique examples versus repeated source/family exposures; training requires the
reviewed exposure hash and a no-overwrite attempt marker. It retains the unchanged
trainer, explicit parent weight mode, bounded phases and composed lease/source-audit
cleanup. New tests exercise the unique-versus-repeated exposure accounting.

The independent partial audit (`proc_4a88`, source `dd5f424b`) passed in14s:
1,333 complete records,346,862 supervised targets; lookup256, sum256, edit437,
recovery384. All native calls/observations, source checksums, independently
reconstructed token masks and host oracles passed; both sandbox cleanup receipts
bound correctly. Both journals were unchanged during the audit. The partial set
remains `training_eligible:false` and is not a completed data authority.

Audit source inventory:17,719 files, SHA
`8cdaaa62734d33a9760e6c2a4998b131dd51dff380adf557633d93c766792f29`.
Retained candidate journal SHA
`ba5a5bf7de11f8edee3c64f431135d1ca54448bfe9712067ee139480bc2653c1`;
verification journal SHA
`4e6c809562f5afa0ae05fdb32de0383c0d897caa340c28559f357dd9a2e6548d`.

The initial data-time estimate was too small. Completion would require a separately
bounded additional data pass for the715 missing records, reusing only independently
audited completed records and preserving the failed root. No such completion pass
has been launched. The32-update learning budget and frozen evaluation remain
unchanged. No optimizer updates have occurred in this new tranche.
