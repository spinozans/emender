# F1 — Lane-chain learning-path repairs

## Result and scope

Implemented E1, E2, E3 and E5, in that order, while the bank remained paused.
The final CPU bank suite passes **101/101 checks**: the original 52 checks (two
lake expectations updated with supervisor approval for F2's intentional E6
change) plus 49 new regression checks. No GPU training, restart, scheduler
submission, held-out read, or promotion was performed.

Changed deployed scripts, all under
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts/`:

- `rl_build_pack.py`
- `rl_train_step.py`
- `rl_bank_lane.py`
- `rl_bank_unit_test.py`

The copies under `/home/erikg/emender/scripts/rl-loop-v1/` are byte-identical to
these deployed scripts. The README describes the deployed-copy convention and
remaining live sibling-module dependencies. `rl_loop_driver.py` did not require
an edit: the adoption gate belongs to the lane, and E5 invokes the existing
codec/retained-prefix and first-action-validator functions from the lane.

No edits were made to `rl_bank.py`, `rl_bank_coordinator.py`, lake supply scripts,
the frozen live `bank/` tree, historical receipts, or `emender-paper`. All test
state is confined to `/home/erikg/emender-scratch/fix-fanout/F1-unit`.

## E1 — Consume sampled, adopted exposure, not packed inventory

### Before

`rl_build_pack.main()` appended all selected receipts to
`consumed-receipts.jsonl` immediately after packing. `rl_train_step.main()`
sampled one pack under `--steps 1`; its sample hashes did not directly identify
the pack or authority-record membership. Failed, skipped, and unsampled
teaching inventory was therefore consumed.

### After

- The packer writes **`packed-receipts.jsonl`**, never the consumed ledger.
- `rl_train_step.sampled_pack_records()` resolves the dataset's actual sampler
  cursor to a concrete `pack-XXXXXXXX` id and complete authority record ids.
- Every completed optimizer step logs `sampled_pack`. The checkpoint event and
  checkpoint payload retain the invocation's entire `sampled_packs` list. The
  event also names the parent, authority manifest and pack manifest SHAs.
- `_adopted_exposure_rows()` verifies the receipt authority/metadata identities
  and resolves only those sampled record ids to receipts.
- `_record_adopted_exposure()` is called only after successful lineage adoption.
  Consumed entries contain the receipt SHA, `sampled_pack_ids`, pack-manifest
  SHA, adopted checkpoint SHA, and exposure cycle.
- Unsampled receipts remain eligible for later windows. Low-signal candidate
  skips do not advance consumption.

**Budget choice:** implemented the task's explicitly permitted **exact sampled
pack log** alternative. The receipts invocation still requests one optimizer
step; this repair does not silently authorize a 32-step/full-window teaching
budget. A 32-receipt window is inventory, not a claim of 32-receipt exposure.
The dataset's full permutation traversal is tested, but a production full-pass
mode/budget was not added or exercised.

Tests: `test_receipt_exposure` (10 checks) uses the real authority writer, real
canonical pack builder and real CPU packed dataset with three packs. It proves
packing consumes nothing, sampled pack membership, preservation of unsampled
receipts, concrete consumed-pack attribution, and coverage over a complete
permutation pass. `test_no_signal_exposure_skip` (2 checks) exercises the actual
SFT stage's low-loss skip, proving unchanged lineage/counters/consumption and
orphan cleanup.

## E2 — Sequential channel chains

### Before

The lane trained PG, anchor and receipts candidates before its adoption loop.
Both anchor and receipts read the old `state["lineage_path"]`; adopting the
receipts sibling discarded the earlier anchor work. Anchor/PG update counters
also advanced before behavioral acceptance.

### After

`_train_cycle_channels()` immediately completes
**train → probe → adopt** for each channel before launching the next.
`_probe_adopt_train_row()` advances and persists lineage, records supersession,
updates total/channel-specific adopted counters, and commits receipts exposure
before returning. A subsequent receipts invocation therefore receives the
anchor-adopted child's path/SHA; a rejected anchor does not become its parent.
PG follows the same sequential rule. A supervisor-approved E2 follow-up also
separates new anchor saves into `training/cycle-N/anchor-checkpoints/`, while
receipts retains `training/cycle-N/checkpoints/`. Both channels previously used
the same `u000001_loss_<4dp>` basename; equal rounded losses could overwrite the
just-adopted anchor even with correct sequencing. Existing on-disk checkpoints
and historical lineage paths are not moved or rewritten; this split applies
only to new saves. The existing metrics `trains[]` array is
preserved, retaining each channel's row in order.

Tests: `test_lane_channel_chain` (5 checks) exercises the real channel
orchestrator and metrics writer with CPU stage stubs, asserting anchor-child
parent identity, exact train/probe/adopt ordering, both retained updates,
supersession order, both metrics rows, and receipts-only exposure commit.
`test_rejected_anchor_chain` (2 E2/E3 checks) proves a rejected anchor does not
enter receipts ancestry and that rejected/adopted rows both remain in metrics.
`test_channel_checkpoint_paths` (3 checks) drives the actual anchor/receipts
command builders and event readers with identically named stub checkpoints,
asserting separate output roots, preservation of both files/SHAs, and the
receipts command's binding to the preserved anchor parent.

## E3 — Probe acceptance is explicit

### Before

A zero-exit `probe` subprocess could write `frame_valid=false`; the lane adopted
its candidate anyway and incremented counters.

### After

After the subprocess returns, `_probe_adopt_train_row()` reads its artifact and
requires **`frame_valid is True`** and the candidate checkpoint SHA to match.
False, missing, non-boolean validity, or wrong candidate identity prevents
adoption. Unreadable probe artifacts fail closed before state mutation. Invalid
candidates retain a metrics row with `adopted=false`, are reclaimed, and do not
advance lineage, supersession, total/channel update counters or consumption.
Successful rows explicitly record `adopted=true`.

Tests: `test_invalid_probe_rejection` (10 checks) invokes a real zero-exit CPU
subprocess through the existing bounded `_spawn` seam. It writes invalid probe
artifacts for anchor, receipts and PG, plus string/missing validity. Assertions
cover unchanged complete state, no persistence/adopted-exposure calls, no
consumption, explicit rejection metrics, and candidate cleanup. The combined
chain regression additionally proves training continues from the last adopted
parent rather than a rejected candidate.

## E5 — Fresh solve irreversible first-action failures

### Before

Only degeneracy-screen failure selected a fresh teacher solve. A clean but
wrong retained first action remained in correction context; no continuation
could replace the sealed criterion's first action.

### After

`_correction_requires_fresh_solve()` preserves the degeneracy trigger and also
checks the byte-bound sealed spec's optional `required_first_action`. It uses
the actual retained-prefix indices and codec semantic turns, ignores private
`think` frames, and delegates subset-exact tool/argument matching to the
first-action validator's `_check_first_action()`.

A retained mismatch selects the existing clean-workspace/no-prefix route. The
teacher correction then has no retained frames and `supervise_from=0`. Correct
first actions with recoverable final failures keep repair continuations.
Dropped finishes, dropped unresolved calls, absent first-action criteria and
clean seed tasks are not falsely treated as irreversible first-action failures.

Tests: `test_fresh_solve_first_action` (11 checks) covers clean tool/argument
mismatches, validator-correct replaceable first actions, recoverable finals,
dropped frames, think frames, degeneracy, absent criteria and seed tasks.
`test_fresh_solve_collect` (6 checks) exercises actual `bank_collect()` routing
with CPU engine/episode/grader/receipt stubs: wrong tool and wrong arguments
produce no prefix, fresh workspaces, and a real receipt-function call with
`supervise_from=0`; a correct first action preserves the mutated workspace,
splice prefix, and `supervise_from=1`.

## Validation and cross-fix coordination

Host: **lambda01**, using `/home/erikg/emender/.venv/bin/python` (Python 3.12,
Torch 2.9.1+cu128). This is not Frontier, so the Frontier-only module activation
was not substituted onto this host.

Exact suite command, run after each fix and after supplemental rejection tests:

```bash
/home/erikg/emender/.venv/bin/python \
  /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts/rl_bank_unit_test.py \
  --scratch /home/erikg/emender-scratch/fix-fanout/F1-unit
```

Logs and results:

| Log under `emender-scratch/fix-fanout/` | Result |
|---|---|
| `F1-E1-tests.log` | 62/62 |
| `F1-E2-tests.log` | 67/67 |
| `F1-E3-tests.log` | 77/77 |
| `F1-E5-tests.log` | 94/94 |
| `F1-final-tests.log` | **101/101** |

The first E1 suite invocation failed at its historical four-bundle lake
expectation because F2's concurrent E6 repair intentionally admits only train
bundles. Supervisor explicitly approved updating this F1-owned test, requiring
both positive admission and explicit development exclusion. `test_lake` now
asserts two train bundles admitted **and** two development bundles skipped with
`reason=non-training-split`; all original 52 checks are retained. No F2-owned
implementation was edited.

Additional checks: all four modified Python files compile with `compile()`;
all four deployed/repository copies compare byte-for-byte; `git diff --check`
passes. Scoped staging only; each fix was committed and pushed to `origin main`
after `git pull --rebase origin main`, without force pushing:

- E1: `d8f04501` — `fix(rl-bank): E1 consume only sampled adopted receipt packs`
- E2: `42a6d608` — `fix(rl-bank): E2 chain channel training through adopted lineage`
- E3: `8c8f5f82` — `fix(rl-bank): E3 reject invalid probe candidates before adoption`
- E5: `02024c80` — `fix(rl-bank): E5 fresh solve sealed first-action mismatches`
- Supplemental E1/E3 rejection regressions and the initial report:
  `3b63a067` — `test(rl-bank): E1 E3 rejection regressions and F1 repair report`.
- E2 checkpoint-path collision follow-up, its three regressions and updated
  report are committed separately after supervisor approval.

Architecture authority reviewed: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, v1
with ADR-003 amendments, and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Relevant
safety intents are **R07** (validated publication/adoption identity), **R12**
(lineage continuity), and **R14/NDP13** (bounded stage subprocesses). Existing
bounded `_spawn` behavior is retained. These prototype CPU checks do **not**
claim production, native elastic, NDP, async-v2.1/V21S or ISP conformance,
checkpoint-resume/optimizer continuity, or a live systems qualification.

## Corrections to prior receipts — history preserved

The following are corrections to documented **intent**, not edits to evidence:

- `gym-starvation-diagnosis-fix-20261006.json` claimed anchor→receipts chaining.
  The audited implementation actually trained siblings. In cycle 1111 both
  channels named the cycle-1110 parent `81dd5a47…`; anchor child `87028a20…`
  was replaced by receipts sibling `bac79a63…`. E2 implements the chain now.
- `era10-receipts-dose-fix-20261006.json` documented larger windows/full-window
  teaching intent and pack-derived dose claims. Cycle 1111 consumed 32 receipts
  and packed 10,522 assistant targets, but the receipts invocation sampled one
  pack containing only 482 targets (**4.6%** of packed inventory). The earlier
  42,349-target step in its log was anchor replay, not receipts teaching.
  E1 now distinguishes inventory from sampled, adopted exposure; it does not
  claim those old windows were taught or that one-step exposure is wash-scale.

All original receipts, consumed ledgers, checkpoint history and bank state were
left unchanged. Historical ledger reconstruction/recovery requires a separate
reviewed action; newly correct accounting does not repair old consumed entries.

## Deviations, limits and next step

- The allowed exact-sampled-accounting branch was chosen instead of silently
  increasing the teaching budget or adding a full-pass training mode.
- E2 extracts two narrow helpers so the actual cycle ordering and acceptance
  seam can be regression-tested; no merge/coordinator/optimizer policy changed.
- E3 additionally binds the valid probe to its candidate SHA and marks rejection
  explicitly in metrics, avoiding an unbound validity assertion.
- The E2 checkpoint-directory split was separately approved after detecting
  the rounded-loss cross-channel filename collision. No historical files were
  moved. The basename remains collision-prone for intra-channel retries in the
  same cycle/channel directory; this is a **residual P2**, not an expanded retry
  or recovery-system redesign.
- F2's train-only lake semantics required the approved test-only coordination
  described above.
- CPU tests do not prove GPU model updates or improved competence. No bank
  resume is authorized by this report; independent reviewer acceptance remains
  required before an operator-approved bounded runtime check.
- Consumption is appended after saved lineage adoption. As in the existing
  filesystem orchestration, lane-state and consumption JSONL are not one atomic
  crash transaction; a crash between those writes requires reconciliation from
  the retained checkpoint/sample logs. No new transactional recovery system was
  introduced or claimed.

Recommended next step: independent review of the scoped commits, then a
separately approved small GPU cycle checking parent-SHA ancestry, sampled-pack
exposure and rejection accounting before any bank resume or scaling.
