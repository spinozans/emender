# Native-agent training mixture v1

2026-09-10. The attended operator requested faster progress, removed separate
licensing review as an internal-training launch gate, and accepted tightly
bounded numerical continuation rather than demanding bitwise execution.
Provenance/notices remain intact. This does not assert legal clearance,
independent evaluation, model capability, or permission to publish data.

## Data and loader: passed

Evidence: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-training-mix-v1`.
Managed process `proc_e387` completed in 69 seconds; 32 clean-export CPU tests
passed. The immutable derivative is
`/mnt/nvme2n1/erikg/sft/e97-4b-native-training-smoke-mix-v1`.

- Authority SHA: `69f26726e32523dd840e363a5b9b7663b5e9ed37bd48a68e0bdbc9827d98a2c7`.
- Pack manifest SHA: `51c8dc8cc06a895ea39be780d211af6d645e48117fe1be9c70f93088b0b393a6`.
- 2,575 whole training records; 4,612,298 input tokens; 2,004,756 targets;
  88 boundary-aware 65,536-context packs. No validation records selected.
- Native: 52 trajectories, 603,944 targets; shuffled problem selection before
  cycling attempts. No repeated source records in this derivative.
- Conversation: 1,291 complete SmolTalk2 no-think conversations, 1,200,776 targets.
  Thinking records are excluded whole, never stripped or recast as public text.
- Retention: 1,232 core/compositional records, 200,036 targets. The historical
  retention mixture's Tulu replay component is excluded.
- Native source reconstruction and actual pinned executor receipts are bound.
  The original native candidate remains immutable and `training_eligible:false`.
  Two historical authorities lacking the newer eligibility field are admitted
  only by their explicit SHA allowlist, not by a generic missing-field bypass.
- Source manifests and all consumed payload files are verified before/after
  copying. Every emitted record's bytes, mask and index are verified; publication
  is fsynced/no-replace. Complete records are shuffled across sources before
  packing, avoiding source-contiguous blocks.
- Native development problems are disjoint; protected canonical and same-basename
  repositories are excluded. This is bounded overlap screening, not a claim of
  exhaustive semantic/fork screening or an independent final holdout. Previous
  lineage exposure is not erased.

The existing trainer loader materialized all 64 scheduled rank-batches on CPU.
Sampler key 974117, eight ranks, eight updates: **415,097 native / 922,093
conversation / 153,288 retention targets** (27.8499% / 61.8656% / 10.2845%).
These are scheduled consumption counts, not the nominal 30/60/10 construction
quotas. Runtime sample IDs and global counts must match
`expected-eight-update-schedule.json`; the trainer now logs those IDs.

Validation commands and exact staged source are retained in `run.sh`,
`source.sha256`, `staged-tree.txt`, `inventory.sha256`, and the logs. Existing
conversation/retention serialization is preserved, not translated into the
source-native tool grammar. Native immediate no-tool examples and complete
fitting/transfer/execution evaluation remain subsequent work, not prerequisites
for a bounded real-data smoke run.

## Restart evidence and revised launch policy

The earlier authored-fixture run `full-sft-restart-v1` (`proc_914c`) **failed its
original bitwise continuation criterion**. Three effective full-model updates,
atomic publication, and exact restoration of all model/optimizer state on all
eight ranks passed. The following update matched displayed losses/norms but
not the terminal digest. No aggregate passing verdict exists. The checkpoint,
per-rank reports and both logs are retained; no tolerance was retroactively
applied. DDP reduction-order/lifecycle effects are a hypothesis, not a measured
root cause. Tensor-level discrepancy has not yet been quantified.

Exact checkpoint integrity/state restoration remains required. Bitwise
post-resume execution is no longer an initial-training launch gate. A numerical
continuation comparison must declare tolerances before running and report actual
errors. First run bounded real data; stop on nonfinite loss/gradient, memory
failure or data identity/accounting failure. No automatic retries or checkpoint
promotion. The initial smoke rate is not a selected SFT learning rate.

Scope: ADR-003 fixed-world safety intent **R07/R12** (atomic committed state and
restore), **R14/NDP13** (bounded failure), **R16** (honest frozen-source evidence),
and **NDP15** checkpoint atomicity. No elastic/native-data-plane/async/Frontier
qualification, communicator shrink, or background-checkpoint claim.
