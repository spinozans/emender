# E97 4B on-policy task-lake and correction execution plan

**Status:** normative implementation companion; Phase A data-safety core is in
progress, and no task source is admitted until its immutable audit passes

**Created:** 2026-09-06

**Behavioral parent:** post-broad exact focused-recipe repair u8

**Behavioral-parent SHA-256:**
`aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39`

**Governing research plan:**
[`EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md`](EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md)

## Decision summary

Build a content-addressed task lake rather than treating any single public
benchmark as the training database. Use fresh SWE-smith tasks as the scalable
executable backbone, license-approved SWE-Gym and R2E-Gym tasks for authentic
repository language and structure, and first-party read/document/recovery tasks
for the intended recurrent assistant product. Public promotion benchmarks,
consumed V3/V4, the real-repository holdout, and the future V5 remain outside
teacher and training access.

During data acquisition, run one independent E97 actor per GPU. The eight GPUs
replicate the 4B model; they do not form DDP and do not perform an outer DiLoCo
merge. CPU workers create isolated fixtures, execute bounded tools, snapshot
state, run deterministic validators, and publish immutable receipts. Remote
Luna/Terra/Sol teachers only propose continuations. No teacher can admit a
record. Training is a separate, alternating phase that stops rollout actors and
uses all eight GPUs for the existing fixed-world DDP path.

The first systems slice is deliberately smaller than the eventual data program:
128 tasks validate end-to-end collection, followed by a bounded 1,200-train-task
and 240-development-task pilot. Scale toward 5,000 tasks only after behavioral
and overlap gates pass.

## Checkpoint lineage and branch decision

The task-lake program builds on the current behaviorally selected checkpoint. It
does not restart from the raw foundation, the collapsed broad terminal, or the
repair-only control.

```text
mature E97 foundation
  step 24,448 / 99,723,771,904 accepted tokens
  SHA 3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568
        |
complete-64K behavioral/document parent u256
  SHA 2f24db49be7bafb0e155bf3698f20193def8e90efc1bd69bdfbfe2b9661b541b
        |
aggressive document-aware epoch
  1,144,364,830 input tokens / 514,953,509 targets
  SHA 7d0e17afd6dde6b4846c2143580fc87d53f1e8a645bfb87acf1f3d2428e7cf37
        |
exact historical focused Pi/instruction mix, 64 updates at 1e-5
  SHA a524b9c8d9d2ca69911db3724fbc003bf9dc8b64dd7f783b98d1cbb324007eb2
        |
exact live-aligned repair, 8 updates at 5e-6
  SHA aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39
        |
new on-policy task collection and correction branch
```

The 1.144B-token stage was one boundary-aware training epoch over many complete
records, not one billion-token trajectory. Its terminal saved `x` policy was
collapsed on Pi behavior, but the exact 64+8 focused sequence restored core
smoke to 120/120 and compositional v2 to 236/240. That restoration is why
`aae654aa...` is the parent.

Use `aae654aa...` in two roles:

1. replicate its saved `x` weights on eight independent rollout actors to
   collect the student's actual successes and failures;
2. initialize a named correction-training branch from those saved `x` weights
   with a fresh Schedule-Free optimizer after the new authority is sealed.

Within a correction-training run, optimize the live Schedule-Free `y` weights,
save promotion checkpoints as averaged `x`, and evaluate paired `x`/`y` behavior
at every aggressive continuation gate. The first training action is an
8-update canary, not a long continuation. If behavior improves, extend through
predeclared 32/64-update gates and optionally apply the proven bounded
live-aligned repair stage. If it fails, retain `aae654aa...` and discard the
branch; no ancestor is overwritten.

The ready 1.35B-target general-continuation authority is not part of this branch
and remains unlaunched pending separate broad-behavior justification.

## Product and runtime boundary

There are two distinct curricula. They must have different versioned tool-schema
and controller digests and must not be silently mixed in one authority.

1. **Read-observe product track:** typed `list_files` and `read`, grounded
   document answers, discovery, comparison, recovery, clarification, and honest
   blockers. This is the immediate product and V5 target.
2. **Repository-repair track:** later typed `edit` and focused `run_tests` in
   addition to read/list. Full public SWE tasks enter only after this bounded
   schema and its sandbox semantics are frozen. Unrestricted `bash` is not the
   target interface.

The historical `read`/`bash`/`edit`/`write` Pi runtime remains useful diagnostic
scaffolding. Results collected under it are labeled with its exact runtime
digest and are not reported as read-observe V5 results.

## Why a portfolio, not one benchmark

No public source simultaneously supplies enough scale, authentic failures,
product-relevant document work, clean licensing, and a defensible untouched
holdout. The task lake therefore separates source material, generated task
identities, execution environments, student rollouts, teacher proposals,
validator receipts, and admitted training records.

### Primary scalable source — SWE-smith

[SWE-smith](https://github.com/SWE-bench/SWE-smith) is the preferred executable
generation framework. Its official repository describes approximately 52,000
published tasks, 250+ repository environments, procedural bug generation, and
the ability to generate additional tasks for new repositories. The framework is
MIT licensed.

Use it to construct fresh task identities from audited repositories:

```text
known-good immutable repository revision
  -> inject controlled defect
  -> prove one or more focused tests fail
  -> preserve the clean and defective tree digests
  -> generate/admit bounded task wording
  -> run the student
  -> validate the final tree with focused and regression tests
```

The framework license does not supersede the license of an underlying
repository. Every repository requires a separate source and redistribution
audit.

### Authentic repository sources — SWE-Gym and R2E-Gym

[SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) reports about 2,400 real tasks
from 11 Python repositories with executable environments and tests. It is useful
for authentic issue language, repository structure, and comparison against
published agent/verifier trajectories.

[R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) reports more than 8,100
procedurally curated executable tasks across 13 repositories. Its commit-derived
construction and automated rewards are a useful diversity complement to
SWE-Gym.

Neither source is bulk-admitted. Each task must pass the same immutable revision,
license, environment, verifier, overlap, and bounded-runtime checks as a
first-party task. Prefer any documented decontaminated subset, then independently
audit it against Emender training and holdout identities.

### Trajectory replay — OpenHands/SWE-Gym

Published OpenHands SFT and verifier trajectories can provide clean-policy
replay and teacher-reference material after license and provenance review. They
are not on-policy corrections because they do not begin from this E97 student's
failure states. They therefore remain a separately counted mixture component.

### First-party product tasks

Generate new executable families that public SWE corpora do not cover:

- direct questions over one bounded document;
- list-then-read discovery with opaque paths;
- comparison and synthesis across two to five files;
- structured JSON/CSV/Markdown extraction;
- missing path and wrong-extension recovery;
- stale environment hints;
- successful empty results;
- conflicting or insufficient evidence;
- clarification and grounded blocker reporting;
- ordinary conversation requiring no tool;
- long sessions used for save/restore/fork demonstrations.

Values, paths, layouts, formats, distractors, and phrasing vary independently.
Task authors may propose instances; deterministic generators and validators own
final task bytes.

### Optional quarantined source — SWE-rebench

SWE-rebench may provide a large, more recent real-issue pool, but it is
quarantined until repository provenance, environment reproducibility, source
licenses, and overlap with every evaluation panel are audited. Its nominal size
is not an admission argument.

## Evaluation firewall

The following are evaluation or diagnostic assets, not training sources:

- SWE-bench Verified and its 500 expert-verified tasks;
- Terminal-Bench and other public agent leaderboards used for promotion;
- Emender's real-repository holdout;
- consumed V3 manifest
  `ef481c637fde5916b8b0fe1f80cc2b4f0a6b88262088cbb33aefe0fed6bd6d09`;
- consumed V4 manifest
  `8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8`;
- the future sealed V5 registry.

V3/V4 may supply abstract failure labels such as `invented_path` or
`ignored_observation`. Their tasks, text, paths, values, fixtures, expected
calls, traces, and transformed derivatives may not enter prompts, validators,
training, rejection sampling, or teacher context.

V5 is generated only after the read-observe runtime, tool schema, task families,
mixture, training stages, checkpoint-selection rule, and thresholds are frozen.
It uses distinct whole generator families and repositories, not merely different
random seeds.

## Source and repository admission

A repository is eligible only when all required receipts are present:

- canonical URL and immutable commit;
- source archive SHA-256;
- detected license plus manual disposition;
- permission to use and redistribute derived task artifacts as planned;
- installation commands with locked dependencies;
- network-free test execution after preparation;
- deterministic focused verifier and bounded regression verifier;
- CPU/RAM/time/output limits;
- no secrets, credentials, submodules, or mutable remote dependencies;
- clean-tree and fixture-tree digests;
- no whole-repository overlap with protected holdouts;
- no exact or normalized semantic collision with protected task registries.

Unknown, conflicting, noncommercial-only, or missing licensing fails closed.
Repository code and generated task/trajectory licensing are recorded separately.

## Task identity and immutable bundle

A task identity is derived, never asserted:

```text
task_identity = SHA256(
  identity_schema || namespace || family_id || generator_source_digest ||
  fixture_tree_digest || canonical_intent_digest
)
```

Every task bundle records at least:

```json
{
  "schema": "emender-e97-onpolicy-task-v1",
  "split": "train | development",
  "task": {
    "namespace": "e97-train-* | e97-dev-*",
    "family_id": "...",
    "identity": "sha256",
    "generator_source_digest": "sha256",
    "fixture_tree_digest": "sha256",
    "intent_digest": "sha256"
  },
  "source": {
    "kind": "swesmith | swegym | r2egym | first-party",
    "repository": "owner/name",
    "revision": "immutable commit",
    "license_receipt_digest": "sha256"
  },
  "runtime": {
    "tool_schema_digest": "sha256",
    "controller_digest": "sha256",
    "sandbox_image_digest": "sha256"
  },
  "limits": {
    "turns": 12,
    "seconds": 300,
    "output_bytes": 16384
  },
  "validator": {
    "spec_digest": "sha256",
    "focused_command": "typed verifier identity",
    "regression_command": "typed verifier identity"
  }
}
```

The bundle does not expose a gold patch to the student or teacher. Gold changes,
when a source includes them, are quarantined for generator validation and
contamination checks. Alternative valid solutions pass when they satisfy the
same deterministic outcome and minefield constraints.

## Pilot composition

### First 128-task systems slice

Run 16 tasks per GPU:

| Source/stratum | Tasks |
|---|---:|
| fresh SWE-smith locate/single-edit/recovery | 64 |
| audited SWE-Gym/R2E-Gym tasks or bounded subtasks | 24 |
| first-party read/document/recovery | 24 |
| ordinary conversation and grounded blockers | 16 |

This slice proves collection and validation mechanics. It is not a training
scale or model-quality claim.

### Bounded acquisition pilot

After the 128-task gate:

| Source/stratum | Train | Development |
|---|---:|---:|
| fresh SWE-smith | 600 | 96 |
| audited SWE-Gym/R2E-Gym | 240 | 48 |
| first-party file/document/recovery | 240 | 72 |
| conversation and grounded blockers | 120 | 24 |
| **Total** | **1,200** | **240** |

Development uses distinct whole generator families or repositories. No training
example is created from a development rollout. A later 5,000-task round is
considered only after this pilot improves held-out behavior.

### Difficulty curriculum

1. locate or read one explicit value;
2. discover an opaque path, then read;
3. make one bounded edit and run one focused check;
4. recover from a missing path, stale hint, or failing check;
5. compare or modify multiple files;
6. solve a bounded real repository issue;
7. execute a long-horizon task with several valid approaches.

At least 70% of the first pilot is levels 1–4. A 4B student should encounter
many learnable decisions; a pool of nearly impossible repository issues would
mainly manufacture long teacher demonstrations rather than useful student-state
corrections.

## Eight-GPU acquisition architecture

For rollout collection, start eight independent, checkpoint-identical actor
processes:

```text
content-addressed task lake
          |
 deterministic CPU dispatcher
          |
  +-------+-------+-------+-------+
 GPU0    GPU1    GPU2    ...     GPU7
 E97     E97     E97             E97
 actor   actor   actor           actor
  +-------+-------+-------+-------+
          |
 raw event logs + state receipts
          |
 deterministic replay/validator workers
       /                             \
 accepted student success       exact failure state
       |                             |
 clean-policy queue          teacher proposal queue
                                     |
                         Luna -> replay -> Terra if needed
                                     |
                         accepted correction authority
```

Required actor properties:

- one GPU and NUMA-local CPU/memory assignment;
- immutable checkpoint SHA-256 and saved-`x`/train-`y` mode recorded;
- isolated sandbox and Triton cache;
- no shared mutable task fixture;
- deterministic task leasing with atomic completion receipts;
- bounded turns, wall time, output, disk, and process count;
- raw stdout/stderr and Pi events retained before interpretation;
- exact system prompt, tokenizer, tool schema, controller, and image digests;
- graceful interruption and idempotent restart.

A task lease does not authorize mutation of the source bundle. Each rollout uses
a fresh copy. The dispatcher may use SQLite or DuckDB as a queue/index, but JSONL,
source archives, fixture trees, traces, and receipts are immutable
content-addressed authorities.

### Rollout variants

The 128-task systems slice uses one greedy rollout per task. Once reproducibility
is proven, the acquisition pilot uses two pinned variants where useful:

1. greedy, temperature zero, the authoritative failure-state rollout;
2. one bounded stochastic rollout with a recorded seed for pass-at-k and
   successful-trajectory acquisition.

Never silently rerun a failed task until it succeeds. Every attempt has a unique
rollout identity and remains in the audit log.

## Progress and no-progress semantics

For each executed call:

```text
action_fingerprint = SHA256(tool_name || canonical_arguments)
progress_fingerprint = SHA256(
  canonical_workspace_state || source_ledger || deduplicated_observations
)
```

The receipt describes post-execution state.

- First `(action, progress)` pair: continue.
- Repeated pair: replace the duplicate result presented to the student with one
  canonical `no_progress` observation and permit a recovery action.
- The same stale pair after that recovery budget: terminate safely.
- The same test after an edit is valid because workspace state changed.
- A repeated read is valid only if it acquires new information.
- Explicitly classified transient failures may have a separately bounded retry
  policy; they are not inferred from prose.

The controller records the authentic attempted result in the audit receipt even
when the student-facing result is the canonical recovery observation.

## Teacher, replay, and admission loop

### Routing

- **Luna:** bulk task-language proposals, routine clean solutions, and short
  corrections. Maximum two proposals per routine state in the first pilot.
- **Terra:** only structurally valid states that Luna failed to solve or an
  explicitly sampled hard-task lane.
- **Sol:** at most a small predeclared audit sample or validator-spec ambiguity;
  never bulk generation.
- **Human:** resolves source licensing, safety policy, validator intent, and
  architecture/objective changes.

Price, provider, model revision, prompts, input/output token counts, and hard
per-run caps are recorded before calls. Provider retention/data-use terms are
part of source admissibility.

### Deterministic acceptance

Every proposal is replayed from the exact original fixture in the same
controller. Admission requires:

- parseable bounded tool calls;
- no minefield or forbidden action;
- required milestones;
- focused and declared regression postconditions;
- a grounded final supported by acquired observations;
- no no-progress cycle;
- task/source/runtime/checkpoint identity consistency;
- training/holdout disjointness;
- complete serialization with no truncation;
- zero target mask on every student-prefix token.

If multiple proposals pass, retain at most two materially distinct trajectories,
ordered by safety, fewer unnecessary calls, fewer total calls, then canonical
action-graph hash. Teacher private reasoning is never a target.

### Correction record

```text
system and user                         loss = 0
student actions and authentic results  loss = 0
canonical no_progress observation      loss = 0
verified corrective assistant suffix   loss = 1 on assistant tokens only
subsequent tool observations            loss = 0
verified grounded final and newline     loss = 1
```

Failed student actions are evidence-bearing context, not positive examples. The
record validator fails if the target suffix is not contiguous, the final is not
targeted, a tokenizer token crosses a context/target boundary, or any consumed
V3/V4 source identity is present.

## Training mixture and alternating schedule

The tentative correction-stage mixture is measured by consumed assistant-target
tokens:

- 35% clean current-policy replay;
- 25% broad instruction/chat/document replay;
- 30% verified student-state corrections;
- 10% successful student trajectories.

This is a starting ablation, not a fixed claim. Compare correction-inclusive and
clean-only controls at matched target tokens. Preserve the proven focused
behavioral structure and exact live-aligned repair rather than substituting a
repair-only shortcut.

Collection and training alternate because both require the eight local GPUs:

1. replicate the selected saved-`x` checkpoint for rollout;
2. collect and verify a bounded generation of tasks;
3. stop actors and seal the authority;
4. build complete boundary-aware packs;
5. run an eight-GPU DDP canary with no redundant outer DiLoCo merge;
6. evaluate every saved `x`, with paired `x`/`y` diagnostics where applicable;
7. collect the promoted checkpoint's new failures, not stale failures forever.

No continuation is authorized by training loss alone.

## Data layout and receipts

A run root contains:

```text
identity/
  run.json
  source-registry.json
  checkpoint.sha256
  runtime-digests.json
leases/
tasks/
sandboxes/
raw-events/
workspace-receipts/
student-rollouts/
teacher-proposals/
validator-receipts/
accepted-records/
rejected-records/
summary.json
```

Each completed task atomically publishes a receipt. Partial directories are not
accepted after interruption. Manifests include counts for attempted, accepted,
rejected, timed out, cycled, validator-failed, and teacher-escalated records by
source, family, difficulty, and rollout variant.

## Gates and stop conditions

### Systems gate — 128 tasks

Require:

- 128/128 reproducible fixture constructions;
- 128/128 raw event and terminal receipts;
- deterministic validator agreement on repeated known-good/known-bad controls;
- exact workspace digest reproduction;
- no task mutation outside its sandbox;
- correct no-progress injection and termination receipts;
- zero V3/V4/holdout collision;
- zero malformed correction masks;
- interruption and idempotent-resume test passed.

### Data gate — 1,200 train tasks

Require at least:

- 55% overall deterministic proposal acceptance after bounded routing;
- 35% correction acceptance after replay;
- 100 accepted records per major training family where the family count permits;
- no authority requiring truncation;
- no mixed checkpoint or runtime identity;
- complete source/license/overlap receipts.

Failure to hit an acceptance floor triggers task/validator/prompt diagnosis, not
unbounded teacher retries.

### Behavioral canary gate

Run an eight-update canary first. Promote or extend only when saved-`x` behavior
improves on fresh development identities in:

- literal and opaque value binding;
- post-observation next-action changes;
- recovery after authentic failures;
- executable completion;
- grounded final answers;
- no-progress-cycle rate.

Core smoke, compositional v2, broad conversation, document comprehension, safety,
and non-overlapping language loss are regression gates. Consumed V3/V4 are
telemetry only. A regression outside a predeclared tolerance stops the stage.

## Implementation sequence and current status

### Phase A — deterministic data-safety core

- [x] Canonical action fingerprints.
- [x] Workspace/source/observation progress fingerprints.
- [x] One canonical recovery observation and per-stale-pair recovery budget.
- [x] Derived task identities.
- [x] Strict student-state correction schema.
- [x] Zero-loss failed-prefix and contiguous target-suffix validation.
- [x] Consumed V3/V4 provenance rejection.
- [x] Masked-SFT correction authority builder.
- [x] Focused and adjacent protocol/SFT regression tests: 60 passed.

Current implementation files:

- `ndm/e97_onpolicy_records.py`;
- `scripts/build_e97_onpolicy_correction_sft.py`;
- `tests/test_e97_onpolicy_records.py`.

### Phase B — source registry and task bundles

- [x] Implement the immutable repository/source audit schema and validator.
- [x] Pin candidate SWE-smith, SWE-Gym, and R2E-Gym framework revisions;
  these remain candidates, not admitted task sources.
- [ ] Audit underlying repository licenses and environment requirements.
- [x] Implement derived task-bundle identities, admitted-source enforcement,
  protected-panel collision checks, and whole-family/repository split isolation.
- [ ] Author the first independent first-party generator families.

Current Phase B files:

- `ndm/e97_task_lake.py`;
- `scripts/validate_e97_task_lake.py`;
- `configs/pi/e97-onpolicy-source-registry-v1.json`;
- `tests/test_e97_task_lake.py`;
- `docs/validation/e97-4b-onpolicy-task-lake-source-registry-v1.json`.

### Phase C — authentic collector

- [x] Convert strict real Pi JSON events into canonical student messages,
  including successful finals, terminal protocol errors, and empty tool output.
- [x] Ingest SHA-pinned explicit post-action workspace, source-ledger,
  observation, and action-linkage receipts.
- [ ] Capture those state receipts inside the live tool controller.
- [ ] Wire the no-progress decision into the live controller.
- [ ] Add atomic task leases, interruption recovery, and deterministic replay.
- [x] Run fresh fake-event adversarial tests and structural parsing checks
  against one historical success and one historical terminal-cycle trace.

Current Phase C ingestion files:

- `ndm/e97_phase_c_collector.py`;
- `scripts/collect_e97_onpolicy_rollout.py`;
- `tests/test_e97_phase_c_collector.py`.

This slice is ingestion-only and does not yet authorize GPU actors.

### Phase D — eight-GPU systems slice

- [ ] Launch one actor per GPU with NUMA and isolated Triton caches.
- [ ] Collect the 128-task systems slice.
- [ ] Validate replay, interruption, and all receipts.
- [ ] Review observed failure taxonomy before any teacher call.

### Phase E — teacher and correction pilot

- [ ] Freeze Luna/Terra prompts, revisions, price snapshot, and budgets.
- [ ] Generate bounded proposals from exact student failure states.
- [ ] Replay and reject deterministically.
- [ ] Seal the first correction and successful-trajectory authorities.

### Phase F — training and DAgger iteration

- [ ] Build the declared matched-token mixtures.
- [ ] Run acquisition and behavioral canaries.
- [ ] Select by saved-`x` behavior, not loss.
- [ ] Recollect failures from the selected updated checkpoint.
- [ ] Freeze V5 only after the complete recipe and gates stabilize.

## Immediate next action

Implement Phase B's source-registry and task-bundle schemas before downloading
or generating public-source tasks. Then implement the fake-event collector and
prove one fresh first-party task can travel through:

```text
task bundle -> isolated rollout -> state receipts -> detected failure ->
verified correction -> zero-loss-prefix SFT authority -> pack reload
```

Only after that vertical slice passes should the project acquire 128 tasks or
start eight GPU actors.
