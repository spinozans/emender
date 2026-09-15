# E97 4B Pi-native tool use and exact-copy programme

**Programme authority:** operator direction of 2026-09-15: build and track the
actual path from the current Pi-fronted OpenHands checkpoint to the bounded Pi
tool surface used in attended work, including web research and exact copying at
long delays.

This document drives execution. Progress and immutable receipts are recorded in
[`validation/e97-pi-native-tool-copy-progress.md`](validation/e97-pi-native-tool-copy-progress.md).
The earlier research questions in
[`EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md`](EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md),
especially RQ2, remain applicable.

## 1. Product target

The target is not arbitrary plugin discovery. It is a fixed, versioned Pi surface:

- local discovery: `fffind`, `ffgrep`;
- file/workspace actions: `read`, `bash`, `edit`, `write`;
- managed subprocesses: `process`;
- research: `web_search`, `source_check`, `fetch_content`,
  `get_search_content`;
- private reasoning and a public final response.

Exact installed schemas and descriptions, not these names alone, are authority.
A capture step must record the tool inventory from the same Pi runtime used for
collection/evaluation. Schema or extension-version changes create a new authority.

The model should research current, uncertain or externally verifiable questions.
It should prefer local discovery for repository questions and avoid gratuitous
web calls for arithmetic or fully supplied facts. Search results are observations:
the model must inspect them, cite or attribute them when appropriate, reformulate
failed queries, and never invent a successful search.

## 2. Representation

Keep the learned five-line assistant frame and private/public separation, but add
a **new Pi-native profile** whose permitted actions are the captured Pi tools plus
private `think` and terminal `finish` pseudo-actions. This is adjacent to, not a
mutation of, the immutable OpenHands-native codec.

For an external Pi action:

1. E97 emits one canonical five-line turn.
2. The provider parses it against the captured schema.
3. Pi executes the real registered tool.
4. On the next provider call, the exact Pi `toolResult` becomes causal native
   context before generation.
5. Argument values are never rewritten by JavaScript/Python transport.

`think` remains private and is resolved internally with a fixed synthetic
observation. `finish.message` becomes the public Pi assistant final. Every
external call/result pair is retained. Transport errors, tool errors and model
termination remain distinct.

## 3. Questions and frozen measurements

### PTCP-01 — Exact-copy baseline

Measure free generation, not teacher-forced likelihood, at approximately128,
1K,8K,32K and64K input tokens. Use held-out opaque strings and whole-template
separation for:

- verbatim needle copy;
- key-to-value selection among distractors;
- exact path emission in a tool argument;
- delayed use in a calculation;
- preservation of an unrelated multiline block during an edit;
- recall of an early tool result after later turns.

Report exact match by operation and distance. Transport replay is a separate
check and cannot count as model copying. Across explicit native task boundaries,
prior text is intentionally absent; persistent files or an explicit memory tool
must carry state.

### PTCP-02 — Tool-choice baseline

Freeze tasks where the correct first action is one of local search, read, shell,
edit/write, process, web search or direct final. Include contrast pairs that
change only whether information is current/external versus supplied/local.
Measure first-call schema/argument validity, observation-dependent continuation,
outcome and unnecessary-search rate.

### PTCP-03 — Web research baseline

Use real Pi search on a bounded frozen query set after source URLs/expected claim
checks are frozen. Include weather/current facts, multi-query research, direct URL
fetch, passage retrieval, empty/error results and query reformulation. Training
and evaluation queries, entities and answer timestamps must be disjoint.

### PTCP-04 — Repository discovery

Preserve the failed0/4 panel. Build the separately specified >=256-record
discovery/correction authority before any admission. Existing failed evaluation
traces remain excluded.

## 4. Data construction

Existing OpenHands data is **rehearsal and task-source material**, not a text-level
Pi conversion authority. An OpenHands task may enter Pi-native data only after its
solution is re-executed with the actual Pi tool and actual result bytes.

The candidate Pi-native authority should contain2,000–5,000 verified records:

|Target share|Capability|
|---:|---|
|35%|fuzzy discovery, read/edit/write, tests and shell|
|25%|web search/fetch, query reformulation and sourced synthesis|
|25%|exact copy, delayed retrieval, binding and preservation|
|15%|tool errors, recovery, process lifecycle and multi-tool composition|

Collection sources:

1. executor-verified authored/teacher trajectories;
2. verified re-execution of eligible OpenHands tasks using Pi tools;
3. current-model on-policy attempts with same-state verified corrections;
4. immutable OpenHands/conversation/retention/bridge rehearsal.

Never supervise tool results, system/user context or harmful student actions.
Authored failure prefixes and failed student actions are zero loss; supervise only
the verified suffix. Deduplicate exact token/mask sequences and report exposure,
not merely record counts.

## 5. Training decision

Do not choose the final mix from intuition. First complete PTCP-01 through03. The
baseline determines whether long-copy needs ordinary SFT, external memory, or an
architecture change, and how much Pi schema acquisition is required.

The currently proposed first tranche is32 updates at LR1e-5 from the exact
representation-bridge live-y checkpoint, using the exercised BF16 stochastic-
rounding Schedule-Free/checkpointed-CE path. Before launch, publish a concrete
admitted authority, exact source/target/exposure accounting, parent identity,
replay quotas and frozen gates. No retries or automatic extension follow a weak
result.

## 6. Acceptance

A candidate must simultaneously pass:

- exact-copy curves reported at every frozen distance, with no aggregate hiding;
- held-out Pi tool schemas and task families;
- current/external questions reliably selecting web research;
- local questions preferring local tools;
- authentic search/tool error recovery;
- repository direct/Pi capability and route parity gates;
- existing bridge fresh, composition, regression and two-task session gates;
- both saved-x/live-y conversation/native/tool retention;
- unchanged security, isolation, cleanup and checkpoint-state controls.

Passing a fixed weather/search demo alone is insufficient. Failure at long exact
copy may justify a bounded external-memory or hybrid-attention follow-up rather
than more repetitions of the same strings.

## 7. Ordered execution

1. Freeze/capture the actual Pi tool inventory and versions.
2. Implement and CPU-qualify the adjacent Pi-native codec/provider.
3. Build exact-copy and tool-choice panels; freeze before model sampling.
4. Run unchanged-checkpoint baselines and publish per-distance/per-family results.
5. Build verified Pi-native training candidates driven by those failures.
6. Audit masks, provenance, deduplication, overlap and fresh verifiers.
7. Publish the exact training authority and gate; then run the authorized tranche.
8. Re-evaluate all frozen panels and retain the outcome without retry.
9. Package an attended demo only from the checkpoint and interfaces that passed.
