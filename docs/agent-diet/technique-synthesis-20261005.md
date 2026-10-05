# Technique Synthesis: Agentic-Task Training Literature → Emender Sealed-Pipeline Design Specs

Date: 2026-10-05
Status: research synthesis (read-only; no code changed)
Audience: emender program (4B sequential-recurrence agent, verified-only RL)

Our machinery referenced throughout: GLM-5.3 authoring of task specs; mechanical
solvability proofs; zero-collision screens; task pool; sealed validators;
degeneracy screen (repetitive-analysis + finish-over-error rejection); teacher
(GLM-5.3) fresh solves on failures; receipts (verified episodes only); per-lane
masked-SFT steps (fresh ScheduleFree optimizer each cycle, no-signal skip below
loss 0.35); 8-GPU DiLoCo soup merges; held-out Stage-B reads every merge; tool
surface list_files/read/write/finish (five-frame protocol); eval harness with
bash/grep/process/web tool choices on held-out panels.

---

## 1. TaskCraft — difficulty-scalable verifiable agentic task generation

Sources: [arXiv:2506.10055](https://arxiv.org/abs/2506.10055),
[HTML v2](https://arxiv.org/html/2506.10055v2)

### (a) What it is
TaskCraft (Shi et al., 2025) is an automated workflow that generates
difficulty-scalable, multi-tool, verifiable agentic tasks with execution
trajectories — ~36,000 tasks at varying difficulty. Its two levers are
*depth-based extension* (chaining tasks so each depends on the prior output) and
*width-based extension* (merging independent subtasks), on top of "atomic" tasks
resolvable by a single tool invocation.

### (b) Their mechanism
- **Atomic task schema:** `q = f(i_T, R) → a` — a question `q` sampled from a
  tool-input index `i_T` (paper titles, image paths, PDF names) and a relation
  `R`, with a candidate answer `a` derived by executing the tool on the
  indexed content. An atomic task is "resolved with a single target tool
  invocation."
- **Depth extension:** recursively build `q^{n+1} = f(q̂^{n+1}, R^n) → a` where the
  intermediate question `q̂^{n+1}` resolves to the *previous* task's input
  `i_T^n`. A search agent retrieves supersets of `i_T^n` to mitigate cyclic
  generation; an LLM derives a superset index `i_T^{n+1}` and relation
  `R^{n+1}` (hierarchical relations like *contains*, *part_of*).
- **Width extension:** `(q₁ + q₂) → (a₁ + a₂)` — an LLM merges and rephrases two
  question strings into one multi-goal task.
- **Verification:** *not* symbolic — it is agentic + LLM-judged. Atomic tasks are
  kept iff a task agent's judged score (2/1/0) strictly exceeds a no-tools
  infer-LLM's score and the agent answer is non-zero (i.e., the task *genuinely
  requires tools*). Extensions are verified "purely through linguistic
  analysis": strict-superset validity (rejecting synonym pseudo-supersets) and
  **information-leakage prevention** (the merged query must not reveal the
  golden answer). Only incremental components undergo agentic validation per
  extension step.
- **Quality loop:** rejection sampling over optimized prompts; bootstrap
  few-shot example selection lifted atomic pass rate 54.9%→68.1% and
  depth@6 41.0%→51.2%; TaskCraft composition beat GPT-4.1 direct generation
  43.0% vs 18.5% pass rate.
- **Trajectories:** recorded while iteratively expanding/validating; SFT on
  3,202 multi-hop tasks with **content masking** over tool-result contexts
  (Chain-of-Action), giving Qwen2.5-3B-Base +14.0% avg Exact Match on
  HotpotQA/Musique/Bamboogle and better RL initialization vs Search-R1.

### (c) What we port into our sealed pipeline
- **P1 — Depth-composition grammar for GLM-5.3 authoring.** Extend the task-spec
  schema with a `chain` field: an ordered list of atomic subtasks where
  subtask k's `read`/`write` inputs must name artifacts produced by subtask
  k−1 (our filesystem artifacts play the role of TaskCraft's `i_T` indexes).
  The mechanical solvability prover runs per-subtask (incremental validation —
  TaskCraft's cost bound), and the composed task enters the pool only after the
  full chain is proven. Depth = number of chained subtasks; width = number of
  parallel independent goals in one spec.
- **P2 — Tool-necessity screen.** Add a leg to the sealed admission screen: run
  GLM-5.3 *with the tool surface stripped* (workspace-only) on the candidate
  spec; if it produces the target answer without tool frames, reject the task
  as not genuinely requiring the list_files/read/write/finish loop. This is a
  mechanical proxy for TaskCraft's agent-score > no-tool-LLM-score rule and
  directly attacks our repetitive-analysis degeneracy (tasks solvable by
  pattern-matching without exploration).
- **P3 — Information-leakage screen.** Before pool admission, mechanically check
  that the composed instruction does not contain (as literal or trivial
  paraphrase — hash/normalized-substring) any golden answer token required by
  the validator. Zero-collision screens gain a new axis: no answer leakage, not
  just no task collision.
- **P4 — Content masking in masked-SFT.** Our per-lane masked-SFT already masks
  loss; adopt TaskCraft's Chain-of-Action discipline explicitly: tool-result
  frames (read outputs) are masked targets; only decision frames (tool choice,
  write payloads, finish content) carry loss. (If our masked-SFT already does
  this, adopt it as a documented invariant.)
- **Difficulty metadata:** store `{depth, width}` on each task spec so the pool
  can stratify and Stage-B panels can be difficulty-balanced.

### (d) Effort and risk
- Effort: P1 is the largest (schema + prover extension + pool changes),
  ~3–5 engineer-days; P2/P3 are screen additions, ~1–2 days each; P4 ~0.5–1 day
  verification. All CPU-side.
- Risk: depth chains raise horizon length, which stresses the five-frame
  protocol and can increase finish-over-error rates (mitigated by our degeneracy
  screen + P2). TaskCraft's own verification is LLM-judged — we deliberately
  keep our *mechanical* proofs as the admission authority and use only the
  structural ideas (composition grammar, anti-leakage), not their judge.

---

## 2. AppWorld — state-verified simulated app worlds (and a minimal notes/office world)

Sources: [arXiv:2407.18901](https://arxiv.org/abs/2407.18901),
[ACL 2024 long paper](https://aclanthology.org/2024.acl-long.850/),
[appworld.dev](https://appworld.dev/), [GitHub](https://github.com/StonyBrookNLP/appworld)

### (a) What it is
AppWorld (Trivedi et al., ACL 2024 Best Resource Paper) is an execution
environment of 9 everyday apps (60K LOC) operable via 457 APIs over 101 SQLite
tables, populated with the digital lives of ~100 fictitious users, plus 750
tasks evaluated by *state-based unit tests* rather than trajectory comparison.
It deliberately includes distractor data and hurdles so careless agents fail.

### (b) Their mechanism
- **Task instantiation:** each Task Scenario blueprint runs a Task Generator with
  three programs — **Setup** (materializes a task-specific copy of the Base DB
  and freezes a start timestamp), **Evaluation** (assertions), **Solution**
  (reference). Each instantiated task = supervisor identity + instruction +
  frozen Task DB + evaluation data with expected values.
- **State verifier construction:** evaluation is a small set of programmatic
  assertions (avg 5.9–8 per task, max 24) over the **database diff** `D^Δ`
  between final and starting state. The pass condition is `D^Δ ⊆ C^expect ∪ C^allow`
  — *all* expected changes present *and no other changes* except allowed ones.
  Assertions look like `test.case(music_player.is_playing, "is_truthy")` or
  `test.case(queue_song_ids, "==", set(...))`, each paired with a natural-language
  statement; the diff is computed by a fast hash-based process.
- **Collateral damage:** the "no unexpected changes" half of the subset check
  catches unintended side effects (e.g., deleting a wish list while completing a
  return).
- **Distractors and hurdles:** Setup deliberately seeds near-miss data (e.g.,
  multiple past orders in different colors/sizes) and realistic obstacles
  (expired default payment card); contrast sets across scenarios vary one detail.
- **Multiple valid solutions:** because only final state is checked, "an order
  receipt may be downloaded from its Amazon API or its confirmation email" —
  process freedom is structural.
- Termination: the agent calls `apis.supervisor.complete_task()` (with an
  `answer` argument for the ~15% QA tasks). GPT-4o solves only ~49% of "normal"
  and ~30% of "challenge" tasks.

### (c) What we port into our sealed pipeline
- **P1 — End-state diff validator (the single highest-value port).** Treat the
  seeded workspace (the files our agent sees via list_files/read/write) as
  AppWorld's Task DB. Extend the task-spec schema with:
  - `seed_manifest`: content-hash manifest of the frozen start state;
  - `expect`: set of file-path + content-hash (or JSON-path/regex) assertions
    that must hold at finish;
  - `allow`: explicit allowlist of paths/fields permitted to differ without
    failing (e.g., scratch files the validator tolerates).
  The sealed validator recomputes the workspace diff at `finish` and asserts
  `D^Δ ⊆ expect ∪ allow` — exactly AppWorld's subset check, implemented as pure
  hashing/string ops, no LLM in the loop. This *subsumes* and sharpens our
  current validators and gives the degeneracy screen a mechanical "collateral
  damage" leg: writes outside the expected surface are rejections even if the
  primary goal is met.
- **P2 — Minimal "notes/office app" world.** A world our sealed pipeline can
  author cheaply inside the current tool surface: a workspace directory tree
  playing "notes/office app": structured `notes/*.md` or `*.json` files with a
  tiny documented CRUD semantics (note ids, titles, tags, created/updated
  timestamps, a trash folder instead of delete). Setup scripts seed the tree;
  distractor files carry near-miss titles/tags; hurdles include an id-conflict
  rule (write to existing id must go through the trash convention) enforced by
  the validator. Tasks = retrieve, aggregate, reorganize, dedupe, and
  merge-note operations with state-diff checks. This is the authoring template
  for an AppWorld-style micro-world with zero new tools.
- **P3 — Contrast sets.** When GLM-5.3 authors a task, automatically emit a
  sibling task that flips one distractor detail (which of two similar notes is
  the correct one) so the pool contains minimal pairs. Pairs cross-check each
  other: if the policy passes both, it is reading; the zero-collision screen
  must treat contrast siblings as one family to avoid double-counting supply.
- **P4 — QA-answer channel.** Mirror `complete_task(answer)`: our `finish` frame
  already carries a payload; make validators distinguish write-state tasks
  (diff check) from answer tasks (exact/normalized-match on the finish payload),
  and mixed tasks (product of legs — see tau-bench).

### (d) Effort and risk
- Effort: P1 ~2–4 engineer-days (manifest hashing + validator leg + spec schema);
  P2 ~2–3 days (seed/distractor authoring templates + a dozen exemplar tasks);
  P3 ~1 day; P4 ~0.5 day.
- Risk: `allow` lists that are too loose admit collateral damage; too tight and
  multiple-valid-solution freedom is lost (AppWorld's own lesson: check *what*,
  not *how*). Mitigate by authoring `expect` from the teacher's fresh solve
  (see tau-bench P2) plus a manual audit of the first ~50 tasks.

---

## 3. tau-bench / tau2-bench — policy documents, API tools, user simulation, factorized rewards

Sources: [arXiv:2406.12045](https://arxiv.org/abs/2406.12045),
[sierra-research/tau-bench](https://github.com/sierra-research/tau-bench),
[tau2-bench docs/evaluation.md](https://github.com/sierra-research/tau2-bench/blob/HEAD/docs/evaluation.md),
[tau2 evaluator AGENTS.md](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/evaluator/AGENTS.md)

### (a) What it is
τ-bench (Yao et al., 2024) benchmarks agents in dynamic conversations with an
LLM-simulated *user*, where the agent has domain API tools and a *policy
document* it must obey. τ2-bench generalizes it (airline/retail/telecom) with a
factorized, per-task evaluation criteria schema. Its two signature ideas for us:
**reward = product of orthogonal components** (state check × communication
check), and **reference trajectories used to derive target state, never as a
required path**.

### (b) Their mechanism
- **Dual-control loop:** the agent sees the policy document + API docs; the user
  simulator (gpt-4o by default; strategies `llm`, `react`, `verify` — a
  verification step on the user's own reply — and `reflection`) sees the task
  instruction, user persona, and scenario. Reward compares the **database state
  at the end of the conversation with the annotated goal state**, plus checks
  that required information was communicated.
- **Factorized reward (tau2):** a task's final reward is the **product** of the
  components listed in `evaluation_criteria.reward_basis`. Components:
  - `DB` (`EnvironmentEvaluator`): the predicted environment's **DB hash** must
    match the hash of a fresh env *after replaying* `evaluation_criteria.actions`
    — i.e., the reference trajectory is executed to *derive* the gold end
    state; "any sequence of tool calls that produces an equivalent DB end state
    passes."
  - `ENV_ASSERTION`: programmatic assertions on the final environment.
  - `COMMUNICATE` (`CommunicateEvaluator`): every required string in
    `communicate_info` must appear in the agent's messages (substring match).
  - `NL_ASSERTION` (`NLAssertionsEvaluator`): LLM-judge statements —
    explicitly **experimental/WIP**.
  - `ACTION` (`ActionEvaluator`): require matching the reference tool calls —
    used in ~9 of ~100 banking tasks and **"not used at all in airline/retail/
    telecom"** because it forbids alternative solutions.
- **Refusal is full reward:** in the worked example (read-only actions only),
  the correct behavior is to refuse cancellation; an agent that does nothing
  but politely refuse scores 1.0.
- **pass^k:** a reliability metric — the probability the agent succeeds on all
  k independent trials (user-simulation nondeterminism makes single-pass
  scores noisy; gpt-4o retail pass^8 <25%).
- **Fault telemetry:** post-hoc LLM labeling of failures by responsible entity
  (user/agent/environment) and type (goal_partially_completed,
  used_wrong_tool, used_wrong_tool_argument, took_unintended_action).

### (c) What we port into our sealed pipeline
- **P1 — Factorized validator schema.** Upgrade sealed-validator task specs to a
  tau-style `evaluation_criteria` object: `reward_basis: ["STATE", "ANSWER"]`
  (our analogues of DB and COMMUNICATE), with the rule **reward = product of
  enabled legs**. `STATE` = the AppWorld-style diff check; `ANSWER` = required
  strings/normalized-match that must appear in the `finish` payload. A
  leg-listing makes the sealed contract explicit per task and lets Stage-B
  report per-leg pass rates for diagnostics.
- **P2 — Replay-to-derive-gold (teacher fresh solves as reference actions).**
  Our teacher (GLM-5.3) fresh solves are exactly tau's `evaluation_criteria
  .actions`. Port the mechanism: run the verified teacher trajectory through
  the sealed validator *once, offline*, snapshot the resulting end state
  (hash manifest + finish payload) and store that snapshot as the task's gold —
  then never compare the policy's *trajectory* to the teacher's, only its end
  state. This mechanically preserves multiple-valid-solutions and removes any
  temptation to grade against the teacher's path. (Caveat: this derives gold
  from one successful solve; combine with author-time expected-state specs
  from AppWorld P1 where the author knows the answer.)
- **P3 — The GLM-judge leg stays diagnostic-only.** If we ever want free-form
  quality statements, port tau2's `NL_ASSERTION` *as a diagnostic channel that
  never enters the reward product* — exactly as tau2 marks it WIP/experimental.
  Verified-only receipts remain gated by mechanical legs only.
- **P4 — Refusal tasks.** Author a task family where the *correct* episode is a
  clean refusal: seeded workspace makes the instruction infeasible or
  policy-violating; the validator asserts *no writes* (empty diff) *and* a
  refusal reason present in the finish payload; reward 1.0 for doing nothing
  but refusing. This is the positive-space counterpart of our finish-over-error
  degeneracy screen: it teaches legitimate, informative refusals rather than
  merely punishing early finishes.
- **P5 — pass^k on Stage-B.** We already read Stage-B every merge; add pass^k
  (k=2–4 repeated attempts per panel task) as a *reliability* signal alongside
  mean pass. Sudden pass^k drops at stable mean flag instability — a cheap
  merge-selection criterion before promoting a soup.
- **P6 — Fault telemetry.** Adapt tau's fault categories (wrong tool, wrong
  argument, unintended write, goal-partially-completed) as a mechanical
  taxonomy over receipts and rejects: our diff validator can mechanically emit
  `unintended_write` and `goal_partial` labels (writes present but expect-set
  incomplete). LLM-labeled fault assignment stays out of the sealed loop.

### (d) Effort and risk
- Effort: P1+P2 ~3–4 engineer-days (validator schema, offline gold-snapshot
  tooling); P4 ~1–2 days (authoring template + validator leg); P5 ~1 day (Stage-B
  reporting); P3/P6 minimal.
- Risk: gold-from-teacher replay bakes in teacher quirks if the teacher solved
  via an unusual-but-valid path — acceptable because we check end state only.
  Refusal tasks risk teaching *over*-refusal; counterbalance by keeping them a
  small fixed fraction of the pool and monitoring refusal rate on ordinary
  tasks in Stage-B.

---

## 4. OpenThoughts-Agent (OT-Agent) — SFT→RL data recipes for small agentic models

Sources: [arXiv:2606.24855](https://arxiv.org/abs/2606.24855) ("OpenThoughts-Agent: Data Recipes for Agentic Models"),
[HTML](https://arxiv.org/html/2606.24855v1),
[launch blog](https://www.openthoughts.ai/blog/agent),
[project blog](https://www.openthoughts.ai/blog/openthoughts-agent),
[GitHub](https://github.com/open-thoughts/OpenThoughts-Agent)

### (a) What it is
OpenThoughts-Agent is an open data-curation project (plus Qwen3-8B/32B models;
OpenThinker-Agent-32B reaches 26.2% Terminal-Bench 2.0, 54.0% SWE-Bench
Verified) built as a two-stage pipeline: SFT on curated traces from strong
teachers, then RL on verifiable tasks. Its >100 ablations make it the best
current evidence on *what actually moves small-model agentic ability*.

### (b) Their mechanism
- **RL task design:** SWE-Smith/R2EGym-style — Docker-captured repository,
  flawed commit with failing tests, natural-language problem statement; rewards
  are "standard binary rewards on verifier success." The best source
  (pymethods2test) was chosen for reproducibility, uniform build environment,
  and "appropriately moderate difficulty ceiling."
- **Task sourcing dominates everything:** task-source choice moved results "up to
  ~30 pp on SWE-Bench Verified-100"; mixing Top-4–Top-8 sources beats Top-1;
  adding more sources hurts (Top-16 "hurts on every benchmark"). LLM task
  filters gave +3 pp; **filtering tasks to those that GPT-5 needs more tokens
  to solve** found tasks worth ~+3 pp.
- **Trace filters:** keep rollouts ≥5 turns; drop timeout and subagent traces —
  at matched ~145M-token budget this beat random subsampling by +3.5 pp avg.
  More rollouts per task plateaued (31.6K→100K); *surface-form augmentation*
  of base problems (~902→21K variants from 997 bases) kept improving. Best
  teacher was GLM-4.7-AWQ (GPT-5.3-Codex was a *worse* teacher, ~−5% on TB2).
- **RL stage:** async RL with **RLOO** (not GRPO), binary verifier rewards, 8B
  cold-start from SFT. Data-source ablation spanned 7.6 pp (pymethods2test
  reward 0.21→0.46). Emergent exploration post-RL: think tokens +116%, tool
  calls +31%, self-correction +81%; judged "legitimate exploration, not reward
  hacking" because per-call error rate rose only +4.1 pp. But the hero run's
  reward peaked ~0.51 then **collapsed to ~0.13** as the policy over-explored
  (agent-timeout rate ≈80%); the deployed checkpoint is taken pre-collapse.
- **SFT+RL > either alone;** base Qwen3-8B "is unable to benefit from agentic
  RL" without SFT; "undertrained" SFT models benefit more from RL.

### (c) What we port into our sealed pipeline
- **P1 — Teacher-solve-cost as a difficulty/quality filter.** Our teacher fresh
  solves already exist as a by-product. Record per task: teacher attempts,
  retries, wall-clock, and frame count. Adopt OT-Agent's token-cost signal:
  prefer pool tasks in the mid-band of teacher cost (not one-frame trivia, not
  unsolvable-ish marathons). This is a mechanical difficulty dial that requires
  zero new authoring machinery.
- **P2 — Trace-admission filters on receipts.** Enforce on receipts (in
  addition to verified-only): minimum interactivity (≥2 tool frames beyond
  list_files, or a policy-scaled floor), and reject degenerate lengths
  (e.g., trivially short solves for tasks tagged non-trivial). OT-Agent's
  +3.5 pp at matched budget is the evidence this beats naive accumulation.
- **P3 — Leading-indicator collapse monitor.** OT-Agent's reward collapse was
  preceded by an explosion of timeouts. Our analogue leading indicators:
  finish-over-error rejects and agent-timeout equivalents (empty/abort
  episodes) as a fraction of attempts. Track per merge in Stage-B reads; if the
  rate climbs steeply while verified-reward is flat or rising, hold the
  previous merge (pre-collapse checkpoint selection, exactly OT-Agent's
  pymethods2test-45 move).
- **P4 — Surface-form augmentation over new-task sprawl.** At fixed budget,
  generating many surface variants of a *verified* base task (reworded
  instructions, renamed files/ids, permuted distractors — keeping the
  validator's semantic content fixed) beat more rollouts per task in their
  ablations. Port as a pool tool: variant generator parameterized on the
  task spec, with the zero-collision screen keyed on *validator semantics*
  (not just surface text) so variants are admitted without colliding.
- **P5 — Keep the SFT-before-RL invariant.** Their result that base models
  cannot benefit from agentic RL without SFT cold start, and that SFT+RL beats
  either alone, is external validation of our masked-SFT→verified-RL loop
  ordering. No change; document as a defended invariant.
- **Not found in the paper (honest gap):** an explicit entropy-maintenance
  mechanism or named exploration-bonus technique — exploration *emerged* and
  was measured, not engineered. We should not claim to be porting "their entropy
  technique"; the portable artifact is the *monitor* (P3), not a regularizer.

### (d) Effort and risk
- Effort: P1 ~1–2 days (telemetry fields + pool filter); P2 ~1 day; P3 ~1 day
  (Stage-B metric + merge-hold rule); P4 ~2–3 days (variant generator +
  semantics-keyed collision screen).
- Risk: teacher-cost filtering can bias the pool toward whatever GLM-5.3 finds
  hard rather than what the 4B needs — monitor by stratified Stage-B pass
  before/after. Variant generation risks collision-screen erosion if keyed on
  surface text; key on validator-semantics hashes. Collapse monitoring is
  observational and safe.

---

## 5. SynthTools — synthetic tool environments with validated tool contracts

Sources: [arXiv:2511.09572](https://arxiv.org/abs/2511.09572),
[HTML v2](https://arxiv.org/html/2511.09572v2),
[GitHub namkoong-lab/SynthTools](https://github.com/namkoong-lab/SynthTools),
[SynthTools-Tasks dataset card](https://huggingface.co/datasets/SynthTools/SynthTools-Tasks)

### (a) What it is
SynthTools (Castellani, Ye, et al., Columbia/Oumi) is a fully LLM-based pipeline
spanning the whole lifecycle of synthetic tool-use environments: 100 fields,
6,800 environments, 73,883 validated tools (from 85,447 proposed), and 79,925
verifiable tasks. Its key contribution for us is not the LLM simulation (we
execute real tools) but the **tool-contract schema, tool validation gauntlet,
mini-task chaining with grounding checks, and difficulty control via distractor
tools**.

### (b) Their mechanism
- **Hierarchical generation:** Field → Sub-domain → Task family → Tools. A tool
  is a tuple `(name, description, parameters, usage, failure_modes,
  output_schema)` with an I/O contract of preconditions, postconditions, and
  error modes.
- **Tool Simulator + Validator:** two-stage simulation — programmatic schema
  validation (AST parsing of call structure, then LLM semantic-constraint
  checks) followed by LLM response generation. The Validator is an engineered
  LLM judge over (tool spec, test call, simulator response) returning
  structured correct/incorrect + confidence + rationale; stress-tested with 3
  failure modes and 3 success modes; 97% accuracy with **0% false-positive
  rate**; tools failing stress probes are filtered (57% of tools had zero
  errors over 8–10 stress calls).
- **Task construction (bottom-up):** per validated toolset, propose sequential
  tool chains; per chain step, generate a *mini-task* + ground-truth tool call;
  an agent attempts it in the simulator; verify by **exact call comparison plus
  an argument-grounding check** ("all arguments grounded in the task
  description or previous interactions, regenerate if not"); on success, update
  environment state. Chaining mini-tasks yields "an explicit solution
  trajectory and automatic verifiability by comparing the final state of the
  environment and the sequence of tool-calls." Each packaged task includes:
  task description, minimal tool set / candidate catalog, ground-truth tool-call
  sequence, and initial/final environment state.
- **Difficulty control:** (1) number of mini-tasks, (2) explicitness vs
  ambiguity of instructions, (3) **adding irrelevant-but-plausible tools to the
  candidate pool**, plus sequencing favoring multi-step workflows.
- **Training use:** midtraining Qwen3 0.6B/1.7B/4B on ~23K trajectories (~200M
  tokens) from a GPT-OSS-120B agent lifted ACEBench/BFCL/API-Bank across all
  sizes (e.g., 4B API-Bank 0.761→0.813), with synthetic-to-real transfer on
  real-API benchmarks.

### (c) What we port into our sealed pipeline (focus: bash/grep/process/web tools)
- **P1 — Tool-contract schema for every new executor tool.** When the gym
  executor gains bash/grep/process/web, each tool gets a SynthTools-style sealed
  contract file: `name, description, parameter_schema, preconditions,
  postconditions, failure_modes, output_schema`. The mechanical solvability
  prover consumes contracts: a task is provable only if every needed call's
  preconditions are reachable and its postconditions imply the validator's
  `expect` set.
- **P2 — Tool validation gauntlet (admission for tools, not tasks).** Before a
  new tool enters the pool's grammar, run stress probes: 3 failure-mode probes
  (bad args, violated precondition, unavailable resource) and 3 success-mode
  probes (known input, novel input, boundary), executed CPU-side against the
  real executor. A tool passes admission only if failure probes fail
  deterministically and success probes succeed deterministically (idempotent
  hashing of outputs). This is SynthTools' 0%-false-positive discipline
  translated from LLM simulation to real execution, which is strictly easier
  for us.
- **P3 — Argument-grounding check in authoring.** Add a leg to the mechanical
  solvability proof: every literal the golden path requires (file names, ids,
  search keys, URLs) must be *derivable* — present in the seeded workspace or
  the task text — never invented by the author. This kills
  hallucination-required tasks at authoring time and complements our
  zero-collision screen with a "no unreachable constants" screen.
- **P4 — Distractor-tool difficulty dial.** For tasks exercising tool choice,
  seed the candidate surface with irrelevant-but-plausible affordances
  (distractor files, distractor processes, decoy web endpoints on held-out
  panels). SynthTools' third difficulty knob transfers directly to our eval
  harness's bash/grep/process/web panels.
- **P5 — Task packaging invariant.** Task specs (and receipts) carry:
  instruction, minimal tool set, ground-truth call sequence (teacher's,
  diagnostic only), and initial/final state manifests. Our P2 from tau-bench
  already converts the ground-truth sequence into end-state gold; SynthTools
  confirms the packaging shape.

### (d) Effort and risk
- Effort: P1 ~2–3 days per tool batch; P2 ~2 days (probe harness is reusable);
  P3 ~1–2 days (prover extension); P4 ~1 day; P5 ~0.5 day.
- Risk: contracts written after the tool exists tend to drift from behavior —
  the probe gauntlet (P2) is the enforcement. Real bash/process tools are less
  deterministic than LLM-simulated ones (timestamps, PIDs); hashing must
  normalize or mask volatile fields, or probes will false-reject. Process/web
  tools on CPU-side eval must be hermetic (no network on the training side).

---

## 6. AgentSynth + OpenComputer — information-asymmetry composition and verifier-grounded task synthesis

Sources: [AgentSynth arXiv:2506.14205](https://arxiv.org/abs/2506.14205),
[AgentSynth HTML v2](https://arxiv.org/html/2506.14205v2),
[ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/fb0f9005938c5a80b1114a42156e99e2-Paper-Conference.pdf),
[OpenComputer arXiv:2605.19769](https://arxiv.org/html/2605.19769v1),
[OpenComputer site](https://echo0715.github.io/OpenComputer/),
[OpenComputer GitHub](https://github.com/echo0715/OpenComputer)

### (a) What it is
**AgentSynth** (Wei et al.) synthesizes 6,000+ long-horizon computer-use tasks by
exploiting *information asymmetry*: subtasks are generated and solved forward
step-by-step (easy), then *summarized* into one long-horizon instruction that is
hard to solve from scratch. **OpenComputer** (Yale NLP Lab) builds verifiable
software worlds by coupling real desktop apps to *programmatic state-verifier
endpoints*, with a *self-evolving* verification layer that repairs checkers
using execution-grounded feedback, and a *verifier-first* task-generation
pipeline.

### (b) Their mechanisms
**AgentSynth:**
- Six LLM agents: **task proposer** (persona + screenshot → small, specific,
  safe task), **executor** (ReAct GPT-4.1 planner + computer-use-preview
  grounding, ≤10 steps), **verifier** (WebJudge-style: extract key requirements,
  select key screenshots, output binary success + completion %; 88% human
  agreement, κ=0.74; accepts only 12% of near-miss perturbations), **reviser**
  (if partially successful, rewrite the task description to match what actually
  happened), **follow-up proposer** (build on prior state; told of failures so it
  proposes simpler), **summarizer** (fold the subtask chain into one abstract
  long-horizon task).
- Difficulty = number of summarized subtasks (levels 1–6): horizon 5→45 steps,
  apps 1.2→3.3, app switches 0.5→4.3. The asymmetry, quantified: their
  generation success stays 65/57/52% at levels 1/3/6 while *evaluation* success
  falls 62%→14%; direct one-shot generation of hard tasks collapses to 11%
  success. Cost ≈$0.60/trajectory.
- Execution-based reject sampling: only subtasks that pass verification are
  kept; failures are revised or downgraded rather than discarding the chain.

**OpenComputer:**
- **State verifiers:** per-app synthetic Python modules in-sandbox exposing CLI
  subcommands with JSON output — `query` endpoints (inspect state: content,
  preferences, history, file I/O, metadata via SQLite, D-Bus, accessibility
  trees, saved files) and `check-*` endpoints; 17.7 endpoints/app avg.
- **Self-evolving verification:** ~15 calibration tasks/app; a strong agent's
  trajectories are cached; an LLM evaluator produces criterion-level reference
  verdicts; the programmatic checker runs on the same final state;
  disagreements attributable to checker bugs feed a bounded debug-fix-retry loop
  (never altering cached trajectories or task objectives). Result: 89.4%
  repair rate on checker errors; human–checker agreement 85.2%→94.1%.
- **Verifier-grounded task synthesis (4 stages):** (1) propose candidate tasks
  from realistic user goals *without conditioning on verifier endpoints*;
  (2) filter for complexity and data-generatability (reject too-short, overly
  linear, trivial, or uninstantiable tasks); (3) **ground in the verifier** —
  keep if an endpoint can check the outcome, otherwise extend the verifier with
  a new endpoint; (4) materialize artifacts. Tasks excluded when criteria are
  not fully verifiable (17 excluded). Periodic per-feature reviews prevent
  coverage collapse. Task instance τ = (instruction, environment, checks);
  reward R = N_pass/N_total with 6.9 checks/task.
- Head-to-head: hard-coded verifiers matched human verdicts on 113/120 tasks vs
  95/120 for an LLM judge; checklist agreement 97.3% vs 92.2%.

### (c) What we port into our sealed pipeline
- **P1 — Information-asymmetry composition for GLM-5.3 authoring.** Change the
  authoring flow for long-horizon tasks: GLM-5.3 authors a *forward chain* of
  small verified subtasks (each individually proven by the mechanical solvability
  prover — incremental cost), then a summarizer prompt folds the chain into one
  abstract instruction; the validator still checks the union of end-states.
  This keeps authoring/proving success high (AgentSynth: flat ~52–65%) while the
  policy faces a genuinely hard composed task. The `depth` field from TaskCraft
  P1 and this are the same machinery; AgentSynth supplies the
  generate-easy/solve-hard evidence and the reviser idea.
- **P2 — Reviser role for the teacher leg.** When a teacher fresh solve fails a
  task, do not just discard: route to a *revise-or-downgrade* step — either
  rewrite the task to the sub-scope actually completed (still mechanically
  proven) or drop its difficulty level. This preserves authoring throughput and
  matches our verified-only invariant (only proven revisions re-enter the pool).
- **P3 — Verifier-grounded authoring: the endpoint rule.** Codify OpenComputer's
  stage (3) as an authoring invariant: **a task may enter the pool only if its
  success condition is expressible as a conjunction of existing sealed
  validator endpoints** (file-manifest hash, path/JSON-path assertion, regex
  presence/absence, finish-payload match, empty-diff). If a candidate task's
  outcome is inspectable but not expressible, the correct response is to
  *extend the endpoint library* (a small, reviewed, mechanical addition), never
  to fall back to LLM judging. This turns our "mechanical solvability proof"
  into an endpoint-grounded proof with a closed, auditable surface.
- **P4 — Self-evolving (calibration) validator upkeep — offline and sealed.**
  Port the calibration loop as a *maintenance* procedure, not a runtime
  component: sample ~15 tasks per task family; run teacher solves; adjudicate
  disagreements between the teacher verdict and the sealed validator
  mechanically-then-human; repair validator endpoints offline (versioned,
  re-hashed, never mutated in place during a cycle). The 85.2%→94.1% agreement
  gain is the evidence this upkeep is worth scheduling per merge cadence.
- **P5 — Anti-coverage-collapse review.** Adopt their periodic per-feature task-
  set review: on each pool expansion, report task-family coverage (which
  validator endpoints, which tool frames, which difficulty bands are exercised)
  and require authoring prompts to target under-covered cells. Cheap, pure
  reporting.
- **P6 — LLM-judge placement.** OpenComputer's 113/120 vs 95/120 result is
  decisive external evidence for our sealed-mechanical-validator stance: LLM
  judges appear only in the offline calibration loop (P4) and tau-style
  diagnostics, never in receipt gating.

### (d) Effort and risk
- Effort: P1 ~3–5 engineer-days (chain authoring + summarizer + incremental
  proving); P2 ~1–2 days; P3 ~2 days (endpoint registry + authoring screen);
  P4 ~3 days initial + ongoing per-merge upkeep; P5 ~1 day.
- Risk: composed/summarized tasks can drift from the sum of their parts (the
  summary must not change what the validator checks — enforce by keeping the
  validator anchored to the subtask end-state union, not the summary text).
  Calibration upkeep risks validator churn; bound it with version pinning and a
  fixed repair budget per cycle (OpenComputer's fixed evolution budget).

---

## Recommended adoption order with rationale

1. **AppWorld P1 (end-state diff validator) + tau-bench P1/P2 (factorized reward,
   replay-to-derive-gold).** Rationale: pure-mechanical, sealed, small effort,
   and it upgrades the *quality of every subsequent verified episode* — the
   foundation receipts and Stage-B stand on. Everything else composes on top of
   a crisp end-state contract. Also adds the collateral-damage leg to the
   degeneracy screen for free.
2. **OpenComputer P3 (verifier-grounded endpoint rule) + P4 (calibration
   upkeep).** Rationale: makes authoring and validator maintenance principled:
   endpoint-grounded proofs, offline repair, and explicit evidence (94.1% vs
   92.2% / 113-vs-95 human agreement) that mechanical beats judged. Low risk,
   CPU-only.
3. **AgentSynth P1/P2 (information-asymmetry composition + reviser) with
   TaskCraft P1's depth/width grammar.** Rationale: the difficulty dial we
   currently lack, obtained by *composing already-proven atomic tasks* rather
   than asking GLM-5.3 to author hard tasks directly (AgentSynth: 52–65% vs 11%
   generation success). Highest authoring-throughput payoff.
4. **TaskCraft P2/P3 (tool-necessity + information-leakage screens) and
   tau-bench P4 (refusal tasks).** Rationale: cheap screens that close the two
   biggest degeneracy loopholes (tasks solvable without tools; answer leakage)
   and add the positive-space counterpart to finish-over-error rejection.
5. **OT-Agent P1–P3 (teacher-cost difficulty filter, trace-admission filters,
   collapse monitor / pass^k in Stage-B).** Rationale: measurement-side ports
   with strong ablation evidence (+3 to +3.5 pp analogues) and almost no
   authoring cost; the collapse monitor protects our per-merge promotion
   decisions as horizons grow.
6. **SynthTools P1–P4 (tool contracts, probe gauntlet, grounding check,
   distractor tools).** Rationale: gate these on the bash/grep/process/web
   executor expansion; the contracts and gauntlet must exist *before* new tools
   enter the authoring grammar, but porting them before the expansion is
   premature scheduling.
7. **AppWorld P2 (notes/office micro-world), P3 (contrast sets), OT-Agent P4
   (surface-form variants).** Rationale: richness/supply-side items; valuable
   but not load-bearing; sequence after the validator and composition
   machinery is stable.

Cross-cutting invariant: every port preserves sealed, mechanical, first-party
authoring; LLMs (GLM-5.3 or judges) touch authoring, teaching, and offline
calibration only — never receipt gating.

## What we deliberately do NOT adopt and why

- **Ingesting external task corpora verbatim (TaskCraft's 36K dataset,
  AppWorld's 750 tasks, SynthTools' 79,925 tasks, tau domain task files,
  OSWorld/AgentSynth tasks).** We author first-party only: external corpora
  bring contamination risk against the very held-out panels our eval harness
  uses, unverifiable provenance (their verifiers are not our sealed
  validators), and licensing ambiguity. We adopt their *generative recipes*,
  not their outputs.
- **LLM-as-judge in the reward path.** tau2 marks its LLM-judge leg
  experimental/WIP; OpenComputer measured hard-coded checkers at 113/120 human
  agreement vs 95/120 for the LLM judge; TaskCraft's own verification is
  LLM-judged and is the weakest part of an otherwise strong design. GLM judging
  stays in offline calibration and diagnostics only.
- **The ACTION reward type (requiring trajectory match).** tau2 itself disables
  it in all its main domains because it forbids valid alternative solutions —
  antithetical to our multiple-valid-solution filesystem world and to the
  exploration we want the 4B to retain.
- **LLM-simulated tools (SynthTools' Tool Simulator).** We can execute real
  tools CPU-side in a hermetic executor; simulation would add a
  distribution-shift layer (their own reason for the 0%-false-positive validator
  gauntlet) with no compensating scale benefit at our pool sizes.
- **Vision/computer-use machinery (AgentSynth's OSWorld/pyautogui stack,
  OpenComputer's GUI loop).** Out of scope for our text-protocol five-frame
  agent; we take the *composition and verifier* ideas, not the modality.
- **Partial credit in receipt gating.** OpenComputer's R = N_pass/N_total is a
  good *diagnostic*; our verified-only binary receipt invariant stays. Partial
  credit leaks unverified gradient into masked-SFT and would undercut the
  no-signal-skip (0.35) logic.
- **Wholesale optimizer/RL-algorithm swaps (RLOO, GRPO).** OT-Agent's RL
  machinery is not portable evidence for changing our ScheduleFree
  masked-SFT + DiLoCo soup design; what transfers is their *data* findings
  (sourcing, filters, collapse monitoring), which is what we adopt.
- **OT-Agent-style raw SFT-scale scaling (100K traces, 24×A100 hero runs).**
  Their own ablation says rollouts-per-task plateaus and surface-form
  augmentation dominates; our verified-only receipts are deliberately scarce,
  and their evidence argues for *variant diversity over volume* — which we
  adopt in miniature — not for loosening verification to chase count.
- **Refusal/communication-heavy user simulation as a training environment.**
  tau's dual-control loop with an LLM user is a superb *evaluation* design but
  puts an LLM inside the episode loop, which our sealed, mechanical,
  verified-only pipeline forbids; we take the factorized reward and refusal
  task *shapes* without the conversational simulator.

---

## Research-process notes (source-quality disclosures)

- TaskCraft: mechanism detail from the paper's HTML (v2). Note their
  "verification" is LLM-based, not symbolic — a commonly mis-described point;
  we kept their structural ideas and rejected their judge.
- AppWorld: from the ACL 2024 camera-ready HTML; assertion examples and the
  `D^Δ ⊆ C^expect ∪ C^allow` subset rule are quoted from the paper.
- tau-bench/tau2-bench: paper abstract + official tau2 evaluation docs and repo
  inspection. The pass^k *formula* is defined in the paper (probability of
  success on all k trials), not restated in the fetched docs — treated as
  direct evidence from the abstract, interpretation for details.
- OpenThoughts-Agent: from the paper HTML (arXiv:2606.24855v1). Important
  correction to the task brief: the RL algorithm is **RLOO, not GRPO**, and no
  explicit entropy/exploration-maintenance mechanism is described — exploration
  was *measured as emergent*. An automated source_check on the RLOO/collapse
  numbers returned "unclear" (no passage-level markers), so those figures rest
  on the direct paper-HTML fetch; confidence medium-high.
- SynthTools / AgentSynth / OpenComputer: from paper HTMLs (2511.09572v2,
  2506.14205v2, 2605.19769v1) with numbers as quoted there. OpenComputer and
  AgentSynth report no *fine-tuning* results on their own trajectories
  (benchmark/evaluation results only) — their value to us is task/verifier
  machinery, which is exactly what we port.

## Missing evidence / open questions

- No public source states a "curriculum schedule" proven to transfer across
  papers; difficulty dials exist everywhere (TaskCraft depth/width, AgentSynth
  levels, SynthTools knobs) but validated *curriculum ordering* for a small
  model like ours is untested — we should treat difficulty metadata as a
  stratification tool first, a curriculum second.
- Whether end-state-only checking (AppWorld/tau style) sufficiently suppresses
  our repetitive-analysis degeneracy when combined with the tool-necessity
  screen is an empirical question only Stage-B can answer.
- OpenThoughts-Agent's collapse dynamics were observed under RLOO with 24×A100;
  whether our masked-SFT + DiLoCo cadence shows the same timeout-explosion
  signature as horizons grow is unknown — the monitor (OT-Agent P3) is the
  hedge, not a guaranteed early warning.

## Next steps (research-side)

- If the program adopts the endpoint rule (OpenComputer P3), a short follow-up
  survey of JSON-path/regex assertion libraries suitable for a sealed,
  CPU-side, hash-based validator implementation would de-risk the endpoint
  library spec.
- A measurement study of our existing teacher fresh solves to calibrate the
  teacher-cost difficulty bands (OT-Agent P1) before wiring the pool filter.
