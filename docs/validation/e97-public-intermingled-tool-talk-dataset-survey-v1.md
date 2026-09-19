# E97 — Public intermingled chat + tool-call dataset survey (v1)

**Purpose.** Survey of public Hugging Face datasets with the INTERMINGLED CHAT + TOOL-CALL
distribution — conversational requests that flow into tool calls and back to plain-text
answers — to teach a small (4B) coding/tool agent the *seam* between pure chat and
agent-trajectory behaviour. Our diet already has chat data and agent-trajectory data in
separate worlds; we need data where a casual user question leads to one tool call, the
result is read, and the answer continues in text.

**Evidence basis.** Dataset cards and Hub API metadata fetched 2026 (latest observed card
revision 2026-08-10). Scale numbers are taken verbatim from the `dataset_info`/card tables
where available; per-record assistant-token estimates are derived from card statistics and
labelled as estimates.

---

## Headline finding: the tool-use seam is already inside our admitted smoltalk2 manifest

`HuggingFaceTB/smoltalk2` (SFT config, 25 source datasets — the manifest our programme
already admitted for its conversation pool) contains **three tool-use splits in the same
repo and config**, carrying the `python_tools` / `xml_tools` tool-definition columns we
already have codec machinery for:

| Split | Rows | Total tokens | Avg turns | Avg assistant-response tokens/row | Provenance / license |
|---|---:|---:|---:|---:|---|
| `smolagents_toolcalling_traces_think` | 9,079 | 63.81 M | 5.34 | 682 | NEW smoltalk2 data, generated with DeepSeek-V3-0324, Apache-2.0 |
| `xlam_traces_no_think` | 59,962 | 29.4 M | 2 | ~456 | Upstream ≈ `Salesforce/xlam-function-calling-60k` (APIGen, CC-BY-4.0) |
| `hermes_function_calling_v1_no_think` | 8,961 | 11.38 M | 5.35 | ~468 | `NousResearch/hermes-function-calling-v1`, Apache-2.0 |

Direct evidence (card `dataset_info` + stats table): all three splits live under
`SFT/…-*.parquet` in the same `HuggingFaceTB/smoltalk2` artifact we already admitted; HF
states the whole mix was decontaminated against evaluation benchmarks and that all *new*
subsets are Apache-2.0 (existing public subsets inherit their upstream licenses).

**What this means for admission:** we used only the conversation pool (620 K records /
~800 M tokens), so the tool splits were almost certainly *excluded from the mixture we
actually consumed* — but they are inside an already-admitted, already-vetted artifact.
Admitting them is the **cheapest large win available**: same provenance, same schema
(`messages` + `chat_template_kwargs.{python_tools, xml_tools}`), no new dataset
onboarding, no new license review beyond the two upstreams already known to us
(CC-BY-4.0 via xLAM, Apache-2.0 otherwise). Estimated supervised budget for all three
splits: ~78 K records, **~38 M assistant tokens** (derived estimate: rows × avg response
tokens from the card's stats table — researcher inference, not a measured count).

Caveat to verify locally (one-line check): confirm which splits our admitted 620 K-record
conversation pool actually covers (the smoltalk2 no_think conversation splits sum to
~885 K rows, so our pool was a selection); and confirm the `xlam_traces_no_think`
`source` column matches Salesforce APIGen records after HF decontamination.

---

## Ranked survey table

| # | Dataset | Scale (rows) | License | Verification quality | Distribution fit (intermingled seam) | Rendering cost |
|---|---|---:|---|---|---|---|
| 1 | **smoltalk2 SFT tool splits** (smolagents_think / xlam_traces / hermes_fc_v1) | 78,002 | Apache-2.0 / CC-BY-4.0 (upstream xLAM) | Upstream APIGen 3-stage (xLAM, executed); HF decontamination on all; smolagents+hermes synthetic | smolagents & hermes: multi-turn, genuinely intermingled; xLAM mostly single-shot (weakest of the three) | **Low** — already in our admitted schema |
| 2 | **Agent-Ark/Toucan-1.5M** | 1,527,259 total (SFT config: 119,287) | Apache-2.0 | **Real MCP tool executions** (495 real MCPs, 2,000+ tools); per-record quality-assessment fields | High: multi-round, multi-turn, multi-tool incl. parallel calls; real results in records | Medium — messages stored as JSON strings, need parse + normalise |
| 3 | **nvidia/Nemotron-Agentic-v1** (a.k.a. Nemotron-Agentic-Tool-Use-v1) | 335,122 multi-turn convs (5.8 GB) | CC-BY-4.0 (ungated) | Synthetic, NVIDIA pipeline; "decide when to call tools, reason over tool outputs"; Glaive-FC-v2 upstream (Apache-2.0) | High: multi-turn conversational; decide-when-to-call emphasis (interactive_agent + tool_calling splits) | Low–medium — OpenAI-style messages + tools |
| 3b | **nvidia/Nemotron-SFT-Agentic-v2** (successor, 2026) | ~1.2 M trajectories (interactive_agent 278,880 + search 5,968 + tool_calling rest; ~25 GB) | CC-BY-4.0 / Apache-2.0 / MIT mix; "ready for commercial use" | Synthetic, curated pipeline; carries `parallel_tool_calls`, `chat_template_kwargs`, per-record metadata | High: single-turn + multi-turn + multi-step, decompose-goal → call → reason over output | Low–medium — same family as v1 |
| 4 | **DomofonResearch/Tool-Reasoning-31K** | 30,764 | Apache-2.0 | Aggregation of xLAM / ToolACE / Glaive / Nous-Hermes / When2Call via `hermes_reasoning_tool_use`; keeps original tool calls + results | **High for the mirror pattern**: 11,943 "relevance" scenarios = tools available but should NOT be called; 11,186 multiturn; 2,710 multistep; 61 % contain ≥1 call; reasoning-before-call present | Medium — reasoning traces + mixed upstream formats to normalise |
| 5 | **NexusProjectsAI/Nexus-Agents-ToolCalling** | 599,267 (v1) / stage1 60,185 / stage2-recovery 16,015 | Apache-2.0 (ungated) | Procedural, **schema-verified** (every call validated against its tool's JSON schema); released model gated by a 27-case behavioural interview eval (27/27 vs 13/27 base) | High-quality seam behaviour but **narrow domain**: 3 skills (setup interview, discovery, task generation); conversational "ask when ambiguous, call when clear" is exactly the seam | Low — OpenAI `messages`+`tools`; but diversity-limited |
| 6 | **Team-ACE/ToolACE** | 10 K–100 K rows, 26,507 APIs | Apache-2.0 | Dual-layer self-evolving verification (rule-based + model-based); execution-grounded per paper | Medium: single/multi-turn API-form dialogues; limited casual-chat preamble | Medium — custom conversation schema |
| 7 | **pyromind/agentic-tool-call-dataset-12k** | 12,000 (10 K short: ~8 turns/3 calls; 2 K long: ~82 turns/40 calls) | Apache-2.0 | Synthetic (Toucan + OpenSeeker derived); reasoning included | High format fit, small scale; intermingled multi-turn with reasoning | Low — OpenAI-style `tool_calls` |
| 8 | **voidful/agent-sft** | 309,322 (27 wired sources, deduped) | **"other"** ⚠️ | Normalised, deduped, quality-tiered mixture in OpenAI-style schema | High breadth (aggregates most of the above families) | Low — but license "other" blocks admission as-is |
| 9 | txchmechanicus/qwen3.5-toolcalling-v2 (dup of Mustafaege) | ~102,393 | Apache-2.0 | Synthetic; reasoning traces | Medium-high; Jupyter-agent/code-execution angle | Medium — Qwen-style messages, thinking content leakage risk |
| 10 | AmanPriyanshu tool-reasoning series (Toucan-333 K cleaned, Hermes, Mobile-Actions 8.7 K, …) | 333 K+ | (per-series, unverified) | "Cleaned/rectified" transforms of upstream sets | Reasoning→call→result→answer pattern explicitly staged | High — custom `<tool_call>` JSON-in-tags format |
| 11 | NousResearch/hermes-function-calling-v1 (direct) | ~10 K class (glaive 5 K etc.) | Apache-2.0 | Synthetic (Glaive-derived) | Multi-turn conversational function calling | Low — but **already inside smoltalk2**; no need to admit separately |
| 12 | AgentTuning/AgentInstruct, Agent-FLAN, FireAct | 1,866 / ~100 K / small | Apache-2.0 / Apache-2.0 (ToolBench upstream ⚠️) / MIT | Executed agent traces (ReAct era) | Multi-turn but **textual ReAct format**, 2023-era tools | High — legacy formats, benchmark-adjacent content |
| 13 | Salesforce/xlam-function-calling-60k (direct) | 60,000 | CC-BY-4.0, **gated (auto)** | APIGen 3-stage: format check → real execution → semantic verify; >95 % human-eval correctness on 600 samples | Mostly single-shot Q→call→answer; weakest seam | Medium — gated repo; **already available ungated via smoltalk2 xlam_traces** |
| 14 | Salesforce/APIGen-MT-5k | 5,000 multi-turn | **CC-BY-NC-4.0** ⚠️ gated | Two-phase verified (blueprints w/ ground-truth actions, LLM committee review) | High (multi-turn agent-human interplay) | — **REJECT on license** (non-commercial) |
| 15 | gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL) | eval set (v1–v4) | Apache-2.0 (repo) | n/a | n/a | — **DO NOT TRAIN**: live public benchmark; training on it = BFCL eval leakage |
| 16 | Magpie-Ultra (v0.1 / v1.0) | 50 K / 1 M | llama3.1 (v0.1) | Synthetic instruction-response | **No tool variant exists**; general chat (our pool already includes smol-magpie-ultra) | n/a |
| 17 | Gorilla/APIBench, UltraTool, Tool-Alpaca | — | — | — | Benchmarks / legacy small corpora, not intermingled chat | Deprioritised (unverified details) |

**Corrections vs the first-pass sweep:** ToolACE is **not 200 K** rows (card: 10 K<n<100 K
with 26,507 APIs). The Nexus card confirms 599,267 rows and the 27/27 behavioural
interview result, but the first-pass "5.4 M tool calls" and "35 %→95 %" figures were
**not found** on the card. APIGen-MT-5k is **CC-BY-NC-4.0** — first pass implied it was
in-range; it is not.

---

## Recommended ADMISSION SHORTLIST (top 5 by value/effort)

1. **smoltalk2 SFT tool splits** — `smolagents_toolcalling_traces_think`,
   `xlam_traces_no_think`, `hermes_function_calling_v1_no_think`.
   ~78 K records, ~38 M assistant tokens (estimate), ~104 M total tokens.
   Same already-admitted artifact; zero new onboarding; Apache-2.0/CC-BY-4.0.
   Admit as a *mixture amendment* to the existing smoltalk2 admission record.
2. **Agent-Ark/Toucan-1.5M** — start with the purpose-built `SFT` config (119,287 rows,
   1.35 GB), expand into Qwen3/OSS configs as budget allows. Apache-2.0, ungated, real
   MCP executions — the strongest verification among all candidates at scale.
3. **nvidia/Nemotron-Agentic-v1** (335 K multi-turn, CC-BY-4.0, ungated) — or jump
   directly to **Nemotron-SFT-Agentic-v2** (~1.2 M trajectories, commercial-use
   statement); the `interactive_agent` split (278,880 rows) is the decide-when-to-call
   component.
4. **DomofonResearch/Tool-Reasoning-31K** (30,764 rows, Apache-2.0) — specifically for
   its 11,943 relevance/no-tool-fits scenarios (mirrors our protected-panel mirror
   pattern: tools in context, correct behaviour = don't call) plus reasoning-before-call.
5. **NexusProjectsAI/Nexus-Agents-ToolCalling** — `stage1` config (60,185 curated rows)
   rather than the raw 599 K v1 corpus. Cheapest source of *canonical seam behaviour*
   (ask-when-ambiguous → call-when-clear → read result → answer in text) in clean
   OpenAI schema; cap its mixture weight because of the narrow 3-skill domain.

Deferred / watchlist: `pyromind/agentic-tool-call-dataset-12k` (clean but small; the
2,000-row long split at ~82 turns is interesting for long-horizon seam behaviour),
`ToolACE` (arrives via DomofonResearch aggregation anyway), `voidful/agent-sft` (blocked
on license "other" — instead use its MIT-licensed `awesome-agent-dataset` catalog to wire
upstream sets individually).

Explicit rejects: **APIGen-MT-5k** (CC-BY-NC-4.0), **BFCL** and **Gorilla/APIBench**
(public evals), any training on **When2Call** (NVIDIA benchmark; its scenarios appear
inside Tool-Reasoning-31K — tag the source slice so it stays separable if When2Call ever
enters an eval panel), **Magpie-Ultra** (no tool variant exists).

---

## Per-dataset assessment detail (six criteria)

### 1. smoltalk2 tool splits (HuggingFaceTB)
- **Distribution fit.** smolagents_traces: 5.34 avg turns, tool-calling agent
  trajectories with reasoning — closest match to "casual ask → call → read → answer".
  hermes_fc_v1: 5.35 avg turns, multi-turn conversational function calling.
  xlam_traces: avg 2 turns — mostly single-shot; call-format ballast, weakest seam
  teaching. No explicit "no tool needed" subset here (gap covered by shortlist #4).
- **Verification.** xLAM: APIGen 3-stage (format / actual execution / semantic), human
  eval >95 % on 600 samples — direct evidence from the Salesforce card. smolagents:
  generated with DeepSeek-V3-0324; execution grounding not stated on the card
  (unverified). hermes: synthetic Glaive-derived, no execution claim. All three carry
  HF's benchmark decontamination pass.
- **Scale.** 78,002 rows / ~104 M total tokens / ~38 M assistant tokens (estimate).
- **License.** Apache-2.0 (new subsets); xLAM upstream CC-BY-4.0. No gating on the
  smoltalk2 repo itself.
- **Format.** Exactly our admitted schema: `messages[{role, content}]` +
  `chat_template_kwargs{python_tools, xml_tools, custom_instructions, enable_thinking}`.
- **Contamination.** HF decontaminated the whole mix against their eval benchmarks;
  public function-calling content is low-risk vs our internal synthetic panels.

### 2. Agent-Ark/Toucan-1.5M
- **Distribution fit.** Trajectories from 495 real MCPs / 2,000+ tools; multi-round,
  multi-turn, sequential + parallel calls; genuine intermingling. Includes `question` /
  `target_tools` / `available_tools` fields — good for tool-retrieval curriculum.
- **Verification.** Real tool executions in authentic MCP environments (card, direct
  evidence); per-record quality-assessment fields support filtering. Card caveat: data
  collected June–September 2025 from community MCPs; responses may be unstable/noisy.
- **Scale.** 1,527,259 rows total: Kimi-K2 518,516 / OSS 457,130 / Qwen3 551,613 /
  SFT 119,287 (~19.5 / ~23.3 / ~21.8 GB raw per generation config). Assistant-token
  budget: unmeasured — needs a counting pass before mixture weighting.
- **License.** Apache-2.0, ungated.
- **Format.** `messages` and tool lists stored as JSON *strings* (string dtype columns) —
  needs one parse-and-normalise step into our codec input.
- **Contamination.** Real-world MCP content; no shared-benchmark material identified.

### 3. nvidia/Nemotron-Agentic-v1 / Nemotron-SFT-Agentic-v2
- **Distribution fit.** Designed for "decide when to call tools" over multi-turn
  conversations; v1 splits: `interactive_agent` + `tool_calling`; 335,122 multi-turn
  conversations (figure from a third-party index of the dataset — medium-high
  confidence). v2 adds multi-step/parallel calls and a `search` split.
- **Verification.** Synthetic, NVIDIA-curated pipelines; v2 states "ready for
  commercial use". Treat as synthetic-verified (schema/pipeline), not execution-grounded.
- **Scale.** v1: 5.8 GB storage; v2: ~1.2 M trajectories, ~25 GB (interactive_agent
  278,880 rows via parquet mirror; search 5,968; tool_calling remainder uncounted).
- **License.** v1 CC-BY-4.0, ungated. v2 CC-BY-4.0 / Apache-2.0 / MIT mixed (per
  component), ungated.
- **Format.** OpenAI-style messages + tools; parse at the structured-message level, not
  the rendered `<function=…>` string level.
- **Contamination.** Glaive-Function-Calling-v2 is a named upstream (Apache-2.0) — a
  public training corpus, not a benchmark. Low risk.

### 4. DomofonResearch/Tool-Reasoning-31K
- **Distribution fit.** Best public source of the *relevance mirror*: 11,943 scenarios
  where tools are available but the correct behaviour is NOT to call; 11,186 multiturn;
  4,925 single-turn; 2,710 multistep; 61 % of rows contain ≥1 tool call;
  reasoning-before-call present.
- **Verification.** Aggregation over xLAM (execution-verified), ToolACE (dual-layer),
  Glaive, Nous-Hermes, NVIDIA When2Call via `interstellarninja/hermes_reasoning_tool_use`;
  original tool calls + results kept. Admit on upstream verification.
- **Scale.** 30,764 rows; token counts not published (estimate tens of M total).
- **License.** Apache-2.0 (card and upstream).
- **Format.** Reasoning-tagged message format; needs our normaliser to split reasoning
  from answer content per our frame protocol.
- **Contamination.** Contains When2Call scenarios — fine for training; tag the source
  column for future eval-exclusion.

### 5. NexusProjectsAI/Nexus-Agents-ToolCalling
- **Distribution fit.** Quintessential seam behaviour: conversational setup interviews
  where the agent asks when ambiguous, calls a tool when unambiguous, reads results,
  answers in text. 2–29 messages per conversation. But only three skills and a small
  tool set — high quality, low diversity.
- **Verification.** Procedural generation with high phrasing/argument diversity
  (unique-message ratio ~0.88), dedup, JSON-schema validation of every call; released
  model gated by 27-case behavioural interview (27/27 vs 13/27 base). Tool *results*
  are synthetic (procedural), not real API returns.
- **Scale.** v1 599,267 rows (21 GB); stage1 60,185; stage2-recovery 16,015 (error
  recovery from broken states — interesting for recovery-behaviour training).
- **License.** Apache-2.0, ungated (Hub API metadata).
- **Format.** OpenAI/`mlx_lm` `messages` + `tools`; renders cleanly through our codec.
- **Contamination.** None identified; internally-generated scenarios.

---

## Rendering plan sketch (OpenAI-style → five-line Analysis/Commentary/Action/Arguments frame)

All shortlist picks except DomofonResearch emit (or normalise to) OpenAI-style messages,
which our codec renders cleanly:

1. **Tool manifest.** smoltalk2: `chat_template_kwargs.python_tools` / `xml_tools` →
   tool registry lines. Toucan/Nemotron/Nexus/pyromind: the `tools` field (list of
   `{type:"function", function:{name, description, parameters}}`) → same registry.
   Registry sits in the preamble; no per-turn repetition.
2. **Assistant turn with `tool_calls[]`.** Each
   `{id, type:"function", function:{name, arguments}}` maps to:
   `Action: <function name>` + `Arguments: <raw JSON arguments string>` (verbatim —
   never re-serialise; preserve argument ordering/whitespace). Assistant text preceding
   the call maps to `Analysis` (reasoning) or `Commentary` (user-facing chat).
3. **`role:"tool"` result messages.** Map to the observation/result slot keyed by
   `tool_call_id`; content preserved verbatim (the "reads result" tokens — supervised as
   context, not targets).
4. **Plain assistant answer after the result.** The seam payoff: renders as an ordinary
   Analysis/Commentary/Action(text) frame — the chat→tool→chat transition we want the 4B
   model to learn, supervised as an assistant target.
5. **Reasoning content.** `_think` splits (smolagents) and DomofonResearch reasoning
   traces route to the reasoning slot per our think/no-think codec policy; never
   concatenate reasoning into the visible answer.
6. **Extra work by source:** Toucan (JSON-string columns → parse first),
   DomofonResearch (split reasoning tags from content), ToolACE and AmanPriyanshu
   (custom tag-based call serialisation → structural regex parse). smoltalk2, Nexus,
   pyromind, Nemotron: direct.

### Admit on source verification vs re-verify locally
- **Admit on source verification:** `xlam_traces_no_think` (APIGen executed + semantic
  verify + HF decontamination — the strongest inheritable chain), Toucan (real MCP
  executions + quality fields), Nexus stage1 (schema-verified + behavioural eval),
  smolagents/hermes splits (HF decontamination).
- **Cheap local lint (no re-execution):** schema-validity of every `tool_calls` payload
  against its manifest entry; tool-result presence after every call; decision
  consistency (call vs action(text)) on DomofonResearch relevance rows.
- **Counting pass before mixture weighting:** Toucan and Nemotron assistant-token totals.
- **No candidate requires re-running tools to admit**; the execution-grounded sets
  (xLAM, Toucan) already carry real results in-record.

---

## Contamination risk summary

- Our protected eval panels are internal synthetic workspaces with unique entity IDs —
  public function-calling corpora cannot collide with them (researcher inference).
- **Do not train on:** BFCL (live public benchmark, v1–v4; its multi-turn v3/v4 is
  exactly the target distribution — which is why training on it would be leakage),
  Gorilla/APIBench (benchmark), When2Call scenarios (benchmark; present inside
  Tool-Reasoning-31K — tag and keep separable), `smolagents/post-train-bench-traces`
  (contains BFCL benchmark traces).
- 2023-lineage sets (AgentTuning/AgentInstruct, FireAct) are HotpotQA/ToolBench-adjacent
  with legacy textual formats — deprioritised rather than rejected.

## Contradictions & corrections vs the first-pass sweep

- ToolACE is **not 200 K** rows (card: 10 K<n<100 K, 26,507 APIs). First-pass overstated.
- Nexus: 599,267 rows confirmed, but "5.4 M tool calls" and "35 %→95 %" were **not
  found** on the card; the card's own claim is the 27/27 vs 13/27 behavioural interview.
- APIGen-MT-5k is **CC-BY-NC-4.0** — rejected for our permissive-only programme.
- Meta released **no public Llama-3.1/3.3 tool-use training chat set** (weights +
  eval artifacts only).
- No official **OpenAI function-calling public training data** exists; the public proxy
  is the Glaive lineage (Apache-2.0), already represented inside smoltalk2.
- **Magpie-Ultra has no tool-calling variant** (general instruction-response; our
  conversation pool already includes smol-magpie-ultra).

## Missing evidence / open questions

- Exact upstream identity of `xlam_traces_no_think` (card omits it from its bullet
  list; row count ≈ xlam-function-calling-60k minus decontamination) — confirm via the
  `source` column on first load.
- Which splits our admitted 620 K-record conversation pool actually covered (determines
  mixture amendment vs recipe line).
- Assistant-token counts for Toucan and Nemotron families.
- Whether smoltalk2's smolagents traces are execution-grounded (card says "generated
  with DeepSeek-V3-0324" only).
- Licenses for the AmanPriyanshu series and microsoft/orca-agentinstruct-1M-v1
  (watchlisted, unverified this pass).
- UltraTool / Tool-Alpaca specifics (legacy, deprioritised, unverified).

## Sources

Kept (load-bearing):
- HuggingFaceTB/smoltalk2 dataset card + README `dataset_info` — https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
- Salesforce/xlam-function-calling-60k card + Hub API — https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k ; APIGen paper — https://arxiv.org/abs/2406.18518
- Salesforce/APIGen-MT-5k card (CC-BY-NC-4.0) — https://huggingface.co/datasets/Salesforce/APIGen-MT-5k ; APIGen-MT paper — https://arxiv.org/abs/2504.03601
- Agent-Ark/Toucan-1.5M card + Hub API — https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M ; IBM blog — https://research.ibm.com/blog/toucan-for-tool-calling
- nvidia/Nemotron-Agentic-v1 card + Hub API — https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1 ; nvidia/Nemotron-SFT-Agentic-v2 card + API — https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2 ; parquet mirror row counts — https://huggingface.co/datasets/tuandunghcmut/Nemotron-SFT-Agentic-v2-parquet
- NexusProjectsAI/Nexus-Agents-ToolCalling card + Hub API — https://huggingface.co/datasets/NexusProjectsAI/Nexus-Agents-ToolCalling
- DomofonResearch/Tool-Reasoning-31K card — https://huggingface.co/datasets/DomofonResearch/Tool-Reasoning-31K
- Team-ACE/ToolACE card — https://huggingface.co/datasets/Team-ACE/ToolACE
- pyromind/agentic-tool-call-dataset-12k card + Hub API — https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k
- voidful/agent-sft Hub API (license "other") + awesome-agent-dataset catalog (MIT) — https://huggingface.co/datasets/voidful/agent-sft ; https://github.com/voidful/awesome-agent-dataset
- NousResearch/hermes-function-calling-v1 Hub API — https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- gorilla/BFCL — https://gorilla.cs.berkeley.edu/leaderboard ; https://github.com/ShishirPatil/gorilla
- txchmechanicus/qwen3.5-toolcalling-v2 card — https://huggingface.co/datasets/txchmechanicus/qwen3.5-toolcalling-v2
- AgentTuning / Agent-FLAN / FireAct — https://github.com/thudm/agenttuning ; https://huggingface.co/datasets/internlm/Agent-FLAN ; https://github.com/anchen1011/FireAct

Rejected/deprioritized as sources: Reddit threads (anecdote), third-party raw dataset
mirrors, papers.cool aggregator PDFs, meta-llama eval datasets (eval-only).

## Next steps

1. Local check: enumerate our admitted smoltalk2 splits; confirm the three tool splits
   were excluded from the 620 K conversation pool; file the mixture amendment.
2. Token-counting pass on Toucan SFT config + Nemotron-Agentic interactive_agent for
   assistant-token budgets per source.
3. Rendering spike: run 200 sampled records from each shortlist pick through the
   Analysis/Commentary/Action/Arguments codec; assert tool-call/result frame alternation
   and post-result text-answer presence (seam ratio is the metric of interest).
4. If When2Call is ever shortlisted as an eval, pre-emptively tag and exclude its slice
   within Tool-Reasoning-31K.

---

*Companion artifact: the same survey is persisted to the subagent research output for
this run. This file should be committed scoped (single-file `git add` of this path),
never `git add -A`.*
