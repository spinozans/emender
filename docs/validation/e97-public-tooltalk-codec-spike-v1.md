# E97 — Public tooltalk codec spike + admission package (v1)

**Status:** STAGED. Nothing in this spike enters a prep. Admission of any source below
requires the operator's explicit sign-off. This artifact is a rendering experiment and
measurement pass, not an admission record.

**Purpose.** Execute the rendering plan sketch from
[`e97-public-intermingled-tool-talk-dataset-survey-v1.md`](e97-public-intermingled-tool-talk-dataset-survey-v1.md)
against the five shortlisted public intermingled chat+tool-call sources: render
OpenAI-style `messages` + `tool_calls` conversations into the Pi-native five-line
Analysis/Commentary/Action/Arguments frame protocol
([`scripts/e97_pi_native_codec.py`](../../scripts/e97_pi_native_codec.py)), measure the
chat→tool→chat seam, verify licenses, and run the protected-panel overlap audit —
producing an admission recommendation per source.

**Renderer:** [`scripts/render_e97_public_tooltalk.py`](../../scripts/render_e97_public_tooltalk.py)
(subcommands `download`, `render`, `overlap`, `verify-licenses`).
**Evidence authority (rendered samples + receipts):**
`/mnt/nvme1n1/erikg/sft/e97-public-tooltalk-codec-spike-v1/` — per-source
`manifest.json` / `records.jsonl` / `metrics.json` / `spot-checks/`, plus
`download-receipt.json`, `license-receipt.json`, `overlap-audit.json`.
**Sample:** 200 records per source (50 for Nemotron), seeded
`e97-public-tooltalk-codec-spike-v1:<source>`, tokenizer `p50k_base`, 64 K context
window accounting. Raw downloads were deleted after rendering; the download receipt
keeps per-file SHA-256 pins.

---

## 1. Which smoltalk2 case holds: NEW SOURCE, not a mixture amendment

Enumerating the admitted artifact
`/mnt/nvme1n1/erikg/sft/e97-4b-smoltalk2-admitted-v1/manifest.json` (620,389 records,
~801 M tokens, admitted Apache-2.0 SmolTalk2 conversation/reasoning SFT):

- `source_counts` contains exactly six subsets: `multi-turn-reasoning-if` (28,217),
  `smoltalk-everyday-convs-reasoning-Qwen3-32B` (2,057),
  `smoltalk-smollm3_smol-magpie-ultra` (406,843), `smoltalk-smollm3_smol-rewrite`
  (53,262), `smoltalk-smollm3_smol-summarize` (96,061),
  `smoltalk-smollm3_systemchats-30k` (33,949).
- `input_files` contains none of the three tool-use splits.
- **NONE of the three tool splits is inside the admitted artifact.** The manifest's
  `excluded_downloaded_subset` field records that
  `smolagents_toolcalling_traces` *was downloaded during the admitted build but
  excluded* with the note "requires Pi tool normalization"; `xlam_traces_no_think` and
  `hermes_function_calling_v1_no_think` were never downloaded by that build.

**Verdict:** the three tool splits are **new sources to admit** (from the same pinned
repo `HuggingFaceTB/smoltalk2` at the already-verified revision
`fc6cc2103c066455aade5d7fbb346039ae36ca5e`), not an amendment of the existing admission
record. The prior conversation-pool admission supplies provenance and revision trust,
but each tool split needs its own admission line, its own lint pass, and — for
smolagents — a policy decision (below).

## 2. Rendering policy (as implemented; all machine-checked)

Per the survey sketch, plus the decisions the sketch left open:

1. **Tool manifest → preamble Protocol registry.** One `Protocol:` line per record
   (the codec's own header form: profile, instructions, tools, pseudo-actions); public
   tool specs are normalized to the Pi registry shape `{name,label,description,parameters}`.
   No per-turn repetition.
2. **Assistant `tool_calls[]` → one five-line frame per call.**
   `Action: <name>` / `Arguments: <source's own JSON serialization when it is strict
   JSON and single-line — never re-serialised>`. Python-literal sources (smolagents,
   hermes) and dict-shaped sources (Nexus) are canonicalized with `compact()`, counted.
   Multi-line verbatim JSON (84 % of Nemotron argument strings) is canonicalized
   because the five-line frame admits single-line Arguments values only.
3. **Pre-call assistant text → `Commentary:` of the first call frame; source reasoning
   (`<think>` blocks, `reasoning_content`, Tool-Reasoning reasoning blocks) →
   `Analysis:`** per the think/no-think codec policy — never concatenated into the
   visible answer.
4. **Plain-text assistant answers (no calls) → `Action: finish` /
   `Arguments: {"message": <answer>}`** (finish.message is the codec's public answer
   channel). SFT transcripts may continue with further `User:` sections after a finish
   frame; supervision is per frame, unlike the runtime episode lifecycle. **This is a
   spike policy for the operator to ratify at admission.**
5. **`role:tool` results → `ToolResult:` sections** in the codec's normalized
   toolResult message shape, content verbatim, supervised as context only.
6. **Parallel calls → consecutive frame/ToolResult pairs.** The codec admits exactly
   one call per turn, so a call group serializes as frame, result, next-frame, result…
   (deferred-frame interleaving; trailing unmatched group members flush at record end
   and count as orphans).
7. **smolagents' `final_answer` tool** is rendered verbatim as a declared-tool call
   (not rewritten to `finish`) and counted separately (`final_answer_after_result`) —
   see metric table. A future admission may map it to the `finish` pseudo-action; that
   transform is deliberate and needs the operator's sign-off.

## 3. SEAM-METRICS table (per-source, eligible sample of 200 / 50)

Definitions: *seam ratio* = records where a plain-text answer (finish frame) directly
follows a ToolResult section. *call→result alternation valid* = every call has a result
and every result a call; the no-terminal-orphan variant forgives one dangling call at
the very end of a record (source-designed terminal answers and rollout-prefix
truncations). *reasoning-before-call* = call frames with non-null Analysis.
*relevance/no-tool* = tools declared, zero calls, ≥1 finish frame. *invalid frames* =
frames failing `parse_turn`/registry/finish validation (reasons split:
malformed / undeclared action). Tokens are `p50k_base` over the full rendered
transcript; assistant-target tokens cover the five-line frames only.

| Source | rows | parse-fail rows | eligible rows | records w/ calls | seam | final-answer-after-result | direct-answer-only | alternation valid | valid (no terminal orphan) | reasoning-before-call | relevance/no-tool | invalid frames | mean asst tok/rec | mean total tok/rec | over-64K rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoltalk2-smolagents (`_think`) | 9,079 | 0 | 9,079 | 100% | 0.000 | 0.615 | 0.385 | 0.000 | 0.990 | 1.000 | 0.000 | 0.240 | 220 | 7,391 | 0% |
| smoltalk2-xlam | 59,962 | 0 | 59,962 | 100% | 0.000 | 0.000 | 0.000 | 0.000 | 0.440 | 0.000 | 0.000 | 0.000 | 67 | 665 | 0% |
| smoltalk2-hermes | 8,961 | 2,451 | 6,510 | 100% | 0.716 | 0.000 | 0.000 | 0.735 | 0.819 | 0.000 | 0.000 | 0.002 | 217 | 994 | 0% |
| Toucan-1.5M SFT | 119,287 | 943 | 118,344 | 70.4% | 0.704 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.296 | 0.001 | 1,144 | 4,127 | 0.5% |
| Tool-Reasoning-31K (When2Call excluded) | 30,764 | 0 | 20,088 | 84.7% | 0.687 | 0.000 | 0.000 | 0.832 | 0.893 | 0.903 | 0.153 | 0.002 | 392 | 1,299 | 0% |
| Nexus-Agents stage1 | 54,648 | 0 | 54,648 | 100% | 0.180 | 0.000 | 0.000 | 0.130 | 0.925 | 0.872 | 0.000 | 0.004 | 469 | 3,760 | 0% |
| Nemotron-Agentic-v1 interactive_agent | 19,028 | 0 | 19,028 | 88.0% | 0.880 | 0.000 | 0.000 | 1.000 | 1.000 | 0.838 | 0.120 | 0.000 | 2,540 | 5,845 | 0% |

Verbatim-argument preservation (frames): xlam 100 %, Tool-Reasoning 100 %, Toucan
100 %, Nemotron 16 % single-line verbatim + 84 % canonicalized (pretty-printed source),
hermes 2 %, smolagents 0 %, Nexus 0 % (python-literal / dict-shaped sources — canonical
`compact()` by necessity).

### Cohort size estimates at full intake (eligible rows × measured means)

| Source | eligible records | assistant-target tokens | total tokens |
|---|---:|---:|---:|
| smoltalk2-smolagents | 9,079 | 2.00 M | 67.1 M |
| smoltalk2-xlam | 59,962 | 4.01 M | 39.9 M |
| smoltalk2-hermes | 6,510 | 1.41 M | 6.47 M |
| Toucan-1.5M SFT | 118,344 | 135.3 M | 488.4 M |
| Tool-Reasoning-31K (When2Call excl.) | 20,088 | 7.87 M | 26.1 M |
| Nexus-Agents stage1 | 54,648 | 25.6 M | 205.5 M |
| Nemotron-v1 interactive_agent | 19,028 | 48.3 M | 111.2 M |

Note these are *sample-means × eligible-rows* estimates; Toucan in particular has
unknown duplication (multi-config generation of overlapping MCP traces) and Nexus
stage1 rows are rollout **prefixes** of the same conversations (§5), so their true
deduplicated cohorts are materially smaller.

## 4. Per-source findings (what the renderer surfaced)

- **smolagents (think).** Every record ends in a `final_answer` tool call with no
  result — a terminal dangling call by design, so strict alternation validity is 0 %
  but no-terminal-orphan validity is 99 %. The seam exists but is *encoded as the
  final_answer call*: 61.5 % of records answer after ≥1 result via `final_answer`, and
  38.5 % of records are direct-answer-only (no external call — the no-tool mirror).
  Reasoning is present on 100 % of call frames. **Defect:** 24 % of call frames invoke
  `visit_webpage`, which is NOT in the row's `xml_tools` registry (the serialized
  registry lists only `final_answer`/`web_search`/`wikipedia_search`) — the smoltalk2
  registry serialization is incomplete relative to the traces' real toolset.
- **xlam.** Pure single-shot call-format ballast: zero tool results in the entire
  split, zero seam, zero reasoning. 100 % verbatim JSON arguments. Exactly the
  "weakest of the three" the survey predicted.
- **hermes.** A seam workhorse for the clean rows: 71.6 % of records answer in plain
  text after a result. **Defect:** 27 % of rows (2,451/8,961) are structurally
  corrupted upstream — the `ExpertQAExtractor` family has both broken tool-registry
  serialization (the registry is fragmented across a prompt preamble) and truncated
  call blocks; the renderer excludes them mechanically. Among rendered records, 18.1 %
  still carry mid-conversation orphans (upstream parallel calls with missing results —
  e.g. three calls, one result, then an answer addressing all three).
- **Toucan SFT.** The strongest verified source: real MCP executions, 100 % call→result
  alternation, 70.4 % seam, **and 29.6 % relevance/no-tool rows** (tools declared,
  correct behaviour is to answer without calling — including explicit
  "my tools cannot do this" declines). 0.79 % of rows (943) have corrupt call-argument
  strings (invalid `\'` JSON escapes) and are excluded mechanically.
- **Tool-Reasoning-31K.** Grounded reasoning on 90 % of call frames; 68.7 % seam; the
  relevance mirror renders exactly as intended (reasoning → "I can't call a tool for
  this" → direct answer). 15.3 % of eligible records are relevance rows. The
  `Nvidia-When2Call` slice is exactly **10,676 rows**, tagged `when2call_excluded` in
  every record and excluded from all eligible metrics/estimates so it can never enter
  a prep silently (When2Call is an NVIDIA benchmark; must stay separable if it ever
  enters an eval panel). 16.8 % of eligible records still have call-only rows without
  results (upstream `single` decision type).
- **Nexus stage1.** The rendered authority here is `stage1/train.jsonl` only:
  **54,648 rows** (the survey's 60,185 is train+validation). Every row is a rollout
  **prefix** ending at an assistant call: 92.5 % of records have only the terminal
  dangling call, i.e. consecutive rows duplicate the same conversation up to each
  step. Reasoning on 87 % of frames; clarifying-question seam in 18 % of records;
  mid-conversation `BOARD STATE` system sections render as System context. The 3-skill
  narrow-domain caveat stands.
- **Nemotron-v1 interactive_agent.** Exact split count **19,028 rows** (the whole-file
  448 MB JSONL was fetched to count and sample uniformly; the 5.3 GB `tool_calling`
  split was NOT fetched — the survey's repo-wide 335,122 figure remains unverified).
  88 % seam, 100 % alternation validity, 84 % reasoning-before-call, 12 % relevance
  rows, zero malformed frames. Long conversations (5.8 K tokens mean, 2.5 K assistant
  target tokens per record) make it the most expensive per record.

## 5. Spot-check by eye (5 per source; full transcripts in each `spot-checks/` dir)

All 35 spot-checks were read by eye. The five-line frames are structurally right,
observations are verbatim and sane, and post-result answers genuinely use the observed
values. Representative excerpts (condensed; `<tool_call>` source tags escaped):

**Toucan SFT** (spots `07913103…`, `58f2d98e…`, `6845f40d…`, `7e9dd557…`, `a424c3ef…`):
```
User: Our mountain-trail guide service is scheduled to lead a 12-person group along the ridge…
Assistant:
Commentary: "I'll fetch the live temperature for your ridge location…"
Action: weather-service-get_live_temp
Arguments: {"latitude": 45.1234, "longitude": -112.5678}
ToolResult: {"temperature": 14.1}
Assistant:
Action: finish
Arguments: {"message":"Current temperature on the ridge: **14.1 °C**\n\n### Compliance assessment\n• **Safety requirement**: The 14.1 °C reading is well above the 5 °C minimum…"}
```
The answer quotes the observation — the seam we are buying. The `6845f40d…` spot is a
decide-NOT-to-call sequence: the user requests a forecast, the tool only returns live
temperature, and the agent answers four times in a row *without* calling (relevance
behaviour). The `a424c3ef…` spot is a pure relevance row (tools declared, no call).

**Nemotron-v1** (spots `23723aca…`, `39f3e7b3…`, `6768e345…`, `a78921ea…`, `ba2060e4…`):
```
Assistant:
Analysis: "The user wants an early check-in request… The user provided Guest ID G01234 and Booking ID B45678…"
Action: process_early_checkin
Arguments: {"reservation_id": "B45678", "requested_time": "10:00"}
ToolResult: {"returns": {"approved": false, "actual_checkin_time": "15:00", "denial_reason": "Room not yet prepared for early check-in"}}
Assistant:
Analysis: "We have the result. The process_early_checkin tool returned an object indicating approved: false…"
Action: finish
Arguments: {"message":"…unfortunately, the room won't be ready before 3:00 PM…"}
```
Grounded pre-call analysis, the denial is read from the observation, the answer uses
it. All five spots follow this shape; `39f3e7b3…` shows a multi-step call→result→call
chain ending in a transfer plus finish.

**smoltalk2-smolagents** (spots `…000000`, `…002369`, `…005293`, `…007001`, `…009077`):
```
Assistant:
Analysis: "I need to confirm both the setting of Kingsolver's novel and the year the Belgian Congo became independent…"
Action: web_search
Arguments: {"query":"Barbara Kingsolver best known novel set in Belgian colony Africa independence year"}
ToolResult: 'Observation:\n## Search Results\n0. [The Poisonwood Bible - Wikipedia](…)…'
Assistant:
Analysis: "I need to visit the Wikipedia page to confirm the exact year…"
Action: visit_webpage            # ← NOT in the declared registry (the 24% defect)
ToolResult: 'Observation:\nThe Poisonwood Bible - Wikipedia…'
Assistant:
Analysis: "I've confirmed the year of independence is 1960…"
Action: final_answer
Arguments: {"answer":"1960"}
```
Reasoning is clean and grounded; the seam is the `final_answer` call. The `…009077`
spot is direct-answer-only (geometry question answered from reasoning, no tool).

**smoltalk2-hermes** (spots `…000004`, `…002567`, `…005054`, `…006634`, `…008516`):
```
User: "Hi, I need to convert 1000 US dollars to Euros…"
Assistant:
Action: convert_currencies
Arguments: {"amount":1000,"from_currency":"USD","to_currency":"EUR"}
ToolResult: "…{'converted_amount': 850.0, 'from_currency': 'USD', 'to_currency': 'EUR'}…"
Assistant:
Action: finish
Arguments: {"message":"Sure, 1000 US dollars is approximately 850 Euros."}
```
Observation is verbatim (upstream wraps results in a `` block — preserved as-is);
the answer uses the result. The `…002567` spot shows the upstream parallel-call defect
(three calls, no results); `…000004` shows a call→result→call(dangling)→finish record.

**Tool-Reasoning-31K** (spots `…000185`, `…007614`, `…015840`, `…023604`, `…030711`):
```
Assistant:
Analysis: "The user wants a Chi-Square test, but the chi_square_independence_test tool requires a contingency table which has not been provided…"
Action: finish
Arguments: {"message":"Apologies, but to perform a Chi-Square test, I need a 2x2 contingency table…"}
```
The relevance/no-tool rows render exactly as the mirror pattern: tools in the
registry, grounded reasoning, no call, direct answer. `…000185` shows a full
call→result→call→result→compare chain with per-step reasoning.

**Nexus stage1** (spots `…000655`, `…011330`, `…025487`, `…041547`, `…054549`):
```
Assistant:
Analysis: "Mapping what they told me to the right tags. Those topics are DONE now — I will not ask about them again."
Commentary: "Tagging it as education, business / saas, or support to start."
Action: propose_tags
Arguments: {"tags":[{"category":"industries","value":"Education"},…]}
ToolResult: 'Proposed: Education, Business / SaaS, Support, Web.'
User: "Hold on — not business / saas or support…"
Assistant:
Analysis: "The user corrected me: that tag doesn't fit their project. Removing only what they rejected…"
Action: remove_tags
```
Clarify-when-ambiguous → call-when-clear → read result → continue: canonical seam
behaviour. Mid-conversation `BOARD STATE` system sections render as System context.
Every spot ends at a dangling call (rollout-prefix rows, §4).

**smoltalk2-xlam** (spots `…000961`, `…014537`, `…028630`, `…045695`, `…059135`):
```
User: "Please provide the list of books that received awards in the years 2011 and 2014."
Assistant:
Action: get_the_awarded_books_of_a_year
Arguments: {"year": "2011"}
Assistant:
Action: get_the_awarded_books_of_a_year
Arguments: {"year": "2014"}
```
Exactly the call-only distribution: correct verbatim Action/Arguments frames, no
results, no answers. Nothing structurally wrong — just no seam content.

## 6. License verification (HF API `cardData`/tags + README; receipt: `license-receipt.json`)

| Source | API license tags | cardData license | gated | verdict vs survey |
|---|---|---|---|---|
| HuggingFaceTB/smoltalk2 | *(none)* | *(none)* | no | **Discrepancy:** the repo carries **no license tag**. The README states "All the new datasets … smolagents-toolcalling-traces … are licensed under Apache 2.0. For the existing public datasets, please refer to the original dataset." smolagents is listed as **[NEW] → Apache-2.0 confirmed**; hermes-function-calling-v1 is an existing public dataset → upstream NousResearch Apache-2.0; **xlam-traces is not in the README's source bullet list at all** (stats table only) → its upstream identity and license remain inferred (≈ Salesforce xLAM, CC-BY-4.0; see §7). |
| Agent-Ark/Toucan-1.5M | apache-2.0 | apache-2.0 | no | matches survey; README: "released under Apache 2.0" |
| DomofonResearch/Tool-Reasoning-31K | apache-2.0 | apache-2.0 | no | matches; README notes upstream mixture (xLAM, ToolACE, Glaive, Nous-Hermes, Nvidia When2Call) "please honor their respective licenses" |
| NexusProjectsAI/Nexus-Agents-ToolCalling | apache-2.0 | apache-2.0 | no | matches |
| nvidia/Nemotron-Agentic-v1 | cc-by-4.0 | cc-by-4.0 | no | matches; per-record `license` field is also `cc-by-4.0`; README additionally cites Apache-2.0 for the Glaive-FC-v2 upstream |

All permissive for our programme. Revisions resolved to the pinned SHAs in every case
(`revision_matches_pin: true`).

## 7. xlam upstream identity (survey open question)

Direct verification against `Salesforce/xlam-function-calling-60k` was attempted and
**blocked**: the repo is gated and this account is not authorized (403). Circumstantial
evidence for the ≈ xLAM identity: 59,962 rows = 60,000 minus 38 (consistent with HF's
benchmark decontamination), APIGen single-shot format, `source` column `xlam-traces`,
`<tools>`-wrapped JSON registries. **Not confirmed.** Consequence: treat
`xlam_traces_no_think` as CC-BY-4.0-inherited **by inference only** until either the
gate is granted or the row is dropped; the split's seam value is nil anyway (§4), so
this does not block any high-value admission.

## 8. Contamination / protected-panel overlap (receipt: `overlap-audit.json`)

Ran the protected-panel overlap machinery (`ndm.e97_protected_overlap` domains) on all
rendered samples against the three fixed sealed panels (pi-core-eval-v3,
pi-core-eval-v4, real-repo-holdout-v1), with candidate records built as
prompt = first user turn, fixtures = tool-result observations, exact-scalar extraction
as usual. **PASS for all seven sources: zero significant entity collisions** — zero
significant exact scalars (≥ 8 bytes), zero significant normalized content overlaps
(≥ 16 bytes). Only trivial sub-8-byte numeric fixture scalars collided (Nemotron 5,
smolagents 1, Tool-Reasoning 19, Toucan 27 — e.g. `4` vs a protected panel operand),
reported as non-entity structural noise per the ledger precedent. Expected result for
public function-calling data, now measured. **No-TOUCH list respected:** BFCL,
Gorilla/APIBench, the When2Call slice (tagged + excluded), NC-licensed sets, and
smolagents/post-train-bench-traces were never downloaded.

## 9. Admission recommendations (STAGED — for operator sign-off)

| Source | Recommendation | Fraction / intake | Re-verify locally vs admit on source verification |
|---|---|---|---|
| smoltalk2-smolagents | **Admit after one policy decision** | full (9,079 rows; 2.0 M asst tok — trivial budget) | HF decontamination inherited; the two open items are local: (a) decide the `visit_webpage` undeclared-registry defect — complete the registry from the smolagents toolset or drop affected records (24 % of frames); (b) decide whether `final_answer` maps to the codec `finish` at prep time |
| smoltalk2-hermes | **Admit** | full *clean* rows (6,510; 1.4 M asst tok); the 2,451 corrupted rows are excluded by mechanical lint | admit on source verification (HF decontamination) + the existing parse lint; mid-orphans (18.1 % of records) need a masking/drop policy decision like (a) above |
| smoltalk2-xlam | **Hold / low value** | if admitted at all, cap as call-format ballast (≤ 10–20 %, ≈ 0.4–0.8 M asst tok) | zero seam, zero results, orphan-call-only; upstream identity/license unconfirmed (gated, §7) — do not admit until identity is resolved |
| Toucan-1.5M SFT | **Admit, subsampled** | e.g. 25–35 K records (≈ 28–40 M asst tok), prioritizing seam + relevance rows | admit on source verification (real MCP executions) + local dedup pass (unknown duplication across the four generation configs) and the 943-row corrupt-argument exclusion (mechanical) |
| Tool-Reasoning-31K | **Admit** | full eligible (20,088 rows; 7.9 M asst tok) | admit on upstream verification (xLAM executed / ToolACE dual-layer); keep the When2Call exclusion tag mandatory; drop or mask the 16.8 % call-only records at prep |
| Nexus-Agents stage1 | **Admit only after prefix-dedup, capped** | small weight (narrow 3-skill domain); dedup rollouts to terminal states first — true cohort is far below 54,648 distinct conversations | local re-verification required: prefix-collapse, then the existing schema-validity claims apply; the 27/27 behavioural-interview evidence supports the seam behaviour itself |
| Nemotron-Agentic-v1 interactive_agent | **Admit, subsampled** | e.g. 50 % (≈ 9.5 K records; ≈ 24 M asst tok); per-record cost is high (2.5 K asst tok) | admit on source verification (synthetic, NVIDIA-curated; CC-BY-4.0); the 5.3 GB `tool_calling` split remains unassessed — separate decision |

Combined recommended first-intake budget at the fractions above (Toucan 30 K,
Nemotron 50 %, everything else full/capped): roughly **70–90 M assistant-target
tokens** across ~120 K records — same order as one of our existing large cohorts,
dominated by Toucan.

**Codec verdict:** the five-line frame protocol renders all seven public sources with
< 0.4 % malformed frames; the only structural amendments any admission needs are the
finish-for-plain-answers policy (§2.4), the parallel-call serialization (§2.6), and the
verbatim-vs-canonical Arguments decision (§2.2) — all of which this spike implements
and measures, and all of which need operator ratification before a prep consumes any of
these sources.

## 10. Reproduction

```bash
PYTHONPATH=. python scripts/render_e97_public_tooltalk.py download \
    --raw-root <raw> --output-root <authority>
PYTHONPATH=. python scripts/render_e97_public_tooltalk.py verify-licenses --output-root <authority>
PYTHONPATH=. python scripts/render_e97_public_tooltalk.py render \
    --raw-root <raw> --output-root <authority>            # 200/source, 50 Nemotron
PYTHONPATH=. python scripts/render_e97_public_tooltalk.py overlap --output-root <authority>
.venv/bin/python -m pytest tests/test_render_e97_public_tooltalk.py   # 15 unit tests
```

The authority dir pins every raw file by SHA-256 in `download-receipt.json` and records
the renderer's own `checker_sha256` in each per-source manifest; raw downloads are
deleted after rendering per policy (re-download from the pinned revisions is
deterministic).
