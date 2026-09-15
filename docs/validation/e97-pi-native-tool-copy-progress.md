# E97 Pi-native tool/copy programme progress

This ledger is append-only in meaning: failures and superseded attempts remain
recorded. The driving design is
[`../EMENDER_E97_4B_PI_NATIVE_TOOL_AND_COPY_PROGRAM.md`](../EMENDER_E97_4B_PI_NATIVE_TOOL_AND_COPY_PROGRAM.md).

## Programme state — 2026-09-15

|Phase|State|Evidence / next action|
|---|---|---|
|Authority and design|**passed**|11-tool manifest SHA `55421905438806223414d96a4e64f1a6ae64772afdb1fbc67fbbfb77e8d5d908`|
|Adjacent Pi-native codec/provider|**CPU scripted control passed**|Real Pi built-in `read` result entered exact causal record; broader tools/panels next|
|PTCP-01 exact-copy panel|**in progress**|Freeze128/1K/8K/32K/64K operations before sampling|
|PTCP-02 tool-choice panel|not started|Use captured Pi schemas and current/local contrast pairs|
|PTCP-03 web-research panel|not started|Freeze timestamped claims/queries and real Pi execution|
|PTCP-04 repository discovery data|proposal frozen|8-record/2,484-target seed; >=256 verified records required|
|Unchanged-model baseline|not started|Run only after protocol and panels pass scripted controls|
|Pi-native training authority|not started|Target2,000–5,000 verified records; exact masks/exposures required|
|Training|not started|Proposed32 updates; publish concrete authority before launch|
|Post-training qualification|not started|All copy/tool/web/repository/retention/session gates jointly|
|Attended demo packaging|current narrow demo only|OpenHands-backed Pi2/2 works; Pi-native surface pending|

## Starting checkpoint and retained evidence

- Model:4,045,972,080 parameters.
- Live-y checkpoint:
  `representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt`,
  SHA `9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa`.
- Current demonstrated behavior: bridge fresh32/32; unchanged direct/Pi8/8 on
  reused structured cases; one real two-task Pi/OpenHands session2/2.
- Repository discovery baseline:0/4 direct and0/4 Pi.
- Long-delay exact free-generation copy: **unmeasured**. The1.0 teacher-forced
  tool-token retention score is not a substitute.
- Current deployed tool contract: Pi-fronted OpenHands actions
  `execute_bash`, `str_replace_editor`, `think`, `finish`; this is not the target
  Pi-native surface.

## Decisions

1. The fixed product tool surface is small; optimize for its real schemas rather
   than arbitrary plugins.
2. Web research is a first-class decision behavior for current, uncertain and
   externally verifiable questions, balanced by local/direct contrast cases.
3. Existing OpenHands trajectories are retained and may supply tasks, but Pi-native
   records require real Pi execution and result bytes.
4. Exact copying is evaluated independently from transport fidelity and NLL.
5. Baseline results drive data composition before a model update.

## Immediate actions

- [x] Capture active `read`, `bash`, `edit`, `write`, `process`, `fffind`,
  `ffgrep`, `web_search`, `source_check`, `fetch_content`, and
  `get_search_content` schemas from the pinned Pi runtime.
- [x] Record extension/package and executable hashes.
- [x] Implement canonical dynamic-tool codec with `think`/`finish` pseudo-actions.
- [x] Add parser, schema, causal-history and private-boundary tests.
- [x] Add a scripted real-Pi call/result/finish control before model sampling.
- [ ] Freeze PTCP-01/02 panels and exact budgets.

## Tool-surface capture evidence

The initial combined capture `proc_a288` inherited open stdin and was stopped
after an unproductive hang; no provider/model/tool ran. Core controls `proc_31fc`
and `proc_964c` timed out for the same reason. `proc_e3e8` was a shell-only
`mkdir` error. Instrumented `proc_553d` and `proc_6750` proved the extension loaded
and provider registered but was never invoked. Closing stdin matched the qualified
launcher: `proc_9a14` completed in one second. The durable grouped capture
`proc_78af` then captured core/process/FFF/web in six seconds; `proc_0b4a` passed
two freezer tests and published the authority.

- Pi:0.85.1; pinned inventory retained from compatibility qualification.
- Process extension:0.12.0.
- FFF extension:0.10.6.
- Web extension:0.29.0.
- Tool manifest: `configs/pi/e97-active-tool-surface-v1.json`, SHA
  `55421905438806223414d96a4e64f1a6ae64772afdb1fbc67fbbfb77e8d5d908`.
- No model generations, tool execution, network access, GPU work or updates.

## Pi-native protocol control

`proc_5d7e` passed nine CPU tests. A real Pi process executed its registered
built-in `read` against a temporary fixture; the exact returned `toolResult` was
paired with the generated call and entered the next E97 prompt before a scripted
`finish`. The control used the new adjacent `e97-pi-native-v1` codec/provider and
never invoked OpenHands. Parser, undeclared-action, result-identity, private-think,
manifest and causal-history checks passed. Zero model generations, network calls,
updates or GPU work occurred.

## Historical session evidence immediately preceding this programme

- Failed model-session preflight `proc_79c2`: session-token plan mismatch; zero
  generations, unchanged weights and cleaned resources.
- Corrected model session `proc_77c4`:2/2; plan SHA
  `5265d88c24c0953b655a3227dc15f2061c50ce1b736958769b5600912b371e78`,
  summary SHA `5217ba6693530eb5dba7f414126d13044f5d2dda1aa66e9246ac1467b6f710de`,
  audit SHA `aeeed83cd9e4ed0b0f9e0a7b06ebb82bd8f43b6f5ddbd1849abec1a521cde1c0`.
