# E97 Pi-native tool/copy programme progress

This ledger is append-only in meaning: failures and superseded attempts remain
recorded. The driving design is
[`../EMENDER_E97_4B_PI_NATIVE_TOOL_AND_COPY_PROGRAM.md`](../EMENDER_E97_4B_PI_NATIVE_TOOL_AND_COPY_PROGRAM.md).

## Programme state — 2026-09-15

|Phase|State|Evidence / next action|
|---|---|---|
|Authority and design|**passed**|11-tool manifest SHA `55421905438806223414d96a4e64f1a6ae64772afdb1fbc67fbbfb77e8d5d908`|
|Adjacent Pi-native codec/provider|**CPU scripted control passed**|Real Pi built-in `read` result entered exact causal record; broader tools/panels next|
|PTCP-01 exact-copy panel|**Stage A negative:0/9**|No valid first turn at128,8K or58K; pure copy capacity remains confounded by Pi-native framing failure|
|PTCP-02 tool-choice panel|**Stage A negative:0/3**|No valid direct, FFF or web first action; Stage B not authorized|
|PTCP-03 web-research panel|not started|Freeze timestamped claims/queries and real Pi execution|
|PTCP-04 repository discovery data|proposal frozen|8-record/2,484-target seed; >=256 verified records required|
|Unchanged-model baseline|**Stage A complete:0/12**|Qualified negative outcomes; rejected-token receipts missing, so stop classifications are not independently byte-reconstructable|
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
- [x] Build and CPU-test tiered PTCP-01/02 panels and exact budgets.
- [x] Freeze from immutable source and run exactly12 Stage A episodes.
- [x] Retain0/12 negative outcomes and block Stage B; do not rerun.

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
updates or GPU work occurred. `proc_a6e3` then passed the honest failure path: one fixed
model-stop result, exact Pi error/history, zero retry and no invented finish.

## Baseline panel preparation

`proc_a428` passed ten CPU tests and a throwaway freeze. The complete predeclared
panel has26 cases: five exact distances each for verbatim copy, key/value binding
and path emission, plus11 first-action tool-choice classes. Stage A is fixed at12
episodes—copy/bind/path at128,8,192 and58,000 filler tokens, plus direct, FFF and
current-web contrasts. Stage B has14 cases and cannot run before Stage A review.
The throwaway panel SHA was
`6da40b2f13a8096e0c0da7af2746b4d611a9dcc78368b80e8c142cf598449932`.
The first immutable freeze `proc_913d` passed tests but produced SHA
`043cc9ef607da5a838a730347a299dc8d7e18e28cd8668167cb33c1bb6c947dd`:
the panel embedded the absolute source-worktree manifest path. This is a
reproducibility failure, retained with zero model generations. The builder now
uses the stable repository-relative authority path. `proc_61f3` reproduced the
corrected panel byte-for-byte from two directories at SHA
`3636c310b4ba82c504ef810af1dca9caabd586943ed4ed799b33e97d39c9b6f1`, and
`proc_740e` reproduced it from immutable source. Before sampling, review found
that58K-delay prompt plus the declared4,096-token generation reservation could
exceed64K. That otherwise valid root is superseded. `proc_c68a` fixed the explicit
per-turn budget at1,024, added a build-time prompt-plus-budget guard and froze SHA
`07cc1d5843ee4060503ecdcf7d2bb813dce51fa0aeba89df367b7696e47994ef`.
`proc_f909` reproduced the final panel from immutable source at SHA
`07cc1d5843ee4060503ecdcf7d2bb813dce51fa0aeba89df367b7696e47994ef`.
Stage A subsequently sampled exactly its12 authorized cases; Stage B remains
unsampled and unauthorized.

## Stage A evaluator preparation

The Stage A execution path registers exact captured core/process schemas through
a safety extension: confined `read` executes, while unexpected `bash`, `edit`,
`write` or `process` calls receive authentic explicit disabled errors rather than
running model code on the host. FFF remains real and read-only; the web extension
is real. `proc_e31e` passed11 tests, including actual Pi safe-read call/result
pairing, and froze the throwaway12-episode plan SHA
`d040e6e3d867795d29f45f043a8db818f9420ebcaa064a0acf5ad17a74f22370`.
The outcome gate is measurement completeness and unchanged weights, not a score;
every fixed model failure remains a result and no retry is permitted.

The first immutable Stage A launch `proc_1506` passed13 CPU tests and froze the
expected plan, then stopped before its first generation: Pi registered the exact
tools in a different order and the extension compared ordered name arrays. The
failed root records `pi_history_mismatch`; model fingerprints are unchanged,
gradients absent, peak HBM8,091,958,272 bytes and the lease released. Registration
order is not schema semantics. Both extension and owner now compare exact name/
schema sets independent of order. `proc_32e1` passed12 tests, including a real Pi
control with all11 Stage A tools registered together and an exact causal `read`.
A new-root Stage A launch was allowed because zero model episodes were consumed.

Corrected launch `proc_89f7` passed15 tests, reproduced plan SHA `d040e6e3...`,
and consumed exactly six fixed episodes before a different harness boundary:
128- and8K-delay copy/bind/path were0/6, all retained as honest
`generation_budget` or `invalid_frame` model failures. Starting the first58K
case then failed before generation because its prompt exceeded Linux's
single-argument byte limit. The six model results will not be rerun. Prompts now
enter Pi via immediately closed stdin rather than argv; `proc_5a31` passed14
tests including a real Pi prompt larger than131KB. A separately frozen
continuation binds the six immutable receipts and permits only the remaining six
Stage A episodes.

## Stage A result and decision

Continuation `proc_6451` passed16 CPU controls, froze plan SHA
`f205876aaf19e81edabe6d81121b731d423c1bfa765f1c53a7a3fcb784416809`,
and completed the remaining six episodes. Combined result: **0/12**, comprising
**0/9** copy/bind/path at128,8,192 and58,000 filler tokens and **0/3** direct,
FFF and current-web tool choice. All12 stopped on the first generation: eight
`invalid_frame` and four `generation_budget`; there were zero valid first
actions, tool calls, tool results, finishes, retries or optimizer updates. Thus
the run does not isolate long-distance memory: the current model first fails to
compose the adjacent Pi-native five-line frame under the full11-tool manifest.
This is actionable curriculum evidence, not evidence that the model lacks all
short or long copy capacity in a simpler interface.

Both model loads remained exact BF16, had no gradients and retained identical
before/after parameter fingerprints. Peak HBM was16,482,376,704 bytes or less;
both leases and all Pi processes closed cleanly. Summary SHA is
`0fcbaccaab22cd0e391244cdc8dc8c9194a42c52d59a8beb271166ee0578a9e3`.

Independent audit `proc_140a` verified all12 panel identities, prompts and prompt
token counts, initial causal records, exact Pi schemas/request histories, fixed
public Pi errors, unchanged source fixtures, model immutability, controller
authorities and cleanup. Its bounded verdict is
`qualified-negative-outcomes-with-incomplete-generation-receipts`; audit SHA is
`5aed2bb6305fd507e4444e24112353903d1dfe2d843e94c6705c4fc962f680fb`.
`proc_4e2a` reproduced that exact audit SHA from immutable commit `a381e003` and
verified its source archive unchanged. The evaluator retained stop reasons but not the rejected generated token IDs,
so the eight/four stop-reason split cannot be independently reconstructed from
bytes. The exact0/12 observable outcomes and no-tool/no-finish histories remain
verified. Future evaluations persist private generation receipts. These episodes
must not be rerun, used for training or reinterpreted.

**Decision:** do not run Stage B. First build verified Pi-native frame/tool-choice
curriculum and diagnostic controls; only then freeze a new, disjoint evaluation.
No training tranche, promotion or RL is authorized by this result.

## DeepSeek teacher bootstrap

LunaRoute exposes `deepseek-4.1-flash` and its background variant. The first
in-session subagent workflow `a59cdcfb...` made zero teacher calls because
`pi-subagents` had loaded before the LunaRoute extension and therefore rejected
the model as absent. Package order was corrected; direct provider smoke
`proc_9c0c` and a fresh-Pi inherited subagent smoke `proc_42a4` then returned
exact expected receipts.

Immutable teacher authority commit `e09e746a` freezes an80-specification pilot:
20 each for framing/copy, local repository work, recovery/composition and web
routing. Controller `proc_12a1` stopped before inference on malformed shell
redirection. Corrected `proc_c073` used one DeepSeek parent and four DeepSeek Pi
subagents and produced all80 specifications. Admission initially rejected the
harmless filename `secretless_target.txt` because `secret` was matched as a
substring. No teacher regeneration was allowed. Boundary-corrected validation
`proc_aa1c` passed the identical raw output:20 per lane,20 failure-prefix tasks,
unique prompts/IDs, bounded relative workspaces and exact task-pool SHA
`9c453abb1a0d4600a4aad0ef570b14e44fd42c69bb8eedfe06f8dd585fd9960c`.
These are task specifications only, not executed traces or training records.

A first direct-task DeepSeek execution `proc_8af2` reached the correct final
`PhaseFrame` but unnecessarily called `bash` first, proving why teacher output
cannot self-admit. That trace is a failed routing attempt; any future correction
must mask the prefix. Before executing shell/edit/process tasks at scale, the
teacher/provider process must be separated from a credential-free bounded tool
sandbox. Host-built-in execution is not an admission path.

The credential-separated proxy smoke `proc_f93d` registered the full frozen
surface but routed local tools over an owned Unix socket into the existing
nonroot, networkless NativeSandbox. DeepSeek returned `PhaseFrame` directly
with zero calls. The process status was nonzero only because the smoke driver
asked the trusted snapshot reader for an unsupported empty name set; the owned
container was removed and the model/tool boundary itself passed.

A deterministic authored real-Pi collector was then qualified before any scale
run. Pilot r1 failed closed after nine complete records because `fetch_content`
correctly blocked a loopback fixture, leaving the authored final ungrounded.
Pilot r2 failed closed after ten complete records because a short public HTML
page produced an incomplete-extraction error. Neither failure published a
candidate authority. Public fetch fixtures were restricted to verified RFC
Editor plaintext.

Pilot r3 (`proc_e75f`) passed all20 records against frozen plan SHA
`e8d6ba0b05423862e90450ad83ad20d8f9d57c50ba1131ea59cae9db1bcc5745`
and source commit `97a2e1f2`:28 genuine Pi calls,4 authentic failure
observations,20 distinct sequences,145,293 total tokens,2,155 supervised
assistant targets,18 families and9 initially classified repository-discovery
records. Independent reconstruction reproduced native records, generated token
IDs, public Pi event projections, causal call/result IDs, terminal finishes,
workspace/web oracles, prefix masks, candidate binaries and summary totals.
Audit SHA:
`b70beac83f2d3a3c6a2d254a0e839f013dd2f526b3232e2fdaa866e11dce3946`.
The output remains `verified-candidates-not-admitted`: training eligibility,
packing and optimizer updates are all false. The repository-discovery label was
subsequently narrowed to genuine search/discovery families and the programme
now asserts at least256 records across at least8 such families at2,000-record
scale.

Full-family qualification exposed two additional fail-closed seams. At recovery
case3, a short managed process emitted an asynchronous lifecycle custom message
that the strict provider rejected as unsupported history. The qualified recipe
now starts a bounded30-second process with notifications suppressed, observes
`READY` through the returned ID, stops that same ID and finishes; focused r3
passed3/3 calls and179 targets. At recovery case4, installed `fffind` expressed
an empty result as `No files found matching pattern` with `isError:false`; this
exact authentic wording is now treated as a maskable failed-search observation.
Focused tail validation then passed both remaining recovery families.

Qualification r4 completed80/80 under plan SHA
`2191d12d4eae092b7f39a79cc7544df88131f48d39e76e586d347ac546b2dd41`:
125 real Pi calls,11 tool-error flags,80 distinct sequences,725,804 tokens,
8,243 assistant targets,26 families, and a maximum record length of46,755
tokens. It covered all11 tools, fetch/retrieve, failed-fetch recovery, process
start/output/stop, all exact-copy distances through32,768 filler tokens and all
recovery variants. Independent reconstruction passed with audit SHA
`8cce3549e8da7279728cca5c41d13cfe8efcd884ad07a1a6661e5791b8d8177e`.
It remains a non-admitted qualification authority with zero model generations,
zero optimizer updates and no packing or promotion authorization.

## Historical session evidence immediately preceding this programme

- Failed model-session preflight `proc_79c2`: session-token plan mismatch; zero
  generations, unchanged weights and cleaned resources.
- Corrected model session `proc_77c4`:2/2; plan SHA
  `5265d88c24c0953b655a3227dc15f2061c50ce1b736958769b5600912b371e78`,
  summary SHA `5217ba6693530eb5dba7f414126d13044f5d2dda1aa66e9246ac1467b6f710de`,
  audit SHA `aeeed83cd9e4ed0b0f9e0a7b06ebb82bd8f43b6f5ddbd1849abec1a521cde1c0`.
