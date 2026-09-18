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

The scale plan froze2,100 one-shot attempts with a2,000-record minimum, no
automatic retries,2,100 unique prompts,616 planned genuine repository-discovery
records across8 families, exact runtime/source identities and plan SHA
`e921356e62c79c676bf029424f51035502572645869526047769742c39990345`.
Collection `proc_0a69` completed2,099 verified and one rejected. The sole
rejection, `pi-native-web-00144-127c37d7`, received the authentic observation
`Error: No search provider available`; it was retained and not retried. The
verified authority contains21,255,796 tokens,215,000 supervised assistant
targets,3,438 real Pi calls,303 tool-error flags,2,099 distinct sequences and
616 repository-discovery records. Authority SHA:
`81c70b767f38df76c39dd9aa402489953d754df492d07fb46516345fef6a3487`.
Independent reconstruction of all transcripts, raw Pi events, call/result IDs,
terminal closures, task oracles, web grounding, generated token IDs, correction
masks and binary indexes passed with audit SHA
`61dc1fb76288db0be974aa574f6d6f89479e0348fabc19179e013cd064ad56ce`.

A frozen decontamination selection excludes the entire87-record
`preservation` family rather than adapting individual examples to protected
content. The resulting2,012-record candidate has19,616,468 tokens,204,294
assistant targets and retains all616 repository-discovery records. Selected
authority SHA:
`8cf83db8d608f6af74bbd2cbe60206ef435fb3003ab2bb63f4eadab17eb05562`;
independent byte-selection audit SHA:
`de977726f0758206ee9defe9c3687d2485e3ba2b80eb9ef9ceafb065c5f6646b`.
The fixed V3, V4 and real-repository protected panels have zero collisions in
all exact and normalized domains. Against the separately frozen Stage-A panel
SHA `07cc1d...`, exact prompts, normalized prompts, task IDs, paths and every
significant content/scalar/final entity have zero collisions. Reported
structural reuse is limited to one family label, one normalized path shape and
trivial `alpha`/`1` content; this is explicitly not treated as entity reuse.
Overlap audit SHA:
`9c104d73b104f4d29a83ca1472b66ef2df681e020a2b9688eb4f87c6ef814a78`.
All source, selected and overlap authorities still declare
`training_eligible:false`, `packing_authorized:false` and zero optimizer
updates.

A nonauthorizing training preparation combines the selected2,012 Pi-native
records with all2,223 records from the already-qualified representation-bridge
authority. It contains4,235 records,25,673,057 tokens and1,052,954 assistant
targets. Preparation manifest SHA:
`76b947c892e5931579ddc5394663e9047a7f530ddafc2c5f194fa72550b2ed89`.
A diagnostic-only boundary pack plan produces438 whole-record64K packs with
explicit recurrent resets and no oversize exclusions; pack SHA:
`25280c14b0323b9981fa296293cfea2e5fa90104d04662fc321cc63275c5bdbc`.
The training Dataset continues to reject this mechanical authority because it
is non-trainable; exposure planning is descriptor-only and never materializes
loss-bearing tensors.

The proposed32-update,8-rank epoch-permutation schedule uses sampler key975424,
selected by training-data-only screening to retain every prior record exactly
once before maximizing new-record exposure. It schedules15,585,765 input
tokens and995,687 targets:147,027 Pi-native,300,316 conversation,126,193
grounded rehearsal,103,665 native,218,384 prior representation bridge and
100,102 retention. All2,223 prior records and1,316 distinct new Pi-native
records are covered. Schedule SHA:
`73e25ee832af63b12847817f09e77af9b3364b89c303e7d9678db25e3eeaaa4e`.
Proposal `configs/pi/e97-pi-native-curriculum-training-proposal-v1.json` was
frozen without authorization; proposal SHA
`96db2e1cee91304b6c295113e517c8c1fd032c40a7fb372d9866dea8ada057be`
and audit SHA
`a27fb97bc1567188a590b9630cae2385175d0152cd23cc546fa46f181da4ba09`.
It proposed32 updates at learning rate1e-5 from checkpoint `9b78628d...`, with
no automatic retry/expansion, no threshold change, no promotion and no RL.

The operator subsequently authorized that exact proposal. A separate admission
preserved the exact first-epoch pack sequence while changing the mechanical
preparation into an explicit32-update authority. Admitted authority SHA is
`ec58b6ae200f3a65a82a1a932bf074caa31885a19eae6694bdcaf1ba53d62865`,
pack SHA is
`bfd592cca7980eee47d1ebe318f554e001339336ac89e01bb77ce8d188717e67`,
admission SHA is
`ea5c9216305010293a372089979e72556e8e936d2ad1b16ed41f3840076a7ef4`
and the admitted schedule SHA is
`2404fb024562cd108169aa157cf5b631ab426e5c783bce0f1f24105337d99e69`.
Its source exposures are exactly the proposed totals.

Authorized run `proc_9453` completed once with no retry:32 updates,
15,585,765 input tokens,995,687 targets, peak rank-0 HBM36,128,739,328 bytes
and final aggregate loss0.3309566. The finite complete checkpoint is
`pi-native-curriculum-training-v1/checkpoints/checkpoint_agent_sft_u000032_loss_0.3310.pt`,
SHA `e9c2d47b24dc419b3ec6c354a77ea585dd00cfc6697a086e1d975ddfc08e3296`.
The GPU lease EXIT trap replaced the launcher's EXIT trap, so the expected
in-run terminal receipt was absent; this controller defect is retained rather
than concealed. Process exit0, lease cleanup, and complete post-run
source/input/inventory checks were bound in a separate controller terminal
receipt. Independent full-checkpoint audit reconstructed all32 sample-ID rows,
token/target clocks, source exposures, optimizer state and4,045,972,080
coordinates, verified finite BF16 tensors and the atomic latest pointer, and
passed with audit SHA
`e654bf94dc3402dfb5cc820c739e3f1bedceee59e617e379d224f788cd42406d`.
The checkpoint remained unpromoted.

The first protected14-case Stage-B attempt used frozen plan SHA
`f198af9982748614803cb495b2a5797806e48ede2d5f0547f78ae451e118787d`,
but inadvertently loaded the Stage-A first-action executor, which deliberately
disables `bash`, `edit`, `write` and `process`. Its first-frame/action projection
was11/14 and7/14, but its full outcomes and original audit verdict are
superseded by retained protocol-invalidation SHA
`8524121c9dddd4fbad8aea9ab018bfbe2feba41e67f7a4a1d3f2228a7fc4535a`.
It is not silently discarded or presented as a valid end-to-end measurement.

A corrected exact-schema executor backed all five local tools with the
hash-pinned NativeSandbox, passed an independent six-call real-Pi smoke, and
was frozen under new plan SHA
`f8f5fca6445a5fe35b050201da9d05c9770ca490452d2cb07fac6056fb690ff1`.
The corrected attempt again measured11/14 valid first frames against required12
and7/14 correct first actions against required10, with one frozen end-to-end
success. The independently reconstructed verdict is
`qualified-negative-gate-failure-corrected-protocol`, audit SHA
`acac2c487461ef1eddd17413aff13c96136d29d2e08c8f277638f131af401669`.
Two pre-existing expected-final inconsistencies (`choice-read-explicit` and
`choice-grep-symbol`) remain recorded without regrading; both primary gates fail
independently. Stage A remains immutable0/12 and checkpoint promotion remains
blocked.

A subsequent non-promoting8-GPU matched diagnostic evaluated the Pi-native
candidate and representation-bridge control in both saved and train modes over
96 execution cases plus32 retention examples. Both bridge modes retained67/96
execution successes: prior regression10/16, prior fresh16/16, fresh32/32,
prior transfer5/16 and composition4/16. Both Pi-native modes scored0/96,
primarily through64--66 invalid frames, despite the partial Pi-native Stage-B
gain. Tool-retention NLL also rose from approximately0.00035 to0.079--0.086,
and native choice matching fell from2/2 to0/2 in both fitting and development
samples. This is strong representation-interference evidence, not a promotion
gate rescue. Independent diagnostic audit SHA:
`0714a89a9bd3d4d4c0b8d8f74b464985720a3b88c83c6493470ac2fe71ead6db`.
Protected outputs remain ineligible for training. The Pi-native checkpoint is
an unpromoted experimental branch; the representation bridge remains the last
fully gated checkpoint. Further optimizer updates and RL require a new explicit
authorization.

A deterministic FP32 task-vector sweep then tested whether the candidate's delta
could be partially merged back onto the parent without the damage. Merges were
computed per tied tensor group in FP32 with a single BF16 rounding (merge
manifest SHA `c7688b4b0c0f34a157dd6b6a94e988bdbcb848bfea8ce3eb72ba87fdcd53a2a2`),
bit-exactness verified on disk. Results across the frozen 96-case execution
panel, the 32-example learning panel, and the corrected Stage-B panel:

| eta | execution | Stage-B valid frames | correct first actions |
|---|---|---|---|
| 0 (parent) | 67/96 | 0/14 | 0/14 |
| 0.1 | 67/96 | 0/14 | 0/14 |
| 0.25 | 69/96 | 0/14 | 0/14 |
| 0.5 | 58/96 | 9/14 | 8/14 |
| 1.0 (candidate) | 0/96 | 11/14 | 7/14 |

At eta 0.1 and 0.25 the merged models are parent-identical on every retention
and likelihood panel while carrying zero Pi-native first-frame capability; at
eta 0.5 the Pi-native gain partially returns (9/14 valid, 8/9 correct among
valid frames, higher action precision than the raw candidate) only as the
OpenHands execution score degrades to 58/96. The gain and the damage are
entangled along this single interpolation direction; no eta dominates the
parent. Stage-B audits:
`2df60379ef93efaa6ef8fc1432abc754b673516cfa4abe09b9de89c4b2650d94`,
`b4aecd37f7a13bdb023644ad71a916ad10047135e8542054ba0a8b7ec29a5da1`,
`6a9651da9cc668d7327662b3d0220339d05cab110416f95ddce076d69aad9959`.
Task-vector arithmetic is therefore not a substitute for a rehearsed repair
tranche; it remains available for specialist branches whose deltas are not
entangled with existing capabilities.

## Repair-tranche proposal (frozen, not authorized)

The repair preparation combines three cohorts in one non-authorizing authority
at `pi-native-repair-preparation-v1`: 115 shortest whole-record OpenHands
execution trajectories (699,784 targets) drawn deterministically (seed 742297,
shortest-first with seeded tie-break, distinct `problem_key`) from the qualified
source-native fulltraj authority (`255b02f3...`, the same corpus that produced
the execution capability), all 2,012 selected Pi-native records (204,294
targets), and all 2,223 bridge rehearsal records (848,660 targets). Records are
emitted in a weighted-fair token-share interleave so greedy 64K boundary packing
produces cohort-mixed packs: 4,350 records, 28,701,962 tokens, 1,752,738
targets, 499 packs, zero oversize exclusions. Authority manifest SHA
`fe1026da32b89276bc484eacdabc73f877dde9f2bd7011aecd601826eed72a08`; pack SHA
`263ef5b8d77a792951cfb1ddb52d8b0ba9740993f576a2d8ada2f99a1bfc4c1b`.
Two earlier selection drafts (median-length slice 33 records; 78-record slice)
are retained at `pi-native-repair-preparation-v1-draft1-median-selection` and
`-draft2-78records` as evidence of the pack-density constraint.

The frozen schedule uses sampler key 700129, screened over keys 700001..700238
with the hard constraint that **every one of the 32 updates contains at least
one record from each cohort**. It schedules 15,217,854 input tokens and 954,424
targets: 362,607 OpenHands rehearsal (38.0%), 135,774 Pi-native (14.2%),
456,043 bridge rehearsal (47.8%). Schedule SHA
`414f6b17a13b76290b5bb552d67e478eaabaa394f0b9ac4c22c71b0b2c459c33`.
Proposal `configs/pi/e97-pi-native-repair-training-proposal-v1.json` is frozen
without authorization; proposal SHA
`3eef7a0aeac1311bdd8dee2468cf63f8d154b6919563e8c8956213d0e2593b05`, audited by
`scripts/audit_e97_pi_native_repair_proposal.py` with receipt SHA
`112123e327bf59d04194ea9533f6851f817daf43561bf8cec80d10d2273e5d4c`.
The frozen dual gate requires BOTH sides: Pi-native Stage-B valid frames >=12/14
and correct first actions >=10/14, AND OpenHands execution >=64/96 with
prior-regression >=10/16, prior-fresh 16/16, fresh >=30/32, transfer >=5/16,
composition >=4/16, plus both x/y retention gates. A direct mechanical scan of
the 115 selected trajectories found zero protected-panel fixture entities
(no `VALUE_*`, `weather_adapter_*`, `sum_all-*`, fixture file names); only
incidental common technical words. The two earlier drafts, the schedule
screening, and the proposal remain non-authorizing: no packing admission, no
optimizer updates, no promotion. Admission, training and evaluation require a
new explicit operator authorization of this exact proposal.

## Authorized repair tranche: trained and dual-gate failed (no promotion)

The operator authorized the exact proposal. Admission nonce 242038 preserved the
frozen epoch-zero pack sequence; admitted authority SHA
`25366863667b7d378af3dcea8ad51acee43cd84ccb4b60a97b6d1c6de5dde46e`, pack SHA
`e4a074d2f0e0908985bf851179fa5a344527c5aa451d4ab0de4f2b5744f0dad0`. Training
`proc_3d19` completed once in 2,738 seconds with no retry: 32 updates,
15,217,854 input tokens, 954,424 targets, final aggregate loss 0.4924.
Checkpoint `checkpoint_agent_sft_u000032_loss_0.4924.pt` SHA
`0085efcdd82299fc9ce16be8e8d8464bcee2b92bdb4cad991e77918de48ed908`, atomic
latest pointer matching. The in-run terminal receipt is again absent (documented
lease-EXIT-trap defect); source and input inventories verified clean after exit.
Independent audit SHA
`46fad67bf63cbdb66ed5b62d7fb508d27ff94ff760136727aa789f3fbc38203a`: exact sample
IDs and clocks, finite BF16 state, source exposures exactly as planned
(362,607 OpenHands rehearsal, 135,774 Pi-native, 456,043 bridge rehearsal).

The frozen dual gate FAILED; no promotion, no rerun, thresholds unchanged.

| Panel | Repair checkpoint | Required | Bridge parent reference |
|---|---|---|---|
| OpenHands execution | 63/96 (prior-regression 5/16, prior-fresh 13/16, fresh 32/32, transfer 8/16, composition 5/16) | >=64/96 with floors | 67/96 |
| Pi-native Stage-B valid frames | 13/14 | >=12/14 (passed) | 0/12 Stage A |
| Pi-native Stage-B correct first actions | 8/14 | >=10/14 (failed) | 0/12 Stage A |
| Tool retention token accuracy | 1.0000 | pass | 1.0000 |
| Native fitting/development choice match | 2/2 and 2/2 | pass | 2/2 and 2/2 |

Stage-B audit SHA
`b936452b84c5af7edffab72ba4e2cf4824ca5d37b1a5307aff06b230f17ef7ff`
(`qualified-negative-gate-failure-corrected-protocol`). The interference is
repaired -- execution rose from the raw candidate's 0/96 to 63/96 with
Pi-native framing simultaneously at its best measured level (13/14 valid
frames) and tool retention exact -- but both gate legs fail: action routing
(8/14 against 10) and the oldest OpenHands regression cohort (5/16 against 10,
with prior-fresh 13/16 against 16). Notably transfer improved 5/16 to 8/16 and
composition 4/16 to 5/16 above the parent, and conversation retention moved
only marginally. The checkpoint remains unpromoted experimental evidence; the
representation-bridge checkpoint remains the last fully gated checkpoint.
Further optimizer updates require a new separate authorization.

## Repair-v2 tranche: Pi-native gate passed for the first time; execution leg still short

The operator authorized the repair-v2 cycle. Four-cohort admission (nonce 33224):
authority `b199edd94f8dfeacd5871e818412fe2ee68aa4461f3fc1b2e8e1cb2451b50485`,
packs `30996a008dea0a30959a1fc475be3c40544fdee557af5aa6274152d0417dbc43`.
Training `proc_a6cd` completed once in 2,664 seconds: 32 updates, 15,308,011
input tokens, 1,311,588 targets, final loss 0.4537. Checkpoint
`checkpoint_agent_sft_u000032_loss_0.4537.pt` SHA
`2f62a5c9acb50d442da161d3f0d698399b1007d4186077d2c58f91dd2258eaa7`; audit SHA
`52947df158b2ad892c924c18b39acdee84d650cd6b62c0fe462402b959236e07`; exposures
exactly as scheduled (276,144 OpenHands, 72,872 focused Pi-native, 308,236
authored concise-turn, 654,336 bridge).

Dual-gate result: **the Pi-native Stage-B gate passed for the first time** --
12/14 valid first frames and 10/14 correct first actions, audit status
`qualified-gate-passed-not-promoted`, SHA
`7ffc0e511c4869f3e745572b5cd6e90c1e0938cef1b170e2bbd96d9077cf64b4`. The v1
diagnosis-driven fixes verified: copy/bind cases now open with `finish`
(over-calling bias resolved), `process` and `edit` now route correctly.
Execution remains short of the frozen leg: 62/96 against 64, prior-regression
still 5/16 and prior-fresh 14/16; transfer held at 8/16, fresh 32/32, tool
retention exact 1.0000, conversation within tolerance. Episode forensics
isolated the remaining execution defect precisely: in every failing recovery
case the model repeats the identical failed tool call (`str_replace_editor view
/testbed/missing.json` and the same error) until turn budget, never attempting
the corrected path -- the error-loop-breaking pattern from the original
grounding-correction training resurfaced. The repair2 checkpoint remains
unpromoted; the dual gate still fails as frozen, and further optimizer updates
require new authorization.

## Historical session evidence immediately preceding this programme

- Failed model-session preflight `proc_79c2`: session-token plan mismatch; zero
  generations, unchanged weights and cleaned resources.
- Corrected model session `proc_77c4`:2/2; plan SHA
  `5265d88c24c0953b655a3227dc15f2061c50ce1b736958769b5600912b371e78`,
  summary SHA `5217ba6693530eb5dba7f414126d13044f5d2dda1aa66e9246ac1467b6f710de`,
  audit SHA `aeeed83cd9e4ed0b0f9e0a7b06ebb82bd8f43b6f5ddbd1849abec1a521cde1c0`.

## Repair-v4 tranche: targeted loopbreak family missed; defect refined to pointer-chasing

The operator authorized the repair-v4 cycle. A new verified collector family
`tool-error-alternate-recovery` (240/240 verified one-shot, 600 real calls, 240
authentic ENOENT errors, 23,910 supervised targets, audit
`5f27d6be2073d87b1700cc01c9f7623ca9e4a147ff62c4096c3f62613feecfb0`, zero
protected-panel entity collisions) was added as a sixth cohort. Admission
nonce 16049; authority `a4203772d2b13db955f88fc0a76ef7c548f68ccc2ca9d38fdcc96f1bb04f60cf`;
packs `5c4f2f39dc8cae77ac520d9b14aa7a84d0d990384480838157acd5d117c4ad47`.
Training `proc_5661` completed once: 32 updates, loss 0.4390, checkpoint
`5d4b82d78df573863b197e0c37a615dd5b9574cf84a0c8a818da308af63b0e26`, audit
`83ac88dea2c5145e22aa335bd847ab195d530ff2db2402de4e9a13cd4bc05448`. The
evaluation's Stage-B phase hit a sed path defect (repair3 directory, repair4
filename) and was rerun standalone with the corrected path; both retained.

Result: execution 66/96 with transfer 10/16 (best measured) and prior-fresh
15/16, but the four regression-recovery cases STILL loop identically
(`str_replace_editor view /testbed/missing.json` eight times), and Stage-B
measured 13/14 valid frames (best) with 9/14 correct actions -- four
`generation_budget` overruns. Episode forensics against the passing parent
trace refined the defect: the parent recovers by inventing a plausible sibling
filename (`catalog.json`), reading a pointer record inside it
(`{"active_path": ...}`), and following that pointer to the real file. The v4
family taught a stated alternate and a Pi-surface fuzzy search -- neither shape
matches this two-hop inventive pointer chase, and the 1.1% target dose was
small. The v4 checkpoint remains unpromoted; the dual gate fails as frozen;
further updates require new authorization.

## Repair-v5 long-horizon arc: capabilities recovered, gate fails on conversation retention

The operator authorized the long-horizon experiment. Forensics on repair-v4
showed the parent recovers by inventing a plausible sibling filename and
following a pointer record inside it (catalog.json -> active_path -> real
file); the v4 stated-alternate family did not teach this two-hop inventive
pointer chase. A new verified collector family `tool-error-pointer-chase`
(240/240 verified one-shot, 880 real calls, 240 authentic ENOENT errors,
34,096 supervised targets, audit
`3e6bb8cfb264d47eb88669132e5e8f17f5cb64b4335d20675c7d4b5805471580`; zero
protected-panel collisions) was collected. An OpenHands rehearsal-slice scan
found repeated non-adjacent identical calls in 57/63 selected records but zero
adjacent repeat runs: the loop is an amplified retry prior, not a taught
adjacency.

The v5 arc design: one frozen five-cohort mixture (bridge 49.6%, authored
23.4%, OpenHands 20.3%, pi-native 4.6%, pointer-chase 2.0%; correction cohort
removed to restore the v2 Stage-B-winning balance), authority
`88f2a8d022fe046b9668d427b3da61ca4d778bcb813e4a1b93d7d43c6161c890`, packs
`6a985cef92bdca55e2fa39e0247cbac260ffeb535d79dc9017f4713e0ecb5208`
(360 boundary-aware packs). Per-update five-cohort stratification is
structurally impossible in a single 128-update schedule (whole OpenHands
records fill 64K packs; 297/360 packs contain none), so the arc is four
chained 32-update segments, each with its own screened stratified key
(760031/760206/760212/760186) and admission (nonces 21912/24932/5055/913),
each parent verified against the prior segment's audited checkpoint by the
extended generic auditor. Segment proposals
`136401ec`/`ec5b3a8b`/`c53141e4`/`5e50c8d9`; audits
`030f1ea6`/`dc8bbf2a`/`2e7edd3f`/`9b7195c9`; training audits
`552db6a3`/`6c4d5aa9`/`ce0a4ac0`/`96bc5923`; loss
0.4319 -> 0.3274 -> 0.2647 -> 0.3221; checkpoints
`ebc3db60`/`695a17bf`/`329dd73b`/`d2276376`. Cheap-tier Stage-B juncture
detectors ran at every segment boundary: 11/14 valid 8/14 correct (u32),
12/9 (u64), 13/9 (u96), 14/11 (u128).

Dual-gate result at u128 (eval
`pi-native-repair5-arc-u128-dual-gate-v1`, execution summary
`33538ee5794e716cb158a6f3adc5cee6c8fc01aeb6b7db5148a60b09ab39a726`,
learning summary
`42849311bf476a9c42738a347393a263afdb84779e42388a2e0b819dc3447f05`):
Stage-B gate PASSED with the best score ever recorded -- 14/14 valid
frames and 11/14 correct actions (audit
`a01622abcaa2edc7bb126b54d6827d47bb08b7bf5ac84cb228ef652fa2384bf0`,
status qualified-gate-passed-not-promoted). Execution 70/96, ABOVE the
parent's 67/96: prior-regression 11/16 (all four loop cases now pass --
the error-repetition loop is broken), fresh 32/32, transfer 8/16,
composition 4/16. The gate still fails: prior-fresh 15/16 (one
invalid_frame on fresh-lookup-0001-world-1) and the conversation-retention
window (record NLL 1.841/1.832 vs limit ~1.72; the 128-update arc overspent
conversational likelihood). No promotion; all four arc checkpoints
retained. The checkpoint-juncture network provides the missing trajectory
data for the next cycle's stopping rule.

## Repair-v5 retention-curve probe: the constraints crossed between u96 and u128

A cheap-tier conversation-retention probe (8 held-out records, 8 ranks, ~4
minutes) scored the four arc checkpoints. Curve: u32 1.5958, u64 1.6355,
u96 1.6932, u128 1.8411 record macro NLL, against the bridge parent's
1.5716/1.5739 and the +0.15 window limit ~1.72. Conversation retention
held the window through u96 and broke in the final segment, exactly where
Stage-B correct actions crossed the 10 threshold (u128: 14/14 valid,
11/14 correct). Mechanism: bridge-cohort train loss fell 0.43 -> 0.26
across the arc while held-out conversation NLL rose -- at ~2.6 epochs of
repetition over the fixed 2,223-record bridge authority the model
memorized the rehearsal slice instead of generalizing. No single arc
checkpoint satisfies both constraints: u96 is within the retention window
(1.6932) but Stage-B correct is 9/14; u128 passes Stage-B and every
execution floor (70/96, prior-regression 11/16 with the loop broken,
transfer 8/16) but fails prior-fresh 15/16 by one invalid_frame case and
the conversation window by ~0.12 NLL. The probe is retained at
`pi-native-repair5-retention-curve-probe-v1`. The next cycle's design
inputs: fresh conversation-side rehearsal diversity (not more epochs of
the same records), the conversation probe added to the cheap juncture
detector tier, and unchanged frozen gates.

## Repair-v6 arc underway; path-alias incident recorded and fully recovered

The v6 arc (fresh SmolTalk2 conversation cohort 20.9% of targets, four
screened stratified keys 770233/770241/770402/770759, dual juncture
detectors) is training. Segment 1 completed and audited (loss 0.6969,
checkpoint `d03e34d3b3b122cafb13f786f53f4e9258f91c4429201047a872ebfce5737991`,
training audit `94a6ef9c3837f762814ba7abe56f0d38e3408a12ea485d7f241cbd27174d7738`);
its Stage-B juncture measured 11/14 valid, 8/14 correct.

Incident: a sed path-alias in the derived v6 juncture script left the
working directory pointing at the repair5 tree, so the v6 u32 Stage-B
evaluation overwrote the v5 u32 artifacts
(`pi-native-repair5-arc-juncture-evals/seg1-u32/`) before the binding check
exposed the contamination (the plan file names its checkpoint). Recovery:
the v6 results were salvaged to
`pi-native-repair6-arc-juncture-evals/seg1-u32/` with the checkpoint binding
verified, and the v5 u32 evidence was deterministically regenerated from
the retained `ebc3db609060179f...` checkpoint, reproducing exactly
11/14 valid, 8/14 correct -- both the restoration and the reproduction
verified. The v5 u32 numbers were additionally preserved in this ledger
and the pushed git history throughout. This is the third occurrence of
the sed-derived-launcher path-defect class (after the repair-v4 Stage-B
checkpoint path and the soup run's stage-b path); going forward launcher
scripts are generated from parameterized templates with a single
paths/checkpoint block and a pre-launch foreign-artifact assertion rather
than by string substitution. A related lease defect (stale
GPU_LEASE_Held flag blocking the second acquire) was handled with the
documented unset workaround.

## Hugging Face canary release: repair-v5-u128 published for collaborative testing

The operator authorized publishing the repair-v5 u128 checkpoint
(`d2276376d43c5b573cec2374b6b52b164bd39d7d38d17b8fe085196cb79e5f9d`) to
`spinozans/emender-e97-4b-pi-instruction-checkpoints` so external
collaborators can probe it and report failures. The release is explicitly
labeled an experimental canary that FAILS the frozen dual gate
(conversation-retention window and one prior-fresh case), with all
measured results and known weaknesses stated in the model card, the full
audit-chain receipt in `releases/repair-v5-u128/release.json`, and the
Pi-native eleven-tool surface, system prompts, and OpenAI-compatible
serve script in `integration/`. The stale u8-era headline (old 120-task
panel world) was replaced by the rewritten model card; prior releases
remain as superseded provenance. Weights verified byte-identical after
upload (8,478,128,544 bytes). This is a testing artifact, not a
promotion; the dual gate remains the only promotion path.

## REPAIR-V6 ARC: THE FROZEN DUAL GATE PASSED FOR THE FIRST TIME

The operator authorized the v6 cycle. The v6 arc kept every v5 cohort and
added a fresh production-admitted SmolTalk2 conversation-rehearsal cohort
(421 records, 20.9% of targets) to fix the conversation-retention
overfitting that killed v5. Preparation authority
`eef2213499ee997b7081f06c84b2cd46ee5a9a2ef2a15720df2737ab369d7beb`, packs
`6ab022abe010fe47d54a2046bb2a5d4e8e593df64aca5181331e7bc82d51867f`
(357 packs). Four chained 32-update segments, screened stratified keys
770233/770241/770402/770759, admissions nonces 24836/2416/16190/48551,
proposals `9abba5fc`/`7993f082`/`8371289d`/`80fdc466`, training audits
`94a6ef9c`/`a7bf9befe`/`03a4f0ed`/`15c6cfbe`, loss
0.6969 -> 0.6273 -> 0.5059 -> 0.4412, checkpoints
`d03e34d3`/`d5c82ec1`/`d8146498`/`b61dfb54`. Dual cheap-tier juncture
detectors (Stage-B + conversation NLL probe) at every 32-update boundary:
u32 11/8 + 1.574; u64 12/10 + 1.583; u96 12/10 + 1.654; u128 12/10 + 1.632.
The conversation window held at EVERY juncture (v5 broke it at u128).

u128 full gate (audit
`a66c9566e12b03f9667acd781cb9ddc6f640e2d5f39620b34565ba156ae90899`):
execution 71/96 best-ever (prior-regression 12/16, loop stays broken)
but prior-fresh 15/16 fails on the persistent lookup case
(fresh-lookup-0001-world-1, failing with a different mode than v5), and
the gate-protocol Stage-B rerun measured 11/9 -- exactly one case
(copy-32768, the 32K exact copy at the generation edge) flipped
valid->invalid relative to the same checkpoint's juncture measurement
30 minutes earlier, exposing a run-to-run noise band straddling the
frozen 12/10 threshold. Learning leg fully passed at u128 (conv NLL
1.6324 within window -- the v5 killer is fixed; dev NLL 0.954 within
window).

u96 (`d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b`)
FULL FROZEN DUAL GATE PASSED (eval
`pi-native-repair6-arc-u96-dual-gate-v1`; execution summary
`c41fa94423dfcd632baba6840b1c2df86f43003c6d8824f3816e54a7aa1ca821`,
learning summary
`8beb9c0cf2888635f5959a8b87453300d5074b45ce3fb53b92ec62452ba98734`,
stage-b audit
`795884d58c9cfe99bcfa28ff6f8de004801eb447542cee7ed639e2a35aa0a003`,
status qualified-gate-passed-not-promoted):
- Execution 71/96 (ties best-ever; parent 67): fresh 32/32 (>=30),
  prior-fresh 16/16 (=16 required), prior-regression 11/16 (>=10),
  transfer 8/16 (>=5), composition 4/16 (>=4) -- ALL FLOORS PASS,
  including the persistent lookup case.
- Stage-B 12/14 valid (>=12), 10/14 correct (>=10) -- PASSED, and the
  gate-protocol run reproduced the juncture measurement exactly.
- Learning: fitting 2/2, development 2/2, tool retention 1.0,
  conversation NLL 1.6544/1.6510 within the +0.15 window, native
  development NLL 0.9593 within window -- x and y both pass.

Both legs of the frozen dual gate pass simultaneously for the first
time in the programme's history. Promotion requires explicit operator
authorization; the u128 arc-end checkpoint is retained with its
near-miss evidence, as are all four segment checkpoints and every
audit chain. RL, repository competence, and production deployment
qualification remain separate later gates.

## PROMOTION: repair-v6 u96 (`d8146498...`) promoted to qualified baseline

On explicit operator authorization, `d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b`
becomes the programme's qualified baseline checkpoint, superseding the
representation-bridge parent `9b78628d...` in that role. Promotion receipt at
`pi-native-repair6-arc-u96-promotion-v1/promotion.json` binds the full gate
audit chain. Per operator direction the v7 cycle and onward chain from the
u128 arc-end checkpoint `b61dfb54...` (retained, unpromoted under the frozen
gate protocol, designated v7 training parent); v7's goal is to clear the
frozen thresholds by more than the measurement noise band (Stage-B 13/11,
prior-fresh 16/16 robust, composition 5/16) with detector-driven cohorts for
the two remaining single-case margins: the persistent lookup case
(finish-without-lookup) and the long-copy generation-budget instability.

## Repair-v7 arc: stopped fail-closed at segment 3 (retention headroom exhausted)

The operator directed v7 and onward to chain from the v6 u128 arc-end
checkpoint b61dfb54 and authorized autonomous overnight operation. Two new
verified families were collected for the remaining single-case margins:
`tool-error-finish-recovery` (150/150 one-shot, 8,171 targets, audit
19a868fa7764d925a3a47b90649b3a8db7f1b59e5a3ecb643168dc4d419635de) for the
persistent lookup case (finish-with-traceback after a failed extraction) and
`long-delay-copy` (180/180 one-shot, 7,224 targets across the 1K-32K delay
tiers, audit a2db00aea3aaaaea228da64d1ffbcaf4c1c56288e632da3580c34ef942c905ac)
for the copy-32768 generation-budget instability; zero exact-entity
collisions. Preparation 91e63bf883a49375a2d4f741b59e42291d63f5514957705b81938687295d66d7
(eight cohorts, 5070 records; a missing-pointerchase defect was caught and
fixed before training), packs b10594fa7493d68e4a17b228c388b5343effd7af7567a21c2eeb46360e3badc4
(440 packs), screened stratified keys 845420/861121/874360/898643 (the
tightest screen yet: 4 keys in ~180K). Auditor generalized for prior-arc
parents with audit-chain verification. Segments 1-3 trained and audited:
checkpoints 06b42f93/0c966bfa/1e2f940e, losses 0.5084/0.3685/0.2568,
training audits acd44958/8a9398bf/8c447c52.

Juncture curve: u32 Stage-B 10/8 + conv 1.671 (early dip absorbing the
fresh-seed conversation slice); u64 Stage-B 12/10 + conv 1.725 (recovered;
copy-32768 fixed by the longcopy cohort -- the targeted family worked);
u96 Stage-B 11/10 + conv 1.806. The retention window (about 1.722/1.724)
was decisively breached; the arc STOPPED fail-closed at three segments per
the pre-declared juncture rule. No v7 checkpoint is promotable.

Forensic conclusion: the u128 arc parent started with only ~0.09 of
retention headroom (conv 1.632 vs the 1.72 window); arcs consume ~0.03-0.06
NLL per segment, so the headroom arithmetic was exhausted by u64. The v6
arc held only because it started from the bridge parent's 1.573 with the
full 0.15 window. Additionally, replacing the conversation slice with a
new-seed SmolTalk2 draw coincided with faster measured drift, though the
dominant factor is the parent headroom. All checkpoints retained; the v8
design: the v6-passing recipe (bridge parent, conversation seed 613117 --
the demonstrated holder) plus the two tiny targeted families at a combined
0.8% of targets, run as a full 128-update arc from full headroom.
