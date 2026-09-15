# Pi session boundaries and miniature repository probes v1

The operator instructed us to continue after the
[unchanged-weight compatibility pass](e97-pi-native-compatibility-v1.md), rather
than stop at each engineering milestone. Continue interface development and
bounded qualification with the checkpoint unchanged. No new training, source
admission, production installation or checkpoint promotion is authorized here.
All earlier training/evaluation budgets and results remain closed and intact.

## Session boundary candidate

`scripts/e97_pi_native_session.py` coordinates explicitly separate native tasks.
It is currently CPU-only control-plane code, **not yet integrated into a real
multi-task Pi transport**. `NativeEpisode.finish` still ends its source record.

- Require an explicit new-task operation, fresh-record acknowledgement and the
  exact prior public-history hash. Never infer a reset from ordinary follow-up
  text, reopen a finished native record, or silently compact history.
- Preserve all private native records and the cumulative public history.
- Reuse the same executor/workspace/shell, while clearly recording that previous
  tasks are **not** part of the new task's model context.
- Reject unverified settlement, stale/modified history, duplicate task IDs,
  active-task replacement and unauthorized continuation. Bound total tasks,
  generated tokens and wall time independently of each task's own budget.

This tests workspace continuity across explicitly independent tasks, not model
conversational memory, cache persistence, resume/fork or long-session competence.
`proc_c220`:44 CPU tests passed, including13 new boundary/state tests. The existing
single-episode adapter and immutable compatibility authority are unchanged.
The combined session/repository/native regression suite (`proc_5281`) passed
all108 tests before freezing the real-sandbox preflight.

## First-party miniature repository probes

`scripts/e97_pi_repository_probes.py` defines four authored repositories:

- Two threshold worlds with opposing inclusive/strict boundary specifications.
- Two selected-row lookup worlds with different field names, reordered inputs,
  absent keys and empty-input cases.

The same user instruction asks the agent to discover the project, repair its
implementation, and run tests without changing tests/documentation. README,
implementation and unittest files are available only through the real sandbox.
These are constrained **authored repository-workflow probes**, not independent
public-repository or benchmark evidence. No protected repository is used.

Before any model sampling, freeze and execute eight-turn authored controls:
discovery, README, failing tests, test source, implementation, repair, passing
tests, finish. Require the actual native test exits to be1 then0 and unchanged
README/tests in the paused-reader snapshot. Independently rerun the original
tests in fresh networkless sandboxes with the returned implementation, and run
the broken original as a negative verifier control.

Returned model code must never execute on the host. A deliberately narrow
pure-function AST guard precedes fresh-sandbox verification; unsupported shapes
are a separate verifier limitation, not proof of incorrect task behavior. The
AST guard is not a semantic oracle or a general Python security boundary. The
original sandbox isolation remains mandatory. Extra workspace files and changed
tests/documentation are never copied into the fresh verifier.

`scripts/qualify_e97_pi_repository_probes.py` is the authored executor preflight;
its responses are scripted, not model-generated or training-eligible. Four
controls, unchanged native eight-turn/token budgets, no GPU/model generation.
Only after preflight/audits pass should a separately frozen bounded model
comparison run. No threshold changes, automatic retries or training follows a
failure. Genuine model failures become diagnosis candidates, not an
excuse to silently replace model decisions with a teacher.

## Completed executor preflight and next frozen diagnostic

`proc_23ba` completed the immutable `2f065096` export in121 seconds. All108 CPU
tests passed again. Four real-Pi authored workflows passed:32 native turns,
28 actual native calls, four failing-then-passing test sequences and unchanged
README/tests. Four fresh verifier sandboxes passed returned repairs; four
negative-control sandboxes failed the broken originals. A separate post-run
auditor reconstructed all32 native prompts/token sequences, bound actual reader
outputs and test exits, and verified all16 distinct container cleanups. Source,
Pi runtime and interpreter audits passed. No model was sampled.

Artifacts: `R/pi-repository-workflow-v1-control` and
`R/pi-repository-workflow-v1-preflight`, with
`R=/mnt/nvme2n1/erikg/e97_systematic_posttraining`.

Before new model sampling, two additional real-Pi scripted CPU controls verify
honest invalid-opening and turn-budget failure termination. The original bridge
still reports an unverified finish for these failures; a separate read-only
checker verifies the exact unchanged public prefix, one empty error message,
owned peer, request counts and absence of extra model/tool retries. A failed task
is not upgraded to successful completion. Unknown connection, executor or
history failures still stop the experiment.

The next model diagnostic is exactly **four cases times two routes = eight
episodes**, one GPU, unchanged bridge checkpoint/live-y mode, original greedy
generator and eight-turn/8,192-token/600-second per-task limits. The direct route
is now the qualified native bridge's owner loop—not the older standalone harness;
the other route uses real Pi. Order alternates by case. No authored repair or
expected source enters the model prompt or sandbox fixture.

Freeze capability separately from measurement/transport: Pi must complete at
least2/4 tasks, with at least one success in each family. Require per-case route
success parity. A task success requires native finish, a passing test command
actually issued by the model, and a passing fresh-sandbox verifier using the
unchanged tests and final snapshot. Unsupported implementation shapes are
reported separately. Even0/4 can be a complete, useful measurement, never a
capability pass. Retain parameter/buffer fingerprints, BF16/no-gradient/HBM
checks, all snapshots and cleanup, and do not add attempts or updates.

## Completed model diagnostic: capability gate failed

`proc_414b`, source `00ca93dc`, completed all eight episodes in506 seconds including
preparation, with116 passing CPU checks. **Direct0/4, Pi0/4**; per-case route parity
passed, but the frozen capability gate failed. Every episode exhausted eight
turns without finishing. All64 native calls were `str_replace_editor view` of
nonexistent `/testbed/testbed.js`, each followed by the authentic missing-path
error. There were **zero directory-discovery commands, test commands or edits**.
All original files were unchanged; all eight fresh verifier runs failed tests.
No verifier-policy rejection obscured a potentially correct implementation.

The independent audit reconstructed every native turn, prompt and generated-token
sequence, checked actual snapshots, fresh verifier exits, failure termination,
unchanged BF16 weights/no gradients, and24 distinct cleanups (eight task agents,
eight readers, eight verifiers). Paired direct/Pi native transcripts and generated
tokens were exact. There was **one unique model-visible trajectory across eight
occurrences**, not eight independent failure mechanisms: the common prompt and
repeated nonexistent-path observations never exposed differing repository files.
Peak allocated GPU memory was8,551,257,088 bytes. Source/Pi/interpreter audits
passed; all GPUs were idle and leases/owned containers absent after closure.

This establishes an observed discovery/error-recovery bottleneck, not a Pi-only
regression. Because the model never reached source or tests, it does **not**
establish whether it could repair the functions once located, nor establish
context erasure or a numerical cause. The earlier8/8 structured-task compatibility
pass remains valid and scoped. No retries, extra model runs, training or promotion
are part of this closed diagnostic budget.

Artifacts: `R/pi-repository-model-v1-control` and `R/pi-repository-model-v1`.

- Plan SHA: `05e97fccdbc54eed1894f260f31eb88cf3a49343c02428a8fcdaea5e21f2d378`.
- Summary SHA: `77f9a708f6bbaf5f9083c5c946f1c34f2b6ec87c7e15bf8d04f9d2afb25f0e6d`.
- Independent audit SHA: `d99c1ce422c3c1d0043d910f5a43601165e6c5edf264856e395ab5cc275f3c1a`.

## Continuing without contaminating the evaluation

The next CPU preparation uses eight **different, training-only** authored sum and
scaling repositories with seed-bound module/test filenames. Four start with an
authored missing-path error, four with clean discovery. The teacher then lists
real files, reads README, observes failing tests, inspects the actual source,
repairs, reruns tests and finishes. Fresh sandbox verification and negative
controls remain mandatory. This is preparation for a later curriculum decision,
not another model attempt at the failed evaluation tasks.

`prepare_e97_repository_discovery.py` rejects exact reuse of the four evaluation
prompts, non-README paths, file contents and repairs; `README.md` is only a shared
convention, not shared contents. This is bounded overlap evidence, not universal
corpus independence. The failed evaluation episodes are not training records.
Authored failure prefixes are explicitly **not genuine model failures** and are
marked for zero loss (`supervise_assistant_from=1`); clean records start at0.
Preparation emits verified source receipts and mask-intent metadata, **not an
admitted or packed SFT dataset**. `training_eligible:false` and all registries
remain unchanged. New training requires a separate explicit budget and frozen
retention/capability gates; no update is automatically authorized by this work.

`proc_1c41` completed this preparation in216 seconds after120 passing CPU tests.
The independent audit verified all eight distinct candidates:52 exact native
prompts/turn token sequences,44 native calls,12 authentic errors,48 potentially
supervised suffix turns and four explicitly masked authored failure prefixes.
Eight fresh verifiers passed the repairs and eight negative verifiers failed the
originals. All104 Pi assistant/tool-result messages preserve public/private
boundaries, and32 distinct containers were cleaned. There were zero model
samples, updates or admission changes.

- Preparation plan SHA: `8d7c3a9a9843bc1464eb74d60010541181b8b631bca32b2e6370ed97b3a575e6`.
- Summary SHA: `3688fed7a0e3402e9af1b91c3225b41be27192f16de72fb401ac18b539257af5`.
- Independent audit SHA: `78d75122c51a139ae98b813af2ec21d7437d60858d2ff482ed296b0daacb5607`.

## Real multi-task Pi lifecycle progress

The single-task extension now has a separately selected session mode driven by
Pi's documented JSONL RPC interface. The owner begins each explicit native task,
strips the exact retained prior public prefix before native generation, settles
only a verified finish, and closes after the configured task count. One Pi
process and executor persist; each model-visible native record resets.

Two failed CPU attempts are retained:

1. `proc_c6f2`: expected Pi startup `model_change` metadata was incorrectly
   treated as an operator model switch, so settlement failed closed and the
   single-task regression tests also detected it.
2. `proc_50f6`: both tasks and all13 expected owner RPCs actually completed with
   two verified closures, but the Python client classified normal stdout EOF
   before its simultaneously ready process-exit descriptor as failure.

The metadata guard now allows Pi's startup record while still binding the active
provider/model at every generation. The EOF/exit ordering is accepted only after
all task settlements and session closure are already verified. `proc_0041`
passed36 focused tests; `proc_c580` passed the expanded **121-test** suite,
including real single-task, honest-failure and two-task Pi loops.

The frozen scripted control used one real Pi process and one qualified OpenHands
sandbox across two explicit native records. Task one established an environment
variable and created a file; task two observed the persistent shell/workspace,
edited and verified that file.

`proc_d8d3`, immutable source `edd3b103`, passed in31 seconds after122 CPU tests.
Both task closures and the final session close were verified. The audit independently
reconstructed all eight native prompts/turn token sequences, checked six real
OpenHands calls and21 owner RPC operations, bound the final paused-reader snapshot
(`seed.txt` unchanged, `shared.txt` changed from `alpha` to `beta`), verified the
environment variable and `/testbed` working directory persisted, and checked18 Pi
messages. Both private native records and the entire public history were retained;
task-one text/private reasoning was absent from task two's native context. The
single agent and reader containers were distinct and cleaned. Source, copied Pi
runtime and interpreter audits passed. Zero model generations or updates occurred.

- Plan SHA: `e49683f6c47e4ebf6433488b0dfc24cde0af00e98fb187269e308feecadc69a5`.
- Summary SHA: `a9370fda914020a8d3d3cb6c48db1e7667dfbb19eb80da1f0217bbbaecfca3`.
- Independent audit SHA: `d32376a78a84abb19ed251f95105b8d8acfae0a21ba6bcc92308dff48f77e16d`.

This establishes bounded lifecycle/executor continuity, **not conversational
model memory, model multi-task success, resume/fork/compaction, or general daily
interactive use**.

The failed-task settlement path is now implemented and passes a real Pi RPC test:
task one terminates with an invalid-opening model failure, its exact public error
and prefix are retained without inventing a native finish, and task two finishes
as a clean native record. The final receipt distinguishes one failed settlement
from one successful closure. Prior-task text is absent from the second native
prompt. `proc_1f5e` passed37 focused tests and `proc_c668` passed the expanded
**123-test** suite. A model failure is continuable only when it matches the fixed
model-stop reasons and exact Pi error shape; transport, executor, history and
identity errors still fail closed.

The first immutable post-safety OpenHands rerun (`proc_f488`) passed69 tests and
the complete two-task executor scenario, then its controller failed while writing
the terminal receipt: Bash expanded `rc=$original` in the same `local` command
under `set -u`. It remains failed controller evidence; no product path failed and
no model ran. A new-root controller split the assignments. `proc_7cc1` then passed
with full source/Pi/interpreter after-audits, and the updated independent audit
again reconstructed two records, eight turns, six calls,21 RPCs,18 Pi messages,
two closures and two distinct cleanups.

- Post-safety plan SHA: `e49683f6c47e4ebf6433488b0dfc24cde0af00e98fb187269e308feecadc69a5`.
- Post-safety summary SHA: `a9370fda914020a8d3d3d3cb6c48db1e7667dfbb19eb80da1f0217bbbaecfca3`.
- Post-safety audit SHA: `7e75b6da8275c97ded7d4f68bd845328730c983cfdf5a67f27108aa06abef9cb`.

The next model-facing check is frozen as a two-episode, one-Pi-session diagnostic:
the unchanged live-y bridge checkpoint must solve the already route-qualified
fresh lookup and fresh edit cases in order, with no supplied calls. It must retain
two separate native records, replay every native prompt exactly, verify task and
session closures, preserve public history without putting task one in task two's
model context, and leave weights/gradients unchanged. Failed tasks remain measured
and do not abort the second task. Both successes are the gate. These reused cases
provide lifecycle integration evidence only—no fresh generalization, memory,
training or promotion claim. The freeze plus expanded suite passed125 CPU tests.

The first model-session launch (`proc_79c2`) failed before generation because the
plan requested65,536 session tokens while `NativeTaskSession` correctly caps the
bounded total at32,768. This was a plan/preflight inconsistency, not a model or Pi
outcome: zero native records or generations occurred, the sandbox cleaned, the
GPU lease released, BF16 weights were fingerprint-identical, gradients remained
absent, and peak HBM was8,091,958,272 bytes. The failed root is retained. The plan
constant and runtime guard now both require32,768; focused tests and a new freeze
passed in `proc_409d`.

The corrected new-root diagnostic (`proc_77c4`) passed in311 seconds from immutable
commit `f7dd795d` after71 frozen-source CPU tests. The unchanged live-y checkpoint
completed both previously qualified tasks in one Pi/OpenHands session:

- lookup: three native turns, two executed calls, exact answer `68cd47b5`;
- edit: five native turns, four executed calls, correct output and exact `done`.

Both task finishes and the session close were verified. The independent audit
reconstructed all eight turns from generated token IDs, replayed every native
prompt hash, paired all six executor observations, regraded both filesystem/final
oracles, checked18 Pi messages and21 owner RPC operations, verified exact history
hashes and absence of task one from task two's native record, and checked two
distinct container cleanups. The first draft of the audit used obsolete helper
names and was corrected before passing; it did not execute the model or workspace.
BF16 weights were unchanged, gradients absent, peak HBM8,540,007,936 bytes, and the
GPU lease released. There were exactly two model episodes, zero retries and zero
updates.

- Plan SHA: `5265d88c24c0953b655a3227dc15f2061c50ce1b736958769b5600912b371e78`.
- Summary SHA: `5217ba6693530eb5dba7f414126d13044f5d2dda1aa66e9246ac1467b6f710de`.
- Independent audit SHA: `aeeed83cd9e4ed0b0f9e0a7b06ebb82bd8f43b6f5ddbd1849abec1a521cde1c0`.

This establishes unchanged-model success for **two reused structured tasks in one
explicit session**. It does not establish conversational memory, implicit
follow-up semantics, fresh repository discovery/generalization, compaction,
resume/fork, Pi-native tools, broad daily use, promotion, or learning.
