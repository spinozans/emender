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
