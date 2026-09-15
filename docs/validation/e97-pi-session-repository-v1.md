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
failure. Genuine model failures would become diagnosis candidates, not an
excuse to silently replace model decisions with a teacher.
