# F9 — staged era-8 task diversity

2026-10-09, lambda01. **Code/tests only: no live admission, teacher API calls,
GPU use, daemon restart, model flip or live pin replacement.** Standing daemon
PID **450261** still has its original start time (02:56:57). Live lake teacher
SHA is `6881b6330cc7949bc6d21bc597c44f2a37cab6fa7ca91ec5681830d36d0449fe`;
repo supply SHA is `23d012883693e230beff847c2bc38f3794d6eb83082177bde224d57ccd9a62f3`.
`F9-evidence.json` binds retained source bytes, staged code, private proof
artifacts and source archive. No external seed download occurred.

## Family designs

**terminal** follows Terminal-Bench's seeded-workspace/multi-step/end-state
*pattern*, explicitly cited in the family module/instruction; no benchmark
instance is a data source. Natural first-party documents/code/config files,
2-6 files with the existing 150-4096-byte rich-workspace floor. Author declares
2-12 bounded reference actions and 1-8 assertions: exact file bytes, fullmatch
regex, absent files, optional read-only command exit codes. At least one
exact/regex artifact is mandatory. Proof rejects an already-satisfied initial
state and executes the reference actions through the real owned Pi process,
not a fabricated CPU/gym trace. Multiple real workspace operations are required.

**conversation** follows clarification/slot-filling, constrained-response and
user-correction conversation-agent patterns, cited in its family instruction.
The private validator spec holds a sealed deterministic user script, 1-3
follow-ups (2-4 user turns including the opening). Behavior classes are
question/json/statement/acted; absent branches reject. The first branch always
requires a question with no prior workspace action. Each intermediate reply
has a bounded fullmatch format constraint. Only the next authorized user turn
is delivered; the whole script is never added to the opening or public fixtures.
Full-exchange grading replays every branch, user correction, final format and
end-state artifact. Skipped turns, acting early, altered user messages or missing
final response fail closed.

`ScriptedEpisodeV8` has its own explicit protocol profile and turn-ending finish
semantics. Ordinary `PiNativeEpisode` and its original provider/irrevocable finish
are unchanged. The Pi provider queues the sealed follow-up **before** completing
the current streamed stop; a late injection after agent_settled raced print-mode
teardown and was rejected during implementation. Real three-user-turn Pi
qualification now passes. Generic canonical token/mask encoding handles all
assistant frames while user messages remain context, not supervised targets.
Conversation corrections are **fresh-only**, consistently selected by synchronous
and pooled correction paths; legacy families keep their existing fresh/splice
rule. This is the approved fail-closed alternative to unsafe user-turn splicing.

## Sealed end-state kind and safety

The additive v8 validator is self-contained and archived under its own member.
All era-2/3/4 validator bytes remain untouched. The new spec kind cannot coexist
with/reinterpret legacy terminal-receipt outcome fields. Legacy verdicts and the
same era-4 proof terminal are tested under both versions, including negative
verdicts; old archived validator bytes are byte-identical. The versioned source
closure accepts the exact old manifest set or the exact additive set, not an
open-ended archive. Registry source_archive pins are rebuilt for the staged
archive; no admission/source/policy schema is weakened.

The owner captures **actual post-episode filesystem state**, not a model echo or
read-back. A new sealed receipt kind binds task identity, exact requested path
coverage and command results. Descriptor-relative O_NOFOLLOW traversal rejects
symlink leaves/parents, nonregular or oversized files. The same private assertions
are captured and graded in proof and bank policy/teacher grading; both focused
and regression validators plus degeneracy screening must pass. Canonical encoding,
positive targets and native close are demonstrated before proof passes. A genuine
Pi transcript with its output removed is a mandatory failing negative control.

Supervisor-approved safety extension: the existing Pi transport offers cwd
isolation, **not an OS sandbox**. `bwrap` is absent on lambda01. Consequently a
finite **allowlist shell grammar** is mandatory on reference AND policy/teacher
new-family actions, before Pi gets an executable call. It allows bounded local
file-processing forms (including a narrow numeric-CSV awk DSL, line-select sed,
mkdir/rm/cp/mv/printf/read operations) and local git init/add/status/log. It rejects
arbitrary interpreters/program loaders, substitutions/expansion, raw newline
commands, absolute/traversal/private-git paths, networking, package installation,
backgrounding and unsupported tools. Assertions allow only read-only forms.
Git system/global config is disabled. Invalid policy actions are retained as
valid native failure frames with positive PG-mask evidence, but never dispatched;
real-Pi policy qualification confirmed no outside-workspace sentinel was created.

This safety ruling deliberately limits the initial terminal diet: offline file
transforms, text/code/config repair, data processing and local-git setup/indexing
are supported; arbitrary executable failing-test runners or package/environment
installation are **not authorized**. The author prompt states the exact grammar.
Such proposed tasks reject instead of silently bypassing the safety boundary.

Validator examples exercised through Pi:

```json
{"kind":"exact","path":"reports/summary.txt","content":"north:14\nsouth:22\n"}
{"kind":"absent","path":"notes/obsolete.txt"}
{"kind":"command","argv":["test","-s","reports/summary.txt"],"exit_code":0,"timeout":2}
{"kind":"regex","path":"answer.txt","pattern":"count=[0-9]+\\n"}
{"kind":"exact","path":"notices/depot.txt","content":"depot=south\n"}
```

The conversation reference first asks which depot, acknowledges north as JSON,
receives a correction to south, then reads/writes/verifies the south artifact.
The opening alone does not reveal the correction. Token is bookkeeping only;
no new-family success is a token echo.

## Seed and morph engine

Author instructions accept a bounded SEED object for invented code/config/tree/
error fragments or previously solved first-party bundles. Local intake verifies
existing admission-receipt/allowlist/overlap and task digests, TRAIN split,
matching passing local proof, author prompt and exact extracted fixture content.
Corpus: **8 admitted solved TRAIN seeds + 4 invented fragments = 12**. Immutable
private corpus SHA is
`1855368bf374849e94c2bfd6fb46b4ef36f025a5b509e04ce7819ef30568d101`.
`F9-seed-manifest.json` records provenance/hashes without copying task text into
the repo. The private corpus and original task text stay lake-local.

Morph CLI reverifies an admitted solved TRAIN seed, requests entity_rename,
chain_depth, distractor or failure_inversion, and emits a complete candidate.
Family and split cannot drift. Prompt SHA, token and fixture-tree collisions with
base/lake/tranche reject; task_identity collisions reject after construction.
There is no cached proof/admission inheritance: variants go through the SAME
schema/natural-world, vocabulary, protected/execution-panel pre-screen, real
proof, both validators, degeneracy, canonical target, full protected overlap,
operator allowlist and admission/injection gates. Sealed script/reference text
also gets the existing protected pre-screen without becoming public fixtures.
Build produces only quarantine; operational supply retains the original later
admission gates. The actual qualified rename variant received a new real-Pi
proof. A deliberately wrong reference plan was rejected without publishing any
quarantine/admission. API-authored morph yield is unmeasured.

## Rotation

Exact 60-cycle shares verified programmatically (every share, not merely sum):

| Family | Share |
|---|---:|
| first_action | 16 (20 -> 16) |
| edit / lookup / recovery / sum | 6 each, unchanged |
| chat | 6 (9 -> 6) |
| writing | 6, unchanged |
| core | 1, unchanged |
| terminal | 4 new |
| conversation | 3 new |

The two largest existing shares supply the seven new slots; no other share,
scheduler lane setting or curriculum/training arithmetic changed.

## Validation and evidence

- **45/45 staged checks passed**, including schema rejects, exact/regex/absent/
  command mechanics, symlink escape rejection, deterministic scripts, full
  exchange/finish checks, old native finish/validator/archive compatibility,
  morph re-validation/dedupe/split isolation, seeded instruction types, verified
  local seed harvest, shared fresh-only correction rule and exact rotation.
- Three passing **actual Pi** proofs: terminal, scripted conversation and renamed
  variant. Both sealed validators, native close, degeneracy and canonical targets
  passed. Complete staged quarantine/source/registry validation and actual
  protected-overlap zero-collision checks passed. No allowlist was minted and
  no task admitted. Wrong reference and unsafe policy controls failed as required.
- Legacy CPU regression: **44 passed / 5 real-Pi tests deselected**. The unrestricted
  original regression invocation produced **44 passed / 5 failed**: the legacy
  v1 provider tests encounter existing `pi_contract_mismatch` with today's Pi/tool
  schema. No legacy provider/test was altered to hide this; the frozen-v2 and
  versioned-v8 real-Pi paths above passed. This is a residual baseline issue.
- Compile, self-contained-validator regeneration equality and whitespace checks
  passed. Independent acceptance review is still required before deployment.

Commands (lambda01; no Frontier/Slurm invocation):

```bash
.venv/bin/python -m pytest -q tests/test_f9_task_diversity.py
.venv/bin/python -m pytest -q tests/test_e97_first_party_validator_protocol_breadth.py tests/test_e97_pi_native_codec.py tests/test_e97_pi_native_tool_bridge.py -k 'not real_pi'
.venv/bin/python scripts/rl-loop-v1/F9-staged/prepare.py --root /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final
source /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final/activate.sh
cd "$F9_STAGE_ROOT"
/home/erikg/emender/.venv/bin/python -m pytest -q /home/erikg/emender/scripts/rl-loop-v1/F9-staged/unit_tests.py
/home/erikg/emender/.venv/bin/python /home/erikg/emender/scripts/rl-loop-v1/F9-staged/harvest_seeds.py
```

Logs: `/home/erikg/emender-scratch/fix-fanout/F9-{tests,staged-tests,cpu-regression-tests,regression-tests}.log`.
Private retained qualification root: lake expansion `F9-staged/qualified-final`.
Versioned archive SHA:
`9c333d4cca1170bc68129b17ba3d50cdbb388c7d05220e6779dc98079329a6e9`.
Tracked staging lives in `scripts/rl-loop-v1/F9-staged`; teacher implementation
remains lake-local, represented in git by `teacher_author.patch`. Detailed
coordinated deployment/rollback boundaries are in `F9-deployment.md`. Rebuild
source authority from the committed deployed bytes before any operator restart;
qualification tranches are intentionally unadmitted, pre-commit test artifacts.

Architecture authority read: RESILIENT_DILOCO_COMPUTE_POOL and GAP_MATRIX.
Applicable safety/evidence intent: **R14/NDP13** bounded local termination,
**R16** honest exact-byte evidence. No resilient-training behavior, checkpoint,
PG arithmetic, systems scale/Slurm queue or numerical-convergence claim is made.

## Risk assessment and PG contrast watch item

New family policy pass rates are **unknown**: deterministic reference successes
are not policy measurement. Family-grouped PG needs **both passes and failures
per family**; an all-fail initial terminal/conversation group provides no useful
within-family contrast and can raise correction/SFT load, while an all-pass group
also loses contrast. Start with only terminal=4/60 and conversation=3/60, watch
per-family outcomes/skip reasons, invalid-action and script rejections, and do
not broaden shares from teacher-proof success alone. Conversations always pay
full fresh-correction cost (approved fail-closed); DeepSeek's prior latency makes
that a reasonable starting choice, not an F9 throughput measurement. If they
later dominate correction volume, splicing scripted user turns is a separate
qualified change. Grammar-only containment limits coverage and needs independent
security review; bwrap defense-in-depth remains unqualified/absent. The old v1
Pi regression failures and coordinated collector/supply deployment are explicit
remaining risks, not silently relaxed admission gates.

**Next:** independent review, then an attended operator deployment/restart and
small per-family policy pass/fail observation; keep the current daemon untouched.
