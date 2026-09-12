# E97 native grounding diagnostic v1

**Completed for u880 before correction:** some exact paths were copied, but all
16 unassisted and all 16 supplied-read episodes failed to finish. The deficit
was not only first-path acquisition. A later separately authorized
[32-update correction](e97-grounding-correction-v1.md) improved autonomous
completion to 7/16 in both x/y, but missed its fresh-value gate; no promotion.

## Question and frozen design

The completed native execution diagnostic scored 0/8 for every model. All 128
final x/y calls returned errors: fabricated paths and failure to recover. That
proves failure to use the supplied paths in those conditions, not a universal
inability to copy strings, nor inability to reason after a successful read.

This bounded follow-up changes no weights, training data or task answers:

- **24 first-turn path probes**: six paths under each of four settings (u880 y/x
  crossed with minimal/source system messages). Paths cover `/testbed`,
  `/workspace`, files and a source-shaped repository directory. The user asks
  for a specific editor view. Score exact `path` separately from the complete
  requested call. These first-turn probes do not dispatch tools.
- **32 multi-turn episodes**: the same paired lookup and sum fixtures from the
  preceding execution panel, under four settings, each both unassisted and
  with one authored successful full-file editor read supplied first. No hidden
  answer is added: the original executor reads the actual input file and its
  authentic observation enters the causal history.
- Source system: verbatim original system **message**, including its `think`
  flag, from the already frozen development example `native-development:1822`.
  It contains general OpenHands workflow instructions, not a task answer or
  task-specific repository path. This contrasts system-message bundles, not
  content-only effects separately from the metadata flag. User prompts, tools,
  file values, tokenizer, generation policy and numerical model code stay fixed.
- Both variants of a fixture stay on the same GPU lane. The final y/x checkpoint
  SHA remains `6b529dd37a1237b13728e8901fe1469b18aa2a0a2c5e4282fc01064cdd4233b1`.

**Assisted outcomes must never count as autonomous successes.** Supplied calls
are separately recorded with authored origin, excluded from generated-call
counts, and checked against exactly one complete fixture view. `assisted:true`
forces `autonomous_success:false`, even if the correct answer is returned.
The original autonomous runner rejects panels containing interventions.

## Interpretation declared before measurement

- Correct paths under source-style but not minimal system: prompt-bundle
  sensitivity, not demonstrated universal inability to copy.
- Incorrect paths even in explicit path probes: a stronger path-grounding
  failure within this protocol; still not a universal string-copy theorem.
- Correct paired answers only after supplied reads: acquisition is blocking
  usable continuation on those cases. Report the assistance, not task success.
- Failure after valid supplied observations: the deficit extends beyond first
  file access. Distinguish wrong answer, unnecessary call loops, format errors
  and failure to finish; do not attribute all failures to arithmetic or memory.

No post-result path normalization, retries, threshold changes, checkpoint
promotion or extra training. Existing negative results stay negative. This is
an intentionally diagnostic, reused panel, not an independent holdout or
repository benchmark.

## Safety and evidence

Reuse the qualified native executor, paused-agent isolated reader, no-follow
file snapshots and host-only oracles from execution v2. Same nonroot/no-network/
no-host-mount/no-GPU tool sandboxes and ownership-checked cleanup. No reasoning
text in public progress; private token/history artifacts remain private.

Same 4,096 tokens/turn, 8,192 generated tokens/episode, eight generated turns,
600-second generation deadline/episode, 65,536 context cap and 45-second tool
RPC deadline. One authored read is explicitly additional in assisted episodes.
Whole command bounded to 5,400 seconds plus 30-second kill grace; eight-GPU
checked lease, local-rank device, NUMA placement, isolated caches, BF16 weights,
both expandable-segment allocator settings and zero torchrun restarts.

ADR-003 safety intent: R07 immutable checkpoint identity, R14/NDP13 bounded
termination, R16 evidence discipline. No elastic/native-data-plane/async,
training-recovery, Frontier or communicator-shrink conformance claim.

Implementation: `scripts/eval_e97_native_grounding.py`,
`scripts/run_e97_native_grounding.sh`, and narrowly scoped intervention support
in `scripts/eval_e97_native_execution.py`. Source, system provenance, examples,
interventions, generation policy and model identities freeze before inference.

Validation: **50 CPU tests passed**, including observation delivery before
generation, assisted immediate finish without a generated tool call, no
assisted-to-autonomous credit, turn-budget attribution, and distinguishing a
copied path from an otherwise incorrect call. Prior actual executor/reader
qualification is bound by the unchanged execution-v2 preflight hash.

Execution root: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-grounding-diagnostic-v1`.
## Completed measurements

`proc_9633` completed in 655 seconds from immutable source `1b1403cd`, including
50 clean-export CPU tests and all 56 probes/episodes. Source inventories passed
before and after execution. All 32 agent and 32 reader containers have cleanup
receipts. The original execution and SFT evidence remain unchanged.

| Setting | Exact path field | Exact requested call | Unassisted completions | Supplied-read completions |
|---|---:|---:|---:|---:|
| u880 y, minimal system | 2/6 | 1/6 | 0/4 | 0/4 |
| u880 y, source system | 2/6 | 2/6 | 0/4 | 0/4 |
| u880 x, minimal system | 1/6 | 0/6 | 0/4 | 0/4 |
| u880 x, source system | 2/6 | 2/6 | 0/4 | 0/4 |

All correct path copies were `/workspace/...` file paths. No setting correctly
copied a `/testbed` target in the explicit path probes. Four probes exhausted
the first-turn generation budget; the other 20 produced valid native frames.
Source-style prompting modestly improved complete-call agreement but did not
rescue task completion. This rejects an absolute claim that the model cannot
copy any path; it does not establish dependable general copying.

Every assisted episode received exactly one authentic successful full-file
observation containing its actual input JSON. All 16 still failed to finish:
nine exhausted eight generated turns and seven exhausted a turn's generation
budget. The 16 unassisted episodes all exhausted eight turns. There were **zero
finish calls**, so these results do not distinguish arithmetic incompetence
from a failure to choose and produce an answer.

After the supplied reads, the models made 87 further external calls; unassisted
episodes made 128. All 215 failed: 199 editor error observations and 16 shell
calls with exit code 1. The shell failures are not identified by an `ERROR:`
text prefix, so prefix-only counting would incorrectly call them successful.
For example, x/minimal attempted repository exploration under the nonexistent
`/workspace/testbed` even after the real numbers-file observation was supplied.

### Interpretation and next decision

The failure extends beyond accessing the first file: the supplied-read
intervention did not restore observation-to-answer behavior. Neither changing
the system-message bundle nor providing the correct initial read was sufficient.
The evidence supports a serious instruction-following/grounding/termination
failure in this native policy, **not** a proven architectural impossibility or
a clean isolated arithmetic/memory diagnosis.

Do not respond by simply extending the same SFT run. A subsequent correction
experiment should directly supervise verified short read/observe/answer/finish
and error-recovery behavior, with decision-level measurements, disjoint probe
values and retention gates before any scale-up. That is a recommendation, not
an automatically launched training budget or a claim that corrections will work.

### Identities

- Panel: `1bed47be493519d5e6e5993a1c0fd8633001be8a1bbba2ffd617030cbbf727ff`.
- Summary: `c3b3a1337c3f7bce03a9e098ada788ae21190f5704c0d7c8c2609cb5390c8069`.
- `supplied-read-audit.json`:
  `2671de71fdbabc5fc9db0f2764d4df20cd810141cae34ca41f97395e8e354e75`.
  This checks all 16 authentic input payloads and authored/assisted attribution;
  it is a delivery audit, not a capability pass.

No generated private reasoning is reproduced in this report.
