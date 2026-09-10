# E97 Open-SWE semantic audit v1

Plan date: 2026-09-09. Audits completed 2026-09-10 UTC.
Local implementation candidates, not training admission.

**Rebuild completed:** the source-native full-trajectory dataset now exists and
passed complete source/mask validation: **5,533 training trajectories / 64,409,332
targets**, plus 436 development trajectories, with zero cross-split problems.
See [the completed rebuild report](e97-open-swe-source-native-rebuild-v1.md).
Earlier sizing figures below remain historical proposals, not current inventory.

## Execution

The read-only full converted-record census completed as managed process
`proc_f87b` (`e97-open-swe-semantic-audit-v1`) in **104 seconds**. All **6 tests
passed** before measurement. No GPU or model inference was used.
The run has a two-hour wall bound, one numerical-library thread, restrictive
output permissions, and no automatic retry. Tests run before the census; any
failure stops the chain and preserves logs/partial evidence.

Frozen root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open-swe-semantic-audit-v1`.
Source/test/recipe/runner inventory SHA-256:
`c7b239c86ce4ac4c610ca2aae4c01a6fa5bbc4798a1ad8c99fb6dbe5e975cd9d`.

Input authority manifest SHA-256:
`248a02e6d977b83474eba6e48589a691e4fc36115602c931cfac4960b73dc196`.
The input's `training_eligible: false` is checked and never modified.

## Frozen scope

- Verify all declared converted streams, metadata, trajectory receipts, and
  archived converter sources before and after measurement.
- Decode and verify serialization hashes for all 13,112 records; reconcile input,
  target, trajectory, assistant-unit, and private-analysis token counts.
- Count tool calls, observed read limits, repository concentration, split exposure,
  multi-record trajectories, and supervised targets after recurrent window resets.
- For recognized source `cat -n` observations, compare the displayed line numbers
  with the converted read's requested offset/limit interval.
- Select bounded training-split examples by a frozen bottom-k hash rule: three per
  tool/window stratum and read-evidence category. No model outcomes select samples.

Published receipts: `results/identity.json`, `results/summary.json`, and
`results/samples.json`. Summary SHA-256:
`3073060e1a218a8bfdf3afae1398acc9a58784d50e0a8410d1fcfe3159335ebf`.
A completed execution is not a semantic pass or admission.

## Completed census

All declared input identities verified before and after measurement. Counts
reconciled exactly: 13,112 records, 10,905 trajectory identities, 558,858 assistant
units, 564,969,738 inputs, and 151,602,474 assistant targets.

| Measurement | Result |
|---|---:|
| Read calls | 110,809 |
| Recognized numbered observations outside requested interval | **15,902 (14.35% of all reads)** |
| Recognized numbered observations within requested interval | 91,914 |
| Recognized header but no tab-prefixed numbered lines | 45 |
| Other/unclassified observation formats | 2,948 |
| Observed read arguments with limit=200 | 53,370 (48.16% of reads) |
| Trajectories spanning multiple recurrent-reset records | **2,141 / 10,905 (19.63%)** |
| Later-window records / first supervised decisions | 2,207 |
| Targets in later windows | 13,893,989 (9.16% of all targets) |
| All targets in multi-record trajectories | 51,587,902 |
| Train-split targets / validation-split targets | 150,132,307 / 1,470,167 |

There were 2,851,049 recognized numbered lines outside their read intervals.
The line matcher is conservative: it requires a source `cat -n` header and a
number followed by a tab. A trailing blank line reduced to a bare number by
source normalization is not counted. Unknown/empty formats are not passes.

Examples selected by the frozen hash rule include a read of lines 1--200 whose
observation displays lines through 580 (then a source clipping notice), and reads
of lines 1--200 displaying lines through 352 and 273. These are concrete
converted argument/evidence inconsistencies, not just different formatting.
Other sampled source observations reject a requested range extending beyond EOF;
that error behavior also needs comparison with the destination tool.

Tool distribution: bash 358,601; read 110,809; edit 62,183; write 16,360;
final 10,905. These are repository-action traces, not a balanced curriculum of
all instruction-following situations.

**Decision:** do not expand training from the unchanged candidate. Determine
provenance and trajectory-level exposure before selecting exclusion or a faithful
rebuild. This does not establish that these mismatches caused the earlier model's
opening-token regression, nor that every window reset removed necessary evidence.

## Limits and follow-up

Line-range evidence is not filesystem replay. An in-range observation is not proof
that its contents, path, formatting, or tool behavior match the deployment runtime.
Unrecognized observations are explicitly unclassified, not counted as passes.
Window-reset counts do not themselves prove that a specific decision lost required
context. No reasoning-quality or model-capability claim follows from this census.

Deterministic examples have been inspected. A separately frozen source-bound
follow-up completed as **`proc_818c`**, `e97-open-swe-source-fidelity-v1`,
in **205 seconds**, after **9 tests passed**, against all original included
trajectories. It checked exact normalized
assistant/observation sequence identity, original read ranges, path roots,
nonempty assistant content fields, extra user messages, and literal path
availability at window starts.

Follow-up root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open-swe-source-fidelity-v1`.
Frozen source inventory SHA-256:
`9e124cb0fe0f6b826b5e3988bb564ae8583ff2bf9c4daef3a662de416a154ba9`. Any
required data correction must produce a separately versioned derivative. Do not
modify this candidate or restart training merely because the corpus is large.

## Completed raw-source reconciliation

All **10,905 normalized trajectory sequences / 558,858 assistant units** matched
the stored converted assistant/observation sequence exactly. Raw shards and
converted inputs were hash-verified before and after the audit. Summary SHA-256:
`fb084f35a680982b83cc3544c3f68472a40bb192ebaabaec28ecc78c5a0ce604`.

| Finding | Count |
|---|---:|
| Out-of-bounds reads originating from an omitted source view_range | **15,788** |
| Out-of-bounds reads originating from a source open-ended view_range | **114** |
| Out-of-bounds reads originating from finite source view_range | **0** |
| Trajectories with at least one bounds mismatch | **8,718 / 10,905 (about 80%)** |
| Targets in those trajectories | 123,704,617 |
| Trajectories with bounds mismatch or multiple records | 8,924 |
| Targets in that union | 129,427,757 |
| Source view_range-error observations | 2,733 |
| Directory views mapped to bash using the following observation | 19,734 |
| Later-window first paths absent from the retained prefix but present earlier | 247 decisions / 243 trajectories |

The source tool definition says that an absent range reads the full file and
`[start, -1]` reads to EOF, subject to source output clipping. The conversion
instead supplies a 200-line bound without transforming the observation into an
observation obtainable under that bound. This is a confirmed conversion-semantic
problem, not unexplained corruption or evidence that the original demonstrations
are unusable. Do not fix it merely by deleting observation tails: later decisions
may depend on them.

The path-availability flags are literal checks, not proof that every flagged path
was impossible to infer. Conversely, they are not a comprehensive memory-dependency
audit. There were no extra pre-finish user turns in the included pool: exactly one
per trajectory. The earlier omitted-user concern is not supported for this pool.

Other measured risks: **193,727 assistant messages have nonempty content fields**
not retained by this conversion; the meaning and redundancy of those fields still
need inspection. Applying the observation normalizer would change 667 file_text,
559 new_str, and 464 old_str payloads (excluding outer whitespace differences).
That flags potentially different displayed evidence and edit/write payloads; it
does not establish 1,690 failed edits.

Discarding every bounds-affected or multi-record trajectory would leave only
**1,981 trajectories / 22,174,717 targets across both splits**, before other
checks. This is an exclusion-envelope calculation, not an admitted clean subset.
**Prefer investigating a source-faithful rebuild over throwing away most of the
agent demonstrations or training the unchanged conversion.**

## Completed source-contract and planning-field audit

Managed process **`proc_8c73`**, `e97-open-swe-contract-audit-v1`, completed in
**133 seconds**, after **13 tests passed**. It inventoried original tool declarations, ignored shell timeout/stdin fields,
nonempty assistant content, and the `think` tool's separate `thought` argument.
The archived converter folds `reasoning_content` from think calls, but not their
`arguments.thought`. The new audit measures that omission and the hypothetical
analysis-cap effect of retaining those thoughts; it does not silently merge
public/private channels or build a training derivative.

The source shell explicitly promises persistent cwd/environment and stdin to
running processes, whereas `configs/pi/e97-core-tools.ts` starts a separate
`bash -lc` invocation per call. Source editor create is no-overwrite, whereas the
native write handler creates or overwrites. These are additional declared-contract
differences to qualify, not runtime replay results.

Frozen root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open-swe-contract-audit-v1`.
Inventory SHA-256:
`ff66f3f78a758747ae06e8eaf4ad626fc0e1bde548c2da6ccb1fea143942506f`.
Summary SHA-256:
`9ad4d0b2bf022ef0c8d9e387467e01e20b79c96b0cf39e2481396cff735cb5af`.

| Finding | Result |
|---|---:|
| Source tool specification variants | 4 (one each for execute_bash, editor, think, finish) |
| Nonempty assistant content | 193,727 messages / 6,771,853 raw tokens |
| Content neither equal to nor contained in reasoning (including longer content) | 191,324 messages / 6,730,177 tokens |
| Source think calls | 3,982 across 3,730 trajectories |
| Nonempty thought arguments | 3,978 |
| Thought arguments not contained in their reasoning_content | 3,977 |
| Raw thought-argument tokens omitted by the converter | **1,331,631** |
| Hypothetical addition of think.thought exceeds current analysis cap | 76 units / 74 trajectories |
| Bash calls with is_input=true treated as new shell commands by conversion | **12 / 11 trajectories** |
| Bash calls with explicit timeout ignored by conversion | 4,106 / 497 trajectories |
| Requested timeout above native 120-second bash bound | 1,900 calls |
| Shell session-mutation lexical candidates | 282,160 calls / 10,904 trajectories |
| Finish messages whose whitespace was collapsed | 10,901 |
| Initial executable source calls | editor 10,413; bash 492; finish 0 |

The omitted content is often related commentary or paraphrase; non-containment
is not proof of unique missing facts. The separate think.thought field, however,
is real source planning text that was not folded by the existing converter. Its
raw token count is not the same as the increase after concatenation/tokenization.
The hypothetical thought-fold count is not an authorized new serialization.

One stdin example submits `y` to a running process. Converting that to a new
`bash("y")` command is not equivalent. Shell regex hits are much weaker: most
may be explicit `cd /workspace/repo && ...` commands that deliberately restate
cwd. Do not treat all 282,160 hits as broken session-dependent steps.

This pool also contains **no immediate-final first responses**. It is not, by
itself, supervision for ordinary no-tool answers under the private-analysis
interface. General instruction examples with a consistent response interface
are required alongside agent demonstrations.

## Completed repair feasibility measurement

**`proc_b826`**, `e97-open-swe-repair-feasibility-v1`, completed in **362 seconds**,
measuring a proposed
source-native representation over the same 10,905 identities. It preserves raw
executable tool names/arguments, paths, observations, and finish-message text;
retains reasoning, nonempty commentary, and think.thought constituents; and
requires one complete trajectory per 65,536-token record. Oversize trajectories
are counted as exclusions, not truncated or reset mid-trajectory.

This is a **sizing/provenance audit only**. No training token files are emitted,
no source is admitted, and the live Pi interface is unchanged. The proposed
placement of source commentary inside internal Analysis is explicitly a visibility
change requiring protocol review. Source-native persistent shell/editor semantics
need a separate versioned runtime qualification; they are not silently added to
Pi's existing four-tool interface.

The goal is to report how much data a coherent replacement could retain, instead
of either trusting the old 151.6M count or discarding 80% of the trajectories.

Frozen root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open-swe-repair-feasibility-v1`.
Inventory SHA-256:
`6bb34f4ff3a0afd05154726d94970394403868d2e62b1306bbdcc6bbce4a651a`.
Summary SHA-256:
`8511a9be3a5918ff4a41b53bb9d0c58e150ddb6ed0053c152f7e08662dacde80`.

| Proposed envelope | Trajectories | Inputs | Assistant targets |
|---|---:|---:|---:|
| All splits | **7,232** | 350,392,455 | **87,107,899** |
| Existing train split | **7,167** | 347,303,240 | **86,333,951** |
| Existing validation split | 65 | 3,089,215 | 773,948 |

Overlapping exclusions: 3,627 whole-trajectory context overflows, 116 trajectories
with over-cap analysis, and three missing explicit final messages. These counts
are not additive. No cross-target BPE-boundary exclusions occurred. Each retained
proposal contains its complete causal history within the 65,536-token envelope.

This is enough candidate volume to justify the next engineering stage; it is
not 87.1M already-admitted targets. Source-native runtime equivalence, commentary
visibility, source/split review, and real replay still gate a training derivative.
Do not reuse its old validation split as an independent holdout without lineage
verification. The previous statement that the entire 151.6M candidate was usable
for the next run is superseded by these measured bounds.

## Admission metadata follow-up

The metadata-only follow-up completed as `proc_5fc5` (seven seconds). The existing
validation split is **not problem-disjoint**: **53/65 validation problems** occur
in training, and **63/65 validation trajectories** share training repositories.
The 7,167 proposed training trajectories cover **3,745 distinct problems**.
Declared licenses are MIT, Apache-2.0, BSD-3-Clause, and BSD-2-Clause, with no
missing metadata. Zero matches were found against the four explicitly protected
repository identities; this is not a complete overlap or licensing clearance.

See `docs/validation/e97-open-swe-admission-metadata-v1.md` for identities, counts,
and limitations. A grouped development split and declared repetition policy must
precede LR-panel freeze; regrouping cannot erase earlier-lineage exposure.

## Next execution step

The source audit is complete enough to specify a faithful rebuild; do not expand
training on the old conversion. In parallel with that design work (one attended
writer, no delegated workers), the independent optimizer prerequisite passed its
bounded local CUDA/NCCL SR gate. That synthetic test loaded no E97 checkpoint and
did not authorize a new SFT run. Subsequent no-update full-model checks isolated
MLP chunk sensitivity and qualified an opt-in FP32-CE scalar path; they did not
clear the failed chunk gate or qualify full-64K training. See
`docs/validation/e97-sr-cuda-qualification-v2.md`.
