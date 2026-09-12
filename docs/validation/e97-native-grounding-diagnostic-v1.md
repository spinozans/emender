# E97 native grounding diagnostic v1

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
Model measurements are pending.
