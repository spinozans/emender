# Native autonomous execution diagnostic v1

**Current result:** corrected execution v2 completed all 32 episodes. Every
model scored **0/8 task completions**. The infrastructure passed; the generated
policies did not. No checkpoint promotion or further training.

## Frozen scope and criteria

After the completed 880-update SFT run, execute **eight new authored cases per
model, 32 episodes total**: parent/y, u128/y, u880/y and u880/x. Each model uses
two GPU lanes; both variants of each pair stay on the same lane. No extra SFT,
RL, checkpoint promotion or first-party registry admission.

Four families, each with two identical user prompts but different environment
contents and required answers:

1. JSON-key lookup with environment-only 20-hex-character values.
2. Integer addition using values available only in a file.
3. Increment a JSON count into a new output file while preserving other fields.
4. Observe an actual missing-file error, consult a catalog, read its target and
   return the value. The first error must precede a later informative observation.

Success requires an explicit correct `finish`, an actual external call,
unchanged source files and, for edits, exact parsed output JSON. Files are read
through a separate immutable-code reader while **all agent processes are
paused**, not through model-provided status or a model-editable test program.
The reader joins only the agent PID namespace, not its cgroup, uses isolated
Python (`-I -S`) in the same pinned read-only image, and reads `/proc/1/root`
through bounded no-follow file descriptors. It receives paths, never gold
answers. It has no host filesystem, GPU, socket or network exposure.
Symlinks, directories, missing, oversized and invalid outputs do not pass.
Gold answers remain host-side; only fixture input files enter the sandbox.

Paired answer success is evidence of environment-conditioned behavior on these
small probes. A one-command program can legitimately solve an edit, so edit
success alone is not proof of model-level observation inspection or memory.
These authored diagnostics are not SWE-bench, independent final-holdout evidence,
or broad reliable-agent qualification. No threshold is adjusted after seeing
outputs and no training examples are emitted.

## Runtime and isolation

- Original native five-line frame, tool declarations and source argument types.
  No supplied assistant header, Pi translation or clipped observations.
- Actual qualified OpenHands 0.53.0 / ACI 0.3.1 executor, immutable image
  `sha256:036df5d48772c58190d5e4702976d7a865f5963358d503a35a36ec00a54726f7`.
- Fresh owned nonroot container per episode: no network, capabilities, GPU,
  host filesystem/socket/port mounts; read-only root, bounded tmpfs, 4GiB RAM,
  two CPUs and 128 PIDs. The Docker socket is host-controller-only.
- Private generation tokens/history and backend logs stay in private evidence;
  progress output contains only case IDs, verdicts and bounded reason labels.
- Full causal prompt replay at each turn, no context compaction or speculative
  cache splicing. Maximum context 65,536, 4,096 generated tokens/turn,
  8,192/episode, eight turns, 600-second episode generation deadline.
  Existing private reasoning cap remains 2,048 tokens / 65,536 UTF-8 bytes.
- Tool RPC deadline 45 seconds; excessive source-request timeouts are not
  silently rewritten. A deadline stops the episode rather than inventing an
  upstream observation. Container startup 120 seconds, authored smoke 900
  seconds, complete command 5,400 seconds plus 30-second kill grace.
- Checked eight-GPU lease, explicit local-rank device, NUMA placement, isolated
  Triton caches, both expandable-segment allocator variables, BF16 weights and
  disabled reduced-precision BF16 GEMM reductions. No optimizer is constructed.
- An authored real-container solver checks all eight fixture/oracle combinations
  before generated calls run. Its success is infrastructure evidence, not model
  success. Failed children are not retried (`--max-restarts=0`).

ADR-003 safety intent: R07 committed checkpoint identity, R14/NDP13 bounded
termination, R16 evidence discipline. This is read-only fixed-world local
inference, not a training-recovery qualification. Elastic/native/async
requirements and Frontier/ROCm/communicator-shrink qualification are unclaimed.

## Implementation and validation

`e97_native_execution_cases.py` owns cases and host grading;
`e97_native_execution_rpc.py` runs only inside the sandbox;
`e97_native_execution_sandbox.py` owns isolated execution and frozen snapshots;
`eval_e97_native_execution.py` freezes, runs and aggregates the panel;
`run_e97_native_execution.sh` enforces the bounded preflight/lease/launch sequence.
The numerical model implementation is unchanged from trainer commit `b8ee034f`.

Validation command:

```sh
/home/erikg/emender/.venv/bin/python -m pytest -q \
 tests/test_e97_native_execution.py \
 tests/test_e97_open_swe_native_runtime_protocol.py \
 tests/test_e97_openhands_native_execution.py
```

Initial CPU validation: **45 passed**, including causal observation delivery
before the next generated turn. The original `proc_b0bf` failed after 129
seconds in the first authored smoke case, before GPU/model evaluation:
`source_files_unchanged:false`. The failed root `native-execution-diagnostic-v1`
and original panel SHA
`7f072bb26dc052888be903ed5a649928962b85640ec983ad3eb0bb0f93f92583`
remain retained, not relabeled.

The native executor correctly read the input. Controlled diagnostic `proc_26cc`
proved Docker archive API cannot find that tmpfs file in either running or
paused state. `proc_d3f5` then proved a separate same-UID reader in the paused
agent PID namespace can read the actual bytes. Neither diagnostic loaded a
model. Roots: `native-execution-snapshot-diagnostic-v1` and `-v2`.

The corrected reader uses no-follow opens at every untrusted path component,
rejects nonregular/oversized/invalid UTF-8 files, and preserves reader stdout,
stderr, inspection and cleanup evidence. Smoke cases now retain their calls,
snapshot and oracle verdict before failing. Updated CPU validation: **46 passed**,
including symlink-parent, FIFO, missing, oversized and invalid-output rejection.
No task, oracle, model, threshold or generation budget changed.

The corrected immutable export (`d6425898`) used
`native-execution-diagnostic-v2` under
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/`. This was an inspected
infrastructure correction, not an automatic failed-child restart.

## Completed generated execution: negative capability result

`proc_a113` completed in 806 seconds, including identity checks, 46 clean-export
CPU tests, all eight authored real-container/oracle checks, model loading and
32 generated episodes. Source inventories passed before and after execution.
All 40 agent containers and 40 isolated reader containers have cleanup receipts.

| Model | Task successes | External calls | Error observations | Finish calls |
|---|---:|---:|---:|---:|
| Parent y | 0/8 | 0 | 0 | 0 |
| u128 y | 0/8 | 48 | 48 | 0 |
| u880 y | 0/8 | 64 | 64 | 0 |
| u880 x | 0/8 | 64 | 64 | 0 |

- Parent: eight invalid native openings, no external dispatch.
- u128: two first-turn generation-budget exhaustions and six eight-turn loops.
- u880 x/y: eight eight-turn loops each. All external calls were editor views;
  none reached informative fixture contents or produced a finish/output file.
- Examples of generated argument errors: `/workspace/testbed/config.json.json`
  instead of `/testbed/config.json`, repeated suffix growth, and repeated
  missing-file views. u128 sometimes selected `/testbed/catalog.json` but used
  an invalid `[1,50]` range on its one-line file. Authentic errors were delivered.
- All original fixture files remained unchanged. No paired family passed.

The earlier four matching first calls were all **repository-directory views**
under `/workspace/<repository>__1.0`. They did not establish general file-path
copying or multi-turn execution. The new final models do produce valid native
frames on all eight prompts, but format acquisition did not transfer to grounded
file access, recovery and completion here. Because no generated policy obtained
the task data, arithmetic and post-successful-read reasoning remain unmeasured;
this is not evidence that either is intrinsically impossible.

Evidence identities:

- Panel: `601ca7525ef616930e912cd45f3565e8b596a422635bb0ed118876782beea854`.
- Model summary: `9f9f8fe513e4433083c64bd1bdbd90f2bceb432550684715ba9a2ee507e00d22`.
- Authored preflight: `8e3ebb01005035d01e636fe7701ff5bbbcad3f960108c9bbffebf52b3c52165f`.
- `causal-framing-audit.json`:
  `b0c2bf673e1b1c59219463ba43dec409bdd26fea5e56aa569f79b32d09fa2989`.

The CPU causal audit reconstructed all 32 histories, checked 186 prompts
against the original dataset renderer, and verified all 176 dispatched arguments
and appended observations exactly. This rules out a recorded framing/argument
translation discrepancy in those paths, not every GPU numerical issue or a
prompt-distribution mismatch.

A reporting bug left the last frame's `reason:"valid"` on 22 eight-turn
exhaustions. These are **turn-budget failures**, not successful episodes; raw
artifacts remain untouched and grading was already false. Future reports now
emit `turn_budget`, with a regression test. Updated CPU suite: **47 passed**.

Next useful diagnostic: separate exact-path acquisition from continuation after
a supplied, authentic successful read; test prompt-style sensitivity as a
separate condition. Such assisted probes must not be relabeled as autonomous
success. Any correction training needs its own frozen data/budget/retention
policy; the completed 880-update budget is not reopened.
