# Native autonomous execution diagnostic v1

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
by the host through bounded Docker archives while **all sandbox processes are
paused**, not through model-provided status or a model-editable test program.
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
before the next generated turn. GPU execution and authored-container
preflight have not yet completed. Immutable exported source, panel/checkpoint
identities, private episodes, sandbox inspections/cleanup and final aggregate
will be retained under `native-execution-diagnostic-v1` in
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/`.
