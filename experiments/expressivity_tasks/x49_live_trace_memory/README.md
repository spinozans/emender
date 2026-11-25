# X49: searchable live context and private neural writes

[Canonical objective and autonomous operating mandate](CHARTER.md) · [Registered arms, dose, controls, stop rules and Lux nomination](PLAN.md).

**Data gate:** [v1 refused: 110 clean calibration documents for 128 required](DATA_STOP_V1.md). No reservation, corpus read, learner or evaluation result under v1. Host/trace qualification continues independently.

**Implementation:** Stage 0 started after public plan `45226903077c18443ad6cc06c6949a25b15159f0`. [Reference learner and credit assignment](IMPLEMENTATION.md) now cover all four arms; CPU checks below are not quality results. No X49 natural training, held-out scoring, GPU lease or Lux job. See [X48](../x48_muon_continuation/RESULT.md) for the starting point and [XSA memory system](../../../docs/XSA_MEMORY_SYSTEM.md) for the distinction between byte evidence, logical addresses and neural values.

## Stage 0a — what we have actually checked

- [`literal.py`](literal.py) builds the same exact overlapping suffix counts and latest eligible follower for A and B. Independent prefix scans include binary bytes, misses, overlaps, truncated horizons, matches more than 1,024 bytes back and fresh-window reset.
- [`values.py`](values.py) uses the existing direct trust head with archive disabled. On the CPU fixture all 256 output scores match direct at birth. After the zero-initialized readout takes an SGD fixture update, a later byte loss reaches the writer and the **correct older host row** in B. A reaches only the current host row at this seam. Rotating just the payload sends that gradient to the wrong-but-eligible donor, preserving the actual follower byte and age bin. Independent host rows isolate this derivative; they are not a trained GDN2.
- [`native_check.py`](native_check.py) compared **654** frontier/horizon answers (248 hits, 406 misses) with the real pinned XSA session, using 1,620 `locate` and 476 `byte` requests on synthetic streams. Live and partially sealed sessions agreed; all four exited successfully. This is correctness, not a natural coverage or throughput estimate. Aggregate JSON SHA-256 `ac3e0d06fa990c7d2087dace84dcc116f58c4160e9affa65cd408de866f71aa8`.

Reproduce from the repository root (first command uses no index binary):

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
python -m pytest -q experiments/expressivity_tasks/x49_live_trace_memory/test_literal.py tests/test_memory_program_direct.py
python -m experiments.expressivity_tasks.x49_live_trace_memory.native_check
```

11 tests passed (7 new, 4 direct regressions). The second command needs the registered `xsa` session executable (SHA-256 `fbc436aac93190368bbed637a5f69cca0fa5848fc81a311e60c4ebe1b4d3bf50`); no weights or natural data. The v1 split reservation subsequently refused (above); do not retry it unchanged.

## Stage 0b — coupling and privately searchable keys

The real X48 step-1024 checkpoint was SHA-verified and strict-loaded into the 149,855,670-parameter host and direct predictor on CPU; fresh host and head optimizers contained no moments. CUDA was not initialized. Separately, on a small differentiable host, **all four arms** performed 4,096-target synthetic updates and exactly reproduced their next transition after restoring weights and both optimizers. This is not a real-host 4K GPU forward, update, or price.

[`trace.py`](trace.py) implements hard, private, one-symbol filing and query choices, not imaginary scratch appended to the input. F keeps its keys fixed; C gets a gradient on its sampled filing and query logits from subsequent external-byte NLL. In a delayed fixture the correct private key found a record when the literal suffix did not. Changing only that payoff changed the score gradient on the one eligible earlier filing, not on too-young writes; the pathwise gradient reached the actual bound earlier host row. This establishes credit flow, not a learned LM gain. [`trace_native.py`](trace_native.py) independently reproduced 542 synthetic F/C frontiers with distinct observed/private XSA sessions (540 private writes, 508 native tag queries, 416 bound observed-byte reads; all four processes closed).

The complete CPU gate at local `aa6505c75` passed 20 tests plus literal and private native checks. Reproduce with `pytest -q experiments/expressivity_tasks/x49_live_trace_memory tests/test_memory_program_direct.py` and both `native_check` and `trace_native` modules under the CPU environment above. The optional `coupled` module audits the retained private host checkpoint. No X49 GPU cost is yet measured. V1 cannot train; the proposed v2 and numeric metering allocation are open decisions.

This directory is for source, tests and short reports. Use private `output/x49-live/` for local manifests, checkpoints, traces and logs. The public source snapshot does not distribute Pile, SXGC binaries, GDN2 weights, or private experiment receipts.
