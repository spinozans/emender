# X49: searchable live context and private neural writes

[Canonical objective and autonomous operating mandate](CHARTER.md) · [Registered arms, dose, controls, stop rules and Lux nomination](PLAN.md).

**Data gate:** [v1 refused: 110 clean calibration documents for 128 required](DATA_STOP_V1.md). No reservation, corpus read, learner or evaluation result under v1. Host/trace qualification continues independently.

**Implementation:** Stage 0 started after public plan `45226903077c18443ad6cc06c6949a25b15159f0`. The first A/B causal-address and delayed-value checks passed on CPU. No X49 natural training, held-out scoring, GPU lease or Lux job. See [X48](../x48_muon_continuation/RESULT.md) for the starting point and [XSA memory system](../../../docs/XSA_MEMORY_SYSTEM.md) for the distinction between byte evidence, logical addresses and neural values.

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

11 tests passed (7 new, 4 direct regressions). The second command needs the registered `xsa` session executable (SHA-256 `fbc436aac93190368bbed637a5f69cca0fa5848fc81a311e60c4ebe1b4d3bf50`); no weights or natural data. The v1 split reservation subsequently refused (above); do not retry it unchanged. Couple A/B to the retained host, verify checkpoint replay and bound the cost before any paid panel. F/C trace learning and the Lux gates remain prospective.

This directory is for source, tests and short reports. Use private `output/x49-live/` for local manifests, checkpoints, traces and logs. The public source snapshot does not distribute Pile, SXGC binaries, GDN2 weights, or private experiment receipts.
