# Memory-program source snapshot

**2026-11-24 update:** start at the [X49 live-context memory plan](../experiments/expressivity_tasks/x49_live_trace_memory/PLAN.md). This publication carries the research tree's committed code, tests and docs through `9902a9fb1689af91325bc6e88fdbbebe22caf334`, including [X47 serving and training](../experiments/expressivity_tasks/x47_xsa_attention/README.md) and [X48's full-dose result](../experiments/expressivity_tasks/x48_muon_continuation/RESULT.md). The X48 aggregate comparison and two registered test-node lists are included; historical per-run logs, checkpoints, private trace data and corpora are not. References to original experiment commits and unshipped receipts are provenance, not promises that a source checkout contains those artifacts. The older snapshot below describes the P1 baseline.

## Earlier P1 publication

This source release contains the committed Python, shell, tests, architecture and experiment reports, and the XSA mount descriptor from the Emender research tree at `5cde505b77c6527f8c565801976e4c3d57036bb3`. Historical commit IDs in the reports identify the original measurement revisions; this publication is a separate code/docs-only commit. New run directories, checkpoints, raw inputs, scores/traces and lease logs are not part of this export. Relative receipt paths in the reports still name the retained local artifacts. Archive/native integration and receipt-verification commands need the registered inputs described there.

Start with [the instruction interface](LEARNED_MEMORY_PROGRAMS.md) and the [real-span training profile](../experiments/expressivity_tasks/x47_xsa_attention/RESULT_MEMORY_PROGRAM_PROFILE.md). The runtime and sampling tests do not need the XSA mounts or training receipts:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest -q \
  tests/test_memory_program.py tests/test_memory_program_sampling.py
```

The next engineering gate is throughput **with equivalent computations**: pack register rows and eligible operands across active documents, then replay fixed chosen programs and check requests, likelihoods, action/operand log probabilities, and gradients through later reads of earlier writes. Price the new path end-to-end on train-only natural windows before allocating a held-out comparison.
