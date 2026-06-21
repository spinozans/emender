# Frontier Kickoff Quality Pass

Task: `frontier-kickoff-quality`
Date: 2026-06-21

## Scope Checked

The initial Frontier kickoff batch includes the three requested planning tracks:

- `synthesize-frontier-handoff`: handoff and paper synthesis.
- `inventory-rocm-readiness`: ROCm/HIP readiness inventory for fused E97 and GDN2 kernels.
- `draft-frontier-execution`: WG execution-plan draft for debug smokes, extended runs, and DiLoCo.

No additional critical kickoff task was added. The requested handoff synthesis,
ROCm readiness inventory, and execution-plan draft are present, and the execution
draft is already ordered after both evidence-gathering tasks.

## Edits Made

Each downstream task was tightened with:

- A concrete artifact path:
  - `docs/FRONTIER_KICKOFF_SYNTHESIS.md`
  - `docs/FRONTIER_ROCM_READINESS_INVENTORY.md`
  - `docs/FRONTIER_EXECUTION_GRAPH_DRAFT.md`
- A file-scope note to prevent unnecessary parallel edits to the same document.
- Acceptance criteria requiring explicit deliverables, evidence separation, and
  WG log summaries.
- Fused Triton recurrence requirements that reject eager or pure-Python
  recurrence validation for prototypes, sanity checks, fallback paths, and
  preliminary experiments.
- Frontier allocation discipline: debug-queue fused-kernel evidence before
  expensive scale or extended runs.

## Dependency Check

The ordering is:

1. `frontier-kickoff-quality`
2. `synthesize-frontier-handoff` and `inventory-rocm-readiness` in parallel,
   with separate artifact paths.
3. `draft-frontier-execution` after both evidence tasks.

This keeps same-file work serialized while allowing independent evidence
collection before the execution-plan draft.

## Validation Result

- Downstream planning tasks now have concrete deliverables and acceptance
  criteria.
- Task descriptions explicitly reject eager/Python recurrence validation paths.
- The batch has clear dependency ordering and non-overlapping file scopes for
  parallel tasks.
- No missing critical kickoff task was found; the rationale is recorded above.
