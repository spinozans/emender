# Quality Pass: Split Erase Witness Batch

Task: `quality-pass-split`
Date: 2026-05-30

Scope: WG graph/task-quality pass only. No source or Lean files were inspected.

## Reviewed Downstream Tasks

- `formal-design-split`: design task for the finite one-step 2D witness.
- `lean-proof-e97`: Lean implementation task for the E97 witness and E88 collapse result.
- `trust-gate-review-2`: formal trust-gate review/repair task.
- `synthesize-split-erase`: synthesis task for proved scope and empirical probe planning.

## Quality Edits Applied

- Clarified that `formal-design-split` may name target theorem statements but must not present them as proved before Lean implementation and trust-gate review.
- Strengthened `lean-proof-e97` validation to require trusted-source checks for every modified trusted Lean file, not only `ElmanProofs.lean` when additional files are touched.
- Strengthened `trust-gate-review-2` validation to record the modified trusted Lean file list, independently run trusted checks, and document any narrowed theorem statements or non-results for synthesis.
- Set `synthesize-split-erase` to graph context and clarified that synthesis must distinguish design targets, proved theorems, trust-gate findings, and non-results.

## Validation Result

- Downstream tasks now clearly separate design, Lean implementation, trust-gate review, and synthesis.
- Code/proof tasks include concrete `cd formal/lean && lake build`, `check_paper_core.sh`, and no-placeholder/trusted-source validation gates.
- Claims are scoped to finite one-step witness/separation unless a stronger theorem is actually proved and trust-gate verified.
- The graph remains sequential for tasks that may touch the same Lean/formal files: design -> Lean proof -> trust-gate review -> synthesis.
