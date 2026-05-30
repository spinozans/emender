# Evaluation: lean-proof-e97

Task: `lean-proof-e97`
Evaluator: `agent-540`
Date: 2026-05-30

## Grade

Overall score: 0.05 / 1.00
Confidence: 0.96
Rubric underspecified: no

The task rubric is sufficiently specific: it asks for a focused Lean
formalization of the 2D E97 split erase/right-direction witness, an E88
impossibility/collapse theorem, trusted theorem-surface integration, scoped
documentation, and concrete validation commands.

## Dimension Scores

| Dimension | Weight | Score | Rationale |
| --- | ---: | ---: | --- |
| Task-specific Lean theorem surface | 0.45 | 0.00 | No implementation commit exists on this branch. The target design names `splitTransitionFromDirs`, `parallel2`, `splitWitnessWriteDir`, `e88_coupled_transition_forces_parallel_split_dirs`, and `e88_cannot_realize_splitWitness_transition` appear only in `formal/lean/E97_SPLIT_ERASE_RIGHT_WITNESS_BLUEPRINT.md`, not in trusted Lean source. |
| E97/E88 semantic match and scope control | 0.20 | 0.05 | The inherited repo already contains earlier E97 all-one specialization and a 1x1 strict witness, but it does not implement the requested 2D nonparallel erase/right-direction separation or collapse lemma. No new overclaim was introduced because no new implementation was introduced. |
| Integration and explanatory artifacts | 0.15 | 0.00 | `PaperCore.lean`, `TRUSTED_PROOF_SURFACE.md`, `E97_GDN2_FORMALISM_FINDINGS.md`, and theorem inventory files were not updated for the requested theorem surface. No final changed trusted Lean file list or naming note was produced by the actor work. |
| Validation and proof hygiene | 0.15 | 0.25 | The current baseline branch passes `cd formal/lean && lake build`, `bash scripts/check_paper_core.sh`, and `bash scripts/check_trusted_no_placeholders.sh ElmanProofs.lean`, but these checks validate inherited source rather than the missing task implementation. No task-specific trusted files were modified. |
| WG delivery hygiene | 0.05 | 0.00 | The worktree had zero commits ahead of `main` before this evaluation artifact; no actor commit or pushed implementation was available to evaluate. |

Weighted total: `0.45*0.00 + 0.20*0.05 + 0.15*0.00 + 0.15*0.25 + 0.05*0.00 = 0.0475`, rounded to `0.05`.

## Evidence

- `git log --oneline main..HEAD` returned no implementation commits before this evaluation artifact.
- `git diff --stat main...HEAD` returned no formal/source delta before this evaluation artifact.
- `rg "splitTransitionFromDirs|parallel2|splitWitness|e88_coupled_transition_forces_parallel_split_dirs|e88_cannot_realize_splitWitness_transition" formal/lean` found the requested names only in `formal/lean/E97_SPLIT_ERASE_RIGHT_WITNESS_BLUEPRINT.md`.
- `formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean` still ends its E97 theorem surface at the existing 1x1 strict split-gate witness and GDN-2 shared-core theorems.
- `wg show lean-proof-e97` reported `Commits ahead: 0` before this evaluation artifact.

## Validation Run By Evaluator

- `cd formal/lean && lake build`: passed, with existing linter warnings.
- `cd formal/lean && bash scripts/check_paper_core.sh`: passed, `12 project source files, no native_decide`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh ElmanProofs.lean`: passed, `2 project source files`.

These validation results show the inherited formal core is buildable. They do
not satisfy the task's expected outputs because the requested one-step 2D split
erase/right-direction witness and E88 impossibility/collapse theorem were not
implemented.

## Final Assessment

This submission effectively did not attempt the implementation task. Existing
prior E97/GDN-2 formalism remains healthy, but the focused theorem surface from
`formal-design-split` is absent from trusted Lean code. The calibrated grade is
therefore near zero, with only minimal credit for preserving a buildable
baseline and not introducing new trust violations.
