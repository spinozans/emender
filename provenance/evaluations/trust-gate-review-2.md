# Evaluation: trust-gate-review-2

Task: `trust-gate-review-2`
Evaluator: `agent-543`
Date: 2026-05-30

## Grade

Overall score: 0.95 / 1.00
Confidence: 0.94
Rubric underspecified: no

The task was sufficiently specified: it required a formal Lean trust gate for
the split erase/write witness, stable theorem names/imports, placeholder and
trust-escape checks, exact modified trusted Lean file accounting, and narrow
finite one-step claim scope.

## Dimension Scores

| Dimension | Weight | Score | Rationale |
| --- | ---: | ---: | --- |
| Task-specific Lean theorem surface | 0.35 | 0.95 | `ElmanProofs.Architectures.SplitGatedDelta` now contains the requested 2D theorem surface: `splitTransitionFromDirs`, `parallel2`, `splitGatedTransition_eq_splitTransitionFromDirs`, `e88_coupled_transition_forces_parallel_split_dirs`, `e88_cannot_realize_nonparallel_split_transition`, `splitWitness_dirs_not_parallel`, `e97_realizes_splitWitness_transition`, `splitWitness_transition_entries`, and `e88_cannot_realize_splitWitness_transition`. |
| Claim scope and semantic match | 0.25 | 0.96 | The result is explicitly transition-factor-level and one-step: E97 realizes `I - u r^T` with `u=(1,1)`, `r=(1,0)`, while E88's coupled `mu I - p p^T` can match a split transition only if the directions are parallel. No broad E88 impossibility theorem is claimed. |
| Proof hygiene and trusted-source safety | 0.20 | 1.00 | `lake build`, `check_paper_core.sh`, the trusted root check, the modified-file trusted check, and direct banned-pattern grep all passed. No `sorry`, `admit`, explicit `axiom`, `opaque`, `unsafe`, or `native_decide` was introduced. |
| Documentation and downstream handoff | 0.15 | 0.93 | `TRUST_GATE_FINDINGS.md`, `TRUSTED_PROOF_SURFACE.md`, `E97_GDN2_FORMALISM_FINDINGS.md`, `PROOF_INVENTORY.md`, and the split-witness blueprint document the final theorem surface and non-results for `synthesize-split-erase`. |
| WG delivery hygiene | 0.05 | 0.90 | Work was tracked through WG logs and artifacts. Existing project linter warnings remain, but they are non-fatal and not introduced as trust escapes. |

Weighted total:

`0.35*0.95 + 0.25*0.96 + 0.20*1.00 + 0.15*0.93 + 0.05*0.90 = 0.957`, reported conservatively as `0.95`.

## Modified Trusted Lean Files Reviewed

- `formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean`

No other trusted Lean source file was modified.

## Final Theorem Surface

All names below are in `SplitGatedDelta` and are in the trusted
`ElmanProofs.PaperCore` import closure because `PaperCore.lean` already imports
`ElmanProofs.Architectures.SplitGatedDelta`.

- `splitTransitionFromDirs`: defines the 2D transition factor
  `lambda I - writeDir eraseDir^T`.
- `parallel2`: defines 2D parallelism by zero determinant.
- `splitGatedTransition_eq_splitTransitionFromDirs`: E97's pointwise erase gate
  realizes the split transition with erase direction `hadamard b k`.
- `e88_coupled_transition_forces_parallel_split_dirs`: if the split transition
  equals an E88 coupled transition, the split write and erase/read directions
  must be parallel.
- `e88_cannot_realize_nonparallel_split_transition`: nonparallel split
  directions cannot be realized by a two-dimensional E88 coupled transition.
- `splitWitness_dirs_not_parallel`: proves `u=(1,1)` and `r=(1,0)` are not
  parallel.
- `e97_realizes_splitWitness_transition`: proves E97 realizes the concrete
  witness transition.
- `splitWitness_transition_entries`: records the concrete entries
  `[[0,0],[-1,1]]`.
- `e88_cannot_realize_splitWitness_transition`: proves no two-dimensional E88
  coupled transition realizes the concrete witness transition.

## Non-Results

The repaired theorem surface does not prove:

- E97 has broad semantic capacity beyond E88.
- E88 cannot match a single selected output matrix when arbitrary value writes
  or other mechanisms are unconstrained.
- A multi-step or trajectory-level separation.
- Better training speed, loss, BPB, kernel throughput, optimizer stability, or
  FLOPs-per-bit convergence for E97 or GDN-2.

## Validation

- `cd formal/lean && lake build`: passed. Final line:
  `Build completed successfully (2195 jobs).`
- `cd formal/lean && bash scripts/check_paper_core.sh`: passed. Output:
  `trusted check passed: 12 project source files` and
  `paper core check passed: 12 project source files, no native_decide`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh ElmanProofs.lean`: passed. Output:
  `trusted check passed: 2 project source files`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh ElmanProofs/Architectures/SplitGatedDelta.lean`: passed. Output:
  `trusted check passed: 3 project source files`.
- `cd formal/lean && rg -n '\b(sorry|admit|native_decide)\b|^\s*(unsafe\s+)?axiom\b|^\s*opaque\b|^\s*unsafe\b' ElmanProofs/Architectures/SplitGatedDelta.lean`: no matches.

## Assessment

The inherited `lean-proof-e97` output did not include the requested 2D theorem
surface, so this trust gate repaired the Lean implementation rather than merely
reviewing prose. The resulting claim is narrow, stable, and checked in the
trusted import closure. The calibrated score is high but not perfect because
the theorem remains the intentionally finite transition-factor witness, not a
stronger update-family or trajectory theorem.
