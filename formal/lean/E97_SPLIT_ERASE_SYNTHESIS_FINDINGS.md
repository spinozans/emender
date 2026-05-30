# E97 Split Erase Witness Synthesis Findings

Date: 2026-05-30

Task: `synthesize-split-erase`

## Bottom Line

The trusted Lean surface now supports a narrow but useful claim:

- E97's pointwise split erase/read gate can realize a finite 2D one-step
  transition factor with different write and erase/read directions.
- In the same 2D transition-factor setting, E88's coupled transition family can
  match such a split factor only when those directions collapse to parallel
  directions.
- The concrete witness uses write direction `u = (1,1)`, erase/read gate
  `b = (1,0)`, erase/read direction `r = b*u = (1,0)`, `lambda = 1`, and zero
  value payload. The transition factor is `I - u r^T = [[0,0],[-1,1]]`.

This is a one-step representational/factorization separation. It is not a
learning theorem, a task-level impossibility theorem, or evidence by itself that
trained E97 or GDN-2 variants will outperform E88.

## Proven Trusted Surface

All theorem names below are in
`ElmanProofs.Architectures.SplitGatedDelta`, imported by the trusted
`ElmanProofs.PaperCore` root.

E97 realization and witness facts:

- `splitGatedTransition_eq_splitTransitionFromDirs`: E97's pointwise
  erase/read gate realizes the explicit split transition
  `lambda I - writeDir eraseDir^T` with `eraseDir = hadamard b k`.
- `e97_realizes_splitWitness_transition`: the concrete witness gates realize
  `splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir`.
- `splitWitness_dirs_not_parallel`: the concrete directions `u=(1,1)` and
  `r=(1,0)` are not parallel.
- `splitWitness_transition_entries`: the concrete transition entries are
  `[[0,0],[-1,1]]`.

E88 limitation facts:

- `e88_coupled_transition_forces_parallel_split_dirs`: if a 2D split transition
  equals an E88 coupled transition `e88DeltaTransition lambdaE88 p`, then the
  split write and erase/read directions are parallel.
- `e88_cannot_realize_nonparallel_split_transition`: nonparallel split
  directions cannot be realized by any 2D E88 coupled transition factor.
- `e88_cannot_realize_splitWitness_transition`: no 2D E88 coupled transition
  realizes the concrete nonparallel witness transition.

Related E97/E88/GDN-2 context already in the trusted surface:

- `e97LinearCore_eq_expanded` and `e97UpdateDirect_eq_expanded`: E97 direct and
  expanded split-gated forms agree.
- `e97_specializes_to_e88_all_one_gates_direct`,
  `e97_specializes_to_e88_all_one_gates_expanded`, and
  `e97_expresses_e88_by_specialization`: all-one E97 split gates recover E88.
- `e97_split_gate_strict_witness_not_e88_all_one`: a separate 1x1 fixed
  state/input witness leaves the all-one-gate E88 subfamily.
- `gdn2LinearCore_eq_e97LinearCore_on_decayed_state`,
  `gdn2LinearCore_identity_decay_eq_e97LinearCore_one`,
  `gdn2LinearCore_eq_expanded`, and
  `e97_and_gdn2_share_split_gated_linear_core`: GDN-2 shares the split-gated
  linear read/write core, applied to a pre-decayed state.

## Design Targets vs Proved Results

The design blueprint
`formal/lean/E97_SPLIT_ERASE_RIGHT_WITNESS_BLUEPRINT.md` targeted a finite 2D
factorization witness. `trust-gate-review-2` verified that the target names were
implemented in Lean and imported by the trusted paper core.

What moved from design target to proved result:

- The explicit split transition `splitTransitionFromDirs`.
- The 2D parallel-collapse lemma for E88 coupled transitions.
- The nonparallel concrete witness and the E88 non-realization theorem for that
  witness.

What remains only design guidance or interpretation:

- Any empirical claim that learned E97 split gates improve optimization,
  sample efficiency, wall-clock throughput, or loss.
- Any broad E88 impossibility theorem beyond the 2D, one-step,
  transition-factor comparison.
- Any claim that E88 cannot match one selected output matrix if arbitrary value
  writes or other unconstrained mechanisms can absorb the transition mismatch.

## Trust-Gate Findings

`trust-gate-review-2` passed after repairing and verifying the Lean surface. The
review artifact is `formal/lean/TRUST_GATE_FINDINGS.md`; the evaluator record is
`provenance/evaluations/trust-gate-review-2.md`.

Validation results recorded by the trust gate:

- `cd formal/lean && lake build`: passed, final line
  `Build completed successfully (2195 jobs)`.
- `cd formal/lean && bash scripts/check_paper_core.sh`: passed with
  `trusted check passed: 12 project source files` and
  `paper core check passed: 12 project source files, no native_decide`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh
  ElmanProofs.lean`: passed with `trusted check passed: 2 project source files`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh
  ElmanProofs/Architectures/SplitGatedDelta.lean`: passed with
  `trusted check passed: 3 project source files`.
- Direct banned-pattern search over
  `ElmanProofs/Architectures/SplitGatedDelta.lean` found no `sorry`, `admit`,
  `native_decide`, explicit `axiom`, `opaque`, or `unsafe`.

The trust gate explicitly scoped the result as finite, one-step, and
transition-factor-level.

## Non-Results

The current theorem surface does not prove:

- that E97 or GDN-2 trains faster than E88;
- that E97 or GDN-2 reaches lower loss, lower BPB, or better downstream
  accuracy;
- that split erase/write gates improve optimizer stability;
- that split gates improve fused-kernel wall-clock throughput or
  FLOPs-per-bit convergence;
- that the finite 2D witness scales to a broad task-level expressivity
  separation;
- that E88 cannot match a single chosen output if value writes, extra steps, or
  other mechanisms are unconstrained;
- any multi-step or trajectory-level separation between E97 and E88.

## Empirical Probe Plan

The witness suggests a targeted empirical probe, not a broad benchmark.

Probe name: 2D asymmetric split-transition fitting.

Task:

- Sample bounded random 2D states `H` from a fixed distribution.
- Fix the target transition
  `T = I - (1,1)(1,0)^T = [[0,0],[-1,1]]`.
- Train small one-head cells to predict `tanh(T H)` from `H` in one recurrent
  step with the additive value payload held at zero or otherwise controlled.
- Use held-out random `H` values so an additive write term cannot memorize one
  selected output.

Models and baselines:

- `E97-split`: one E97-style split erase/write cell with trainable or supplied
  `k`, erase gate `b`, write gate `w`, and zero value payload.
- `E88-coupled`: matched one-step E88 cell with transition
  `mu I - p p^T` and zero value payload.
- `E88-extra-resource` controls: E88 with two steps, two heads, or an
  unconstrained value write, to check whether additional resources close the
  gap and to avoid overstating the one-step theorem.
- `GDN-2-split-linear`: the same split gate without E97's temporal state
  nonlinearity, to isolate factorization from nonlinear-state effects.
- `FreeLinear2x2` oracle: an unconstrained 2x2 transition matrix, expected to
  fit the target and calibrate optimizer health.

Expected measurable outcomes if the formal factorization matters in practice:

- E97-split and `FreeLinear2x2` reach near-zero held-out transition MSE.
- E88-coupled plateaus above the E97-split error floor under the one-step,
  zero-value-payload constraint.
- The learned E97 directions have nonzero determinant
  `u0*r1 - u1*r0`, while E88's fitted coupled transition remains symmetric in
  the off-diagonal factor.
- E97-split reaches a fixed MSE threshold in fewer steps and with a higher
  seed success rate than E88-coupled.
- Extra-resource E88 controls may close the gap; if they do, the result should
  be reported as support for the exact one-step factorization reading, not as a
  broad E97-over-E88 claim.

Reusable code candidates:

- `experiments/expressivity_tasks/transition_witness.py` is an adjacent
  numerical witness harness for one-step transition separations, but it targets
  NDM/GDN/M2RNN rather than the E97 split erase/read factor.
- `ndm/triton/e88_triton_forward.py` already exposes `erase_gate` and
  `value_write_gate` in the PyTorch reference and Triton path, which is a useful
  scaffold for an E97-style split-edit probe.

No experiment was run for this synthesis task. Extending either script into the
probe above should be a small follow-up task with its own validation criteria,
including a zero-value-payload control and a held-out-state test.

## Paper Update Recommendation

Recommendation: later, not in this task.

Rationale:

- The formal theorem surface is stable enough to cite, and the trust gate
  passed.
- The current task was a synthesis and probe-planning task, not a paper edit.
- The paper's formal section is already dense and also contains an outdated
  source-file count in the Lean trust paragraph, so even a small theorem-set
  insertion should be made in a focused paper-edit pass with a fresh
  `paper/build.sh` run.

Concrete follow-up edit:

- In `paper/main.typ`, add a short theorem-set paragraph in Section 7 after
  "Theorem set C: update-family separation" and before the "Theorem set C'"
  multi-step subsection.
- State that `SplitGatedDelta.e97_realizes_splitWitness_transition`,
  `SplitGatedDelta.e88_cannot_realize_splitWitness_transition`, and
  `SplitGatedDelta.e88_coupled_transition_forces_parallel_split_dirs` prove a
  finite 2D one-step transition-factor separation: E97 realizes
  `I - (1,1)(1,0)^T`, while E88's coupled `mu I - p p^T` can match a split
  transition only when the split directions are parallel.
- Add one explicit sentence that this is not a training, throughput, BPB, or
  task-level theorem; it motivates the 2D asymmetric split-transition empirical
  probe described here.

No paper file was edited by this task.
