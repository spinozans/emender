# E97/GDN-2 Formalism Findings

Date: 2026-05-30

## Bottom Line

Yes, the comparison can be formalized, but only at the structural/algebraic and
coarse resource-signature levels currently represented in Lean.

The sorry-free core supports this stance:

- E97 split erase/read and write gates are a clean extension of the
  delta-correcting E88/Emender update.
- E97 specializes to E88 by setting both split gates to all ones, so E97 is at
  least as general as E88 at the one-step update level.
- A concrete 1x1 witness shows E97 is not merely the all-one-gate E88
  subfamily on that fixed state/input.
- A concrete 2D split-direction witness shows E97's split erase/read direction
  can be nonparallel to its write direction in one transition factor, while an
  E88 coupled transition `mu I - p p^T` can only match such a split transition
  when the two split directions are parallel.
- GDN-2-style split erase/write shares the same split-gated linear read/write
  core as E97, but remains on the scan-compatible/no-temporal-state-nonlinearity
  side of the resource signature.
- E97 remains in the same leading per-head recurrent-state cost class as E88
  under the current Lean cost model. If precomputed split-gate application is
  counted explicitly, the model records `6 * d * d + 2 * d`, bounded by
  `7 * d * d` for `d >= 2`.

The Lean core does not prove better training efficiency, lower loss, better
wall-clock throughput, better FLOPs-per-bit convergence, optimizer stability, or
a broad task-level E97/GDN-2 expressivity win. Those remain empirical claims.

## Changed Modules And Files

The E97/GDN-2 theorem surface is in these trusted Lean modules:

- `ElmanProofs.Architectures.SplitGatedDelta`
  (`formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean`)
- `ElmanProofs.Architectures.RecurrentResourceFormalism`
  (`formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean`)
- `ElmanProofs.PaperCore`
  (`formal/lean/ElmanProofs/PaperCore.lean`)

The trust-gate batch also touched
`formal/lean/ElmanProofs/Expressivity/E88ExceedsE1HCapacity.lean`, but only to
clarify that its capacity theorem is formal state-capacity evidence, not an
empirical outcome claim.

Supporting documentation and inventory files updated by the upstream formalism
and trust-gate work:

- `formal/lean/E97_RESOURCE_COST.md`
- `formal/lean/E97_GDN2_PROOF_PLAN.md`
- `formal/lean/FORMALIZATION_GAP_AUDIT.md`
- `formal/lean/PROOF_INVENTORY.md`
- `formal/lean/TRUSTED_PROOF_SURFACE.md`
- `formal/lean/TRUST_GATE_FINDINGS.md`

## Theorem Surface

All names below are in the trusted `ElmanProofs.PaperCore` import closure.

### SplitGatedDelta Helpers

Module: `ElmanProofs.Architectures.SplitGatedDelta`

- `hadamard_onesVec_left`: for any `v : Vec n`,
  `hadamard (onesVec n) v = v`.
- `hadamard_onesVec_right`: for any `v : Vec n`,
  `hadamard v (onesVec n) = v`.

These are helper facts used to make the all-one-gate specialization exact.

### E97 Algebra

Module: `ElmanProofs.Architectures.SplitGatedDelta`

- `e97LinearCore_eq_expanded`: for all `lambda`, `H`, `k`, `b`, `w`, and `v`,
  `e97LinearCore lambda H k b w v =
  splitGatedTransition lambda k b * H + outerKV k (e97WriteValue w v)`.
- `e97UpdateDirect_eq_expanded`: for all `lambda`, `H`, `k`, `b`, `w`, and `v`,
  `e97UpdateDirect lambda H k b w v =
  e97UpdateExpanded lambda H k b w v`.
- `e97_specializes_to_e88_all_one_gates_direct`: for all `lambda`, `H`, `k`,
  and `v`, `e97UpdateDirect lambda H k (onesVec K) (onesVec V) v =
  e88DeltaUpdateDirect lambda H k v`.
- `e97_specializes_to_e88_all_one_gates_expanded`: for all `lambda`, `H`, `k`,
  and `v`, `e97UpdateExpanded lambda H k (onesVec K) (onesVec V) v =
  e88DeltaUpdateExpanded lambda H k v`.
- `e97_expresses_e88_by_specialization`: combines the direct and expanded
  all-one-gate equalities above, giving constructive inclusion of E88 in E97.
- `e97_split_gate_strict_witness_not_e88_all_one`: for arbitrary
  `lambdaE97` and `lambdaE88`, the 1x1 zero-state/unit-input E97 update with
  unit read gate and write gate `2` is not equal to the E88/all-one-gate update
  on the same state/input. Informally, E97 produces `tanh 2` where the E88
  specialization produces `tanh 1`.

### 2D Split Erase/Write Direction Witness

Module: `ElmanProofs.Architectures.SplitGatedDelta`

- `splitTransitionFromDirs`: defines the one-step transition factor
  `lambda I - writeDir eraseDir^T` in two key dimensions.
- `parallel2`: defines two-dimensional parallelism by zero determinant:
  `a 0 * b 1 = a 1 * b 0`.
- `splitGatedTransition_eq_splitTransitionFromDirs`: E97's pointwise
  erase/read gate realizes the split transition with erase/read direction
  `hadamard b k`.
- `e88_coupled_transition_forces_parallel_split_dirs`: if
  `splitTransitionFromDirs lambdaSplit writeDir eraseDir` equals an E88
  coupled transition `e88DeltaTransition lambdaE88 p`, then `writeDir` and
  `eraseDir` are parallel. This is a necessary one-step collapse condition.
- `e88_cannot_realize_nonparallel_split_transition`: any nonparallel split
  directions are outside the two-dimensional E88 coupled transition family at
  the transition-factor level.
- `splitWitnessWriteDir`, `splitWitnessEraseGate`, `splitWitnessEraseDir`, and
  `splitWitnessValue`: the finite witness uses `u = (1,1)`, `b = (1,0)`,
  `r = b*u = (1,0)`, and zero value payload to isolate the transition factor.
- `splitWitness_dirs_not_parallel`: proves the witness directions are not
  parallel.
- `e97_realizes_splitWitness_transition`: proves E97 realizes the witness
  transition through its pointwise erase gate.
- `splitWitness_transition_entries`: records the concrete transition entries:
  `I - (1,1)(1,0)^T = [[0,0],[-1,1]]`.
- `e88_cannot_realize_splitWitness_transition`: no two-dimensional E88 coupled
  transition realizes the concrete nonparallel split transition.

### GDN-2 Linear Core

Module: `ElmanProofs.Architectures.SplitGatedDelta`

- `gdn2LinearCore_eq_e97LinearCore_on_decayed_state`: for all `D`, `H`, `k`,
  `b`, `w`, and `v`,
  `gdn2LinearCore D H k b w v = e97LinearCore 1 (D * H) k b w v`.
- `gdn2LinearCore_identity_decay_eq_e97LinearCore_one`: for identity decay,
  `gdn2LinearCore (1 : Matrix (Fin K) (Fin K) Real) H k b w v =
  e97LinearCore 1 H k b w v`.
- `gdn2LinearCore_eq_expanded`: for all `D`, `H`, `k`, `b`, `w`, and `v`,
  `gdn2LinearCore D H k b w v =
  splitGatedTransition 1 k b * (D * H) + outerKV k (e97WriteValue w v)`.
- `e97_and_gdn2_share_split_gated_linear_core`: records both definitional
  equalities: E97's linear core is `splitGatedLinearCore lambda H k b w v`, and
  GDN-2's linear core is `splitGatedLinearCore 1 (D * H) k b w v`.

### Resource Signatures

Module: `ElmanProofs.Architectures.RecurrentResourceFormalism`

- `e97_split_gated_resource_signature_like_e88`: for all `layers`, `heads`,
  and `d`, E97 has matrix state, temporal nonlinearity, delta-correcting write,
  `.sequentialManyProgram` implementation mode, and `scanCompatible = false`.
- `gdn2_split_erase_write_is_scan_compatible_linear_state`: for all `layers`,
  `heads`, and `d`, GDN-2 split erase/write has matrix state, delta-style write,
  no temporal nonlinearity, `.parallelScan` implementation mode, and
  `scanCompatible = true`.

### Cost Claims

Module: `ElmanProofs.Architectures.RecurrentResourceFormalism`

- `e97_e88_same_leading_per_head_recurrent_cost`: for all `d`, `H`, and
  `depth`, `flopsPerToken (e88NDM depth H d) = e88PerHeadStateUpdateCost d`,
  `flopsPerToken (e97SplitGated depth H d) =
  e97PerHeadStateUpdateCostCoarse d`, and
  `e97PerHeadStateUpdateCostCoarse d = e88PerHeadStateUpdateCost d`.
- `e97_precomputed_split_gate_overhead_linear_in_d`: for all `d`,
  `e97PrecomputedSplitGateOverhead d = 2 * d`.
- `e97_precomputed_split_gate_overhead_bounded_by_state_scalars`: for
  `d >= 2`, `e97PrecomputedSplitGateOverhead d <= squareStateScalars d`.
- `e97_precomputed_gates_same_quadratic_cost_class_as_e88`: for `d >= 2`,
  E88 cost is `6 * squareStateScalars d`; E97 with precomputed split-gate
  application is `6 * squareStateScalars d + 2 * d`; E88 is no more expensive
  than that counted E97 expression; and the counted E97 expression is bounded
  by `7 * squareStateScalars d`.

## Formal Claims Vs Empirical Claims

Formal claims supported by Lean:

- E97 direct and expanded split-gated update forms are algebraically equal.
- All-one E97 split gates recover E88 exactly.
- A finite 1x1 split-write-gate witness leaves the all-one-gate E88 subfamily.
- A finite 2D split erase/write direction witness proves a one-step
  transition-factor separation: E97 realizes `I - u r^T` for nonparallel
  `u=(1,1)` and `r=(1,0)`, while E88's coupled transition can match a split
  transition only when the split directions are parallel.
- E97 and GDN-2 share a split-gated linear read/write core, with GDN-2 applying
  it to a pre-decayed state and not adding E97's state nonlinearity.
- The current resource signature places E97 with E88 on nonlinear,
  non-scan-compatible sequential recurrent updates and GDN-2 on the
  scan-compatible linear-state side.
- Under the current cost model, E97 and E88 share the same leading recurrent
  update cost class; explicitly applying precomputed split gates adds a
  separately counted `2 * d` term bounded by one `d * d` state-scalar pass for
  `d >= 2`.

Empirical claims not proved by Lean:

- E97/GDN-2 trains faster than E88, GDN, Mamba2, or M2RNN.
- E97/GDN-2 reaches lower loss or better BPB.
- The split erase/write gates improve FLOPs-per-bit convergence.
- The update is faster on real kernels after fusion, bandwidth, and launch
  overheads are counted.
- Gate generation from token features is cheap, stable, or useful in training.
- The 1x1 strict witness scales to a broad task-level expressivity separation.
- A broad impossibility result about all E88 behavior. The 2D split-direction
  theorem compares only one-step transition factors and does not claim E88
  cannot match a single selected output matrix when arbitrary value writes or
  other mechanisms are allowed.

## Validation Results

Copied from `formal/lean/TRUST_GATE_FINDINGS.md` for the E97/GDN-2 trust gate
on 2026-05-30:

- `cd formal/lean && lake build`: passed.
  Final line copied there: `Build completed successfully (2203 jobs).`
- `cd formal/lean && bash scripts/check_paper_core.sh`: passed.
  Copied output: `trusted check passed: 12 project source files` and
  `paper core check passed: 12 project source files, no native_decide`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh
  ElmanProofs.lean`: passed.
  Copied output: `trusted check passed: 2 project source files`.
- Independent grep over the changed trusted E97/GDN-2 files found no matches for
  banned placeholders/trust escapes:
  `sorry`, `admit`, `native_decide`, explicit `axiom`, `opaque`, or `unsafe`.

Local validation for this synthesis task:

- `cd formal/lean && lake build`: passed.
  Final line: `Build completed successfully (2203 jobs).`
  The build emitted non-fatal Lean linter warnings in proof modules; it exited
  0.
- `cd formal/lean && bash scripts/check_paper_core.sh`: passed.
  Output: `trusted check passed: 12 project source files` and
  `paper core check passed: 12 project source files, no native_decide`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh
  ElmanProofs.lean`: passed.
  Output: `trusted check passed: 2 project source files`.

Paper status: paper text was left untouched for this synthesis task. Because no
paper files were modified, `paper/build.sh` and visual paper review were not
run.

Trust-gate-review-2 validation after adding the 2D split-direction theorem:

- `cd formal/lean && lake build`: passed.
  Final line: `Build completed successfully (2195 jobs).`
  Existing non-fatal Lean linter warnings remain.
- `cd formal/lean && bash scripts/check_paper_core.sh`: passed.
  Output: `trusted check passed: 12 project source files` and
  `paper core check passed: 12 project source files, no native_decide`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh
  ElmanProofs.lean`: passed.
  Output: `trusted check passed: 2 project source files`.
- `cd formal/lean && bash scripts/check_trusted_no_placeholders.sh
  ElmanProofs/Architectures/SplitGatedDelta.lean`: passed.
  Output: `trusted check passed: 3 project source files`.
- `cd formal/lean && rg -n '\b(sorry|admit|native_decide)\b|^\s*(unsafe\s+)?axiom\b|^\s*opaque\b|^\s*unsafe\b'
  ElmanProofs/Architectures/SplitGatedDelta.lean`: no matches.

## Author-Facing Wording

If the paper needs a compact sentence later, the theorem surface supports this
limited wording:

> The Lean core treats E97 split erase/write as a strict one-step extension of
> the E88 delta-rule family: all-one split gates recover E88 exactly, a finite
> 1x1 witness leaves that subfamily, and the counted recurrent-state update
> remains in the same leading per-head cost class. This is structural evidence,
> not a proof of improved training efficiency or loss.

This wording is not applied to the paper in this task.

## Recommendation

Keep E97/GDN-2 as a structurally justified candidate and empirically test it
next; do not claim it wins on efficiency from Lean alone.

The next empirical step should be a small, controlled E97 CMA/S5 probe against
E88 and GDN/GDN-2-style baselines at matched or cohort-matched geometry. The
probe should report loss/BPB, S3/S5 or state-tracking accuracy, wall-clock,
kernel counters, and FLOPs-per-bit with gate generation and split-gate
application included. If that probe does not show a practical gain, the formal
result still supports documenting E97 as a clean extension, but not promoting it
as the racer.
