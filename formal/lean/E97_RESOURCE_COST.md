# E97 Resource-Cost Comparison

This note records the trusted Lean surface for comparing E97 split-gated delta
updates against E88/Emender and GDN-2-style split erase/write. It is deliberately
limited to algebraic inclusion, coarse recurrent-state cost, and resource
signature properties.

## Lean Theorem Surface

All names below are in the trusted `ElmanProofs.PaperCore` import closure.

### Same Coarse Cost Class

Module: `ElmanProofs.Architectures.RecurrentResourceFormalism`

- `e97_e88_same_leading_per_head_recurrent_cost`
  - Under the existing `flopsPerToken` / state-scalar model, matched square
    E97 and E88 heads both cost `6 * (d * d)` per recurrent-state update.
- `e97_precomputed_split_gate_overhead_linear_in_d`
  - Applying precomputed split gates costs exactly `2 * d`: one key-axis
    product and one value-axis product.
- `e97_precomputed_split_gate_overhead_bounded_by_state_scalars`
  - For `d >= 2`, the `2 * d` gate-application term is bounded by one
    `d * d` state-scalar pass.
- `e97_precomputed_gates_same_quadratic_cost_class_as_e88`
  - With precomputed gate application counted explicitly, E97 is bounded by
    `7 * (d * d)` while E88 is `6 * (d * d)`, so both remain in the same
    quadratic per-head recurrent-state cost class.

### Generality and Strict Witness

Module: `ElmanProofs.Architectures.SplitGatedDelta`

- `e97_specializes_to_e88_all_one_gates_direct`
  - Direct E97 specializes to direct E88 with all-one split gates.
- `e97_specializes_to_e88_all_one_gates_expanded`
  - Expanded E97 specializes to expanded E88 with all-one split gates.
- `e97_expresses_e88_by_specialization`
  - E97 weakly generalizes E88 by constructive all-one-gate specialization.
- `e97_split_gate_strict_witness_not_e88_all_one`
  - A 1x1 zero-state/unit-input witness with write gate `2` produces `tanh 2`,
    while every E88/all-one-gate setting on the same state/input produces
    `tanh 1`; injectivity of `tanh` proves these differ.

The strict witness is intentionally finite and narrow. It proves that the E97
update family is not only the all-one-gate E88 subfamily on that concrete
state/input. It does not prove a broad task-level expressivity separation.

### Resource Signatures

Module: `ElmanProofs.Architectures.RecurrentResourceFormalism`

- `e97_split_gated_resource_signature_like_e88`
  - E97 has matrix state, full-state temporal nonlinearity, delta-correcting
    writes, sequential-many-program execution, and no scan-compatible recurrent
    state path.
- `gdn2_split_erase_write_is_scan_compatible_linear_state`
  - GDN-2-style split erase/write has matrix state and delta-style writes, but
    no temporal state nonlinearity and a scan-compatible/parallel-scan resource
    signature.

Module: `ElmanProofs.Architectures.SplitGatedDelta`

- `e97_and_gdn2_share_split_gated_linear_core`
  - E97 and GDN-2 share the same split-gated linear read/write core, with GDN-2
    applying it to a pre-decayed state.

## Scope Limits

The formal surface supports the following answer:

- Same cost class: yes, under the coarse per-head recurrent-state
  `flopsPerToken` model; with precomputed gate application counted separately,
  E97 remains quadratically bounded by a conservative `7 * d * d`.
- More general: yes, weakly by all-one-gate specialization, and with a concrete
  strict 1x1 split-gate witness.
- GDN-2 comparison: GDN-2 remains scan-compatible/no-temporal-state-nonlinearity
  in the resource signature, while E97 remains nonlinear-state and
  non-scan-compatible like E88/Emender.
- Empirical efficiency: unresolved.

The Lean theorems do not count gate generation from token features, kernel
fusion choices, memory bandwidth, optimizer effects, training stability,
hardware throughput, empirical learning efficiency, empirical superiority, or
FLOPs-per-bit convergence.
