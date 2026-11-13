# Emender operator algebra — status and citation index

The compact result is **conditional on a fixed decision and numerical-gate
schedule**. The arbitrary-operator result uses a faithful free-vector-space
embedding and makes no compactness claim. There are no unfinished proofs in
this addition. The historical nonlinear E88/NDM `emender` signature is retained;
the X30 slot store has its own semantic signature.

## Validation and reproducibility

- Baseline verified at `5cdb2f12668802096e11b2a040c30f3169e0c55e`:
  `lake build` exited 0, `Build completed successfully (2198 jobs).`
  The taxonomy and its definitional scan labels, the P1 gap, and all five
  specified prior composition/scan proof idioms were read and checked.
- Delivered Lean root: `lake build` exits 0,
  **`Build completed successfully (2207 jobs).`**
  Exact output: `build_logs/operator_algebra_lake_build.log`.
  The new proof modules have no warnings; replayed warnings in older modules
  predate this work.
- `scripts/check_paper_core.sh` checks the entire trusted project import
  closure for proof placeholders, explicit axiom/opaque declarations, and
  `native_decide`: **passed, 25 project source files**.
  Evidence: `build_logs/operator_algebra_trust.log`.
- `lake env lean scripts/audit_operator_algebra.lean` audits **all declarations**
  in the eight new modules, rejecting axiom declarations and any transitive
  dependency other than `propext`, `Classical.choice`, and `Quot.sound`.
  It produces `OPERATOR_ALGEBRA_AUDIT.json`, the exact theorem/file/line/axiom
  inventory: **passed, 626 declarations and 92 source theorems**.
  Evidence: `build_logs/operator_algebra_axioms.log`.
- `typst compile EMENDER_OPERATOR_ALGEBRA.typ EMENDER_OPERATOR_ALGEBRA.pdf`
  exits 0 with Typst 0.14.2. The PDF has **9 pages**. All nine rendered pages
  were inspected; mathematical subscripts and theorem-box wrapping were checked.
- `git diff --check` passes. `AGENTS.md` and `CLAUDE.md` remain unchanged and
  have matching SHA-256 hashes. Unrelated existing worktree changes are excluded.

The final delivery reports the exact commit and push result. The retained
validation manifest pins the proof source files independently of Git metadata.

```text
SORRY_COUNT=0
SORRY_REASONS=[]
NEW_AXIOM_COUNT=0
NEW_AXIOMS=[]
```

`OPERATOR_ALGEBRA_AUDIT.json` is the machine-greppable kernel dependency summary;
the text above is not a substitute for running the audit.

## Proved

### 1. Algebra, arbitrary embeddings, and the P1 gap

`MemoryAlgebra.AffineOperator S` contains a real-linear endomorphism and a
constant injection. Its monoid product applies the **left operand first**:
`(L₁,b₁) * (L₂,b₂) = (L₂ ∘ L₁, L₂ b₁ + b₂)`. Elementary labels need not themselves
form a closed family. `MemoryOperator Op S` supplies the actual step function,
an embedding, and an equality on **every** state.

Principal theorems in `Architectures/MemoryOperator.lean`:

- `MemoryAlgebra.AffineOperator.thenOp_assoc`
- `MemoryAlgebra.AffineOperator.apply_one`
- `MemoryAlgebra.AffineOperator.apply_mul`
- `MemoryAlgebra.AffineOperator.linear_mul`
- `MemoryAlgebra.AffineOperator.injection_mul`
- `MemoryAlgebra.AffineOperator.affine_identity`
- `MemoryAlgebra.memoryOperator_fold`
- `MemoryAlgebra.memoryOperator_scanCompatible`
- `MemoryAlgebra.conditional_scan_exists`
- `MemoryAlgebra.every_update_family_multiProgrammed`
- `MemoryAlgebra.independent_fold_projection`

`MemoryAlgebra.IsMultiProgrammedRecurrentStack` is the missing P1 type-level
predicate, parameterized by the update family. It requires at least two
programs and local dependence on each program's own prior state. It applies
even to nonlinear updates; it is not equivalent to scan compatibility.
`RecurrentResourceFormalism.slotStore_multiProgrammed` and
`RecurrentResourceFormalism.gdn2_multiProgrammed` instantiate it.

For literally arbitrary state transitions, `Architectures/LiftedMemory.lean`
encodes a full state as a basis vector of `X →₀ ℝ`. Named witnesses:

- `MemoryAlgebra.LiftedMemory.encode_injective`
- `MemoryAlgebra.LiftedMemory.lift_correct`
- `MemoryAlgebra.LiftedMemory.lift_comp`
- `MemoryAlgebra.LiftedMemory.arbitrary_family_scan`

This is a change of state space, not a claim that every nonlinear recurrence
is affine on its original coordinates or has efficiently composable summaries.

### 2. Emender and GDN2 semantic instances

`EmenderMemory.Store B K V` has numeric keys, values, per-slot strength, and
age. `State` additionally contains Boolean live flags. Ages are represented
over ℝ. `Op` implements all five requested operations; `Write` restricts a token
to at most one write; `TokenDecision` fixes pop, write, decay, and death data.

The direct arithmetic and diagonal encoding are independently defined and
equated by `EmenderMemory.step_eq_diagonal`. Named operation witnesses:

- `EmenderMemory.append_overwrites`
- `EmenderMemory.correct_full_overwrite`
- `EmenderMemory.correct_disabled`
- `EmenderMemory.correct_preserves_key`
- `EmenderMemory.merge_delta_rule`
- `EmenderMemory.merge_one_eq_correct`
- `EmenderMemory.merge_preserves_key`
- `EmenderMemory.pop_strength`
- `EmenderMemory.decay_live_strength_age`
- `EmenderMemory.decay_dead_frozen`
- `EmenderMemory.only_append_allocates`
- `EmenderMemory.append_allocates_only_selected`
- `EmenderMemory.freeMask_iff`
- `EmenderMemory.pop_death_iff`
- `EmenderMemory.decay_death_iff`

APPEND writes `(key, gate * value)`, sets strength **1**, and age **0** before
the birth-token decay. CORRECT overwrites the full **value** vector, preserving
the key, strength, and age. MERGE is `old + beta * (new - old)` and never
allocates. Both module entry points use binary write enables, so a fired MERGE
also specializes to overwrite; the generic magnitude remains modeled. POP and
DECAY kill at `<= free_threshold` without zeroing dead numeric rows.

Decomposition and compact closure witnesses:

- `MemoryAlgebra.slotInjection_eq_onehot`
- `MemoryAlgebra.DiagonalOperator.toAffine_thenOp`
- `MemoryAlgebra.DiagonalOperator.toAffine_composeAll`
- `MemoryAlgebra.diagonal_slot_decomposition`
- `EmenderMemory.step_decomposition`
- `EmenderMemory.token_step_eq_fold`
- `EmenderMemory.token_summary_correct`
- `EmenderMemory.token_decomposition`

The exact decomposition is `D ⊙ s + ∑ j∈A, e_j ⊗ b_j`. `D` includes overwrite
and MERGE erasure as well as POP/DECAY. `A` includes age injections, not just
selected write slots. The one-hot theorem certifies the outer-product shape;
no separate matrix-rank API theorem is claimed.

The same interface wraps the existing GDN2 definition, not a substitute model:

- `GDNMemoryOperator.embed_correct`
- `GDNMemoryOperator.gdn2_conditional_scan`
- `GDNMemoryOperator.gdn2_fold_scan`

### 3. Conditional taxonomy and named obstructions

`MemoryAlgebra.ScanCompatibleRelativeTo step schedule` requires a fixed affine
representation for each scheduled step, valid on every numeric state.
`RecurrentResourceFormalism.SemanticArchitectureSignature` carries the actual
update family and exposes `scanCompatibleRelativeTo`. The Boolean field remains
a backward-compatible unconditioned classification, explicitly documented as
insufficient evidence. The X30 slot store and E88/NDM have distinct signatures.

- `EmenderMemory.token_conditional_scan`
- `RecurrentResourceFormalism.slotStore_scan_relative`
- `MemoryAlgebra.routedRun_eq_replay`
- `MemoryAlgebra.routedRun_eq_scan_of_trace`
- `EmenderMemory.replay_eq_checked`

`TraceMatches` is a hypothesis that recorded operators equal the decisions on
the routed trajectory. `TokenDecision.ValidFor` checks the two death masks and
post-write live mask against the supplied numeric state. These hypotheses do
not prove that decision generation itself is parallel or state independent.

All counterexamples are proved:

- `MemoryAlgebra.Obstructions.hard_correct_eq_gate`: hard-gated CORRECT on a
  zero value, writing one, equals `1[g > 1/2]`.
- `MemoryAlgebra.Obstructions.hard_gate_witness` and
  `hard_gate_not_affine_on_unit_interval`: gates `0, 1/2, 1` violate the affine
  identity; the obstruction already holds on `[0,1]`.
- `MemoryAlgebra.Obstructions.deathThenDecay_formula`: checking death at `1/2`
  before another decay `1/2` gives `x` for `x <= 1/2` and `x/2` otherwise.
- `MemoryAlgebra.Obstructions.death_threshold_witness` and
  `state_dependent_death_not_affine`: the numeric strength channel itself
  violates the affine identity when death masks are recomputed.
- `EmenderMemory.age_activity_not_one_slot`: global age increments refute a
  one-slot bound on the complete numeric injection.

Associativity of function composition does not fail. The false unconditional
claim is compact affine closure on the original state/gate coordinates.

### 4. Sequence, chunk, prefix, and tree scan equivalence

In namespace `MemoryAlgebra.AffineOperator`:

- `foldl_eq_composeAll`
- `composeAll_eq_function_composition`
- `composeAll_append`
- `composeAll_flatten`
- `chunked_fold_equiv`
- `prefix_scan_equiv`
- `chunk_prefix_equiv`
- `composeAll_linear`
- `composeAll_injection`
- `injection_propagation`
- `prefix_injection_propagation`
- `injection_eq_suffix_sum`
- `prefix_product_injection_formula`

`MemoryAlgebra.ScanTree.summary_eq_composeAll` and `ordered_tree_equiv` license
any ordered parenthesization, including balanced trees. Empty schedules/chunks
are covered. The indexed formula propagates injection `i` through the suffix
strictly after `i`, uses no inverses, and handles overwrites/zero diagonal entries.
`EmenderMemory.token_fold_scan` and `token_chunked_scan` specialize to tokens.

### 5. Activity and sparse evaluation

In namespace `MemoryAlgebra`:

- `slotSupport_card_le`
- `slotSupport_injection_subset`
- `slotSupport_injection_card`
- `sparse_reconstruction`
- `diagonal_support_subset`
- `support_add_subset`
- `diagonal_composition_support`

In namespace `EmenderMemory`:

- `injection_support_subset`
- `injection_activity_bound`
- `content_injection_support_le_one`
- `sparse_step`
- `age_injection_dense`
- `age_activity_not_one_slot`
- `token_activity_bound`
- `token_content_support_le_one`

Per-token **content** support is at most one slot. Full numeric injection
support is at most B and can equal B because age increments apply to every
live slot. A diagonal multiplier never spreads support. Sparse reconstruction
is valid for any certified activity superset; neither reading/routing sparsity
nor constant activity for an arbitrary long chunk follows.

## Assumed, estimated, and open

1. **Source refinement:** the Python-to-Lean correspondence is manually checked
   at the equation level. Python execution, the allocator, native fp32 correction,
   timestamp management, and the timer wheel are not verified.
2. **Complete fixed schedule:** targets, binary enables, all gate magnitudes,
   residual-derived append gate inputs, and live/death masks must be supplied.
   Detachment does not remove forward dependence on state. `cell.py` records
   replay error; `naive_reference.py` can recompute it even with a routing trace.
3. **Architectural validity:** APPEND selects a free slot; CORRECT/MERGE select
   the architecture's matched slot. Those selector obligations are external.
   The algebra quantifies over all supplied decisions without silently assuming
   their producer has been verified.
4. **State omitted:** softmax read outputs, usage, pressure, allocator structures,
   optimizer state, and cross-layer input generation are outside the affine
   store. Boolean liveness is handled separately. Integer-age/range invariants
   and lazy-age representation equivalence remain open refinements.
5. **Numerics and learning:** exact real arithmetic only. No floating-point
   reassociation/parity/stability proof, rounding error bound, or gradient proof.
6. **Cost estimates:** with C tokens per chunk, inter-chunk serial depth is
   O(ceil(T/C)); including concurrent local passes gives an ideal O(C+T/C), or
   O(C+log(T/C)) with balanced summary scan. Decision generation can remain O(T).
   No per-token read/write volume is removed; summaries/replay may add traffic.
   These are assumptions/estimates, not Lean performance theorems.
7. **Universal embedding:** arbitrary transitions scan on the lifted full-state
   basis, with no compactness, finite-dimensionality, or efficient work claim.
8. **Runtime implementation:** no new GPU kernel or training run is included.
   A numerical implementation and its equivalence tests are subsequent work.

## Reference semantics hashes

Read-only source references, SHA-256:

```text
f43fa9c4e2be220494f5c17f903f529d97c2517845d1d1d67f4ac0fe455233f1  experiments/expressivity_tasks/x30_tape_cell/cell.py
68fe25baa015031df86ee810beb2b41474b89612f74d3d51519589bd46d4e119  experiments/expressivity_tasks/x30_tape_cell/naive_reference.py
```

## Exact changed-file list

Paths below are relative to `formal/lean/`. The complete list of source theorem
names and locations is also in `OPERATOR_ALGEBRA_AUDIT.json`; the index above
lists the principal citation surface and deliberately excludes autogenerated
structure/recursor lemmas.

```text
.gitattributes
ElmanProofs/Architectures/MemoryOperator.lean
ElmanProofs/Architectures/DiagonalMemory.lean
ElmanProofs/Architectures/EmenderMemory.lean
ElmanProofs/Architectures/EmenderToken.lean
ElmanProofs/Architectures/GDNMemoryOperator.lean
ElmanProofs/Architectures/LiftedMemory.lean
ElmanProofs/Architectures/ConditionalScan.lean
ElmanProofs/Expressivity/MemoryScan.lean
ElmanProofs/Architectures/RecurrentResourceFormalism.lean
ElmanProofs/PaperCore.lean
FORMALIZATION_GAP_AUDIT.md
EMENDER_OPERATOR_ALGEBRA.typ
EMENDER_OPERATOR_ALGEBRA.pdf
OPERATOR_ALGEBRA_STATUS.md
OPERATOR_ALGEBRA_AUDIT.json
scripts/audit_operator_algebra.lean
build_logs/operator_algebra_lake_build.log
build_logs/operator_algebra_trust.log
build_logs/operator_algebra_axioms.log
build_logs/operator_algebra_memo_compile.log
build_logs/operator_algebra_validation.txt
```

The root `ElmanProofs.lean` already imports `PaperCore`; adding the new imports
to `PaperCore` wires the formalization into the default `lake build`. No
toolchain, Lake dependency, Python source, training artifact, or external repo
was changed.
