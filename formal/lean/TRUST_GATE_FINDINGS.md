# Trust Gate Findings

**Date:** 2026-05-24  
**Gate result:** PASSES non-vacuously

## Environment

- ripgrep 14.1.1 (rev 4649aa9700), available in bash via `~/.local/bin/rg`
- Lean 4 / Lake via leanprover/lean4:v4.26.0 (elan)
- Mathlib v4.26.0

## Checks run

### `check_trusted_no_placeholders.sh ElmanProofs.lean`

Scanned 2 project source files in the import closure of `ElmanProofs.lean`
(`ElmanProofs.lean` + `ElmanProofs/PaperCore.lean`).

`rg` ran and found zero matches for `sorry`, `admit`, `axiom`, or `opaque`.

**Result:** trusted check passed: 2 project source files

### `check_paper_core.sh ElmanProofs/PaperCore.lean`

Scanned 8 project source files in the import closure of `ElmanProofs/PaperCore.lean`:

- ElmanProofs/Architectures/M2RNNComparison.lean
- ElmanProofs/Architectures/OnlineMemory.lean
- ElmanProofs/Architectures/RecurrentResourceFormalism.lean
- ElmanProofs/Expressivity/E1HDefinition.lean
- ElmanProofs/Expressivity/E88ExceedsE1HCapacity.lean
- ElmanProofs/Expressivity/S5NDMRealization.lean
- ElmanProofs/Expressivity/S5Tracker.lean
- ElmanProofs/Expressivity/S5Witness.lean

`rg` ran and found zero matches for `sorry`, `admit`, explicit `axiom`/`opaque`,
or `native_decide` in the paper core import closure.

**Result:** trusted check passed: 8 project source files; paper core check passed:
8 project source files, no native_decide

### `lake build ElmanProofs.PaperCore`

Built 2197 modules (full Mathlib + all project modules) with exit code 0.

Final project modules built:
- [2193/2197] ElmanProofs.Activations.Lipschitz
- [2194/2197] ElmanProofs.Expressivity.E1HDefinition
- [2195/2197] ElmanProofs.Architectures.RecurrentResourceFormalism
- [2196/2197] ElmanProofs.Expressivity.E88ExceedsE1HCapacity
- [2197/2197] ElmanProofs.PaperCore

**Result:** Build completed successfully (2197 jobs).

## Conclusion

Gate passes non-vacuously. The `rg` scan ran over real source files and found no
placeholder proof terms, explicit axioms/opaque declarations, or kernel-bypassing
`native_decide` calls in the trusted paper core import closure. The `lake build`
kernel type-checked the full proof chain to exit 0.

The paper's 'trusted Lean 4 core with no placeholders' claim is supported by this
conclusive build run.

## Logs

- `build_logs/2026-05-24_lake_build.log` — full `lake build` output (2197 jobs)
- `build_logs/2026-05-24_trust_gate.log` — trust gate scan output

---

## Post-Execution Check — 2026-05-24 (after F1/F3/F4/F5 batch)

**Gate result:** PASSES non-vacuously

### New modules now in the PaperCore closure

The following module was added to the transitive closure of `ElmanProofs/PaperCore.lean`
since the initial trust gate run:

- `ElmanProofs/Expressivity/NDMRealizesS5.lean` (added by F1)

Full closure is now **9 project source files** (was 8):

| File | Added by |
|------|---------|
| ElmanProofs/Architectures/M2RNNComparison.lean | pre-existing |
| ElmanProofs/Architectures/OnlineMemory.lean | pre-existing |
| ElmanProofs/Architectures/RecurrentResourceFormalism.lean | pre-existing / F3 / F4 |
| ElmanProofs/Expressivity/E88ExceedsE1HCapacity.lean | pre-existing |
| ElmanProofs/Expressivity/NDMRealizesS5.lean | **F1** |
| ElmanProofs/Expressivity/S5NDMRealization.lean | pre-existing |
| ElmanProofs/Expressivity/S5Tracker.lean | pre-existing |
| ElmanProofs/Expressivity/S5Witness.lean | pre-existing |
| ElmanProofs/PaperCore.lean | pre-existing |

### Theorem names from F1, F3, F4 the paper can cite

| Theorem | Module | Line | Task |
|---------|--------|------|------|
| `ndm_realizes_s5_tracker` | `ElmanProofs.Expressivity.NDMRealizesS5` | 391 | F1 |
| `ndm_m2rnn_flop_class_equiv` | `ElmanProofs.Architectures.RecurrentResourceFormalism` | 1165 | F3 |
| `multiProgrammed_admits_m2rnn_and_ndm` | `ElmanProofs.Architectures.RecurrentResourceFormalism` | 1264 | F4 |

All three theorems were verified by name (`rg`), confirmed present at the stated
line numbers, and checked to contain no `sorry` or `admit` in either source file.

### Checks run

1. `check_trusted_no_placeholders.sh ElmanProofs.lean` — **exit 0**, 2 source files
2. `check_paper_core.sh ElmanProofs/PaperCore.lean` — **exit 0**, 9 source files, no native_decide
3. `lake build ElmanProofs.PaperCore` — **exit 0**, 2198 jobs (full Mathlib + all project modules)

Final project modules built:
- [2194/2198] ElmanProofs.Activations.Lipschitz
- [2195/2198] ElmanProofs.Expressivity.E1HDefinition
- [2196/2198] ElmanProofs.Architectures.RecurrentResourceFormalism
- [2197/2198] ElmanProofs.Expressivity.E88ExceedsE1HCapacity
- [2198/2198] ElmanProofs.PaperCore

### No new placeholders

`rg` scanned all 9 project source files and found zero matches for `sorry`,
`admit`, explicit `axiom`/`opaque`, or `native_decide` in the trusted paper
core import closure. Gate still passes with no regressions.

### Post-execution logs

- `build_logs/2026-05-24_post_execution_trust_gate.log` — trust gate scan output
- `build_logs/2026-05-24_post_execution_lake_build.log` — full `lake build` output (2198 jobs)
