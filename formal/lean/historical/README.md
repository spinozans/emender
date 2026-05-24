# Historical Lean Sketches

This directory contains pre-extraction sketches and retired theoretical approaches that were
explored during the NDM formalization but are **not part of the trusted paper core**.

## Purpose

The `formal/lean/ElmanProofs/PaperCore.lean` import closure defines the trusted boundary for
the paper's Lean-backed claims. Files in this `historical/` directory are outside that boundary
and may contain:

- `sorry`, `admit`, axiom, `opaque`, or `native_decide` placeholders
- Exploratory proofs that were abandoned in favor of cleaner approaches
- Overclaims that were refined or removed in the revised paper narrative
- Sketches from earlier theoretical directions (e.g., the TC0/parity-centered story that was
  superseded by the fixed-state/NC1/S5 framing)

## Trust Boundary

**Files in `formal/lean/ElmanProofs/PaperCore.lean` (and their transitive import closure):**
- Are checked by `scripts/check_paper_core.sh` to have zero placeholders
- Are kernel-type-checked by `lake build ElmanProofs.PaperCore` 
- Support the paper's claims in §6–7

**Files in `formal/lean/historical/`:**
- Are NOT checked by the trust-gate pipeline
- May contain exploratory or incomplete proofs
- Should NOT be cited as formal evidence unless their claims are first ported into
  `PaperCore` and re-proved without placeholders

## Workflow for Recycling Historical Sketches

When a historical sketch contains a useful idea:

1. Copy the needed statement into a new small module or an existing trusted module
   in `PaperCore` or its import closure.
2. Weaken it until it is true and checkable.
3. Prove it without `sorry`, `admit`, `axiom`, `opaque`, or `native_decide`.
4. Add it to `PaperCore.lean` **only** if it directly supports a paper-facing claim.

The old files are a mine of ideas, not a debt pile to clean up.

## Current Contents

**As of 2026-05-24:** This directory is a placeholder. All current Lean modules
(`Activations/Lipschitz.lean`, the architecture modules, and the expressivity modules)
are in the trusted core. No files have been moved here yet.

If future work introduces exploratory sketches with placeholders, they should be
moved into the corresponding subdirectories here (e.g., `historical/Expressivity/`
for expression-related sketches, `historical/Architectures/` for architecture sketches).

## Motivation for Existence

This directory is created proactively to:

1. **Prevent future ambiguity** about the trusted boundary. Readers can see at a glance
   which files are in scope for paper claims and which are not.
2. **Formalize the quarantine policy.** By creating the directory and documenting its purpose,
   we make it clear that historical sketches have a reserved location, separate from the
   active build.
3. **Support the trust-gate narrative.** The paper's §7 states "all claims are Lean-checked
   with no placeholders". This directory clarifies that the statement refers to the PaperCore
   closure, not the entire repository.

See `formal/lean/TRUSTED_PROOF_SURFACE.md` and `formal/lean/FORMALIZATION_GAP_AUDIT.md`
for the related strategy and task list.

---

*Last updated: 2026-05-24 (task: lean-f5-quarantine-historical)*
