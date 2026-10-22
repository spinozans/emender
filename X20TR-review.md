## Review

**Verdict: ACCEPT-WITH-CONDITIONS.** The mathematical argument is sound on inspection. Documentation scoping and independent runtime verification remain conditions.

### Correct

- **Abstraction and quantifiers:** `RecurrentModel` permits any state type, initial state, and token-indexed transition; `T` is a horizon parameter (`LinTanhExpressivity.lean:73–81`). Consequently, `tanh_computations_are_lin_one_dim` quantifies over **any** recurrent model, including vector-valued TANH at every dimension—not merely the concrete scalar example (`:263–269`).
- **Terminal output only:** `Realizes` applies the decoder after the entire word (`:84–85`). This proves equality of terminal functions, not matching intermediate states, prefix outputs, or online behavior.
- **Faithful scalar interpretation:** The compiled LIN model carries a natural number (`:113`, `:236–237`), not literally a real vector. It faithfully embeds into one exact real coordinate: natural-number casting preserves `b*s+x` and is injective. An arbitrary real-valued-state decoder can extend the natural-state decoder off the reachable states. **That bridge is mathematically valid but not separately formalized here.**
- **Proof structure:** The accumulator identity establishes positional expansion; the tail bound separates different leading digits; induction establishes equal-length injectivity (`:127–229`). Universality then follows by extending the target along the injective state map (`:92–105`, `:246–256`). Compensation simply chooses the target to be the other model’s terminal output (`:263–269`). No hidden TANH approximation argument is needed.
- **Scope:** The actual alphabet is `Fin b`, with `1 < b`; other finite alphabets require ordinary injective token relabeling. The decoder may depend on both target and horizon. No learning algorithm, finite-precision guarantee, robustness result, or decoder-size bound is proved.
- **Corrections/open problems:** Identity-token preservation is proved (`:297`); general finite automata admit one-hot linear transition representations. The document correctly identifies bounded-precision separation and TANH interpolation as unproved. “LIN dominates” is justified as **non-strict containment**, not a demonstrated strict advantage.

### Findings and exact conditions

**P1 — Scope the advertised no-separator verdict consistently.**  
`THEOREMS.md:12–26` rejects a “task family” separator without stating precision in the verdict itself; `:151–156` again asserts no function-class impossibility without the fixed-horizon qualification. The Lean commentary similarly uses unrestricted language (`LinTanhExpressivity.lean:256–262`). Later caveats help, but do not make these claims individually accurate.

**Condition text:**

> State the verdict as: “COMPENSATION / LIN dominates non-strictly at each fixed finite horizon over a finite alphabet, with exact scalar state, no imposed precision budget, and an arbitrary terminal decoder that may depend on the horizon and target.” Apply these qualifications to the unrestricted no-separator claims. Use `d₀ ≥ 1` when invoking `1 ≤ d₀`. Do not present precision/parameter efficiency as the only remaining distinction: learnability, decoder complexity, and robustness are not addressed.

**P2 — Uniformity is not simply the precision caveat.**  
`THEOREMS.md:123–126` says the uniform version “reduces to” precision. The current encoding also loses length: `[0]` and `[0,0]` both reach zero even with exact arithmetic. Horizon-dependent targets assigning different outputs to those words cannot share this construction’s decoder.

**Condition text:**

> Replace “the uniform version reduces to the precision caveat” with “A shared decoder across horizons is not proved; this positional encoding also has cross-length collisions, independently of finite precision.”

This does not establish that a different uniform encoding is impossible.

**P2 — Correct the contraction hypothesis.**  
`THEOREMS.md:138` gives `c<1` as a contraction condition. For example, `c = -2` satisfies it but expands distances.

**Condition text:**

> Replace `c<1` with `|c|<1` (or `0 ≤ c < 1`). Keep identity-token preservation distinct from contraction or perturbation-resistant latching.

### Required parent audit

I cannot execute commands with the supplied read-only inspection tools. Therefore I **did not independently run** Lake or inspect runtime `#print axioms` output; the parent-reported green build is not my execution evidence.

Run this CPU-only audit, which changes no tracked source:

```bash
cd /home/erikg/emender/formal/lean
lake build ElmanProofs.Expressivity.LinTanhExpressivity &&
lake env lean --stdin <<'LEAN'
import ElmanProofs.Expressivity.LinTanhExpressivity

#print LinTanhExpressivity.linState_inj
#print LinTanhExpressivity.lin_one_dim_realizes_any_function
#print LinTanhExpressivity.tanh_computations_are_lin_one_dim

#print axioms LinTanhExpressivity.linState_inj
#print axioms LinTanhExpressivity.lin_one_dim_realizes_any_function
#print axioms LinTanhExpressivity.tanh_computations_are_lin_one_dim
LEAN
```

**Audit condition:** successful build and declaration checks; headline axiom dependencies restricted to standard Lean axioms (`propext`, `Classical.choice`, `Quot.sound`), with no `sorryAx` or additional assumptions.

**Merge verdict: BLOCK pending the conditions above.** No defect found in the three headline proofs by source inspection. This reviews the current committed file contents, not a historical commit diff. No files were edited.
---

## Parent attestation (2026-10-22)

X20TR conditions satisfied:
- Audit condition: run by the parent VERBATIM (lake build ElmanProofs.Expressivity.LinTanhExpressivity +
  #print axioms on all three headline theorems): Build completed successfully (2065 jobs);
  linState_inj / lin_one_dim_realizes_any_function / tanh_computations_are_lin_one_dim each depend
  only on [propext, Classical.choice, Quot.sound]. No sorryAx. Output retained.
- P1: THEOREMS.md verdict restated exactly per the condition text (non-strict domination, fixed
  horizon, finite alphabet, exact scalar state, no precision budget, arbitrary terminal decoder
  dependent on horizon/target; d0 >= 1; learnability/decoder-complexity/robustness listed as
  unaddressed alongside precision). Lean theorem commentary restated with the same scope
  qualifications. lake build re-verified green after the comment edit.
- P2 (uniformity): "reduces to the precision caveat" replaced with the condition text verbatim
  (shared decoder across horizons not proved; cross-length collisions [0] vs [0,0] independent
  of finite precision; different uniform encodings not excluded).
- P2 (contraction): c<1 corrected to 0 <= c < 1 with the |c|<1 contraction note (c=-2 expands);
  identity-token preservation kept distinct.

Verdict: X20TR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X20T compensation theorem closed.
