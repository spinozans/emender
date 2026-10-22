## Review

**Scope:** Reviewed the seven Lean modules, both receipts, and relevant supplied-paper definitions/proofs. Snapshot review relies on the parent’s attestation that these paths equal commit `6899754e`. Build success (2198 jobs) and the fourteen declaration axiom audits are **parent-supplied evidence**, not independently executed here.

### Correct
- R1’s scalar-decay recurrence identity and DPLR algebra are proved; parity recognition is correctly left external (`ArmMapping.lean:161–219`, `PrecisionRegime.lean:307`).
- Exact integer zero masking, configuration embedding, balanced-block discrimination, and the witness’s cardinality bound are established (`CounterMachine.lean:111–132,194–202,319–406`).
- WFA evaluation, its deterministic specialization, and the literal matrix obstruction are established (`WeightedAutomata.lean:134–209`).
- The omega simulation/collapse and exact corner realization have appropriately explicit abstraction/reachable-orbit assumptions. R4 proves homogeneous-product identities and padding/depth arithmetic—not circuit-class membership.

### Findings

1. **P1 — The precision-schedule centerpiece is vacuous.**  
   `countModel_needs_growing_budget` assumes a **global**, all-length abstraction for every `n` (`PrecisionRegime.lean:180–186`). But `h 0` already contradicts `countModel_no_omega_finite_state`, regardless of the schedule. Even a growing schedule cannot satisfy this premise.  
   **Condition:** introduce length-restricted precision assumptions and prove the corresponding lower bound, or explicitly demote this theorem to an impossible-global-abstraction corollary and mark the Def.-10 schedule bridge unproved.

2. **P1 — R2’s closed-form scope exceeds its proof.**  
   `cmRealStep` retains discrete control, Boolean equality decisions, and arbitrary lookup functions (`CounterMachine.lean:132,152–155`). Only `zeroMaskForm` receives an arithmetic-expression correctness theorem; there is no assembled closed-form transition proof, including for the one-counter witness. Thus THEOREMS O14’s “proved for the one-counter witness” is unsupported (`THEOREMS.md:447–452`). Additionally, the machine definition permits additive updates only (`CounterMachine.lean:76–91`), whereas the paper’s Lemma 3 includes reset updates.  
   **Condition:** complete those constructions, or widen O14 and consistently describe `cmModel_fold` as an exact embedding of the defined additive-update machine—not the completed closed-form Lemma-3 simulation.

3. **P1 — Exact-real upper-bound transfer is still asserted despite the acknowledged datatype gap.**  
   `MAPPING_RECEIPT.md:111,181,223–224` says the paper’s upper bounds apply; `THEOREMS.md:84,195–199` repeats that placement. Yet `MAPPING_RECEIPT.md:283–285` explicitly says transfer to the exact-real class is not automatic. `expTaylor_two_ne_exp` establishes a discrepancy, not a transfer theorem or computational-class strictness.  
   **Condition:** qualify every placement by the paper-compatible rational approximation semantics, or supply the bridge. Remove unsupported “strictly stronger” conclusions.

4. **P1 — The cycle obstruction is overinterpreted.**  
   `cycle3_not_dplr` excludes one literal transition matrix, not all encodings of three-cycle behaviour, nor constructions with fewer than four layers. Nevertheless, `THEOREMS.md:375–376` and `MAPPING_RECEIPT.md:289–292` claim behavioural strictness and four-layer necessity. For example,
   \[
   B=\begin{pmatrix}0&-1\\1&-1\end{pmatrix}
    =\operatorname{diag}(-1,0)-(1,1)^T(-1,1)
   \]
   is DPLR and cycles `(1,0) → (0,1) → (-1,-1) → (1,0)`.  
   **Condition:** restrict the claim to literal matrix realizability in the specified coordinates; describe the paper’s four-layer construction as sufficient, not proved necessary.

5. **P1 — E97 ReLU/softplus lower bounds are not established by the citation.**  
   `MAPPING_RECEIPT.md:225` and `THEOREMS.md:433–436` place those configurations under the paper’s lower bounds. The paper constructs **ReLU MLP** transitions; E97’s cited implementation applies one configured elementwise activation (`ndm/models/e88_fla_hybrid.py:1356–1374`). Non-saturation alone supplies no simulation theorem, and the paper supplies no softplus construction.  
   **Condition:** label these configurations as candidate hosts requiring a realization proof, not configurations already covered by the lower bounds.

6. **P2 — “Accepts exactly \(a^k b^k\)” overstates the witness.**  
   `balancedCM` ignores the previous phase when updating (`CounterMachine.lean:207–215`), so it also accepts `abab`. The actual `SeparatesBalanced` theorem is narrower and correct.  
   **Smallest fix:** change the recognizer comments to balanced-block discrimination.

7. **P2 — Robust decoding is not implemented by the supplied readout.**  
   `enc_robust_decode` proves uniqueness of nearby corners (`StreamingTanhRealization.lean:120–139`); the model readout instead requires exact corner equality and otherwise returns the initial-state output (`:188`). A small perturbation of a noninitial corner therefore need not decode correctly.  
   **Smallest fix:** distinguish geometric separation from an implemented robust decoder and from finite-precision dynamical guarantees.

8. **P2 — The training interpretation remains stated as a theorem consequence.**  
   `MAPPING_RECEIPT.md:231–234` says shallow-prefix learning is “exactly what” the fixed-budget theorems predict. Those theorems provide neither a 3721-state capacity obstruction nor a learnability bound; THEOREMS itself acknowledges this (`THEOREMS.md:93–99`).  
   **Smallest fix:** remove the predictive/necessary causal claim.

### Verdict

**Acceptance: BLOCKED. Merge verdict: BLOCK.**

Conditions 1–5 require proof completion **or consistent narrowing of the acceptance claims**. The existing kernel-checked mathematics can remain; green builds and standard-axiom audits do not resolve these statement-to-claim gaps. No files were edited.
---

## Parent attestation (2026-10-22)

All eight X22TR conditions applied via the reviewer's NARROWING option (commit 592eef0f,
pushed) and verified by the parent: (1) countModel_needs_growing_budget demoted in the
docs to an impossible-global-abstraction corollary with the Def-10 schedule bridge marked
UNPROVED; (2) the cmRealStep discrete-control/lookup scope stated, only zeroMaskForm
carries arithmetic correctness; (3) all exact-real placements qualified to
rational-approximation semantics, every 'strictly stronger' conclusion removed (grep:
zero occurrences in both docs); (4) cycle3_not_dplr restricted to literal matrix
realizability with the reviewer's rot3-is-DPLR counterexample documented verbatim
(B = [[0,-1],[1,-1]] = diag(-1,0) - (1,1)^T(-1,1)); four-layer = sufficient, not necessary;
(5) E97 ReLU/softplus configurations relabeled candidate hosts requiring realization
proofs; (6) balancedCM comments corrected to balanced-block discrimination; (7)
enc_robust_decode's geometric separation distinguished from an implemented robust decoder;
(8) the predictive causal claim about shallow-prefix learning removed (grep: gone).
Statements and proofs unchanged (docs and Lean doc comments only). Parent-run lake build
green (2198 jobs) after the edits; the 14-declaration axiom audits stand unchanged.

Verdict: X22TR BLOCKED -> ACCEPTED (conditions 1-5 satisfied by consistent narrowing
per the reviewer's stated alternative; 6-8 applied). The X22T re-proof is closed with all
claims scoped to the compiled kernels.
