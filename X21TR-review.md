## Review

**Acceptance verdict: ACCEPT-WITH-CONDITIONS.** The Lean results checked are sound as stated; several headline interpretations exceed them. **Merge verdict: BLOCK pending the conditions below.**

### Correct

- **Streaming simulation and one-hot realization:** `StreamModel.runFrom_sim` preserves every emitted output, not merely terminal outputs. `matVec_onehot` uses the correct matrix orientation; the realization has dimension `n`, token-dependent `0/1` matrices, one-hot reachable states, and one shared readout.  
  Evidence: `formal/lean/ElmanProofs/Expressivity/StreamingExpressivity.lean:126–135,243–315`.

- **Q3 impossibilities:** both counting and balanced-block discrimination are genuinely activation-free and dimension-free, conditional on `BoundedAbstraction`. Counting uses distinguishable outputs; balanced discrimination uses congruence under a common suffix. Neither requires surjectivity.  
  Evidence: same file, `372–424`.

- **Literal cycle results:** scalar affine and scalar TANH steps both satisfy the proved no-three-cycle result. The dimension-two construction is correct: its matrix is
  \[
  \begin{pmatrix}0&-1\\1&-1\end{pmatrix},
  \]
  with characteristic polynomial \(x^2+x+1\). Its coefficients and orbit coordinates belong to `{-1,0,1}`, so are exactly representable in ordinary finite formats, including bf16. The characteristic polynomial is checked algebraically here; Lean proves the cube identity and explicit orbit, not a characteristic-polynomial theorem.  
  Evidence: same file, `437–536`.

- **Refuted witnesses:** the two exact TANH fixed points are correctly constructed over real arithmetic. Balanced-block counting cannot separate the activation classes under the finite-abstraction hypothesis.  
  Evidence: same file, `402–424,578–584`.

### Findings and acceptance conditions

#### 1. P1 — Q1/Q2 claim unproved activation and fixed-dimension equalities

`THEOREMS.md:25–28,46–60` claims equality at every dimension and `LIN = TANH`. Only LINEAR receives the automaton realization theorem. O1/O2 correctly acknowledge the missing activation result (`241–249`), contradicting the headlines.

**Condition:** replace those claims throughout both files with:

> Both activation classes admit the finite-automaton upper bound under the stated abstraction hypothesis. Every n-state automaton has a dimension-n LINEAR realization. No TANH converse or equality at fixed dimension is established.

The corresponding Lean documentation at `StreamingExpressivity.lean:317–321` also needs correction.

#### 2. P1 — The “characterization” declaration lacks the reverse abstraction bridge

`BoundedAbstraction` constrains **all** states (`149–155`). The reverse conjunct of `bounded_precision_class_characterization` proves only stream equivalence and reachable one-hot states (`322–332`); it supplies no bounded abstraction.

This distinction is substantive. For a two-state swapping automaton with distinct outputs, the ambient `finLinModel` has three distinguishable states: its two one-hot vectors and the zero vector. Zero always produces the fallback output; swapping distinguishes it from the corresponding one-hot state. Therefore the ambient model need not admit a two-valued abstraction.

Likewise, `rot3Model.out = id` prevents any finite abstraction on its entire rational-pair state space (`525–528`), although its initialized orbit is finite.

The forward theorem additionally assumes a **surjective** quotient (`202`), absent from the advertised “at most n” formulation.

**Condition:** either:
- add the invariant-domain/reachable-state bridge and address nonsurjective abstractions; or
- explicitly downgrade the formal headline to “surjective abstraction collapse plus finite-reachable one-hot realization,” without claiming the current declaration proves the stronger characterization.

#### 3. P1 — No literal three-cycle does not establish a function-class dimension hierarchy

The inference at `THEOREMS.md:67–68,192–193` and `StreamingExpressivity.lean:426–432` is invalid under the present semantic abstraction and arbitrary readout.

**Counterexample:** scalar LINEAR transition \(s\mapsto s+1\), initial state zero, with
\[
q(s)=\lfloor s\rfloor\bmod 3,\qquad \mathrm{out}=q.
\]
This is a surjective three-valued abstraction and emits `1,2,0,…`. Its real state never cycles, but its observable behavior implements the three-cycle automaton.

**Condition:** describe the results only as a separation in **literal state-cycle realizations**, not automaton/stream-function expressivity. A stronger hierarchy requires additional hypotheses and proof.

Also remove “you need d ≥ number of automaton states” at `THEOREMS.md:211–212`: `d=n` is sufficient, not proved necessary, and the stated two-dimensional three-cycle already contradicts that blanket necessity claim.

#### 4. P1 — The bf16 landing confuses coordinate precision with total channel bits

`THEOREMS.md:93–102` substitutes `B=16` into a definition where `B` measures the **whole channel**.

For 1,024 bf16 coordinates, the nominal bit-pattern upper bound is \(2^{16384}\), not \(2^{16}\). The one-hot theorem guarantees LINEAR realizations for automata with at most 1,024 states; it does not characterize every function available at that shape.

**Condition:** distinguish:
- whole-channel finite-state upper bound;
- the ≤1,024-state one-hot LINEAR construction;
- empirical learning/generalization.

Retain “4× transfer is consistent, interpretation only,” adding that these results neither identify the learned automaton nor predict unseen-horizon correctness.

#### 5. P1 — O4 attributes an unsupported floating-point model to Weiss

`THEOREMS.md:120–124,144,253–258` identifies Weiss’s setting with bounded mantissa/unbounded exponent. The parent-supplied source passages establish the paper’s stated finite-precision, input-bound counting separation, **not that exact numerical model**.

Fixed mantissa with unbounded exponent also does not make unit-increment counting exact indefinitely.

**Condition:** withdraw that identification. Describe Weiss’s precision/range assumptions separately, and say their relationship to this formal abstraction requires explicit modeling. Do not assert blanket failure of `BoundedAbstraction` merely from unbounded numerical range.

#### 6. P2 — Literature cleanup

- **Delete anchor 8** (`THEOREMS.md:149`). The indicator series of \(a^n b^n\) is not a rational weighted series in the standard sense: its Hankel submatrix indexed by \(a^i,b^j\) is an arbitrarily large identity matrix.
- **Restrict anchor 4** (`145`) to the weighted linear representation/channel and linear-readout subcase. Arbitrary decoded LINEAR stream functions are not generally rational series.
- Merrill 2019’s supplied passage specifies **\(O(\log L)\) bits at length \(L\)** and asymptotic/saturated acceptance—not this fixed total-bit model. Preserve that distinction.
- Merrill 2020 and Delétang’s labels are supported at the supplied abstract level. Add “in their tested task subset” to Delétang’s empirical outcome.
- Siegelmann–Sontag remains correctly labeled secondary-via-Weiss; no original-source verification or super-Turing conclusion is established.

O5/O6 appropriately leave learnability and the exact-arithmetic uniform-decoder question unaddressed.

### Validation and limitations

No files modified. The committed source was inspected directly; no committed-range diff artifact was supplied.

Execution tools were unavailable. Parent supplied:
- successful `lake build ElmanProofs.Expressivity.StreamingExpressivity`;
- axiom audits for 13 headline declarations, all using only `propext`, `Classical.choice`, and `Quot.sound`, with no `sorryAx`.

Literature verification above uses the exact passages supplied by the parent, not an independent browser fetch. Acceptance requires the corrections above and re-verification of any changed Lean declarations.
---

## Parent attestation (2026-10-22)

All six X21TR conditions applied VERBATIM by the conditions worker (commit 8d78a435,
pushed) and verified by the parent: (1) Q1/Q2 equality claims replaced with the
reviewer's exact text in THEOREMS.md and the Lean doc comment ('Both activation
classes admit the finite-automaton upper bound... No TANH converse or equality at
fixed dimension is established' - present in both files); (2) headline downgraded
to 'surjective abstraction collapse plus finite-reachable one-hot realization' with
the swapping-automaton/zero-vector counterexample and surjectivity note included;
(3) three-cycle results rescoped to literal state-cycle realizations with the
floor-mod-3 counterexample; the d>=automaton-states necessity claim removed;
(4) bf16 landing rewritten with the 2^16384 whole-channel bound vs the <=1024-state
one-hot guarantee vs empirical generalization as distinct categories, plus the
neither-identifies-the-learned-automaton-nor-predicts-unseen-horizon clause;
(5) Weiss float-model identification withdrawn, replaced with the finite-precision/
input-bound/range-arguments description and the explicit no-blanket-BoundedAbstraction-
failure clause; (6) anchor 8 deleted, anchor 4 restricted to linear-readout, Merrill
2019 O(log L) framing preserved distinctly, the Deletang verbatim abstract sentence
retained with the tested-task-subset qualifier, Siegelmann-Sontag unchanged.

Verification: parent-run lake build green (2198 jobs) after the comment-only Lean
edits; 13 headline axiom audits unchanged (standard axioms only); no theorem
statements or proofs touched (the diff is documentation and THEOREMS.md).

Verdict: X21TR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X21T unbounded-horizon theory
closed with all headline interpretations scoped to the compiled theorems.
