# Math-notation spec for §7 spine theorems (in-paper)

**Scope.** This spec governs **§7 inline use only**: the symbols, the
quantitative-bound notation, the witness/margin notation, the theorem-statement
style (inline form), and the machine-verification anchor convention used by the
two-to-four spine theorems lifted into §7 math notation. The arXiv PDF carries
the spine theorems inline; it does *not* carry full prose proofs.

The full statement list, the lemma DAG, and the prose proofs live in a
separate `leanblueprint`-style **web companion** that ships after the arXiv
post. Conventions for that companion (proof-presentation style, lemma citations,
length norms, reinterpretability-of-proofs discipline) are deferred and live
at the bottom of this file under "Deferred: blueprint companion spec".

**Status.** Decisive for the in-paper §7 scope. No "either/or" choices left
for the §7 spine-theorem render tasks.

---

## 1. Symbols for the core objects

These symbols are **fixed**. Render tasks must not substitute alternatives.

| Object | Symbol | Typst form | Notes |
|---|---|---|---|
| Emender matrix state at head $h$ | $S_h$ or $S$ | `S_h`, `S` | Matches paper §3 exactly. Drop the $h$ subscript when the theorem statement is single-head. |
| M²RNN matrix state | $H$ | `H` | Matches paper §3 ablation block and Lean side-by-side. Never use $H$ for the Emender state. |
| Key vector (write address) | $k$ | `k` | $L^2$-normalised in body; theorem statements should say "with $\|k\| \le 1$" or "$\|k\|=1$" as the hypothesis requires. |
| Value vector | $v$ | `v` | |
| Query vector (read address) | $q$ | `q` | |
| Prediction error / delta | $\delta$ | `delta` | $\delta = v - S^{\!\top} k$. Body §3 uses the same symbol. |
| Scalar decay | $\lambda$ | `lambda` | Theorem-statement parameter. Body §3 uses $d_h \in (0,1)$ for the learned per-token decay; the trusted-core theorems use $\lambda$ as a free scalar with $\lambda > 0$ stated. Bridge the two by a single sentence in §7 preamble; do not rename. |
| State dimension (common) | $d$ | `d` | $d = N = V$ in body §3; the trusted-core theorems use $d$ for the common matrix dimension. |
| State rows (keys) | $N$ | `N` | |
| State columns (values) | $V$ | `V` | |
| Number of heads | $H$ (count) | `H` | Distinct from M²RNN state $H$ by context (italic in math, prose in body). When both appear in the same theorem, write "$\hat H$ heads" for the count. |
| Time / step index | $t$ | `t` | $t = 0, 1, 2, \dots$. |
| Layer index | $\ell$ | `ell` | |
| Head index | $h$ | `h` | |
| Slot index | $(i, j)$ | `(i, j)` | $i \in \{0,\dots,N-1\}$, $j \in \{0,\dots,V-1\}$. |
| Norm | $\|\cdot\|$ | `norm(dot)` | Default is $L^2$ (Euclidean); annotate when ambiguous. |
| Inner product | $\langle k, q \rangle$ or $k^{\!\top} q$ | `<k, q>` or `k^T q` | Either form; pick one per theorem. The Lean `keyDot` is $\langle k, q \rangle$. |
| Identity matrix | $I$ | `I` | Dimension inferred from context. |
| Iterated tanh | $\tanh^{\circ k}$ | `tanh^(compose k)` or `tanh^(circle.small k)` | Used only when the $k$-step separation spine theorem references it inline. |
| Witness state (2D) | $S^\star$ | `S^*` | The "lower-left" witness matrix in the one-/two-/$k$-step separation. Replaces Lean's `lowerLeftState`. |
| Witness key (2D) | $k^\star$ | `k^*` | Replaces Lean's `mixedKey`. |
| Architecture predicates | $\mathrm{Emender}$, $\mathrm{M^2RNN}$, $\mathrm{M^2RNN\text{-}CMA}$, $\mathrm{GDN}$ | `"Emender"`, `"M"^2"RNN"`, `"M"^2"RNN-CMA"`, `"GDN"` | Use the typst function `upright()` or quoted string form already used in §3. |
| External-forget class | $\mathcal{F}_{\mathrm{ext}}$ | `cal(F)_"ext"` | Defined inline at first use as "the class of fixed-weight M²RNN-style resources with row, column, or cell external forget gates". |

### Architecture-discriminating predicates

The full body of theorems uses the **external-forget class** $\mathcal{F}_{\mathrm{ext}}$
defined as

$$
\mathcal{F}_{\mathrm{ext}} := \{\text{row-forget}\} \cup \{\text{column-forget}\} \cup \{\text{cell-forget}\}
$$

with each variant fixed-right ($H W$ with $W$ data-independent), raw-write
($+ k v^{\!\top}$ outer product), and a forget interpolation that depends only
on the within-step inputs. **Render tasks must keep all three sub-classes named
inside the theorem statement when the Lean statement quantifies over them.**

---

## 2. Notation for quantitative bounds

### Saturated slot

$$
|S_{i,j}| > 1 - \eta, \qquad \eta \in (0, 1).
$$

Typst: `abs(S_(i,j)) > 1 - eta`. The symbol $\eta$ (`eta`) is reserved for
the saturation slack throughout. When the post-update slot is involved write
$|S'_{i,j}| > 1 - \varepsilon$ with $\varepsilon$ (`epsilon`).

### Bounded input

Pre-activation perturbation bound: $|\delta| \le M$ where $\delta$ is a scalar
perturbation to a pre-activation entry and $M \in \mathbb{R}_{>0}$.

For vector inputs the convention is $\|v\| \le 1$ and $\|k\| \le 1$, with the
norm being $L^2$ unless explicitly subscripted. **Do not drop the hypothesis
when it appears in the Lean statement.**

### Margin / bound-away-from-zero

For separation theorems, the margin appears as

$$
\bigl\| \text{traj}_{\mathrm{E}} - \text{traj}_{\mathrm{R}} \bigr\| \ge \mu, \qquad \mu > 0,
$$

with $\mu$ written out explicitly whenever the proof produces it. For the
existential form ("there exists $\mu > 0$") write "$\mu > 0$" and exhibit the
value used in the proof. **Never replace an explicit constant with "$> 0$"
when the proof produces an explicit constant.**

### Release threshold (counter-delta release)

$T^\star := \lambda |S_{i,j}|$. Any write contribution $W$ with $W$ in the
opposite sign of $S_{i,j}$ and $|W| > T^\star$ releases the latch. The
quantitative companion gives post-update magnitude at least $\tanh(\mu)$
where $\mu := |W| - T^\star > 0$.

---

## 3. Notation for witness constructions

### The 2-D alphabet

The state space is $\mathbb{R}^{2 \times 2}$. The witness state is

$$
S^\star = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix},
$$

and the witness key is $k^\star = (\tfrac{1}{\sqrt 2}, \tfrac{1}{\sqrt 2})^{\!\top}$,
the "mixed" key. Typst:

```typst
S^* = mat(0, 0; 1, 0), quad
k^* = vec(1/sqrt(2), 1/sqrt(2)).
```

### The $k$-step witness sequence

The $k$-step witness input sequence is

$$
[(k^\star, 0), (0, 0), (0, 0), \dots, (0, 0)] \in (\mathbb{R}^2 \times \mathbb{R}^2)^k.
$$

Render this in Typst as

```typst
[(k^*, bold(0)), (bold(0), bold(0)), dots, (bold(0), bold(0))] in (RR^2 times RR^2)^k.
```

The leading pair injects the witness; the $k-1$ trailing zero pairs are
the "filler". In prose call this "the $k$-step witness" and refer back to
$S^\star, k^\star$ rather than re-displaying the values inside subsequent
prose.

### Trajectory-difference norms

For a state-space resource $\mathcal{R}$ with update rule $\mathcal{R}.\text{update}$,
the $k$-step trajectory starting from $S^\star$ on the witness input is
$\mathcal{R}^{(k)}(S^\star)$. Trajectory difference between Emender and
$\mathcal{R} \in \mathcal{F}_{\mathrm{ext}}$ is

$$
\Delta^{(k)} := \mathcal{R}^{(k)}(S^\star) - \mathrm{Emender}^{(k)}(S^\star) \in \mathbb{R}^{2 \times 2}.
$$

Entrywise margin at slot $(0,0)$ is the standard distinguishing-entry argument:
$\Delta^{(k)}_{0,0} = -\tanh^{\circ k}(-1) \neq 0$.

---

## 4. Theorem-statement style (§7 inline form)

Theorem statements inside §7 are written as **a single English sentence (or
two)** that carries every Lean-side hypothesis inline, ends in the displayed
or inline bound, and is followed by a parenthetical "(Lean: `<qualified
name>`)" anchor. The existing §7 already uses this convention for theorem
sets A–F; subsequent spine-theorem renders must match it.

Form template:

```typst
For every Emender slot saturated with $abs(S'_(i,j)) > 1 - epsilon$ and
every pre-activation perturbation $delta$ with $abs(delta) <= M$ and
$epsilon + M < 1$, the post-tanh slot moves by at most
$2 (epsilon + M) abs(delta)$ (Theorem F1, *saturation insensitivity*;
Lean: `EmenderLatching.emender_saturation_insensitivity`).
```

### Rules

- **No proofs in §7.** The inline form is statement + ~1-sentence intuition
  only. Full proofs live in the blueprint companion (see "Deferred" section
  below); the in-paper version cites the Lean anchor and moves on.
- **One sentence per theorem statement.** If the sentence is unreadable at
  one sentence, that is a sign the theorem belongs in the appendix /
  companion rather than inline. The four-to-six load-bearing latch /
  separation results are short enough to read as one sentence; the longer
  capacity-style targets are not §7 candidates.
- **Hypotheses inline, not numbered.** Numbered (i), (ii), (iii) form belongs
  in the verbose appendix/companion style, not §7 inline.
- **Theorem name + Lean anchor at the end.** Format
  `(Theorem F1, *saturation insensitivity*; Lean: \`<qualified name>\`)`.
  Variants like `(Lean: ...)` alone are acceptable when the §7 prose already
  named the result; the bare `(Lean: ...)` form is the existing §7 default.

### Faithfulness discipline (fidelity > cleanliness)

Every §7 inline statement must include every hypothesis the Lean statement
quantifies over. In particular:

- The **external-forget class** qualifier ("for every fixed-right raw-write
  M²RNN-style resource with row, column, or cell external forget gate") must
  appear verbatim in the math statement whenever the Lean statement uses
  `FixedRightRawExternalForget2` or equivalent. Do not collapse to "every
  M²RNN-style resource".
- The **fixed-weight** condition ("with fixed weights $W$") must appear
  whenever the Lean statement abstracts over $W$ as a parameter.
- **Existential vs. universal** quantifiers must match. The Lean
  `emender_m2rnn_k_step_separation_exists` is existential over a witness
  sequence; render that as "there exists a $k$-token input sequence …", not
  "for every $k$-token input sequence …".
- **Norm hypotheses** ($\|k\| \le 1$, $\|v\| \le 1$) are dropped only if the
  Lean statement does not contain them.
- **The $k \ge 1$ lower bound** in the $k$-step separation statement is kept,
  not silently dropped.

The principle: **fidelity > cleanliness.** A statement that is slightly uglier
than it could be but matches the Lean exactly is correct; a statement that
reads cleanly but omits a hypothesis is wrong.

---

## 5. Machine-verification anchor convention

### Anchor text

```
Machine-verified in Lean: ElmanProofs.Architectures.<Module>.<theorem_name>
```

In §7 inline form the anchor appears in parenthetical short form
"(Lean: `<qualified name>`)" — already the existing §7 convention. The
**full** "Machine-verified in Lean:" phrase is reserved for the longer
appendix / companion form (deferred).

The fully qualified Lean name is the anchor. The actual URL rendering helper
is a separate task (`v22-lean-url-helper`); this spec decides anchor text and
placement, not URL rendering.

### Placement

**Parenthetical, at the end of the theorem-statement sentence, before the
sentence-ending period.** This matches existing §7 conventions where
`(Lean: \`emender_realizes_s5_tracker\`)` is the trailing parenthetical.

Concretely, the structure of a §7 inline entry is:

```
<English sentence carrying hypotheses and conclusion>
(Theorem <label>, *<name>*; Lean: `<qualified name>`).
```

For theorems whose Lean statement is split across two or more sub-theorems
(e.g., the F-set companions), list the qualified names separated by commas
inside the same parenthetical.

---

## 6. Discipline for "ugliness"

The Lean has several specifications that are mathematically faithful but
verbose: the external-forget sub-class taxonomy (row/column/cell), the
fixed-right parameterisation condition, the orthonormal-key family
restriction, the unit-key inner-product hypothesis ($\|k\|^2 = 1$), etc.
**These must appear in the §7 inline math statement, translated to standard
notation, not paraphrased away.**

Examples:

- *Lean:* `FixedRightRawExternalForget2` with three sub-cases.
  *Math (correct):* "every fixed-right raw-write M²RNN-style resource with
  row, column, or cell external forget gate, i.e., every
  $\mathcal{R} \in \mathcal{F}_{\mathrm{ext}}$".
  *Math (forbidden):* "every M²RNN-style resource".
- *Lean:* `keyDot k k = 1`.
  *Math (correct):* "$\|k\|^2 = 1$" or "$k$ is a unit key".
  *Math (forbidden):* (silently omitted).
- *Lean:* `1 ≤ k` (premise of $k$-step separation).
  *Math (correct):* "for every $k \ge 1$".
  *Math (forbidden):* "for every $k$".

The principle stated as a slogan: **fidelity > cleanliness.** A theorem
statement is allowed to read slightly more verbosely than it would in a
non-machine-verified paper, because the qualifier you elided is the qualifier
the proof relied on, and a reader in 2056 trying to re-verify the result in
a different system will need it.

---

## 7. Worked example: F1 (saturation insensitivity), §7 inline form

The worked example is **§7-inline only**: a one-to-two-sentence English
statement carrying every hypothesis, a displayed or inline bound, and the
machine-verification anchor.

```typst
For every Emender slot saturated with $abs(S'_(i,j)) > 1 - epsilon$ and
every pre-activation perturbation $delta$ with $abs(delta) <= M$ and
$epsilon + M < 1$, the post-tanh slot moves by at most
$2 (epsilon + M) abs(delta)$ (Theorem F1, *saturation insensitivity*;
Lean: `EmenderLatching.emender_saturation_insensitivity`). The bound
shrinks linearly in both the saturation slack $epsilon$ and the
perturbation magnitude $M$: at deep saturation a bounded write barely
moves the slot, which is the quantitative content of the latching half
of the Emender primitive.
```

The first sentence is the theorem statement (hypotheses inline, displayed
bound at the end, parenthetical label-and-Lean anchor). The second sentence
is the one-sentence intuition.

No prose proof is rendered in §7. The proof lives in the blueprint companion
that ships after the arXiv post.

---

## Summary of the seven decisions (in-paper / §7 scope)

For the `wg log` summary record:

1. **Symbols.** State $S$ (Emender), $H$ (M²RNN); inputs $k, v, q$; indices
   $t, \ell, h, (i,j)$; scalars $\lambda, \eta, \varepsilon, M, \mu$;
   dimensions $N, V, d, H$ (count); external-forget class
   $\mathcal{F}_{\mathrm{ext}}$.
2. **Bounds.** Saturation $|S_{i,j}| > 1 - \eta$; bounded inputs
   $\|v\| \le 1$, $\|k\| \le 1$; margins $\ge \mu > 0$ explicit; release
   threshold $T^\star = \lambda |S_{i,j}|$.
3. **Witnesses.** 2-D alphabet with $S^\star$, $k^\star$; sequence
   $[(k^\star, 0), (0, 0), \dots, (0, 0)]$ of length $k$; trajectory
   difference $\Delta^{(k)}$ with entrywise margin at $(0,0)$.
4. **Theorem statements (§7 inline).** One English sentence carrying all
   Lean-side hypotheses, ending in the displayed or inline bound, followed
   by `(Theorem <label>, *<name>*; Lean: \`<qualified name>\`)`. No proofs
   in §7; one-sentence intuition only.
5. **Fidelity > cleanliness.** External-forget sub-class taxonomy,
   fixed-weight conditions, unit-key inner products, $k \ge 1$ lower bounds —
   all preserved verbatim in the math statement.
6. **Machine-verification anchor.** Parenthetical short form
   `(Lean: \`<qualified name>\`)` at end of statement sentence; full
   "Machine-verified in Lean:" phrase reserved for blueprint companion.
7. **Ugliness discipline.** Translate Lean qualifiers (external-forget
   sub-class, fixed-right, orthonormal-key) into standard notation; do not
   paraphrase them away.

**Worked theorem rendered.** F1 (saturation insensitivity) as it would
appear in §7: one statement sentence + one intuition sentence + Lean anchor.
Typst-renderability confirmed via scratch compile (`paper/data/_scratch_F1_render.typ`,
deleted after verification).

---

## Deferred: blueprint companion spec

The following conventions are **deferred** to a separate
`leanblueprint`-style web companion that ships after the arXiv post. They
do not govern the in-paper §7 spine theorems; they govern the companion
where the full statement list, the lemma DAG, and the prose proofs live.

This section is a placeholder so the conventions are not lost; the companion
task will adopt or revise them. Not load-bearing for this task.

### D.1 Hierarchy

- **Theorem.** Top-level result. Numbered, named. Used for the spine results
  (latching set F, separation sets C/C′/D, etc.).
- **Proposition.** Standalone but non-spine result. Numbered, named.
- **Lemma.** Auxiliary used inside one or two adjacent proofs. Numbered, may
  be unnamed.
- **Corollary.** Direct consequence of a stated Theorem. Numbered, named.

### D.2 Display equations vs. inline

Display when: the equation is a step referenced again later; the equation is
longer than a typical typeset line; or the equation is the conclusion of the
theorem. Inline for single-symbol references inside English prose.

### D.3 Proof environment

Open with bold `*Proof.*` and close with $\square$ at the right margin.
Multi-paragraph proofs put the $\square$ at the end of the last paragraph.

### D.4 Case splits and induction

Case splits: numbered list (`+` enumeration), one bullet per case, max one
level deep (factor as a Lemma otherwise). Induction: open with
`*Proof by induction on $k$.*` then `*Base case.*` and `*Inductive step.*`
blocks.

### D.5 MVT, tanh-monotonicity, and similar real-analysis citations

Cite the underlying mathematical fact by its standard name. Examples:

- "By the mean value theorem applied to $\tanh$ on $[Z, Z + \delta]$ …"
- "Since $\tanh$ is strictly monotone increasing …"
- "By injectivity of $\tanh$ …"
- "Since $\tanh' = 1 - \tanh^2 \le 1$ …"
- "By the chain rule …", "by the triangle inequality" — all written in prose.

### D.6 Proof length norm

Target: 5–15 minutes for a competent reader to verify, per theorem.
Operationally: 10–30 lines of typeset prose for the companion form. Longer
than ~40 lines suggests factoring into Lemma + Theorem. Shorter than ~5
lines suggests the proof has been compressed past the point a non-Lean
reader can follow.

### D.7 Reinterpretability discipline

Proofs in the companion must not cite Lean-specific tactic names,
Mathlib4-specific lemma names, or any other proof-assistant-specific
construct. Cite the underlying mathematical fact. If the fact is
non-standard, state it inline with full hypotheses; do not externalise to
a Lean lookup.

Forbidden in companion prose proofs (non-exhaustive):

- `Real.tanh_strictMono`, `Real.tanh_injective`, `Real.tanh_zero`,
  `tanh_bounded`, any other Mathlib identifier.
- Tactic names: `simp`, `linarith`, `ext`, `fin_cases`, `induction'`, etc.
- Type-class names: `Fintype`, `Decidable`, `LinearOrder`, etc.
- Module / namespace prefixes: `OnlineMemory.`, `Matrix.`, etc.

The Lean side of the trusted core is the verification, *not* the source of
mathematical authority. A reader in 2056 with a different proof assistant
must be able to recover the proof from the companion prose alone.

### D.8 Verbose theorem-statement form

```
*Theorem (F1, saturation insensitivity).* Let S in RR^(N times V) be an
Emender state matrix and let lambda > 0 be a scalar decay. Fix a slot
(i, j), pre-activation entry Z = lambda S_(i,j) + W, and post-activation
entry S'_(i,j) = tanh(Z). Assume:

  (i) The slot is saturated: abs(S'_(i,j)) > 1 - epsilon for some
      epsilon in (0, 1).
  (ii) A scalar perturbation delta to the pre-activation entry has
       abs(delta) <= M.
  (iii) epsilon + M < 1.

Then

  abs(tanh(Z + delta) - tanh(Z)) <= 2 (epsilon + M) abs(delta).

† Machine-verified in Lean: ElmanProofs.Architectures.EmenderLatching.emender_saturation_insensitivity.
```

Numbered hypotheses (i), (ii), (iii) when three or more; inline "and"
otherwise. The machine-verification footnote sits at the end of the
theorem statement, before the proof begins.

### D.9 Status of D.1–D.8

These items were drafted under the original task scope. The author's
scope correction moved them to the blueprint companion. The companion
task (TBD task ID) will adopt or revise them; treat the text above as a
starting draft, not a finalised spec.
