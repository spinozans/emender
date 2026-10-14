# Emender reader first editorial roadmap

## The story to tell

Emender makes recurrent memory a place to compute. Each input reads the current memory, writes a correction, and transforms the state that the next step will receive. The systems work makes that sequential computation practical at language-model scale. The experiments then ask which computations the trained model actually learns, including the cases where its expected advantage fails.

That is the paper's through-line. A reader should understand the mechanism and why it is useful before encountering the theorem inventory, artifact audit, or numbered gaps. The mathematics and qualifications remain essential, but their job is to delimit a claim the reader already understands.

This roadmap accompanies `paper/main.reader-first.typ` and `paper/reader-first.pdf`, an alternate editorial version based on `paper-1p3b-spine` at [`f17b3be20b395cef914cd7034e8691c997164230`](https://github.com/spinozans/emender/commit/f17b3be20b395cef914cd7034e8691c997164230). It leaves the canonical manuscript, its PDF, all TMLR submission artifacts, experiments, proofs, and data untouched. It is a proposed reading experience, not a replacement scientific authority or a submission.

## What a reader should remember

1. **Mechanism:** the model repeatedly computes with its carried state. Delta correction and state nonlinearity are distinct ingredients: GDN also uses delta correction; nonlinear state is the contrast with GDN.
2. **Practicality:** many small sequential memory programs can run in parallel. The current E97 model trained through 150.8 billion Pile tokens on an eight-GPU workstation at approximately 1.3 billion parameters.
3. **Evidence:** a tightly parameter-matched GDN2-MLP control occupies the same training-loss band, while direct probes show task-dependent differences. The positive modular-quadratic result and the negative S5 transfer belong together.
4. **Interpretation:** structural expressivity, practical training, and learned capability are related questions with different evidence. None can be substituted for another.

The point is an experimentally usable architectural option and a research question about computation through memory. A proof inventory is not the opening pitch, and the measured result does not need a universal superiority claim to matter.

## Recommended reading order

### 1. Introduce a running computation

Start with an input sequence that changes an evolving state. An association can be revised, a controller can change its response, or an iterative algorithm can apply another transformation to its current answer. This explains the role of memory before discussing neural-network families.

Suggested opening:

> A sequence can require more than a record of what arrived. Its elements may be instructions that change a running calculation, measurements that alter a feedback response, or observations that revise an earlier association. In each case the current state is part of the next computation.

Then name the scientific question: can a model learn these updates and remain practical to train at language-model scale?

### 2. Explain the memory update

Use a short verbal execution trace before the full equations:

> Read what the memory currently predicts at the supplied key. Write the correction toward the supplied value. Transform the updated state with tanh, then carry that state into the next step.

Two distinctions should be clear here. Delta correction makes a write depend on the existing content. Applying tanh to the state introduces a nonlinear operation into the temporal path. Neither alone establishes that a trained model remembers indefinitely or discovers a particular algorithm.

Keep the complete E97 equations, gate definitions, normalization note, and pseudocode in the architecture section. The reader should reach them with an intuitive model of what each line does.

### 3. Explain the computational cost and the systems response

The nonlinear update generally prevents the scan used by linear-state recurrences. That cost creates the systems question. Width-axis multi-programming addresses it by running many small memories in parallel across heads and batch elements while leaving each memory's time loop serial.

Suggested transition:

> The computational choice creates a systems problem. A nonlinear update generally loses the scan structure used to parallelize linear-state recurrence. We instead preserve the sequential update and run many small memory programs side by side.

Keep the geometry, throughput measurements, memory use, and CUDA scope nearby. Do not turn practicality into a speed win: the literal 1.3 B paired geometry gives near-tied throughput, and the small-scale speedup does not reproduce at this scale.

### 4. Establish that the model trains

Introduce the E97/GDN2-MLP pair in words, then give its exact identities and recipe. The high-level result is the narrow training-loss band at matched tokens. The source tables and operational records remain the authority for the numbers.

Suggested section opening:

> The systems design earns its place if the model can be trained on a substantial language-modeling workload. We compare E97 with a closely parameter-matched GDN2-MLP control under the same long-run recipe. The result is a narrow training-loss band at matched tokens.

Make the evaluation boundary explicit once here and retain it in the limitations: the current 150 B pair has no held-out tokenizer-invariant BPB evaluation or downstream panel. The predecessor E88/M²RNN-CMA/GDN study does have held-out measurements. It is a different cohort, not an evaluation of the new pair.

### 5. Ask what the trained state can compute

The transition should be a question the bulk-loss result leaves open:

> Similar language-modeling loss can coexist with different behavior on explicit computations. We therefore test tasks whose required state updates are known.

Preserve the small-model controls, update-map intervention, and 1.3 B paired tests as distinct experiments. The positive result is higher terminal modular-quadratic accuracy and better length robustness under the recorded protocol. The important negative is that the earlier S5 ordering does not transfer to this pair. These are complementary observations, not a main result and an inconvenient footnote.

The full paper must retain the single seed, common cap and stopping rule, unequal realized steps, intermediate GDN2 performance, long-length decline, non-chance GDN2 plateau, and S5 curriculum OOM boundary. The opening summary can be brief if the detailed protocol and interpretation remain directly attached to the results.

### 6. State what the formal results establish

Lead the formal section with the theorem set used by the narrative: temporal composition and the conditioned affine state channel. Then present the finite-state tracker, representability comparisons, resource accounting, split gates, and latching results.

Suggested orientation:

> The experiments measure learned behavior. The formal results describe the update maps and the resource classes in which they are compared.

The reader needs the hypotheses before extrapolating the result. In particular, an affine fold under a fixed state-independent schedule does not make a whole GDN2-MLP stack globally linear. Counting executed tanh compositions is not a lower bound on the shallowest possible representation. The S5 lookup artifact is not a trajectory proof for the recurrent tanh cell.

Put the trust-gate implementation, recursive import count, extraction inventory, and build details after the theorem explanations. They support inspectability; they do not explain the mechanism.

### 7. Extend the question beyond language modeling

Place applications after the formal section so the reader knows what the structural comparison does and does not imply. Keep the six existing examples and their equations. Their common pattern is repeated transformation of a carried state; their distinct regimes and counterexamples matter.

A useful lead is:

> The iterated-map probes ask a question that arises well beyond language modeling: can an architecture keep applying a nonlinear transformation to its own evolving state?

Treat the application section as a research agenda. The existing EXHIBIT, ANALOGY, and OPEN labels help separate an equation's structural resemblance from a measured E97 advantage. Preserve the contractive, fixed-support, alternative-encoding, finite-precision, and error-metric exceptions. There are no measured deployments, energy savings, control guarantees, or biological-regulation results in this paper.

### 8. Close on the scientific result and next test

The conclusion should return to computation in memory, its demonstrated trainability, and the task-dependent probe outcomes. End with the experiments and formal questions that can distinguish the next hypotheses. Avoid ending the scientific argument with journal positioning or self-assessment of correctness.

Reproducibility and submission-format status belong in a separate paragraph or subsection. Keep them, but let the scientific conclusion finish its own thought first.

## Changes implemented in the alternate

- Rewrote the abstract around the memory operation, practical training, measured behavior, and then formal scope. It retains the positive and negative 1.3 B findings and identifies the outstanding evaluation boundary.
- Rewrote the introduction as a connected explanation of the mechanism, systems choice, comparison, and implications. Removed the repeated opening surveys and multiple contribution summaries. The detailed comparison to related architectures remains in the body and related work.
- Replaced the background's opening with a direct explanation of where nonlinearity occurs. Retained pangenomics as motivation while clearly distinguishing it from the experiments reported here. Removed the unsupported generalization that byte-level pangenomic processing rules out routine use and the unquantified claim about the first linear-recurrent attempts; these are not needed to establish the paper's contribution.
- Added short result-oriented openings to the language-modeling and computational-probe sections. The tables, numerical results, recipes, and associated qualifications are unchanged.
- Renamed four section headings to follow the reader's questions: computing with memory, making the updates practical, training at 1.3 B, and learning explicit computations.
- Moved formal results before the application examples. Moved theorem set H before set A so the stated temporal-composition result appears where the section says the argument starts. The theorem-set text and equations remain unchanged.
- Moved formal verification metadata to the end of the formal section.
- Moved the complete claim/evidence/scope/non-claim table from the introduction to the end of the limitations. The original table remains intact apart from its heading and a now-obsolete forward-reference sentence.
- Rewrote the conclusion around the mechanism and results. Kept the exact negative S5 finding, the held-out boundary, and O1/G3. Put the preprint-format caveat under a reproducibility and packaging heading.

This is a substantial framing and flow pass, not a line edit of every technical paragraph. The technical body is deliberately conservative. The next pass below would shorten it further without changing the evidence.

## Suggested next cuts and moves

### Consolidate repeated interpretations

The statement that loss alone does not determine capability appears in the introduction, language-modeling section, probe section, and conclusion. Each occurrence should do a different job: pose the question, report the loss result, answer with the probes, and summarize the inference. Remove additional repetitions that merely restate the slogan.

Similarly, keep a short local caveat beside each claim and a complete treatment in limitations. Repeated full lists of G1–G3 and O1 can become a precise cross-reference after the first substantive explanation. Do not remove the caveat from the result that needs it.

### Reduce the survey burden before the architecture

The background and taxonomy are useful reference material but delay the concrete mechanism. A later pass could keep the linear-state criterion, one comparator equation, and the task definitions in the main narrative, moving the longer family-by-family taxonomy to an appendix. This alternate retains that material to avoid making a second large structural change before the author reviews the new opening.

### Put protocol detail at the point of use

Long identities and provenance tables are essential for exact reproduction, but an initial results paragraph should say what was compared and what happened. Retain the tables immediately afterward. The same rule applies to the historical CMA search records: begin with their role in selecting the two models, then show the retained evidence and missing winner-series information.

### Separate the three comparison families consistently

Use E97/GDN2-MLP for the current 150 B-token pair; E88/M²RNN-CMA/GDN for the predecessor 1.3 B study; and the exact tested arms for small-scale probes. Avoid shortening all three to “the nonlinear and linear models” when the surrounding claim depends on the particular comparator. The older held-out values are retained measurements. The newer pair's missing held-out evaluation neither invalidates them nor supplies an evaluation of the newer pair.

### Make application examples cumulative

The six applications currently each restate the broad scope boundary. A future shortening pass can give that boundary once, then use each example to teach a distinct point: quantized feedback, state constraints, changing support, driven nonlinear dynamics, contraction as a countercase, and actuator saturation. Preserve the specific exceptions; cut repeated general disclaimers.

### Join predictions to the experiment that resolves them

The existing predictions and future-work sections overlap. A later revision can pair each hypothesis with its decisive comparison and falsifier, then put longer formal-development targets in an appendix. This alternate leaves both sections verbatim so no proposed experiment, falsifier, or open mathematical target disappears during the framing pass.

## Scientific boundaries retained

- Executed nonlinear composition is not a minimal circuit-depth lower bound.
- The GDN affine-fold result conditions on a state-independent parameter schedule; it is not a statement that the full neural architecture is globally linear.
- Delta-versus-raw-write separation is inside the formalized resource class. Exact finite-k non-equality is not a uniform finite-precision margin.
- Generator lookup plus external group composition is not a recurrent tanh tracker realization. O1 remains open.
- Constrained affine/reflection realizability remains open. No TC0-versus-NC1 separation is assumed.
- The latching hold theorem's decay hypothesis does not establish default persistence in the production range.
- The current 1.3 B result is single-seed and task-dependent. The modular-quadratic comparison has unequal realized training; S5 does not reproduce the expected ordering.
- The old cohort's held-out losses are descriptively close. No statistical-equivalence test is supplied, and those measurements do not evaluate the current pair.
- The applications are structural examples and hypotheses, not measured deployments.

## Build and review

From the repository root, with the Typst CLI installed:

```sh
typst compile --root . paper/main.reader-first.typ paper/reader-first.pdf
```

The alternate uses the same local figure, data, and bibliography files as the canonical paper. It does not rebuild or replace TMLR artifacts. A Python `typst` package can compile the same source when the CLI is unavailable; give it the repository root and a writable package cache. `reader-first-vs-canonical.diff` shows the proposed manuscript edits against the canonical source; it is provided for review and has not been applied to that source.

`conservation.json` records the source identities and exact-content checks. All 12 standalone display-equation blocks and eight fenced pseudocode blocks are preserved. The formal theorem blocks, complete limitations, predictions, future work, historical appendices, and bibliography call are retained verbatim. No proof or experiment has been rerun as part of this editorial work. PDF compilation and visual inspection establish document readability, not scientific replication.
