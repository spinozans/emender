# E97 current-pair editorial roadmap

## The paper's argument

E97 makes carried memory part of an ongoing computation. Each input reads the current state, writes a correction, and applies a nonlinear transformation before that state enters the next step. Spatial parallelism across small heads and batch elements makes this sequential computation practical. The current E97/GDN2-MLP 1.3 B comparison then asks whether the architectural distinction matters for what the trained models learn to compute.

The main empirical story is one study: comparable training loss, closely matched parameter counts and measured throughput, an E97 advantage on an iterated nonlinear map, and a strong GDN2-MLP result on S5 group-state tracking. The mathematical mechanism makes these task choices intelligible. Their interpretation concerns the computations learned under the recorded protocols, not a universal ranking of full-network expressivity.

This revision supersedes the first reader-first draft's cross-cohort framing. Older benchmark results and their extended interpretations have been removed from the alternate paper, including its appendix. A short development register retains process history and source pointers. See `EXCLUDED_HISTORICAL_EVIDENCE.md` for the evidence-scope decision. The canonical paper, original results, and proof sources remain untouched.

## Recommended flow

### 1. Explain the computation before the classification

Open with a sequence whose observations change a running calculation or revise an association. Explain what memory does in the next computation, rather than starting with an inventory of architectural families or proof qualifications.

Suggested opening:

> A sequence can require more than a record of what arrived. Its elements may be instructions that change a running calculation, measurements that alter a feedback response, or observations that revise an earlier association. In each case the current state is part of the next computation.

Then explain the E97 operation in execution order: read the current prediction, write its correction, transform the updated state, and carry it forward. Both compared models use delta correction. The mechanism of interest is the nonlinear operation in E97's temporal state path, with decay placement and other trained-stack differences stated explicitly.

### 2. Make the systems result concrete

A serial recurrent update is useful only if it can be trained at a meaningful scale. Explain width-axis multi-programming as many small memory programs running side by side, while each one processes time sequentially. The current E97 configuration has 216 heads per layer, 11 layers, and batch 4: 9,504 recurrent programs per token, each with a 32-by-32 state tile.

Use the actual pair's measurements. Parameters are 1,286,589,072 versus 1,286,713,448, a 0.0097% difference. The seven-GPU DDP preflight at matched batch 4 reports 3,211 versus 3,290 tokens/s per GPU and 28,942 versus 35,715 MB peak memory. E97's larger feasible batch follows from lower memory use. These are comparable throughput measurements, not identical speed, a per-token speed win, or a measured matched-FLOP result. Distinguish this preflight from the eight-GPU local-DiLoCo long runs.

### 3. Establish comparable language-model training

Introduce the E97/GDN2-MLP pair, the shared data/tokenizer/chunk/optimizer/DiLoCo recipe, and the separately selected geometries and learning rates. Keep the exact run identities and matched-token table.

Suggested result paragraph:

> The two current models reach comparable training loss after each family's geometry and learning rate have been selected by search. The small, window-dependent difference does not establish a loss ordering or statistical equivalence. It establishes that the nonlinear recurrence trains competitively under this substantial shared workload.

The 150.79 B-token comparison is training-log evidence. Held-out language-model and downstream evaluation of these checkpoints remain pending. The earlier cohort's measurements must not be used to fill that gap.

### 4. Connect the iterated-map result to the hypothesis

The task repeatedly applies a nonlinear transformation to a running value. That is the computation the architecture was designed to place in memory. On the current pair, E97's higher terminal accuracy and greater length robustness support the proposed computational bias.

Keep the actual results: 99.96% versus 76.09% at trained length, and 76.07% versus 53.36% at eight times that length. Keep the full table and protocol alongside them. The single seed, common cap and plateau rule, unequal realized steps, better intermediate GDN2 checkpoint, non-chance long-length plateau, and declining E97 accuracy remain material qualifications. The experiment supports the mechanism hypothesis without uniquely isolating its cause or proving exact algorithm recovery.

### 5. Give S5 its own computational interpretation

S5 tests composition of finite-group actions. The current GDN2-MLP result is strong: it leads the reported long-horizon comparison at every evaluated length. Both curves decline with length; the attempted length-512 GDN2 curriculum encountered OOM after one completed optimizer step.

Describe this directly as learned group-state tracking under the recorded protocol. Do not call it a failed replication of an older E88 or 8 M experiment. Do not reduce the paper to “mixed results.” The meaningful contrast is between the computations tested and the learning efficiency observed within each task.

### 6. State the relevant mathematics after the empirical question is clear

Lead with nonlinear temporal composition and the conditioned affine state fold. Retain the adjacent split-gated-core identities, S5 tracker and routing statements, and conditional latching results with their hypotheses and theorem names.

The key boundary is structural: a fixed, state-independent GDN parameter schedule gives an affine fold in initial state. The full GDN2-MLP stack contains nonlinear components. Counting executed tanh compositions does not prove a minimal-depth lower bound; the routing theorem does not rule out affine group actions. O1 and G1–G3 remain open in their stated senses. No empirical gap should be promoted into a full-stack impossibility result.

Unrelated raw-write comparisons, predecessor resource counts, and interpretations based on older probes have been removed from this focused manuscript. The full proof project remains in the repository.

### 7. End with utility and decisive next tests

The applications section illustrates why computation through state matters for feedback, iterative optimization, and driven dynamics. Its examples remain structural hypotheses rather than measured deployments.

The next tests should arise from the current pair: additional seeds, common realized budgets and checkpoint-selection controls, a current-scale state-nonlinearity ablation, held-out language-model evaluation, and matched scaling. This keeps the research program broad without making older exploratory results carry the present argument.

## What was cut or moved, and why

- Removed older E88/GDN1 language-modeling, downstream, S5, and related benchmark tables and interpretations from the alternate. The author no longer relies on them as evidence for this paper.
- Removed the small-scale probe sections, broader architecture-search leaderboards, predecessor occupancy measurements, and “micro and mega” synthesis. They are distinct experiments, not arms or replications of the current pair.
- Replaced the long historical appendix with a short development/source register. It records evolution and where to inspect the original work, without extracting benchmark meaning from it.
- Replaced the broad historical predictions with tests of the current model pair and relevant formal targets.
- Kept current-pair numeric tables and their data-driven computations. The reader-first table module now loads only current-pair result data.
- Replaced the old claim inventory with a connected account of mechanism, practicality, learned computation, and formal scope.

## Files and validation

Base source: `paper-1p3b-spine` at `f17b3be20b395cef914cd7034e8691c997164230`.

- Alternate manuscript: `paper/main.reader-first.typ`
- Rendered alternate: `paper/reader-first.pdf`
- Current-pair table module: `paper/figures/expressivity_1p3b.reader-first.typ`
- Evidence-scope note: `paper/editorial/EXCLUDED_HISTORICAL_EVIDENCE.md`
- Review diff and conservation record: the other files in `paper/editorial/`

Build from the repository root:

```sh
typst compile --root . paper/main.reader-first.typ paper/reader-first.pdf
```

The conservation record now distinguishes retained current-pair equations/results from deliberately excluded historical material. It no longer claims that every old paragraph or display remains in the alternate. Proof files, result JSONs, canonical source, bibliography, and existing submission artifacts are unchanged. Rendering and editorial checks are not reruns of training experiments or Lean proofs.
