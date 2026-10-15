# Emender reader first transfer review

Date: 2026-10-15. Companion: [Frontier training integration plan](FRONTIER_TRAINING_INTEGRATION.md).

## Recommendation

Use the reader-first manuscript as the organizing text for the next canonical revision. Bring across substantive correctness fixes from the canonical manuscript and add a properly developed Frontier training result. Do not attempt another sentence-by-sentence cleanup of the current 65-page structure.

The reader-first version explains why the computation matters, how the memory update works, how it becomes trainable, and what the trained models actually learn. The canonical version incorporates some of those paragraphs but retains the old evidence hierarchy, theorem inventory, historical comparisons, and repeated scope discussion around them. The problem is structural, rather than a handful of unfortunate sentences.

This is a proposed editorial plan. No manuscript or release artifact was changed for this review.

## Versions reviewed

- Canonical paper: `paper/main.typ` and `paper/current.pdf` at `4b62d192b936fdb80ca75e1e842da3fbd8d91034`. Its PDF is 65 pages. Verified that the later CI merge `baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e` changes no file under `paper/` relative to this commit.
- Reader-first alternative: `paper/main.reader-first.typ` and `paper/reader-first.pdf` at `fba15ce50641715b764948e7a0e9e69985eff2ed`. Its PDF is 30 pages.
- Editorial decisions: `paper/editorial/READER_FIRST_ROADMAP.md` and `paper/editorial/EXCLUDED_HISTORICAL_EVIDENCE.md` on the reader-first branch.
- Frontier evidence: the companion integration plan verifies the accepted-token anchors, checkpoint identity, loss definition, corpus transition, and available systems data. The complete raw Frontier loss trajectory remains to be recovered.

## What failed to transfer

### The abstract became an inventory again

The roadmap says: “The abstract carries motivation and findings. The methods carry seeds, stopping rules, evaluation status, and checkpoint selection.”

The canonical abstract instead inserts: “This is a single-seed, common-cap comparison with unequal realized training over the tested lengths.” It then adds “matched-signature delta-versus-raw-write separation,” “minimal circuit-depth lower bounds,” “a full-stack impossibility,” and “constrained affine-channel realizability.” These are several different research questions arriving before the reader has understood the central contribution.

Retain the measured accuracies, the GDN2 group-tracking result, and the precise conditional affine-state statement. Give the actual experiment protocol beside the experiment. Delete the abstract's catalog of unproved propositions. Removing imagined counterclaims does not require removing real assumptions.

There is also a small sign of incomplete transfer: “Emender explores a complementary path” remains after the sentence naming textual scratchpads was removed. Either establish the other path or simply state Emender's approach directly.

### The introduction explains the implementation before the operation

The reader-first introduction is about 90 source lines; the canonical introduction is about 350. In the alternative, the memory operation precedes width-axis multi-programming. Canonical lines 130–137 explain E97's serial time loop and parallel heads before lines 190–201 explain what E97 does to state.

The canonical first empirical paragraph also accumulates the workstation experiment, estimated bpb, averaging-window behavior, larger-scale uncertainty, old 8 M results, several Lean claims, O1, and a blanket-impossibility discussion. It is followed by another empirical paragraph and another formal-summary paragraph. The later contributions list and “Claims, evidence, and scope” table repeat much of that material.

Replace the introduction as a connected sequence: the human need; the state operation; parallel execution; the current training evidence including Frontier; what the two tasks reveal; the precise mathematical question. Remove the claim matrix and second contribution inventory.

### The evidence-scope decision was reversed

The reader-first evidence note states: “The author has withdrawn confidence in using the older E88/GDN1 empirical comparisons and their associated interpretations as evidence for this paper.” It explicitly excludes them from the appendix too, while preserving the source records.

Canonical line 2010 introduces the current E97/GDN2 computational probes. Its next subsection is “State-count headroom at 8 M parameters,” followed by the old permutation results and reconstructed seed figure. The appendix begins with the reader-first source-register paragraph, including “The empirical study in this paper is the current E97/GDN2 pair,” then continues into predecessor-cohort systems, downstream, language-modeling, and state-tracking tables.

Remove those historical benchmarks and their interpretations from the paper. Retain a brief development register with repository pointers. This implements the author's evidence decision; it does not claim that every historical number has been proven false. Add Frontier on its verified E97 merits rather than treating it as another historical comparison or silently conflating it with the paired experiment.

### The section order and headings were left behind

The alternative moves from “Computing with a nonlinear memory” to “Making sequential updates practical,” “Language modeling at 1.3 billion parameters,” and “What the current models learn to compute.” Formal results follow the empirical question under “What the update rules establish.”

Canonical keeps “Architecture,” then a long “Formal Results” section, before systems and either empirical result. Its most relevant temporal-composition statement appears late as theorem set H. The reader encounters a taxonomy of proofs before knowing which observation they explain.

Use the alternative's order and descriptive headings. Begin the mathematical discussion with the state operations used in the present comparison. Keep theorem names and their hypotheses in that section; the full proof project can remain independently available.

### The applications lost their human route into the research program

The reader-first applications start with feedback and movement, then signal processing, iterative optimization, conversation and perception, and biological dynamics. The canonical opener borrows the balance and motion examples but the sections still begin with sigma–delta conversion. Feedback arrives last, and the conversation/perception and broader biological motivation did not transfer.

Restore the reader-first progression. For each example, identify the evolving state, the nonlinear operation, and a useful measurement. Preserve the distinction between an application motivating future work and a demonstrated application result. State that distinction through the actual proposed experiment, rather than repeating labels and caveats throughout the section.

### The conclusion was copied and then interrupted

The alternative concludes: “Emender makes recurrent memory a site of ongoing computation.” It connects practical training, the iterated-map result, the GDN2 group-tracking result, and the research program in three paragraphs.

Canonical inserts a training-log bpb qualification into paragraph one, then adds a separate paragraph beginning “The modular-quadratic comparison uses one seed, a common cap, unequal realized training...” followed by O1 and G3. A further applications sentence repeats the final paragraph. The conclusion also carries predecessor reproducibility and reconstructed-seed discussion.

Restore the three-paragraph conclusion, update the training sentence with verified Frontier evidence, and put reproducibility in a dedicated section. Explain the next decisive experiment once in Open questions.

## Passage plan

1. **Abstract, canonical lines 23–50.** Replace the whole abstract with the candidate below. Keep one clear mechanism, the training result, both computational findings, and one precise formal statement.
2. **Introduction, lines 109–459.** Use reader-first lines 109–198 as the base. Insert one concise Frontier sentence in the practical-training paragraph after its provenance is checked. Preserve the shared-MLP explanation and name GDN2 naturally after defining the recorded GDN2-MLP arm once.
3. **Section order and formal discussion.** Move the core formal discussion after the current-pair computational results. Lead with nonlinear temporal composition and the conditional affine fold. Keep relevant split-gate identities, tracker definitions, routing hypotheses, and conditional latching in their natural places. Review later canonical factual corrections against each retained theorem paragraph rather than reverting them wholesale.
4. **Language modeling, lines 1698–2005.** Keep the exact pair identities, matched-token statistics, search provenance, and actual-geometry measurements. Remove cross-family search-scale and predecessor comparisons as support for the present study. Add “Extended training on Frontier” after the matched-token result, before detailed selection provenance. Give the trajectory its own figure if the underlying records support one, and report what the training data establish.
5. **Computational results, lines 2006–2434.** Replace the historical-study interlude with the reader-first current-pair presentation. Preserve the full current tables and local protocol: seed, cap, plateau stopping, unequal completed steps, terminal checkpoint selection, better intermediate GDN2 accuracy, length-dependent degradation, and the failed length-512 GDN2 curriculum. Describe each task's computation and observed ordering directly.
6. **Applications, lines 2435–2647.** Restore feedback/movement, signal processing, optimization, conversation/perception, and biological dynamics. Retain concrete equations and questions; eliminate repeated abstract representation/impossibility discussions already handled in the mathematical section.
7. **Open questions and conclusion, lines 2768–3188.** Use the alternative's concise learning/evaluation, recurrent-realization, and scale/implementation organization. Consolidate duplicate predictions and future-work material into those questions. End with what the architecture enables and which current-pair experiment would most strengthen the explanation.
8. **Appendix, line 3189 onward.** Keep the short source register. Remove historical benchmark tables and conclusions from this manuscript while preserving repository sources. Add reproducibility details needed for the verified Frontier result alongside the current pair's records.

A useful final reading test is whether a new reader can explain the memory operation, the reason it is trainable, and what each experiment found before encountering theorem labels or audit identifiers.

## Proposed abstract

Fast interaction and feedback require models to update what they know as new inputs arrive. Emender performs this computation in bounded recurrent memory: each input corrects the memory's prediction and applies a nonlinear transformation to the state carried forward. Many small memory heads execute in parallel, making the sequential update practical to train. At 1.3 billion parameters, E97 trains through 150.8 billion Pile tokens on one eight-GPU workstation, reaching training loss comparable to a closely parameter-matched Gated DeltaNet-2 control. An extended Frontier run reaches 513.0 billion cumulative tokens, with its final segment trained on CommaPile. Fine-tuning the workstation pair on an iterated nonlinear map gives E97 99.96% accuracy versus GDN2's 76.09% at the trained length, and 76.07% versus 53.36% at eight times that length. GDN2 leads on long-horizon S5 group-state tracking. Formal results characterize E97's repeated nonlinear state transformation and GDN's affine state composition under a fixed, state-independent parameter schedule. Together, these findings establish a practical route to training nonlinear recurrence and support further study of computation through memory in feedback, optimization, and driven dynamics.

The Frontier investigation has now verified the continuation lineage and the change of corpus to CommaPile. This wording identifies the cumulative training extent without treating the endpoint as a same-corpus loss improvement. The remaining abstract findings come from the retained current-pair study.

## Frontier insertion language

The following lead incorporates the separately verified continuation and corpus identity:

“The same 1.3-billion-parameter E97-MLP architecture was continued on Frontier from the 150.79-billion-token checkpoint. The retained continuation checkpoint reaches 513.01 billion cumulative accepted training tokens, with its final segment using CommaPile mainmix. Its final recorded training-loss average is 2.2798 nats/token.”

The checked Frontier configuration uses 256 nodes and 2,048 ranks, batch 4, context 2,048, and pure-average DiLoCo with K = 40. The loss is the rank-0 mean over the last 100 logged windows, representing 1,000 local updates; it is recorded before the optimizer enters evaluation mode. It should not be described as a fresh evaluation of the final checkpoint. Since the workstation experiment uses The Pile and the final continuation segment uses CommaPile, subtracting their final losses would not measure same-distribution improvement.

The investigation verified cumulative-token anchors at 150.794, 199.615, 400.942, and 513.014 billion tokens. Those anchors can support an honest continuation timeline. Intermediate segment loss logs remain unlocated, so a continuous Frontier loss curve should wait for the underlying measurements. The old `docs/experiments/figures/e97_diloco_loss_curve_20260623.png` is a workstation curve and must not be presented as Frontier evidence. The integration note should bind these facts to the fetched records, execution code, and checkpoint manifests.

## Sources

- [Canonical manuscript at the reviewed commit](https://github.com/spinozans/emender/blob/4b62d192b936fdb80ca75e1e842da3fbd8d91034/paper/main.typ)
- [Reader-first manuscript](https://github.com/spinozans/emender/blob/fba15ce50641715b764948e7a0e9e69985eff2ed/paper/main.reader-first.typ)
- [Reader-first roadmap](https://github.com/spinozans/emender/blob/fba15ce50641715b764948e7a0e9e69985eff2ed/paper/editorial/READER_FIRST_ROADMAP.md)
- [Historical evidence decision](https://github.com/spinozans/emender/blob/fba15ce50641715b764948e7a0e9e69985eff2ed/paper/editorial/EXCLUDED_HISTORICAL_EVIDENCE.md)
