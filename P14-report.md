# P14 — editorial de-hedging pass

## Contract and scope

Executed section by section in manuscript order, starting directly at P13’s final
commit `a0d1654c` on `paper-1p3b-spine` (HEAD already equaled the requested
base; no unrelated commit or rebase conflict). **Hedge removed, scope preserved.**
There are **55 removal/reword sites: 39 rhetoric and 16 scope-preserving rewords**.
Counts are sites, not individual deleted words; multiple deletions in one span
are fully shown in that row. **32 kept judgment-call entries** are listed below.

No section was merged, split or reordered. Math, numbers, tables, figures,
pseudocode, bibliography and Lean/literature citations were preserved.
The hypervolu mirror was not touched. No training, evaluation, Python, Slurm,
native build or WorksGood machinery was run; this was a manuscript-only pass
on lambda01. Resilient-DiLoCo requirement IDs are not applicable.

## Section-by-section disposition

| Section | Rhetoric | Scope-preserving reword | Total sites |
|---|---:|---:|---:|
| Abstract | 1 | 0 | 1 |
| 1 Introduction | 4 | 4 | 8 |
| 2 Background | 3 | 0 | 3 |
| 3 Architecture | 1 | 2 | 3 |
| 4 Systems | 1 | 3 | 4 |
| 5 Language Modeling | 5 | 1 | 6 |
| 6 Expressivity | 2 | 2 | 4 |
| 7 Applications | 2 | 1 | 3 |
| 8 Formal Results | 10 | 2 | 12 |
| 9 Related Work | 3 | 0 | 3 |
| 10 Limitations | 0 | 0 | 0 |
| 11 Conclusion | 1 | 0 | 1 |
| 12 Testable Predictions | 0 | 0 | 0 |
| 13 Future Work | 1 | 0 | 1 |
| Appendix A Lineage | 1 | 0 | 1 |
| Appendix B Predecessor cohort | 4 | 1 | 5 |
| Appendix C Held-out bpb | 0 | 0 | 0 |

Zero-edit sections were read and retained for scope, not skipped. Figure/table
caption rhetoric (including the taxonomy’s “not a strawman” and downstream
“null, reported as a null”) was left untouched under the exclusion contract.

## Full removal table

Old line numbers refer to `a0d1654c:paper/main.typ`; new line numbers refer to
the delivered source. Each row shows the complete replaced span, including
case/punctuation and line breaks. The exact zero-context unified source diff is committed
at `paper/tmlr/validation/dehedging-revision.diff`; the table’s 55 replacements
replay that entire diff with no unlisted changes.

| ID | Section | Old → new line | Removed/replaced span | Replacement | Category | One-line reason |
|---:|---|---|---|---|---|---|
| 1 | Abstract | 40 → 40 | <code>The negative transfer is explicit:</code> | <code>The negative transfer is:</code> | rhetoric | Remove commentary on how explicitly the negative is presented; retain the verbatim S5 boundary. |
| 2 | 1 Introduction | 134 → 134 | <code>The field's response has been to route around<br>the obstruction rather than meet it: parallelize the time axis by other<br>means,</code> | <code>Alternative routes parallelize the time axis by other<br>means,</code> | rhetoric | Remove rhetorical avoidance framing; retain Newton and attention alternatives. |
| 3 | 1 Introduction | 147 → 146 | <code>set aside as the expensive case to avoid rather than the case to<br>measure.</code> | <code>regarded as too expensive to<br>measure.</code> | rhetoric | Remove the avoid-versus-measure contrast; retain the historical reason for the untested setting. |
| 4 | 1 Introduction | 168 → 167 | <code>parity survives at 7 B and beyond is named openly as a limitation in</code> | <code>parity survives at 7 B and beyond remains a limitation in</code> | rhetoric | Remove self-praise about disclosure; larger-scale uncertainty remains. |
| 5 | 1 Introduction | 232 → 231 | <code>The two responses are complementary, not<br>rival:</code> | <code>The two responses are complementary:</code> | scope-preserving-reword | State the relationship positively; retain the hybrid/update-rule explanation. |
| 6 | 1 Introduction | 255 → 253 | <code>In both cases the<br>pure-recurrent setting was set aside rather than filled.</code> | <code>These routes use attention or time-axis parallelization.</code> | scope-preserving-reword | Replace rhetorical detour framing with the stated attention/time-parallel methods; the following comparison remains pure recurrent. |
| 7 | 1 Introduction | 328 → 325 | <code>The<br>  contribution is the architectural *option* plus the *mechanism*,<br>  not a training-speed win — bulk loss is a convergent null across<br>  the tuned families.</code> | <code>The<br>  contribution is the architectural *option* plus the *mechanism*;<br>  training speed supplies no win, and bulk loss is a convergent null across<br>  the tuned families.</code> | scope-preserving-reword | Retain the speed non-claim while removing contribution-is-not-X framing. |
| 8 | 1 Introduction | 377 → 374 | <code>To make the scope legible up front, the table below<br>states</code> | <code>The table below<br>states</code> | rhetoric | Remove commentary on the paper’s conduct; retain every claim/scope/non-claim row. |
| 9 | 1 Introduction | 454 → 451 | <code>One scope note belongs with M²RNN specifically. We do not claim that<br>M²RNN lacked width or batch parallelism, or missed that a nonlinear<br>recurrence is serial in time; the distinction is that its published<br>scaling route emphasizes *hybrid* use,</code> | <code>M²RNN uses width and batch parallelism for a nonlinear<br>recurrence that is serial in time. Its published<br>scaling route emphasizes *hybrid* use,</code> | scope-preserving-reword | Replace defensive framing with the acknowledged systems premise; keep the hypothesis and non-isolation boundary. |
| 10 | 2 Background | 649 → 645 | <code>a<br>single design grid rather than a list of unrelated tricks. Reading a<br>head by the *dynamics* of its per-token state map — not by the<br>experiment that first produced it — fixes it as one point in a<br>two-axis grid:</code> | <code>a<br>single design grid. The *dynamics* of a head’s per-token state map<br>locate it in a two-axis grid:</code> | rhetoric | Remove rhetorical contrasts with tricks and discovery history; preserve both taxonomy axes. |
| 11 | 2 Background | 685 → 679 | <code>included only<br>once both axes have earned it.</code> | <code>included as<br>the combination of the two axes.</code> | rhetoric | Replace personified justification with the taxonomy construction; future-work status remains. |
| 12 | 2 Background | 691 → 685 | <code>This grid makes the relationship to the linear-recurrent baseline<br>exact rather than rhetorical.</code> | <code>The adjacent split-gated cores locate the linear-recurrent baseline<br>in this grid.</code> | rhetoric | Remove self-evaluation of exactness; the following block retains all decay-placement and inclusion boundaries. |
| 13 | 3 Architecture | 875 → 869 | <code>repeats it across a residual stack rather than changing the per-head<br>equation.</code> | <code>repeats the same per-head equation across a residual stack.</code> | scope-preserving-reword | State equation reuse positively rather than denying a change. |
| 14 | 3 Architecture | 1063 → 1056 | <code>Three remaining choices are<br>deliberate omissions rather than additions: there is</code> | <code>Three remaining choices are:<br>there is</code> | rhetoric | Remove meta-commentary about deliberate design conduct; enumerate the same omissions. |
| 15 | 3 Architecture | 1073 → 1066 | <code>The 1.3 B stack's shape is not arbitrary. Ingredient (d) sets its character —</code> | <code>Ingredient (d) sets the 1.3 B stack’s shape —</code> | scope-preserving-reword | Use the positive design explanation instead of denying arbitrariness; search-selected values remain. |
| 16 | 4 Systems | 1167 → 1160 | <code>Throughput comes from width, not from time.</code> | <code>Width-axis multi-programming supplies throughput while each head’s<br>time loop remains serial.</code> | scope-preserving-reword | Replace not-X framing with the same axis and serial-time constraint. |
| 17 | 4 Systems | 1219 → 1213 | <code>That the card stays busy is a claim about occupancy, and it is measured<br>directly. Driving</code> | <code>Driving</code> | rhetoric | Remove telling-language preceding the occupancy measurement; keep the entire measured protocol and statistics. |
| 18 | 4 Systems | 1243 → 1236 | <code>This figure is occupancy, not peak arithmetic, and the two should not<br>be run together. Model-FLOPs utilization (MFU)</code> | <code>Occupancy and peak-arithmetic utilization measure different quantities.<br>Model-FLOPs utilization (MFU)</code> | scope-preserving-reword | Keep the metric distinction without instructing the reader how to read. |
| 19 | 4 Systems | 1265 → 1258 | <code>so the standing<br>is parity-class, not superiority.</code> | <code>so the standing<br>is parity-class.</code> | scope-preserving-reword | Drop the redundant opposite of parity; the preceding slower-kernel comparison remains. |
| 20 | 5 Language Modeling | 1343 → 1336 | <code>Holding the channel-mixer (MLP) fixed on both arms is what<br>makes this a comparison of token mixers — the recurrence — rather than<br>of everything at once.</code> | <code>Holding the channel-mixer (MLP) fixed on both arms<br>makes this a comparison of token mixers — the recurrence.</code> | rhetoric | Remove the rhetorical everything-at-once contrast; geometry and learning-rate differences remain in the table. |
| 21 | 5 Language Modeling | 1475 → 1467 | <code>Two boundaries on this measurement are stated plainly. First, it is a</code> | <code>Two boundaries apply to this measurement. First, it is a</code> | rhetoric | Remove self-evaluation of disclosure; keep held-out and single-seed limitations. |
| 22 | 5 Language Modeling | 1499 → 1491 | <code>but the statement is descriptive closeness with the reported<br>ordering, not a statistically established win or equivalence:</code> | <code>but these values show descriptive closeness with the reported<br>ordering; statistical superiority or equivalence remains unestablished:</code> | scope-preserving-reword | Replace the statement-is-not construction while retaining the exact statistical boundary. |
| 23 | 5 Language Modeling | 1507 → 1499 | <code>a reader must be able to verify that<br>GDN2-MLP was optimizer-hardened, not strawmanned. The full provenance<br>chain is as follows.</code> | <code>the search protocol and records must establish<br>GDN2-MLP’s optimizer hardening. The provenance<br>chain is as follows.</code> | rhetoric | Remove defensive strawman framing; retain the baseline-hardening evidence requirement. |
| 24 | 5 Language Modeling | 1638 → 1630 | <code>Once sensibly tuned,</code> | <code>Once tuned,</code> | rhetoric | Remove decorative evaluation of tuning quality; same tuned-family condition. |
| 25 | 5 Language Modeling | 1640 → 1632 | <code>— which is why the 1.3 B loss null is expected rather than<br>surprising, and why the paper's positive claims live in</code> | <code>— consistent with the 1.3 B loss null. Capability and formal results appear in</code> | rhetoric | Remove expected-versus-surprising rhetoric and commentary on the paper’s claims; retain the null and section pointers. |
| 26 | 6 Expressivity | 1724 → 1715 | <code>selection-criterion* asymmetry, stated here in<br>full.</code> | <code>selection-criterion* asymmetry.</code> | rhetoric | Remove commentary on completeness of disclosure; keep the selection asymmetry. |
| 27 | 6 Expressivity | 1842 → 1832 | <code>To verify that the $S_5$ result is not a single-task artefact we run a<br>six-task canonical sweep covering</code> | <code>Alongside the $S_5$ probe we run a<br>six-task canonical sweep covering</code> | rhetoric | Remove telling-language about validation intent; keep every task, budget, result and counterexample. |
| 28 | 6 Expressivity | 2048 → 2038 | <code>GDN2-MLP approaches a roughly 50% plateau at the<br>longest lengths, explicitly *not* the declared</code> | <code>GDN2-MLP approaches a roughly 50% plateau at the<br>longest lengths, above the declared</code> | scope-preserving-reword | Express the non-random plateau positively; the baseline value and limitation remain. |
| 29 | 6 Expressivity | 2093 → 2083 | <code>This hardware limitation and the<br>single seed remain part of the result, not hidden protocol changes.</code> | <code>This hardware limitation and the<br>single seed constrain the result.</code> | scope-preserving-reword | Keep the OOM and seed boundaries; remove defensive hidden-changes language. |
| 30 | 7 Applications | 2155 → 2145 | <code>The depth-duality gives a precise lens, but not an application-level<br>lower bound.</code> | <code>The depth-duality characterizes executed state-channel composition;<br>application-level lower bounds remain open.</code> | scope-preserving-reword | State the theorem role positively and keep the lower-bound gap. |
| 31 | 7 Applications | 2292 → 2282 | <code>The control is the important countercase:</code> | <code>The control is a countercase:</code> | rhetoric | Remove decorative importance; contraction and the absent depth wall remain. |
| 32 | 7 Applications | 2334 → 2324 | <code>local feedback and correction rather than a large reasoning<br>model per sensor.</code> | <code>local feedback and<br>correction.</code> | rhetoric | Remove the rhetorical contrast with a reasoning model; deployment and energy claims still require measurements. |
| 33 | 8 Formal Results | 2343 → 2333 | <code>The claims in this section are checked by a machine rather than argued<br>on paper. A trusted Lean 4</code> | <code>A trusted Lean 4</code> | rhetoric | Remove commentary contrasting machine checks with paper argument; retain the complete trust description. |
| 34 | 8 Formal Results | 2346 → 2335 | <code>Trust rests on what those proofs<br>are *not* allowed to assume. The import closure of the</code> | <code>The import closure of the</code> | rhetoric | Remove telling-language before the explicit trust restrictions; retain every forbidden construct. |
| 35 | 8 Formal Results | 2376 → 2364 | <code>*What to take from set A: at fixed width and precision the Emender is a</code> | <code>*At fixed width and precision the Emender is a</code> | rhetoric | Remove reader-direction prefix; retain the finite-state scope summary. |
| 36 | 8 Formal Results | 2443 → 2431 | <code>*What to take from set C: in a single step the delta-correcting write</code> | <code>*In a single step the delta-correcting write</code> | rhetoric | Remove reader-direction prefix; retain the one-step matched-shape abstract-domain summary. |
| 37 | 8 Formal Results | 2469 → 2457 | <code>*Positive embedding.* The separation is about resources, not raw<br>  capability. *Given* the extra read-then-delta resource,</code> | <code>*Positive embedding.* The separation concerns the specified resource<br>  class. *Given* the extra read-then-delta resource,</code> | scope-preserving-reword | Replace not-raw-capability framing with the resource-class scope; embedding and missing-resource condition remain. |
| 38 | 8 Formal Results | 2477 → 2465 | <code>*What to take from set C′: the one-step gap does not average out under<br>composition — it persists at every length on a constructed input.*</code> | <code>*The one-step gap persists at every length under<br>composition on a constructed input.*</code> | scope-preserving-reword | Drop reader-direction and redundant negation; exact finite-length witness scope remains. |
| 39 | 8 Formal Results | 2518 → 2506 | <code>*What to take from this: the clean k-step result is the ceiling reachable</code> | <code>*The k-step result is the ceiling reachable</code> | rhetoric | Remove reader-direction and decorative clean; retain the direct-composition limit and open target. |
| 40 | 8 Formal Results | 2523 → 2511 | <code>The k-step separation above is the *clean ceiling reachable from the</code> | <code>The k-step separation above is the *ceiling reachable from the</code> | rhetoric | Remove decorative clean; keep the direct-composition scope. |
| 41 | 8 Formal Results | 2582 → 2570 | <code>*What to take from set D: the updates share a quadratic envelope in a</code> | <code>*The updates share a quadratic envelope in a</code> | rhetoric | Remove reader-direction; keep the synthetic-cost and unverified-FLOP boundary. |
| 42 | 8 Formal Results | 2614 → 2602 | <code>*What to take from set E: a simple signature-field predicate is shared</code> | <code>*A signature-field predicate is shared</code> | rhetoric | Remove reader-direction and decorative simple; retain proxy and concurrency-certificate boundary. |
| 43 | 8 Formal Results | 2631 → 2619 | <code>*What to take from set G: over abstract real-valued gate/key/state</code> | <code>*Over abstract real-valued gate/key/state</code> | rhetoric | Remove reader-direction; retain abstract-domain and production-inclusion qualifications. |
| 44 | 8 Formal Results | 2689 → 2677 | <code>*What to take from set F: conditional slot-level saturation, hold,</code> | <code>*Set F establishes conditional slot-level saturation, hold,</code> | rhetoric | Replace reader-direction with a declarative summary; retain the production-default non-claim. |
| 45 | 9 Related Work | 2864 → 2852 | <code>design (#ref(&lt;sec:formal&gt;), set G) rather than by coincidence.</code> | <code>design (#ref(&lt;sec:formal&gt;), set G).</code> | rhetoric | Remove the rhetorical coincidence contrast; retain the adjacent-core relation and formal pointer. |
| 46 | 9 Related Work | 2886 → 2874 | <code>is therefore not a<br>strawman: it is the same raw-write family after CMA-ES has moved it into</code> | <code>is the same raw-write family after CMA-ES has moved it into</code> | rhetoric | Remove defensive strawman framing; retain stable pure-recurrent configuration and all confound qualifications. |
| 47 | 9 Related Work | 2912 → 2899 | <code>included here as a peer, with the note that its<br>nonlinear-block share is small.</code> | <code>included here as a peer with a small<br>nonlinear-block share.</code> | rhetoric | Remove disclosure meta-language; retain the small nonlinear share. |
| 48 | 11 Conclusion | 3088 → 3075 | <code>The negative<br>is equally load-bearing:</code> | <code>The negative<br>transfer is:</code> | rhetoric | Remove commentary on argumentative importance; retain verbatim S5 failure and GDN2 ordering. |
| 49 | 13 Future Work | 3251 → 3238 | <code>It is the sharpest single experiment for fixing where<br>the Emender-versus-raw-write gap actually sits, and</code> | <code>This experiment would locate<br>the Emender-versus-raw-write gap, and</code> | rhetoric | Remove superlative ranking and decorative actually; retain proposed test and conditional-versus-family scope. |
| 50 | Appendix A Lineage | 3342 → 3329 | <code>It is here for reproducibility of<br>the design history, so the choices in #ref(&lt;sec:arch&gt;) read as the residue of<br>measurements rather than as bare assertions.</code> | <code>The design history documents the measurements behind<br>the choices in #ref(&lt;sec:arch&gt;).</code> | rhetoric | Replace paper-conduct and bare-assertion contrast with the positive provenance function. |
| 51 | Appendix B Predecessor cohort | 3614 → 3600 | <code>Critically, the train-loss ordering does *not* reproduce held-out.</code> | <code>The train-loss ordering does *not* reproduce held-out.</code> | rhetoric | Remove decorative critically; preserve the ordering reversal. |
| 52 | Appendix B Predecessor cohort | 3661 → 3647 | <code>where the architectures<br>genuinely come apart.</code> | <code>where the architectures<br>come apart.</code> | rhetoric | Remove decorative genuinely; same predecessor-specific comparison. |
| 53 | Appendix B Predecessor cohort | 3788 → 3774 | <code>Two facts must<br>be kept apart. First,</code> | <code>Trained-length fit and length extrapolation differ. First,</code> | rhetoric | Replace reader instruction with the positive metric distinction; preserve both observed outcomes and impossibility boundaries. |
| 54 | Appendix B Predecessor cohort | 3853 → 3838 | <code>We therefore<br>do *not* claim a pure-expressivity wall for either raw-write or<br>linear GDN; the statement here is the matched-FLOP efficiency one:</code> | <code>These observations support a matched-FLOP efficiency reading;<br>a pure-expressivity wall for either raw-write or linear GDN remains<br>unestablished:</code> | scope-preserving-reword | Replace defensive author-claim framing while retaining the expressivity non-claim and budget conditions. |
| 55 | Appendix B Predecessor cohort | 3864 → 3849 | <code>every length; as noted above, the delta update too</code> | <code>every length; the delta update too</code> | rhetoric | Remove recap meta-language; retain the delta plateau. |

## Judgment calls — kept for the author

These are 32 candidate clusters retained because their modal, negative or
contrastive wording is load-bearing or ambiguous. No stronger claim was
substituted. Line numbers locate the retained anchor, not every sentence in a
cluster. This is an editorial disposition, not resolution of the research gaps.

| ID | Section | Old → new line | Kept anchor | Reason / author call |
|---:|---|---|---|---|
| J1 | Abstract | 38 → 38 | <code>single-seed,</code> | Single-seed/common-cap/unequal-training and non-invariant-extrapolation qualifications limit the reported gap. |
| J2 | Abstract | 30 → 30 | <code>they do not lower-bound minimal circuit depth</code> | These negatives define the Lean result’s scope; removing them would invite a circuit-depth or full-stack impossibility reading. |
| J3 | 1 Introduction | 265 → 262 | <code>under the addressing-bottleneck hypothesis</code> | May benefit is hypothesis modality, not apologetic uncertainty; width/depth/head-count changes still prevent mechanism isolation. |
| J4 | 1 Introduction | 178 → 177 | <code>Its generator-table lookup construction is not a proof</code> | The lookup-versus-recurrent-trajectory distinction is the F1 gap. |
| J5 | 1 Introduction | 199 → 198 | <code>not the production decay below one</code> | The hold theorem’s decay domain differs from production; motivation is not a persistence guarantee. |
| J6 | 1 Introduction | 375 → 372 | <code>What we claim, and what we do not</code> | The heading identifies the claim/scope/non-claim table; keep its boundary function and leave all table and caption text untouched. |
| J7 | 2 Background | 539 → 535 | <code>This structural tradeoff is not a universal historical</code> | The historical claim is narrower than a lineage-wide decline in expressivity. |
| J8 | 2 Background | 572 → 568 | <code>Containment in TC#super[0] alone does not prove</code> | Circuit containment needs arithmetic assumptions and an unproved complexity-class separation; affinity is not a group-action obstruction. |
| J9 | 2 Background | 639 → 635 | <code>Exact recognition over all words, not finite-length accuracy</code> | The recognizer theorem does not classify a trained model from finite-length learning failure. |
| J10 | 2 Background | 663 → 657 | <code>within the restricted transition analyses motivating this grid</code> | The eigenvalue placement claim depends on restricted analyses, not affinity alone. |
| J11 | 3 Architecture | 860 → 854 | <code>It is not a full-stack implementation.</code> | Algorithm 1’s projected-input primitive is not the whole production model; the corresponding Algorithm 2/3 signature boundaries are also kept. |
| J12 | 3 Architecture | 1017 → 1010 | <code>memory rather than guarantee persistent bindings</code> | Saturation and latching motivate production memory but do not prove it for production decay. |
| J13 | 4 Systems | 1294 → 1287 | <code>a source-portability route</code> | Triton source portability is not ROCm correctness/performance parity; CUDA-only evidence and single-arm continuation boundaries remain. |
| J14 | 4 Systems | 1246 → 1239 | <code>a conservative lower bound</code> | MFU’s qualifier follows from omitted recurrent/gate FLOPs; slightly higher is quantitative uncertainty, not decorative hedging. |
| J15 | 5 Language Modeling | 1338 → 1331 | <code>literally identical recipe</code> | Literally/exactly matched tokens identify the comparison’s binding; retain these precision words rather than weaken them. |
| J16 | 6 Expressivity | 1704 → 1695 | <code>not a consequence<br>of the lookup theorem</code> | State-count headroom supports a learning-failure hypothesis; it does not prove nonlinear tracker realization or exclude constrained capacity. |
| J17 | 6 Expressivity | 1923 → 1913 | <code>not an unconditional four-seed mean</code> | Successful-seed conditioning and unequal subsets prevent an unconditional causal or algorithm-recovery claim. |
| J18 | 6 Expressivity | 2105 → 2095 | <code>a gap plausibly reflects inductive bias</code> | Plausibly expresses mechanism uncertainty given selection and optimization confounds; graceful rather than invariant transfer remains. |
| J19 | 7 Applications | 2138 → 2128 | <code>This is a structural family, not a claim</code> | The examples share a structural family, not weights, training or computational difficulty. |
| J20 | 7 Applications | 2165 → 2155 | <code>Here *EXHIBIT*</code> | EXHIBIT/ANALOGY/OPEN definitions and every application-specific exception delimit structural versus untested empirical claims. |
| J21 | 8 Formal Results | 2438 → 2426 | <code>not a missing proof of the already-proved lookup identity</code> | This not-X-but-Y distinction is substantive: F1 requires a stronger architecture/trajectory theorem. |
| J22 | 8 Formal Results | 2523 → 2511 | <code>ceiling reachable from the</code> | Clean was removed, but ceiling reachable by direct composition was kept: changing that characterization is an author/formal-scope call, not an editorial deletion. |
| J23 | 8 Formal Results | 2602 → 2590 | <code>Resource scope, not a parameter-efficiency lower bound</code> | The resource-class witness supplies no minimal-parameter corollary; keep exact-real, synthetic-cost, proxy and production-domain non-claims. |
| J24 | 9 Related Work | 2824 → 2812 | <code>not an achieved production-default property</code> | The fast-weight ancestry paragraph still distinguishes conditional latching motivation from achieved production behavior. |
| J25 | 10 Limitations | 2933 → 2920 | <code>Formal gaps: G1–G3 and F1</code> | All limitation paragraphs are load-bearing: hypotheses, missing evaluation/search records, unequal stops, S5 failure/OOM, comparator, scale, backend and citation boundaries. Entire section kept byte-for-byte. |
| J26 | 11 Conclusion | 3099 → 3086 | <code>rather than<br>priority, universal nonlinear superiority</code> | The TMLR contribution boundary excludes priority, universal superiority and an unproved complexity separation; this contrast carries scope. |
| J27 | 12 Testable Predictions | 3216 → 3203 | <code>conjecture</code> | Expected/should/conjecture and falsifiers mark predictions rather than established results. Entire section kept byte-for-byte. |
| J28 | 13 Future Work | 3243 → 3230 | <code>a genuine difference</code> | The genuine-difference-versus-optimization alternative states the unresolved scientific interpretation; both conditional/family readings remain. |
| J29 | Appendix A Lineage | 3382 → 3368 | <code>addressing geometry is a stability hypothesis</code> | The reshape is not an isolated-cause ablation; the abstract specialization is not production strict inclusion. |
| J30 | Appendix B Predecessor cohort | 3801 → 3786 | <code>The accurate statement is therefore neither</code> | Keep-and-report: the neither-length-generalizes-nor-fails passage delimits trained-length fit versus extrapolation and contains math mentions. Its stale “The delta update is also length-bounded” prose pointer does not match the existing “The delta update also plateaus under these recipes” heading; retained for author disposition, not silently repaired. |
| J31 | Appendix B Predecessor cohort | 3845 → 3830 | <code>does not establish exact algorithm recovery</code> | Transferable structure does not exclude finite-context heuristics or establish recovery; the solvable controls are diagnostic, not a causal theorem. |
| J32 | Appendix C Held-out bpb | 3920 → 3905 | <code>— *not* the fixed</code> | The measured tokenizer-specific normalization differs from the fixed training-curve constant; this is scope, not a rhetorical contrast. Entire appendix kept byte-for-byte. |

## Semantic spot-check: 15 removed-hedge sites

The old and new text and neighboring qualifications were manually compared at
each site below. The companion source-anchor check passed for every site; its
record is `paper/tmlr/validation/p14-semantic-check.json`. These checks do not
claim a new proof of the manuscript’s science.

| Removal ID | Section / old → new line | Result | Scope comparison |
|---:|---|---|---|
| 4 | 1 Introduction, 168 → 167 | PASS | The 1.3 B replication still concerns update/tuning rather than scale; no larger-scale claim was added. |
| 9 | 1 Introduction, 454 → 451 | PASS | M²RNN’s width/batch parallelism and serial recurrence are acknowledged positively; the mechanism remains a hypothesis and the runs do not isolate it. |
| 12 | 2 Background, 691 → 685 | PASS | The grid still describes adjacent cores with different decay placement; no production inclusion or parameter remapping is inferred. |
| 14 | 3 Architecture, 1063 → 1056 | PASS | The same three omissions and E88-only no-write-gate condition remain; E97’s split gates and ablation gain are unchanged. |
| 16 | 4 Systems, 1167 → 1160 | PASS | Width supplies parallel throughput, while each head’s time loop remains serial; no scan capability was introduced. |
| 17 | 4 Systems, 1219 → 1213 | PASS | The occupancy measurement retains E88 identity, CUDA device, sampling window, every statistic and the explicit non-transfer to E97. |
| 18 | 4 Systems, 1243 → 1236 | PASS | Occupancy remains distinct from MFU; the 6N convention and omitted recurrent FLOPs still make the MFU a lower bound. |
| 22 | 5 Language Modeling, 1499 → 1491 | PASS | Both numerical orderings remain descriptive; neither significance nor equivalence is asserted. |
| 26 | 6 Expressivity, 1724 → 1715 | PASS | The state-tracking-based nonlinearity selection and its directional confound remain intact. |
| 28 | 6 Expressivity, 2048 → 2038 | PASS | The roughly 50% GDN2 plateau is above the 1/256 declared random baseline, not invariant or a hard ceiling; one-seed and unequal-stop limitations remain. |
| 29 | 6 Expressivity, 2093 → 2083 | PASS | The length-512 stage still failed at sub-batch 1 after one update; hardware and seed constraints are retained and S5 non-replication remains verbatim. |
| 30 | 7 Applications, 2155 → 2145 | PASS | The statement characterizes executed composition only; the state-independent schedule, decoder hypothesis, alternative representation caveat and EXHIBIT/ANALOGY/OPEN definitions remain. |
| 38 | 8 Formal Results, 2477 → 2465 | PASS | Persistence still means exact non-equality at each finite length on the constructed witness, with no uniform margin or S5-generator bound. |
| 44 | 8 Formal Results, 2689 → 2677 | PASS | Set F still supplies only conditional slot-level bounds; the hold requirement lambda >= 1 is unchanged and is not a production-decay theorem. |
| 54 | Appendix B Predecessor cohort, 3853 → 3838 | PASS | The predecessor result remains an efficiency reading under tested budgets; optimization confounding and the unestablished expressivity wall both remain. |

## P13 provenance mechanism and enumerated guard delta

The task authorizes regenerated baselines using P13’s mechanism. The canonical
guard first rejected the edited source against P13’s digest (expected negative
check); it was not disabled or relaxed. `canonical-baseline.json` now records:

- Previous P13 source SHA-256 (retained as `p13MainTypSha256`):
  `989ea5f940d70be77b2d435fecc616040fe14d67ca54def6255a271d215059ee`.
- P14 source SHA-256 (`mainTypSha256`):
  `c899544128ebd08948edf721e86f1b15d249ecfbbfee00166b3a9e6b7314f7ad`.
- Added `p14BaseCommit: a0d1654c`; appended the P14 directive, report and exact
  diff path to `revision`. P13 part-1 digest and revision history remain.
- Frozen accepted snapshot `eb2b96363e3ff800d7eefe2ba7386e7cf673d1cf` and its PDF remain
  unchanged. The bibliography amendment rule, style provenance and guard
  implementation are unchanged.

All 20 fixed numeric/scope samples and the full symbol audit are deep-identical
to P13. Source-derived long-fragment coverage changes **2,587 → 2,567** because
prose leaves changed; every current long fragment passes in all three PDFs.
No sample, qualification or mathematical oracle was rebased away. Page counts
remain 65 / 58 / 58. Conversion counts remain 98 headings, 20 figure/table
captions and 21 formal statements; only the positive-embedding statement
excerpt in `conversion-inventory.json` changes (removal 37).

The existing layout-split policy is unchanged. Every changed split record in
`drift-check.json` is enumerated here (unchanged records omitted):

| Disposition | PDF | Fragment | Split index |
|---|---|---|---:|
| Removed | canonical | <code>(the M²RNN family) accumulates without</code> | 8 |
| Removed | submission | <code>FLOPs-per-bit curve over more than two orders of magnitude of training</code> | 31 |
| Removed | preprint | <code>FLOPs-per-bit curve over more than two orders of magnitude of training</code> | 31 |
| Removed | submission | <code>2 E97, 1 e97-linear, and 4 GDN2 seeds in this cell,</code> | 18 |
| Removed | preprint | <code>2 E97, 1 e97-linear, and 4 GDN2 seeds in this cell,</code> | 18 |
| Removed | submission | <code>shown statistically equivalent by a seed-level or paired equivalence</code> | 40 |
| Removed | preprint | <code>shown statistically equivalent by a seed-level or paired equivalence</code> | 40 |
| Added | submission | <code>control here shows what the raw-write family does in a stable</code> | 26 |
| Added | preprint | <code>control here shows what the raw-write family does in a stable</code> | 26 |
| Added | canonical | <code>address under which the old content is retrieved, and a value-axis</code> | 30 |
| Added | submission | <code>Input: arm, seed set, task generator, recorded study recipe<br>1. For each seed and arm, initialize from scratch or bound weights<br>   as specified by that study; initialize its optimizer.<br>2. Generate training batches at the declared trained length(s).<br>3. Optimize the declared objective until the study's stop rule.<br>   Record completed stages/updates and any failed stage.<br>4. For each declared evaluation length, generate held-out batches.<br>5. Run grad-free forward; count correct predictions on the mask.<br>6. Save per-seed records; aggregate only the declared seed subset.<br>Return: accuracy-by-length records and protocol/failure metadata</code> | 246 |
| Added | preprint | <code>Input: arm, seed set, task generator, recorded study recipe<br>1. For each seed and arm, initialize from scratch or bound weights<br>   as specified by that study; initialize its optimizer.<br>2. Generate training batches at the declared trained length(s).<br>3. Optimize the declared objective until the study's stop rule.<br>   Record completed stages/updates and any failed stage.<br>4. For each declared evaluation length, generate held-out batches.<br>5. Run grad-free forward; count correct predictions on the mask.<br>6. Save per-seed records; aggregate only the declared seed subset.<br>Return: accuracy-by-length records and protocol/failure metadata</code> | 246 |
| Added | submission | <code>kernel choice. The reported successful-seed subsets nevertheless</code> | 15 |
| Added | preprint | <code>kernel choice. The reported successful-seed subsets nevertheless</code> | 15 |

SHA256SUMS retains the exact same 15 paths. Six digests change: main.typ,
current.pdf, submission.pdf, preprint.pdf, arxiv-source.tar.gz and
canonical-baseline.json. The other nine (including both vector drawings,
bibliography, build wrapper, frozen accepted PDF and vendored style) are
unchanged. Generated TeX, PDFs, archive, extracted PDF text, build logs and
current guard receipts were regenerated, not hand-edited.

## Validation evidence

Durable receipts are under `paper/tmlr/validation/`; full execution logs and
editing/checking scripts are under
`/home/erikg/emender-scratch/fix-fanout/P14-validation/`.
No source regression test was changed or weakened. Three task-specific scratch
checks were added (preservation/replay, semantic anchors and independent report
parsing); all passed.

| Command | Result / retained evidence |
|---|---|
| `node .../P14-validation/verify-report.mjs` | PASS: independently parsed the report’s 55 exact old/new rows and reasons, line numbers, 39/16 counts and 32 judgments; replayed source/diff exactly, checked frozen figure/PDF/bibliography bytes and untouched formal/data/results/figure sources. p14-checks.txt. |
| `node .../P14-validation/preservation.mjs` | PASS: 55 exact replacements; 2,896 numeric tokens, 743 literal math spans, 310 raw/Lean spans, 151 literature citations, 187 ref pointers, 98 headings, 17 figure blocks, two standalone tables, eight pseudocode blocks and bibliography directive unchanged. Excluded-block overlap check passes; limitations, predictions and Appendix C byte-identical. Receipt: p14-preservation.txt. |
| `node .../P14-validation/semantic-check.mjs` | PASS: 15 source-anchor checks plus manual old/new scope comparison; p14-semantic-check.json. |
| `typst compile paper/main.typ paper/current.pdf` | PASS: 65 pages. |
| `node paper/tmlr/convert.mjs` | PASS: 98 headings, 20 captions, 21 formal statements, byte-identical bibliography. |
| `TECTONIC_BIN=/tmp/tectonic bash paper/tmlr/build.sh` | PASS: anonymous/identified PDFs 58/58 pages; no missing glyphs, unresolved refs, overflows or errors. build-output.txt, submission-build.txt, preprint-build.txt. |
| `bash paper/tmlr/bundle-arxiv.sh`; fresh extraction; `/tmp/tectonic --keep-intermediates --keep-logs main.tex`; `pdftotext -raw`; `cmp` against preprint-text.txt | PASS: fresh standalone archive build; extracted raw text byte-identical to preprint. arxiv-build-output.txt, arxiv-build.txt, p14-checks.txt. |
| `LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu PDFTOTEXT=/tmp/p9-poppler/usr/bin/pdftotext PDFINFO=/tmp/p9-poppler/usr/bin/pdfinfo node paper/tmlr/verify.mjs` | PASS: 20 samples × 3 PDFs, 2,567 source fragments, 707 math expressions, 441 attachments, eight S primes in each PDF (two in drawings), identity/bibliography/source guards. verify-output.txt, symbol-verification.txt, drift-check.json. |
| `node --test paper/tmlr/verify-symbols.test.mjs paper/tmlr/notation-audit.test.mjs` | PASS: 17/17, including missing-prime and prior-conversion-defect negative regressions. symbol-tests-output.txt, symbol-regression-tests.txt. |
| `node paper/figures/validate_assembly.js` | PASS: 13 sections, three appendices, 57 references, G1–G3/F1/U3.1, 17-file trust inventory and checkpoint bindings. assembly-check.txt. |
| `bash formal/lean/scripts/check_paper_core.sh` | PASS: 17 local project sources, no placeholders/added axioms/opaque/native_decide. p14-checks.txt. |
| `(cd formal/lean && lake build ElmanProofs.PaperCore)` | PASS: 2,196 jobs (cached build, not clean Mathlib bootstrap); existing linter warnings only. p14-checks.txt; full scratch lean.txt. |
| `node formal/lean/scripts/test_local_import_closure.js` | PASS: recursive imports, dependency exclusion, five forbidden tokens and compiler-error fail-closed fixtures. p14-checks.txt. |
| `(cd paper/tmlr && sha256sum -c validation/SHA256SUMS)` | PASS: all 15 paths. |
| `git diff --check` | PASS. Explicit-path commit, push, remote-hash and empty-index checks are the delivery gate below. |

## Diagnostics, residual risks and delivery

A scratch replacement initially mismatched bullet indentation and aborted
before writing; corrected. The preservation check initially located a forward
reference rather than a section heading; corrected its heading lookup and
replayed it. The independent report checker initially tried reading the locally
frozen accepted PDF as a Git blob; it is ignored/untracked, so its unchanged
bytes were verified against P13’s committed SHA256SUMS receipt instead.
The staged whitespace check flagged the unified-diff receipt’s blank context
lines as trailing spaces; the receipt was regenerated with `--unified=0`, retaining
all exact changes without context-only blank lines. The check was not disabled.
The final preservation/replay, semantic checks and regressions all
pass; these diagnostics did not change the manuscript outside the reported table.
Underfull TeX spacing warnings and Typst PDF’s pre-existing Poppler
“Suspects object is wrong type (boolean)” diagnostic remain; the guard finishes
successfully. No acceptance check was disabled.

The 32 kept judgment entries, the stale Appendix B prose pointer (J30), and
P13’s normalization/production-fidelity/protocol/open-research calls remain
author decisions. No new experimental, formal or deployment claim is certified.
Independent reviewer acceptance is still required.

Delivery is one requested commit,
`editorial pass: de-hedging per operator directive`, including this report and
all explicit task paths, then a push to `origin/paper-1p3b-spine`. The final
response and scratch delivery receipt identify the resulting hash and remote
comparison. The seven pre-existing untracked review files are not included;
no unrelated file is staged. Next: independent editorial/scope review; the
supervisor handles any hypervolu mirror update.
