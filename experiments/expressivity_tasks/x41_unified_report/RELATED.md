# Related work for Emender Memory

Source: [X32 survey](../x32_comparables/REPORT.md), especially its must-cite list and full-PDF addendum. This is a citation-ready bibliography and positioning guide, not a new exhaustive literature search. X32's confidence/missing-evidence qualifications survive. The Merrill, DeltaStack and Delétang primary landing pages were also checked for title, authors and publication identity on 2026-10-28.

## Structural generalization and memory management

Use Delétang for the prior empirical phenomenon and Merrill for the restricted SSM theory. Do not turn fixed-state shorthand into a theorem about every fixed-dimensional nonlinear recurrence or every delta-rule extension. Stack/queue/deque and differentiable interpreter work precede architectural operations.

## Complete X32 must-cite bibliography

**Must cite (claim-bearing):**

1. Delétang et al., *Neural Networks and the Chomsky Hierarchy*, ICLR 2023 — [arXiv:2207.02098](https://arxiv.org/abs/2207.02098). Cite for: structured memory ("a stack or memory tape") is what enables non-regular generalization. **This is the sentence a reviewer will use to say "already known".**
2. Merrill, Petty, Sabharwal, *The Illusion of State in State-Space Models*, ICML 2024 — [arXiv:2404.08819](https://arxiv.org/abs/2404.08819). Cite for: fixed-state models provably cannot state-track; our 0.50 controls are theory-predicted.
3. Nguyen et al., *Push, Pop, Parallelize* (DeltaStack), ICML 2026 — [PMLR v306](https://proceedings.mlr.press/v306/nguyen26i.html), [code](https://github.com/teeann/DeltaStack). Cite for: closest cousin; delta-rule stack ops; chunk-parallel training; the bounded power-of-2 stack; and to position our budget-boundary analysis as the delta.
4. Yang et al., DeltaNet, NeurIPS 2024 — [arXiv:2406.06484](https://arxiv.org/abs/2406.06484). Cite for: the delta rule as error-correcting overwrite; the gated-delta-rule control.
5. Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, 2021 — [arXiv:2102.11174](https://arxiv.org/abs/2102.11174). Cite for: the fast-weight framing of our writes.
6. Graves et al., NTM — [arXiv:1410.5401](https://arxiv.org/abs/1410.5401); Graves et al., DNC — [Nature 538](https://www.nature.com/articles/nature20101). Cite for: learned addressing lineage and its failure modes; the "fix the management" motivation.
7. Joulin & Mikolov, Stack-RNN, NeurIPS 2015 — [paper page](https://proceedings.neurips.cc/paper_files/paper/2015/hash/26657d5ff9020d2abefe558796b99584-Abstract.html); DuSell & Chiang, NS-RNN — [ACL 2020](https://aclanthology.org/2020.conll-1.41/), [ICLR 2023](https://arxiv.org/pdf/2210.01343); Suzgun et al. — [arXiv:1911.03329](https://arxiv.org/abs/1911.03329). Cite for: architectural-memory precedents on Dyck/CFG.
8. Wu et al., Memorizing Transformers, ICLR 2022 — [arXiv:2203.08913](https://arxiv.org/abs/2203.08913). Cite for: inference-time store-size enlargement as prior art on our budget knob.
9. Karami, Pascanu, Mirrokni, Lattice, 2025 — [arXiv:2504.05646](https://arxiv.org/abs/2504.05646). Cite for: fixed-slot budgeted compression with novel-content-only updates = prior art on MERGE.
10. EDA, 2026 — [arXiv:2606.26560](https://arxiv.org/abs/2606.26560). Cite for: decoupling erase address from write address = prior art on CORRECT's address structure.
11. Behrouz et al., Titans, NeurIPS 2025 — [arXiv:2501.00663](https://arxiv.org/abs/2501.00663); Sun et al., TTT, ICML 2025 — [arXiv:2407.04620](https://arxiv.org/abs/2407.04620). Cite for: the fixed-state test-time-memory alternative and why capacity is the axis they do not have.
12. Berges et al., Memory Layers at Scale, ICML 2025 — [arXiv:2412.09764](https://arxiv.org/abs/2412.09764); Lample et al., PKM, NeurIPS 2019 — [arXiv:1907.05242](https://arxiv.org/abs/1907.05242). Cite for: the "memory = capacity" reading we are *not* making.
13. Kaiser & Sutskever, Neural GPU — [arXiv:1511.08228](https://arxiv.org/html/1511.08228v3); Price, Zaremba, Sutskever, *Extensions and Limitations of the Neural GPU* — [arXiv:1611.00736](https://arxiv.org/html/1611.00736v2). Cite for: OOD claims must be distribution-conditional (a discipline we must meet).
14. Bošnjak et al., Differentiable Forth, ICML 2017 — [PMLR v70](https://proceedings.mlr.press/v70/bosnjak17a.html); Reed & de Freitas, NPI — [arXiv:1511.06279](https://arxiv.org/abs/1511.06279); Kurach et al., NRAM — [arXiv:1511.06392](https://arxiv.org/abs/1511.06392). Cite for: learned-vs-fixed management is a long-standing axis, and hybrids are precedented.
15. Raven — [arXiv:2607.25357](https://arxiv.org/abs/2607.25357); RAM-Net — [arXiv:2602.11958](https://arxiv.org/pdf/2602.11958). Cite for: current (2026) fixed-slot-with-routing state of the art — a reviewer will know these.
16. Katharopoulos et al., linear attention — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236); Gu & Dao, Mamba — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). Cite for: the fixed-state compressor family.


## Closest comparisons and the claims they constrain

| Comparison | Prior contribution | Narrow distinction / required qualification |
|---|---|---|
| Delétang | Structured-memory OOD generalization; memory-tape terminology | Direction is known; cite distribution and depth protocol |
| Merrill | State-tracking limitations of analyzed SSMs | Theoretical scope must be retained; no new impossibility proof |
| DeltaStack | Delta-rule stack, learned soft pointer, parallel training | Architectural age-1 routing, MERGE + pressure, measured failure boundary; boundedness is shared |
| DeltaNet / fast weights | Corrective overwrite and chunkwise training | The delta rule itself is prior art |
| EDA | Separate erase and write addresses | Temporal age-1 correction is the local distinction, not invention of address decoupling |
| Lattice | Budgeted fixed-slot compression/update | No broad claim to invent merge or novelty-sensitive storage |
| Raven / RAM-Net | Fixed slots and sparse/learned routing | X32 mechanism evidence is abstract-level; no comparative performance claim here |
| Memorizing Transformers | Larger inference-time stores | Store-size knob is prior art; boundary measurement is narrower |

## “This is just X” rebuttal guide

| Objection | Response licensed by receipts |
|---|---|
| Just an NTM/DNC | Shared external-memory lineage; management rules here are architectural, while content/gates remain learned |
| Just a stack | READ/POP are content-addressed; age-1 correction and MERGE exceed a LIFO-pointer description; no unbounded guarantee |
| Just DeltaStack | Agree it is closest; do not deny its capacity theorem, ablation or controls; distinguish failure-boundary telemetry and recurrence |
| Just a KV cache | Shared slot/read intuition; identify the explicit correction/pop/decay/merge program rather than claiming a wholly new category |
| More state than controls | Parameter matching is not state-byte matching; a full state-accounting comparison remains future evidence |
| Pressure is only a diagnostic | Agree: no adaptive policy capability is established; report false negatives and early-warning denominator |
| A fitted one-run boundary | Agree on limited seeds and budgets; retain the larger arm's pressure-free hole |
| Titans/TTT already have memory | Compare state organization and update program, not the word “memory” |

## Naming and disambiguation

The operator selected **Emender Memory**, with *emendare* as the rationale for age-1 revision. X32 flags collision of “tape machine” with NTM and Delétang's memory tape, and warns that bounded occupancy must not be called unbounded growth. The architectural ops answer the stack-only/LIFO objection; they do not abolish the finite budget. This is operator terminology plus design description, not a literature-derived naming priority claim.

## DeltaStack addendum: retained refinements

The following is the repository's full-PDF verification receipt, reproduced with attribution. Its numerical literature results are prior survey evidence, **not** newly verified X41 experimental measurements; PAPER.md relies on the mechanism/capacity distinctions rather than its benchmark numbers.

## ADDENDUM A — DeltaStack PDF verification (added by the parent session, post-survey)

The survey flagged the unread DeltaStack PDF as the single largest evidence gap. The parent retrieved and read the full paper (PMLR 306:92579–92592; local copy at /tmp/deltastack.pdf, fetched 2026-10-26). Corrections and confirmations to the survey's DeltaStack claims:

**Confirmed from the paper body:**
- Management is **learned**: actions are `softmax(W_α x_t) ∈ Δ³` over {PUSH, POP, NO-OP}; the pointer is a continuous scalar drift `p_t = p_{t−1} + α_push − α_pop` sharpened by a **learnable** Laplacian kernel (γ = softplus(γ̂)). The survey's code-level reading was correct.
- Theory: single-layer DeltaStack recognizes Dyck_{k,D} with embedding dim d = ⌈log₂ k⌉+2 **and stack capacity S ≥ D**; DeltaNet recovered as a special case; NC¹-complete state tracking inherited. Expressivity only — the paper's own Discussion concedes "expressivity does not guarantee learnability."
- LM results: 340M/760M on FineWeb-Edu; 760M zero-shot avg 52.0 vs DeltaNet 51.4; length extrapolation to 16k on CodeParrot/OpenThoughts-Math maintained vs DeltaNet.
- Formal languages (their Table 1, 5-run avg): DeltaStack 79.4 avg vs DeltaNet 65.5; Stack-RNN (sequential reference) 96.3. Per task: Dyck 94.3, Stack Manipulation 96.8, Reverse String 98.8, **Mod Arith w/ Brackets 35.2** (they explicitly struggle; learnability admission in §6).

**Corrections to the survey:**
1. "No capacity analysis" must be softened. They **do** have a stack-size ablation (their Fig. 3: S ∈ {8, 16, 24, 48} on Parity / Reverse String / Stack Manipulation) and their Theorem 4.4 carries an explicit **S ≥ D capacity condition**, with a remark that deeper test-time nesting generalizes by increasing S "without retraining" — a *theoretical* claim from the construction, with no empirical beyond-trained-depth receipt.
2. What remains absent, verified against the full text: any **failure-boundary analysis** — what happens when S < D. Saturation behavior is unexamined in the paper (the pointer clamp lives in the code); there is no degradation policy, no saturation telemetry, and no empirical curve of collapse depth vs budget with matched fixed-state controls. Their capacity story is a monotone-benefit ablation plus a sufficiency theorem, not a boundary measurement.
3. Their baseline breadth is real (DeltaNet, Gated DeltaNet, Gated DeltaProduct, RWKV-7, Transformer, Stack-Attention, LSTM, Stack-RNN at trained lengths) — our "matched controls" discipline is less distinctive than the survey already noted; our distinction is the *boundary characterization*, not the existence of controls.

**Strategic consequences (updated):**
- The quadrant claim (zero-learned-management × content-driven occupancy) survives: DeltaStack sits firmly in the learned-management, bounded-state quadrant.
- DeltaStack's S ≥ D theorem **supports our framing**: capacity binds expressivity — we are the ones who measure where the boundary empirically bites, expose saturation as a signal (pressure), and handle it gracefully (merge) rather than assuming it away or clamping silently.
- Their Mod-Arith-w-Brackets 35.2 and expressivity-vs-learnability concession strengthen the "learnability is the frontier" thesis this program already adopted (X26c/X27).
- Their chunkwise O(LCd) parallel training is the axis where they lead; the X31 parallel decomposition (decay = prefix product, append = scan, age-1 = shift-local, pop/merge = sparse sequential corrections) is the direct response and must be cited against their WY-representation trick.

## Items still excluded from claim-bearing use

X32's unverified Grefenstette, kNN-LM, Strobl and Sarrof references remain verification leads, not verified claim support here. Unidentified “Differentiable Pushdown (Du et al.)” and “MOVES (Sander et al.)” are not silently replaced with invented citations. Third-party AI DeltaStack summaries remain excluded. Broad causal explanations for DNC's historical adoption are interpretations, not established causal evidence.
