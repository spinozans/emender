# X43 — the fork-resolved language model: living program ledger

The fork-resolved LM uses the corpus's exact combinatorial structure (xsa/chi) as its deterministic spine and EmenderLM as the organ of decisions. The complementarity hypothesis is that the index carries what the neural store is weakest at—exact corpus continuation—while the certified emender carries what the index cannot: out-of-distribution (OOD) behavior and abstraction. This is a program hypothesis, not an established generalization result. Paper 1, [X41](../x41_unified_report/PAPER.md), establishes the memory's scoped boundary properties; this is Paper 2, the decision space over sequences. The chi formalism's two-sided bounds are the intended theoretical spine, and the receipts below are its empirical legs. Exact structural decomposition into walks and decisions is the proposed distinction from kNN-LM/infini-gram style retrieval/interpolation; no novelty priority or comparative performance claim follows from this positioning.

The theoretical bridge is still open: DERIVATION_PART1C.md defines finite-body per-position forks, not minimum suffixient cardinality chi. There are 636,727 fork positions but 425,221 xsa witnesses in the historical container. Do not substitute either for the other or assert an unproved bound. Formal two-sided inequalities, their constants, boundary convention and application to decision cost remain a registered derivation task; no theorem receipt is claimed here.

Positioning and related work: [POSITIONING.md](POSITIONING.md) (sweep-level; full-claim verification pending).

## Claim ladder

All losses below are natural-log nats/byte. The machine-readable bindings at the end are part of this document and the sole claim registry. Every receipt there names its repository, exact path, full commit and SHA-256. Registered rungs have predictions and no outcome receipts. Historical RESULT files are immutable evidence; their earlier “blocked”/“not run” statements are superseded only by the explicit later receipts below.

### C1 — measured: fork structure cross-validated

Independent direct reverse-SA/LCP and decoded FM backward-extension routes agree on 1,115,394/1,115,394 labels and maximal context lengths: 636,727 forks, 478,667 forced, maximum context 123. Receipt bindings: C1. The xsa byte-contract gates at upstream 8be5206 are green (H1–H4, 41 Rust tests); they repair the old absent-ff query behavior. This separates the original label agreement from the later upstream query repair; it does not claim the historical build came from that later commit.

### C2 — measured: loss decomposition

Forks occupy 57.0854% of corpus positions and carry 65.928% / 82.004% of Emender / transformer held-out NLL. Full held-out denominator: 111,539 targets (64,041 fork; 47,498 forced). Forced/fork NLL is 1.761594/2.528119 for Emender and 0.773061/2.612681 for transformer. Receipt bindings: C2. These are descriptive corpus-wide labels, with held-out structure and potentially longer contexts than the neural 128-byte resets.

### C3 — measured: fusion crossover

Final-test denominator: 55,770 targets, distinct from C2. Train-count feature fusion plus forced walk yields Emender 1.673012 versus transformer 1.746996, reversing standalone 2.222975 versus 1.924201. Receipt bindings: C3. The crossover is conditional on this post-hoc recipe, corpus and seed; even train-count arms select contexts using corpus-wide labels. Corpus-count walk gives about 0.5708, a transductive diagnostic containing target/future information, not a causal ceiling. Train-count exact oracle NLL is infinite; zero targets are retained.

### C4 — refuted: fork-only writes improve decisions without forced-loss cost

The preregistered joint improvement prediction fails: fork-only eligibility worsens both losses. At 180 seconds baseline forced/fork NLL is 1.844641/2.576447 versus 1.998269/2.665702; at fixed step 1000 it is 2.111915/2.726599 versus 2.126162/2.729965. Receipt bindings: C4. The tested internal write-mask route is closed for this program; the two-system route is the next design choice. This does not rule out every internal architecture. Actual writes fall from .462504 to .204370, not to fork density; inference used oracle masks.

### C5 — measured: pipeline receipts, with failures retained

Corpus → tape → query and independent container CRC/LF checks are receipted. Receipt bindings: C5. Early wrong-path build and cyclic-seam failures remain failures. Historical part1d query correctness was 12/13, with virtual-END aliasing for absent ff; upstream 8be5206's byte-contract receipt subsequently reports absent-symbol and boundary-byte gates green and legacy text refusal. XML entities remain verbatim, so the unchanged Shakespeare body in the raw wrapper is not well-formed XML. Upstream gates do not retroactively certify every historical command. Dirty local query rerun files are not silently adopted as committed evidence. All-256-symbol corpora remain refused; no end-to-end latency claim is established.

### C6 — measured: ordinal crossover survives; fork-aware recipe fails absolutely

The preregistered direction survives **only as a corpus-label diagnostic**: primary all-forced standalone NLL is Emender 6.064164 versus transformer 6.693707 (delta −0.629543); fork NLL is 10.433393 versus 11.516522 (delta −1.083128). Actual NVIDIA GDN-2 fork NLL is 11.360357. All six aware/control arms completed matched 180-second budgets. These large losses do not validate the architecture foundation: the tested normalization/OOD-training recipe fails against every retrained control.

Using the exact C3 empty-support abstention rule, aware standalone/fusion is Emender 6.694547/6.694321, transformer 7.324090/7.312338, GDN-2 7.233323/7.195213. Historical C3 fusion remains 1.673012/1.746996. Retrained neural-only controls have fork NLL 2.527107/2.538039/2.380693; their C3-compatible fusion is 1.676996/1.699410/1.602624, so the Emender/transformer post-hoc crossover also reproduces in fresh controls. Full strata, parameters, steps, tokens, timings and failures: [RESULT_C6.md](RESULT_C6.md), bindings C6.

Training self-counts produce zero OOD target examples among 572,686 training forks, but final test has 17,066 nonempty-support zero-target forks. Their aware NLL is 18.159355/20.142119/19.707956. Uniform OOD routing and self-containing training counts are a failed recipe. The selector and labels are corpus-wide; **no causal training-only selector/arm was executed**, leaving that broader C6 requirement open. One seed/corpus, no significance or ingredient-ablation claim. Do not infer sound architecture foundations merely from the ordinal sign or declare C3 a post-hoc artifact. The initial optional GDN-1 constructor was caught before its training; its runner interruption (exit 143) and actual GDN-2 continuation (exit 0) are retained.

Historical registration notes below preserve the protocol state before execution and scoring.

C6 execution preregistration (2026-11-02): [PREREGISTRATION_C6.md](PREREGISTRATION_C6.md), SHA256 `e70de380d2cfe34d4c7a66abad42a926b724da60cc7d2929d8f8313a04b968b9`. Primary comparison is corpus-label diagnostic two-system standalone overall NLL (Emender < transformer); secondary fork NLL, fusion and retrained controls are separate. The frozen protocol explicitly defers the causal training-only context selector and limits any verdict accordingly. At registration, no training/evaluation had begun.

C6 additive C3 comparison clarification: [PREREGISTRATION_C6_C3_COMPARISON.md](PREREGISTRATION_C6_C3_COMPARISON.md), SHA256 `46fbbd048d6a2c6ee3cc110b36b7851233081ebd9c7d209d35617f50d85d6973`. Registered while the first arm is still training, before any held-out inference: C3-compatible rows abstain at empty-support forced positions; primary all-forced diagnostic remains unchanged.

C6 optional-arm correction: [PREREGISTRATION_C6_GDN2.md](PREREGISTRATION_C6_GDN2.md), SHA256 `72debb3246ed183bc1dbee2becb1602ec4cb42d2e8bcb65b50e659d072ffa167`. Before optional-arm training, the initial GDN-1 constructor was identified and its runner terminated (exit 143); four Emender/transformer receipts were retained. Actual GDN-2 uses the existing external NVIDIA loader with FLA 0.5.1. At that correction point, no final-test loss had been inspected or scored.

### C7 — registered: designed dense two-system prototype

Prediction: fork-masked loss, branch-set conditioning plus an OOD channel, interval features in the gate and exact-count teachers reduce final-test NLL below C3's 1.673012 Emender result. Report gap closure (1.673012 − designed NLL)/(1.673012 − 0.570865), while explicitly distinguishing the causal/train-count system from that transductive diagnostic denominator. No numeric closure is assumed. Ablate each ingredient with matched budgets; evaluation rules and causal index must be frozen before execution.

### C7a — refuted: unconstrained joint residual training fails dense comparison

Both 180-second boosted heads lose under identical C3 calibration-feature fusion plus forced walk. Total/fork NLL: Emender boosted **3.718402/4.902387**, dense **1.673012/2.424886**, C6 compound masked **6.694321/10.433005**; transformer boosted **14.647364/19.057854**, dense **1.746996/2.558332**, compound masked **7.312338/11.496303**. The separately deployed uncalibrated joint gate is worse still: total/fork **3.860648/5.147120** and **18.651658/25.947240**. No training positions were deleted, and both neural softmax and gate received backprop through the mixture.

Transformer joint neural weight collapses to about 2.4e-6 on one-branch positions versus C3 dense .965; self-containing counts provide no zero-target training events. This refutes the scoped prediction that this unconstrained objective preserves anchors and improves both losses, not residual training in general. Emender processed only 229 steps (unexplained throughput loss); transformer 26,380. Wall clocks matched, steps did not. C6 controls remain compound conditioning/normalization/masking arms, not isolated masking evidence. One seed, corpus-wide selectors, no causal generalization claim; GDN2 deferred. [RESULT_C7a.md](RESULT_C7a.md), bindings C7a.

Registration preceded training in a07a8eea: [PREREGISTRATION_C7a.md](PREREGISTRATION_C7a.md), SHA256 `2a2b5fa7003c86facd81e5081bfd5b82260e573d22ad8d1c71d4768e64668978`. Primary common calibration gate and secondary uncalibrated joint gate were frozen in this protocol; dense and C6 controls were reused, not retrained. Seventy independent loss/denominator checks and six CPU tests pass.

### C7b — refuted: primary Emender loses; secondary transformer fixed-blend wins

emender C3-fused total/fork/empty-support NLL: fixed 1.711331/2.476626/2.523966; floor 4.552704/5.956247/9.051871; dense 1.673012/2.424886/2.444723; starved 3.718402/4.902387/7.269978;

transformer C3-fused total/fork/empty-support NLL: fixed 1.655556/2.435391/2.288062; floor 5.281165/7.443949/8.962798; dense 1.746996/2.558332/2.466627; starved 14.647364/19.057854/30.744151;

The primary Emender fixed arm loses all three columns; secondary transformer wins all three. The gate-floor arm loses for both. This mixed receipt does not settle training integration globally at inference.

Fixed training has no learned routing and retains .5 direct true-target CE on covered positions; empty support is pure true-target. Secondary gate-floor .3 prevents probability-weight collapse, but does not bound gradient responsibility. Gates are calibrated/frozen under unchanged C3 inference. Controls reused; four sequential 180-second arms. Signed decisive deltas are separate for total, fork and empty-support columns, with full OOD-fork/forced denominators and calibration trust telemetry. [RESULT_C7b.md](RESULT_C7b.md), bindings C7b.

Registration committed before training in 00f39a22; [PREREGISTRATION_C7b.md](PREREGISTRATION_C7b.md), SHA256 `3c685cd6e59555e031379daa5c841f1a74cf85667b2a52a758876b8b65472949`.

One seed/corpus, transductive selector and no zero-target training exposure; equal wall clocks, unequal steps and cold/warm kernel costs. No universal residual-learning or scale-up claim. Six tests and independent receipt replay pass.

### C8 — registered: redundancy ladder

Prediction: controlled duplication 2x/10x/50x reduces per-position decision density and increases forced-walk gains; test whether crossover changes monotonically with redundancy rather than assuming it. Generate synthetic corpora from the same base distribution, separate duplicated training material from independently sampled held-out material, and report both per-position forks and suffixient chi separately. This tier-1 curve is intended to predict Wikipedia/Pile behavior; those transfers remain unmeasured. Freeze generator, sizes, splits and seeds before execution.

### C9 — measured: increased overlap; full-context OOD recovery failed

The precommitted prediction holds for the retained dense transformer: walk/fork generation has greater training-corpus overlap at every n=4..32 than neural-only generation. At n=16, novel rates are 93.090% versus 96.259%; longest copied spans are 21–50 versus 20–28 bytes. Four fixed prompts per mode generate 4096 bytes each in total. Forced walks account for 1676/4096 steps. Active-interval OOD departures fall from 963 to 724; full 128-byte context remains unsupported for 4082/4096 walk/fork steps, with **zero full-context re-entries**. Every empty full context obtains a shorter supported suffix immediately; this is backoff recovery only. Self NLL worsens from .930636 to 1.036002 nats/byte; no independent quality claim. [RESULT_C9.md](RESULT_C9.md) retains all n-gram rates, samples, run lengths, and limitations. Protocol committed in 644bff97 before generation; outcomes in 0d9b8bce; bindings C9. One dense head, four prompts, transferred frozen C3 gate, and no escape from forced singleton support limit inference.

### C10 — registered: index as teacher

Prediction: mimetic initialization from exact training-only branch distributions lowers fork NLL or time to a preregistered loss threshold against random initialization at equal total training budget, including teacher cost. Keep held-out bytes out of teacher targets and distinguish support from OOD. Freeze distillation objective, temperature, smoothing/zero-support policy and initialization duration before execution. No teacher-training receipt exists.

### C11 — refuted: dense device inputs fail both heads; swap untested

P1 raw fork, P2 empty-support preservation and P3 fused-total improvement all fail for both Emender and transformer under 180-second budgets. Exact distributions and interval features entered the residual stream; full dense CE and the internal memory stayed unchanged. This closes the tested input-channel/dense-loss cell negatively. Dense head gradients cannot bypass a learned mixture gate here, yet shortcut conditioning still does not preserve OOD quality. Do not equate routing-free training with an empirical OOD guarantee.

[RESULT_C11.md](RESULT_C11.md) contains both heads' fused/unfused total/fork/OOD tables and steps. P4/P5 swap and 100/200-step adaptation remain registered and unexecuted; no swappability result. One seed, corpus-wide transductive context selection, self-containing train distributions, reused historical dense controls and unequal step counts limit interpretation. Registration cc137968, [PREREGISTRATION_C11.md](PREREGISTRATION_C11.md), SHA256 `e682a17d36ec33630096bfbafa89724974eb8282df7d9d175265c46e0424f8e6` preceded training. Four CPU tests and independent 56-pair loss replay pass.

### C12 — measured: disjoint gains pass; masking does not rescue enabled self-index

P1/P2 pass for both heads: disjoint-X training improves raw Y-fork NLL Emender **2.529165 → 2.434766** (−0.094399), transformer **2.634117 → 2.470846** (−0.163271). Empty-X-support NLL is preserved: dense/disjoint **2.216768/2.214777**, **1.901476/1.810907**. Both heads completed both 180-second arms. X Darwin scientific prose (950,692 bytes) and unchanged Y Shakespeare drama (1,115,394) were constructed/hash-bound before training. No Y material was inserted into X; 74.3662% of training positions are empty-support and 84.6791% zero-target. Whole-task direct copying has infinite final-test NLL.

This is a positive scoped result beyond self-containing copy training, not an isolated operator-skill proof. Shared English branch distributions can still predict some Y targets, and corpus-wide Y context lengths remain transductive. Reused historical dense controls differ in steps/precision and active parameters. Therefore the requested negative closure condition was not met, but “any gain can only be operator semantics” does not follow. Whole-device removal worsens the disjoint heads; empty support retains context-length metadata, so P2 does not establish unplugged robustness.

P3 passes in both eval modes for both heads versus C11, yet enabled self-index remains harmful: masked/dense OOD-self is Emender **4.858708/2.444723**, transformer **3.390723/2.466627**. With all device inputs removed, masked OOD is **2.425617/2.075162**, restoring an independent mode. Shortcut prevention preserves the unplugged head; it does not rescue the enabled self-index interface at this tier. [RESULT_C12.md](RESULT_C12.md) has the complete disjoint and masking tables, recovered C11 feature-removal controls, costs and residual risks. Original C11 with-feature predictions remain unchanged; exact-step recovery passes whole-vector probability parity for both heads.

Preregistered in de8e342e before feature construction/training: [PREREGISTRATION_C12.md](PREREGISTRATION_C12.md), SHA256 `3a7b1e86f6fc7836025cf2c2527416efca7f27a38e1baca3d83f43862483837b`. Missing-control recovery preregistered in d1744b2e before replay/scoring: [PREREGISTRATION_C12_CONTROL_REPLAY.md](PREREGISTRATION_C12_CONTROL_REPLAY.md), SHA256 `1ee68ea51f96ed32a6bb124bf4095ea291180d907de8b6ab061e9f12699547cc`. Six CPU tests, 2,049 independent substring feature checks, 98 loss/denominator checks and both runner exits pass. One seed/pair; no uncertainty, matched-step causal device ablation or isolated semantic-operator claim. No self-index scale-up recommendation. Bindings C12.

### C13 — measured: coverage scaling is mixed; calibration and learned consultation stack

P1 passes for Emender but fails for transformer: frozen C12 X→self total/fork changes **2.116847/2.434766 → 1.834650/2.397116**, **1.816094/2.470846 → 1.835328/2.523442**. P2 is mixed by column: raw self inputs lose total to dense C3 fusion for both, win fork for both, and lose/win OOD-self for Emender/transformer. P3 shows complementary integration: self plus C3 calibration/forced walk yields **1.650482/2.379249** and **1.694013/2.470112** total/fork, beating dense C3 fusion in both columns for both heads. OOD-self is unchanged by stacking: **2.450782/2.410592**, versus dense **2.444723/2.466627**. The tested deployment recipe retains both learned consultation and calibration/forced walk; the policy does not replace fusion.

The requested tier-1 integration comparison is closed, including the failed monotonic-scaling prediction for transformer. The remaining registered C7/C8/C9/C10/C11 swap/adaptation work is not thereby completed. Full map below: all rows share final-test total/fork denominators 55,770/32,415; OOD-self and OOD-X have distinct denominators 9,049/41,668. Historical C11 OOD-X was not scored.

| Head | Training / eval integration | Total | Y fork | OOD-self | OOD-X |
|---|---|---:|---:|---:|---:|
| emender | Dense / neural | 2.222975 | 2.529165 | 2.444723 | 2.216768 |
| emender | Dense / C3 calibration + forced walk | 1.673012 | 2.424886 | 2.444723 | — |
| emender | C11 self inputs / neural | 3.872928 | 5.330632 | 6.551557 | — |
| emender | C11 self inputs / C3 fusion | 3.870401 | 5.327873 | 6.551557 | — |
| emender | C12 disjoint / X inputs | 2.116847 | 2.434766 | 2.423721 | 2.214777 |
| emender | C12 disjoint / all zero | 3.078657 | 3.040701 | 4.267185 | 3.202396 |
| emender | C12 disjoint / self inputs | 1.834650 | 2.397116 | 2.450782 | 1.849784 |
| emender | C12 disjoint / self inputs + C3 fusion | 1.650482 | 2.379249 | 2.450782 | 1.649322 |
| transformer | Dense / neural | 1.924201 | 2.634117 | 2.466627 | 1.901476 |
| transformer | Dense / C3 calibration + forced walk | 1.746996 | 2.558332 | 2.466627 | — |
| transformer | C11 self inputs / neural | 4.953106 | 7.317413 | 6.060058 | — |
| transformer | C11 self inputs / C3 fusion | 4.941959 | 7.298356 | 6.060058 | — |
| transformer | C12 disjoint / X inputs | 1.816094 | 2.470846 | 2.383454 | 1.810907 |
| transformer | C12 disjoint / all zero | 2.474033 | 2.871956 | 2.918144 | 2.464290 |
| transformer | C12 disjoint / self inputs | 1.835328 | 2.523442 | 2.410592 | 1.818276 |
| transformer | C12 disjoint / self inputs + C3 fusion | 1.694013 | 2.470112 | 2.410592 | 1.683711 |


Predictions were hash-bound before recovery/inference in `67c26772`, [PREREGISTRATION_C13.md](PREREGISTRATION_C13.md), SHA256 `d8da80b016bb619a04d5c811597d19c8b587c4ea6ac538436bac88a3d5d0778f`. No new training arm: missing C12 weights were reconstructed at their exact original 6,293/29,640 steps, with **zero** whole-held-out probability error in all four X/all-zero parity vectors (111,539 targets each), then frozen. No checkpoint files. Broker GPU 2; all 60 independent loss/denominator and fusion replays pass. Bindings C13; [RESULT_C13.md](RESULT_C13.md).

Scope: unchanged C11 training-prefix counts and device geometry, not future-containing full-corpus counts. “Full coverage” still has empty self support. Metadata changes with the requested geometry, so this measures coverage plus an interface shift. One seed/pair, transductive selectors, historical unmatched controls and no uncertainty estimate. Device-off degradation proves live dependence, not that nothing is encoded in weights; shared English distributions leave partial copying possible. No isolated semantic-operator or causal coverage-only claim.

### C14a — measured: provisional Pile/Wikipedia coverage; branch integrity gate failed

10,000 eligible interior Pile token boundaries against the 100 GB Wikipedia array give 16/64/256/1024-byte support **36.59% / 2.36% / 0.59% / 0.09%**. Median maximal supported context is 12 bytes (p90 26, p99 150; cap 1024, nine censored). Conditional observed-next-token first-byte agreement is **69.3333%** over 9,900 nonempty contexts. Centered 256-byte spans match at 5/1000, **all whitespace only**; no Wikipedia document-component fraction is established. Supplemental locate checks all 59 supported 256-byte contexts and nine 1024-byte contexts against corpus bytes. One long non-whitespace match is a statistics table; overlap is sparse and length/complexity dependent, not guaranteed disjointness.

The independent audit **failed** on two UINT64_MAX right-extension counts. Both affected feature rows are explicitly rejected; raw failures and a minimal upstream reproducer are preserved. The remaining receipt audit passes after quarantine, including 73 direct byte checks and 100 independent decoder checks; this does not certify query-engine correctness. No sxgc changes. AGC FFI has no SXI2 query API; the persistent-interface request is a handoff only. Host batch branch throughput is 326.22 successful features/s including process overhead: a repeated-batch 40M extrapolation is 34.06 hours before context search/transfer. Kernel-only throughput uses the default many-core Rayon pool despite `-j 1`; **no single-core budget is established**. [RESULT_C14_PHASE1.md](RESULT_C14_PHASE1.md), [C14_FEATURE_BRIDGE.md](C14_FEATURE_BRIDGE.md); bindings C14a. Training remains blocked on upstream query repair, a complete causal token geometry/cache, and an honest production budget.

### C14 — measured: both dense byte baselines trained; independent validation green

The byte registration and operator-authorized dense amendment bind this result. Take 3 superseded shard materialization with read-only pile.txt offsets: all 7,831,213 selected document receipts matched, with two token-truncated documents excluded. Independent validation rehashed every admitted document slice: 7,831,211 documents / 43,494,394,389 selected bytes. No corpus text was materialized for this binding or take 4. Gaps, omitted documents and capped suffixes never enter model inputs.

Both heads completed the registered seed-3401, batch-16, context-128, BF16, all-zero-consultation protocol on broker physical card 2. Emender (2,133,700 trainable parameters) completed 111,596 steps / 225,880,702 targets in 2700.022528 active seconds; transformer (2,133,748) completed 480,353 steps / 972,375,733 targets in 2700.000951 seconds. Emender reused its surviving checkpoint without retraining; transformer trained in take 4. Both complete evaluations use the original frozen 3,431,242 within-document reset windows and all 434,204,084 real targets. Calibration and final-test documents are disjoint; padding is excluded from loss and state resets between windows.

| Split / position stratum | Targets per arm | Emender NLL | Transformer NLL |
|---|---:|---:|---:|
| calibration / all | 220,107,062 | 2.277418521 | 1.421162234 |
| calibration / window_reset_1_16 | 27,784,009 | 2.289452649 | 1.723704214 |
| calibration / window_interior_17_128 | 192,323,053 | 2.275680007 | 1.377455414 |
| final_test / all | 214,097,022 | 2.276134617 | 1.404434206 |
| final_test / window_reset_1_16 | 27,031,327 | 2.288481328 | 1.711359374 |
| final_test / window_interior_17_128 | 187,065,695 | 2.274350496 | 1.360082971 |

Losses are nats/byte. Strata describe positions 1–16 and 17–128 within independent windows, all within documents; they are not support/fork strata. Six boundary/padding tests pass. Independent CPU validation checked source/offset/window consistency, checkpoint and dependency hashes, and every recorded loss/stratum aggregate: maximum NLL replay error **0.0** for both heads against tolerance 1e-8. This validates aggregation of recorded losses, not independent model inference. [RESULT_C14_DENSE_TAKE4.md](RESULT_C14_DENSE_TAKE4.md), [training.json](receipts/c14_dense/training.json), [validation.json](receipts/c14_dense/validation.json); current bindings C14.

The initial single-shard preflight failure remains historical evidence in RESULT_C14_DENSE.md. The take-4 inventory found Emender training complete, its evaluation partial at 52,289,140 targets, and transformer absent. The original Emender checkpoint had no earlier hash receipt; its take-4 inventory hash matches the final checkpoint. **Retained failure:** optional byte-exact parity against the old partial prediction file failed. The partial file had already been overwritten by the authorized full reevaluation, so numerical differences cannot be quantified; no historical byte parity is claimed. [Recovery failure receipt](receipts/c14_dense/emender_recovery_parity.json).

This is a single-seed, fixed-time dense reference; the arms saw different target counts. Consultation P1/P2/P3, support/fork strata, device-off robustness, calibration stacking and significance remain untested. Native consultation feature prerequisites remain separate and unresolved; no consultation gain follows from these dense results.

Native consultation preflight (2026-11-03): xsa `88d4e0744a05359614d28db5f0ed160a677d8ef5` was built in an isolated clean clone and synchronized ff-only; ~/sxgc was not edited. Former-sentinel counts are 0/0; “the ” has 313,603,799 occurrences and 137 branches. An independent native fixture passes 261 patterns covering every query byte, exact branch counts, EOF and closure. This resolves the former universal-transport/native-query preflight blocker; it does not supply complete train/valid caches.

The bounded extractor measured **1,034,056 positions / 8,192 training windows / 219 documents in 260.307391 seconds (3,972.442 features/s)**, including maximal causal suffix selection, complete 256-byte branch distributions and gzip output. Native throughput is 31,530.865 queries/s, with 6.802295 queries per emitted position. Prefix 16/64-byte support is 47.931962%/0.819579%; no 256/1024-byte contexts are eligible under 128-byte resets. These source-order raw-byte prefix rates are not the phase-1 token-boundary sampling population or full-stream/held-out coverage. Cache replay passes every row and rechecks the unchanged dense eval-window hash and checkpoint hashes. Eleven tests pass.

**Full-size consultation work remains incomplete.** The frozen stream has 43,052,359,095 training and 434,204,084 validation targets. Linear scenarios for this implementation are 125.44/1.27 days and 2.0836 TB combined compressed cache, versus 807.65 GB free at preflight; float32 inputs would occupy 45.226 TB. These are prefix extrapolations, not scaling guarantees or impossibility claims for better caching. No exhaustive cache job or GPU training was launched. Complete caches, transfer costs, both consultation arms, paired support-stratified losses, device-off P2, P1/P3 verdicts, and Pile-scale coverage swaps/stacking remain unexecuted. No dense model, evaluation window, budget or protocol was changed. [Serving preflight result](RESULT_C14_SERVING_PREFLIGHT.md), [work status](receipts/c14_serving/work_status.json); additional bindings C14 below retain failed fixture attempts as well as the accepted native oracle and benchmark.

Cache scaling continuation (2026-11-03): unchanged causal selector with exact cross-chunk LRU reuse, sparse binary gzip and native `-j128` re-extracted the **same 1,034,056-position prefix in 169.832325 seconds (6,088.688/s)**. Exact row replay and every coverage/histogram diagnostic match preflight. Fourteen tests pass; dense checkpoints, offsets, evaluation-window and model dependency hashes are unchanged. Storage is 34,266,350 bytes (33.137809 bytes/position); the combined pipeline is 1.53 times faster with 31% less storage. These jointly changed settings do not isolate a cause. Prefix-specific scenarios still project to **81.84 training days, 0.83 validation days and 1.441 TB**, versus 807.20 GB free. No new unique positions, complete cache, consultation arm, paired table or prediction verdict resulted. The full-cache gate remains unsatisfied and no full-size job remains running. [Scaling result](RESULT_C14_CACHE_SCALE.md), [remainder receipt](receipts/c14_cache_scale/scale_comparison.json); additional bindings C14.

## Growing design memo

### Dense mode — selected direction; implementation registered

Use a dense neural decision head in a two-system design. C4's refutation motivates abandoning the tested internal write-mask route; C3 motivates the separate exact walk. Decision evidence: C4 and C3 bindings. C6's relative sign survives but its absolute fork-aware recipe fails; it does not establish dense-mode superiority. C7 remains unmeasured. No kernel rewrite or speedup is claimed.

### Branch-set conditioning — tested joint recipe; benefit unestablished

C6 exposed a 256-byte support mask plus explicit OOD/empty-support channel, interval counts, branch count and entropy through an embedding projection. The jointly tested fork-aware recipe is much worse than controls; there is no isolated conditioning benefit. Keep the encoding and normalization as a failed scoped prototype, not a selected winning design. Decision receipt: C6 bindings; C7 requires fresh ablations and a causal selector.

### Interval-aware gates — registered placeholder

Consider context length, interval/support size and uncertainty in the gate (C7). C3 already measures a calibrated feature gate and correlated support/context strata; it does not causally establish a feature benefit. Additional interval-feature selection and ablations remain undecided. Decision evidence for existing gate only: C3 bindings.

### OOD channel — normalized prototype tested; training exposure failed

C3 and C6 retain 2,709 empty-support forks and 17,066 nonempty-support zero-target forks. C6 normalized supported-byte logits with one aggregate OOD logit and routed OOD mass uniformly outside support; empty support abstained uniformly. Probability arithmetic passed, but self-containing training counts supplied zero OOD target events and the recipe failed badly on held-out zero-target forks. Before C7, register leave-one-out/prefix-only exposure or another justified OOD objective and isolate normalization. Do not scale the C6 recipe or claim OOD benefit. Walk exit/re-entry remains undecided (C9). Decision receipts: C3/C6 bindings.

### Joint residual learning — tested; unconstrained recipe failed

C7a retained every dense position in a jointly differentiated model/index mixture. It still failed the dense comparison for both heads, with transformer index-gate collapse and poor held-out neural predictions. Self-containing exact counts can attenuate neural gradients at forks too, so merely retaining positions is insufficient to guarantee useful anchors. No constrained-gate, causal-selector, leave-one-out, or dense auxiliary-loss variant was tested. Do not scale this recipe as established integrated learning. Decision receipts: C7a; C6 remains a compound failed recipe rather than an isolated masking ablation.

### Exact index teachers — registered placeholder

Exact branch distributions are proposed teachers for mimetic initialization (C10) and designed heads (C7). Training-only support, zero counts and teacher construction cost require explicit accounting. Decision receipt: none; no demonstrated benefit yet.

## Session discipline: green means documented

Run `/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/validate_program.py` before a result is called documented and before committing a status transition. The validator checks exact committed bytes and current receipt bytes, hashes, claim order and statuses. The audit implementation binding is updated to its C14-capable commit; historical validator versions remain available in their original commits. It audits bindings, not scientific entailment or fresh model execution. The sxgc repository defaults to the sibling checkout; `--sxgc-repo` can point to another checkout containing the pinned commit. Missing evidence fails closed.

As results land, update this ledger immediately: record the protocol before execution, then bind every adopted outcome by path + commit + hash and change status only when its receipt is available. No orphan outcome receipts: every newly adopted result must belong to a claim; older run artifacts remain under their existing manifests. Commit the receipt first before promoting a registered claim, so its exact commit can be pinned without self-reference. Registered claims must have no outcome bindings; design motivations may refer to measured claims. Do not commit .pt files. Keep failures and limitations visible. The earlier no-training session left C6 registered. The subsequent C6 execution is now receipted above, including its failed absolute recipe and interrupted optional-arm setup.

## Exact receipt bindings

Paths are repository-relative. `emender` is this repository; `sxgc` is the external source repository. Full commits below pin evidence independently of movable branch names.

```json claim-bindings
{
  "schema": 1,
  "claims": [
    {
      "id": "C1",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/part1d/cross_validation.json",
          "commit": "a6111985113c905006faa451970e7754b8736caa",
          "sha256": "ac3dff82c75af68cda298b7732c36d8b7b89d7265e53727f6dc0668859b95599"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/DERIVATION_PART1C.md",
          "commit": "c3e741fc039f0466167b0184d4bc4504b101bf17",
          "sha256": "1cb9d60458a60a927ff823daf5e1928540ed81671367d93bd91bc575a5db56bd"
        },
        {
          "repo": "sxgc",
          "path": "tools/binbwt/byte_contract_results.json",
          "commit": "8be5206c24f05cb05e813be4b28c3b3a87e47274",
          "sha256": "d50db92a163ae392715b200924a3656019f2a2b793e42c31d63f0268a3f0843c"
        }
      ]
    },
    {
      "id": "C2",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/part1c/decomposition.json",
          "commit": "41249523e115ab397e7ffe3342d97d9702a2069e",
          "sha256": "288356ce247cfa97ded5e6fbb76f55448311e56f82ff6ba187e83bffba0d5814"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_PART1C.md",
          "commit": "41249523e115ab397e7ffe3342d97d9702a2069e",
          "sha256": "464354541f6b0cf0aa1e56129281c8e504b4ec4191298de6cd5b4e30ba8ca2e0"
        }
      ]
    },
    {
      "id": "C3",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/part2/fusion.json",
          "commit": "97eb18e80a8089997e514d2618b2cbd67f8f7653",
          "sha256": "63f0ca948451e6ef7950f5aad11c22eec4b7abd13d7e6d725952733242fedf99"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_PART2.md",
          "commit": "97eb18e80a8089997e514d2618b2cbd67f8f7653",
          "sha256": "29a2308a3bd3ca14cf65bd30cb906e2d631492ff468be8c2c090ad049b459dd6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/DERIVATION_PART2.md",
          "commit": "97eb18e80a8089997e514d2618b2cbd67f8f7653",
          "sha256": "1dc14d241f94576e0600e4eb4baa5b4c122fdc4d61405b41df684f9bf6ef78eb"
        }
      ]
    },
    {
      "id": "C4",
      "status": "refuted",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/part3/results.json",
          "commit": "661ef992e7e2df8af653b0b5eccb327035feb587",
          "sha256": "808066a2c7c1bf3c2f02a64cf8f80bb6de0c84282ff79363c1b0502714b730df"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_PART3.md",
          "commit": "661ef992e7e2df8af653b0b5eccb327035feb587",
          "sha256": "1eaae84b469cf104872046aacaf9180c0d067711d60cb0f391578f4c0581d261"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_PART3.md",
          "commit": "b816217e365f57023d76c0bce42a64cec89e591c",
          "sha256": "cb85bf7bb74391d1ceea7fdb148b9fe2941dcca4e8936d814e5f536380dd92af"
        }
      ]
    },
    {
      "id": "C5",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_PART1B.md",
          "commit": "661ef992e7e2df8af653b0b5eccb327035feb587",
          "sha256": "628121693b4fdfd93831f04b28457a3990f31521434a4d2e2ff4a285747a98ed"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_PART1D.md",
          "commit": "a6111985113c905006faa451970e7754b8736caa",
          "sha256": "e84c71ed380c2551fed1cdd379be0d51573ee09f4119354240f232d64929d6d1"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/part1d/gates.json",
          "commit": "a6111985113c905006faa451970e7754b8736caa",
          "sha256": "4eb5be354c482b8d8d7aed25fbab0c7c6ffd4f1e21c56513cedb7ebf71085537"
        },
        {
          "repo": "sxgc",
          "path": "tools/binbwt/byte_contract_results.json",
          "commit": "8be5206c24f05cb05e813be4b28c3b3a87e47274",
          "sha256": "d50db92a163ae392715b200924a3656019f2a2b793e42c31d63f0268a3f0843c"
        }
      ]
    },
    {
      "id": "C6",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/c3_comparison.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "c13fea0c23ae1451ef0a83a547e7a0f611ce0d6ee142a4dea07bf2b7c3d19179"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/c3_scoring.txt",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "c13fea0c23ae1451ef0a83a547e7a0f611ce0d6ee142a4dea07bf2b7c3d19179"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/calibration_freeze.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "c7e2a525acef96f4dab78f78635b8de8ffcb841227ca9c68102329f15e45a3f9"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/console.txt",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "3502cf2dd8c8d0cc4ceebf9073954dbbb7837dde4b0949684950ae217abcc99e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/console_gdn2.txt",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "253d071213a3753cf8f0e8ce41f867f8870a224eea4fd4ff1046dc14a6d38a41"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/dependencies.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "3cf3dbd8b0222b6d901d8dd8004bf6ee1481fbe2813fbb6b98cb84c3bb0e8075"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/emender_aware_prediction.npz",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "aac307eed8d657c3ba6861cfcdcf0b7362315bd904d3cb0464ca312b272689b7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/emender_control_prediction.npz",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "e5e4d3096113d694004830d3bb9f86d428b8939f02c3b8ea8a9a498e65798c04"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/frozen_gates.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "f1dda42b5d328001a1fd08a2cec9e91effa3749a5e74f24a2142903e0e7f3e9c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/gdn2_aware_prediction.npz",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "e6371a7e78da444bc51eb7f09fb29669b944512ea3edc43267aca8e181df1141"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/gdn2_control_prediction.npz",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "6d7254927a0dd756b1b04ac625e3f164fc0ceeb9ec24e5c58162c16139a3d88b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/gdn2_source.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "9d016a954a21d1d69a1413e3e17b060043c77b576617fd7992098bd8eae43e46"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/gpu.csv",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "719774c9669804ed7898598d9eb3db2f6908b748f9ee9bfd657b2bcd3e0544a9"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/index.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "b7e1cfb7f3dc3e8440a9cea8bd00f0714337c5b73f51b5c417547211152e2dd6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/index.npz",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "fffc6fb8e9215ab77f77c7053af92f85f3f04804f92feca1748b4ae34bce1d40"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/interruption.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "0a2db3093efe059404a8cd7a8071fe9b99de3d53e95196464810ceea43edfefd"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/lease.txt",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/lease_gdn2.txt",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/ood_exposure.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "4d3c178ba01285c9b9fc02888661f96b91cbcfaf238f41f49c9ca858a0fd4763"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/results.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "8fd2286904da1dafb7d76ed523f72e658d4b17b14b996c5afd8e6fe94b79c58e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/scoring.txt",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "8fd2286904da1dafb7d76ed523f72e658d4b17b14b996c5afd8e6fe94b79c58e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/terminal.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "1f31fb9d7c0d695c6c19102dbedc15ba143038f918df070bfccce009bba53a6a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/terminal_gdn2.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/tests.txt",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "0b8df7669b362c2265440d9b4992404796c6e63be49186af73b11c457944d026"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/training.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "17d3e58e553663b96b24ea4e40612be25c918c52c755c57222a49a56b3823531"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/transformer_aware_prediction.npz",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "849c6fd8903494ed2bf7955f6200833149d98ac5e4cd570b6797c13dcbfbfefa"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/transformer_control_prediction.npz",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "39980f9514467f0dfc93f035e4b5305fb7a48637c0c65e7143f461fda325c69b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c6/validation.json",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "4c743603c2a0357514784eb38ac522c4f8e291c3c5fc0344a4306601a6e1dafe"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C6.md",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "545dc645388a7cb424fa23a1b79059a0a75bccc5f4841a24f013f1457f61729e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C6.md",
          "commit": "db00edf799d51c2b9f0e877d3a8355ebc0b1a5f6",
          "sha256": "e70de380d2cfe34d4c7a66abad42a926b724da60cc7d2929d8f8313a04b968b9"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C6_C3_COMPARISON.md",
          "commit": "64a8cce690beb7facf95fe5ddedc87db8406f335",
          "sha256": "46fbbd048d6a2c6ee3cc110b36b7851233081ebd9c7d209d35617f50d85d6973"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C6_GDN2.md",
          "commit": "a6ca3ec80f710428a432bd42868937177e6e14e5",
          "sha256": "72debb3246ed183bc1dbee2becb1602ec4cb42d2e8bcb65b50e659d072ffa167"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c6.py",
          "commit": "bceb3c92b67506417f85b0cdbda8edb9f24d316b",
          "sha256": "0482c82636f59ec52cfb92aca9ca63627118a2c373496919d5326e5ecfc92ff4"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c6_gdn2.py",
          "commit": "f73b62ba432b4fb9afb809a0c6597117464668ea",
          "sha256": "21bd31860b873326a251a78ec86d9b93cd7031d8a36735812a0032e76cc2542e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c6_entry.py",
          "commit": "60116387d724189d6e645ddebc017584d02c7149",
          "sha256": "f961223cdac7c46d113ea27927ad17d28c0173379d7594aaad7e918108cd2f3b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c6_c3_compare.py",
          "commit": "d58adfd8e5c124badd691001e314270d43f9e519",
          "sha256": "62bdd239afa093d16ff1aceb8f5b08632c11575e5198480251948e66d5d8f6a6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c6_leased.sh",
          "commit": "60116387d724189d6e645ddebc017584d02c7149",
          "sha256": "a4163c3a9a7ee57e9682dcb471a5dc211d249720a748885fad6ba37c126d0d27"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c6_gdn2_leased.sh",
          "commit": "f73b62ba432b4fb9afb809a0c6597117464668ea",
          "sha256": "943bbb5865e0bc5d7020238a9dd5e74a428a520d6482bb8d9a3745d51511dcc7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c6.py",
          "commit": "bceb3c92b67506417f85b0cdbda8edb9f24d316b",
          "sha256": "f2df8cb9dc767f37162c58b25f7b492fea3f04d1da0bcdab3b99d33ab8aa7128"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c6.py",
          "commit": "e20d824a66825d2be96dd95a2dc0f3bbc95a0dfc",
          "sha256": "f194508b159b04cf5c70a2c5af47de4ed7763b4cf7c6313a692296acce963f7f"
        }
      ]
    },
    {
      "id": "C7",
      "status": "registered",
      "receipts": []
    },
    {
      "id": "C7a",
      "status": "refuted",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C7a.md",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "e2b72f2845f678cfef28c59696a9575f33835a541b842a871cd0e12ec2e3f10c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C7a.md",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "2a2b5fa7003c86facd81e5081bfd5b82260e573d22ad8d1c71d4768e64668978"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c7a.py",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "26c70aa7c67560f981cd324869a0330698859f05ba6e246bfde0ee98fbbad0d2"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c7a_score.py",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "0bc2570192e83b644578fa0eb809874ec0349bb50e8785cfd32c06a2b333cc94"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c7a.py",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "b74c4c2044cd7583978d0dbb39186f5c38fae5d55440b53ac34bb010b266caec"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c7a.py",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "114de3d72b7348acaf0bc8788c22c4cd7454d04260dbef21ff4d5297fffdf7fa"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c7a_leased.sh",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "5efc12a2bd13b24c88175de6a8685a64d49a38a69692c117d21fe8e7bbf33d4f"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/calibration_freeze.json",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "e5f27894e79b6036e21dcf9e21cbbd8fb7960d5a457926c49a64e64896d3546d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/console.txt",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "b63d5e29d5294fe37e623a9c0d13d963d288aad623cc85e7589c0f59bd5de61e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/dependencies.json",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "0310b38633f20b452b5284180fa215965809b837f1e3c1ebb838639031a18be6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/emender_boosted_prediction.npz",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "52c753beb9ca0d146d36060812e7ac54250e421bd3a84ccc887e6d52df4a9542"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/frozen_gates.json",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "19bad95715cf91578dfba73900aff1d971848daaecdb13bfc355e5e1a2ee2259"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/gpu.csv",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "c49ee0ec3d8bfd0b64339ad73910f18c46254fc593f0fe2cbaa4c14c810f2bef"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/lease.txt",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/manual_notes.json",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "1da7302b21c338372a9156678b11d2f3068bce227465a492beb60115888bec5c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/results.json",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "cefa54e889214ee015f6715d2ad9fcca40cd5822b0c84d604c906a290bbcedc7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/scoring.txt",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "d2658b458edf9e68587b0d58b970c9b66345e1faeeed03ebd82f2563dac8569d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/terminal.json",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/tests.txt",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "5dd52afefc03d5d25a95b116c9ceb3cbd05173e45d9c94059ce9e4f3d3af751e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/training.json",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "81b5e68a0ca72472c2143c6547b3a024ad6d7cd4af97fd2e4512119944de6a42"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/transformer_boosted_prediction.npz",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "da27f88e45aaca57636b879f2b6f61d472a82cbe4bad0de586017419f88af040"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7a/validation.txt",
          "commit": "370cd773a62fde8ad3b2925539c0227986e91018",
          "sha256": "926d436a77c32b825f1e9601b3b4a93d3c30693907810ad116e1b82a6259e6e5"
        }
      ]
    },
    {
      "id": "C7b",
      "status": "refuted",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C7b.md",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "afad80f815bfa0cc03ebab90b14e33f5834a3d2d714ce52acee4c03a9b7036c6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C7b.md",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "3c685cd6e59555e031379daa5c841f1a74cf85667b2a52a758876b8b65472949"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c7b.py",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "69eeb9112b02669c5c5b2157a8a6330aa5d6bc5f4dcfb4be5f890735a152c707"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c7b_score.py",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "939d0e748e91f8b92d6773517a4c7b227e22d4fc824c2b72a02fd1e14f9260ba"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c7b.py",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "e1b4a86754b579d969eac964f1c9977eceab60be74ce28ea969a9e993cac59dc"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c7b.py",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "7fae58e570d18d29bd3a3aa983cfffecff8bcf0ef4c820989ef7a305212ec47b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c7b_leased.sh",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "520850b64ca91be4022d942f991f17315e7a206c7cfae228c088a7c66c90bd00"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/calibration_freeze.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "c4853c7b93a5e43f15b56a7839c546866747c85cd6b9f50b20bac2beaefdd7af"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/console.txt",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "72939f7267c88b492909e9d3cf21cf6a1264c5b9e599faa97e97641ce032de4b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/dependencies.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "e749a43a6965056f5603687be35d5c23543917573ff27a5e22aed81e194f25c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/emender_fixed_prediction.npz",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "de9419124afcdd9d06d0728418f221fb839f970d8a8518c6ecadcc4072900bbe"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/emender_floor_prediction.npz",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "d7337a123ed80387088417d9f9c2ea3179d7c1aea5c2907ea5c62e3d68fb0cc1"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/frozen_gates.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "910d316365941cc34c1db89d89f0c5200a6081cfe1658846a39dd85b6eaba0d8"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/gpu.csv",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "c2cb3d9381da5240ec503d3371e5531fa8bf04ec2111cbb30ec7b2b8307cc2d4"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/lease.txt",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/manual_notes.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "765891f8fe8ab9cbdd35651139884eed88f1f1221f59f6d8b0a0f0d0913e5b18"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/results.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "2d44bfe2dbe45388269ce9ca8672915ae116b61fe1f7c18ce6abcc51ad0c014b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/scoring.txt",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "f008cc7868c308f9a32b0b5e3610f55ead44f2f24bd0e9ad926d785725e322e1"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/teacher.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "dcefe7bed1449a0237a32c22ac176de150b0211823ef46d762c71a5688d8655c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/terminal.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/tests.txt",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "bd58bc1df5241f5a4c7004cf6c39eb1539541ba9a4e36a5c58e28ba6e40396d5"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/training.json",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "c0dd6ceb4a9a34da30a2c3a5575e9c0be694c286a4fe6eb0361f1d2ad3c11b32"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/transformer_fixed_prediction.npz",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "516c645de5dae8c12204da7a548a6c494e95ae77c2724f7fd22e7dee0bf8a1c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/transformer_floor_prediction.npz",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "e3e36a2246c31f6eed947af7da710708d98adafb08e9c58891641063694886de"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c7b/validation.txt",
          "commit": "e702c80337d40de162e6dd923a0e75c6dbc24ec5",
          "sha256": "fd3257b7fda81ab096f8f6f97b4d77537f8f41e137c00465f6079bee8d8b2068"
        }
      ]
    },
    {
      "id": "C8",
      "status": "registered",
      "receipts": []
    },
    {
      "id": "C9",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C9.md",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "198d1aa8afed11cc99b9ab0e4989b02d9cb1c162cbfa71f8c17bf876e194357d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C9.md",
          "commit": "644bff97eb6f64fe6148d768f931fdea181a7f6d",
          "sha256": "68deea733b11b7dee6c73581c88e03bf89f55560142b68873c0f2c397b15aaac"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c9.py",
          "commit": "a8b536e063baed1fe6a20474d5a706851116b4f0",
          "sha256": "5e7ae5dfcf732cd75d3ab79495177f765f36a847fad0c22a51f3166bdc052ba9"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c9.py",
          "commit": "a8b536e063baed1fe6a20474d5a706851116b4f0",
          "sha256": "32c28d3bd5889df3dd37034092d24c9da810f4701d35fa662118785869f366b5"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c9.py",
          "commit": "a8b536e063baed1fe6a20474d5a706851116b4f0",
          "sha256": "88361f1899db5e62065cea172098db3f7b683e85fdd86d3b09940cfaeebfd8ad"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c9_leased.sh",
          "commit": "a8b536e063baed1fe6a20474d5a706851116b4f0",
          "sha256": "4f84ab8be6447d301a5d8e8891c6b285b238ced635782f571ee6c3a69aff0f9e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/binding.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "ccd743018b770ed14694148ae6e436065fb223e2e73be214ed3cf96421577d68"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/console.txt",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "fbfaa6d56d0766d94964c800383e548a92668e40280a42eefc5916ed5d9ae04b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/gpu.csv",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "d6c64825b01e5fc039f91f1eca6ee86cdd247ad6a6d36a7ecfc3f807374c0432"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/lease.txt",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/manifest.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "a3db6b54dca51481449254d109d69b4760fa1ce5b08c4ec501cb0ee6fcee6071"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/neural_only_0_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "6c2e34bb7061d08af6e7da668035d8dcf6cf8df786aa0ded0256b121ab4b301f"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/neural_only_1_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "3735a16c628cc2981e25f3368b33469ee6ac21155fe00a54ac7d070721130754"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/neural_only_2_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "4fa75d4832338e6b06d3c70f472e6fc158cb338d644e2c0a5fc5283af466c18b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/neural_only_3_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "02d95237f5b5f1ec1f4a924f5d204a621256611bb1366530da2fba99d0ef36f2"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/results.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "b520945c00827d9d3fb7ad5304a3ecd0d5d044dc49f420d3b7621387e1a26f70"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/samples.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "3745e293ad9d2736d30ab196aa224172d6dc5e1fa79869d716f1ab3c9efcdef2"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/terminal.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/tests.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "0cb04c97574c86bb0f71ca9c70f4c66d44793de1061b9c84b78dea1639ca7c38"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/validation.txt",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "1345bad3a3c75d812cc1758aa46a84740d9abce1ab6ce65b547e0b14b686a413"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/walk_fork_0_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "18d63d0bd23e502533eeaddbe2a275f3953c2382165f8e818be1b4c834a37114"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/walk_fork_1_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "a807d45a885f270cf33695da9b8e5183dc187ba4cd6439a9273421d94560bb5f"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/walk_fork_2_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "36548fff4dc128219410230c0c95a38ebfd91dba32723f8c74e7a6762eb3c2ab"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c9/walk_fork_3_steps.json",
          "commit": "0d9b8bced36cb09be85d0228cb006e14972dbcbb",
          "sha256": "a6efd2e5e866e7b5194d77c4ed3a359cd89a3e218f22e75b037181529506d026"
        }
      ]
    },
    {
      "id": "C10",
      "status": "registered",
      "receipts": []
    },
    {
      "id": "C11",
      "status": "refuted",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C11.md",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "7331f6950c0622cabad40d3e5b367f985d867b9b928053b59329618b1e76a700"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C11.md",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "e682a17d36ec33630096bfbafa89724974eb8282df7d9d175265c46e0424f8e6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c11.py",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "0be42152ea88ed9979ea2c1e106fadbad8db6fd8108b4c9a8f8360177597fac3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c11_score.py",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "7fd212ab928418ee6e2dbeafd2ad12449e17a2e2eaa542ef0c9d11708cd862eb"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c11.py",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "40e3ba3bb27722a7e9b334ee290c95cef03739f8e056640c103d61e5a3123f37"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c11.py",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "d889a9aa7b2c1e1baec87b1ad7b4506c6b9139c5e4c59da18bcb14e663793a3d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/calibration_freeze.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "5010177035bc57d97dfe010521890b66db9c0742842a58d798e7dcfa91c20a65"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/console.txt",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "49f28d17a680749b2981fcecb2a8bf745a4d6ea38b0f41f133d512a06d47a6d8"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/device.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "43abf401305855286476f97e5b590ace6a7b16ac18ae9b3d4a432b1b759eac8d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/device.npz",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "02cd2fde202d95bece52de3a7a21bd02739856a864bbd4a1e67156f1bf8c2c9e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/emender_ops_prediction.npz",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "1c6e3aed49114b583824c094d962b40d176891603fb1f2193e3e081da32ba90a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/frozen_gates.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "7f23cc794f03a676f98d006bb78dff32f32351e63d7c064213a0b1ccc9f50bee"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/gpu.csv",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "f98073353f895f8074c5c9c06d700ada1f22782fc4d6e9f409ba27a4a80d2456"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/lease.txt",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/manual_notes.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "26badec12de7a01dac6c095709e813ae0f650247e5fc8ee84aa65289200d8c5d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/results.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "fb3d5224fc2f425df8c6e6c62d20af7983c204b1dde6f56f04f12def4dd97580"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/teacher.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "dcefe7bed1449a0237a32c22ac176de150b0211823ef46d762c71a5688d8655c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/terminal.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/training.json",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "730219d2efd593655eaeef02bc871f0011a52bb68620963af676cfc55975c0f8"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c11/transformer_ops_prediction.npz",
          "commit": "70d5faa2f551d4371c4af287ddea8e5363dc2564",
          "sha256": "cf8e80fde6c3870cc97a755f91594a14dc356fdcf36a1b5a04a999a073450283"
        }
      ]
    },
    {
      "id": "C12",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C12.md",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "6db6b9693cac175719ced853d0b420e3a3965eeb5263b31d9e166e1fea6346e9"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C12.md",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "3a7b1e86f6fc7836025cf2c2527416efca7f27a38e1baca3d83f43862483837b"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C12_CONTROL_REPLAY.md",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "1ee68ea51f96ed32a6bb124bf4095ea291180d907de8b6ab061e9f12699547cc"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c12.py",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "c2d95d2d47489bb1d114fc069a811e56acb7dca00742cfd2234fb3b0198cb842"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c12_score.py",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "7204289336657cbfb7008eb00075f35d9e8c04bfcf50a6e542a458d8a667b499"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c12_control_replay.py",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "33938c72a40e5a3375ca71ef6a3d637cd794cb51597f47d72dbe6e0114fedaf5"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c12.py",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "005f1435bf2aa7453cd2d6acc58a080d8e21a52ad02b9ca9aae5313229854632"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c12.py",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "51f7b738254078bb0fb36e992990b89d64e26a64e11194b12a79fe31d78305d6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c12_leased.sh",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "a9c24b3c2312c050d87fc0cf05dd73a44c79cace41841635721f7132606c5988"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c12_control_replay_leased.sh",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "5a39579c56a9c9415ea197319c696d5ab3124ab204bd3925bd1cef5ea12d9540"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_program.py",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "9835ed73b212a0f7b26d30d9eb7fde85c5d5d45f8016e81904a2d4c6401391a2"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/console.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "af89acfbaf7f766a5e9ca9e84d4d819ea84b7064409063db8370c74340300b46"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/control_replay.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "57e7ace3e148285cfa1983b19d0cb91402b3bd9353471f7a540026a77b28fdac"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/corpora.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "825f6dd80229142e8d813c2003855468977e07934be9213b29823bdfe66e6cab"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/corpus_x.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "7124fc713f7da52adf8a10fda02eaa9a9a3e1f16cd622e4abe85f9d6e3d97686"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/dependencies.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "f46d52aad013eb9d6cf99e0ef9a32010f359af90d20237544a7124e4c8c65e2c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/device.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "1f13701b243c0f916a84b30cdabc222b23dfd9032d83ac266bceb9596edc9cef"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/disjoint.npz",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "1ef70ce4d3d00781864b197e89dbf81182ee120e788573dd38b1400585a054cb"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/emender_c11_replay_prediction.npz",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "1626ddabb2b155a7a1e5683748c0f729d1b57b3ba7ddb80853dbeac44823e756"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/emender_disjoint_prediction.npz",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "f47eddbffc69d0646c4db89ad04440a12e1fd52fabe96ffe92d23d72778460fe"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/emender_masked_prediction.npz",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "ce86dd9d14778ca1d02b1e864987852626df3c921651c8551c52bfe1db1ce55a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/gpu.csv",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "282c222785b263b14ae3fddf10c44283dc11689badf8ec778bd33a23b101ce77"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/lease.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/manual_notes.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "c914260632f54805aa70a8a688414078542e5b0f16232f1cf61b582c018cdae8"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/prepare.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "0b06b2e3a330bb91bb2a413c9cdb053dd7d154cb92f87f9e4ef0361381e24566"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/replay_console.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "9afe18b427eead43e406c2dbf0e502ab694b06a8c09236a36ed6f24da29b2ef7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/replay_gpu.csv",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "7c9af04ebe18687b914176c8347a6e54e52decbd27d4e0718c74f71a30a14567"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/replay_lease.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/replay_terminal.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/results.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "a85895e0041fffdb37cc69a8a1d8368ce2a35430410c6ac5dacc4a9da61c68b2"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/scoring.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "f9bd324c12c48c7d21034143463d2c5f068c300c82e0f06547c4fad5d0f7a036"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/terminal.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/tests.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "2eb452dacbf8dabb8764b982e8852984d8ec0f97c12671806cdc494d8a024537"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/tests_initial_import_failure.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "77a3ad175e044932aae668734ece1b1f4573922dfd40c76a65ba3d827465fd0e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/training.json",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "2f34aded82b64da09acdb69b1e7b8c448bdfc25ac346d4e27edd863ba3ae8116"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/transformer_c11_replay_prediction.npz",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "eb66ee0cb209a0cbda8a37bdc1f8a461a52180adc60ea8f997be1a1c51c55acc"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/transformer_disjoint_prediction.npz",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "8cb59b73ca45fac3b5df88c1c73c35d5529b494266f5e2c20f69a21edcc47f7d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/transformer_masked_prediction.npz",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "b763bd4d411c96e056d41e04265b5b889cea46a23d57c728a33b06f221054561"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c12/validation.txt",
          "commit": "6c06d27d448642edaf6bad10b96527f2113dfe8a",
          "sha256": "86a3b0631384c7b6adeddd53de4923b5641ce1c6d53390609bacb8017e034956"
        }
      ]
    },
    {
      "id": "C13",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C13.md",
          "commit": "67c26772f620aa6b44733043f397d5f343f4e198",
          "sha256": "d8da80b016bb619a04d5c811597d19c8b587c4ea6ac538436bac88a3d5d0778f"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C13.md",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "87eafa52e7cea44021e0aaccb4347d5e41bca82f4ad008135148e778421db29a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c13.py",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "d26ce3b7437e061a38813708e2fbc60f7502f326c1a56846faad32a87adcbf27"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c13_score.py",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "4dbb29fb58e7715cbbc09a2c7b6c91c4633613a45c70686c05d898a5e731f410"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c13.py",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "01cacf81f55f3a891c790d9c13fe0f1cd00a673b534eb949029f0007ac1ae9aa"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/run_c13_leased.sh",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "e3d0ff2fd5cf1c8c5b12fd51d506243630afef755566bc691ae7d41bae6ad5ac"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/calibration_freeze.json",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "02447680b2171e4bc889cb44cfc0d2d5297994b4f0bfa57b8e0fedd213582049"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/console.txt",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "8598dc85c00e0d788156f34510fe707b6b42534bff9dc3ed7624822eff3b7bf7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/dependencies.json",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "162694daa580f068d47e93f5c8232b3e06385ffd4617548dd416e3a7e60e91cc"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/emender_self_prediction.npz",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "0f50ba790597d951e27fa5be4f652e7dafcb74e1f6aa566458644e08c2f62e00"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/frozen_gates.json",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "9755e9cf1b6c0a64386447125de4fd637796d1b3cfcb9413df0c0a46d828cb80"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/gpu.csv",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "05ee5c2570903fc36928369de5560286856ed5457d49a8026cb771646e2aea07"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/lease.txt",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/manual_notes.json",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "0ce5b63395c8aaa3e14e7929c7d0e608ba13791cdae571c8571a13509ea48d53"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/recovery.json",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "7695a91bb7814c87f1dabd338181a7064a1c876fc5baea45a5edb68f04146222"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/results.json",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "40e418adaa6682ca3c165f4b42b252f2f10890ac8d5e84a8cdcc5c5b26751dfb"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/scoring.txt",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "40e418adaa6682ca3c165f4b42b252f2f10890ac8d5e84a8cdcc5c5b26751dfb"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/terminal.json",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "d24219f6de94b955cf36d25bd2aca88e2d35eef45b9317b7083f55ec8fe9653c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/transformer_self_prediction.npz",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "409a764c442e69a017465f6a3d0d9a7a631db0117e31cac9d391fe7afc398b1a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/validation.txt",
          "commit": "ccf76f92166a644fa9c5d9f6d2ca9fa7c3605c87",
          "sha256": "69103ad8bee4fc0c03a5ce4e2f94e8841165525777cb501a4e40da6ea40055c5"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c13/verification_notes.json",
          "commit": "9536e54871d4909c0e3b7aa8cebef6bb033bde8e",
          "sha256": "396af148e050b9d15dc3a1589e1f4257400fa21b7e2a4d92deb3d6e83b3b57bb"
        }
      ]
    },
    {
      "id": "C14a",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14/MANIFEST.json",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "377c3457b68e1936b8fa623c5861cc71d30f8a663e812f33acb049033d6e8fbf"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14/coverage.json",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "b692b6bf2503c533bacea9e03b5de811a170c9b9b88a4d3ca39e152b885d40fa"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14/validation.json",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "f31d71655b6af1da5db1bb2ac784790cb7ca53b666ec6b63fe2f988aaeb89c04"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14/count_range_failures.json",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "87bb18655f3e6c7c772c75f634e5e14f05548af337de47829be7daf2cbf7439f"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14/initial_validation_failure.txt",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "e7db0b9da65219a20cc3355a10c0ba6e35e9c0ad3be35262b7ea6f728be7d8c7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14/budget.json",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "37be33d58f1d866dd490e75a71463f10544131ee5ab011f129edba71746aba69"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C14_PHASE1.md",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "8539962e9f43983580fde72d0818f9a686d348597f01136955576328db991b06"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/C14_FEATURE_BRIDGE.md",
          "commit": "6ff620c5ffe8b531a10ea5352bb7f8f7ba5c3f81",
          "sha256": "323f146143846cf8aefa61272a058d0db5f48900adad11d3b6a71f20d4bdc089"
        }
      ]
    },
    {
      "id": "C14",
      "status": "measured",
      "receipts": [
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C14_DENSE.md",
          "commit": "8b54a9b2b1733a45f8e7d35053bd9a4ab07ef2de",
          "sha256": "254658dced3446a69af7a05be1ff321e4c94f11eaa4de36580abb26df6302412"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_C14_DENSE.md",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "9ebc8e6db9030197d81245987a0d92203a6afda6f4eab11a476731ea58cf4c3f",
          "supersedes": "644bff97eb6f64fe6148d768f931fdea181a7f6d (the original dense-only registration; amended by take-4's recovery inventory commit 024ed3f5, results committed 76365ea9)"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c14_dense_source.py",
          "commit": "024ed3f554c40fc9101f0cd9420ba509a85e99fe",
          "sha256": "711b05a8cf103aa5d0f60b78712d761b702fd342516210e4d9ad6b4c369b6de6",
          "supersedes": "644bff97 original; re-pinned to take-4 pile.txt-offset rewrite"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c14_dense_train.py",
          "commit": "024ed3f554c40fc9101f0cd9420ba509a85e99fe",
          "sha256": "d19e98190d5bb2bbfce02c047dbe386960a577ddee4fbc4ffea23174ceec14c8",
          "supersedes": "a8b536e0 original; re-pinned to take-4 pile.txt-offset streaming version"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/source_preflight.json",
          "commit": "024ed3f554c40fc9101f0cd9420ba509a85e99fe",
          "sha256": "c9c016a662f2d871d41fbe06a8d6198395394afad9e7f25d3edee1a8bff5e2e4",
          "supersedes": "8b54a9b2 shard-00 preflight; re-pinned to take-4 full-coverage preflight (pile_span superseded it)"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense_source_console.txt",
          "commit": "8b54a9b2b1733a45f8e7d35053bd9a4ab07ef2de",
          "sha256": "e6b751cd730f8d122bd8ed3ca8c8c7056504a6c051dfe250dcd0483a59740990"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C14_DENSE_TAKE4.md",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "e4d9df4f2996968d5aa661b41e3ad918575ac9ec38923617bbc269a0aa76a85a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c14_dense_train.py",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "d19e98190d5bb2bbfce02c047dbe386960a577ddee4fbc4ffea23174ceec14c8"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c14_pile_span.py",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "7e1f64a20418d5c41ce4d85ea9c1ad9f6bc90e6720f5f5773660b0a46727f8bd"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c14_pile_batches.py",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "63980bef3f3e5a35dc4bace9f7f4b6d9098949ff47c1bc161ff6ceb89888b8fe"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c14_dense.py",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "c34d9aec058ba12371d6090d529d6a70deae1371b487f6b052ed4776531d09dd"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/pile_span.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "c02e7882f19f2b3841c1fbd1899d7affca5d4f793742cf74307ecf8d55650668"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/eval_construction.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "4b3ff2605e876c726720e8bd28e43417339c5b5f4859e67f35c63cfc1455ec11"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/inventory_take4.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "7cb3296bffcc387480f5ad046a877dc642b2e2c265f14a4ece24966935be21a8"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/training.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "b07127aca785b0a4dda337c3b5168d20a6c4f89d2fdaecaa371b366743dd68a7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/checkpoints_take4.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "0667b20378564684a2b3db76dfe6b063d51fecd7382b233a23faf6df08bb1607"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/validation.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "862bd027748b08f28d1aa6116322844701e8869e51677d489ac135da44a831bb"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/manual_notes_take4.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "4606adbda8fea338048cee114fd8bdab58e13ec8645e20de89de6a33fdc318f0"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/emender_recovery_parity.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "1e52a52eddd92ea5a994db595f771851cbbf099d22d7cf9a29f75017c54e73d7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/boundary_tests_take4.txt",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "0c9ddeda3b88edcec874f692d726140f408d546e357fd0dbea32511639573f4e"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/continuation_launch.json",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "888b1320217b59ae4e94bda437da47b3efa6e4b9ca2d9665b354b08ab2c7f04a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_dense/training_code_take3.py.txt",
          "commit": "76365ea9e00a50f6469ca956a4946b40a70602e7",
          "sha256": "6b74a86356850fcccb0aa61b8a844a18717e82d302610e307f3b41c2a9f5cffc"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C14_SERVING_PREFLIGHT.md",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "a2e73360b4452ae4ea5c5b6523d26b1c9a061415b035f3fce12338e7abec29b6"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c14_branch_serve.py",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "23c48b72b9b5506121c142294b6c6de221d7cef2863d869960daea40d468a0e1"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c14_serving_verify.py",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "ef20cd6c2d5c67dfdc93e5beac02b5c568278cd21a7e6e3e6b6232fafd1015c7"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/validate_c14_serving.py",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "eb67d306ac2746de0978025238ddd702e1da96dbdee2a5632e8a95331003a4b1"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c14_branch_serve.py",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "053798d42ad09bea20c1f36f0401437e7c8ee75212f5475efae686b4b06f9ae0"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/MANIFEST.json",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "3761064a408d61421b604e3da08345a05b9f4464e4330f865a1377984840310a"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/benchmark_summary.json",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "55d357e8fc85fa8c8ea7e0f6e5ff142e2e2fd16669a01c21a5f82cdbf9d82ef1"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/benchmark_validation.json",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "65c1bf2b92bd561c14eb5d8f4f3c611ad5ce177a96c67fd8b0a44f098c616750"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/fixture_random/fixture.json",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "94056881452de6009764079fa33b6b7e3f74c89f384e20dcd2ea99ce75a38e4c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/provenance.json",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "a371bdd4953059be6a08ff8b1b8dd8d4b107408824c378344ee625ba4fe07427"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/scope.json",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "a31f45deb59b330d617e98c1f0582104ae67c8e1cd34d62bb07221c80f6c91f2"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/work_status.json",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "0be7207c03a8f74b03ed14b3549ebfcc7f1cd063570e9f8a4335736cec8f2e13"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_serving/tests.txt",
          "commit": "e437d7a2b0856950b6e762746f99e15463f30830",
          "sha256": "36b08a93f1957176ff11c4c03f47d90aff7719b9d896e276a6291fc2d9e0ad67"
        },
        {
          "repo": "sxgc",
          "path": "xsa/src/branches.inc.rs",
          "commit": "88d4e0744a05359614d28db5f0ed160a677d8ef5",
          "sha256": "badad614b0a245bfdf10c771d7a873da1434f8858827e48d0c4cfeaa0ef17778"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/RESULT_C14_CACHE_SCALE.md",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "dbeb7d204d93df8953e7b8d81ba3dc3e87c8310e35f0847988dd3e7cedad085c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/c14_cache_scale.py",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "1f5feecebf57f93dd150a46ac38a1cba416055a6e75fb0b36f7359db305662ce"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/MANIFEST.json",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "384f0c46d1bf659a5ac5ad659e5f81476b73f4bd20f6eb1170822e71772219bb"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/benchmark_summary.json",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "de61c6f3bd3c4e810dcc6f257badb21d480d73e28b48e3c6f2ae3c70c17f2f54"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/extraction_console.txt",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "b92e6b2b531e30718a6ba15b4358b5ac35b3cd2e3dbd9f90a94e1375e41d15db"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/frozen_identity.json",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "9f4a8eea2745d497d7fd3a7b585cce3e17de36f98a54c22b04c2bb908622afc4"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/manual_notes.json",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "ea5f9c8718b7e66498b1063bf96426e853fd9d4b95a8f78aa0bbd3a03a1b5d64"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/native.stderr",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "e60f3bb9e2c041798a4e887f6d2a3f9035d1275afda25b90820958c002a8cfdf"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/parity.json",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "df5668d02702b57fdb5b150216be05c9a5c6174658fa70971ef3ab43dcff7664"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/parity_console.txt",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "6134606fe15c1d4d6ed734bb936e6847a39ccc73ec7625691f9bdc0fe3df556c"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/program_validation_before_binding.json",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "1a67b9a7d74d526e017b560f2fe6a9a3e2768b839db3703f58b97f66f8830719"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/scale_comparison.json",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "1b2894268234ad694725aa48608cabe80fd4a24c53b00931c3f6bfe028e79c7d"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_cache_scale/tests.txt",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "11cbd5b9e997fb4afed7c63e6e392881ac3c76abe4f66736712fa82e12d8c854"
        },
        {
          "repo": "emender",
          "path": "experiments/expressivity_tasks/x43_fork_resolved/test_c14_cache_scale.py",
          "commit": "92f7544fe67ef962de9a790aa3b2ea92d3018016",
          "sha256": "6a2970db7f57b6a74d5184d30d61caab731ba098e9d777c6dd3b8aceaf9cc5a5"
        }
      ]
    }
  ]
}
```

## Operator-authorized dense and generation rungs

PREREGISTRATION_C14_DENSE.md permits dense-only launch without the consultation
bridge. Its later zero-materialization pile.txt binding supersedes the historical
single-shard failure: both dense heads are trained and completely evaluated, and
independent validation is green. Consultation predictions remain untested.
PREREGISTRATION_C9.md froze prediction, prompts, budgets, measures and re-entry
before generation; the C9 result and raw sample/step receipts are now bound above.

## C14 registered-scale stream audit (2026-11-03; blocked)

The frozen dense receipts contradict the requested approximate 19M-position
stream. Replaying the source-order document-offset sampler through each recorded
step count reproduces exactly 225,880,702 training targets for Emender
(1,785,536 windows), and 972,375,733 for transformer (7,685,648 windows).
Both start at document 0/start 0; the last windows are document 41993/start
112512 and document 179298/start 17792 respectively. Seed 3401 initializes the
models; the sampler does not shuffle. The shared paired evaluation contains
434,204,084 targets in 3,431,242 frozen windows. These are receipt counts,
not new inference or coverage measurements.

The bounded union of the two nested training prefixes and evaluation therefore
contains 1,406,579,817 positions, rather than 19M or the full corpus. Neither
training prefix nor evaluation was shortened. A fixed time budget can produce
different prefix lengths once consultation processing is added; imposing both
exact frozen prefix lengths and 2700 seconds needs an explicit protocol decision
before training. Per the stop-on-change instruction, extraction and training
were not launched. No caches were allocated and no dense artifacts changed.
P1/P2/P3 and C13 remain untested. The exact replay, identities, source hashes,
limits and unavailable supervisor channel are recorded in
`receipts/c14_registered_scale/stream_audit.json`. Uncommitted assembly work
was left untouched. This audit does not waive any launch gate or claim the
requested paired table is complete.

## C14 registered-stream resumption (2026-11-03; incomplete)

The operator resolved the preceding stream-count conflict in favor of the frozen
dense receipts. Exact training prefixes, paired evaluation windows, seed and
registered budgets remain binding. The earlier ~19M estimate is superseded;
the audit's 1,406,579,817-position union stands. No dense model was retrained.

A separate timing premise needs correction: the banked 121,430 features/s is
**assembly-only**, explicitly excluding native extraction. The existing complete
selector/cache benchmark measured 6,088.69 selected features/s. Linear projection
at that prefix-specific rate is about 64.2 hours for the union and 10.3 hours for
the Emender prefix, rather than 3.5 hours for the union. These are scenarios,
not full-stream measurements. See `receipts/c14_registered_resume/resolution.json`.

`c14_registered_cache.py` starts the exact Emender prefix in bounded attended
slices, retaining immutable sparse binary shards outside git. Each write records
its exact size before allocation, uses exclusive creation and `posix_fallocate`,
and checks the 150 GB free-space floor before and after writing. No files are
deleted. Resume verifies every completed shard's SHA-256 and extraction identities;
unreceipted shards fail closed for inspection. Every chunk checks all compact
rows exactly and independently exhausts all suffix lengths at three hash-bound
sample contexts. Source windows use the existing per-document hash checks.
Completed chunks are progress, not complete arm caches or training acceptance.

The paired table, coverage-calibrated P1/P2/P3 verdicts, consultation training,
and C13 coverage-scaling/stacking remain outstanding. The runner currently covers
only the Emender training prefix; evaluation and transformer continuation need
phase runners before their extraction. Full caches and launch validation remain
required. No GPU lease was acquired. The enabled catalog contains no
`contact_supervisor` channel; no supervisor reply or approval is claimed.

The first attended extraction slice completed **1,424,203 positions in 11,264
windows**, in 242.04 seconds including per-chunk parity and write guards
(5,884.18 positions/s). Its 88 immutable shards occupy 134,849,956 bytes.
All 1,034,056 banked oracle positions match row-for-row and feature-bit-for-bit
(SHA-256 `70971034f5cf8ddf79e0f91c8bb578d09ced536e1f0a3d8a0a89557dc93b4c73`).
The 261 native fixture patterns and independent selected feature bits pass again;
17 targeted unit tests pass. The ledger validator remains green (17 claims,
264 historical bindings); this does not certify the incomplete consultation arm.

`receipts/c14_registered_resume/resume.json` records **224,456,499 remaining
Emender targets**, the exact next window index **11,264**, cache hashes, free
space and resume command. `extraction_journal.jsonl` receipts each completed
shard and its sample checks. `parity.json` contains the new verification result.
The extraction process has exited; no training or background extraction remains.
The full work order is incomplete and no completed-arm or paired-table verdict
is claimed.
The complete 42,352,061,848-byte source index was also rehashed against the
frozen byte registration: SHA-256
`9f2b3194b1c270c77886622dd528d7f8df142c761ae79417efa175ba1ead0462`
(`receipts/c14_registered_resume/index_sha256.txt`). It matches exactly.

## C14 managed union extraction (2026-11-03; running, science pending)

The operator accepted the banked **5,884.18 targets/s end-to-end** rate and
closed optimization work. `scripts/systemd/emender-velocity-c14-union.service`
is enabled and running under the lingering user manager (`Linger=yes`), outside
the attended shell. It uses only `/home/erikg/emender/.venv/bin/python`, CPU
queries, and an empty CUDA device list; no GPU lease was acquired. Extraction
order is the exact Emender prefix, immutable paired evaluation, then only the
transformer continuation. The transformer cache reuses the identical Emender
prefix, yielding the audited 1,406,579,817-position union. At the accepted rate,
the arithmetic is **66.4 hours** for the complete union; the earlier ~64-hour
projection used 6,088.69/s. This correction does not reopen optimization.

`c14_union_service.py` retains the proven legacy Emender extraction identity and
resumed at window 11,264. A managed stop/start subsequently resumed exactly at
window 13,952. Committed shards remain immutable and hash checked; an interrupted
uncommitted shard with a durable write intent is retained outside the committed
namespace and replayed. Torn final journal bytes are also retained. Unexpected
orphans or committed corruption fail closed. Every shard retains complete compact
row parity and three exhaustive suffix checks with context/feature hashes. The
150,000,000,000-byte floor includes the pending allocation; exceptions stop the
unit with a failure receipt, whereas abnormal process death is restarted with a
bounded retry limit. Bytes are journaled per shard and summed at each phase
boundary. An exclusive process lock prevents concurrent union writers.

The launch snapshot at **06:29:09 UTC** records **16,896 windows / 2,138,614
positions / 355,314,089 cache bytes**, with **803,231,301,632 bytes free**.
The latest shard hash was independently rechecked. The post-restart startup
slice measured **3,521.45 targets/s over 106.62 seconds**, below the accepted
baseline: rate matching and first-hours acceptance are **not yet verified**.
The service remains running, without further optimization. The baseline scenario
projects the Emender boundary at **2026-11-03 17:02:53 UTC**, paired evaluation at
**2026-11-04 13:32:45 UTC**, and union completion at **2026-11-06 00:47:10 UTC**.
These are projections conditional on the banked rate, not completion promises.
Document-dependent branch density, host contention and disk usage can change them.

`receipts/c14_union_service/` retains launch state, exact resume index, journal
snapshot and hashes, console, systemd state and refreshed parity. The complete
1,034,056-position banked oracle again matches feature-bit-for-feature-bit
(SHA-256 `70971034f5cf8ddf79e0f91c8bb578d09ced536e1f0a3d8a0a89557dc93b4c73`);
261 independent native fixture patterns pass. This verifier captured the original
11,264-window prefix while the managed service started; its journal digest is
live-time evidence, not an atomic complete-cache certification. The separate
launch journal snapshot defines the exact committed prefix above. The service
rehashes the entire native source index at each start. Frozen paired-window and
offset hashes plus the first three document-bounded eval windows were checked;
unit tests cover cross-document reset prevention and frozen identity rejection.
All 21 targeted tests pass; the program ledger remains green at 17 claims and
264 historical bindings. The initial new-test collection failed on relative
imports and was corrected before the successful run.

No phase boundary has arrived in this session. The service receipts boundaries
and continues extraction; it does not launch unattended training. At the Emender
boundary, the next attended session must apply the registered consultation
protocol, features as inputs, tier-1 feature dropout, and broker cards 2/4.
Paired evaluation requires its completed frozen cache. No dense artifact was
changed, no checkpoint was committed, and no consultation result, paired table,
P1/P2/P3 verdict, or C13 coverage/stacking result is claimed. First-hours monitoring
and rate acceptance remain outstanding. The enabled tool catalog has no
`contact_supervisor`; no supervisor reply is implied. Operational and recovery
commands are in `scripts/systemd/README.c14-union.md`.

The final launch receipt (`launch.json`) supersedes the startup snapshot
(`startup.json`) for resume state: 19968 windows / 2524696
positions, next window 19968, 390170471 cache bytes at
2026-11-03T06:31:44.669784+00:00. A final managed restart resumed at window
18304 after binding the native executable to the frozen oracle and
requiring a full index rehash at every phase. Its short slice measured
4473.92 targets/s; baseline rate acceptance remains pending. The current
Emender boundary scenario is 2026-11-03T17:04:23.408194+00:00.
Manual start-limit counters were reset after installation checks; the service
was left enabled and running.

## C14 union extraction first-hours verification (2026-11-03; early snapshot)

`receipts/c14_union_first_hours/REPORT.md` supersedes the startup rate/projection
observation above. At 06:48:12 UTC, the service remains active/running, PID 1821314,
NRestarts=0, with **57,600 windows / 7,284,742 positions**.
All 450 committed shards in that snapshot were rehashed; three newly extracted
document-bounded windows and nine positions match the exhaustive native suffix
oracle in exact rows and feature bits. Eight targeted tests pass; validator green
(17 claims, 264 bindings). No extraction failure, service stop or rate optimization.

The original 4,473.92/s slice includes startup; the same slice after its first
completed shard is **6,609.01/s**. The longest interval outside the verifier's
extra native process measures **5,612.91/s over 303.20s**,
4.61% below baseline. Earlier slower blocks overlap external benchmarking and
memory pressure. The full invocation including startup/diagnostic contention is
4,814.02/s. This supports a startup/contention/content explanation for the
short-slice loss, not a demonstrated persistent implementation regression; the
remaining stream's baseline rate is not certified. The clean-rate Emender scenario
is **Nov 3 17:37 UTC**, versus 19:25 including diagnostic contention and 17:07 at
the accepted baseline. These are conditional projections.

Free space is **802.69 GB**, above the 150 GB STOP line. Cache bytes are
856,216,169; recent-tail and cumulative-prefix union scenarios are
137.86 GB and 165.32 GB, respectively, around the approximate 159 GB plan.
That plan is not a guaranteed bound. The diagnostic probe's loading interval was
excluded from clean rate accounting and its temporary host contention retained
in receipts. Observation covers the first ~18 minutes, not a full hour; ongoing
health and rate remain operational risks. All caches and consultation science
remain incomplete; no boundary or training acceptance is claimed.
