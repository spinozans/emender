# X43 — the fork-resolved language model: living program ledger

The fork-resolved LM uses the corpus's exact combinatorial structure (xsa/chi) as its deterministic spine and EmenderLM as the organ of decisions. The complementarity hypothesis is that the index carries what the neural store is weakest at—exact corpus continuation—while the certified emender carries what the index cannot: out-of-distribution (OOD) behavior and abstraction. This is a program hypothesis, not an established generalization result. Paper 1, [X41](../x41_unified_report/PAPER.md), establishes the memory's scoped boundary properties; this is Paper 2, the decision space over sequences. The chi formalism's two-sided bounds are the intended theoretical spine, and the receipts below are its empirical legs. Exact structural decomposition into walks and decisions is the proposed distinction from kNN-LM/infini-gram style retrieval/interpolation; no novelty priority or comparative performance claim follows from this positioning.

The theoretical bridge is still open: DERIVATION_PART1C.md defines finite-body per-position forks, not minimum suffixient cardinality chi. There are 636,727 fork positions but 425,221 xsa witnesses in the historical container. Do not substitute either for the other or assert an unproved bound. Formal two-sided inequalities, their constants, boundary convention and application to decision cost remain a registered derivation task; no theorem receipt is claimed here.

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

### C9 — registered: generation character

Prediction: the walk/fork alternation loop increases corpus n-gram overlap versus neural-only arms, while an explicit OOD channel permits drift and re-entry. At fixed prompts, sampling settings and generated-byte budgets, report n-gram overlap, longest copied spans, OOD run lengths, re-entry rates and walk/fork proportions; separate novelty from plagiarism judgments. Freeze n values, prompts and re-entry policy before execution. No generation receipt exists.

### C10 — registered: index as teacher

Prediction: mimetic initialization from exact training-only branch distributions lowers fork NLL or time to a preregistered loss threshold against random initialization at equal total training budget, including teacher cost. Keep held-out bytes out of teacher targets and distinguish support from OOD. Freeze distillation objective, temperature, smoothing/zero-support policy and initialization duration before execution. No teacher-training receipt exists.

### C11 — refuted: dense device inputs fail both heads; swap untested

P1 raw fork, P2 empty-support preservation and P3 fused-total improvement all fail for both Emender and transformer under 180-second budgets. Exact distributions and interval features entered the residual stream; full dense CE and the internal memory stayed unchanged. This closes the tested input-channel/dense-loss cell negatively. Dense head gradients cannot bypass a learned mixture gate here, yet shortcut conditioning still does not preserve OOD quality. Do not equate routing-free training with an empirical OOD guarantee.

[RESULT_C11.md](RESULT_C11.md) contains both heads' fused/unfused total/fork/OOD tables and steps. P4/P5 swap and 100/200-step adaptation remain registered and unexecuted; no swappability result. One seed, corpus-wide transductive context selection, self-containing train distributions, reused historical dense controls and unequal step counts limit interpretation. Registration cc137968, [PREREGISTRATION_C11.md](PREREGISTRATION_C11.md), SHA256 `e682a17d36ec33630096bfbafa89724974eb8282df7d9d175265c46e0424f8e6` preceded training. Four CPU tests and independent 56-pair loss replay pass.

### C12 — registered: disjoint device and masked self-index inputs

P1: disjoint-X device inputs improve raw Y-fork NLL over dense; P2: empty-X-support NLL preserved; P3: 50% masked self-index training improves OOD against C11. Primary Emender, secondary transformer; four matched 180-second arms. Corpora constructed and hash-bound before training. [PREREGISTRATION_C12.md](PREREGISTRATION_C12.md), SHA256 `3a7b1e86f6fc7836025cf2c2527416efca7f27a38e1baca3d83f43862483837b`. Shared English contexts limit a positive result's operator-skill interpretation; a miss closes this tested tier. No outcomes yet.

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

Run `/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/validate_program.py` before a result is called documented and before committing a status transition. The validator checks exact committed bytes and current receipt bytes, hashes, claim order and statuses. It audits bindings, not scientific entailment or fresh model execution. The sxgc repository defaults to the sibling checkout; `--sxgc-repo` can point to another checkout containing the pinned commit. Missing evidence fails closed.

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
      "status": "registered",
      "receipts": []
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
      "status": "registered",
      "receipts": []
    }
  ]
}
```
