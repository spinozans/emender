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

### C6 — registered: fork-aware crossover

Prediction: the C3 crossover survives training-in: fork-aware Emender's final-test NLL is below fork-aware transformer's under matched 180-second training budgets. A reversal or tie refutes the directional prediction; report deltas without claiming significance. Train byte-level Emender and transformer, and GDN2 if adding its byte-level variant is cheap; record omission explicitly otherwise. From initialization use fork-masked loss, branch-set conditioning and walk-forced bytes as consumed free inputs. Equal corpus, split, seed, context, optimizer and parameter target; receipt actual parameters, steps, tokens, compilation/wall cost and index cost. Keep C3's 55,769 calibration / 55,770 final-test separation; freeze calibration before final evaluation. Compare directly with 1.673012/1.746996, plus retrained neural-only controls. Report overall, fork, forced, covered, unsupported and zero-target loss. Corpus-label diagnostics and causal train-index arms must be separate. Before execution freeze exact branch/OOD normalization, mask alignment, backoff and reset rules, gate fitting and budget accounting in this ledger. No run or outcome receipt exists yet; next session executes after protocol completion.

C6 execution preregistration (2026-11-02): [PREREGISTRATION_C6.md](PREREGISTRATION_C6.md), SHA256 `e70de380d2cfe34d4c7a66abad42a926b724da60cc7d2929d8f8313a04b968b9`. Primary comparison is corpus-label diagnostic two-system standalone overall NLL (Emender < transformer); secondary fork NLL, fusion and retrained controls are separate. The frozen protocol explicitly defers the causal training-only context selector and limits any verdict accordingly. No training/evaluation has begun.

### C7 — registered: designed dense two-system prototype

Prediction: fork-masked loss, branch-set conditioning plus an OOD channel, interval features in the gate and exact-count teachers reduce final-test NLL below C3's 1.673012 Emender result. Report gap closure (1.673012 − designed NLL)/(1.673012 − 0.570865), while explicitly distinguishing the causal/train-count system from that transductive diagnostic denominator. No numeric closure is assumed. Ablate each ingredient with matched budgets; evaluation rules and causal index must be frozen before execution.

### C8 — registered: redundancy ladder

Prediction: controlled duplication 2x/10x/50x reduces per-position decision density and increases forced-walk gains; test whether crossover changes monotonically with redundancy rather than assuming it. Generate synthetic corpora from the same base distribution, separate duplicated training material from independently sampled held-out material, and report both per-position forks and suffixient chi separately. This tier-1 curve is intended to predict Wikipedia/Pile behavior; those transfers remain unmeasured. Freeze generator, sizes, splits and seeds before execution.

### C9 — registered: generation character

Prediction: the walk/fork alternation loop increases corpus n-gram overlap versus neural-only arms, while an explicit OOD channel permits drift and re-entry. At fixed prompts, sampling settings and generated-byte budgets, report n-gram overlap, longest copied spans, OOD run lengths, re-entry rates and walk/fork proportions; separate novelty from plagiarism judgments. Freeze n values, prompts and re-entry policy before execution. No generation receipt exists.

### C10 — registered: index as teacher

Prediction: mimetic initialization from exact training-only branch distributions lowers fork NLL or time to a preregistered loss threshold against random initialization at equal total training budget, including teacher cost. Keep held-out bytes out of teacher targets and distinguish support from OOD. Freeze distillation objective, temperature, smoothing/zero-support policy and initialization duration before execution. No teacher-training receipt exists.

## Growing design memo

### Dense mode — selected direction; implementation registered

Use a dense neural decision head in a two-system design. C4's refutation motivates abandoning the tested internal write-mask route; C3 motivates the separate exact walk. Decision evidence: C4 and C3 bindings. Dense-mode superiority itself is unmeasured (C7); no kernel rewrite or speedup is claimed.

### Branch-set conditioning — registered placeholder

Expose supported branch identities/counts to the head rather than only mixing output probabilities (C6/C7). C3's support-stratified fusion motivates the candidate, but no conditioning ablation has decided encoding, normalization or capacity. Decision receipt: none; freeze before training.

### Interval-aware gates — registered placeholder

Consider context length, interval/support size and uncertainty in the gate (C7). C3 already measures a calibrated feature gate and correlated support/context strata; it does not causally establish a feature benefit. Additional interval-feature selection and ablations remain undecided. Decision evidence for existing gate only: C3 bindings.

### OOD channel — requirement selected; implementation registered

C3 reports 2,709 empty-support forks and 17,066 nonempty-support zero-target forks on final test. A designed branch head must retain normalized probability outside the observed branch set, and abstain explicitly on empty support. Decision evidence: C3 bindings. Encoding, calibration and walk exit/re-entry remain undecided (C6/C7/C9); no deployment claim.

### Exact index teachers — registered placeholder

Exact branch distributions are proposed teachers for mimetic initialization (C10) and designed heads (C7). Training-only support, zero counts and teacher construction cost require explicit accounting. Decision receipt: none; no demonstrated benefit yet.

## Session discipline: green means documented

Run `/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/validate_program.py` before a result is called documented and before committing a status transition. The validator checks exact committed bytes and current receipt bytes, hashes, claim order and statuses. It audits bindings, not scientific entailment or fresh model execution. The sxgc repository defaults to the sibling checkout; `--sxgc-repo` can point to another checkout containing the pinned commit. Missing evidence fails closed.

As results land, update this ledger immediately: record the protocol before execution, then bind every adopted outcome by path + commit + hash and change status only when its receipt is available. No orphan outcome receipts: every newly adopted result must belong to a claim; older run artifacts remain under their existing manifests. Commit the receipt first before promoting a registered claim, so its exact commit can be pinned without self-reference. Registered claims must have no outcome bindings; design motivations may refer to measured claims. Do not commit .pt files. Keep failures and limitations visible. C6 remains registered in this session; no GPU training was started.

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
      "status": "registered",
      "receipts": []
    },
    {
      "id": "C7",
      "status": "registered",
      "receipts": []
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
    }
  ]
}
```
