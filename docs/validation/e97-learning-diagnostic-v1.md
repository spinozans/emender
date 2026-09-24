# E97 read-only learning diagnostic v1

Date: 2026-09-09. Local, uncommitted. Status: all four GPU likelihood
measurements complete. No new training, parser relaxation or promotion.

## Question

The 5e-5 u64 checkpoint retained legacy tasks in train/y (120/120 core,
236/240 compositional), but both y and x failed all 24 analysis-protocol
development tasks before any tool dispatch. Determine whether correct targets
became more likely on actually consumed training examples, and whether the
single-record forward loss and tokenwise serving predictions disagree.

This is not yet an observation-dependent reasoning test: protocol failure
prevented the earlier development runs from receiving any real observations.

## Verified consumption, not nominal stage mixture

Reconstructed all 512 pack selections from the exact training sampler source,
world 8, key 974121, epoch-permutation mode, updates 1–64. Counts match the
terminal reload receipt exactly: **30,561,243 inputs / 17,984,082 targets**.

| Source | Consumed target tokens | Record occurrences |
|---|---:|---:|
| Agent/private-analysis | 1,920,701 | 169 |
| Conversation | 8,864,511 | 8,650 |
| Compositional | 3,656,404 | 20,231 |
| Core retention | 1,166,722 | 9,744 |
| Documents | 2,375,744 | 1,799 |

There were **40,593 unique source records** and **169 unique agent trajectories**.
These are not 40,593 unique tasks or semantically unique contents. Agent targets
were **10.68%** of consumption at u64, versus 30% in the complete-stage quota.
The 17.98M total must not be described as 17.98M analysis/agent targets.

The epoch permutation uses an affine modular map. Its initial pack IDs here
are **849, 851, 853, 855, ...**: no-replacement coverage does not imply a good
random shuffle or source-balanced prefix. The early mixture skew is an actual
confound. It does not alone explain the original 2e-6 full-stage failure, which
also produced zero development successes.

## Frozen probes

Root: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/learning-diagnostic-v1`.

`probes.json` SHA-256:
`4b6ec6efae0fe7d9a132ffeae3a680d2ab81320aa43b8b997118c1f1900a3370`.

- First eligible records in actual update/rank/pack order: 12 agent examples
  from distinct trajectories, six conversation examples and six core examples.
- Complete context before the first supervised target, bounded at 4,096 tokens;
  score up to 64 contiguous supervised target tokens. No context truncation.
- All 24 existing, already-consumed development prompts; no invented reference
  reasoning for these. Compare first-token predictions and fixed marker-prefix
  likelihoods (`Analysis: `, `Action: `, `Final: `, `The `).
- Total prefix tokens across the 48 probes: 25,940; maximum prefix: 2,137.
- Explicitly **training_eligible=false**. Diagnostic snippets are not new SFT
  records, full trajectories, an independent holdout or a replacement gate.

The initial CPU freeze attempts rejected missing selection quotas because null
trajectory IDs collapsed non-agent uniqueness. Fixed the identity fallback to
use record identity when trajectory identity is null, before publication and
before model measurement. No partial probe file was published.

## Measurements and limits

Compare parent train/y, prior full-stage 2e-6 train/y, and u64 5e-5 train/y and
saved/x, using exact checkpoint hashes from the completed evaluation.

For each prompt:
- Tokenwise serving top-five first-token predictions and marker log-likelihoods.
- For training examples, teacher-forced target NLL and first-target rank.
- The same boundary-aware loss API as the trainer, evaluated without gradients
  on the same single-record prefix/target bytes; compare with serving NLL.
- Full-prefix forward first-token prediction versus tokenwise serving.

This does **not** qualify packed 64K backward gradients, full-turn acquisition,
optimizer updates, or numerical equivalence across all kernels. Shape-dependent
BF16 differences are reported rather than automatically called corruption.
Different checkpoints' likelihoods must be compared on the same frozen probes.

## Execution

- CPU tests: **20 passed**, covering prefix/target bounds, scoring, aggregate
  completeness/identity/membership, and eight-rank CUDA routing.
- New scripts: `scripts/freeze_e97_learning_diagnostic.py`,
  `scripts/probe_e97_learning_likelihood.py`,
  `scripts/aggregate_e97_learning_likelihood.py`.
- Read-only source snapshot inventory SHA-256:
  `5973205ddc9f573526275e35d37f6fea4ef1aa0b7d520a828b91f3b8ff1db7e6`.
- Managed process: `proc_8aef`, `run-probes.sh`, exclusive eight-GPU lease,
  explicit local-rank device selection, NUMA placement and isolated Triton
  caches. A failure stops the chain; no automatic failed-run retry.
- Parent train/y completed first. Mean target NLL on the 12 agent probes:
  **2.230531 serving vs 2.230687 boundary-aware forward**. All 12 agree on
  first-token argmax; 0/12 predict the correct Analysis first token. Across all
  48 prompts, 47/48 agree on first-token argmax. The six core probes have serving
  NLL **0.000734** and 6/6 correct first tokens. This is consistent with strong
  familiar-template fitting, and no gross forward/serving discrepancy on these
  parent probes; it is not yet a comparison with the trained candidates.
- Parent summary SHA-256:
  `1de9b7a831333e721b79f573fac667563557caf74f38cb77f7620bfced6c525d`.
- Marker interpretation caveat: the frozen marker strings end in a standalone
  space token (p50k ID 220), unlike the space-prefixed word/quote token normally
  following a header. Therefore their full `logprob_sum` is the probability of
  that explicit token sequence, **not** a reliable marginal textual-prefix
  probability. Comparisons should use the stored per-token scores through the
  colon (or `The`), dropping the final standalone-space token after verifying
  the token prefix. This is an offline projection of already stored scores;
  no model rerun, prompt change or active-source mutation is required. Target
  NLL, first-target ranks and forward/serving comparisons are unaffected.
- The prior full-stage **2e-6 train/y** checkpoint completed. On the same
  12 consumed agent prefixes, serving target NLL improved **2.230531 →
  1.979452**, with improvement on all 12 examples. Boundary-aware forward NLL
  was **1.979265**, and full-prefix/serving first-token argmax agreed on all
  48 probes. Thus the measurements do show learning; neither a wholly dead
  optimizer nor a gross train/serve mismatch explains this bounded result.
- Nevertheless **0/12** training probes predict the required Analysis first
  token. Its ranks range from **3 to 118**. Mean header-token log-probability
  through `Analysis:` (excluding the standalone-space token) improved
  **−20.2543 → −11.7665**, while `Action:` remains strongly preferred
  (**−0.00128 → −0.05261**). On the 24 development prompts, the corresponding
  Analysis-header mean improved **−18.7541 → −12.7452**, but Action remains
  preferred. These are mean log-probabilities of explicit token sequences,
  not task-success rates or arithmetic mean probabilities.
- Prior 2e-6 train/y summary SHA-256:
  `1f3e56a25908ff998f2679a0f0309a0904d04166f755fa952401a08f56992683`.
- **u64 5e-5 train/y** completed. All 12 consumed agent examples improved in
  overall 64-target serving NLL: mean **1.196274** (boundary-forward **1.195609**).
  All 48 full-prefix/serving first-token argmax predictions agree. However,
  **0/12** select Analysis, and its first-token likelihood worsened versus
  the parent on **11/12** examples. Correct-token ranks range from **5 to 5,618**.
- Separating the crucial first decision from teacher-forced continuation:

  | Checkpoint | Mean first-token NLL | Remaining 63-target NLL |
  |---|---:|---:|
  | Parent train/y | 18.1522 | 1.9778 |
  | Prior 2e-6 train/y | 10.7660 | 1.8400 |
  | u64 5e-5 train/y | **28.3201** | **0.7657** |

  The better overall loss hides a severe first-decision regression. The later
  token scores are conditional on supplying the correct preceding tokens; they
  do not establish successful free generation or fresh-task reasoning.
- u64 train/y mean header-token log-probabilities: on training agent prefixes,
  **Analysis −28.3230 / Action −0.001119**; on development prefixes,
  **Analysis −26.0695 / Action −0.0000170**. This is not explained by a gross
  forward/serving mismatch on these probes. It motivates inspecting protocol
  decision supervision, competing source gradients and effective updates rather
  than selecting a new LR from average loss alone. Different LR runs have
  different consumption budgets; this is not a matched-budget LR comparison.
- u64 train/y summary SHA-256:
  `85605757de49693121b4b21a26ce778f3dc05efe9b9320137fc39c3fffaa40d0`.
- **u64 5e-5 saved/x** completed. Its 12 agent-probe serving NLL is **0.870080**
  (boundary-forward **0.871005**), mean first-token NLL **12.6052**, and remaining
  63-target NLL **0.68381**. Despite this strong teacher-forced continuation
  likelihood, **0/12** select the correct first token (ranks 14–1,854). All 48
  full-prefix/serving first-token argmax predictions agree.
- On the six core probes, saved/x NLL is **0.115095**, versus **0.000488** for
  train/y; first-target top1 is 2/6 versus 6/6. Even seemingly low aggregate
  cross-entropy can conceal critical protocol errors and poor task success.
- Saved/x mean header-token log-probabilities: training agent prefixes,
  **Analysis −12.6130 / Action −1.19898 / The −3.89163**; development prompts,
  **Analysis −11.9628 / Action −5.68225 / The −1.01547**. The descriptive-prose
  preference on development prompts matches the free-generation observations.
- Saved/x summary SHA-256:
  `248d41db51edeaaa0b69179f0faedf0dff7f77c7bcf5bbb26d36133b18faf0bd`.
- `proc_8aef` completed successfully in **1,202 seconds**. All four likelihood
  evaluations are complete. These are completed diagnostics, not capability
  successes. Both u64 representations learned conditional continuations while
  failing the autonomous opening decision. This supports an isolated prefill
  comparison and marker-specific gradient investigation; it does not prove
  useful fresh-task reasoning, identify the sole cause, or select a new LR.
- No optimizer step or training run is authorized by this diagnostic.
