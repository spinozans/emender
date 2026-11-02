# X43 C12 — disjoint gains survive; masking preserves the unplugged head, not the self-index interface

P1 and P2 pass for both heads. Disjoint-index Emender improves raw Y-fork NLL **2.529165 → 2.434766** (−0.094399); transformer **2.634117 → 2.470846** (−0.163271). Empty-X-support NLL does not collapse. P3 also passes for both heads with and without features relative to C11, but enabled self-index OOD remains much worse than dense. Masking prevents catastrophic unplugged-head degradation; it does **not** rescue the enabled self-index interface.

**Operator-skill verdict:** this is positive evidence for training with a foreign index beyond the globally self-containing copy solution. It is not an isolated proof of abstract operator skill: X distributions can still supply predictive English byte priors on matching contexts, the Y-selected context length is transductive, and historical dense controls differ in steps/precision. A foreign index cannot copy the whole Y task, but that fact alone does not make every gain non-copying. The preregistered sign passes, so the requested negative closure condition is not met; do not close the training-side question as a failure or claim universal operator competence.

**Shortcut-prevention verdict:** yes for retaining a useful all-zero-input mode; no for rescuing the feature-enabled self-index interface at this tier. Primary masked Emender OOD is 4.858708 with features versus dense 2.444723, although it improves C11’s 6.551557. With all features removed it is 2.425617. Secondary transformer follows the same distinction (3.390723 enabled, 2.075162 removed, dense 2.466627). The transformer without-feature mode improves dense total/fork too; primary Emender without-feature total/fork remain slightly worse than dense.

## Frozen protocol and corpora

Main preregistration commit `de8e342eb2ed2bfc942acb493f2f03d712b11e3d` precedes feature construction and training; runner frozen in `de3ce28f7dcb4d4169ceac9910c07e861cfc8ba2`. Predictions hash `3a7b1e86f6fc7836025cf2c2527416efca7f27a38e1baca3d83f43862483837b`. C11 recovery additive registration commit `d1744b2e1d8379108ebed162aefde8edd856e94f`, hash `1ee68ea51f96ed32a6bb124bf4095ea291180d907de8b6ab061e9f12699547cc`, precedes recovery and C12 scoring. No model was selected or extended using held-out results.

X is Charles Darwin’s scientific prose, *On the Origin of Species*, [Project Gutenberg #1228](https://www.gutenberg.org/ebooks/1228.txt.utf-8), retaining exact bytes strictly between the Gutenberg START/END marker lines. Y is the unchanged canonical [TinyShakespeare dramatic corpus](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), preserving historical controls. Two different authors/genres, separate works; shared English substrings are permitted.

- X: 950,692 bytes, SHA256 `7124fc713f7da52adf8a10fda02eaa9a9a3e1f16cd622e4abe85f9d6e3d97686`.
- Y: 1,115,394 bytes, SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Both hashes and the downloaded-source hash were committed in `receipts/c12/corpora.json` before training. No Y bytes were indexed in X.

Arm 1 trains dense CE on the first 90% of Y. For each Y context selected by the existing direct-SA maximal-repeat rule, an exact reverse-SA query over X produces a 256-byte branch distribution, log1p occurrence count, entropy, X support>1 bit, and log1p context length. There is no backoff. Empty support has zero distribution/count/entropy/bit but retains context length. Thus empty support differs from physically removing all 260 features. The Y fork bit and Y target are not input features. Y’s context selector still uses corpus-wide structure.

74.3662% of training positions have empty X support; 84.6791% have zero X target probability, including 103,527 nonempty-support zero-target positions. The exact foreign-copy loss on final test is infinite: 47,447/55,770 targets have zero X probability (41,668 empty-support; 5,779 nonempty zero-target). The other 8,323 targets have positive X probability, so conditional distribution copying remains a possible contributor to gains.

Arm 2 uses unchanged C11 self-index features, independently dropping all 260 inputs together with probability .5 per training occurrence, without rescaling. The memory and full 256-byte CE remain unchanged. Each frozen model is evaluated with features and with all inputs zero. No inference fusion or learned calibration enters these tables.

## Disjoint table

All values raw nats/byte on the same 55,770 final targets. Fork is the preregistered **Y fork** stratum (32,415); OOD-X is empty X support (41,668), which differs from C11’s OOD denominator. Dense has no feature channel; its same predictions are reused with the new masks.

| Head | Training | Eval features | Total | Y fork | OOD-X |
|---|---|---|---:|---:|---:|
| emender | dense | none | 2.222975 | 2.529165 | 2.216768 |
| emender | disjoint | X index | 2.116847 | 2.434766 | 2.214777 |
| emender | disjoint | all zero | 3.078657 | 3.040701 | 3.202396 |
| transformer | dense | none | 1.924201 | 2.634117 | 1.901476 |
| transformer | disjoint | X index | 1.816094 | 2.470846 | 1.810907 |
| transformer | disjoint | all zero | 2.474033 | 2.871956 | 2.464290 |

P1 fork deltas are −0.094399 / −0.163271; P2 OOD-X deltas are −0.001991 / −0.090568 for Emender / transformer. The primary OOD improvement is tiny and has no uncertainty estimate. Whole-device removal worsens both heads, so an empty-support pass does not establish unplugged robustness. X-fork (7,507) and nonempty zero-target strata remain in `results.json`; Emender nonempty-X-zero-target NLL worsens 2.796943 → 2.925646 despite its aggregate gains.

## Masking table

Raw nats/byte; OOD-self is empty self-index support (9,049), with the same total and Y-fork denominators as above. Dense “either” means the model has no device channel. C11 with-feature rows reuse original probabilities. C11 without-feature rows use exact-step recovered weights, adopted only after the preregistered whole-held-out-vector parity gate.

| Head | Training | Eval features | Total | Y fork | OOD-self |
|---|---|---|---:|---:|---:|
| emender | dense | either (no channel) | 2.222975 | 2.529165 | 2.444723 |
| emender | c11 | self index | 3.872928 | 5.330632 | 6.551557 |
| emender | c11 | all zero; recovered | 7.213058 | 7.322476 | 6.025565 |
| emender | masked | self index | 3.650585 | 5.262262 | 4.858708 |
| emender | masked | all zero | 2.257021 | 2.559500 | 2.425617 |
| transformer | dense | either (no channel) | 1.924201 | 2.634117 | 2.466627 |
| transformer | c11 | self index | 4.953106 | 7.317413 | 6.060058 |
| transformer | c11 | all zero; recovered | 6.335313 | 6.543489 | 6.482301 |
| transformer | masked | self index | 3.777592 | 5.892060 | 3.390723 |
| transformer | masked | all zero | 1.647425 | 2.185056 | 2.075162 |

P3 enabled OOD improvements versus original C11 are 1.692849 / 2.669335 nats for Emender / transformer. With features removed, OOD improvements versus original C11’s enabled row are 4.125940 / 3.984896. Both modes satisfy P3; the enabled rows still lose dense in every table column. Against recovered C11 without-feature weights, masking also improves all three columns for both heads.

## Execution and validation

Sequential 180-second loops, seed 4301, batch 16, context 128, AdamW .0003/.01, clipping 1, BF16 and float32 normalized logits. Feature preparation/model initialization/inference are outside loop budgets; first-forward compilation inside each loop is charged. Cache warmth and steps are not matched.

| Arm | Configured trainable parameters | Steps | Tokens | Training seconds | Kept features |
|---|---:|---:|---:|---:|---:|
| emender_disjoint | 2133700 | 6293 | 12888064 | 180.023335 | 100% |
| transformer_disjoint | 2133748 | 29640 | 60702720 | 180.002195 | 100% |
| emender_masked | 2133700 | 5945 | 12175360 | 180.002171 | 50.0190% |
| transformer_masked | 2133748 | 29068 | 59531264 | 180.002370 | 50.0195% |

C11’s original Emender used 2,015 steps versus 6,293 disjoint / 5,945 masked here; historical dense used 5,552 Emender / 35,853 transformer steps. Reused dense controls also differ in compute precision and lack the extra 66,560-parameter device projection. Thus equal loop budgets do not isolate the device as the causal source of a gain.

C11 recovery replays its original optimizer steps and initialization, not a new scientific budget arm. Both heads pass whole-vector `allclose(rtol=1e-5, atol=1e-7)` on all 111,539 held-out target probabilities; maximum absolute error is 1.7881393432617188e-7 for each. Recovery costs 53.931467s / 172.274370s. Original with-feature probabilities remain the table reference; recovered weights supply only the otherwise missing feature-removal rows. No checkpoint files were written.

Six CPU tests pass: binary/finite-boundary substring intervals against exhaustive search, target-independent cross-index query, dense input-projection gradients and hook cleanup, and fail-closed ledger audits. Independent validator is green: 2,049 substring-count feature checks (separate `bytes.find` route), 98 loss/denominator replays, corpus/protocol/code/prediction hashes, four ≥180s budgets, dropout rates, GPU binding, C11 recovery parity and verdict signs. Main and recovery runners exit 0, each broker-leased physical GPU 2 from allowlist 2/4. Python `/home/erikg/emender/.venv/bin/python` only. No forbidden GPU, Frontier, Slurm, WorksGood or subagents. Program validator is run again after committed receipt binding.

Failures retained: initial pytest relative-import collection failure was fixed before training; its log is retained. The first independent-validator attempt was interrupted because repeated lazy NPZ decompression was slow; arrays are now loaded once, and the unchanged checks pass. No training run failed.

## Residual risks and manual notes

One seed and one corpus pair; no uncertainty/significance estimate, causal prefix-only selector, matched-step feature-free retrain, precision-matched dense control, or semantic-operator versus cross-corpus-prior ablation. Context lengths use held-out Y structure and may exceed the 128-byte neural reset. Foreign features can encode predictive byte priors directly; disjoint works do not make partial distribution copying impossible. Therefore P1’s positive sign is a scoped device-input result, not identification of abstract operator skill. P2 tests empty index support with remaining metadata, not total input removal. IID feature dropout does not exactly simulate real empty-support feature rows, which retain fork/length metadata; the enabled masking arm still fails dense OOD badly. Do not scale the self-index interface as rescued.

Manual notes: `contact_supervisor` is absent from the complete active tool catalog; no substitute supervisor/orchestration service was used. Work stayed in this attended thread. Unrelated pre-existing workspace changes were preserved; their global whitespace warnings are outside this change. No `.pt` files produced or staged. No resilient DiLoCo behavior changed or exercised; its requirement IDs are inapplicable. Exact dependencies and manual notes are retained under `receipts/c12/`.
