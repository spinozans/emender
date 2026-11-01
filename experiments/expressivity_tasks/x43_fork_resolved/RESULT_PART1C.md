# X43 part 1c: direct-SA decomposition

Exact corpus and both checkpoints/per-target arrays match their retained SHA-256
receipts. The reversed finite corpus SA/LCP computation completed on CPU; no GPU,
xsa source change, retraining, interpolation, or fork-driven gate was used.
The preregistered follow-ups remain bound and unrun.

The derivation in DERIVATION_PART1C.md fixes occurrence, boundary, and orientation
conventions. There are 636,727 fork positions in 1,115,394 bytes
(57.0854%). This is chi_position/n, **not a proven minimum
suffixient cardinality**. The complete context-length histogram, 30 highest
branch-entropy contexts, forced-byte distribution, labels and model/baseline
metrics are in receipts/part1c/.

Held-out targets: 111,539; 64,041 fork and 47,498 forced.

| Arm | Forced NLL | Fork NLL | Forced accuracy | Fork accuracy | NLL at forks |
|---|---:|---:|---:|---:|---:|
| Emender | 1.761594 | 2.528119 | 48.345% | 25.449% | 65.928% |
| Transformer | 0.773061 | 2.612681 | 81.454% | 31.694% | 82.004% |

Unigram forced/fork NLL: 3.219009 / 3.442457; accuracy 19.121% / 11.766%.
Exact training-count oracle covers 74.199% of forced targets and 91.474% of
fork targets. On covered forced targets its NLL is 0 and accuracy 100%.
On covered forks accuracy is 23.980%; 33,582 observed targets have zero training
count, so unsmoothed oracle NLL is infinite. Positive-support-only fork NLL is
1.010703, a conditional diagnostic, not the full oracle loss or a universal
ceiling. Uncovered targets are excluded from oracle metrics and counted explicitly.

Validation: 511 tiny/random strings agree with independent exhaustive context
search; known abac (all fork) and aaaa (all forced) examples pass. 100 seeded
corpus targets independently pass overlapping bytes.find scans, maximality,
and branch-diversity checks. The initial hand-written abac expectation was
wrong (the unique b context backs off to empty); it failed before corpus execution
and was corrected. Final validation passes.

Residual risks: corpus-wide contexts include held-out information and can exceed
models' 128-byte reset windows; labels are descriptive rather than causal.
No equality to xsa chi or optimal suffixient-set size is established, and the
xsa endpoint failure remains unresolved. Single seed, short training, unequal
token budgets and training precision remain; no architecture ranking follows.
The requested contact_supervisor tool is unavailable in this session.
