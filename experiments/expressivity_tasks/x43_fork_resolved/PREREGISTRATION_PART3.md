# X43 part 3: bound before training and evaluation

Prediction: masking the existing constant policy write gate to fork positions
will lower fork NLL without raising forced NLL relative to unmodified Emender.
The primary comparison is the full original 111,539-target held-out set; report
part-2 final-test separately (the quoted 2.223 baseline is that subset, whereas
full held-out baseline is 2.201701). Any fork improvement with forced regression
fails the joint prediction. Report an honest null or regression without tuning.

This explicitly requested experiment is an in-corpus label-conditioned probe,
not a causal or deployable result. Use the retained part1c corpus labels, now
independently cross-validated against the fixed xsa. Training uses only sampled
training positions, but their classification itself sees the full corpus.
This supersedes the older follow-up's training-only mask requirement for this
probe; that stronger causal experiment remains unrun. No production gate claim.

Time alignment: input byte at absolute offset t is consumed, read occurs before
write, and loss predicts byte t+1. The write stores the representation of byte t;
therefore its eligibility is fork[t], NOT fork[t+1]. This stores ambiguous
consumed decisions for later reads. The same mask applies to all layers/heads.
The immediate current-layer read is unchanged, while subsequent reads and
higher-layer representations can change. POP, decay and expiry remain intact.

Implementation: existing wire logit becomes -infinity on ineligible positions;
otherwise it is unchanged. The kernel's existing sigmoid threshold disables
all write cases there. Gate is fork[t] AND original policy threshold, not an
unconditional write at every fork. Thus actual density <= fork density; the
operator's approximate density prediction is tested, not assumed. The kernel
still computes routing evidence; no branch-computation speedup is asserted.
No learned management, parameter, read operator or capacity changes.

Four arms in fixed order: baseline, independent Bernoulli .5, independent
Bernoulli chi (training slice density of receipted labels), fork-only. Masks
are sampled per occurrence using independent seed 4303 and broadcast to heads.
All arms initialize seed 4301, original Candidate policy, width/layers/count,
corpus/split, 128 context, batch 16, AdamW 3e-4/.01, clip 1, bfloat16 training,
fp32 eval, original 180s loop including first-use costs and no warmup. Byte RNG
4301 remains separate from mask RNG. Save a step-1000 checkpoint for a secondary
matched-step comparison if all arms reach it; do not choose steps using eval.
Time endpoints may have unequal steps: report them and throughput explicitly.
Original baseline and transformer predictions are retained comparisons, not
retrained transformer arms. No hyperparameter search or result-driven reruns.

Evaluate after training, resetting every 128 inputs, complete original held-out
coverage. Compute overall/forced/fork NLL and accuracy, NLL share at forks,
paired loss deltas. Receipt mean gate probability, actual event density,
prewrite/live occupancy per layer/head over held-out windows, with exact
zero forbidden writes checked on every scan. Also report eligibility density.
All-true parity, all-false zero-write, alignment, finite gradient checks precede
scientific training. Checkpoint files stay local and are never committed.

Inference without labels: (1) walk a fixed training-only index with observed
left context, compute continuation support and abstain on unmatched contexts;
this can approximate classification but differs from corpus maximal-context
labels and needs causal validation and latency accounting; (2) a fixed internal
proxy using match-evidence/write-strength distribution, e.g. thresholding a
confidence statistic, needs a separately registered calibration and accuracy
receipt. Neither option is implemented or validated here. Both use only past
observations/model state, never an unavailable future corpus label. The probe
uses oracle labels at held-out inference explicitly.

GPU: broker lease physical 2 or 4 only. Python: repository .venv/bin/python.
The requested contact_supervisor tool is absent from the active tool catalog;
no supervisor coordination can be sent through it.
