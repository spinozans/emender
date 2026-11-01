# X43 fork program: bound before follow-up execution

Part 1 is a pipeline shakedown, seed 4301, raw vocabulary 256, contiguous
90% training / 10% held-out bytes. Both arms use context 128, batch 16,
AdamW lr 0.0003, clip 1, weight decay 0.01, and 180 seconds training each.
The pure arm reuses X42 velocity Candidate (memory plus MLP), width 256,
two blocks, two memory heads, target 2M trainable parameters. The control
is a two-layer causal transformer, width 256, four heads, FF width chosen
to match the pure trainable count. Report actual counts and duration.
Evaluation resets state every nonoverlapping 128-byte window; every held-out
target after the first held-out byte is evaluated once. No hyperparameter search.

## Index stop condition and interpretation

The prescribed `cd ~/sxgc && cargo build --release` failed: no Cargo.toml.
The index leg stops; no surrogate fork labels or fork measurements will be
reported as xsa results. Full-context uniqueness needs a context-length or
minimal-context convention: a whole observed prefix usually has one occurrence.
BWT run boundaries are in suffix-array order and encode changing *preceding*
characters. They cannot be assigned directly to natural-order next-byte targets
without the index's orientation, suffix mapping, sentinel convention, and precise
chi definition. Resolve those from the operator's xsa once buildable. Receipt
both in-corpus descriptive labels and train-only branch counts separately to
expose held-out leakage. Forced/fork NLL table remains blocked until then.

## Follow-up A: exact-count interpolation (NOT RUN)

Freeze both trained models. Build branch counts on training bytes only, using
the index's resolved context convention and explicit no-match backoff. Split
held-out bytes contiguously into gate calibration (first half) and final test
(second half). At forks q = (1-g) p_model + g p_counts, g=sigmoid(a), with one
scalar a per model fit only on calibration NLL, initialized 0. Forced positions
retain model softmax. Report final test overall/forced/fork NLL, fork accuracy,
learned g, count coverage, and paired differences to frozen model predictions.
Never tune on final test. Include separate explicitly in-corpus oracle diagnostic.

## Follow-up B: fork-driven Emender writes (NOT RUN)

Matched seed, bytes, optimizer, initialization, steps and capacity; compare
baseline writes, independent Bernoulli write mask density 0.5, Bernoulli density
chi, and fork-only writes of density chi. Chi and masks must be computed from
training-only index evidence; do not access future held-out labels at inference.
Distinguish an in-corpus oracle-mask diagnostic from a deployable causal gate.
Bind the resolved mask time alignment (write after consuming byte t versus
prediction of t+1), effective writes, and treatment of unmatched contexts before
execution. Report memory occupancy, actual write density, throughput, overall
and decomposed losses. This preregistration authorizes no follow-up execution.
