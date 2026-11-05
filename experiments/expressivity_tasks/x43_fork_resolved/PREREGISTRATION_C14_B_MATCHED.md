# C14-B matched-exposure unbudgeted twin

Registered before training. This independent fresh seed-3401 twin uses all
lifecycle operations, nonbinding cap 257, batch 16, context 128, BF16, AdamW
lr 3e-4, decay .01, clip 1, unchanged keep=.5 consultation dropout and document
resets. Stop at exactly 35,419 updates and 71,677,042 real targets. Padding is
masked. This explicitly replaces the fixed 2700-second budget for this new arm
only: matched exposure by design, about 13,600 active seconds (3.8h) at the
honest ~2.6 steps/s. No kernel optimization or X31 recertification is prerequisite.

Source comparability follows HARVEST.json: all 23 source hashes are required
from c190956d in a detached local worktree; replay SHA256
58a8acb855a9659779904665ab510b6ddf08c734302333d31b74fe0eed79bbec binds.
Use the original source, avoiding even the optimized replay's limited-input
identity qualification. The new runner changes only stop/report conditions,
checks its own registered hash and each frozen source hash, and writes an
inspectable effective main. Fresh gates inherit the source-matching historical
battery and data bindings explicitly; they do not claim a fresh battery run.
Run preflight before launch. Fresh checkpoint and receipts are independent;
never overwrite original arms. Exact model/optimizer/RNG/cursor/telemetry state
is saved every 120 active seconds and on SIGINT/SIGTERM. Resume with the same
leased wrapper and --resume. Cards 2/4 are permitted through the shared broker;
card 1 and union service are untouched. PB3 retains operation counts, peaks,
occupancy samples and capacity merges. Require peaks <257 and zero capacity
merges; sampled occupancy is not a stationary mean.

PT1: compare frozen-window final-test NLL against budgeted 1.584700014.
Twin lower: the unbudgeted configuration learns better at matched data and
budget pressure's observed value was compute shaping. Twin higher: evidence
for curation pressure beyond compute in this recipe. Equality is a result.
Capacity and implementation remain confounded; one seed is not a general law.
PT2: report all 66 on/off split/stratum rows with identical denominators to
frozen triangle arms. PT3: measure whether the descriptive 9x dense-anchor
improvement per target extrapolates or saturates; do not treat endpoint ratios
as sample complexity or interpolate an unmeasured curve.

Paired evaluation is required after complete training, on the unchanged
3,431,242 document-bounded windows / 434,204,084 targets, including separate
214,097,022-target final test. Previous full evaluation took ~52,111 seconds;
3.8h estimates training only. No quality verdict before terminal evaluation.
The existing append-only arm has 75,394,452 targets, not 71,677,042: report
that exposure honestly; a three-arm exactly matched table is unavailable
without an exact retained append-only checkpoint. Do not relabel its endpoint.

Manual notes: supervisor transport is absent from the enabled catalog; no
supervisor contact is claimed. No resilient DiLoCo work (gap IDs inapplicable).
