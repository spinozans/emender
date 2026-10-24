# X26c RAW-DEFAULTS IMM control — registered before optimizer updates

Trigger: X26 completed both scalar-contract IMM arms at 60k without the
Figure2-like pattern; X26b completed both native-task arms at 60k with partial
ID learning but OOD balanced accuracy only 50.09–51.25%, despite high raw
accuracy. `receipts/trigger.json` retains all four terminal/result/validation
hashes and native confusion matrices/recalls; neither failure is an architecture
impossibility result. In particular, do NOT describe X26b as chance ID learning.

## Intervention

Execute the byte-identical `imm_unbound/rnn.sh` and `delta.sh` via bash from
LinearRNN@dbfa365ad680f425dfb79a9e78b5e0d7915757bb, which execute their own
unmodified trainers, native collate, loader, optimizer, scheduler and stopping.
Fresh seed0, no checkpoint restore. Only environment/data/output path knobs
are set; env -i clears accidental hyperparameter overrides. Their default
GRU2x256/drop.3 and DeltaNet1x256/h4/drop.2, batch256, lr3e-4, 30k step cap,
500 epoch cap, weight decay .01/.005, clip1, 1000-step linear warmup followed
by cosine to .1LR, auxiliary weight .3, eligible-length curriculum5->100/150
native epochs, epoch-tail partial batches, eval every2epochs, finalAcc patience
100/60 and strict improvement checkpoint selection govern. Native validation
is mean-of-batch accuracy, not pooled balanced accuracy; retain this unchanged.
No OOD checkpoint selection, no fused replacement, no reconciliation loop.

Reuse the hash-identical frozen X26b native data (generator as shipped, seed0,
T_train100/gap100, 700/200/100 per-length train/val/test, unbalanced). All five
native contract differences (uniform entries, int64 overflow, ==0 label,
BOS+matrix tokens and per-length unbalanced sampling) remain deliberate.
These data counts and T100 are the carried controlled dataset configuration,
not a claim that gen.py CLI defaults (T300/200/50/50) generated Figure2.

## Three pre-registered outcomes (qualitative, per arm and jointly)

- RAW-SUCCEEDS: shipped pipeline reaches the Figure2-like near-perfect pattern
  with balanced accuracy decisively above chance, not prior-tracking. This
  implicates the X26b->X26c config/loop bundle, NOT a uniquely identified delta.
  Receipt every candidate delta in COMPARISON.md. No exact figure-coordinate
  reproduction claim: public revision-to-Figure2 provenance remains unverified.
- RAW-FAILS: shipped code in our environment does not attain that pattern;
  environment/config fragility or an environment-broken assumption remains.
  Identify where determinable; failed imports/errors/censors are NOT scientific
  training failures and must be separated from completed training outcomes.
- MIXED / ID-only: ID learns but OOD does not, or the arms disagree. Qualify
  the OOD claims using the paper Figure2 caption's matrix-length bins
  [1,100], [101,200], [201,300], not scalar-token lengths. Do not force a binary
  verdict. No post-hoc numerical threshold will be invented.

## Prior drift / mandatory external reporting

Figure2 reports RAW accuracy on a task whose prior drifts across bins. X26b
class1 priors: test_bin0 46.79%, bin1 76.54%, bin2 89.61%. Thus raw OOD accuracy
alone—even close to 90%—is not task learning. Every interpreted accuracy report
must include balanced accuracy, class0/class1 recalls, confusion matrices,
class counts and majority baseline. Per-length reporting likewise retains
both recalls and balanced accuracy (null if either class absent).

Training logs are upstream's unmodified raw, mean-of-batch diagnostics, NOT
standalone outcome claims. After successful exit, a SEPARATE evaluator loads
only their selected best-ID checkpoint, verifies its SHA256 before/after, and
uses their dataset/collate/eval bucket ordering/AMP to report pooled metrics on
validation and all three test bins. No training instrumentation, loss edits,
monkeypatches, alternate checkpoint selection or optimizer updates in evaluator.

## Gates and standard chain

Real N300/batch256 backward+eval zero-update smokes in each isolated environment
must pass. Hash-bound code/data/trigger/smoke registration commit+push BEFORE
launch. Physical cards4/5 only with UUID and PID/starttime/host receipts;
standalone shell leases, no WorksGood machinery. Persistent60s identity-safe
400GiB free /40GiB per-rung /7day wall guards TERM then KILL own process groups;
errors vs infrastructure right-censoring separate. Watcher probe/status/terminal
paths are receipted; native notification arming remains supervisor-owned.
Independent review required; mutable logs/checkpoints are not committed.
Local lambda01 CUDA, no Frontier or resilient DiLoCo behavior changed:
R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 not applicable.
