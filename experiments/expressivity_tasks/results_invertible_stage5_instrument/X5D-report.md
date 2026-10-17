# X5D — instrument verification and budget reconciliation

**Verdict: no task/depth-plumbing bug; no production fix and no relaunch of the six X5 runs.**
The X5 runner reproduces the known one-step learning behavior. The apparent
contradiction compared a **10K-update T=4** exit with **much longer-trained T=1**
results. The `<4` entries remain accurate descriptions of the registered
10K-budget experiment, **not established capability frontiers**. Longer-budget
T=2/T=4 competence is not established by this diagnostic either.

Base: `ca426f1c34b40aef16a84d26296d992bcae0fdf3`, local host `lambda01`.
Evidence root: `experiments/expressivity_tasks/results_invertible_stage5_instrument/`
(abbreviated **D/** below). Existing X5 results are unchanged.

## Approved diagnostic interpretation

Before launch, the live supervisor approved independent scratch seed-0 probes,
ordered T=1, T=2, T=4, for both original architectures. T=1 has X3-matched maximum
budgets (E97 60,000; GDN2 18,500), a 97% reproduction threshold and X3's 250-update
observation grid. T=2/T=4 retain X5's 10,000 budget, 90% threshold and 500-update
grid. This explicitly amends the task's proposed binary interpretation: low
T=2/T=4 results at 10K do **not** require declaring a task bug or an intrinsic
frontier when X3's own T=1 already needs more than 10K. The supervisor ruled this
budget miscalibration, not an architecture limitation. See
`D/diagnostic_protocol.json` for the approval text and limits; its portable
transcript was written after launch, not claimed as a timestamped preregistration.
No carry-forward curriculum or original six-run matrix relaunch was performed.

## 1. Data verification — scalar/hand decoding

`verify_invertible_stage5_instrument.decode_batch` checks the entire batch using
scalar `(x,y) <- (y,(x+y*y+c) mod 61)` arithmetic, independently of the generator's
`forward`/`trajectory` helpers. It verifies two disjoint init tokens, exactly four
drivers, six real tokens, four supervised targets per row, and joint class
`61*x+y`. Prefix targets are zero and masked; driver targets are all unmasked.

Historical seed-0 first training batch, **reconstructed** from the archived task
source, seed and generator call (not falsely labeled an archived training dump):

| Init tokens → state | Four drivers | Composed states | Joint target classes |
|---|---|---|---|
| `[112,160]` → `(51,38)` | `53,8,35,44` | `(38,23),(23,26),(26,2),(2,13)` | `2341,1429,1588,135` |
| `[92,138]` → `(31,16)` | `51,32,22,18` | `(16,33),(33,39),(39,51),(51,35)` | `1009,2052,2430,3146` |
| `[79,124]` → `(18,2)` | `25,29,43,54` | `(2,47),(47,44),(44,13),(13,23)` | `169,2911,2697,816` |

For the first row, the four new second coordinates are:
`(51+38²+53)%61=23`, `(38+23²+8)%61=26`,
`(23+26²+35)%61=2`, `(26+2²+44)%61=13`.
Thus the input is `[112,160,53,8,35,44]`, target
`[0,0,2341,1429,1588,135]`, mask `[F,F,T,T,T,T]`.
The `+2` is a **length** convention: init x encodes as `x+61`, init y as
`y+122`; init tokens are not coordinates shifted by two.

Actual archived seed-0 **frozen T=4 validation** rows:

| Init tokens → state | Four drivers | Composed states | Joint target classes |
|---|---|---|---|
| `[62,162]` → `(1,40)` | `33,60,2,12` | `(40,48),(48,25),(25,4),(4,53)` | `2488,2953,1529,297` |
| `[96,169]` → `(35,47)` | `54,48,56,56` | `(47,41),(41,7),(7,24),(24,29)` | `2908,2508,451,1493` |
| `[106,152]` → `(45,30)` | `49,12,43,12` | `(30,18),(18,0),(0,0),(0,12)` | `1848,1098,0,12` |

For validation row 1 the modular numerators are `1634,2404,675,53`, yielding
`48,25,4,53`. All **512 rows in each of six archived pools** pass the scalar
check: `[512,6]`, 2,048 supervised positions. Arrays exactly match regenerated
X2/X3 split pools and their archived array hashes. `D/data_verification.json`
contains decoded examples for all seeds/arms, arrays/file hashes and findings.
The new T=4 probes additionally dump their **actually generated** first training
batch directly from inside X5's generator call in `*/instrument_receipt.json`.

## 2. One-step reproduction through the actual X5 runner

`verify_invertible_stage5_instrument.probe` calls **X5's unchanged `run()`**.
It overrides only the explicit diagnostic constants/pool-depth selection and
wraps the original task generator to receipt, not alter, returned arrays.
Models, tied heads, ScheduleFree optimizer, lr, clipping, masks, padding,
forward/backward, and checkpoint/eval-x handling are the actual X5 path.
E97 has 9,900,800 trainable parameters; real FLA GDN2 has 9,887,056 and
`allow_neg_eigval=True`. All probes are independent scratch initializations;
none loads another probe's weights or optimizer. One leased GPU runs them serially.

| Arm | T=1 budget | X5 accuracy at 10K | First ≥90% | First ≥97% / stop | Exit validation accuracy |
|---|---:|---:|---:|---:|---:|
| E97 | 60,000 | 2.3438% | 48,500 | 57,000 | 97.8516% (501/512) |
| GDN2 | 18,500 | 14.8438% | 13,000 | 13,750 | 98.2422% (503/512) |

The **original X3** matched-lr T=1 validation receipts are also only
2.3438% / 14.8438% at 10K. X3 first reaches 90% at 48,500 / 13,000 and 97% at
54,750 / 13,750. Its original run stops at 60,000 / 18,500; the headline
97.66% / 100% is the selected-checkpoint **test** result, not a 10K result.
Our diagnostic threshold is validation-only; these columns are explicitly
validation metrics, not mislabeled test reproductions.

GDN2 reproduces **all 13,750 per-update training losses exactly**, and its
validation trajectory exactly through that stop. E97 reproduces the delayed
learning curve and first 90% boundary, but is not bitwise identical: maximum
per-update loss difference is 1.271396 nats; at 57K, X3 validation is 98.4375%
versus 97.8516% here. We claim trajectory/regime reproduction, not deterministic
E97 bitwise equivalence or a measured cause for its numerical differences.
`D/{e97,gdn2}_T1/x3_equivalence.json` retains every matched validation boundary
and the training-loss comparison; the full logs remain available.

## 3. Depth plumbing and historical-run evidence

The executed chain is:

1. X5 `DEPTHS=(4,8,16,32,64,128)` is copied into config; it is an effective
   constant, not an ignored CLI depth option.
2. `run()` loops `for depth in DEPTHS`, calling
   `task.generate_batch(32, depth+2, rng)` at every update.
3. `InvertibleQuadraticTask.generate_batch(B,T,rng)` draws `[B,2]` init
   coordinates and `[B,T-2]` drivers, concatenates them, then composes every
   driver into targets at columns `2:`. Thus depth 4 → T=6 → four drivers.
4. X5 `validation_pools` passes that same depth to X3 `frozen_pools`; X3 p=61
   delegates to X2, which calls the same task with `length+2`.
5. `padded_forward` only **right-pads** to a multiple of 16; T=1/2/4 use real
   lengths 3/4/6, all padded to 16, then logits are sliced back to the real
   length **before** loss/metrics. Init inputs remain visible. The mask hides
   only their loss. `HybridLadderLM.forward` embeds all tokens and returns
   unshifted logits; the runner uses external CE on `logits[mask],y[mask]`.
   No next-token shift, teacher-forced targets, or pad targets enter CE.

The archived launch runner SHA-256 is
`4f88b07bf67dbec69b604d5b3c66d2bf1738cf02568b83252fd577e04f988582`.
Each original config binds that snapshot. X3/X2/task source hashes still match
the launch sources; the current X5 difference is only the already-reported
stage-entry **checkpoint retention** guard, not data or training behavior.

Every original `training.jsonl` has exactly 10,000 consecutive updates marked
`depth=4`; **per-batch lengths, inputs and hashes were not recorded**. That
receipt gap is a finding, not evidence that the batches secretly had T=128.
Every original `result.json` records **1,920,000 input tokens**:
`10,000*32*6`, versus `10,000*32*130=41,600,000` for 128 drivers.
More strongly, all **six original final checkpoint NumPy RNG states exactly
match replaying 10,000 four-driver batches** from their recorded seed. Each
checkpoint's SHA-256, depth and step were independently verified. See
`D/historical_rng_audit.json`. Together these corroborate actual four-driver
training, while honestly distinguishing reconstructed arrays from archived ones.

## 4. Reconciliation and interpretation limits

There is **no difference in the first-position mathematical task**: both paths
supply `(x0,y0,c1)` and predict the same joint next state. There is a substantial
**experiment** difference: 10K updates at four-position mean CE versus up to
60K updates at one-position CE. T=4's first position receives one quarter of the
per-batch CE weighting; the other three positions require longer compositions.
Fresh-driver sampling changes the subsequent RNG stream and the frozen pool
seed includes depth. None of this changes the prefix convention, masking or
input visibility. The first target being learnable in isolation at ~14K–57K
therefore does **not** force 25% aggregate accuracy after 10K mixed-position
updates. At 10K even the isolated one-step task has not reached 90%, let alone
97–100%. The purported 25% lower bound assumed a learned one-step solver that
these budgets demonstrably did not yet provide.

The original six observations still say “no ≥90% crossing at T=4 within 10K.”
They do **not** locate the first intrinsically unlearnable composition depth,
and the absence of later-stage transitions is not a measured T=8–128 failure.
The similarity to old T=128 aggregate accuracies is not evidence of a hidden
length parameter: decoded data, RNG states and token accounting contradict it.
The approved conclusion is **no-bug + budget reconciliation**. A larger-budget
carry-forward curriculum is a separate supervisor decision; no such launch or
claim that 60K necessarily solves T=2/T=4 is made here.

Uniform marginal reference: `ln(3721)=8.221747728` nats. For this invertible map,
a predictor restricted to a proper driver suffix with at least two omitted
drivers has full-state accuracy `1/3721 = 0.0268744961%` under uniform independent
inputs. This is a suffix-only information bound, not a capacity bound on either
recurrent model and not an explanation inferred from the measured accuracies.

## 5. T=2 then T=4: independent scratch, 10K updates each

| Arm | Drivers | Real/padded length | Exit correct/count | Exit accuracy | Best validation | Final per-driver accuracies |
|---|---:|---|---|---:|---:|---|
| E97 | 2 | 4 / 16 | 5/1024 | 0.4883% | 1.8555% | 0.9766%, 0% |
| GDN2 | 2 | 4 / 16 | 22/1024 | 2.1484% | 2.1484% | 4.1016%, 0.1953% |
| E97 | 4 | 6 / 16 | 13/2048 | 0.6348% | 0.7812% | 1.9531%, 0.1953%, 0.1953%, 0.1953% |
| GDN2 | 4 | 6 / 16 | 8/2048 | 0.3906% | 0.8301% | 1.5625%, 0%, 0%, 0% |

All four stop on the original 10K budget, not a threshold crossing. At a
**matched 10K** comparison T=1 is already below threshold, so these runs do not
identify a first intrinsically failing depth after a fully learned one-step
stage. They are not a curriculum carried forward from the successful T=1 runs.
In particular, GDN2's T=4 first-position 1.5625% yields exactly
`1.5625%/4 = 0.390625%` aggregate, not ~25%: its first-position solver has not
been learned to ~100% under this training budget/objective.

GDN2 T=4 reproduces **every original seed-0 X5 training loss exactly**, as well
as its 0.390625% exit. E97's new 0.634765625% versus original seed-0 0.537109375%
is the same low-accuracy regime, not bitwise reproduction (maximum per-update
loss difference 0.294308 nats). No model/task/recipe fix was necessary.

Every observed new training stream's SHA-256 matches a complete replay, and its
final checkpoint RNG matches that replay. Shapes and mask counts were checked
on **every generated batch**. All six actual models also have exactly equal
first-three-token logits with the full sequence versus the truncated
`(init_x,init_y,c1)` prefix: max absolute difference **0**, measured under the
same BF16 autocast/right-padding path. Per-position panels use the final frozen
eval-x checkpoint, with no checkpoint selection on those panels. See
`D/verification_summary.json`, each run's `instrument_receipt.json`,
`position_metrics.json`, `validation.json` and `training.jsonl`.

## 6. Validation, changes and next step

**114 tests passed** across the task, X2/X3 and X5 protocol/aggregation suites.
New tests hand-check a known four-driver example; validate T=1/2/4/8/16/32/64/128
length/encoding/mask contracts; reject corrupted targets/init/masks/drivers;
distinguish four- from 128-driver RNG replay; and test causal right-padding,
prefix visibility and removal of padded logits. The existing end-to-end mocked
X5 loop test now asserts actual generator lengths `(32,6),(32,10),(32,10)` across
stage advancement plus cumulative token accounting. No production runner,
generator or model source changed.

Commands run from the repository root with local `.venv/bin/python`:

```bash
.venv/bin/python -m experiments.expressivity_tasks.verify_invertible_stage5_instrument --data-only
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_invertible_quadratic.py tests/test_invertible_stage2.py \
  tests/test_invertible_stage3_8m.py tests/test_aggregate_invertible_stage3_8m.py \
  tests/test_invertible_stage5_curriculum_8m.py \
  tests/test_verify_invertible_stage5_instrument.py \
  tests/test_aggregate_invertible_stage5_curriculum_8m.py
bash -n experiments/expressivity_tasks/launch_invertible_stage5_instrument.sh
bash experiments/expressivity_tasks/launch_invertible_stage5_instrument.sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.expressivity_tasks.audit_invertible_stage5_instrument
git diff --check
```

The launcher issues six serial `--arm/--depth/--out` commands retained
in `D/receipts/`; each exits zero. A trailing separator space emitted by the
launcher was removed from command receipts solely to pass the staged whitespace
check; before/after hashes are in `receipts/command_receipt_normalization.json`.
Argument vectors and launch snapshots are unchanged. `receipts/probes/` retains the exclusive GPU
lease, source snapshots, package versions, source/base identity, disk/GPU
snapshots and final successful exit. `audit_invertible_stage5_instrument.py`
independently verifies current source hashes against configs/launch snapshots,
models/recipe/budgets, consecutive updates, final checkpoint hashes, replayed
stream/RNG identities, aggregate versus per-position metrics, causal-prefix
checks and all six original checkpoint/RNG receipts. Final portable evidence is
bound by `D/artifacts_sha256.json`. Checkpoint **payloads** remain local and
Git-ignored, as in X3/X5; their paths and hashes are committed. The full
checkpoint audit therefore requires those retained local payloads, not merely
a fresh checkout of the portable receipts.

Reviewed design authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` (ADR-003) and
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. These are isolated local task/model probes,
**not resilient DiLoCo or production E97 training**. R16 evidence discipline is
followed without claiming its systems-scale gate; no R/NDP/V21S/ISP conformance,
retired-production requirement satisfaction, Frontier submission or Slurm
Partition/QOS binding is claimed.

**Next:** independent review of this instrument/budget reconciliation, then a
separate supervisor-approved stage-budget/curriculum design before any relaunch.
The diagnostic does not test longer T=2/T=4 budgets, carry-forward bootstrap,
other seeds, or whether either architecture ultimately composes to T=128.
