# X26b native IMM control amendment (before optimizer updates)

Trigger: X26 RNN IMM best validation51.78% at35500, DeltaNet51.78% at58000;
both complete60000 updates without the qualitative near-perfect ID target.
`receipts/trigger.json` binds their terminal/result/validation receipts. This
fires X26/X27's accepted conditional native-task comparison, not a generator-bug
or architecture-impossibility verdict. Sources remain upstream LinearRNN
`dbfa365ad680f425dfb79a9e78b5e0d7915757bb`; no upstream bytes edited.

## Supervisor-approved control and predictions

Run the shipped `imm_unbound/gen.py` UNMODIFIED, seed0, T_train100/gap100,
700/200/100 examples PER LENGTH (train/validation/test), UNBALANCED default.
Deliberately preserve int64 overflow, ==0 labels and native dataset/collate.
BOS + N matrix-vector tokens, NO STATE teacher forcing. Native GRU2x256/drop.3
and native DeltaNet1x256/h4/drop.2; unmodified type embedding +9-vector projection,
LayerNorm/GRU and RMSNorm/DeltaNet heads. No scalar encoder substitution.

Freeze qualitative predictions before updates:
(a) both learn native IMM: localizes failure vs X23E to the task-code/contract
BUNDLE, consistent with Figure2's qualitative target, NOT an exact reproduction;
(b) native also fails: investigate reconciled batch/LR/WD/curriculum protocol;
raw script defaults are only a proposed NEXT separately registered control.
Mixed outcome localizes to model-by-contract interactions; do not force either
branch. No post-hoc numerical success threshold or single-field attribution.
Because native data are imbalanced, report confusion matrices, class recalls,
balanced accuracy and per-length accuracy alongside raw accuracy and majority
baselines. High majority-class accuracy alone does not show task learning.

## Reconciled protocol retained

AdamW batch128, lr1e-4/wd1e-4, betas(.9,.999),eps1e-8, clip1, bf16, no warmup.
Use native BCE final +.3 native BCE over all intermediate MAT outputs. Native
collate/label/loss/model code executed directly, not reimplemented. Eligible-pool
curriculum5->100/150epochs; shuffle seed23000, epoch tails carried to128 exactly,
as X26, not native bucket/partial-batch scheduling. Validation at0/every500;
three consecutive100% ID validations or60000 updates. Best ID accuracy, tie BCE;
OOD never selects/halts. Frozen native pools70k/20k/10k/10k/10k, hashes recorded.
Model max_len301 covers all registered bins without positional embeddings.
Fresh seed0, no checkpoint restore. Errors vs infrastructure censoring separate.

## Standard chain / graph censor

Max-bin N300/batch128 zero-update backward+eval smokes must pass for BOTH arms.
Commit+push registration/code/gates before optimizer updates. Standalone shell
leases physical cards4/5 only, PID/starttime/host and GPU UUID verification,
persistent60s identity-verified disk40GiB/rung, free400GiB, wall7day guard,
TERM then KILL. Watcher wiring emitted; native notification arming supervisor-owned.
No WorksGood machinery. Local lambda01 Python3.12/CUDA, not Frontier;
R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 not applicable.

Supervisor additionally rules eager X26 graph RNN resource-infeasible:
step153 at59890.76s, recent12.3549 updates/hour (~202days for60k), GPU2 active10%,
CPU100%; not wedged, but censored INFRA-RESOURCE-INFEASIBLE, not failed science.
Original worker group terminated using verified identity, standard guard/terminal
receipt retained. Identical architecture relaunch awaits a performance-optimized
scan matching the eager oracle in forward AND parameter gradients under an
X24-style pre-update gate; no ungated relaunch authorized. Its separate gate
and launch disposition must be receipted before any graph updates.

Independent acceptance review required. Mutable training/status/checkpoints and
183MiB native pools are not committed; deterministic generation command and hashes are.
