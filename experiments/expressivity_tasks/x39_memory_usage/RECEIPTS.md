# X39 — memory-usage regularization: pre-registration

## Part 7 measured dispositions (post-run)

| Arm | Pre-registered | Measured PPL / control ratio | Per-head slope | Disposition |
|---|---|---|---|---|
| 1 decay, horizon 128 | 100M; PPL ratio <= 1.05; slope <= .5 | 6.92328267 / 1.00975219 | 0.0000/0.0000/0.0000/0.0000 | provisional compression screen passes; source attestation fails |
| 2 gate cost, lambda 0.01 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.91114998 / 1.00798265 | 0.9575/0.9602/0.9294/0.9402 | quality passes; strong compression fails |
| 3 poverty, S128 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.89398492 / 1.00547915 | 0.0000/0.0000/0.0000/0.0000 | capacity-capped; rich transfer is the disposition test |
| 3 same weights, S2944 | 100M; PPL ratio <= 1.02; slope <= .5 | 6.87446791 / 1.00263262 | 1.0000/1.0000/1.0000/1.0000 | quality passes; strong compression fails |
| 4' semantic keys + match wire | Operating lambda; 100M; parity + ReplayBackward | Not run | Unmeasured | Optimized recurrent wire absent; incomplete, not a trained null |

**Provisional measurements: unchanged-source attestation failed.** External X36 backward/replay_setup edits occurred during the single training process; `receipts/part7_integrity.json` retains the mismatches. These rows are not fully certified compression receipts.


The operator part-7 instruction authorizes the ordered usage runs and exact B64 comparisons. Original pre-training registration bytes are retained in each run. The selection rule and quality bands were not changed after results. Full trained adjudication: `receipts/part7_adjudication.json`.

| Lambda | PPL at 20,021,248 tokens | Ratio to control | Within 2% |
|---|---|---|---|
| 0.001 | 14.67135383 | 1.00008214 | True |
| 0.01 | 14.78951122 | 1.00813641 | True |
| 0.1 | 15.21379435 | 1.03705794 | False |

### Part 7 structural-lessons ledger

| Structural lesson | Trained evidence and limit |
|---|---|
| Decay bounds retained memory without reducing writes | Every registered length peaks at 64 slots/head; slope about zero, allocation .5/token/head, zero rich pressure; short-context PPL costs .975%. This is recent retention, not semantic deduplication. |
| Write-enable cost reduces the allocation coefficient | Selected lambda .01 gives slopes .9294–.9602: fewer writes, but the <=.5 strong-compression criterion fails in all heads. PPL costs .798%. |
| Capacity caps can mimic compression | S128 occupancy must be interpreted alongside pressure and the same trained weights at S2944; low capped slope alone is not learned compression. |
| Semantic retrieval is not trained deduplication evidence | Prior constructive CPU wire passes its two ablations, but no optimized recurrent wire or trained semantic result exists. |

No strong compression transfers to rich deployment.


## Part 6 amendment — B64 exact-exposure control (2026-10-27)

Provenance: the operator's part-6 task explicitly resolves the part-5
B8-control/B64-training blocker by authorizing a new control training run.
This supersedes the historical no-control-retraining instruction below.
Train the X34 tape_lm skeleton and policies with the optimized X36 lane,
seed 3401, unchanged tokenizer and affine packed-data ordering, batch 64,
context 512, and unchanged AdamW settings. Record validation at update 611
(20,021,248 target tokens) and 3052 (100,007,936 target tokens). These are
the matched exposures for the three pilot lambdas and full arms respectively.
The largest lambda in {.001,.01,.1} whose pilot perplexity is at most 1.02
times the new control at update 611 is selected; no qualifier is an honest
null. No B8 endpoint may substitute for either comparison.

The new B64 control also supplies the family reference for re-adjudicating
X37/X38 quality bands. Their historical B64-versus-B8 comparisons are not
batch matched; existing conclusions must retain that comparability caveat
until the new control and their exact exposure/protocol checks are available.
This amendment does not itself establish any quality band or compression.

The part-5 directive and its blocker receipt were requested as inputs but are
absent from this checkout's X39 files and recent history. The operator's
part-6 text supplies the blocker resolution above. No contact_supervisor
tool is exposed (tool discovery checked names and descriptions); no other
orchestration service is authorized as a substitute.

### Part 6 pre-registered versus measured disposition

| Arm / receipt | Registered requirement | Measured disposition |
|---|---|---|
| B64 control | X34 config, B64, 100M; exact 20M pilot reference | Completed 3052 updates / 100,007,936 tokens, PPL 6.8564175640; update 611 / 20,021,248 tokens PPL 14.6701488443 |
| Control occupancy | T 512/1024/2048/4096, two windows; transcript slope about 1 | Slopes 1/1/1/.9997886161; packed-window pressure zero; complete profiles and NLL in part6_control_evaluation.json |
| Arm 1 decay | Full 100M; slope plus <=1.05 PPL ratio, decay NLL shape | Not trained; all trained-arm metrics unmeasured |
| Arm 2 gate cost | Three 20M pilots, largest lambda within 2%; independent 100M | No pilots; no operating point selected; no no-qualifier null established |
| Arm 3 poverty | Full S128 training; S128 and S2944 deployment, <=1.02 PPL ratio | Not trained; disposition and transfer unmeasured |
| Arm 4' semantic + match wire | Operating lambda, full 100M, <=1.02 PPL ratio | Optimized wire not implemented/certified; not trained |
| Per-arm compression | Slope <=.5 plus passing quality; allocation/pressure/story/depth/context | No trained-arm evidence; no compression success claimed |

### Structural-lessons ledger (scope retained)

| Lesson | Evidence and limit |
|---|---|
| Append gate has no allocation-count gradient | Historical source-hashed CPU counterexamples; amplitude affects stored content, hard routing supplies no count derivative |
| Certified ISA supplies no semantic deduplication | Repeated/zero-residual keys still append in historical CPU witnesses |
| Read-to-gate wire necessary and sufficient for repeat suppression | Only within the part-4 hand-set alias witness: semantic+wired suppresses repeats; both required ablations do not. No general necessity/sufficiency or learned LM result |
| Poverty dose-response | Synthetic S8/16/32 recall CE 3.882920/2.166516/0; no trained S128-to-S2944 transfer result |
| Decay equilibrium | CPU deaths/equilibrium and part-4 isolated GPU death parity; no decay-arm trained occupancy or quality result |

Part-4 match evidence uses 1/sqrt(K), whereas certified READ uses 2/sqrt(K).
Its comment equating those scales must not be treated as trained-wire parity.
`part6_family_comparability.json` retains X37/X38 source-artifact hashes and
new matched B64 quality ratios; X38 pilot comparisons use update 611 and full
comparisons use update 3052. These do not select an X39 gate-cost lambda.

## Part 4 operator amendment — cleared arms and match wire

The part-4 instruction clears arms 1–3 independently of the old semantic
failure. B64 remains binding. Arm 4' is a permitted lane-wiring deviation:
add a detached pre-read maximum scaled match score to the write-enable logit,
with empty-store evidence zero. Keep the sigmoid detached and thresholded;
do not alter certified READ/POP/WRITE/DECAY operations. The CPU witness uses
write bias 2 and match coefficient -1, zero error coefficient and POP disabled.
Both identity-keys/wired and semantic-keys/unwired ablations are required.
This is a constructive circuit receipt, not learned repeat selectivity.

The amendment does not select a lambda or authorize inventing a pilot control.
The exact-exposure B64 control required by the existing 2% selection rule is
absent; the immutable X34 checkpoint is B8 at 100,003,840 tokens. That unresolved
comparison limits lambda selection, not the validity of CPU mechanism checks.
No X39 training has run. The added broker wrapper executes only the targeted
decay receipt; it is not a training runner.

Registered before any X39 training. Status: **blocked at mechanism preflight**.
This registration preserves the operator's four levers; it does not silently
replace their semantics. No X39 scientific outcome is claimed.

## Control and invariants

Reuse X34 `runs/pilot_tape_lm/latest.pt` and
`receipts/probe_pilot_tape_lm.json` without retraining. Six blocks, width 256,
four attention heads, FFN 1024, RoPE, tape at block 3, four heads with identical
policies, context 512, seed 3401, X34 tokenizer and packed data order.
AdamW lr 3e-4, betas (.9,.95), weight decay .1, clip 1, fp32.
No X38 diversity machinery. X31/X33/X34/X36/X37/X38 remain read-only.

The requested effective batch is 64; **X34 actually used 8** (4096 target
tokens/update). Grouping eight consecutive X34 batches preserves exposure order
but changes optimizer updates. Consequently an X34 checkpoint is not a
batch-matched control for the requested training. This discrepancy must be
resolved before claiming an otherwise matched experiment or selecting lambda.
100M at batch 64 rounds up to 3052 updates / 100,007,936 target tokens;
20M rounds up to 611 updates / 20,021,248 tokens. Record exact exposure.

## Requested arms

0. Frozen, already-trained transcript control; no retraining.
1. Only decay changes: `decay_rate = exp(log(1e-3)/128)` with the existing
   free threshold 1e-3. Receipt both requested double and float32 constants and
   the T2 horizon `log(free_threshold)/log(decay_rate)`. This is the lifetime
   of untouched strength-1 slots, not a guaranteed cap under refresh/POP.
2. Append cost: lambda in {1e-3, 1e-2, 1e-1}, 20M tokens per trial. Select
   the largest lambda with held-out perplexity <=1.02 times control at the
   exact same exposure; no qualifying trial means **no operating point**.
   A selected lambda gets an independent seed-3401 100M run. The detached
   residual magnitude must remain detached. **Blocked:** the existing gate
   scales appended values; it does not gate allocation. Discrete allocation
   rate has no differentiable logit path. A gate-strength penalty must not be
   labeled an append-rate penalty. Defining a new allocation decision and its
   gradient requires an operator decision and new oracle semantics.
3. Train with S=128/head, evaluate the same trained weights at S=128 and
   S=2944. All other policies stay symmetric. Receipt R-CAP directly; scarcity
   is not a guarantee of learned selectivity.
4. A single shared learned W_in before frozen per-head-seeded key geometry,
   adapting X37 arm-c machinery, without X38 objectives. **Blocked:** X31/X36
   write matching addresses the age-1 slot, not a key-similarity match; changing
   keys affects reading, not concept novelty routing. R-ZERO-RESIDUAL allocates
   a new slot too. Making repeated concepts stop appending requires changed
   routing semantics and oracle/replay certification, beyond a W_in change.

## Primary compression receipt

Evaluate T={512,1024,2048,4096}, two held-out packed windows per T, tape reset
per window, plus the first 16 validation stories (cap 4096 with truncation
flags). Preserve per-head mean live count over tokens, peak live, final live,
and occupancy profile; do not substitute final count for mean count.
Fit ordinary least squares of log(peak live) on log(T), separately per head,
using maximum peak across the two windows. No zero-count log pseudocount:
zero occupancy is reported as degenerate and its slope is undefined.
Slope near 1 is transcript behavior; <=0.5 is strong compression only when
quality also passes. Budget-capped slope alone does not establish compression.

X34's existing control receipt already covers all four lengths; analyze it
directly rather than spending another GPU run. It cannot supply mean occupancy
or profiles that were not recorded. Retain source hash and mark missing fields.

## Secondary receipts and quality bands

Distinguish (a) write-enabled fraction, (b) successful allocations/token/head,
(c) append requests/token/head including R-CAP, (d) sigmoid gate strength.
The operator's predicted control "append rate 1" describes (a), whereas
physical successful allocation is about .5. Pre-register selective successful
allocation <.5, but report all four and avoid a denominator switch.

Receipt per-head pressure nonzero fraction, mean/max, raw R-CAP fraction and
W64 saturation (>=90% R-CAP in a trailing 64-token window). X33 establishes
an exhaustion detector, sometimes 96 depth before collapse, not a general
health detector. At control T<=4096, ~T/2=2048 remains below S=2944:
zero pressure in a rich deployment at these lengths cannot distinguish it
from control. A longer saturation stress is a separate, clearly labeled probe.

At matched tokens and fixed X34 validation windows, register short-context
perplexity ratios <=1.05 for decay, <=1.02 for append cost, poverty (each
deployment budget separately), and semantic keys. These are acceptance bands,
not predictions of measured results. For decay additionally report the entire
per-position NLL curve, 128-token bins, and each context's perplexity ratio;
long-context degradation is expected and must not be hidden by averaging.
Use story occupancy profiles to distinguish recent retention from flat growth.

Nulls: no lambda operating point; unchanged semantic append behavior; scarcity
does not transfer to rich deployment; low slope solely from destructive
saturation; poor read quality despite reduced occupancy. All are valid outcomes.

## Execution gates

Use only `/home/erikg/emender/.venv/bin/python`. GPU work must be leased via a
local run_leased.sh wrapper using the broker, cards 0–6 only, excluding 7 and
the occupied/certified cards. Sequential arms. No GPU run is authorized by
this audit while the mechanism blockers remain unresolved. Before every
training invocation, require fresh oracle parity for its exact configuration,
assert ReplayBackward in its loss graph, and record source/config/data hashes,
parameter totals/trainable counts versus base_lm, physical GPU lease, and
preregistration hash. No .pt files in git. Failed gates remain failures.

The Intercom contact_supervisor tool was not exposed in this session; tool
discovery found no supervisor or intercom tool. Pending decisions are recorded
here rather than guessed: allocation gate semantics, semantic match/no-op
semantics, and the batch/control mismatch. Training is not implemented or run.

## Part 2 amendment — operator redesign

The task instruction resolves batch policy to B64 and replaces arm 2's append
amplitude cost with `lambda * mean(sigmoid(W_write(x)))`, before the certified
detached write-enable threshold. Arm 4 combines a **single shared** learned
W_in before frozen per-head geometry with arm 2's selected cost. Keys alone
remain non-deduplicating; R-ZERO-RESIDUAL still appends. Earlier statements
requiring new allocation/no-op semantics are superseded for these redesigned
arms. No certified semantics change is authorized or needed for the gate cost.

Original `receipts/preflight.json` is retained unchanged. New CPU controller
and decay evidence is in `receipts/redesigned_mechanisms.json`; it contains no
training or GPU certification. Mechanism acceptance remains incomplete until
poverty loss and repeat-selective learned gating are demonstrated. No operating
lambda has been chosen. All quality bands, honest nulls and trained-arm metrics
above remain applicable, with arm 4 now using the combined objective.

## Part 3 measured mechanism disposition

Source-hashed machine evidence: `receipts/part3_mechanisms.json`.

| Pre-registered mechanism/prediction | Measured | Disposition |
|---|---|---|
| Decay deaths and equilibrium without saturation | Prior CPU receipt: deaths, equilibrium, no R-CAP | CPU passed; GPU parity unrun |
| Write cost indirectly reduces allocation through detached threshold | Prior costs 0/.001/.01/.1: .5/.5/.3359375/0 | CPU passed; lambda quality unrun |
| Poverty saturates, fires pressure, harms recall loss | S=8/16/32: peaks 8/16/24; R-CAP 64/32/0; recall CE 3.882920/2.166516/0 | CPU passed; no trained transfer result |
| Semantic reads confirm stored content and gate closes only with semantic keys | Repeat target read .090664 -> .999920; repeat gate and allocation remain 1 -> 1 | **Failed; stop before GPU** |
| Decay optimized-lane death timing matches CPU bit-for-bit | Not run after semantic failure | Uncertified |
| Four arms trained with parameter match and ReplayBackward | Not run after semantic failure | Incomplete |
| Occupancy slopes <=.5 with acceptable perplexity, allocation/pressure/profile and decay curves | No trained-arm measurements | Incomplete |
| Poverty disposition at S=128 and S=2944 | No trained weights or evaluations | Incomplete |

These CPU losses are synthetic binding recall cross entropy, not perplexity
bands. Semantic failure is a hand-set construction counterexample, not proof
that all learned contextual gates fail. No operating lambda was selected.
Do not launch GPU work on the basis of the ten passing regression tests:
the regression explicitly preserves the failed semantic mechanism status.
