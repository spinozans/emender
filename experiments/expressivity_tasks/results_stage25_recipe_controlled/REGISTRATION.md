# X25 — append-only recipe-controlled separator amendment

Operator approved 2026-10-23 ("go"); TASK_DIRECTIVE.md is the authority. This new
namespace amends, never rewrites, X23E/X24 registration or hot-control trajectories.
All five rungs are fresh seed0 initializations; commit AND push precede launch.
This is explicitly an evidence-informed recipe amendment, not a claim that no
prior task outcomes were inspected. Zero X25 optimizer updates before registration.

## Primary recipe and unchanged scientific predictions

Merrill Section6 paper protocol is now **primary**: ordinary AdamW lr1e-4,
weight_decay1e-4, betas(.9,.999), eps1e-8, batch64, clip norm1. LR is constant
from update1, with no warmup/cosine (none specified in Section6). Retain the
paper's 60K cap and three consecutive perfect ID-validation evaluations; use
X23E's eval-every500 and validation-best accuracy/loss checkpoint selection.
No test/OOD-based selection. X10-E97/X11-GDN2 transferred hot recipe (warmup5000
to1e-3, cosine to1e-5, wd.01) is demoted to **recipe-confounded control**, not an
architectural training verdict. The original X23E/X24 registrations explicitly
flagged its deviation/unvalidated cross-task transfer risk.

Predictions and thresholds are **UNCHANGED** from X23E/X24:
- Graph: E97 learns ID and generalizes OOD; GDN2 learns ID but degrades OOD;
  all four graph outcomes are informative. Integer: both learn ID and OOD.
- **activation-binding**: softplus >=99% ID and both OOD bins where tanh drops
  >=5pp ID-to-far-OOD; supports practical activation bottleneck/candidate host,
  not a production qualification or discharge of O10.
- **not-the-activation**: softplus also drops >=5pp despite ID learning; practical
  bottleneck elsewhere, O10 open.
- **fails-to-train**: inspected numerical instability/nonfinite logits or gradients
  recorded honestly, not an expressivity theorem. No automatic retries/staging.
- ID failure is separate from OOD degradation. Report exact counts/losses for all
  bins, including intermediate or tanh-also-generalizes outcomes. One seed is
  exploratory. Incomplete bins, OOM/disk/wall/external interruption are censored.

All task definitions, N bins, unary serialization, exact integer arithmetic,
class balance, final-only supervision, shared immutable pools/seeds, sample draws,
model shapes, legacy precision and fused engines remain X23E/X24. The actual
X23E datasets directory is linked, not replicated/regenerated. E97-softplus uses
X24's qualified plain-softplus fused engine, not centered softplus or eager FP32.
Historical max-bin **zero-update** smokes and X24 kernel-match gate are reused
and SHA-bound: no model/engine/input-shape changes in X25. They do not establish
stability at trained weights. The shared loop only exposes unchanged-default WD
and receipts the actual optimizer; the X25 wrapper supplies lr/wd1e-4.

## Three null observations and gradient evidence

`receipts/hot_recipe_evidence.json` binds source hashes, exact selected-checkpoint
test counts, terminal records, last updates/validation and preclip gradient windows.
1. Integer pair at60K: selected-checkpoint ID5080/10000 (E97),5020/10000(GDN2);
   losses .691642/.694782 (near ln2). Both OOD bins also near chance. This is not
   a claim every training/final-validation loss was ln2 (GDN2 final val1.214898).
2. Graph pair incomplete at supersession: tanh last update7916, validation7500
   10441/20000=.52205; softplus last update4852, validation4500
   10338/20000=.5169. Operator's earlier snapshots were ~52%@7500 and ~51%@4687.
   X24 preclip grad norms were repeatedly ~12–18; last500 updates range
   .1264–50.4111, mean13.5007. Tanh last500 range .0147–65.4027, mean1.9083.
   These are gradient-instability signatures, not direct state-norm measurements.
3. GDN2 graph crashed update5340 with nonfinite logits; last logged5339,
   validation5000 .5094. Last500 preclip grads .0666–81.6761, mean9.2495.
   The joint observations motivate the recipe-confound hypothesis; they do not
   prove a causal LR mechanism or architectural impossibility.

Both live graph arms are **RECIPE-CONFOUNDED/SUPERSEDED**, not training verdicts.
Their identity-verified owned process groups were terminated; terminal/result/
CENSORED marker receipts and `recipe_superseded.json` retain checkpoint hashes.
Existing best.pt and all streams stay in place as hot-recipe control data; no
checkpoint deletion, restart, namespace reuse or changed historical registration.

## GDN2 nonfinite diagnosis (existing evidence only)

`receipts/gdn2_nonfinite_diagnosis.json` and `gdn2_crash_traceback.txt` retain
what is extractable. Training fields: step, loss, lr, grad_norm, elapsed_s,
peak_allocated_bytes. No state norms or kernel telemetry appear in stream/crash.
Failure is detected in a training forward pass, before update5340's backward.
Nonfinite logits alone do NOT establish recurrent-state magnitude growth/overflow.
No new instrumented run. Diagnosis rides on the paper-recipe GDN2 graph arm's
outcome, which doubles as a numerical-stability check, not an automatic causal proof.

## Authorized launches and standard operational chain

Fresh namespace results_stage25_recipe_controlled:
- e97_graph_seed0: card0, tanh.
- e97softplus_graph_seed0: card7, plain softplus.
- gdn2_graph_seed0: card1, stock FLA negative-eigenvalue-enabled GDN2.
- e97_integer_seed0: card2, tanh.
- gdn2_integer_seed0: card3.

Mod97 NOT relaunched, queue note only, out of scope this pass. Controls not rerun.
No hardware/precision/architecture amendments, new seeds, or other tasks.

Registration/source/dependency hashes are committed+pushed first; independent
trainer verifies worktree AND HEAD blobs, launcher verifies matching remote ref.
Shell GPU leases only, no WorksGood. Record physical GPU snapshot, commands,
worker/trainer/watch PID/starttime/host, leases, config and dataset hashes.
Disk preflight>=400GiB free, <=40GiB/rung, wall<=7days; retain validation-best
checkpoint plus atomic staging only. Independent60s persistent watcher enforces
bounds with identity-verified owned-group TERM, KILL after30s. Hash-validated
result/terminal/marker receipts persist on exit, infrastructure is right-censored.
`receipts/watcher_wiring.json` supplies exact status/terminal paths and --once
probe for supervisor native arming. Resource guard runs independently; no native
notification arming is claimed by this worker.

Single-host local CUDA research, not resilient DiLoCo or Frontier Slurm. Normative
resilient requirement IDs R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 are not
applicable; no resilient conformance, Partition/QOS binding or scale gate claimed.
Independent acceptance review required; launches are not convergence results.
