# X26c Growth Report — Raw-Defaults Control Outcome (RAW-FAILS)

Registration commit **832c2115** (pushed origin/main before launch; ancestry
verified). Upstream snapshot **dbfa365ad680f425dfb79a9e78b5e0d7915757bb**
(14/14 files hash-verified vs `results_stage26_replication/sources/manifest.json`).
This report adjudicates the terminal outcome under the four interpretive
conditions of the accepted X26CR machinery review. It does not re-litigate
machinery; see `receipts/` and `ENVIRONMENT.md`.

## What ran (unmodified fidelity)

Both upstream bash launchers executed unchanged: their `train_rnn.py`
(`rnn_type=GRU`) and their `train_deltanet.py`, on their `imm_unbound` integer
task, with the frozen hash-identical X26b native datasets (generator as shipped,
seed0, T_train100/gap100, unbalanced, all five native contract differences
deliberately preserved). Their shipped hyperparameters: batch 256, 30k steps,
lr 3e-4, shipped BCE + 0.3 auxiliary loss (supervisor's CE misinstruction
corrected and receipted as concession #21). Isolated venvs; NumPy 2.5.4;
FLA 0.6.0; unavoidable torch 2.9.1+cu128 reuse receipted. Zero instrumentation
inside their training loop. A separate external evaluator loaded only their
selected best-ID checkpoint (SHA256 verified before/after), their
dataset/collate/eval bucket ordering/AMP, and reported pooled metrics.

## Terminal receipts

Both arms COMPLETE: `markers/*.DONE`, `receipts/watcher_terminal.json`
states `complete` for both, no training errors, no censors. Physical cards:
GRU card4/trainer PID1693325, Delta card5/trainer PID1693643; guard
PID1692488. Terminal steps at the shipped 30k default.

## Results (external frozen-checkpoint evaluator, pooled)

Prior disclosure (X26b class-1 priors): bin0 46.79%, bin1 76.54%,
bin2 89.61%. Raw accuracy alone is prior-tracking, not task learning.

| arm | bin | raw acc | balanced acc | class0 recall | class1 recall |
|---|---|---|---|---|---|
| GRU (rnn_integer_seed0) | 0 (ID) | 0.6296 | **0.6301** | 0.622 | 0.639 |
| GRU | 1 (OOD) | 0.7466 | **0.5145** | 0.077 | 0.952 |
| GRU | 2 (far OOD) | 0.8813 | **0.5049** | 0.030 | 0.980 |
| DeltaNet (delta_integer_seed0) | 0 (ID) | 0.6169 | **0.6115** | 0.695 | 0.528 |
| DeltaNet | 1 (OOD) | 0.7134 | **0.5200** | 0.156 | 0.884 |
| DeltaNet | 2 (far OOD) | 0.8769 | **0.5012** | 0.027 | 0.975 |

Training-console native diagnostics agree in band (e.g. GRU finalAcc
62.51/75.11/88.26 across bins) and are retained in the arm logs; they are
upstream mean-of-batch diagnostics, not pooled outcome claims.

Per-length receipts retained in each `result.json` under
`metrics.test_binN.by_matrix_length` with both class recalls and balanced
accuracy per length. Signature: matrix_length 1 -> balanced accuracy 1.0
(the trivial sub-case), short lengths partial, and progressive collapse to
single-class prediction at longer lengths (class-1 recall up to ~1.0 with
class-0 recall down to ~0.03-0.17 at OOD bins).

## Adjudication

**Pre-registered outcome cell: MIXED / ID-only, for both arms.** ID
learning is genuine but partial; OOD collapses to predominantly
prior-tracking with little balanced discrimination. The Figure-2-like
near-perfect pattern did not transplant. Per the registration, no binary
verdict is forced and no post-hoc "learns" threshold is invented. Details:

- **Partial in-dist signal, genuine**: bin0 balanced 63.0% (GRU) / 61.2%
  (DeltaNet) with BOTH classes predicted (recalls 62/64 and 70/53). This
  is comparable to the X26b native-protocol runs (GRU 64.38%, Delta
  61.89% balanced ID) and numerically lower in these single-seed
  observations — the raw bundle bought no more in-dist learning than the
  reconciled protocol did.
- **OOD = predominantly prior-tracking, with little balanced
discrimination**: balanced accuracy collapses to
  50.1-52.0% while class-1 recall reaches 88-98% and class-0 recall falls
  to 3-16%. The raw OOD accuracies (71-88%) are explained by the bin
  priors (77%, 90%) plus near-exclusive class-1 prediction. A constant
  class-1 classifier scores 76.5%/89.6% at bins 1/2; the models land within
  ~2-5 points of that baseline.

Under the X26CR interpretive conditions:

1. Terminal receipts + external frozen-checkpoint metrics exist for both
   arms — adjudication is permitted.
2. Prior disclosure and per-class recalls retained everywhere (table above;
   per-length tables in `result.json`).
3. **Classification: completed-failure, not environment-error.** Training
   completed cleanly (no import failures, no nonfinite deaths, no
   censors). The environment is theirs-as-recoverable: their repository
   specifies NO package versions (unpinned requirements), so the exact
   original environment is unrecoverable from the repo; our torch
   2.9.1+cu128/CUDA stack and GPU hardware differences cannot be fully
   excluded as environment deltas. This residual environment uncertainty
   is receipted, not adjudicated away.
4. **Bundle framing**: this failure identifies the shipped
   config+loop+environment bundle. It does NOT identify a unique causal
   delta (candidate deltas remain receipted in
   `results_stage26b_native_control/COMPARISON.md`), and it is NOT a
   verified refutation of their Figure 2: the published numbers may have
   been produced in an original environment the unpinned repository does
   not specify. What is established: the result does not survive
   transplantation to the environment the repository permits
   reconstructing.

## Program-level conclusion

The Merrill-26a empirical Figure-2 pattern did not reproduce in any of
the three fidelity levels run: X26 (their models, our reviewed contract,
reconciled protocol), X26b (their models on their native task code), and
X26c (their raw unmodified pipeline). In X26b and X26c, the raw-accuracy
numbers that superficially approach the figure's band are confounded with
a 47%->90% class-prior drift the paper does not mention, and the
prior-proof metrics show little balanced discrimination at OOD; X26's
evaluation pools were exactly balanced (raw equals balanced accuracy
there, per the trigger receipts), so its verdicts rest on the
prior-proof metric directly. What these runs establish: the published
pattern did not survive transplantation to the environment the
repository permits reconstructing. This is NOT a refutation of Figure 2 —
the original environment is unrecoverable — but any downstream use of
Figure 2 as evidence must disclose this three-level non-reproduction and
the prior-drift confound in the paper's raw-accuracy evaluation.

## Scope and limits

n=1 seed per arm, single host, qualitative verdict per registration (no
post-hoc numerical threshold invented). No claim that gen.py CLI defaults
(T300/200/50/50) generated Figure 2; the carried dataset configuration is
the controlled one, receipted in REGISTRATION.md. Public
revision-to-Figure2 provenance remains unverified.
