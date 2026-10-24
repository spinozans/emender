## Review

**Verdict: ACCEPT-WITH-CONDITIONS — control machinery only, not training outcomes.**

No issues found.

### Correct

- **Shipped-pipeline fidelity:** `scripts/launch_merrill26_raw_defaults.sh:38–44` invokes the upstream bash launchers under `env -i`, overriding environment/data/output paths rather than training hyperparameters. Both launchers specify batch256, lr3e-4, 30k-step/500-epoch caps, auxiliary weight0.3, and their native regularization, curriculum, scheduler and stopping configuration (`…/imm_unbound/rnn.sh:31–63,180–197`; `…/imm_unbound/delta.sh:33–86,185–208`).
- **Correct loss and collate:** shipped final BCE plus weighted auxiliary BCE remains intact (`…/train_rnn.py:400–421,490–494`; `…/train_deltanet.py:413–432,502–506`). Native collate supplies BOS+MAT inputs, final-MAT labels and output-only intermediate auxiliary targets—not STATE teacher forcing (`…/train_rnn.py:229–295`; `…/train_deltanet.py:233–294`). The mistaken CE instruction and concession #21 are explicitly receipted in `results_stage26c_raw_defaults/ENVIRONMENT.md:3–12`.
- **Environment disclosure:** prelaunch and runtime receipts record torch2.9.1+cu128, CUDA12.8 and NumPy2.5.4. Delta uses FLA0.6.0; GRU merely exposes inherited, unused FLA0.5.1. Separate venvs reuse project CUDA torch/dependencies through `.pth`; this is explicitly **not hermetic isolation** (`ENVIRONMENT.md:16–37`; both `receipts/*/environment_runtime.json:2–9`). Setup logs show upgrades installed locally without uninstalling base-environment packages.
- **External evaluator, zero loop instrumentation:** the worker waits for successful upstream exit before starting the evaluator (`scripts/launch_merrill26_raw_defaults.sh:70–75`). Evaluation uses strict checkpoint loading, eval/no-grad operation, native dataset/collate/bucketing and AMP, with checkpoint SHA256 checked before/after (`experiments/expressivity_tasks/run_merrill26_raw_defaults.py:46–110`). It reports pooled balanced accuracy, both recalls, confusion matrices, class counts, majority baseline and per-length metrics.
- **Prior drift and predictions:** `REGISTRATION.md:32–63` freezes raw-succeeds/raw-fails/mixed outcomes and prohibits raw-only interpretation. Frozen class1 rates are **46.79%, 76.54%, 89.61%** (`results_stage26b_native_control/receipts/data.json:55–84`).

### Trigger evidence

All four triggers are completed 60k runs, hash-bound in `receipts/trigger.json`.

| Trigger | Balanced accuracy: ID / OOD1 / OOD2 |
|---|---|
| X26 GRU | 52.30% / 49.99% / 50.53% |
| X26 Delta | 51.48% / 51.05% / 50.68% |
| X26b GRU | 64.375% / 51.133% / 50.086% |
| X26b Delta | 61.892% / 51.255% / 50.287% |

X26 pools contain exactly5000 examples/class, so pooled raw equals balanced accuracy; historical per-class recalls were not logged. X26b class0/class1 recalls are:

- GRU: ID **64.292/64.458%**, OOD1 **5.584/96.681%**, OOD2 **1.155/99.018%**.
- Delta: ID **65.439/58.346%**, OOD1 **7.161/95.349%**, OOD2 **1.155/99.420%**.

These support **partial ID learning, OOD prior-tracking**, not chance ID learning (`receipts/trigger.json:46–175`).

### Validation and provenance

- Receipted **46/46 tests**, prelaunch and postlaunch (`receipts/tests.txt:6–14`; `tests_postlaunch.txt:6–14`).
- Both N300/batch256 backward+eval smokes passed with zero optimizer updates (`receipts/smoke_{rnn,delta}_integer.json:2–12`).
- Both evaluator smokes passed, checkpoint unchanged, covering native boundary lengths1/100/101/200/201/300 (`receipts/evaluator_smoke_{rnn,delta}.json:2–12`).
- Launch verification records GRU **card4/PID1693325**, Delta **card5/PID1693643**, physical UUIDs, and guard **PID1692488** (`receipts/launch_verified.json:10–186`).
- Registration-before-launch is supported by `receipts/launch.json:2–10`, supplied `x26c_history.txt`, and the parent’s explicit DAG audit. Upstream14/14 hash fidelity relies on the parent’s supplied hash audit; I inspected the current source seams, not a committed-range diff. Tests were reviewed through receipts, not independently executed.

### Exact conditions

1. **No scientific adjudication until terminal receipts and external frozen-checkpoint metrics exist.**
2. Every interpreted result must retain balanced accuracy, both recalls and prior/baseline disclosure; unchanged upstream raw diagnostics are not outcome evidence.
3. Classify completed training separately from pipeline/evaluator errors and infrastructure censoring. Diagnose environment-broken versus completed failure where evidence permits; otherwise retain uncertainty.
4. Preserve the registered bytes and three-outcome framing; successful control identifies the configuration/loop/environment bundle, not a unique causal delta or verified Figure2 reconstruction.
5. Parent completes its designated audits and supervisor-owned notification arming before unattended acceptance.

- **Fixed:** None; read-only review.
- **Merge verdict:** **OK with notes.**
---

## Parent attestation (2026-10-24)

X26CR conditions satisfied: (1-4) The interpretive conditions are adopted verbatim as the
outcome-adjudication protocol: no scientific adjudication until terminal receipts and
external frozen-checkpoint metrics exist; every interpreted result retains balanced accuracy,
both recalls, and prior/baseline disclosure (unchanged upstream raw diagnostics are not
outcome evidence); completed training classified separately from pipeline/evaluator errors
and infra censoring, with environment-broken vs completed-failure diagnosed where evidence
permits and uncertainty retained otherwise; the registered bytes and three-outcome framing
preserved, with a successful control identified as the configuration/loop/environment BUNDLE,
not a unique causal delta or a verified Figure-2 reconstruction. (5) Parent audits complete:
the reviewer's exact 46-test suite (5 files, per receipts/tests.txt) reruns green under the
parent - 46 passed in 7.60s; upstream 14/14 hash fidelity re-verified (parent audit supplied
to the reviewer); registration-before-launch DAG ancestry confirmed (832c2115 precedes
059a881b); supervisor-owned notification armed BEFORE this acceptance: process proc_c5c6
(PID 1704278) watches the chain with native wake - both arms running per its latest poll.
Local HEAD equals origin/main.

Verdict: X26CR ACCEPT-WITH-CONDITIONS -> ACCEPTED. X26c raw-defaults machinery closed;
outcome adjudication proceeds only under the four interpretive conditions above.
