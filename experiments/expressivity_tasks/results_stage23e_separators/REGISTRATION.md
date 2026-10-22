# X23E — theorem-bearing separator tasks (pre-training registration)

Authority: operator task X23E (2026-10-22), Merrill, Jiang, Li, Lin & Sabharwal,
*Why Are Linear RNNs More Parallelizable?* (ICML 2026), archived here as
`PAPER_SOURCE.md`. The live supervisor resolved the contradictions below BEFORE
training. No optimizer updates, experimental dataset generation, or outcome-based
recipe selection preceded this registration. Max-length smoke tests use disposable
seed239999, random untrained models, backward/gradient checks, and **zero updates**.
All eight smoke receipts, including observed peak allocation, are bound in JSON.

## Protocol authority and explicit deviations

1. **Graph primary = Definition 11**, source s, sorted deterministic directed
   edges, target t, unary node indices, single causal left-to-right pass, accept
   iff s reaches t (including the zero-edge path). Appendix F instead says
   binary serialization; that reading is NOT run. Sampling follows F's two
   disjoint increasing bucket paths, p=.5 for interior bucket assignments,
   source0/targetN in the same bucket for positives and different for negatives.
   F calls this n vertices yet names endpoints0 and n: we explicitly use its
   endpoint convention, vertices0..N (N+1 vertices), and bin by N. No relabeling
   of binary-encoded inputs as the unary problem. Source/edge/pair/target
   delimiters make empty unary0 unambiguous. EOS and SOS are framing only.
   Maximum N300 worst-case input is **90,904 tokens**, not 300 tokens.
2. **Integer primary = Definition 12 + Theorem5 positivity**: exactly9N tokens
   in {-1,0,1}, row-major 3x3 blocks; P=A1...AN by right multiplication; label
   1 iff P[0,0]>0. Products use Python arbitrary-precision integers, NEVER float,
   int64 accumulation, clipping, or modular surrogates. F instead uses
   P[0,0]==0 and optional intermediate clipping; that is a different target and
   is NOT run. Its entry probabilities(.45,.10,.45) are retained. Conditional
   rejection produces balanced binary labels at each sampled length; N itself
   is drawn uniformly and never redrawn when rejecting a matrix sequence.
   Input contains no length header, product, intermediate target, or answer.
3. **Z_97 secondary = authorized adaptation, not exact replication**. F leaves
   its prime m and query index unspecified, uses invertible iid matrices and
   prefix-entry classification (unlike main-text binary BCE). We explicitly
   choose m97, query(0,0), final-entry 97-way classification, no prefix targets
   or headers. Entries are uniform in{-1,0,1}, each matrix rejected until its
   exact determinant is nonzero mod97. Arithmetic reduced mod97 each step.
   Natural 97-class sampling is retained, with full label histograms; the main
   section's balanced-BINARY requirement does not imply 97-way rejection.
   Mod97 is finite state and is NOT the theorem-bearing arithmetic separator.
4. **Control = X-series ParityTask(mode='final')**: N iid bits (conditioned on
   an exactly balanced parity), followed by its predict-marker2, final binary
   parity target. This is inside both recurrent classes; GDN negative eigenvalues
   are enabled. It shares the length bins and sample/optimizer budgets, not
   the graph's quadratic unary token count. Matching semantic N is not a claim
   of matched token count, arithmetic complexity, or equal learnability.
5. **Main Section6 dataset/stopping template**: fixed70,000 training examples,
   fixed20,000 validation examples, independent10,000-example test splits in
   N[1,100], [101,200], [201,300]. Binary splits are exactly balanced. Fixed
   independent PCG64 split seeds23001..23005; same pools across architectures,
   generated only AFTER registration. Samples at tiny N can naturally coincide
   across splits; no claim of disjoint exhaustive finite input support. Compact
   NPZs/labels/length hashes are receipted and shared under a file lock, never
   regenerated after publication. Training draws64 examples with replacement
   per update using PCG64 seed23000. Supervision is final position only.
6. **Strongest architecture-specific recipe family, NOT the paper recipe**:
   the paper Section6 says AdamW1e-4/wd1e-4/batch64; F says batch128 and its
   extracted LR table is incomplete. Batch64 is primary here. Approved strongest
   known recipe family for these exact architectures is X10-E97/X11-GDN2 ordinary
   AdamW: betas(.9,.999), eps1e-8, all-parameter wd.01, clip norm1. Linear warmup
   updates1..5000 to1e-3, cosine to1e-5 at60,000. This compresses the historical
   500K schedule to the paper's60K cap, explicitly an unvalidated cross-task
   transfer, not a demonstrated optimum on separators. No ignition-lottery
   ScheduleFree/default-optimizer confound, no silent restart or teacher forcing.
   Recipe receipts: `results_invertible_stage10_optimizers_8m/registration.json`,
   `results_invertible_stage11_gdn2_adamw_8m/{registration.json,REGISTRATION.md}`;
   implementation `run_invertible_stage{10_optimizers,11_gdn2_adamw}_8m.py`.
7. Effective batch64 uses variable padded-token-budget microbatches (8192 graph,
   16384 others), sum losses /64 before accumulated backward. One long example
   may exceed that budget; **no cropping/truncation**, sequence chunking, recurrent
   state carry across examples, or alternate model equations. Validation baseline
   and every500 updates; early stop after three consecutive nonbaseline perfect
   20K validation evaluations, else60K. Select lowest validation-error/loss
   checkpoint, then evaluate all three full test bins once. Test/OOD never
   controls stopping, staging, hyperparameters, or checkpoint selection.

## Architecture control

Stock X-series recurrent cores: dim256, depth4, MLP ratio2; E97 tanh-state,
raw_write=False, heads32/state32, expansion1, existing split-edit Triton kernel
and fused evaluation path, legacy precision. FLA fla-gdn expansion8.625 with
allow_neg_eigval=True, stock chunk recurrence (the wrapper's n_heads32/n_state32
arguments do not override FLA's stock two 128-key-dimension heads). Float32
parameters, bf16 autocast;
no precision adapter or custom kernel. Only tied task vocabulary changes:
graph8, integer/control4, mod97=97 (only first3 scalar ids used as real inputs).
Binary BCEWithLogitsLoss uses class1-minus-class0 logit; mod97 uses CE.
The recurrent shapes are identical to historical stock shapes, not equal total
parameters after changing the vocabulary. All rungs are fresh seed0 initializations,
no historical checkpoint loading or learned preprocessing.

## Predictions locked before training

- Graph: E97 succeeds ID and generalizes OOD; fla-gdn succeeds ID but degrades
  OOD. If fla-gdn also generalizes, report a real practical finding, not an
  invalidated experiment or theorem disproof. A finite length sweep at practical
  precision cannot refute an asymptotic conditional complexity upper bound.
- Exact integer positivity: both succeed (inside DPLR's theorem-bearing class).
- Control: both succeed. Secondary adapted mod97: both learn final-entry tracking.
- Operational success means >=99% test accuracy in a bin; graph degradation means
  at least5 percentage-point ID-to-farthest-OOD drop. Report exact counts/losses
  in ALL bins even between thresholds. For mod97 use same99% criterion, chance1/97.
- Four graph outcomes are all informative: E97-only OOD, both OOD, neither OOD,
  or GDN-only OOD. Neither is a learnability/resource finding, never proof of
  inexpressibility; GDN-only challenges the anticipated practical inductive bias.
  Failed controls flag a recipe/learnability confound, not selective architectural
  inability. ID failure is reported separately, not rebranded OOD degradation.
  One seed is exploratory, not robust cross-seed evidence. Missing thresholds,
  incomplete validation/tests and infrastructure stops are right-censored.

## Launch chain and bounds

Rungs: card0 E97 graph, card1 GDN graph; card2 E97 integer, card3 GDN integer;
card4 E97 control, card5 GDN control; card6 E97 mod97 THEN GDN mod97. Card7 unused.
Before full runs all eight max-bin forward/backward smokes must pass; exact peaks
are embedded in registration.json. No optimizer update in a smoke test.

Registration, all operational/model/task sources, test receipts and smoke receipts
are SHA256-bound, committed AND pushed BEFORE launch. Trainer independently checks
worktree and HEAD blobs. Launcher verifies remote branch commit, records registration
SHA/commit, physical GPU snapshot, host/PID/starttime identities, commands, leases,
trainer configuration, dataset hashes, logs, checkpoint hash and terminal receipts.
GPU leases are native shell locking only, NOT WorksGood service/orchestration.
Persistent watcher polls60s, validates terminal result hashes and marker equality,
serializes secondary card6 rungs, and writes growth/censoring evidence.

Keep only validation-best full-resume checkpoint plus atomic staging (at most2
payloads); overwrite removes the prior best rather than accumulating checkpoints.
Disk guard: >=400GiB free and <=40GiB per rung; device allocation OOM is censored;
wall bound7days including data/validation/test time, independently enforced by
watcher. Guard signals only identity-verified owned process groups (TERM then KILL).
Infrastructure failures never count as optimization/expressivity failures. Software
errors are errors with no scientific verdict, not success. No automatic retries,
extensions, changed lengths, post-result primes, or post-result recipes authorized.
This is standalone single-host research, not resilient DiLoCo; its normative
requirement matrix does not apply and no resilient-training behavior is changed.
