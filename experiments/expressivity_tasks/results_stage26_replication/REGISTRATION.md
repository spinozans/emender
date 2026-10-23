# X26 pre-registration: partial replication under reconciled sources

Registered before any optimizer updates. Target is Merrill et al.2026 Figure2's
**qualitative** pattern, not an exact reproduction claim. Read DISCREPANCIES.md
for the exact Table1 extraction, upstream receipts and approved precedence.

## Frozen predictions and interpretation

1. Their task-specific nonlinear RNN on DGC: ~100% in-distribution accuracy,
   strong length generalization in both OOD bins.
2. Their DeltaNet (and RWKV-7 if subsequently qualified/registered) on DGC:
   high in-distribution accuracy, substantially degraded OOD relative to RNN.
3. Their RNN and DeltaNet on IMM: near-perfect in-distribution accuracy.
   These are qualitative targets; no unreported numeric Figure2 coordinates
   or post-hoc success thresholds are asserted.
4. A failure of their models on our contract does NOT by itself prove a
   generator bug; it triggers the native-tokenization control and the
   field-by-field comparison. That control requires a new registration,
   not automatic launch or mutation of this instrument. Compare graph
   distribution/numbering, encodings, labels (>0 vs==0), product order,
   balancing, overflow/clipping, headers and teacher forcing before attribution.
5. Numerical/software failures are errors, resource/wall failures are censored;
   missing threshold crossings are not evidence of theoretical inability.
   No architecture-divergence claim is licensed before the control succeeds.

## Four primary rungs

`rnn_graph_seed0` card2; `delta_graph_seed0` card3;
`rnn_integer_seed0` card4; `delta_integer_seed0` card5, all lambda01.
Only these four are authorized. Start fresh seed0; no checkpoints restored.
DGC nonlinear model = upstream clamped ReLU1x256 with emb128 and mean pool.
DGC DeltaNet = upstream1x256/h4, emb128, mean pool.
IMM nonlinear = upstream GRU2x256/drop.3; IMM DeltaNet1x256/h4/drop.2.
IMM native encoders replaced by scalar embedding4x256 preserving our task;
aux.3 >0 targets at every intermediate matrix's ninth scalar, output-only.
Curriculum T5->100/150 eligible-pool epochs. No warmup (paper overrides repo).
These reconciled dimensions differ from Table1 L2 and are not misrepresented.

Both optional RWKV configs failed zero-update max-bin probes (`v_first=None`
upstream FLA); neither is trained. Repo-default DGC control deferred because
its shell script uses an unrecognized flag; no 'unmodified' control claimed.

## Tasks, training, evaluation

Use the actual shared X23E frozen compact pools via datasets symlink; unchanged
reviewed Def11 typed unary graph and Def12 exact integer-positivity scalar
stream. Pool generation, tokenization and oracles are not edited. Balanced
train70k / validation20k / ID10k in n1–100, OOD10k each in101–200/201–300;
split seeds23001–23005. Dataset receipt hashes bind all already-existing pools;
missing graph test pools are materialized only at terminal evaluation with the
same frozen generator/spec/seed. Never use OOD for stop or checkpoint selection.

AdamW constant lr1e-4/wd1e-4, betas(.9,.999), eps1e-8; batch128 effective
examples via length-aware gradient accumulation; norm clip1.0; BCEWithLogitsLoss
on DGC two-logit margin or IMM scalar last-token logit. Auxiliary loss normalized
across the entire effective batch's intermediate prefixes, independent of
microbatch boundaries. bf16 autocast; upstream DGC recurrence forces fp32 inputs
as in its implementation (autocast behavior unchanged). No NaN replacement.
Evaluate ID validation at update0 then every500 updates; stop after three
consecutive100% validation evaluations or60000 updates. Choose best validation
accuracy, tie-break validation BCE; evaluate frozen ID/OOD tests only afterward.
Training is shuffled eligible-pool passes, epoch tails carried to retain128;
this sampling adaptation and curriculum timing are disclosed.

## Launch / guard contract

Source+dependency SHA checks and HEAD blob verification; registration commit
must match pushed upstream ref before spawn. Four zero-update max-bin backward
and eval gates required. Standalone shell lease per idle physical GPU, worker
and trainer PID/starttime/host receipts; no WorksGood service machinery.
Independent60s guard: min400GiB free,40GiB/rung,7days; TERM then KILL only
identity-verified owned groups. Resource censoring is not a training verdict.
Custom Python unroll of very long unary graphs may be slow; no throughput or
completion-time promise. Native notification arming belongs to supervisor;
watcher_wiring.json provides probe and terminal target after launch.

Local .venv Python3.12/CUDA, not Frontier. Resilient requirement IDs
R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 are not applicable.
Independent acceptance review remains required.
