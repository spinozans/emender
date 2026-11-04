# Registered B1-v2 retraining

Completed 35,419 steps / 71,677,042 raw-byte targets in 2700.032370 active
seconds: **13.117991 steps/s**, 149.736x the first attempt's 0.087607 steps/s.
The unchanged seed-3401, batch-16, context-128, BF16, AdamW 3e-4/decay .01,
clip-1 protocol ran on leased card 2. Dropout keep fraction was measured, not
assumed (see training_summary.json). Every window resets at its own document;
cache/source hashes and row identities remained checked. Every backward passed
the native/replay agreement check and nonfinite-gradient clipping guard.

The original production kernels and B1 forward kernel remain unchanged. The
82-test launch battery and cache bit gates passed before the fresh run.
Compilation/graph capture, source/cache read and native decode, assembly,
transfer, forward, backward, updates and telemetry were inside active time.
Only the standing checkpoint save/hash stalls were excluded; their individual
durations remain in the receipt. Full raw telemetry/curve are preserved in
training.json.gz, byte-identical to the working training.json. Observer samples
are occupancy_time_samples.jsonl and retain cumulative findability denominators.

Selected checkpoint (outside git):
/home/erikg/c14_dense_bytes/emender_c14_b1_v2_resume.pt
SHA256: 9c87a6a306b708608d7f466dbd62c70c53263c5322f24963b91b94afc8f4def8
Next training window: 566704. Optimizer and CPU/CUDA/dropout RNG states are in
the checkpoint. The active budget is complete; do not continue training it.

The 237-step first attempt's checkpoint and receipt hashes were checked and
are unchanged. Its raw training receipt and evaluation artifacts are also
retained as compressed first_attempt_* archives here. Their old PB labels are
not the operator-requested verdict. The 684-step slow-decode engineering attempt
is retained separately with its relocation manifest; neither attempt is mixed
into the final selected model.

Residual exposure risk remains: passive v2 ran 88,520 steps / 179,162,804
targets at 32.784951 steps/s. B1-v2 saw 40.0067% of that target stream at the same
registered active budget. Removing the 374x original deficit does not establish
equal exposure or an exposure-isolated architecture comparison. Report this
with the PB results. One seed and nonzero-weight findability also do not prove
generality or causal management benefit. Frozen-window paired evaluation and,
if PB1 moves, the registered conditional controls remain next. No PB verdict is
claimed by this training phase.
