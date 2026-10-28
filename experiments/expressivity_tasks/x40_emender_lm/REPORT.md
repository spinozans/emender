# X40 implementation and launch receipt

The architecture and ordered experiment runner are implemented. **The study
is running; this is not a completed scientific result or an honest trained
null.** At this snapshot, 110 of3052 TinyStories updates completed
(3,604,480 target tokens); final quality, context extrapolation,
compression and curriculum-depth outcomes are unmeasured. The running
process is PID4041271 under broker wrapper PID4041193 on physical card0.
The live curve and train log continue after this snapshot.

Validation completed before this training invocation:
- Unchanged X31 suite via the X36 adapter:109 passed (oracle.log/oracle.xml).
- Independent sequential X31-state evidence/error construction at B64/T512
  for every hierarchy policy, then bitwise full-scan values/state/traces/
  pressure/gradients against unchanged X31: all four passed (parity.json).
- Whole pure stack B64/T512 backward:24 ReplayBackward nodes, finite gradients
  (smoke.json). Every subsequent optimizer step asserts24 replay nodes.
- CPU architecture/count/T2/wiring tests:3 passed (cpu.log/cpu.xml).
- Total parameters72,162,960 versus widened base72,162,982: relative gap
  3.0486545e-7, below0.1%. Trainable counts21,719,112 versus72,162,982.

The implementation extends X39 locally with the requested error term in the
detached enable wire and composes the read-only X36 replay. Training uses
no attention. Keys and token features are frozen/detached as registered;
queries, W_in and value/content paths learn. Thresholds/ISA are unchanged.
All upstream experiment files remain untouched by this work. No WG service,
worker, dispatcher, Slurm or Frontier execution was started. Resilient
DiLoCo requirement IDs are not applicable to this single-GPU experiment.
AGENTS.md and CLAUDE.md were not changed and compare equal.

Run artifacts bind source/data/config hashes. The initial preflight preceded
final runner editing; registration.json and the current parity/smoke receipts
are from the fresh training invocation, after those edits. preflight.log
records the earlier successful smoke invocation. One initial wrapper attempt
failed with permission-denied before chmod; it acquired no GPU and ran no
Python. No .pt artifacts are written.

Residual risks and pending evidence:
- No final100M-token quality result exists yet. Training loss is not validation
  perplexity and does not establish any registered quality band.
- Five-length context evaluation,24 compression slopes, pressure/routing
  profiles and16-story hierarchy counts await pretraining completion.
- Stage3 follows the original X28E30000-update budget, which is substantially
  longer than the informal brief-probe wording. It has not started.
- No matched fine-tuned transformer depth receipt was identified. The runner
  reports its own OOD bins but explicitly withholds comparative separation.
- X34 transformer/hybrid long-context references used by the evaluator are
  historical B8 receipts, with that protocol limitation retained.
- The runner has no model/optimizer checkpoint or resume path. Interruption
  loses learned state; a pre-existing curve prevents accidental relaunch.
- The contact_supervisor tool is unavailable (name/description discovery
  returned no matches). No alternative orchestration service was used.
- Code-corpus training is deferred until items1–4 complete.

Monitor `receipts/curve.jsonl` and `receipts/train.log`. Terminal training
status is written to `receipts/training.json`; final context/compression to
`receipts/evaluation.json`, and depth to `receipts/depth.json`. Absence of
those terminal artifacts must never be treated as success.
