# Matched twin evaluation harvest

Do not declare quality before terminal paired evaluation. Active detached launch and verification are in EVAL_LAUNCH.json / eval_detached_launch.json / eval_survival.json. New kernel installed only in this process; frozen training checkout and arms untouched. No checkpoint is committed.

Watch eval_console.txt and eval_resume.json in this directory. Success requires eval_terminal.json exit_code 0, eval_resume.json status complete, 3,431,242 windows / 434,204,084 real targets, eval_table.json and verdicts.json present. Final-test total must be 214,097,022 for each mode. Failure/nonzero is failure, never completion. The inherited verdicts.json uses historical PB labels/reference; it is raw scaffolding, not PT1's causality conclusion.

If interrupted: verify launcher/child death and broker lease state first; retain failed terminal/logs. Resume detached with the same leased wrapper and --resume, using exactly the same amended source/hash config; shard-atomic eval_resume.json is authoritative. Do not start a concurrent evaluator. No need to restore training state: training is terminal exit0 at 35,419 steps / 71,677,042 targets. RESUME_STATE.json describes old training recovery history, not current evaluation.

Completion commit requirements:
- PT1: compute B1_on/final_test/total NLL minus budgeted 1.584700014. Negative: matched-exposure unbudgeted learns better, observed budget pressure value was compute shaping. Positive: evidence for curation pressure beyond compute in this recipe. Equality is a result. Keep capacity/implementation confounding and single-seed limits explicit.
- PT2: publish all 66 paired on/off split/stratum rows and verify identical denominators against frozen triangle arms. Append-only endpoint has 75,394,452 targets; do not call three arms exactly exposure-matched.
- PT3: report measured improvement per target vs dense anchor; assess descriptive 9x extrapolation/saturation without claiming sample complexity or interpolating unmeasured curves. Runtime projection is not a quality result.
- Update PROGRAM.md and theory note Part I with terminal results and causality ruling; bind committed receipts correctly. Run .venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/validate_program.py and retain green output. Commit complete receipts, 66-row table, PT1-PT3 interpretation, PROGRAM/theory updates; no .pt files.
- Retain checkpoint/source/index/window/journal hashes, masked document-boundary evidence, terminal markers and final process/broker verification. Never touch card1, union service, ~/sxgc or /mnt/nvme2n1.
