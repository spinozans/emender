# STATUS — E97 interleaved teacher-generation pilot v1 (worker session)

Task: PILOT of the operator's generator+guide design — DeepSeek 4.1-flash
generator (LunarRoute background) attempting NEW terminal-bench-style tasks
through the real Pi sandbox, GLM 5.3 guide reviewing each trace, plus
WildChat-seeded open-ended conversations; measure records/hour per lane,
verification pass rate, guide-rejection distribution, token yield/hour,
queue latency. CPU only; GPUs untouched; never git add -A; long jobs
nohup'd. Do NOT launch the standing program — the pilot informs the
operator's scale decision.

**State: COMPLETE (measured, verified, reported; nothing admitted; no
standing program launched).**

Workspace (authoritative evidence):
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-teacher-generation-pilot-v1/`
— REPORT.md (measured rates, rejection distributions, 3 full verified trace
exemplars + 1 conversation, contamination-discipline receipt, scale-up
recommendation), STATUS.md, metrics-final.json, checkpoints (with .bak
pre-fix copies), candidate authorities for both arms, exemplar files.

Results (targets >=300 terminal + >=200 conversation were both exceeded):
- Terminal: 356/370 verified (96.2%); 2,551,345 tokens; 275,854 supervised
  assistant targets; 91.6 verified/lane-hour; ~71K target tokens/lane-hour.
- Conversations: 230/295 verified (78.0%); 162,764 tokens; 131,796 targets;
  70.7 verified/lane-hour.
- Generator: lunaroute/deepseek-4.1-flash-background; guide:
  lunaroute/glm-5.3-flash-background. Real Pi executions only (frozen
  pinned runtime + owner bridge + serve_pi_native_tools); mechanical task
  verification is the final bar; verified records rendered in the canonical
  Pi-native / conversation codecs (p50k tokens + masks + idx + jsonl),
  `verified-candidates-not-admitted`, `training_eligible:false`, zero
  optimizer updates.

Key findings:
- A standing bank of 2–6 background lanes exceeds the 30–80 lane-hour/epoch
  replenishment need with large headroom; queue latency and authoring
  diversity — not cost (LunarRoute advertises zero cost) — are the binding
  constraints.
- Two harness defects found, fixed, documented mid-pilot (prompt-whitespace
  session guard; clean-tree checker self-pollution), with auditable
  re-attempt/reverify receipts; naive pass rate would have read 87.6%
  before the corrections vs 96.2% after.
- Seven frictions to fix before scale-up (diversity bank, notification seam,
  conversation guard looseness, JSON budgets, episode deadline, guide JSON
  robustness, guide strictness policy) — see REPORT.md.

Repo artifacts (scoped commits): scripts/collect_e97_teacher_generation_pilot.py
(94e554f4 + follow-ups), docs/validation/e97-teacher-generation-pilot-v1.md.

GPUs untouched; CPU only; no `git add -A`; long jobs nohup'd throughout.
