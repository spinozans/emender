# E97 interleaved teacher-generation pilot v1 — validation record

Pilot per the operator's design: a generator+guide arrangement on LunarRoute
background slots continuously producing verified terminal-task traces and
conversations, measured for a scale decision. **No standing program was
launched; nothing was admitted to training; zero optimizer updates.**

- Evidence workspace (authoritative):
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-teacher-generation-pilot-v1/`
  (REPORT.md with full measured rates, exemplar traces, contamination
  receipt, scale-up recommendation; STATUS.md; checkpoints and authorities).
- Collector: `scripts/collect_e97_teacher_generation_pilot.py`
  (freeze/collect/aggregate/reverify; committed).
- Arrangement: `deepseek-4.1-flash-background` generator attempts authored
  terminal-bench-STYLE tasks through the REAL Pi sandbox (frozen pinned pi
  runtime + owner bridge + `serve_pi_native_tools`; every tool result is a
  real execution), and writes WildChat-seeded conversations;
  `glm-5.3-flash-background` guide reviews every candidate trace/record;
  mechanical verification is the final bar.
- Outcome: 356/370 verified terminal records (96.2%; 2,551,345 tokens,
  275,854 supervised assistant targets; ~92 verified records per lane-hour,
  ~71K target tokens per lane-hour) and 230/295 verified conversations
  (78.0%; 162,764 tokens, 131,796 targets; ~71 verified records per
  lane-hour). Authorities rebuilt and independently validated
  (records/masks/index/token counts, every record decodes).
- Two harness defects were found, fixed, and documented mid-pilot
  (prompt-whitespace session-guard; clean-tree checker self-pollution by
  `__pycache__`), with pre-fix checkpoints preserved (`.bak` files) and an
  auditable `reverify` pass for the 32 checker-affected cases.
- Contamination discipline: tasks authored deterministically in-repo from
  seeded synthetic banks; no Terminal-Bench instance/corpus consulted or
  reproduced; frozen forbidden literal/number discipline enforced at freeze,
  generation, and guide time; WildChat seeds filtered (English, non-toxic,
  moderation-unflagged, deduplicated) and preserved byte-exact.
- Recommendation: a standing bank of 2–6 background lanes exceeds the
  30–80 lane-hour/epoch replenishment need by a wide margin; fix the seven
  documented frictions (diversity bank, notification seam, guards, budgets)
  before any launch. Launch is the operator's decision, uninformed by
  anything this pilot authorized.
