# P1 — Section 6 integration

Worktree: `/home/erikg/emender-paper`; branch: `paper-1p3b-spine`.
Scope: `paper/main.typ`, Section 6 supporting tables, this report. No edits
made to the applications bridge or `paper/OUTLINE.md`.

## Milestone 1: completed expressivity slot

- Tables and prose read committed JSONs at Typst build time; measurements are
  not manually retyped. Added full length-grid modular-quadratic and S5 tables
  and the five-arm, three-seed contractive-control table.
- Terminal modular-quadratic accuracy: E97 99.96% at T128, 90.16% at T512,
  76.07% at T1024; GDN2-MLP 76.09%, 56.55%, 53.36% respectively. Its
  T8192 accuracy is 50.53%, **not** the declared random baseline 0.390625%.
- Correction approved by supervisor: the brief and earlier phase-2 commit
  title's “baseline by 8x” and “never exceeds 76%” are false against the JSONs.
  No current table/caption repeats them. GDN2's interim T128 trajectory probe
  reached 84.84%, so terminal accuracy must not be treated as a hard ceiling.
- Cap 18,269 is authoritative in `budget.json`; the pre-registration commit
  title's 18,263 is a title-only typo. Identical caps/plateau rules yielded
  **unequal realized training**: E97 2,500 steps, GDN2 2,200. The 4.35 GPU-h
  figure is projected capped training; actual training was 2,334.8 GPU-s
  (0.649 GPU-h), excluding loading/final evaluation.
- GDN-8M has 7,937,536 parameters versus E97's 7,376,896 (+7.6% rounded);
  seed-mean accuracy 0.896261 at T128 to 0.854350 at T1024. Measured mixed
  logistic Lyapunov exponent −0.20118: contractive control, not wall vehicle.
- Both initial S5 and capacity-summary negatives are quoted verbatim. S5
  does not replicate the predicted small-scale ordering. The capacity table
  flags GDN2's one T512 optimizer step and OOM-infeasible curriculum stage.
- D1 cited as compositional-state-depth duality; D2 as per-token-action routing,
  **not** a circuit-depth lower bound or full-stack impossibility.
- Replaced pending-slot bindings in contributions, claim table, limitations,
  and prediction; retained historical predecessor findings in Appendix B.

### Source file + immutable commit

All paths below are under `paper/results/expressivity_1p3b/` unless specified.

| Numbers/claim | Source | Commit |
|---|---|---|
| Full modular-quadratic grid, baseline, seed, scored positions | `iterated-map/ft_{e97,gdn2}_modular_quadratic_sweep.json` | `9dc3afd7` |
| Recipe, plateau stops, realized steps/time, interim probes | `iterated-map/ft_{e97,gdn2}_modular_quadratic.json` | `9dc3afd7` |
| Pre-registered cap, throughput, projected GPU-time | `iterated-map/budget.json` | `1989656c` |
| Five-arm parameter counts and seed-mean control grid | `iterated-map/phase1_8m_control_table.json` | `384e657c` |
| Logistic mixed-driver exponent | `iterated-map/lyapunov_logistic_map.json` | `0038ad84` |
| Pre-registered control/wall protocol | `iterated-map/PLAN.md` | `0038ad84` |
| Initial no-separation negative | `SUMMARY.md` | `5e1e60ef` |
| E97 S5 length grid | `capacity/cap_e97_s5_sweep.json` | `f1ed5615` |
| GDN2 S5 grid/OOM caveat and negative | `capacity/cap_gdn2_s5_{train,sweep}.json`, `capacity/SUMMARY.md` | `6f916843` |
| State-channel duality | `formal/lean/ElmanProofs/Expressivity/CompositionalDepthDuality.lean` | `32cfcc9c` |
| Routing bridge and explicit gaps | `formal/lean/ElmanProofs/Expressivity/S5GDNBridge.lean`, `formal/lean/GAPS-D2.md` | `82eec14d` |

Validation so far: `typst compile paper/main.typ /tmp/P1-section6.pdf`
and `git diff --check` passed. GPU experiments were not rerun; this task
integrates frozen results, not resilient-training behavior.

## Remaining task milestones

U2 CUDA scope and U3 per-sweep decomposition will be recorded below when
implemented and validated.
