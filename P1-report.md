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

## Milestone 2: U2 closed — CUDA-only evidence scope

`grep -nE 'CUDA|ROCm|Portable|throughput' paper/main.typ` located the
backend claim in Section 4. Throughput heading and prose now explicitly
scope all performance measurements and budget smokes to CUDA only.
Triton source portability is not reported as ROCm correctness, throughput,
or parity; the separate single-arm Frontier continuation cannot establish
paired portability. Removed the unmeasured dispatchability/port-time claim.
No ROCm measurement was invented. `typst compile` and `git diff --check`
passed after this milestone.

## Milestone 3: U3 closed as an audited decomposition plus explicit gap

At the Section 5 loss-parity binding, the new table computes all 14 retained
sweep rows directly from the normalized CSVs: snapshot status, candidate
trajectories, retained generation records, generation-zero best candidate
mean-loss fitness, best observed fitness, and their difference. Failed and
partial sweeps remain visible. This is an observational within-sweep gain,
not a causal architecture-versus-tuning decomposition.

These historical/redo snapshots differ from the later eleven-family
leaderboard and **are not mapped to the carried-in final winner searches**.
The additional E97-MLP and GDN2-MLP rows identify those searches explicitly.
GDN2's retained winner/count/generation values remain cited in the existing
provenance prose, rather than copied or silently substituted from a snapshot.

Numbered **Open item U3.1** covers only those two carried-in winner rows:
“carried-in winner first-generation fitness unavailable; within-sweep gain
not computable from retained artifacts”. Missing first-generation spread and
E97 candidate/generation totals in this normalized snapshot are also stated.
No unavailable quantity is estimated; failed sweeps' missing generation-zero
fitness stays a dash, not zero.

Sources:
- `paper/results/cmaes_burst_v2/cmaes_eval_summary.csv` and
  `cmaes_generation_summary.csv` at
  `899e91cd6716d0867fba928306f9437a7af97cbc`.
- Carried-in GDN2 metadata: `docs/repro/lb_gdn2_mlp_20260612/best.json` at
  `f2e88dc2d4cb3e3fc3142efe8c485da8db3f6ac3` (first-generation data absent).
- Carried-in E97 identity: existing provenance in
  `experiments/lb_compare_20260613/REPRODUCTION.md`; no normalized winner
  generation series is retained.

## Final validation and acceptance evidence

- Added `paper/figures/validate_section6.js`: CPU-only, independent result/CSV
  parsing compared against **rendered Typst query cells**, not merely source
  text. Checks all 7 wall-grid rows, 5 control rows, 6 S5 rows, 14 snapshot CMA
  rows plus both U3.1 rows; verifies verbatim negatives, common-cap versus
  unequal-realized-step caveat, non-baseline GDN2 result, CUDA scope, and no
  remaining TODO-EXPRESSIVITY-1P3B.
- `node paper/figures/validate_section6.js` — PASS, all rows match.
- `typst compile paper/main.typ /tmp/P1-final.pdf` — PASS (Typst 0.14.2).
- `typst query` independently exposed the figure bodies for the numeric tests.
- `typst compile paper/main.typ '/tmp/P1-page-{p}.png' --format png
  --pages 21,26-28 --ppi 85` — PASS. Visually inspected CMA and all three new
  expressivity table pages: no clipping; tightened the CMA count header to
  avoid crowding. These local validation renders are not committed.
- `git diff --check` — PASS. No new GPU run, Slurm submission, native build,
  Python execution, or resilient-training change was needed.
- Milestones committed and pushed to `origin/paper-1p3b-spine`: Section 6
  `f734a0b5`; CUDA/U2 `3d3885c1`; U3/validation is the commit containing this
  final report. No files are left staged at completion.

Residual risks: single-seed 1.3 B probes; common cap but unequal realized
training; interim-versus-terminal variation; GDN2 S5 curriculum OOM; U3.1
winner-series evidence gap. Existing held-out bpb/downstream and formal
inventory-refresh TODOs are outside P1. Review gate remains required.
Recommended next step: independent review of the frozen-data bindings and
U3.1 disclosure; recover winner generation logs and run pre-registered
multi-seed/common-budget follow-ups separately if authorized.
