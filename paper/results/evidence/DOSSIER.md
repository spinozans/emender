# Claim → Artifact Provenance Dossier

**Generated:** 2026-10-06
**Worktree:** `/home/erikg/emender-paper` (branch `paper-1p3b-spine`, HEAD `29bb9d0d`)
**Paper source:** `paper/main.typ`
**Machine-readable companion:** [`DOSSIER.json`](DOSSIER.json)

This dossier binds each load-bearing claim in the Emender paper to the
committed artifacts that carry it, so the paper can point readers at a
"committed run artifacts" surface. Each artifact records its path, the
most recent commit that touched it, and what it shows.

- **Commit** = `git -C /home/erikg/emender-paper log --format=%h -1 -- <path>`.
- **Bound** = at least one committed artifact exists at a known commit.
- **Bound-partial** = a committed artifact exists, but a sub-claim's raw run
  data or an auxiliary measurement is not committed.
- **Unbound** = no committed artifact.

## Summary

| Status | Count |
|---|---:|
| Bound (incl. partial) | 17 |
| Unbound | 4 |
| **Total claims** | **21** |

Bound-and-partial claims: 4 of the 17 carry a noted weakness (raw run JSON
or a sub-measurement not committed). They are flagged in the table below.

## Bound claims

| # | Claim (short) | Section | Primary artifact(s) | Commit | Status |
|---|---|---|---|---|---|
| C1 | E88 1.273B reaches 0.973 bpb on The Pile, ~23 GPU-days, single GPU | §5, §1 | `paper/results/figure_2/E88_NDM.csv`, `AS_OF.md`, `figure_2.png`; `paper/review/THROUGHPUT.md` | `4a18eda1`, `d86ee36b` | bound |
| C2 | Multi-programming: 22,200 programs/token, 370 heads × 32×32, depth 12 | §1, §4 | `formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean`; `formal/lean/PROOF_INVENTORY.md` | `f6dfe8ae`, `8b5e9502` | bound |
| C3 | Per-architecture CMA-ES protocol; search selected high head count | §5 | `paper/results/cmaes_burst_v2/cmaes_eval_summary.csv`, `SYNTHESIS.md`, `CMAES_LOG_INVENTORY.md` | `899e91cd`, `14e7dd82` | bound-partial (per-sweep counts not decomposed; see U3) |
| C4 | Train-loss bpb 0.973 / 0.973 / 0.979 at v0.3 | §5 | `paper/results/figure_2/{E88_NDM,FLA_GDN,M2RNN_CMA}.csv`, `AS_OF.md` | `4a18eda1` | bound |
| C5 | Held-out bpb 0.966 / 0.966 / 0.961; 5-slice statistical tie | §5, App. B | `paper/review/HELDOUT_MULTISLICE.md`, `heldout_multislice_slices.json`, `BPB_FULL_TABLE.md`, `PILE_BPB_MEASURED.md`, `scripts/measure_pile_bpb*.py` | `b1dfca62`, `76c46332`, `fdca5edc` | bound |
| C6 | Common-Pile contamination control | App. B | `paper/review/COMMA_PILE_BPB.md`, `comma_slice.json` | `76c46332` | bound |
| C7 | M2RNN-paper diverges at step 8,400 (grad 4.2e7); CMA reshape stable | §5 | `paper/results/figure_2/SOURCES.md`; `docs/M2RNN_E88_COMPARISON.md` | `53816515`, `cbcc726c` | bound-partial (raw log off-repo) |
| C8 | FLOPs-per-bit convergence under CMA-ES (N=4) | §5 | `paper/results/cma_flop_rate/convergence.png`, `summary.csv`, `thresholds.csv`, `SOURCES.md`; `docs/CMA_FLOP_RATE_FINDING.md` | `0e007010`, `193ddaea` | bound (upstream data in `~/elman`) |
| C9 | 8M S5/S3 separation (E88 0.7918 vs baselines) | §6 | `paper/figures/s5_expressivity_seeds.png`, `plot_expressivity_seeds.py`; `paper/ndmpapernotes.md` | `da77796c`, `aea195db` | bound-partial (8M raw run JSON not committed) |
| C10 | Six-task canonical sweep; length extrapolation | §6 | `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`, `LENGTH_EXTRAP_RESULTS.md` | `cbcc726c` | bound-partial (raw run JSON not committed) |
| C11 | [E,E,G,G] AABB hybrid degrades state tracking | §6 | `paper/results/figure_4_hybrid/canon_hybrid_AABB__*.json`; `paper/figures/hybrid_degradation_seeds.png` | `b2b80e2d` | bound |
| C12 | 1.3B symmetric-budget S5: E88 > M2RNN > GDN at every length | §6 | `paper/review/s5_symmetric_data/{e88,gdn,m2rnn}.json`, `S5_SYMMETRIC_BUDGET.md`, `S5_SYMMETRIC_RESULTS.md`; `experiments/.../s5_symmetric_20260603/eval/summary.json` | `4b89e64e`, `1ba292ed` | bound |
| C13 | 1.3B to-competence S3/S5 + solvable controls | §6 | `paper/review/s3_s5_finetune_v03_data_tocomp/{e88,gdn,m2rnn}.json`, `S3_S5_FINETUNE_V03.md` | `56d7a2d3` | bound |
| C14 | QA/reasoning panel numbers at 1.3B | §6 | `paper/results/qa_reasoning/{racer_panel_300item_progression,reasoning_panel_latest,fact_panel_latest}.csv`, `SOURCES.md` | `43e7cda8` | bound |
| C15 | Lean update-family separation (sets C, C′, D) | §7 | `formal/lean/ElmanProofs/Architectures/{RecurrentResourceFormalism,MultiStepSeparation,M2RNNComparison,OnlineMemory}.lean`; `PaperCore.lean`; `PROOF_INVENTORY.md`; `LEAN_FRONTIER.md` | `f6dfe8ae`, `aea195db`, `cbcc726c`, `dd35ac12`, `8b5e9502` | bound |
| C16 | Lean S5 tracker + finite-state ceiling + realizes S5 (sets A, B) | §7 | `formal/lean/ElmanProofs/Expressivity/{S5Witness,S5Tracker,S5NDMRealization,NDMRealizesS5}.lean`; `PROOF_INVENTORY.md` | `cbcc726c`, `aea195db`, `8b5e9502` | bound |
| C17 | Triton kernel implementation + CUDA correctness tests | §1, §4 | `ndm/triton/e88_triton_{forward,backward,optimized}.py`; `tests/test_e88_triton{,_optimized}.py` | `3cf6439e`, `0631411b` | bound-partial (ROCm untested; see U2) |

## UNBOUND CLAIMS

These claims appear in the paper but have **no committed artifact** in this
worktree. Reported as gaps, not softened.

### U1 — Lean latching (Theorem set F) — **unbound**

`paper/main.typ` §7 "Theorem set F: latching" and the §1 Lean-spine
paragraph claim a trusted Lean proof of saturation insensitivity
(`EmenderLatching.emender_saturation_insensitivity`), sign-preserving hold
(`emender_latch_holds_by_default`), and counter-delta release
(`emender_latch_releases_on_counter_delta`,
`..._release_quantitative`).

**Gap:** No Lean module `ElmanProofs/Architectures/EmenderLatching.lean`
exists in the worktree, none of these theorems appear anywhere under
`formal/lean/ElmanProofs/`, the module is not in the `PaperCore` import
closure, and the path does not appear anywhere in this worktree's git
history (`git log --all --name-only`). The cited names occur only in
`paper/main.typ` and `paper/data/math-notation-spec.md`. The three latching
theorems the paper leans on are not backed by committed Lean source here.

### U2 — ROCm/CUDA parity — **unbound**

The abstract/introduction claim the Triton multi-programming kernel "runs
identically on CUDA and ROCm."

**Gap:** No ROCm/AMD test or benchmark is committed.
`paper/review/THROUGHPUT.md` measures an NVIDIA RTX 6000 Ada (CUDA) only.
No committed artifact demonstrates ROCm parity.

### U3 — "Four CMA-ES sweeps" replication — **unbound**

The §5 figure caption and the §Limitations "Evidence structure" paragraph
claim the within-class ordering is "replicated across four CMA-ES sweeps
(250+ configs/architecture)."

**Gap:** No committed per-sweep artifact enumerates four sweeps at 250+
configs/architecture. The aggregated dataset
(`paper/results/cmaes_burst_v2/`) covers 1,285 configurations total across
13 architecture-family panels, but is not decomposed into the claimed four
250+ sweeps. The claim is asserted only in the paper caption and in the
audit note `paper/review/rigor.md`.

### U4 — HuggingFace v0.3 release of checkpoints + configs + kernel — **unbound**

The abstract and the §10 availability statement claim the three 1.3B
checkpoints, the per-architecture CMA-ES configurations, and the Triton
kernel are released to the v0.3 HuggingFace target.

**Gap:** No checkpoint or release manifest is committed in this worktree;
the v0.3 weights and CMA-ES configs live off-repo (HuggingFace / `~/elman`).
The Appendix B readback (Emender 0.96613 / GDN 0.96612 / M2RNN-CMA 0.96132)
is documented, but the released artifacts themselves are not in-repo.

## Notes and caveats

- Commit hashes are the **most recent** commit touching each path, not
  necessarily the commit that first added it.
- Two Lean namespaces cited in the paper differ cosmetically from the actual
  source namespaces: the paper cites `EmenderRealizesS5.emender_realizes_s5_tracker`
  (actual namespace `NDMRealizesS5`) and `S5EmenderRealization.*` (actual
  namespace `S5NDMRealization`). The theorems exist; only the printed
  namespace labels differ. These are **not** gaps.
- The `bound-partial` weakness for C3, C7, C9, C10, C17: for C9/C10 the raw
  per-seed run JSON is not committed (only the summary Markdown and the
  rendered figures); for C7 the divergent M2RNN-paper log lives in `/tmp`;
  for C3/C17 the aggregate dataset/kernel is committed but the specific
  sub-measurement claimed (four-sweep decomposition / ROCm) is not.
- The paper source is Typst (`paper/main.typ`), not LaTeX.
