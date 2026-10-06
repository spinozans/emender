# Coherence Pass Notes — paper spine rebind to E97 / GDN2-MLP

Task: `paper-coherence-pass` (Task B, CPU only). Worktree branch
`paper-1p3b-spine`. This note lists what changed and why, in
claim → evidence → file form, per the task contract. Companion edits:
`paper/main.typ` (commits 8d313ea4, bfd6f91e, 08887cbf, 88109ce1,
c2e8381c) and `paper/OUTLINE.md` (commit b4eabb72).

## The spine (operator direction, verbatim binding)

Lean realizability → learned at 8 M → 1.3 B double-pairing (E97 vs GDN2,
CMA-ES-hardened baselines, matched tokens, same 2k-chunk Pile recipe)
with loss parity → expressivity suite at 1.3 B closing the
theory→empirics loop (slot for Task A).

## Claim → evidence → file

### §0/§1 — Viability and the pairing

| Claim (as now stated) | Evidence | File(s) |
|---|---|---|
| E97 (1,286,589,072 params) trains through 150,793,748,480 tokens on The Pile on one 8-GPU workstation; final last-100 loss 2.4365 nats/token; est. 0.93 bpb (bytes/token 3.783 from frozen args.json) | Final consensus checkpoint + sha + token/step clock | `docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md`; `docs/E97_LOSS_CURVE_BPB_REFRESH_20260630_0755.md` (bytes/token provenance); launcher `scripts/launch_emender_8gpu_diloco.sh` |
| GDN2-MLP (1,286,713,448 params; +0.0097%) trained under the identical recipe to 152,280,498,176 tokens; last-100 loss 2.4034 | Release validation receipt with step/tokens/sha | `docs/validation/gdn2-huggingface-152b-control-release.md`; launcher `scripts/launch_gdn2_mlp_8gpu_diloco.sh` |
| Both arms: same Pile file, 2,048-token chunks, p50k_base, bf16 schedule-free AdamW, per-rank batch 4, 8-island DiLoCo K=250 plain average (outer lr 1.0, beta 0), 65,536 tokens/optimizer step | Launch manifests (both launchers); token-per-step validated from args.json and log deltas | `scripts/launch_emender_8gpu_diloco.sh`; `scripts/launch_gdn2_mlp_8gpu_diloco.sh`; `docs/GDN2_ETA_20260724.md` |
| Loss parity at matched tokens: cutoff at E97's terminal authority, 92,037 aligned records, 0 single-arm records; raw 2.4770/2.4324, smoothed 2.4370/2.4267 (Δ −0.0103), last-100 mean Δ +0.0139 (sign flip), last-1000 Δ −0.0085; early (5.118 B) smoothed Δ −0.0713 | Matched-cutoff alignment table | `docs/GDN2_OPS_SNAPSHOT_20260728.md` (final); `docs/GDN2_E97_MATCHED_LOSS_COMPARISON_20260723.md` (early) |
| Pair released: HF repos + immutable tags + shas | `https://huggingface.co/spinozans/emender-e97-1.3b` tag `tokens-150793748480`; `https://huggingface.co/spinozans/gdn2-mlp-1.3b` tag `tokens-152280498176` | `docs/validation/e97-huggingface-dense-base-and-cli-agent-release.md`; `docs/validation/gdn2-huggingface-152b-control-release.md` |

### §5 — CMA-ES matched-control transparency (search breadth, budgets, configs)

| Claim | Evidence | File(s) |
|---|---|---|
| Search protocol: CMA-ES popsize 8, 15-min/candidate, ctx 2048, p50k, Pile, bf16 fused-only, 1,300 M ±3% target | Leaderboard methodology + reproduce recipe | `docs/E97_RAW_1P3B_LEADERBOARD.md`; `docs/REPRODUCE_E97_RAW_1P3B.md` |
| Breadth: 11 families, 64–128 evals each, best avg losses 5.9511 (e97-raw) … 6.4606 (transformer, searched under the same budget) | Committed leaderboard | `docs/E97_RAW_1P3B_LEADERBOARD.md` |
| 1,285 normalized candidate trajectories / 13 family panels / 192,147 step-loss rows behind the sweeps | Committed normalized corpus | `paper/results/cmaes_burst_v2/` (`cmaes_trajectory_points.csv.gz`, `cmaes_eval_summary.csv`, `NORMALIZATION.md`, `FIGURE_NOTE.md`) |
| Carried-in winners: E97-MLP (dim1792/nh216/ns32/dep11/mlp2.2623, lr 1.007e-3, fitness 5.8606) and GDN2-MLP (dim2176/nh30/dep12/exp1/ratio3.2587, lr 4.7431e-4, 104 evals/13 gens, fitness 5.8949, eval 86) | Committed comparison + per-search best.json | `experiments/lb_compare_20260613/{LEADERBOARD,REPRODUCTION}.md`; `docs/repro/lb_gdn2_mlp_20260612/best.json` |
| Disclosed asymmetry: earlier redo sweep had gdn2-mlp at 64 evals (< 96 target, provisional); the 150 B run launched from the 104-eval search | Both records | `docs/E97_RAW_1P3B_LEADERBOARD.md`; `docs/repro/lb_gdn2_mlp_20260612/best.json` |
| Convergent-loss null at search scale: 5 CMA-best 1.3 B models within a 0.088-bpb held-out band (2.0126–2.1013, non-avg, fixed 131,072-token Pile-tail slice); E97-MLP ahead of GDN2-MLP on both primary metrics (5.8606 vs 5.8949; 2.0911 vs 2.1013) but inside the noise band | Committed leaderboard with corrections | `experiments/lb_compare_20260613/LEADERBOARD.md` (read the CORRECTIONS header, not the stale auto-verdict); `docs/RESEARCH_ASTRAY_POSTMORTEM.md` addendum |
| FLOPs-per-bit convergence at 480 M, N=4 families (E1 1.3394, E88 1.2451, FLA-GDN 1.1403, Mamba2 1.1555 final nats at matched FLOPs; shared curve over >2 orders of magnitude) | Committed result package | `paper/results/cma_flop_rate/` (`README.md`, `summary.csv`, `thresholds.csv`, `SOURCES.md`) |

### §4/§5 — Throughput and memory at the pair geometry

| Claim | Evidence | File(s) |
|---|---|---|
| Matched-conditions per-GPU throughput tie at the literal pair geometry: 3,211 vs 3,290 tok/s (0.976×), memory 28.9 vs 35.7 GB; E97 fits bs6 (4,470) where GDN2 OOMs; small-scale 1.26–1.56× speedup NOT reproduced at 1.3 B | Preflight benchmark | `experiments/preflight_100b/RESULTS.md` |
| E88-era occupancy microbench (100% util, 7,492 tok/s, MFU 15.7%, ≈91% of FLA chunked-scan) retained, explicitly attributed to the predecessor instance | Prior paper measurements | `paper/main.typ` §4 (unchanged numbers, re-attributed) |
| CUDA-only production scope (ROCm portability not claimed) — U2 scoping | No ROCm production artifact exists for the pair | `paper/main.typ` §4 "Portable kernels" |

### §6 — Expressivity (8 M + iterated maps + 1.3 B slot)

| Claim | Evidence | File(s) |
|---|---|---|
| 8 M S5/S3 probes (0.79/0.36/0.22 at S5 T=128; 3 seeds), capacity non-binding | Unchanged from prior draft | `paper/ndmpapernotes.md` lines 153–173; `paper/review/` data; figures `paper/figures/s5_expressivity_seeds.png` |
| Iterated-map separation (modular_quadratic, T=128→4096): E97 length-invariant (0.982→0.981 at p=256/dim1024) where e97-linear (0.619) and gdn2 (0.560) collapse; 8-seed confirm gap +0.17…+0.32; two predicted-negative families (contractive map 0/24 grok; anbncn sign-flipping) | Committed sweep + confirmation | `experiments/grok_symmetric_width/RESULTS.md`; `experiments/grok_symmetric_width/CONFIRM.md` |
| e97 vs e97-linear differ only by tanh vs identity on the same fused kernel (causal isolation) | Kernel source + results | `experiments/grok_symmetric_width/RESULTS.md`; `ndm/models/e88_fla_hybrid.py` |
| Downstream panels null (predecessor cohort, labeled as such): fact 0.345/0.373/0.358/0.353, reasoning 0.319/0.350/0.336/0.324 (E88/FLA-GDN/M²RNN/Mamba2) | Committed CSVs | `paper/results/qa_reasoning/fact_panel_latest.csv`; `paper/results/qa_reasoning/reasoning_panel_latest.csv` |
| 1.3 B expressivity on the E97/GDN2-MLP pair: **slot** — protocol fixed (zero-shot first; matched fine-tune per the 8 M suite if inconclusive), results land in `paper/results/expressivity_1p3b/`, negatives reported verbatim | Task A in flight | `paper/main.typ` §6 "Expressivity at 1.3 B on the trained pair (slot)" with TODO-EXPRESSIVITY-1P3B markers |

### §7 — Formal (wording fixed to the Lean artifact)

| Claim | Evidence | File(s) |
|---|---|---|
| Trusted surface: PaperCore import closure, 12 modules, no sorry/axiom/opaque/native_decide; machine-extracted inventory 667 entries at extraction commit | Committed extraction JSON + build | `paper/data/lean-trust-gate-theorems.json`; `formal/lean/PROOF_INVENTORY.md` |
| S5 capacity floor: `S5Inseparability.capacity_lower_bound_120`, `small_recognizer_cannot_realize_s5`, `exists_distinguishing_word_pair`, generator surjectivity `run_surjective` | In extraction JSON | `paper/data/lean-trust-gate-theorems.json`; `formal/lean/ElmanProofs/Architectures/S5Inseparability.lean` |
| Realization: `NDMRealizesS5.emender_realizes_s5_tracker` (namespace fixed from the draft's wrong `EmenderRealizesS5.*`); lookup-table counts `S5NDMRealization.*` (fixed from `S5EmenderRealization.*`) | In extraction JSON | same |
| k-step separation set C′, one-step set C, FLOP-class set D, multi-programming set E — all verified present by name | In extraction JSON | same |
| New set G (split-gated relations): `e97_specializes_to_e88_all_one_gates_*`, `e97_realizes_splitWitness_transition`, `e88_cannot_realize_splitWitness_transition`, `e97_and_gdn2_share_split_gated_linear_core`, `gdn2_split_erase_write_is_scan_compatible_linear_state`, `e97_split_gated_resource_signature_like_e88`, E97/E88 cost-class theorems | Committed Lean files, post-snapshot (SplitGatedDelta + RecurrentResourceFormalism E97 rows) | `formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean`; `formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean` |
| Latching set F: `EmenderLatching.emender_saturation_insensitivity`, `emender_saturation_two_inputs_close`, `emender_latch_holds_by_default`, `emender_latch_releases_on_counter_delta`, `emender_latch_release_quantitative` | Recovered proofs, commit `696299b4` (from wg/agent-281 `6d68e5b0`), `lake build` clean in both trees (U1 closed) | commit `696299b4`: `formal/lean/ElmanProofs/Architectures/EmenderLatching.lean` |
| NC1/TC0 wording: realization + 120-state floor Lean-proved; Barrington completeness and the linear-state ceiling cited, not formalized; TC0 membership statements cited; bounded-precision raw-write T(d) transfer flagged as conjecture | Audit + inventory + JSON | `formal/lean/FORMALIZATION_GAP_AUDIT.md`; `formal/lean/PROOF_INVENTORY.md` (Claimed-but-Not-Formalized list); `formal/lean/TC0_CLAIMS_STATUS.md` |

### §9 — Limitations (new/changed)

- **Named scale limitation** ("Scale: fine at 1.3 B — does it survive at 7 B+?"): the
  pairing replicates across update rule and tuning regime, not across scale;
  the single-arm E97 513 B Frontier continuation cannot answer a paired
  question; the matched 3–7 B pair is the named follow-up
  (`docs/validation/e97-35b-moe-150b-job5208321.md` is the continuation's
  provenance; explicitly out of the paired claim).
- **Transformer baseline**: one deferral sentence, no apology
  (per operator direction).
- **Training-log basis of the parity** and pending held-out bpb for the
  150 B pair: TODO-NEEDS-EVIDENCE in §5 and §9.

## What moved (nothing deleted)

| Old location | New location |
|---|---|
| §5 predecessor cohort (E88/M²RNN-CMA/GDN racers, per-architecture CMA protocol incl. H=370 narrative, M²RNN gradient-conditioning stability control, held-out bpb tie, Common-Pile control, matched-compute tie) | Appendix B "The predecessor 1.3 B cohort (E88 era)" (`<sec:appendix_cohort>`), text unchanged except heading/self-references |
| §6 predecessor 1.3 B fine-tune study (symmetric-budget S5 tables/figures, to-competence cross-check, micro/mega prose, QA/reasoning panel prose) | Same Appendix B, retitled subsection |
| Held-out bpb methodology appendix | Kept, retitled "(predecessor cohort)", intro re-pointed at Appendix B |
| Lineage appendix A | Extended E63→E88→E97 with a new E97 stage bullet |

## Open TODOs left in the paper (deliberate, marked, not softened)

1. **TODO-EXPRESSIVITY-1P3B** (§6): the 1.3 B expressivity slot on the
   E97/GDN2-MLP pair — Task A's deliverable.
2. **TODO-NEEDS-EVIDENCE** (§5, §9): held-out, tokenizer-invariant bpb for
   the 150 B pair (predecessor-cohort pipeline exists and is cited);
   downstream capability panels for the pair.
3. **TODO-NEEDS-EVIDENCE** (§7): refresh `lean-trust-gate-theorems.json`
   extraction to cover the two post-snapshot modules (SplitGatedDelta,
   EmenderLatching at `696299b4`).
4. §9/§12 name the broader-search GDN2 re-run (removing the 64-eval
   provisional standing of the earlier redo sweep) and the matched 3–7 B
   scale-up as the load-bearing follow-ups.

## Deliberate non-changes

- The commented-out ALTERNATE ABSTRACTS block at the top of `main.typ`
  still carries E88-era text; it is non-rendered A/B reference material
  and was left for history.
- The 8 M probe suite keeps its original "GDN (FLA-Gated DeltaNet)"
  baseline labels and numbers; §6 now discloses that the probes used the
  E88-lineage default configuration of the Emender (the E97 instance
  postdates them), and the 1.3 B slot tests the current instance.
- Per-sweep CMA-ES decomposition claims (U3) were not added; §5 reports
  per-family eval counts and trajectories instead.
