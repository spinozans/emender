# lb-compare — final apples-to-apples comparison of the 4 (+1) CMA-best 1.3B models

**Task:** lb-compare. Final apples-to-apples comparison of the CMA-best models
at THEIR OWN found geometries: held-out BPB on one shared disjoint slice, plus
formal-separator length-extrapolation at each found cell.

## Models (each at its OWN CMA-found 1.3B geometry)

| Arm | Cell / level | Geometry | Search avg-loss | Source task |
|---|---|---|---:|---|
| pure-E97 | E97 split-edit, raw-write (no MLP) | dim2432 nh416 ns16 dep10 lr9.85e-4 bs3 | 5.9511 | e97-raw 1.3B leaderboard |
| Emender-mix | typed-gdn2-lm mixture, f=0.971 (~97% e97_delta + 3% gdn2_recall) | dim2432 nh212 ns32 dep10 lr1.144e-3 bs2 | 6.0756 | lb-emender-mix |
| gdn2-mlp | GDN-2 mixer + SwiGLU MLP | dim2176 nh30 dep12 mlp3.259 lr4.74e-4 bs4 | 5.8949 | lb-gdn2-mlp |
| m2rnn | M2RNN matrix-to-matrix RNN (XMA fused) | dim3072 nh346 ns16 dep13 lr1.04e-3 bs4 | 6.0636 | lb-m2rnn2 |
| emender-mlp | E97 split-edit **DELTA** (e88_raw_write=0 — delta-correcting, NOT raw-write; verified across all 520 eval args) + SwiGLU MLP (fair MLP counterpart of gdn2-mlp) | dim1792 nh216 ns32 dep11 mlp2.26 lr1.007e-3 bs4 | 5.8606 | lb-emender-mlp |

Param counts verified byte-for-byte against each source's recorded actual_params.

## Protocol (identical for every model)
- **Construction:** reused `scripts/cmaes_search_v2.build_train_command` → `train.py`
  so each model is built BYTE-IDENTICALLY to its CMA search (same level, kwargs,
  fused kernels).
- **Training:** pile.txt (`/home/erikg/elman/data/pile.txt`) seed42, p50k_base,
  ctx/chunk 2048, schedule-free AdamW, bf16 uniform, fused kernels asserted
  (E97 Triton split-edit / m2rnn XMA / gdn2 external). 15-minute train budget
  per model — the SAME budget the CMA search used per candidate.
- **Held-out:** ONE fixed disjoint slice from the FAR TAIL of pile.txt (offsets
  ≥ 90% of file), 64 chunks × 2048 = 131072 scored tokens, p50k_base, saved to a
  tensor and scored byte-for-byte identically by every model on the schedule-free
  AVERAGED weights. BPB = (CE_nats/ln2) / 3.878 bytes-per-token.
- **Formal separators:** `train_hybrid.py` on `anbncn_viability` (a^n b^n c^n),
  `dyck_depth_unbounded` (unbounded Dyck depth), `modular_counter`; trained T=128,
  evaluated T∈{128,256,512,1024} (Délétang length-extrapolation). Each arm keeps
  its FOUND cell + head-composition + n_state; dim=512/depth=4 fixed across arms =
  capacity/width control (literal 1.3B width is infeasible for 10k-step synthetic
  tasks and confounds capability with parameter count). 2 seeds.

## Artifacts
- `build_heldout_tensor.py` — extracts the fixed disjoint held-out slice.
- `run_bpb.py` — held-out BPB driver (reuses the CMA construction path).
- `run_separators.py` — formal-separator length-extrap driver.
- `orchestrate.sh` — single GPU-lease-owning orchestrator (no cross-agent contention).
- `aggregate.py` → `LEADERBOARD.md` — unified tables + ranking.
- `bpb_results.json`, `sep_results.json` — raw measured results.
- train.py patch: `--heldout_tensor` (additive, off by default).

## Results & verdict
See `LEADERBOARD.md` (generated from the raw JSONs) and the VERDICT section
appended after the run completes.

## CORRECTIONS (post-review — the verdict below overstates against the Emender)

1. **`emender-mlp` is E97-DELTA (split-edit, delta-correcting), NOT raw-write.** All 520 eval
   args show `e88_raw_write=0`. The "raw-write" labels in this doc for emender-mlp are wrong.
   (Only `pure-E97` is the raw-write variant.) So the capability-retaining delta cell — not the
   recall-sacrificing raw one — is the arm that beat gdn2-mlp.
2. **On the PRIMARY metrics, emender-mlp LEADS gdn2-mlp**, it does not "lose" or merely "tie":
   search avg-loss 5.8606 < 5.8949, and non-avg held-out 2.091 < 2.101. gdn2-mlp wins *only* on
   the schedule-free *averaged* basis, which this very run flags as the inferior/artifact basis.
   The fair MLP-vs-MLP loss fight **leans Emender** (within the 0.088 noise band).
3. **The separators are the GROK-SUPPRESSED battery** (LR pinned 3e-4, no weight-decay sweep,
   short schedule-free training, dim512/dep4, 2 seeds) — the unreliable capability metric. And
   **`modular_counter` is bounded/finite-state counting, where linear-state is *expected* to
   win** — it is NOT the Emender's nonlinear-in-time (unbounded-counting / step-growth) claim.
   So "loses the discriminating separator" neither tests nor refutes that claim.
4. **Therefore "clean NO-GO / gdn2-mlp best all-around" is not supported.** Honest status: LM
   loss = tie with the Emender on the good side; capability = UNDETERMINED pending a proper grok
   test (AdamW + wd-sweep + train-to-grok on unbounded separators, with a GDN-2 width control).

## VERDICT (summary — see LEADERBOARD.md for full tables) — SUPERSEDED BY CORRECTIONS ABOVE

**LM held-out BPB: TIE — the Emender does NOT win.**
Non-averaged held-out (basis consistent with the CMA search avg-loss) puts all 5 within a
0.088-BPB band (pure-E97 2.013 · m2rnn 2.034 · Emender-mix 2.040 · emender-mlp 2.091 ·
gdn2-mlp 2.101) = a statistical tie at single-seed/15-min budget. Averaged (schedule-free
eval) weights rank gdn2-mlp 2.155 < emender-mlp 2.178 < pure-E97 2.283 < m2rnn 2.300 <
Emender-mix 2.738, but the averaged weights are uniformly worse than the final weights here,
by an architecture-dependent margin (MLP cells Δ≈0.05–0.09; mixer/split-edit Δ≈0.27–0.70,
worst for Emender-mix) — an averaging artifact, not a capability gap.

**Formal separators: the Emender does NOT win.**
- anbncn_viability: TIE (all solve, comparable extrap).
- dyck_depth_unbounded: TIE (all solve T=128, all collapse at length).
- modular_counter (discriminating): gdn2-mlp solves (0.97/1.00), Emender-mix partial (0.86),
  m2rnn/emender-mlp seed-variable, **pure-E97 robustly FAILS (0.25 ≈ baseline).** The counting
  capability is the GDN-2 recall cell's, not the E97 nonlinear-in-time heads'. Capacity ruled
  out (emender-mlp 14.5M ≈ gdn2-mlp 15.5M but 0.48 vs 0.98; m2rnn 2.7M beats emender-mlp 14.5M).

**Overall: clean NO-GO for the Emender at 1.3B — ties on LM bpb, loses the discriminating
separator to gdn2-mlp. gdn2-mlp is the best all-around. Upholds the convergent-loss / capability
null from emender-real-1p3b, emender-cap-sweep, opt-1p3b, lb-emender-mix.**

### Methodological finding (reusable)
At a short (15-min) train budget, schedule-free AdamW's polyak-AVERAGED eval weights are
*worse* than the final weights, by an architecture-dependent margin — large enough (Δ up to
0.70 BPB) to flip a 1.3B held-out leaderboard. Any "leaderboard methodology = averaged weights"
comparison at short budget must report both, or it will mis-rank mixer-only / split-edit cells.
