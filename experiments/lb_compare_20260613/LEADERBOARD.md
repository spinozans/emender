# lb-compare — apples-to-apples leaderboard (REAL measured data)

All 5 CMA-best models at THEIR OWN found 1.3B geometry. SAME protocol: pile.txt seed42 train (15-min budget, matching the CMA search), bf16 uniform + fused kernels (E97 split-edit Triton / m2rnn XMA / gdn2 external), p50k_base, ctx 2048, schedule-free AdamW. Held-out = ONE fixed disjoint pile.txt-tail slice (64 chunks / 131072 scored tokens, byte-for-byte identical for every model). Held-out BPB = (CE_nats/ln2)/3.878 bytes/token.

## 1. Unified table — search avg-loss vs held-out (same slice)

Held-out reported in BOTH weight modes: **non-avg** = the final/training weights (same basis as the CMA search avg-loss, which is a non-averaged training-trajectory mean); **avg** = schedule-free polyak-averaged eval weights (the 'leaderboard methodology'). At this 15-min budget the averaged weights are uniformly worse than the final weights, by an architecture-dependent margin (see verdict).

| Model | Params (M) | Search avg-loss | train-loss(last100) | Held CE (nonavg) | Held **BPB (nonavg)** | Held CE (avg) | Held BPB (avg) | steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure-E97 | 1265.6 | 5.9511 | 5.7437 | 5.4102 | **2.0126** | 6.1367 | 2.2829 | 1197 |
| Emender-mix | 1273.2 | 6.0756 | 5.6120 | 5.4844 | **2.0402** | 7.3594 | 2.7377 | 1789 |
| gdn2-mlp | 1286.7 | 5.8949 | 5.8751 | 5.6484 | **2.1013** | 5.7930 | 2.1550 | 861 |
| m2rnn | 1275.0 | 6.0636 | 6.0843 | 5.4688 | **2.0344** | 6.1836 | 2.3003 | 999 |
| emender-mlp | 1286.6 | 5.8606 | 6.2945 | 5.6211 | **2.0911** | 5.8555 | 2.1783 | 892 |

**Held-out BPB ranking — NON-AVG (primary; lower=better):** pure-E97 2.0126 < m2rnn 2.0344 < Emender-mix 2.0402 < emender-mlp 2.0911 < gdn2-mlp 2.1013

**Held-out BPB ranking — AVG (schedule-free eval):** gdn2-mlp 2.1550 < emender-mlp 2.1783 < pure-E97 2.2829 < m2rnn 2.3003 < Emender-mix 2.7377

## 2. Formal separators — length-extrapolation accuracy (train T=128)

Matched capacity (dim=512, depth=4) across all arms = capacity/width control; each arm keeps its FOUND cell + head-composition + n_state. Accuracy averaged over seeds. Random baseline noted per task.

### anbncn_viability  (random baseline ≈ 0.500)

| Model | params(M) | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|---:|
| pure-E97 | 3.7 | 1.000 | 0.936 | 0.872 | 0.824 |
| Emender-mix | 7.4 | 1.000 | 0.932 | 0.872 | 0.825 |
| gdn2-mlp | 15.5 | 1.000 | 0.950 | 0.875 | 0.818 |
| m2rnn | 2.7 | 1.000 | 0.948 | 0.896 | 0.840 |
| emender-mlp | 14.5 | 0.998 | 0.922 | 0.842 | 0.780 |

### dyck_depth_unbounded  (random baseline ≈ 0.004)

| Model | params(M) | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|---:|
| pure-E97 | 3.8 | 0.993 | 0.761 | 0.397 | 0.207 |
| Emender-mix | 7.5 | 0.999 | 0.879 | 0.475 | 0.246 |
| gdn2-mlp | 15.6 | 0.999 | 0.863 | 0.472 | 0.247 |
| m2rnn | 2.8 | 0.993 | 0.755 | 0.397 | 0.207 |
| emender-mlp | 14.6 | 0.999 | 0.840 | 0.446 | 0.232 |

### modular_counter  (random baseline ≈ 0.200)

| Model | params(M) | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|---:|
| pure-E97 | 3.7 | 0.251 | 0.225 | 0.212 | 0.206 |
| Emender-mix | 7.4 | 0.859 | 0.601 | 0.400 | 0.300 |
| gdn2-mlp | 15.5 | 0.979 | 0.872 | 0.577 | 0.400 |
| m2rnn | 2.7 | 0.595 | 0.435 | 0.317 | 0.261 |
| emender-mlp | 14.5 | 0.480 | 0.349 | 0.274 | 0.237 |

## 3. Honest verdict

**Does the Emender win, tie, or lose — on bpb AND on the formal separators?**

### LM held-out BPB → TIE (Emender does NOT win)
- On the **non-averaged** held-out (the basis consistent with the CMA search avg-loss),
  all 5 models land in a **tight 0.088-BPB band** (2.013–2.101). pure-E97 (2.013),
  m2rnn (2.034), Emender-mix (2.040) are nominally ahead of emender-mlp (2.091) and
  gdn2-mlp (2.101), but this ordering is **within single-seed / 15-minute-budget noise**
  and does not match the search avg-loss ordering (which put the MLP models first). The
  honest read is a **statistical tie across all five architectures**.
- On the **averaged** (schedule-free eval) held-out, gdn2-mlp (2.155) and emender-mlp
  (2.178) lead and Emender-mix is last (2.738). **This is an averaging artifact, not a
  capability gap:** at the 15-min budget the schedule-free polyak average is uniformly
  worse than the final weights, and the penalty is architecture-dependent — small for the
  MLP cells (Δ≈0.05–0.09 BPB) and large for the mixer-only / split-edit cells
  (Δ≈0.27 for pure-E97/m2rnn, **Δ≈0.70 for Emender-mix**). Reporting averaged-only would
  falsely brand Emender-mix a big LM loser; its actual learned model (non-avg 2.040) is
  squarely competitive.
- **Verdict (LM bpb): the Emender variants TIE the GDN-2 / m2rnn baselines. No win.**
  This upholds and extends the convergent-loss null established across the prior emender
  tasks (emender-real-1p3b NO-GO, emender-cap-sweep NULL, opt-1p3b null, lb-emender-mix
  near-pure-E97).

### Formal separators → Emender does NOT win; the counting capability is GDN-2's, not E97's
Matched capacity (dim=512, depth=4; per-arm params 2.7–15.6 M reported as the width
control), train T=128, eval T∈{128…1024}, 2 seeds, LR matched at 3e-4 for every arm.
- **anbncn_viability (a^n b^n c^n):** all 5 solve in-distribution (acc≈1.0) and extrapolate
  comparably (T=1024 ≈ 0.78–0.84, m2rnn best). **TIE.** Note gdn2-mlp/emender-mlp have
  2–4× more params yet do no better → not capacity-bound.
- **dyck_depth_unbounded (unbounded Dyck depth, Weiss–Goldberg–Yahav):** all solve T=128
  (≈0.99) and all **collapse toward baseline at length** (T=1024 ≈ 0.21–0.25; Emender-mix
  0.246 and gdn2-mlp 0.247 marginally best, within noise). No architecture truly
  extrapolates unbounded counting. **TIE (all weak).**
- **modular_counter (the discriminating task):** robust contrasts across seeds —
  **gdn2-mlp solves it (0.97/1.00 @T128, 0.40 @T1024)**, Emender-mix partial (0.86/0.86),
  m2rnn and emender-mlp seed-variable (0.39–0.82 / 0.29–0.67), and **pure-E97 robustly
  FAILS (0.25/0.25 @T128 ≈ random baseline 0.20).** Capacity is ruled out: emender-mlp
  (14.5 M) ≈ gdn2-mlp's capacity (15.5 M) but scores 0.48 vs 0.98, and m2rnn (2.7 M) beats
  emender-mlp (14.5 M). **The counting capability lives in the GDN-2 recall / linear-state
  cell, not the E97 nonlinear-in-time split-edit heads** — adding 3% gdn2_recall lifts pure-
  E97 from 0.25 (fail) to Emender-mix's 0.86. This directly refutes the premise that the
  Emender's nonlinear-in-time heads buy a counting/capability advantage; consistent with
  "linear wins on finite-state / modular counting" (linvsnonlin-separator-is-counting).

### Overall
**The Emender (typed nonlinear-in-time mixture) neither wins LM held-out BPB nor wins the
formal separators against the GDN-2 baselines at matched conditions — it ties on LM and
loses the one discriminating separator (modular_counter) to gdn2-mlp.** The pure-E97 cell
outright fails modular counting. **gdn2-mlp is the best all-around model**: tied-best LM
bpb (non-avg), best LM bpb (avg), and decisive winner on modular_counter. This is a clean
NO-GO for the Emender as an LM-or-capability improvement over GDN-2 at 1.3B.

### Caveats (honesty)
- Single training seed for the LM held-out arms, 15-min budget → the LM band (0.088 BPB)
  is within noise; the "tie" is the robust claim, not the fine ordering.
- Separator LR fixed at 3e-4 (canonical expressivity LR), NOT each model's found LM LR;
  matched across arms but not per-arm tuned. The modular_counter gaps (0.25 vs 0.98) are
  far larger than plausible LR sensitivity, and pure-E97's failure is seed-robust.
- modular_counter learnability is seed-sensitive for m2rnn / emender-mlp (2 seeds); the
  gdn2-mlp-solves / pure-E97-fails / Emender-mix-partial contrasts are seed-robust.
- Separators run at matched capacity (dim512/depth4), NOT the literal 1.3B width: capability
  is an architectural property of the cell, and the 1.3B width is infeasible for 10k-step
  synthetic tasks and would confound capability with parameter count.
