# Seed-Maturity Threshold for SAFE Horizontal DiLoCo Scaleout
### "Where do we seed from before going horizontal on Frontier?"

**Task:** `seed-maturity-threshold` (Architect / design document with rationale).
**Author:** agent-1957. **Date:** 2026-06-20.
**Status:** **COMPLETE.** Phase 1 (CPU existing-data analysis + LMC design + the
threshold rule, §1–§3/§5–§8) **and Phase 2 (the empirical in-basin-vs-blow-up
matrix, §4)** are both delivered with **no GPU** — Phase 2 is harvested from the
per-merge loss trajectories the production DiLoCo runs already wrote to disk
(`harvest_merge_continuity.py` → §4.2 matrix across maturity {0, 528 M, 1.233 B,
1.966 B} × islands {2, 4, 6} × β {0, 0.9}). Both phases passed an **independent
4-lens adversarial audit** (§10). The only piece still GPU-gated is the *strict
held-out centroid barrier* on the *larger-`K` drift axis* (`lmc_probe.py`, armed
and non-blocking, §4.4) — a refinement that sharpens the SGD-stability knee but
does not gate the §5 rule, which is now decided on measured data.

---

## 0. The question, and the answer in one line

**Question.** What is the MINIMUM single-model training maturity (loss / BPB /
tokens) at which horizontal DiLoCo averaging is SAFE — replicas stay in the SAME
loss basin so the periodic weight-average is benign or beneficial (SWA) — versus
the under-trained regime where data-diverged replicas land in DIFFERENT basins
and the average sits on a loss barrier (blow-up)?

**Answer (headline).** At the **validated Frontier operating point** — plain
weight-average (outer `β=0`), merge interval `K=250` local steps, island count
`I ≤ 4` — the **safe-seed maturity floor is 0 tokens (random-init), and this is
now DIRECTLY MEASURED**, not inferred. The full maturity × island × β matrix is
read from real on-disk DiLoCo merge trajectories (§4): a **from-scratch I=4 run
stays in-basin through 91 merges** (converges loss 9.14 → 2.91, min 2.71, no
compounding; the small per-merge train-loss barrier is window-dependent, ~+0.09 at
W=4, §4.3), and any real seed (528 M, 1.233 B) **keeps descending with no
divergence** through up to **358 merges** (held-out −0.10 … −0.13 BPB — *within* the
±0.17 reference-noise band, i.e. non-harmful, not a calibrated SWA gain). The
factors that decide safety in this experiment are **NOT seed maturity** but:

1. **the outer optimizer** — plain average `β=0` is in-basin at every maturity; the
   **clean β-isolated control** (`swell_i4`: β=0 vs β=0.9, same 528 M seed/I=4/lr)
   shows `β=0.9` **BLOWS UP** (loss → 51; held-out +1.8 → +35 BPB); β=0.9 also
   diverges from the *most mature* 1.966 B seed at I=6 (+0.46/merge sustained — but
   that cell is **confounded** with `outer_lr=0.5`/islands, §4.2), so maturity does
   **not** rescue β=0.9: it is an outer-optimizer overshoot, not a basin-mismatch; and
2. **(carried, not re-measured here) the drift per merge window** — LMC theory and
   prior `diloco-periodic` work say the barrier grows with how far replicas wander
   between merges (`K`), but **every cell in this matrix is `K=250`**, so `K` is an
   asserted gate, not one demonstrated by these data (the larger-`K` axis is the
   GPU-gated refinement, §4.4).

A genuine **maturity threshold could only re-appear in the stressed regime** (much
larger `K` / many more islands), where Frankle's SGD-stability theory predicts an
early seed tolerates *less* drift before the merge crosses a basin than a mature
seed does. That drift-induced threshold is **the one open axis** — the measured
matrix is confined to **K=250, I≤4 (β=0) / I≤6 (β=0.9), maturity 0…1.966 B**, and
the strict held-out *centroid* barrier on the larger-`K` drift axis is the
GPU-gated refinement (§4.4). Within the measured envelope the **operating-point
rule** (§5) is decided: **you do not need to bank an expensive seed to scale out at
`K=250, β=0, I≤4` — random-init is already in-basin; the merge cadence and the
outer-optimizer choice are the gate, not the seed.**

This is a **GO-leaning, non-fatal** result reported under strict NULL discipline:
the experiment looked hard for the "too-early seed blows up" red flag at the
operating point and **did not find it**; it relocates the real risk to two knobs
that *are* under our control.

---

## 1. Theory hook — Linear Mode Connectivity / the SGD-stability point

DiLoCo's outer step with `outer_lr=1, β=0` is exactly the **average of the
replica weights**:

```
W_consensus  =  (1/I) Σ_i  W_i          (plain local-SGD average)
```

The relevant safety quantity is the **centroid instability gap** (the LMC
"instability analysis" of Frankle, Dziugaite, Roy & Carlin, *Linear Mode
Connectivity and the Lottery Ticket Hypothesis*, 2020):

```
gap  =  L(W_consensus)  −  mean_i L(W_i)
```

This is the gap *at the centroid* — a lower bound on Frankle's sup-over-path
barrier, and the **operationally relevant** quantity here because the `β=0`
DiLoCo merge lands *exactly* at the centroid. We call it "barrier" below as
shorthand for this centroid gap.

- **`barrier ≤ 0`** → the children are linearly mode-connected: the line (and the
  centroid) between them has no loss barrier, so the average is *as good as or
  better than* the children. Averaging is benign — the SWA (stochastic weight
  averaging) regime, where the centroid is a flatter, better minimum.
- **`barrier ≫ 0`** → the children sit in *different* basins; the centroid lands
  on a ridge between them. Averaging destroys the model (the DiLoCo "blow-up").

**Frankle's empirical law.** Take a shared parent at training time `t`, fork `I`
children, train each on different SGD/data noise. There is an **early
"SGD-stability" point** `t*`: forking *after* `t*` gives `barrier ≈ 0` (stable —
the children stay mode-connected); forking *before* `t*` gives `barrier > 0`
(unstable — they diverge into different basins). The barrier **worsens with
(a) more children `I`** (the centroid is further from each child) **and (b) more
independent SGD between forks** — in DiLoCo that "amount of SGD" is the **merge
interval `K`** (drift from the shared anchor is *monotone in `K`*, which is the
axis the probe sweeps; `K·lr` is a heuristic ordering, not a literal distance —
it drops ‖g‖ and the √K diffusion travel).

**Mapping onto DiLoCo.** A K-step DiLoCo run is a *repeated* instability test:
every `K` steps the replicas have drifted (monotonically in `K`) from the shared
anchor and we average them. So the safety condition is

```
barrier(maturity m, merge interval K, islands I)  ≤  0 .
```

The **seed-maturity threshold** is the maturity `m*` above which this holds for
the `(K, I)` we intend to run. Theory predicts `m*` is *small* when `K` is small
(little drift to cross a basin) and grows as `K` and `I` grow. The existing
record (§2) confirms the small-`K` corner; the probe (§3) measures how `m*` moves
as drift grows.

---

## 2. What the EXISTING data already implies (no-GPU analysis)

Three prior REAL campaigns on this exact emender-mlp 1.286 B model
(dim1792/nh216/ns32/dep11, fused split-edit Triton, schedule-free, bf16,
commapile/pile ctx2048) bear directly on the threshold. All used **plain average
`β=0`** and **`K=250`** unless noted, and all scored **offline, fused, y-mode** at
**matched total tokens**. Degradation `= consensus_BPB − single-GPU_BPB` at
matched tokens; **negative ⇒ DiLoCo better ⇒ SWA benefit**. ⚠ **Held-out tensor
provenance differs by row** (rows 1–3 = clean disjoint p50k slice `07005c39`;
row 4 = pile-tail `8e1198ab`; row 5 = commapile `cb0a478c`) — each *degradation/
barrier is differential on its own fixed tensor and so tensor-robust*, but
**absolute BPB magnitudes are NOT comparable across rows**.

| # | source | seed maturity | islands `I` | `K`, `β` | merges | basin verdict | matched-token degradation |
|---|---|---|---|---|---|---|---|
| 1 | `diloco-scaling-law` | **scratch (random init), scored @176 M tok** | 4 | 250, 0 | (rank-0 proxy) | parity (no barrier) | **+0.009 BPB** (within ±0.12 noise) |
| 2 | `diloco-scaling-law` | 528 M | 2 | 250, 0 | 22 | in-basin (SWA) | **−0.115 … −0.135** |
| 3 | `diloco-scaling-law` | 528 M | 4 | 250, 0 | 4 | in-basin (SWA) | **−0.075 … −0.129** (monotone-narrowing) |
| 4 | `diloco-seed-race-i4` | **1.233 B** | 4 | 250, 0 | **358** | in-basin (SWA) | **−0.082 … −0.131** (7/7 pts, sign-p=0.0156); → 1.052 @ 4.17 B |
| 5 | `diloco-longhorizon` | **random init (seed 42), NOT a seed** | **7** | 250, 0 | many | **no merge blow-up**, but held-out BPB **COLLAPSES** (recipe, not basin) | **NOT an SWA point** — held-out 2.03→3.19 @215 M for **both** DDP & DiLoCo (constant-LR roll-over, memory `fix-long-horizon`) |
| 6 | `diloco-scaling-law` | 528 M | 4 | 250, **0.9** | 4 | **BLOW-UP** | **+1.85 → +35.13** (train loss 6.08→8.81→14.75→19.57) |

**Row 5 caveat (adversarial-audit fix).** The only `I=7` run is **random-init**
(arm B `dil_b0_long`, `--seed 42`, *no `--resume`* — "1.286 B" is the *parameter*
count, not a seed maturity), on commapile/bs6 with a third held-out tensor. Its
held-out BPB **collapses** past ~64 M tokens — but so does the **single-GPU DDP
baseline** (constant LR, `warmup_steps=0`, no decay), so this is a **training-
recipe roll-over, not a merge/basin blow-up** (no NaN, no per-merge spike). So
row 5 is a "**`I=7` β=0 DiLoCo runs through many merges without a merge-induced
blow-up**" datapoint **only** — it is **not** an SWA/in-basin benefit and **not**
from a mature seed. There is **no measured `I>4` run from a mature seed.**

### The maturity ladder — tokens vs loss vs held-out BPB (CPU, from disk)

The "maturity" axis is best read in **tokens**, because both the train loss and
the held-out BPB are **non-monotone** under the constant-LR schedule-free recipe
(the known long-horizon roll-over, memory `fix-long-horizon`) — neither loss is a
clean maturity proxy. The single-GPU reference ladder on disk
(`…/ref_emender_mlp/runs/levelE97_100m_20260615_211750/`, 8192 tok/step):

| step | tokens | train loss (filename) | held-out BPB (scaling-law / seed-race) |
|---:|---:|---:|---:|
| 21 500 | 176 M | 3.778 | 1.436 |
| 43 000 | 352 M | 3.613 | 1.301 |
| 64 500 | 528 M | 3.125 (best train in {21.5k…107.5k}) | **1.385** (held-out *bump*) |
| 86 000 | 705 M | 3.246 | 1.264 |
| 107 500 | 881 M | 2.997 | 1.301 |
| 129 000 | 1.057 B | 3.144 | — |
| 150 500 | 1.233 B | 3.044 | ~1.23 (seed-race seed) |
| 193 500 | 1.585 B | 3.169 | — |
| 236 500 | 1.937 B | 3.078 | 1.170 |
| 244 141 | 2.000 B | 3.117 | 1.173 |

Note the **train-vs-held-out divergence at 528 M** (best train loss 3.125, yet a
held-out *bump* 1.385) — a concrete reason the threshold must be stated in
**tokens**, and a reason the SWA averaging *helps*: the consensus escapes that
constant-LR bump (it is below the single-GPU envelope precisely where the seed
sits on a bump).

### What the table says about the threshold

- **Maturity axis (rows 1–4):** across the held-out maturity ladder —
  **random-init-scored-@176 M → 528 M → 1.233 B** — averaging is **in-basin at
  every measured point**. The lowest cell (from-scratch I=4, scored at 176 M
  tokens) reaches single-GPU parity (`+0.009`, within noise), and *any* real seed
  keeps descending with no divergence. **In this §2 held-out view this looked like
  "no floor down to 176 M," and 0-token safety was an LMC inference.** ⚠ that
  held-out point is a **rank-0 replica proxy** (step 21600 ∉ 250ℤ), not a true
  post-merge consensus. **§4 now UPGRADES this to a direct measurement:** the same
  from-scratch run's **91-merge train-loss continuity** (converges 9.14 → 2.71, no
  compounding) measures the 0-token cell in-basin *at the actual merge boundaries* —
  so 0-token/random-init safety is **measured, not merely inferred** (within
  `K=250, I≤4, β=0`; the strict held-out scratch barrier remains the GPU refinement,
  §4.4 #2).
- **Island axis (rows 1–4):** `I ∈ {2, 4}` are **in-basin (SWA)** through the
  merges at `K=250, β=0` from real seeds. `I=7` exists **only** as the
  random-init recipe-confounded row 5 — it shows **no merge-induced blow-up** at
  7 islands but is **not** an SWA datapoint and **not** from a mature seed. So the
  measured *safe-SWA* island range is `I≤4`; `I>4` safety (and from a mature
  seed) is **unmeasured** and is a probe target (§3).
- **The ONE blow-up (row 6) is the outer-momentum knob, not maturity/islands.**
  Same fused recipe, same 528 M seed, same `I=4`, same `K=250` — only `β:0→0.9`
  — and the consensus catastrophically diverges. This is an **outer-optimizer
  overshoot** (`β=0.9 × outer_lr=1.0` over-steps the pseudo-gradient), confirmed
  independently by `diloco-loss-parity` (momentum "OVERSHOOTS — worst at ~2.5").
  **β=0 is mandatory; β=0.9 is forbidden — independent of how mature the seed is.**

### The gap the existing data leaves (why a fresh probe is needed)

Every existing cell sits at **`K=250`** — *frequent* merges, hence *small* drift
per window (`≈ 250·1e-3`). Frankle's law says the barrier is induced by **drift**
and **earliness**; at small drift even a random-init parent stays mode-connected
(row 1 proves it). So the existing record cannot, by itself, exhibit a
maturity threshold — it never stresses the drift axis that *creates* one. A naive
reading ("DiLoCo is always safe, seed maturity doesn't matter") would **fail the
task's NULL discipline**: it has not been tested against the larger-`K` /
earlier-seed regime where LMC predicts the barrier appears. That regime is what
the probe in §3 measures.

---

## 3. The fresh probe — barrier-vs-drift across the maturity ladder

`experiments/seed_maturity_threshold/lmc_probe.py` (REAL, FUSED, no eager, no
mock). It measures the **Frankle instability barrier as a function of drift** at
each maturity and island count, to locate the maturity knee `m*(K, I)`.

**Mechanism.**
- *Parent* = a single-GPU emender checkpoint at maturity `m` (full schedule-free
  state incl. base-iterate `z`), drawn from the validated reference ladder:

  | name | step | tokens | train loss |
  |---|---|---|---|
  | 176 M (very-early) | 21 500 | 176 M | 3.778 |
  | 528 M (early) | 64 500 | 528 M | 3.125 |
  | 1.06 B (mid) | 129 000 | 1.057 B | 3.144 |
  | 2.0 B (mature) | 244 141 | 2.000 B | 3.117 |

- *Replicas* = `I` independent continuations of the parent on **disjoint data
  streams** (`--seed = base + island`, the same `data_seed = seed + rank`
  sharding train.py uses for real data-parallel DiLoCo), run through the
  **validated `train.py` single-GPU path** — the `[fused-guard]` split-edit
  Triton kernel asserts on every replica, **no eager fallback**, constant LR
  `1.007e-3` (schedule-free, NOT the AdamW cosine), `gate_activation=silu`.
- *Snapshots*: each replica saves every `snap` steps up to `K`, so **one pair/
  tuple of replica runs yields the barrier at every drift** `d ∈ {snap, …, K}`
  (the instability-vs-amount-of-SGD curve) without retraining per drift.
- *Consensus(d)* = the **SF-aware average** of the `I` matched-step snapshots,
  computed offline with the **exact `train.py:diloco_merge` semantics** for the
  `β=0` local-SGD branch: arithmetic mean of the eval-`x` model weights + mean of
  the SF base-iterate `z`; schedule-free clocks preserved (verified byte-for-byte
  against the merge code path).
- *Scoring*: **offline only** — `scripts/eval_checkpoint.py --y-mode train`,
  fused, on the fixed pile-tail held-out tensor (md5 `8e1198ab`, the
  seed-race / offline-eval-references standard). **No inline held-out** during the
  training windows.
- *Barrier(d)* `= bpb(consensus_d) − mean_i bpb(replica_i,d)`.
  `≤ +0.02 ⇒ in-basin`; `> +0.02 ⇒ basin mismatch / blow-up.`
- *Multi-merge mode* (`merges>1`) re-seeds all islands from the consensus and
  repeats, testing whether a positive barrier **compounds** (runaway, the `β=0.9`
  signature) or **self-heals** across merges — i.e. loss continuity through merges.

**Why drift `K` is the stress axis.** Existing data nails the `K=250` corner
(safe at all maturities). To find `m*` we sweep drift up to `K=4000` (16× the
operating point) at `I=2` across all four maturities, then add `I=4` at the two
ends of the ladder. If even `K=4000` from the 176 M seed stays in-basin, the
operating-point "floor ≈ 0" generalizes to aggressive cadence (a strong GO); if
the early seed's barrier turns positive at some drift while the mature seed's
stays flat, that crossing **is** the maturity threshold.

**Ordered cell plan (decisive-first; results stream to `lmc_barrier_results.csv`):**

| # | maturity | `K` (drift) | `I` | merges | what it decides |
|---|---|---|---|---|---|
| 0 | 176 M | 4000 | 2 | 1 | early seed + high drift — **most likely barrier** |
| 1 | 2.0 B | 4000 | 2 | 1 | mature seed + high drift — does drift alone blow up a mature seed? |
| 2 | 528 M | 4000 | 2 | 1 | maturity sweep |
| 3 | 1.06 B | 4000 | 2 | 1 | maturity sweep |
| 4 | 176 M | 2000 | 4 | 1 | island-count modifier at the early end |
| 5 | 2.0 B | 2000 | 4 | 1 | island-count modifier at the mature end |
| 6 | 176 M | 1000 | 2 | 3 | multi-merge continuity, stressed cell (compound vs heal) |
| 7 | 2.0 B | 250 | 4 | 3 | multi-merge continuity, safe operating-point cell |

**Confound stack (every cell must clear ALL before a barrier reading is admissible):**
1. **fused, no eager** — `train.py` `[fused-guard]` asserts per replica; eval fused.
2. **disjoint data** — `--seed = base + island` ⇒ each replica a distinct stream
   (real instability, not replicated work).
3. **full SF state incl. `z`** loaded on every replica (`--resume`).
4. **plain average `β=0`** (the offline merge is the `β=0` local-SGD branch;
   `β=0.9` is the known divergent knob, excluded).
5. **`gate_activation=silu`** passed explicitly (train.py default `sigmoid` is a
   silent-correctness trap on resume).
6. **matched-step snapshots** — barrier compares consensus and replicas at the
   *same* step (same drift), and consensus is a true post-merge average.
7. **offline y-mode eval** on a fixed tensor — same scoring for replicas and
   consensus; no inline held-out.
8. **capacity** — identical 1.286 B geometry across parent, replicas, consensus
   (strict-load).

---

## 4. MEASURED — in-basin-vs-blow-up across the maturity × island × β matrix

**Phase 2 is delivered (CPU, no GPU) by harvesting the per-merge loss trajectory
that every production DiLoCo run already wrote to disk.** The strict held-out
*centroid* barrier `bpb(consensus) − mean_i bpb(replica_i)` needs per-replica
pre-merge weights, which the production runs do not checkpoint (they save only the
merged consensus), so it requires fresh replica training on a GPU — and the box is
saturated with INDEFINITE jobs (`--steps 1e8`), so that refinement is hard-gated
(see §4.4). **But the task's own validation — "in-basin-vs-blow-up measured (loss
continuity through merges)" — is read directly, with no GPU, from the per-step
train-loss trace + the explicit `>>> [DiLoCo] merge #N at step S` markers in each
run's `run.log`.** This is Frankle's instability signal in train-loss space at the
exact `β=0` centroid the merge lands on, and we **cross-check it against the
offline-scored held-out consensus BPB degradation already on disk** (two
independent modalities; both differential, both robust to the held-out tensor).

### 4.1 The measurement
For a merge at step `S`, the **detrended per-merge jump** = (post-merge windowed
train loss) − (pre-merge descent linearly extrapolated across the merge gap). The
extrapolation removes the local descent so the merge's *own* effect is isolated
(critical for the fast-descending from-scratch run, where the raw jump is biased
*downward* by the steep descent). `jump ≈ 0` ⇒ the average is in the same basin
(continuous / SWA); `jump ≫ 0` and **compounding** ⇒ the average crosses a barrier
(blow-up). Held-out degradation = `consensus_BPB − single-GPU_BPB` at matched
**total** tokens (negative ⇒ DiLoCo better ⇒ SWA). Harvested by
`harvest_merge_continuity.py` → `merge_continuity_results.csv`; combined with the
held-out CSVs by `build_threshold_table.py` → `final_threshold_table.csv`.

### 4.2 The matrix (all REAL on-disk runs; emender-mlp 1.286 B, fused, β as noted, K=250)

| cell (run) | seed maturity | `I` | `β` | merges | loss envelope (first→last, max) | **detrended train jump / merge** | held-out degr. (consensus−1GPU) | verdict |
|---|---|---|---|---|---|---|---|---|
| `stab_k250` | **scratch / 0 tok** | 4 | 0 | **91** | 9.14 → 2.91 (max 9.14) | **+0.087 ± 0.029** | **+0.0085** (parity, n=1 ⚠) | **IN-BASIN** |
| `swell_i2_k250` | 528 M | 2 | 0 | 22 | 3.54 → 3.09 (max 3.59) | −0.045 ± 0.035 | **−0.126** (n=4, SWA) | **IN-BASIN** |
| `swell_i4_k250` | 528 M | 4 | 0 | 22 | 3.52 → 3.10 (max 3.64) | +0.118 ± 0.057 | **−0.101** (n=4, SWA) | **IN-BASIN** |
| `seed_race_i4` | **1.233 B** | 4 | 0 | **358** | 3.38 → 3.11 (max 3.61) | **−0.026 ± 0.013** | **−0.104** (n=30, SWA) | **IN-BASIN** |
| `swell_i4_mom_k250` | 528 M | 4 | **0.9** | 22 | 3.54 → **16.95** (max **51.4**) | **+7.26 ± 1.32** | **+17.03** (n=4, +1.85…+35.1) | **BLOW-UP** |
| `outer_mom_i6` ⚠ | **1.966 B** | **6** | **0.9** | ~34† | 2.76 → **~6.7** (max 8.52) | **+0.46 ± 0.10** | (live; not yet scored) | **BLOW-UP** |

† `outer_mom_i6` is a **snapshot of a still-running job** (~34 merges at harvest;
the committed CSV advances as it runs); the BLOW-UP verdict is **stable across all
snapshots** (31→35 merges, jump +0.42→+0.46, loss sustained ~6.5 ≫ seed 3.1). ⚠ **It is NOT a clean β
isolation:** it changes **three** variables vs the β=0 cells at once
(`outer_lr 1.0→0.5`, `I 4→6`, seed `528 M→1.966 B`), and there is **no β=0 I=6
control** (`seed_race_i6` crashed at startup, 0 merges). The **clean,
single-variable β isolation** is the pair **`swell_i4_k250` (β=0, in-basin) vs
`swell_i4_mom_k250` (β=0.9, blow-up)** — same 528 M checkpoint, same
`outer_lr=1.0`, only `outer_beta` differs. `outer_mom_i6` is **corroboration** that
β=0.9 still diverges from a far more mature seed (and at *half* the outer_lr — which
strengthens, not weakens, the β attribution), **not** independent isolation.

**The blow-ups are smooth/sustained divergence, not NaN.** A separate caution:
`native_k250.MERGEBUG_loss63` is a β=0, K=250, I=4 run with the **same outer config
as the safe cells** that nonetheless diverged (max loss 63.4, never recovers below
4.49 ≈ seed) — because it ran the **pre-fix schedule-free merge code** (the
double-step bug fixed in commit `473b4c9`/`177cca9`, "diloco-stability-k250 PASS").
It is excluded as a *documented, since-fixed code defect*, but it carries a real
lesson (§4.5): **merge-implementation correctness is itself a safety precondition** —
the maturity-independence result holds only with the corrected merge.

### 4.3 What the matrix decides
1. **β=0 is IN-BASIN at EVERY measured maturity — including random-init (0 tokens).**
   "In-basin" here = **bounded-and-self-correcting**, not jump-free: each β=0 cell
   has real per-merge train-loss bumps, but they **do not compound** and the loss
   **ends at its global minimum**. The from-scratch I=4 run converges through
   **91 merges** (9.14 → **2.71 min**; 79/91 intervals recover below their
   pre-merge level; jump-vs-merge-index slope ≈ +0.0003, i.e. flat, no runaway).
   Its detrended per-merge barrier is small and **window-dependent**
   (**+0.012 at W=2 [n.s.] → +0.087 at W=4 → +0.146 at W=6**); an adversarial
   pseudo-merge *placebo* control confirms real merges exceed the curvature-only
   baseline at W≥3 (t=2.2–5.4), so the bump is a **genuine (small) barrier, not a
   detrend artifact** — but the IN-BASIN verdict rests on the **tail/min** (loss
   ends at its minimum, no compounding), *not* on the jump magnitude. Real seeds
   are net non-harmful (jump ≤ 0 for 528M-I2 and the 358-merge 1.233B cell;
   held-out −0.10…−0.13, **within the ±0.17 BPB single-GPU reference-noise band** —
   read as "**no divergence**", not a calibrated SWA gain). The two modalities
   (train-loss jump, held-out degradation) **agree per cell**.
2. **β=0.9 BLOWS UP at EVERY measured β=0.9 cell.** The **clean, β-isolated**
   control is `swell_i4_k250` vs `swell_i4_mom_k250` (same 528 M seed, same I=4,
   same `outer_lr=1.0`, only β: 0 → in-basin, 0.9 → +7.26/merge catastrophe, loss
   → 51). `outer_mom_i6` **corroborates** that β=0.9 also diverges from the *most
   mature* 1.966 B seed (+0.46/merge sustained), but it is **confounded** (also
   `outer_lr=0.5`, also I=6, no β=0 I=6 control) — so it shows maturity does not
   *rescue* β=0.9, without cleanly isolating β at I=6.
3. **⇒ Within the MEASURED envelope (K=250, I∈{2,4}, β=0, `outer_lr=1.0`, corrected
   merge code) the safe-seed maturity floor is ≤ 0 tokens (random-init starts
   in-basin), DIRECTLY MEASURED** — an upgrade over the Phase-1 inference. The
   gating knob **in this experiment** is the **outer optimizer (β must be 0)**;
   seed maturity and island count do **not push any measured β=0 cell out of basin**
   (though island count *does* significantly move the per-merge barrier: 528 M
   I=2 −0.045 → I=4 +0.118, z≈−2.45 — it just stays in-basin). **Drift `K` is NOT
   re-measured here** (all six cells are K=250); the "`K` is a gate" claim is
   carried from prior `diloco-periodic`/`loss-parity` work, not demonstrated in this
   matrix, and the larger-`K` axis is the GPU-gated refinement (§4.4).

### 4.4 What this does NOT yet measure (the GPU-gated refinement, armed non-blocking)
The §4.2 matrix answers the task's in-basin-vs-blow-up + threshold question from
real merge trajectories. The remaining cells are GPU-gated; the driver
(`run_lmc_probe.sh`, detached) polls `acquire 1 --no-wait` every 30 s and runs
`lmc_probe.py` only on a genuinely idle GPU (hard fail-fast guard; never clobbers a
racer), appending to `lmc_barrier_results.csv` when the box frees. Ranked by value
(from the adversarial audit, §4.5):
1. **β=0 I=6 control** — resume `seed_race_i6` (β=0, K=250, `outer_lr=1.0`,
   currently 0 merges) to ≥20 merges from the 1.966 B seed. **Single cheapest cell
   that de-confounds the I=6 blow-up**: if β=0/I=6 stays in-basin it isolates β
   (vs `outer_lr`/islands) for the mature-seed divergence and fills the one
   coverage hole the floor-≤0 claim leans on.
2. **Strict consensus held-out barrier for the scratch cell** — score per-replica +
   consensus checkpoints at a *true* merge boundary (a step ∈ 250ℤ, not the rank-0
   `@21600` proxy) for `stab_k250`, replacing the n=1 held-out point with a measured
   held-out LMC centroid barrier at maturity 0 (closes the train-loss→held-out
   inference gap, audit Lens 2).
3. **`outer_lr`/β decoupling** — one β=0.9 cell at `outer_lr=1.0` (or one β=0 cell
   at `outer_lr=0.5`), cheapest as a resume of the 528 M `swell_i4` checkpoint, so
   the divergence attributes to β cleanly rather than the β/`outer_lr` bundle.
4. **Large-`K` drift axis** (`K` up to 4000) — the `lmc_probe.py` sweep, to locate
   the SGD-stability knee on the drift axis the existing record never stresses.

None of these gate the §5 operating-point rule; they sharpen its boundaries.

### 4.5 Adversarial confound audit (NULL discipline)
The §4 empirical claims were run through a **4-lens adversarial audit** (5 agents:
detrend-validity, train-loss↔held-out proxy, blow-up classification, coverage/
overclaim, + synthesis). **Verdict: `stands-narrowed` on all four lenses, zero
refutations; every disputed number reproduced exactly against the raw logs.** The
audit drove the narrowing now folded into §0/§4.2/§4.3/§5/§6:
- **(major) Clean β isolation rests on ONE pair** (`swell_i4` β=0 vs β=0.9); the
  I=6 cell confounds β with `outer_lr`+islands and has no β=0 control → §4.2 ⚠.
- **(major) Merge-code correctness is itself a gate** — a same-config β=0 run
  (`native_k250.MERGEBUG`) diverged to loss 63 with the pre-fix merge → safety is
  established only post-fix (§4.2 note).
- Scratch held-out "+0.0085 parity" is a single **rank-0 proxy** (n=1, step
  21600 ∉ 250ℤ) → lead the floor claim with the **91-merge train-loss continuity**,
  not the held-out point.
- 528 M "SWA benefit" sits **within the ±0.17 BPB ref-noise band** → "no
  divergence", not a calibrated gain; 1.233 B uses a clamped reference for 23/30
  points (direction holds on the 7 real-overlap points, mean −0.109).
- `K`/drift gate is **asserted, not measured** here (all cells K=250).
- Island count **does** move the per-merge barrier (significant) but **does not push
  any β=0 cell out of basin** → softened from "does not gate".
- Per-merge jump is **window-dependent** (+0.012 W=2 → +0.146 W=6) → reported as a
  range; the in-basin verdict is **tail/min-based**.

---

## 5. The rule — "seed here, then scale out"

Stated as an operating envelope, because the record proves safety is **conditional
on `(β, K, I, merge-code)`**, not a single token count.

**The one-sentence threshold (audit-hardened, most-defensible form):** *the
safe-seed maturity floor is **≤ 0 tokens (from-scratch starts in-basin)** within the
**MEASURED envelope `K=250`, `I ∈ {2,4}`, `β=0`, `outer_lr=1.0`, with the corrected
schedule-free DiLoCo merge**; the gating knob in this experiment is the **outer
optimizer** (every β=0 cell self-corrects and stays in-basin; both β=0.9 cells
diverge), while maturity and island-count do **not push any measured β=0 cell out of
basin** — but `I=6`, `K>250`, and the β×maturity cross are **unmeasured**, and
merge-implementation correctness is itself a precondition.*

### 5a. At the validated Frontier operating point (`β=0`, `K=250`, `I≤4`, `outer_lr=1.0`)
- **No seed-maturity floor: random-init (0 tokens) already starts in-basin —
  DIRECTLY MEASURED.** The from-scratch I=4 DiLoCo converges through **91 merges**
  (loss 9.14 → 2.71, no merge blow-up; lead evidence is the train-loss continuity,
  §4.3). It also ties single-GPU at matched tokens (+0.0085 held-out) — but that
  held-out point is a single rank-0 proxy (n=1), so the *robust* support is the
  91-merge continuity, not the BPB point. **There is no safety reason to bank any
  seed at this operating point; the cheapest safe seed is "whatever you would train
  anyway," including nothing.**
- **A mature seed is a bonus, not a requirement:** the 528 M and 1.233 B cells show a
  real seed keeps descending with **no divergence** through up to 358 merges (held-out
  −0.10…−0.13, *within* the ±0.17 BPB reference-noise band — read as non-harmful, not
  a calibrated SWA gain). If you already have a 1–2 B seed (we do), use it — it is a
  head-start, **not** a blow-up-averting necessity.

### 5b. HARD constraints that DO gate safety (independent of maturity)
- **Outer optimizer = plain average `β=0` (`outer_lr=1.0`).** The clean β-isolated
  control (`swell_i4` β=0 vs β=0.9, same seed/I/lr) shows β=0.9 DIVERGES (loss → 51;
  held-out +1.8 → +35); β=0.9 also diverges from the most mature 1.966 B seed at I=6
  (confounded with `outer_lr=0.5`/islands, but corroborating). **This is the single
  most important knob — do not enable outer momentum on this box/recipe.**
- **Corrected merge code is a precondition.** A β=0 run on the *pre-fix*
  schedule-free merge (`native_k250.MERGEBUG`) diverged to loss 63 at the same
  `(β, K, I)` as the safe cells. Use the post-`473b4c9`/`177cca9` merge; the
  maturity-independence result holds **only** with it.
- **Bounded merge interval / drift.** `K=250` is the measured-safe cadence at every
  maturity. Larger `K` is **NOT re-measured in this experiment** (all cells K=250);
  the larger-`K` safety boundary is carried from prior `diloco-periodic` work and is
  the GPU-gated refinement (§4.4) — treat `K>250` as unverified until the probe lands.
- **Full SF state on the seed** (`--resume` incl. `z`), `gate_activation=silu`,
  **fused** (no eager) — correctness gates, not performance knobs.

### 5c. When a real maturity floor *could* re-appear (stressed regime — unmeasured)
If the Frontier plan forces **aggressive cadence** (`K ≫ 250`, e.g. to amortize a
slow interconnect) or **many islands** beyond the measured `I≤4` (β=0) range, then
per §1/§7 Frankle predicts a maturity floor re-appears: the early seed's barrier
turns positive at high drift while the mature seed's stays flat. **This regime is
UNMEASURED in this experiment** (all cells K=250; β=0 only to I=4). The GPU-gated
probe (§4.4) is what measures that knee; until it lands, treat `K≫250` / `I>4` as
**conservative-seed territory** — either seed past a healthy maturity (≥0.5–1 B is
cheap and gives a wide basin) **or** shrink `K`, rather than scaling out
from-scratch at aggressive cadence on the strength of the K=250 result alone.

---

## 6. Confound audit (NULL discipline) — defending the narrowed claim

The narrowed claim "**β=0 averaging keeps DiLoCo in-basin from any maturity
(floor ≤ 0 tokens) within `K=250, I∈{2,4}, outer_lr=1.0`, corrected-merge, while
β=0.9 diverges**" survived a **4-lens adversarial audit (§4.5): stands-narrowed,
0 refutations, all numbers reproduced**. The full stack:

| confound | resolution |
|---|---|
| eager vs fused | all measured cells are the fused split-edit Triton kernel (`[fused-guard]`, no eager path); the GPU probe asserts the same. Eager numbers non-transferable (NON-NEGOTIABLE #1), excluded. |
| wrong island count | measured in-basin at `I ∈ {2,4}` (β=0) and the live `I=6` blow-up is β=0.9 only; **no β=0 I=6 control exists** (`seed_race_i6` crashed at 0 merges) → §0/§5 refuse to claim β-isolated I=6 safety; it is the #1 follow-up (§4.4). Island count *does* move the per-merge barrier (528 M I2→I4, z≈−2.45) but does not push any β=0 cell out of basin. No extrapolation to 100s–1000s islands. |
| **β not cleanly isolated at I=6** | the clean single-variable β control is the `swell_i4` β=0/β=0.9 pair (same seed/I/lr); `outer_mom_i6` confounds β with `outer_lr=0.5`+I=6 and is cited only as corroboration, not isolation. |
| **merge-code correctness** | a β=0 run on the *pre-fix* merge (`native_k250.MERGEBUG`) diverged to loss 63 → safety is conditional on the post-`473b4c9`/`177cca9` merge; stated explicitly, not hidden. |
| outer schedule | restricted to `β=0` for the safe claim; the `β=0.9` divergence (blow-up out of basin) is reported as the binding constraint. |
| plain vs momentum average | the offline/the production merge is the `β=0` local-SGD branch (`outer_lr=1`), byte-matched to `diloco_merge`. |
| matched tokens | held-out degradations at matched **total** tokens (seed-aware piecewise: seed ×1, DiLoCo ×I); the train-loss continuity is matched-**step** (same drift). |
| held-out noise floor & extrapolation | the single-GPU reference is non-monotone (±0.17 BPB constant-LR band) → 528 M "SWA benefit" is reported as **no-divergence within noise**, not a calibrated gain; the 1.233 B cell uses a clamped reference for 23/30 points (direction holds on the 7 real-overlap points, mean −0.109). |
| scratch held-out is n=1 | the +0.0085 "parity" is a single rank-0 proxy (step 21600 ∉ 250ℤ) → the floor claim **leads with the 91-merge train-loss continuity**, not the held-out point; the strict held-out scratch barrier is GPU-gated (§4.4 #2). |
| train-loss vs held-out proxy | the per-merge jump is rank-0 *train* loss, a looser proxy than the held-out centroid barrier; **cross-checked** against the offline-scored held-out degradation (both agree per cell), and the in-basin verdict is **tail/min-based** (loss ends at its global min, no compounding), robust to the window-dependent jump magnitude. |
| **drift `K` not measured here** | all six cells are K=250; "`K` is a gate" is carried from `diloco-periodic`, **not** demonstrated in this matrix → the larger-`K` axis is explicitly the GPU-gated refinement (§4.4 #4), not asserted as safe. |

**Three things this analysis explicitly does NOT claim:** (1) β-isolated safety at
`I=6` / 100s–1000s islands (unmeasured; the I=6 blow-up is confounded); (2) safety
at `K>250` (unmeasured here); (3) token-*efficiency* — DiLoCo is matched-token
*safe* (in-basin) but carries a large-batch sample-efficiency cost
(`diloco-longhorizon`), a separate axis from this basin-safety question.

---

## 7. LMC / basin interpretation

The measured matrix (§4) is a clean instance of Frankle's law at the **small-drift**
end, and it now shows the SGD-stability transition *quantitatively*. At `K=250`
even a random-init parent is **already essentially past** the practical
SGD-stability point *for that drift* — 250 steps of `1e-3` SGD is not enough
independent travel to cross a basin — so all `I` replicas stay linearly
mode-connected and their average self-corrects each merge (loss ends at its
minimum). Consistent with Frankle, the residual per-merge barrier is **mildly
positive at random-init** (detrended +0.09 train-loss/merge, ~3 SEM, non-fatal:
the run still converges 9.14 → 2.71) and **relaxes toward zero / net-negative as
the seed matures** (the 358-merge 1.233 B cell is −0.026) — the early replicas
diverge slightly more before the merge, exactly the "earlier fork tolerates less
SGD" prediction, but the drift at K=250 is small enough that even the random-init
barrier never compounds. The divergence (`β=0.9`) is **not** a basin-mismatch — it
is the outer optimizer *overshooting* the consensus (`1/(1−β)=10×` step), a
step-size instability **orthogonal to LMC**, which is why it ignores maturity
(it diverges even from the 1.966 B seed).

What remains for the GPU probe (§4.4): push the **drift axis** (`K·lr`) at each
maturity to find *how far* replicas can wander before the average crosses a
barrier. Frankle predicts the tolerated drift **grows with maturity** — the mature
2 B seed should stay mode-connected to larger `K` than the random-init seed. That
barrier-vs-drift crossing is the SGD-stability knee on the one axis this matrix
holds fixed; it sharpens the largest-safe `(maturity, K, I)` envelope but does not
move the K=250 operating-point verdict.

---

## 8. Frontier de-risking recommendation

- **The gate is NOT "bank a huge seed."** At the planned operating point
  (`β=0`, `K=250`, `I≤4`, `outer_lr=1.0`, corrected merge) the cheapest safe seed
  is **free**: the from-scratch I=4 DiLoCo stays in-basin through **91 merges**
  (DIRECTLY MEASURED, §4.2) and ties single-GPU at matched tokens. **Do not spend
  granted compute buying seed maturity for *safety*.** Use the mature 1–2 B seed we
  already have because it is a *head start* (and keeps descending with no
  divergence), **not** because scaling out from earlier would blow up. **Caveats
  the allocator must act on:** `I>4` (β-isolated), `K>250`, and the β×maturity
  cross are **unmeasured**; merge-code correctness is a precondition.
- **Spend the de-risking budget on the knobs that actually gate safety:**
  (1) pin the outer optimizer to plain average **`β=0` (never `0.9`)** — the single
  binding constraint (`β=0.9` diverges out of basin); (2) use the **corrected schedule-free merge** (post-`473b4c9`);
  (3) keep `K=250` (the only measured-safe cadence) unless the §4.4 probe has
  cleared a larger `K` — if the interconnect forces `K≫250`, seed past a healthy
  maturity first rather than scaling out from-scratch at aggressive cadence.
- **Run the cheap pre-flight before betting the allocation, in priority order
  (§4.4):** (1) the **β=0 I=6 control** (`seed_race_i6` to ≥20 merges) — the single
  cheapest cell that de-confounds the I=6 blow-up; (2) the strict held-out scratch
  barrier at a true merge boundary; (3) the `lmc_probe.py` barrier-vs-drift sweep
  for the chosen cadence — all far cheaper than discovering a basin barrier
  mid-allocation.

---

## 9. Artifacts
**Phase 2 empirical (CPU, delivered):**
- `experiments/seed_maturity_threshold/harvest_merge_continuity.py` — parses the
  production DiLoCo `run.log`s for per-step loss + `merge #N at step S` markers,
  computes the detrended per-merge train-loss jump and the in-basin/blow-up verdict.
  → `merge_continuity_results.csv`, `merge_continuity_summary.json`.
- `experiments/seed_maturity_threshold/build_threshold_table.py` — combines the
  train-loss continuity with the held-out BPB degradation (two modalities).
  → `final_threshold_table.csv` (the §4.2 matrix, script-reproducible).
**Phase 2 GPU refinement (armed, non-blocking):**
- `experiments/seed_maturity_threshold/lmc_probe.py` — the strict held-out
  barrier-vs-drift probe (single-GPU sequential replicas → offline SF-aware merge →
  offline y-mode eval); hard fail-fast GPU guard; resumable; decisive-first plan.
- `experiments/seed_maturity_threshold/run_lmc_probe.sh` — non-blocking
  self-leasing driver (`acquire 1 --no-wait` poll loop; never blocks/clobbers).
- `experiments/seed_maturity_threshold/lmc_barrier_results.csv` — streamed per-drift
  strict barrier rows when the box frees.
**Real source runs harvested (on disk):** `/mnt/nvme1n1/erikg/diloco_sweep/`
  `stab_k250`, `swell_i2_k250`, `swell_i4_k250`, `swell_i4_mom_k250`,
  `seed_race_i4`, `outer_mom_i6` (+ `native_k250.MERGEBUG_loss63` as the pre-fix
  counter-example). Prior REAL evidence reused: `experiments/diloco_scaling_law/`,
  `experiments/diloco_seed_race_i4/`, `experiments/diloco_100b/longhorizon/`.

---

## 10. Adversarial-audit hardening notes

### 10.0 Phase-2 empirical audit (the §4 matrix)
The §4 empirical claims were run through a **second 4-lens adversarial audit**
(detrend-validity, train-loss↔held-out proxy, blow-up classification, coverage/
overclaim + synthesis; 5 agents, each re-deriving numbers from the raw `run.log`s).
**Verdict: `stands-narrowed` on all four lenses, ZERO refutations; every disputed
number reproduced exactly** (the detrended jumps, the held-out degradation ranges,
the `swell_i4` same-checkpoint β A/B, `seed_race_i6`=0 merges, the `native_k250`
pre-fix loss-63 config). Eight caveats from that audit are folded into
§0/§4.2/§4.3/§4.4/§4.5/§5/§6 — the two material ones: **(1)** the clean β isolation
is the `swell_i4` pair only (the I=6 blow-up confounds β with `outer_lr`/islands;
no β=0 I=6 control); **(2)** merge-code correctness is itself a gate (a same-config
β=0 run diverged to loss 63 on the pre-fix merge). The audit also confirmed (via a
pseudo-merge placebo control) that the small from-scratch per-merge barrier is a
**genuine** effect, not a detrend artifact, while the IN-BASIN verdict correctly
rests on the tail/min (no compounding), not the window-dependent jump magnitude.

### 10.1 Phase-1 design audit (the existing-data analysis + probe)
The §1–§3 design was run through a 4-lens adversarial audit (numbers-fidelity,
NULL-discipline red-team, LMC-theory, probe-implementation). Verdict: **mechanism
sound; probe implementation + LMC interpretation confirmed correct; two scoping
fixes applied** (now folded into §0/§2/§5/§8):

- **Floor claim scoped (was overclaimed):** "floor ≈ 0 tokens" → "**no measurable
  floor down to the lowest cell run (176 M tokens)**"; true 0-token/random-init
  safety is labelled an **LMC inference**, not a measurement.
- **Row 5 relabelled (was mislabelled):** the `I=7` longhorizon run is
  **random-init** ("1.286 B" was the parameter count) and its held-out BPB
  **collapses for a recipe reason shared with the single-GPU baseline** — it is a
  "no merge blow-up at I=7" datapoint only, **not** SWA, **not** a mature seed.
  There is **no measured `I>4` run from a mature seed**.

Confirmed-correct by the audit (survived the red-team): all Rows 1–4/6 numbers
reconcile to `degradation_summary.json` / `seed_race_i4_degradation.csv` /
checkpoint filenames; matched-total-token accounting is consistent; the y-mode
eval scores the consensus at exactly `mean_i(y_i)` (schedule-free `y` is affine in
`x,z`, averaging is linear) so the barrier lives in the real train-weight space
DiLoCo continues from; the probe averages the **same** quantities as
`train.py:diloco_merge`'s `β=0` branch (eval-`x` + base-iterate `z`, clocks
preserved); the fused/no-eager/disjoint-data path and the fail-fast GPU guard are
genuine; and the `β=0.9` +35 BPB is correctly diagnosed as an **outer step-size
instability** (`1/(1−β)=10×` overshoot), orthogonal to LMC.

Residual caveats (carried, low-impact):
- **§2 degradation ≠ §3 barrier.** The §2 *degradation* is vs the single-GPU
  reference (SWA-vs-baseline evidence); the §3/§4 *barrier* is vs the
  mean-of-replicas (the strict in-basin test). The former is looser; do not
  conflate them — both point the same way, but the barrier is the rigorous LMC
  quantity.
- **Row 1 is a rank-0 replica proxy** (step 21600 ∉ 250ℤ), not a true consensus —
  read it as "no barrier *visible* at 176 M," not a clean consensus number.
- **Mixed held-out tensors across §2 rows** (07005c39 / 8e1198ab / cb0a478c):
  per-row signs are tensor-robust (differential), cross-row absolute BPB are not.
- **`in_basin` gate = `barrier ≤ +0.02 BPB`** is a chosen convention (the
  differential noise floor on a fixed tensor, well inside the ±0.12 absolute
  band), not a derived threshold; §4 may report a small CI instead.
- **2.0 B-seed drift labels are offset** (parent step 244141 ∉ 250ℤ → drifts
  {109,359,…}); bucket those cells by the `drift_tokens` column, not face-value
  step labels (barrier values are unaffected).
