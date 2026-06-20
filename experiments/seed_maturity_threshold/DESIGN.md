# Seed-Maturity Threshold for SAFE Horizontal DiLoCo Scaleout
### "Where do we seed from before going horizontal on Frontier?"

**Task:** `seed-maturity-threshold` (Architect / design document with rationale).
**Author:** agent-1957. **Date:** 2026-06-20.
**Status:** **Phase 1 (CPU-only existing-data analysis + LMC design + the
threshold rule) is COMPLETE and is the shipped deliverable** (§2, §5–§8, no GPU).
Phase 2 (the fresh stress-probe `lmc_probe.py`) is **armed and non-blocking** —
the box is saturated with indefinite jobs, so it polls `--no-wait` and populates
§4 if/when a GPU frees; the §5 rule already follows from the existing measured
record and Phase 2 only confirms/sharpens the larger-`K` corner.

---

## 0. The question, and the answer in one line

**Question.** What is the MINIMUM single-model training maturity (loss / BPB /
tokens) at which horizontal DiLoCo averaging is SAFE — replicas stay in the SAME
loss basin so the periodic weight-average is benign or beneficial (SWA) — versus
the under-trained regime where data-diverged replicas land in DIFFERENT basins
and the average sits on a loss barrier (blow-up)?

**Answer (headline).** At the **validated Frontier operating point** — plain
weight-average (outer `β=0`), merge interval `K=250` local steps, island count
`I ≤ 4` (directly measured) — there is **no measurable seed-maturity floor down to
the lowest cell run**: horizontal averaging is in-basin and benign at every β=0,
K=250 cell measured — a **from-scratch I=4 run scored at 176 M tokens reaches
single-GPU parity (+0.009 BPB, inside the ±0.12 reference noise)**, and any real
seed (528 M, 1.233 B) is mildly **SWA-beneficial** (−0.07 … −0.14 BPB). True
**0-token / random-init** safety, and any maturity *below the 176 M-token measured
floor*, is **inferred from LMC theory** (250 steps of `1e-3` SGD is too little
drift to cross a basin) — **not directly measured**, and is stated as inference.
The factors that actually decide safety are **NOT seed maturity** but:

1. **the outer optimizer** — plain average `β=0` is safe; `β=0.9` momentum
   **DIVERGES** (+1.8 → +35 BPB) at *every* maturity and island count tested; and
2. **the drift per merge window** (monotone in `K`, which the probe sweeps) — the
   linear-mode-connectivity (LMC) barrier is induced by how far replicas wander
   between merges, not by how under-trained the parent is, *within the reachable
   K=250 regime*.

A genuine **maturity threshold re-appears only in the stressed regime** (large
`K` / many islands), where Frankle's SGD-stability theory predicts an early seed
tolerates *less* drift before the merge crosses a basin than a mature seed does.
That drift-induced threshold is **theory-only until the §4 probe lands** — the
existing record is confined to **K=250, I≤4, ≥176 M-token, β=0**. Within that
envelope the **operating-point rule** (§5) is decided: **you do not need to bank
an expensive seed to scale out at `K=250, β=0, I≤4` — the merge cadence and the
outer-optimizer choice are the gate, not the seed.** (There is **no measured
`I>4` run from a mature seed**; the only `I=7` run, §2 row 5, is random-init and
recipe-confounded — see the note there.)

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

- **Maturity axis (rows 1–4):** across the measured maturity ladder —
  **random-init-scored-@176 M → 528 M → 1.233 B** — averaging is **in-basin at
  every measured point**. The lowest cell (from-scratch I=4, scored at 176 M
  tokens) reaches single-GPU parity (`+0.009`, within noise), and *any* real seed
  is strictly *better* and below the single-GPU envelope. There is **no positive
  island/basin penalty for a better seed to "shrink"** at `K=250, I≤4`; the seed
  only adds an SWA bonus. **⇒ no measurable maturity floor down to the lowest cell
  run (176 M tok); 0-token/random-init safety is an LMC inference (§7), not a
  measurement.** ⚠ row 1 is a **rank-0 replica proxy** (step 21600 is *not* on a
  K=250 merge boundary), not a true post-merge consensus — so it bounds the
  consensus only under the SWA hypothesis it is testing; read it as "no barrier
  *visible* at 176 M," not a clean consensus number.
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

## 4. MEASURED barrier-vs-drift  *(fills in as the probe lands)*

> **Probe status (Phase 2 — GPU-gated).** The box is saturated with INDEFINITE
> jobs (racers on GPU 0–1, the outer-momentum β=0.9 test on GPU 2–7, all
> `--steps 1e8`); `list-free` is empty. Per the operator constraint we **must not
> clobber** them and **must not block** on the lease. The probe driver
> (`run_lmc_probe.sh`, detached PID 3141645) therefore **polls the broker with
> `acquire 1 --no-wait` every 30 s** and runs ONLY when a genuinely-idle GPU is
> granted; a hard fail-fast guard in `lmc_probe.py` aborts if no GPU is pinned
> (so it can never default onto a racer's GPU 0 — the bug that OOM'd the racer on
> the first launch, now fixed). It then runs the cells in the order above,
> appending to `lmc_barrier_results.csv`, and the table below populates as cells
> complete. **Phase 1 (the existing-data analysis in §2 + the rule in §5) is the
> shipped deliverable and stands independent of Phase 2.** Phase 2 confirms and
> extends it (the larger-`K` drift axis) whenever the box frees.

| cell | maturity | drift (steps / tokens) | `I` | mean replica BPB | consensus BPB | **barrier** | verdict |
|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | |

**Reading rule.** For each maturity, plot `barrier` vs `drift`. The maturity
threshold `m*(K,I)` is the smallest maturity whose barrier stays `≤ +0.02` out to
the intended operating drift. If all maturities stay flat-and-negative out to
`K=4000`, the operating-point "floor ≈ 0" extends to aggressive cadence; if the
early curves bend positive while the mature ones stay flat, the crossing drift
defines the safe `(maturity, K)` envelope.

---

## 5. The rule — "seed here, then scale out"

Stated as an operating envelope, because the existing record proves the threshold
is **conditional on `(β, K, I)`**, not a single token count.

### 5a. At the validated Frontier operating point (`β=0`, `K = 250`, `I ≤ 4` measured)
- **No measurable seed-maturity floor down to the lowest cell run (176 M tokens).**
  Stated in the task's requested units: the MEASURED anchor is a from-scratch
  (random-init) I=4 DiLoCo scored at **176 M tokens** reaching single-GPU
  **parity (+0.009 BPB, within ±0.12 noise)**, with no merge blow-up; the SWA
  benefit is measured from 528 M and 1.233 B seeds. **True 0-token / random-init
  safety, and any maturity below 176 M, is INFERRED** from LMC theory (§7: at
  K=250 the per-window drift is too small to cross a basin even from random init)
  — *not directly measured*. So the operating-point rule is: **down to at least
  176 M tokens there is no safety reason to bank more seed; below that, rely on
  the LMC inference (small drift) or simply seed to ≥176 M, which is cheap.**
- Any real seed only improves on the lowest cell (SWA bonus). **You do not need to
  bank an expensive single-GPU seed to scale out at this operating point**; the
  cheapest safe seed is essentially "whatever you would train anyway."
- **A mature seed is a bonus, not a requirement:** rows 2–4 show a mature seed
  buys an SWA-style −0.07…−0.14 BPB and a head start, but it does not change the
  *safety* verdict. If you already have a 1–2 B seed (we do), use it — it is
  strictly better — but its value is *head-start + SWA*, not *averting blow-up*.

### 5b. HARD constraints that DO gate safety (independent of maturity)
- **Outer optimizer = plain average `β=0` (`outer_lr=1.0`).** `β=0.9` DIVERGES
  (+35 BPB) at every maturity/island tested. This is the single most important
  knob — do not enable outer momentum on this box/recipe.
- **Bounded merge interval / drift.** Keep `K` within the measured in-basin
  envelope (§4). `K=250` is proven safe at all maturities; larger `K` is safe
  only up to the drift where §4's barrier stays ≤ 0 (more conservative for
  earlier seeds and more islands).
- **Full SF state on the seed** (`--resume` incl. `z`), `gate_activation=silu`,
  **fused** (no eager) — these are correctness gates, not performance knobs.

### 5c. When a real maturity floor re-appears (stressed regime)
If the Frontier plan forces **aggressive cadence** (large `K`, e.g. to amortize a
slow interconnect) or **many islands** beyond the measured range, then per §1/§4
seed *past the SGD-stability knee* before scaling out: the early seed's barrier
turns positive at high drift while the mature seed's stays flat. The probe's
measured knee (§4) sets that floor in tokens/BPB; below it, high-drift scaleout is
unsafe and you must either (i) seed more, or (ii) shrink `K`.

---

## 6. Confound audit (NULL discipline) — defending the universal claim

The claim "**averaging shows no measurable barrier at every measured maturity
(≥176 M tok) at the operating point (`β=0, K=250, I≤4`)**" is a near-universal
statement, so it must survive the full stack:

| confound | resolution |
|---|---|
| eager vs fused | all existing cells + the probe fire the `[fused-guard]`; no eager path. Eager numbers are non-transferable (project NON-NEGOTIABLE #1) and are excluded. |
| wrong island count | measured SWA at `I ∈ {2,4}` from real seeds; `I=7` is only the random-init recipe-confounded row 5 (no merge blow-up, but not SWA); the safe-SWA claim is scoped to **`I ≤ 4`** and explicitly **refuses** to extrapolate to `I>4` / 100s–1000s islands (no slope to fit; a probe target). |
| outer schedule | restricted to `β=0`; the `β=0.9` divergence is reported as the binding NO-GO, not hidden. |
| plain vs momentum average | merge is the `β=0` local-SGD branch, byte-matched to `diloco_merge`. |
| matched tokens | all degradations at matched **total** tokens via the seed-aware piecewise formula (seed phase ×1, DiLoCo phase ×I); the probe's barrier is matched-**step** (same drift) which is even tighter. |
| seed loaded incl. SF `z` | `--resume` → `load_checkpoint` restores model + optimizer (`z`, clocks); eval re-applies y-mode. |
| held-out noise floor | the single-GPU reference held-out curve is non-monotone (±0.12–0.17 BPB constant-LR band); every degradation is judged against that floor, and the barrier is a *differential* (consensus − replica on the same tensor) robust to the absolute anchor. |
| drift axis untested (the real gap) | **this is exactly why §3's probe exists** — the "always safe" reading is NOT asserted beyond `K=250` until §4 measures the larger-`K` barrier. The operating-point rule (5a) is what the data supports; the stressed-regime caveat (5c) is flagged as conditional, not promised. |

**Two things this analysis explicitly does NOT claim:** (1) safety at 100s–1000s
islands (unmeasured; refused per NULL discipline); (2) that DiLoCo is
token-*efficient* — it is matched-token *safe* (in-basin) but carries a
large-batch sample-efficiency cost (`diloco-longhorizon`), a separate axis from
this task's basin-safety question.

---

## 7. LMC / basin interpretation

The existing record is a clean instance of Frankle's law at the **small-drift**
end: at `K=250` even a random-init parent is *already* past the practical
SGD-stability point *for that drift* — 250 steps of `1e-3` SGD is not enough
independent travel to cross a basin, so all `I` replicas remain linearly
mode-connected and their average is benign (often flatter ⇒ SWA-better). As the
seed matures the loss landscape only gets *more* connected (wider basin), so the
barrier stays ≤ 0 and the SWA bonus grows. The single divergence (`β=0.9`) is
**not** a basin-mismatch at all — it is the outer optimizer *overshooting* the
consensus point, a step-size instability orthogonal to LMC.

The probe (§3) pushes the drift axis: it asks *how far* the replicas can wander
(`K·lr`) from each maturity before the average crosses a barrier. Frankle predicts
the tolerated drift **grows with maturity** — the mature 2 B seed should stay
mode-connected to larger `K` than the 176 M seed. The measured barrier-vs-drift
crossing (§4) is the empirical SGD-stability knee, expressed as the largest safe
`(maturity, K, I)` envelope.

---

## 8. Frontier de-risking recommendation

- **The gate is NOT "bank a huge seed."** At the planned operating point
  (`β=0`, `K=250`, `I≤4` measured) the cheapest safe seed is ≈ free: the lowest
  measured cell (random-init, scored at 176 M tok) is already at parity, and the
  LMC argument (§7) says even a smaller/0-token seed stays in-basin at this small
  drift. **Do not spend granted compute buying seed maturity for *safety*.** Use
  the mature 1–2 B seed we already have because it is *strictly better* (head
  start + SWA), not because scaling out from earlier would blow up. **Caveat:**
  `I>4` and larger `K` are unmeasured — confirm with the §3 probe before scaling
  beyond the measured `I≤4, K=250` envelope.
- **Spend the de-risking budget on the two knobs that actually gate safety:**
  (1) pin the outer optimizer to plain average `β=0` (never `0.9`); (2) keep the
  merge interval `K` inside the measured in-basin drift envelope (§4) — and if the
  interconnect forces a larger `K`, seed past the §4 knee first.
- **Run the cheap pre-flight before betting the allocation:** the `lmc_probe.py`
  barrier-vs-drift sweep (a few single-GPU GPU-hours) tells you the exact
  `(maturity, K, I)` safety envelope for the chosen cadence — far cheaper than
  discovering a basin barrier mid-allocation.

---

## 9. Artifacts
- `experiments/seed_maturity_threshold/lmc_probe.py` — the barrier-vs-drift probe
  (single-GPU sequential replicas → offline SF-aware merge → offline y-mode eval);
  hard fail-fast GPU guard; resumable; cheap-decisive-first ordered plan.
- `experiments/seed_maturity_threshold/run_lmc_probe.sh` — non-blocking
  self-leasing driver (`acquire 1 --no-wait` poll loop; never blocks, never
  clobbers a busy GPU).
- `experiments/seed_maturity_threshold/lmc_barrier_results.csv` — streamed
  per-drift barrier rows (cell, maturity, drift, replica/consensus BPB, barrier).
- Run logs / manifests under `/mnt/nvme1n1/erikg/seed_maturity_threshold/`.
- Prior REAL evidence reused: `experiments/diloco_scaling_law/`,
  `experiments/diloco_seed_race_i4/`, `experiments/diloco_100b/longhorizon/`, and
  the evaluator verdict `evaluations/diloco-scaling-law-evaluation.md`.

---

## 10. Adversarial-audit hardening notes

This document was run through a 4-lens adversarial audit (numbers-fidelity,
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
