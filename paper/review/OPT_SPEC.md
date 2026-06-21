# OPT_SPEC — "OPTIMIZATION NOT ARCHITECTURE" investigation (design doc)

**Task:** `opt-spec` · **Role:** Architect (design only, no GPU) · **Date:** 2026-06-10

This document is the **shared experimental contract** for the
"OPTIMIZATION-NOT-ARCHITECTURE" line. It defines (1) the optimization goal and
its metric, (2) the shared harness / battery / metrics every probe MUST use,
(3) the four lever-probe designs, (4) the mandatory controls, (5) the
small→1.3B protocol, and (6) the aggregation/synthesis plan. It is written so
that **four probe agents build directly comparable runs** and **one synth agent
aggregates them into a single GO/NULL verdict.**

---

## 0. Why this line exists — the thesis

**Architecture is exhausted on this substrate.** Every recent "new cell" lever
came back NULL at convergent loss:

- `rot` (complex-eigenvalue) — periodic/clock capability NULL; matched-real
  solves it too ([[complex-eig-capability-periodic-null]]).
- TTT / `refit` (momentum + K-step inner optimizer) — convergent-capability
  NULL; the momentum-off special case (= GDN-2's gated-delta rule) does
  everything ([[ttt-capability-convergent-null]]).
- `nlmem` (MLP fast-weight memory) — HONEST NULL; ties GDN-2 only on XOR,
  loses elsewhere ([[nlmem-capability-inert-then-null]]).
- The within-layer `e97_raw + gdn-neg` mixture is **capability-complete but
  LM-dominated** — it only *ties* GDN-2 on held-out BPB while costing 2.6×
  (`E97_WITHIN_LAYER_SYNTHESIS.md`: ties BPB, LM-dominated at 2.6× cost).

The standing result across all of these is a robust **convergent-loss null**:
at matched compute and convergent loss, exotic function-class moves do not beat
the GDN-2 (gated-delta) special case.

**The thesis of this line.** If a *better* model for the GDN+nonlin within-layer
mixture exists, it is a **better local optimum reached by a better TRAINING
REGIME**, not a new cell. The optimized object is the **training** — init,
per-head-type learning rates, normalization placement, gate/decay init, and the
minimal load-bearing cell — **with the architecture/function-class held fixed.**

This investigation is the honest, pre-registered test of that thesis. The null
hypothesis is explicit and falsifiable (§1.4): *even the best training regime
ties GDN-2-default at convergent loss*, which would extend the convergent-loss
null from architecture to optimization. A real win must clear the bar in §1.4.

---

## 1. Optimization goal and metric (PRE-REGISTERED)

### 1.1 The optimized object

We optimize a **training regime** `R` applied to a **fixed substrate cell** `M`
(the within-layer GDN+nonlin mixture; §2.2). `R` ranges over: init / placement,
per-head-type LR groups, normalization placement + gate/decay/dt init, and the
minimal cell ablation. `M`'s function class — the head-type set and the kernels
— is **frozen**. No probe may add a head type, change a kernel, or widen the
function class; that would be an architecture move and is out of scope.

### 1.2 "Better local optimum" — defined precisely

A **better local optimum** is a single trained instance of `M` that
**SIMULTANEOUSLY holds all capability corners that single configurations trade
off against each other**, *at convergent loss*. The corners that demonstrably
trade off on this substrate are:

| corner | witness task(s) | who owns it today | who loses it |
|---|---|---|---|
| **recall** | `mqar_recall` | GDN-2 (0.99) | all-delta / refit / e97_raw (~0.17) |
| **counting** | `modular_counter` K=5, `dyck_depth_unbounded`, `anbncn_viability` | refit / all-delta (0.99) | GDN-2 (0.86, high-variance) |
| **step-growth / nonlin** | `modular_quadratic` p=64, `iterated_nonlinear_map` | refit / `nonlin` / e97 (1.00 / 0.94) | GDN-2 (0.84) |
| **track / depth** | `s5_permutation` (running) | `reflect` / gdn-neg (1.00) | positive-eigenvalue-only cells (~0.10) |
| **latch** (control corner) | `flag_hold_recall` | everyone (1.00) | — (least discriminating; §1.3) |

The documented trade-off is the crux: **GDN-2 owns recall but is the weak
counting/step-growth arm; the all-delta/refit/e97 substrate owns
counting/step-growth but goes recall-blind** (`TTT_CAPABILITY_RESULTS.md` §1;
`E97_WITHIN_LAYER_SYNTHESIS.md` Q1). A "better local optimum" is the training
regime that makes **one** trained mixture hold **both** sides at once — not by
a new cell, but by *how it is trained*.

### 1.3 The Joint Capability Coverage (JCC) metric

For a trained config `X` evaluated on the shared battery (§3):

1. **Per-corner specialist ceiling `S_c`** — the best accuracy any *single
   specialist* reaches on corner `c` under the shared protocol, measured once
   from the control arms and **frozen** as the denominator (recall→GDN-2;
   counting/step-growth→refit-del/all-delta; track→gdn-neg/reflect; nonlin→
   `nonlin`; latch→any). Frozen ceilings make every probe's JCC comparable.

2. **Per-corner held ratio** `r_c(X) = acc_X(c) / S_c`, clamped to `[0, 1]`,
   averaged over seeds and the eval-length grid (§3.3). A corner is **held** if
   `r_c(X) ≥ τ`, with **τ = 0.95** (pre-registered).

3. **Headline JCC (worst-corner ratio):**
   `JCC(X) = min_c r_c(X)` over the **scored** corners
   `{recall, counting, step-growth, track}`.
   The `min` (not mean) is deliberate: the thesis is "holds ALL corners
   *simultaneously*", so a config that aces recall and fails counting must score
   low. `latch` is **excluded from the headline** (it is solved by every arm —
   the least-discriminating witness, `SPECIALIZATION_STUDY_RESULTS.md` §1‡) and
   reported only as a saturation sanity check.

4. **Reported alongside the headline** (for diagnosis, never as the headline):
   - **`#corners-held`** — count of scored corners with `r_c ≥ τ` (0–4).
   - **Harmonic mean** of `r_c` over scored corners (smoother than `min`,
     surfaces within-sweep gradients).
   - The **full per-corner `r_c` table** at every eval length.

The headline is `min_c r_c` because it is the only aggregate that cannot be
gamed by trading a hard corner for an easy one — the exact failure mode every
prior cell exhibited.

### 1.4 The decision rule (pre-registered GO / NULL bar)

Let `B` = the GDN-2 control (§4) trained under the *same* compute and
convergence gate, in the *same* probe — **`B` = fair-default GDN-2 at small
scale (§4.1), `B` = CMA-ES-best GDN-2 at 1.3B (§4.2)**. A regime `R` is a
**real win** iff:

> **`JCC(R) − JCC(B) ≥ Δ*` where `Δ* = max(0.03, 2·SE_seed)`**, the gain is
> **positive on the worst corner** (i.e. `R` does not win the mean by trading
> corners), and it **holds at 1.3B** (§5).

`SE_seed` is the across-seed standard error of `JCC` measured on `B` (3 seeds).
`Δ* = 0.03` is the floor; if seed noise is larger, the noise band governs. A
result with `|JCC(R) − JCC(B)| < Δ*` is recorded as a **convergent-loss NULL
extended to optimization** — a publishable negative finding, not a failure.

**Secondary metric (tiebreak / bonus only):** genuine held-out BPB edge at 1.3B
on the averaged (schedule-free) weights, anchored exactly as
`E97_WITHIN_LAYER_SYNTHESIS.md` (never raw train loss). A BPB edge does NOT
substitute for the JCC bar; it is reported as an additional axis.

### 1.5 The convergent-loss requirement (non-negotiable)

The whole thesis lives "at convergent loss", so **every run must train to loss
convergence, not to a time budget.** Concretely:

- Train to a **fixed generous step budget** (§3.2) that the controls
  demonstrably plateau within, AND
- **Report a convergence certificate** per run: the relative loss improvement
  over the final 20% of steps (`(L_{80%} − L_{final}) / L_{80%}`). A run is
  **converged** iff this is `< 0.02` (2%). Non-converged runs are excluded from
  JCC and flagged for a longer-budget re-run.

This is what separates this line from the time-bounded LM screens (where rank =
step-count, `E97_GENERALIZATION_AUDIT`); here we compare *plateaus*, not
*progress*.

---

## 2. Shared harness

### 2.1 Trainer, precision, optimizer (identical across all probes)

- **Driver:** `experiments/expressivity_tasks/train_hybrid.py` (the same trainer
  the TTT/nlmem/CMA batteries used).
- **Precision:** `--disable_autocast` (fp32) — exact algorithmic tasks; no bf16
  gate ambiguity (the autocast hybrid-path inertness bug, `E97_FUSED_LM_KERNEL`).
- **Optimizer:** `--optimizer schedulefree` (schedule-free AdamW,
  `weight_decay=0.01`, betas `(0.9, 0.95)`), matching every prior battery.
- **GPU:** ALWAYS via the broker lease — `eval "$(scripts/gpu_lease.sh N)"`;
  the runners are broker-aware (read `CUDA_VISIBLE_DEVICES`, round-robin over
  leased ids). Never hand-pick GPUs.
- **Seeds:** `{42, 123, 456}` (3 seeds) for every arm — JCC is seed-averaged and
  `SE_seed` drives the decision band.

### 2.2 The fixed substrate cell `M`

The within-layer GDN+nonlin mixture, realized as the **`typed-gdn2`** layer
(`ndm/models/typed_head_mixture.py`), the same layer the TTT battery drove. Its
9 typed slots are:

```
TYPE_NAMES = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
              'gdn2_nonlin_shell', 'e97_raw', 'e97_delta', 'refit']
#  idx 0          1        2       3       4          5             6        7      8
```

The substrate `M` for the optimization line is the **2-side capability-complete
mixture**: a **`gdn2_recall` (recall backbone) slice + a counting/step-growth
slice** (`e97_delta`/`refit-del` for counting, `nonlin` for step-growth), with
`--gdn_allow_neg_eigval 1` so the recall side also covers `track` via the
negative along-key eigenvalue (`E97_WITHIN_LAYER_SYNTHESIS.md` Q1: gdn-neg gives
recall+track from one head type). The head allocation is set via
`--head_type_logits` (9-way, softmax→largest-remainder over heads). This cell is
**frozen**; probes vary only the training regime around it.

> **Default substrate allocation (the "house mixture"):** 50% `gdn2_recall`
> (neg-eigval on) + 25% `e97_delta` + 25% `nonlin`, at `n_heads=32`. Probes that
> need a different split MUST hold it identical across their own arms and report
> it; the **synth agent re-tests any cross-probe combination** (§6).

### 2.3 Shape (small scale) — identical to the TTT battery

`--dim 256 --n_heads 32 --n_state 32 --expansion 1.0 --mlp_ratio 2.0
--depth 4`. This is the exact shape `TTT_CAPABILITY_RESULTS.md` used, so the
control numbers there are directly reusable as cross-checks. `mlp_ratio 2.0` is
the fixed O(depth) nonlinear readout present in every arm. Param-matching
between arms within a probe is done by flexing **only `mlp_ratio`** (a fungible
param sink), never the recurrent head composition.

### 2.4 Flag surface (already in `train_hybrid.py`)

The levers map onto existing flags (verified present):

| concept | flag | notes |
|---|---|---|
| head allocation | `--head_type_logits "l0,...,l8"` | 9-way; idx per §2.2 |
| knob LR group | `--knob_lr_mult F` | separate LR for `{lam,beta,igain,gamma}_raw` knobs |
| free-gain / reflection / igain caps | `--lam_max --beta_max --igain_max` | form knobs (held fixed unless a probe owns them) |
| corner placement | `--corner_mixture` (+ spread-init in the cell) | the unified-corner allocation |
| gate | `--use_gate {0,1}` | output gate on/off |
| decay parametrization | `--decay_mode` | GDN decay schedule (`mamba`, …) |
| neg eigenvalue | `--gdn_allow_neg_eigval 1` | track via reflection |
| state nonlinearity | `--linear_state --state_activation` | tanh vs linear (the m2rnn/E88 knob) |
| delta vs raw / split | `--split_edit`, `--phi`, `--refit_has_mom`, `--refit_newton_steps` | within-frozen-substrate write knobs |

The single piece of **new plumbing** the line needs is in **opt-headlr** (§5.1):
the current `knob_lr_mult` makes ONE knob group for all heads; opt-headlr must
split it into **per-head-TYPE** groups. That is a trainer change, not an
architecture change, and is scoped inside that probe.

---

## 3. Shared capability battery

Every probe runs the **same battery, same K/p, same train length, same
eval-length grid, same step budgets**. This is what makes the four probes
aggregatable. The battery is a superset of the TTT battery so its controls
cross-check.

### 3.1 Tasks (REAL deterministic generators — no mocks, per global rule)

| corner | task (`--task`) | arg | scored? |
|---|---|---|---|
| recall | `mqar_recall` | — | ✅ |
| counting | `modular_counter` | `--K 5` | ✅ |
| counting | `dyck_depth_unbounded` | — | ✅ (counting, length-extrap) |
| counting | `anbncn_viability` | — | ✅ (counting) |
| step-growth | `modular_quadratic` | `--K 64` (=p) | ✅ |
| step-growth | `iterated_nonlinear_map` | — | ✅ |
| track | `s5_permutation` (running) | — | ✅ |
| latch | `flag_hold_recall` | — | sanity only (§1.3) |
| control | `parity` | — | sanity (all arms must solve) |
| joint | `mixed_probe` | — | reported (all corners in one stream) |

Corner aggregation: a corner's `acc_X(c)` is the **mean over its witness tasks**
(recall/track/step-growth corners average their listed tasks; counting averages
its three). The frozen ceiling `S_c` is computed the same way on the specialists.

### 3.2 Step budgets and convergence (per §1.5)

- Hard-gap tasks (`modular_counter`, `modular_quadratic`, `dyck_depth_unbounded`,
  `s5_permutation`, `iterated_nonlinear_map`, `mqar_recall`, `mixed_probe`):
  **12000 steps** (1.5× the TTT battery's 8000, to guarantee plateau for the
  convergence certificate — verify controls plateau; raise to 16000 if any
  control fails the `<2%` gate).
- Controls (`parity`, `anbncn_viability`, `flag_hold_recall`): **5000 steps**.
- `batch_size 32`, `seq_len 128`, `lr 5e-4` base (the TTT battery's lr; each
  probe may sweep lr *only as part of its own lever* and must include the 5e-4
  point).

### 3.3 Eval-length grid (length-extrapolation = the algorithm-vs-memorization signal)

Train at T=128; eval at **T ∈ {128, 256, 512}** for the hard gaps,
**{128, 256}** for controls (matching the TTT battery so numbers cross-check).
JCC is computed **per eval length and averaged**, AND reported per-length (the
extrapolation gradient is where prior wins/losses concentrated — `CMA_CAPABILITY`
§1, the gains widen at T=1024).

### 3.4 Output schema (MANDATORY — this is what makes probes aggregatable)

Every run writes `{label}.json` to the probe's `results_<probe>/` dir, and every
probe ships an `aggregate_<probe>.py` that emits one **shared-schema** row per
(arm, seed) into `results_<probe>/JCC_ROWS.jsonl`:

```json
{
  "probe": "opt-headlr",
  "arm": "headlr_recall1x_count10x",
  "regime": {"knob_lr_mult_recall": 1.0, "knob_lr_mult_count": 10.0, "...": "..."},
  "substrate": {"head_type_logits": "...", "n_heads": 32, "gdn_allow_neg_eigval": 1},
  "seed": 42,
  "params": 8000000,
  "converged": true,
  "conv_certificate": 0.004,
  "per_corner_acc": {"recall": 0.97, "counting": 0.95, "step_growth": 0.93, "track": 0.99, "latch": 1.0},
  "per_corner_ratio": {"recall": 0.98, "counting": 0.96, "step_growth": 0.99, "track": 0.99},
  "per_length_ratio": {"128": {"...": "..."}, "256": {"...": "..."}, "512": {"...": "..."}},
  "jcc_min": 0.96,
  "corners_held": 4,
  "jcc_hmean": 0.98,
  "bpb_proxy": null
}
```

The frozen ceilings `S_c` live in a single committed file
`experiments/expressivity_tasks/opt_ceilings.json` (written by the **first**
probe to run the controls, or by a tiny shared `compute_opt_ceilings.py`; §6),
so all four `aggregate_*.py` divide by the identical denominators.

---

## 4. Controls (mandatory)

**Two control regimes by scale — small probes use a *fair-default* GDN-2; the
1.3B validation uses *CMA-ES-optimized* controls (Erik's directive on `opt-1p3b`:
no hobbled baselines, best-vs-best).**

### 4.1 Small-probe controls (the 4 probes)

- **GDN-2 fair-default (`B`) — in EVERY probe.** The special case: `fla-gdn`
  with `--gdn_allow_neg_eigval 1`, default init/norm/gate/decay, same shape,
  same compute, same convergence gate. **It must be "reasonably tuned, NOT
  hobbled"** (per the `opt-headlr` task): give `B` a short base-LR sanity sweep
  (e.g. `{3e-4, 5e-4, 1e-3}`) and report its **best** LR as the baseline, so a
  regime never "wins" merely because `B` ran at the wrong LR. `B` is the
  "architecture is already enough" arm; a regime only wins if it beats this.
- **Substrate-default-trained (`B₂`) — in every probe.** The §2.2 house mixture
  trained with the *default* regime (no lever applied). Isolates the lever's
  effect from the substrate: `JCC(R) − JCC(B₂)` is the lever's pure contribution;
  `JCC(R) − JCC(B)` is the contribution vs the incumbent.
- **Kernel constraint:** probes use the **existing fused cells — NO new kernels.**
  The only new code allowed in the line is **optimizer-side plumbing** (the
  per-head-type LR param-grouping in `opt-headlr`, §5.1); that is not a kernel.

### 4.2 1.3B controls — CMA-ES-OPTIMIZED, best-vs-best (Erik, `opt-1p3b`)

At 1.3B we do **NOT** compare against default-trained controls. Both controls
are **CMA-ES-optimized at the 1.3B param budget for this application**, so each
sits at *its own* optimum — fair best-vs-best, no suboptimal-geometry artifacts:

- **CMA-ES-best GDN-2** — CMA over GDN-2's geometry/hyperparameters (dim/depth/
  heads/N/lr/decay/etc.) at ~1.3B, reusing the `cma_capability` /
  `cmaes_search_v2` machinery. This is the incumbent at *its* optimum.
- **CMA-ES-best m2rnn** (raw-write power-separation foil) — CMA over m2rnn's
  geometry at ~1.3B (`level=m2rnn`; a CMA m2rnn 1.3B config already exists at
  `hf_v03_fix_staging/m2rnn-cma-1.3b/config.json`: dim 1920, depth 21). m2rnn is
  the raw-write family where the state-nonlinearity knob is **load-bearing in the
  opposite direction** (`M2RNN_LINEAR_ABLATION.md`: linear ≫ tanh) — the honest
  "different write rule entirely" foil. If the optimized GDN+nonlin regime cannot
  separate from CMA-best m2rnn on coverage, that bounds the claim.

The optimized mixture `R*` is one arm; CMA-best GDN-2 and CMA-best m2rnn are the
control arms. **The CMA-over-controls protocol is finalized by `opt-synth`**
(§6) before `opt-1p3b` launches. The §1.4 decision rule at 1.3B therefore uses
`B = CMA-best GDN-2` (the strongest possible incumbent), making any GO verdict
maximally honest.

---

## 5. The four lever-probe designs

All four share §2 harness, §3 battery, §4 controls (`B`, `B₂`), §1 metric. Each
sweeps **exactly one lever**, holds everything else at the §2.2/§2.3 defaults,
emits the §3.4 schema, and ships a `RESULTS.md` with the JCC table + the §1.4
verdict vs `B`.

### 5.1 `opt-headlr` — per-head-type learning rates

**Hypothesis.** Recall heads and counting/nonlin heads want different LRs.
`CMA_CAPABILITY` §3 found recall-style heads want *gentle* knob refinement (the
q/k/v/o projections absorb the task) while the hard corners need their knobs
*driven*; `UNIFIED_LEARNABILITY` + the 10–20× knob-LR finding say the same. A
*single* LR forces one side to under- or over-train, so the mixture lands on a
compromise that drops a corner. **Distinct per-head-type LRs let both sides
converge onto their corners simultaneously.**

**New plumbing (scoped here).** Extend the `knob_lr_mult` param-grouping in
`train_hybrid.py` (lines ~432–452) from one knob group to **per-head-type knob
groups**: tag each `{lam,beta,igain,gamma}_raw` (and optionally the per-head
q/k/v/o projection) by the head-type slot it belongs to, and assign a group LR
multiplier per type-class. Two type-classes suffice: **recall-class**
(`gdn2_recall`) vs **compute-class** (`count`/`nonlin`/`e97_delta`/`refit`).

**Sweep (arms).** LR-multiplier ratio between the two classes, recall-class held
at base:

| arm | recall-class mult | compute-class mult |
|---|---|---|
| `headlr_uniform` (= B₂) | 1× | 1× |
| `headlr_c2` | 1× | 2× |
| `headlr_c5` | 1× | 5× |
| `headlr_c10` | 1× | 10× |
| `headlr_c20` | 1× | 20× |
| `headlr_r_slow` | 0.5× | 10× | (recall gentler, compute driven — the CMA pattern) |
| `headlr_inverted` | 10× | 1× | (falsifier: drive recall, gentle compute) |
| `B` GDN-2-default | — | — |

3 seeds × 8 arms × 10 battery tasks. The 10–20× range is mandatory (it is the
range that worked before). `headlr_inverted` is the pre-registered falsifier: if
it wins, the story is "drive recall", not "drive compute".

### 5.2 `opt-initspread` — init-spread / placement

**Hypothesis.** `SPECIALIZATION_STUDY` is unambiguous: horizontal head-type
coverage is governed by **placement, not pressure** — spread-init + knob-LR is
the only learnable form that covers all four corners; a generic-center init
collapses (0/4). This probe **tunes the placement** on the frozen substrate.

**Sweep (arms).** Vary the init/placement, knob-LR held at the §5.1 default
(klr=20 unless 5.1 has already reported a better compute-class value — synth
reconciles):

| arm | init | corner_mixture | spread |
|---|---|---|---|
| `spread_off` (≈ B₂ center) | generic center | uniform | off |
| `spread_on_uniform` | spread-init | 0.25/0.25/0.25/0.25 | on |
| `spread_on_skew` | spread-init | 0.40/0.28/0.01/0.31 (CMA difficulty-skew) | on |
| `spread_fixedpop` | hard fixed-assignment | uniform | n/a (buffers) |
| `spread_mag_lo` / `spread_mag_hi` | spread-init, ±knob spread magnitude | uniform | on |
| `B` GDN-2-default | — | — | — |

The skew arm tests `CMA_CAPABILITY`'s finding that the capability-optimal mixture
is **skewed by corner difficulty** (starve latch, feed track/nonlin), now under
the JCC `min`-aggregate (which *penalizes* starving a *scored* corner — a sharper
test than CMA's mean fitness). `fixedpop` is the robust floor from
`SPECIALIZATION_STUDY`.

### 5.3 `opt-norm` — normalization placement, gate-bias init, decay/dt init

**Hypothesis.** The basin a corner converges to is set at init by the
gate/decay/norm. Recall wants near-1 retention (decay→0, gate open); counting
wants a different decay regime. A better *joint* init of norm placement + gate
bias + decay/dt lets both corners settle without one washing out the other —
without touching the function class.

**Sweep (arms).** One axis at a time, then the best-of-each combined (combined
re-tested, not assumed additive):

| axis | arm values |
|---|---|
| norm placement | pre-norm (default) vs post-norm of the recurrent head; RMSNorm on q/k/v on/off |
| gate-bias init | sigmoid output gate init **open** (bias `+2`) vs **closed** (`−2`) vs default `0` (`--use_gate 1`) |
| decay / `A_log` / `dt_bias` init | `lam_max ∈ {1.0, 1.3, 1.59}` (the CMA winner 1.59); decay-init "slow" (retention→1) vs "fast" |
| `beta_max` (reflection depth for track) | `{1.5, 2.0, 2.75}` (CMA winner 2.75 carried track to T=1024) |
| `B`, `B₂` | controls |

Hold `head_type_logits` and LR at §2.2/§5.1 defaults. Report which single init
axis moves JCC most; the combined "best-init" arm is the candidate this probe
forwards.

### 5.4 `opt-minimal` — minimal load-bearing core for counting + recall

**Hypothesis (subtractive).** GDN-2 is a special case of the Emender
(`EMENDER_TAXONOMY.md`); which of its pieces are *necessary* to hold counting
AND recall simultaneously? Identifying the minimal cell (a) defines the smallest
substrate the other three levers should optimize, and (b) tests whether some GDN
machinery is dead weight for joint coverage (echoing the recall short-conv
caveat in `TTT_CAPABILITY` §1).

**Sweep (ablation arms).** Start from the §2.2 substrate (full) and remove one
piece at a time, scoring on the joint `{recall, counting, step-growth, track}`:

| arm | ablation |
|---|---|
| `min_full` (= B₂) | nothing removed |
| `min_no_conv` | short-conv on q/k/v removed (the FLA recall plumbing) |
| `min_no_gate` | `--use_gate 0` |
| `min_no_beta` | delta-strength β fixed (no input-dep) |
| `min_no_negeig` | `--gdn_allow_neg_eigval 0` (drops track — confirms it is load-bearing) |
| `min_linear_state` | `--linear_state 1` (tanh→linear; the m2rnn/E88 knob, both directions documented) |
| `min_no_decay_inputdep` | decay made static (not input-dependent) |
| `min_no_mlp` | `--mlp_ratio 0` (is the O(depth) readout load-bearing for counting?) |
| `B` GDN-2-default | — |

The output is a **necessity table**: for each piece, `ΔJCC` when removed. The
minimal core = the smallest arm whose `JCC ≥ JCC(min_full) − Δ*`. This probe
*also* feeds §6 — if a piece is dead weight for JCC, the 1.3B candidate drops it.

---

## 5.5 Small → 1.3B protocol

1. **Small (the 4 probes).** All at the §2.3 shape (dim 256 / 32 heads / N32 /
   depth 4 / mlp 2.0), matched compute, convergent loss, 3 seeds, full battery.
   Each probe emits `JCC_ROWS.jsonl` + `RESULTS.md` + its §1.4 verdict vs `B`.

2. **Synth selects (§6).** The synth agent reads all four `JCC_ROWS.jsonl`,
   builds the unified leaderboard, and emits **one candidate regime `R*`** =
   the per-lever winners composed onto the minimal core from `opt-minimal`.
   Because levers may interact, `R*` is **re-run at small scale once** (3 seeds,
   full battery) to confirm the composed regime's JCC before paying for 1.3B.

3. **1.3B validation.** Validate `R*` + **CMA-ES-optimized** controls at 1.3B,
   matched compute (§4.2 — best-vs-best, no hobbled baselines):
   - **Arms:** `R*` (the optimized regime on `typed-gdn2-lm`), **CMA-best GDN-2**
     (`gdn-matched-lm` / `gdn2-mlp` at its CMA-tuned geometry), **CMA-best m2rnn**
     (`level=m2rnn`). Each control sits at *its own* CMA optimum for the 1.3B
     budget; param-matched to ~1.3B. The §1.4 bar uses `B = CMA-best GDN-2`.
   - **Wiring (verified):** `LadderLM` levels exist — `typed-gdn2-lm`
     (TypedHeadMixtureLayer), `gdn-matched-lm`, `m2rnn`, `gdn2-mlp`,
     `e98-cma-lm`. The optimized regime's knobs pass through
     `layer_kwargs=` (the documented candidate-knob path).
   - **Two measurements (both required by the task):**
     1. **Capability coverage at scale** — run the §3 battery on the
        **1.3B-shaped cell** (scale the head config to the 1.3B dims, train on the
        probes), exactly as `E97_WITHIN_LAYER_SYNTHESIS.md` Q1 / the scale-pilot
        spot-check did (winner recall 0.96 / track 1.00 / count 1.00 @0.48B).
        Compute JCC at scale; the §1.4 bar must hold.
     2. **Held-out BPB** — the LM run on the committed slices
        (`COMMA_PILE_BPB` / `heldout_multislice`), averaged (schedule-free)
        weights, token-matched AND wall-clock-noted (the within-layer wall-loss turned
        on wall-clock; report both).
   - **Verdict:** GO iff `R*` clears §1.4 at 1.3B on JCC (worst-corner,
     beyond noise) AND does not regress held-out BPB vs `B`. Otherwise NULL
     (optimization does not beat the convergent-loss null) — recorded honestly.

---

## 6. Aggregation / synthesis plan

A dedicated **`opt-synth`** task (`--after` the four probes; the 1.3B validation
is a further `--after opt-synth` task) does:

1. **Freeze the ceilings first.** Before any probe scores JCC, run
   `compute_opt_ceilings.py` (the controls + specialists on the §3 battery) to
   write `opt_ceilings.json`. Every `aggregate_*.py` divides by these identical
   `S_c`. (If probes run before synth, the first probe to finish writes the
   ceilings from its `B`/specialist arms and commits the file; later probes
   consume it. The synth agent verifies all four used the same ceilings hash.)

2. **Unified leaderboard.** Concatenate the four `JCC_ROWS.jsonl`, drop
   non-converged rows (flag for re-run), seed-average, and rank every arm across
   all probes by headline `JCC = min_c r_c`. Report per-corner `r_c`, `corners-
   held`, hmean, and per-length curves. The control `B` appears in all four
   probes — its JCC across probes is a **harness-consistency check** (must agree
   within `SE_seed`; disagreement = a harness drift bug to fix before trusting
   any probe).

3. **Per-lever verdict.** For each probe, state whether its best arm clears
   §1.4 vs `B` (real win / NULL). This is the per-lever contribution.

4. **Compose `R*`.** Build the candidate regime from the per-lever winners
   (head-type LRs from opt-headlr, placement from opt-initspread, init from
   opt-norm) on the **minimal core** from opt-minimal. **Re-run `R*` once at
   small scale** (levers can interact; additivity is not assumed). If the
   composed JCC < the best single-lever JCC, fall back to the best single-lever
   regime and note the interaction.

5. **Finalize the CMA-over-controls protocol (Erik, `opt-synth`).** Before
   `opt-1p3b` launches, `opt-synth` specifies the CMA-ES search for BOTH 1.3B
   controls (§4.2): the search space (geometry/HPs per control), budget, fitness
   (capability coverage + held-out BPB at ~1.3B), and the reuse of
   `cma_capability.py` / `cmaes_search_v2.py`. This guarantees the 1.3B controls
   are best-vs-best, not default-trained.

6. **1.3B verdict & write-up.** After the 1.3B validation, write
   `paper/review/OPT_SYNTHESIS.md`: the leaderboard, the per-lever
   contributions, `R*`, the 1.3B JCC + BPB vs **CMA-best GDN-2** and **CMA-best
   m2rnn**, with token + wall-clock accounting, and the final GO/NULL call
   against §1.4. If NULL, frame it as the convergent-loss null *extended from
   architecture to optimization* — a clean, honest negative that closes the line
   (consistent with the standing nulls in §0).

### Task graph (fork-join)

```
opt-spec (this doc)
   └─ opt-headlr     ┐
   └─ opt-initspread ├─ (4 parallel probes, shared harness/battery/metric)
   └─ opt-norm       │
   └─ opt-minimal    ┘
        └─ opt-synth        (--after all 4; composes R*, re-runs R* small)
             └─ opt-1p3b    (--after opt-synth; R* + GDN-2 + m2rnn at 1.3B)
```

Each probe task's `## Validation` MUST require: (i) the §3.4 schema emitted with
frozen ceilings, (ii) the convergence certificate per run, (iii) `B` and `B₂`
controls present, (iv) a `RESULTS.md` with the §1.4 verdict vs `B`.

---

## 7. Validation checklist (this task, `opt-spec`)

- [x] **Optimization goal defined** (§1.2) and **JCC metric defined** (§1.3) —
  worst-corner ratio over frozen specialist ceilings at convergent loss, with a
  pre-registered GO/NULL bar (§1.4) and convergence requirement (§1.5).
- [x] **4 lever-probe designs specified** (§5.1–5.4) with **shared harness**
  (§2), **shared battery** (§3), **shared metrics + output schema** (§3.4).
- [x] **Controls specified** — GDN-2-default `B` + substrate-default `B₂` in
  EVERY probe (§4); **m2rnn at 1.3B** (§4, §5.5).
- [x] **small→1.3B protocol specified** (§5.5) — matched compute, capability
  coverage (battery on the scaled cell) + held-out BPB, exact `LadderLM` levels.
- [x] **Aggregation/synthesis plan written** (§6) — frozen ceilings, unified
  leaderboard, per-lever verdict, `R*` composition + small re-run, 1.3B verdict,
  fork-join task graph.
- [ ] **Doc committed** (done at end of this task).

---

## 8. Grounding (sources reconciled)

- `TTT_CAPABILITY_RESULTS.md` — the recall↔counting trade-off; convergent-loss
  null on the TTT richness lever; the battery + shape this spec reuses.
- `E97_WITHIN_LAYER_SYNTHESIS.md` — the GDN+nonlin mixture is capability-complete
  but LM-dominated (ties GDN-2 BPB at 2.6× cost → not adopted as architecture) → motivates the optimization line; the
  `gdn-neg ⇒ recall+track` mechanism; held-out BPB anchoring discipline.
- `SPECIALIZATION_STUDY_RESULTS.md` — placement (spread-init/fixedpop), not
  pressure, covers the corners → opt-initspread.
- `CMA_CAPABILITY_RESULTS.md` — knob-LR / mixture-skew / form-knob (beta_max,
  lam_max) effects → opt-headlr, opt-initspread, opt-norm; the multi-cap fitness
  this JCC sharpens (min vs mean).
- `M2RNN_LINEAR_ABLATION.md` — the raw-write foil and the state-nonlinearity
  knob's opposite-direction effect → the 1.3B m2rnn control + opt-minimal's
  `linear_state` arm.
- `EMENDER_TAXONOMY.md` — GDN-2 = {decay,reflect}×linear special case → the
  function-class boundary opt-minimal ablates within.
- Harness: `experiments/expressivity_tasks/train_hybrid.py` (flags verified),
  `tasks/` (generators verified), `ndm/models/typed_head_mixture.py` (9 slots),
  `ndm/models/ladder_lm.py` (levels verified), `scripts/gpu_lease.sh` (broker).

*Design deliverable for `opt-spec`. No GPU used. `paper/main.typ` untouched.*
