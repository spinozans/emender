# HANDOFF — Emender / E97 architecture research → 100B scaleout

**Audience:** a fresh research + engineering team with **zero prior context** —
new models, new systems, no knowledge of this project's tooling or history. This
document is **self-contained**: it defines the architecture from scratch, states
every validated finding with committed provenance (file paths + git commits), and
lays out the scaleout plan, the artifact map, the methodology guardrails, and the
open questions. It is a synthesis of committed measured artifacts, written to be
**calibrated** — neither overselling nor underselling the results.

> **One-paragraph orientation.** This project studies a recurrent-RNN cell called
> the **Emender** (lead instance tag **E97**), a *gated-delta linear/nonlinear
> recurrence* that generalizes the linear "Gated DeltaNet-2" (GDN-2) baseline. The
> headline measured results, fairly compared at 1.3B parameters: (1) on **language-model
> loss**, the lead config `emender-mlp` **ties-or-leads** `gdn2-mlp` — it is ahead on
> both primary metrics but inside the single-seed noise band; (2) on **capability**,
> there is a **real, robust temporal class separation** — the nonlinear cell
> extrapolates a class of iterated maps where the linear cell memorizes-then-collapses
> — but it is **specific to one task family (`modular_quadratic`)**, *not* a general
> law; (3) on **throughput**, the two cells are a **tie** at 1.3B (an earlier
> 1.26–1.56× speedup did not survive to scale). The plan is to train a 1.3B
> `emender-mlp` seed model to **100B tokens** locally and track whether the
> modquad-specific capability surfaces as *measurable real-LM* capability divergence
> from `gdn2-mlp`, then hand the seed to a Frontier-class HPC scaleout.

---

## Table of contents

1. [Architecture: the Emender / E97 cell + MLP, from scratch](#1-architecture)
2. [Validated findings (calibrated, with provenance)](#2-validated-findings)
3. [Scaleout + frontier plan (primitives, gaps, phases)](#3-scaleout--frontier)
4. [Artifact map: where everything lives](#4-artifact-map)
5. [Methodology guardrails (avoiding the documented failure modes)](#5-methodology-guardrails)
6. [Open questions / immediate next steps](#6-open-questions--immediate-next)

---

## 1. Architecture

### 1.1 The family: a dynamics taxonomy of head-types

The Emender is a **mixture of recurrent "head-types"** inside a single layer. Each
head-type is one per-token state-update map of the general form

```
S_t = f( A_t · S_{t-1} + B_t )
```

where `S_t` is the head's matrix state, `A_t` is an input-dependent state transition,
`B_t` is the input-driven write, and `f` is an optional pointwise nonlinearity. The
taxonomy is read from the **dynamics**, on **two independent axes** (canonical
reference: `paper/review/EMENDER_TAXONOMY.md`; paper anchor `paper/main.typ`,
subsection *"A dynamics taxonomy of head-types: eigenvalue placement × saturation"*):

**Axis 1 — eigenvalue placement in the complex unit disk** (where the along-key
eigenvalue of `A_t` sits):

| Placement | Name | Dynamics |
|---|---|---|
| real, positive | `decay` | stored value fades toward zero (vanilla GDN) |
| real, negative | `reflect` | sign flips each step — the tracking lever (lets a recurrence track non-solvable group structure such as S₅) |
| complex pair | `rot` | rotation / oscillation of the stored value |

**Axis 2 — state map: linear vs saturated:**

| State map | Suffix | Canonical fn | Dynamics |
|---|---|---|---|
| linear | *(none)* | identity | unbounded along eigendirections |
| + saturation | `-nonlin` | `hardtanh` (or smooth `tanh`) | latches a driven slot → finite-state regime |

The named cells form a grid: `decay`, `reflect`, `nonlin` (= **E97**, the saturated
delta-correcting head), `rot` (= complex-eigenvalue head), `rot-nonlin` (reserved).

**Key structural claim — GDN-2 is a special case of the Emender.** Gated DeltaNet-2,
the linear-recurrent baseline, is *exactly* the restriction of the Emender to the
**real diameter of the unit disk with no saturation**: the `{decay, reflect} × linear`
sub-grid. The Emender generalizes GDN-2 along two axes: real diameter → full unit disk
(adds `rot`), and linear state → saturated state (adds `nonlin`). The Emender layer is
the within-layer pool over this grid; **GDN-2 is its linear, real-axis corner.** This
matters for fair comparison: any Emender-vs-GDN-2 result is a comparison of a model to
its own restriction.

> **Naming note.** The experiment tags **E97 / E88 / E98 / E99** are retired as
> architecture names and kept *only* as historical run identifiers. E97 = the `nonlin`
> head (real × saturation). E88 is the original 1.3B instance tag. Use the dynamics
> names (`decay`/`reflect`/`nonlin`/`rot`) in new work.

### 1.2 The E97 cell — split-edit gated-delta recurrence with per-step nonlinearity

The lead head-type is the **E97 split-edit delta** cell. It is a gated-delta-rule
recurrence with *separate* gates on the key (erase/read) axis and the value (write)
axis, plus a per-step bounded nonlinearity on the state. Per (batch, head), with state
`S` of shape `[N, V]` (canonical source: kernel docstring in
`ndm/triton/e97_chunked.py`; cell in `ndm/models/e88_fla_hybrid.py`,
`use_split_edit=True`):

```
read_key_t  = e_t ⊙ k_t                      # erase/read gate on the key axis,  [N]
write_val_t = w_t ⊙ v_t                      # value-write gate,                 [V]
delta_t     = write_val_t − S_{t-1}^T read_key_t          # delta correction,    [V]
S_t         = f( decay_t · S_{t-1} + k_t · delta_t^T )    # state update,     [N,V]
out_t       = S_t^T q_t                                   # readout,             [V]
```

- **`decay_t`** is a Mamba2-style input-dependent exponential decay
  (`g = −exp(A_log) · softplus(a_proj(x) + dt_bias)`), giving the along-key eigenvalue.
- **Short depthwise convolutions + SiLU** are applied to `k, v, q` after projection;
  `q, k` are L2-normalized (FLA-GatedDeltaNet design elements).
- **The per-step nonlinearity `f`** is the saturation axis. The paper writes the cell
  as `S ← tanh(d·S + k·(silu(v) − S^T k)^T)`. Three ablation variants isolate the
  pieces, all byte-identical code except for the marked change:
  - **`emender` / `nonlin` / E97** (lead): `f = tanh`, delta-correcting. This is the
    capability-retaining cell.
  - **`e97-linear`**: `f = identity` (drop the per-step nonlinearity). Isolates the
    causal effect of the nonlinearity — same kernel, only `tanh` vs `identity` differs.
  - **`e97-raw`**: drop the delta correction, write the gated value directly.

When the per-step nonlinearity is **on**, the recurrence is genuinely nonlinear in time
and must run a sequential scan; when **off** (`e97-linear`), the recurrence is an
*affine asymmetric gated-delta* form `S_t = (decay_t·I − k_t·read_key_t^T)·S_{t-1} +
k_t·write_val_t^T`, which admits a **chunked-parallel** kernel (intra-chunk matmuls,
recurrent state threaded across chunks — the same trick FLA uses for GDN-2). The fused
Triton kernel `ndm/triton/e97_chunked.py` implements the chunked form; the sequential
fused path lives in `ndm/models/e88_fused.py`.

### 1.3 The lead configuration: `emender-mlp`

The seed config handed to scaleout:

```
emender-mlp = E97 split-edit DELTA (e88_raw_write=0, delta-correcting; NOT raw-write)
            + bias-free LLaMA-style SwiGLU MLP  (w3(silu(w1 x) · w2 x))
geometry:   dim 1792 · n_heads 216 · n_state 32 · depth 11 · mlp_ratio 2.2623
params:     1,286,589,072  (measured, byte-identical to its CMA search)
```

The SwiGLU MLP is added by `MixerMLPWrapper` / `SwiGLUMLP` in
`ndm/models/ladder_lm.py` (post-mixer RMSNorm + MLP), making it a fair counterpart of
the `gdn2-mlp` control (GDN-2 mixer + the *same* SwiGLU MLP). **The MLP matters:** the
MLP was worth ~0.42 search-loss to GDN-2; the canonical fair fight is **MLP-vs-MLP**
(`emender-mlp` vs `gdn2-mlp`), not naked-cell-vs-MLP (see §5).

The primary control:

```
gdn2-mlp = GDN-2 mixer (the linear {decay,reflect} corner) + SwiGLU MLP
geometry: dim 2176 · n_heads 30 · depth 12 · mlp_ratio 3.2587 · use_conv
params:   1,286,713,448  (measured)
```

Both configs are built **byte-identically** to their CMA searches via
`scripts/cmaes_search_v2.build_train_command` → `train.py` (`--level E97` / `--level
gdn2-mlp`). All param counts verified against each source's recorded `actual_params`.

---

## 2. Validated findings

All three findings below are from committed, measured artifacts. The calibration is
deliberate: the project's own post-mortem (`docs/RESEARCH_ASTRAY_POSTMORTEM.md`)
documents ~10 days of comparisons that were quietly rigged *against* the Emender and
had to be corrected by the PI — so the discipline here is to read the **primary
measured metric head-to-head** and to state scope honestly in both directions.

### 2.1 LM loss → `emender-mlp` TIES-OR-LEADS `gdn2-mlp`

**Provenance:** `experiments/lb_compare_20260613/{LEADERBOARD,REPRODUCTION}.md`
(commit `8acd929`, corrections `ece1b16`). Apples-to-apples: all 5 CMA-best 1.3B
models at their *own* found geometry, same protocol — pile.txt seed42, 15-min train
budget (matching the CMA search), bf16 uniform + fused kernels, p50k_base, ctx 2048,
schedule-free AdamW. Held-out = ONE fixed disjoint pile.txt-tail slice (131,072 scored
tokens, byte-identical for every model). BPB = (CE_nats/ln2)/3.878 bytes/token.

| `emender-mlp` vs `gdn2-mlp` | `emender-mlp` | `gdn2-mlp` | winner |
|---|---:|---:|---|
| CMA search avg-loss | **5.8606** | 5.8949 | emender-mlp (−0.034) |
| held-out BPB, **non-avg** (primary basis) | **2.0911** | 2.1013 | emender-mlp (−0.010) |
| held-out BPB, averaged (inferior basis) | 2.1783 | 2.1550 | gdn2-mlp (+0.023) |

**Calibrated reading.** `emender-mlp` is ahead on **both primary metrics** (the search
avg-loss and the non-averaged held-out BPB, which is the basis consistent with the
search). It loses only on the schedule-free *averaged-weights* basis — which the run
itself flags as the inferior/artifact basis (at a 15-min budget the polyak average is
uniformly worse than the final weights, by an architecture-dependent margin up to ~0.70
BPB for mixer-only cells). The margins (0.034 search, 0.010 held-out) sit inside the
~0.088 single-seed / 15-min noise band. **Honest statement: `emender-mlp`
ties-or-beats `gdn2-mlp` and is never clearly worse; on both primary metrics it is the
one ahead.** Not "worse"; not "gdn2 wins."

> ⚠️ **Provenance caveat — read the corrections, not the auto-verdict.** The
> auto-generated VERDICT block inside `LEADERBOARD.md` says "clean NO-GO / gdn2-mlp best
> all-around." That verdict is **superseded** by the CORRECTIONS at the top of both
> files (and by `RESEARCH_ASTRAY_POSTMORTEM.md` Addendum): it (a) mislabeled
> `emender-mlp` as raw-write when it is the delta cell, (b) leaned on the averaged
> (artifact) ordering, and (c) used a grok-suppressed separator battery (see §2.2). The
> corrected, calibrated reading is the table above. **Follow-up owed:** multi-seed BPB to
> push the margin out of the noise band.

This LM tie is **robust across architecture, optimization, and scale** — it is the
"convergent-loss null": many different cells converge to nearly the same LM loss at
matched compute (consistent with `emender-real-1p3b`, `emender-cap-sweep`, `opt-1p3b`).
The fair MLP-vs-MLP fight is the one that leans Emender.

### 2.2 Capability → a REAL temporal class separation, but `modular_quadratic`-specific

**Provenance:** `experiments/grok_symmetric_width/{RESULTS,CONFIRM}.md` (commits
`d489955`, `ce7ff39`). This is the load-bearing capability result, and it has two
parts — a strong positive and a sharp scope limit.

**The positive (real, seed-robust, mechanistically isolated).** On `modular_quadratic`
(`x_t = (x_{t-1}² + c_t) mod p`, per-position supervision), trained at sequence length
T=128 and evaluated by **length-extrapolation** out to T=4096:

- the **nonlinear** cell `e97` is **length-invariant** (e.g. p=256/dim1024: 0.982 @
  T=128 → 0.981 @ T=4096);
- both **linear-state** arms (`e97-linear` and `gdn2`) **memorize the train length and
  collapse** toward baseline at far T, **even at maximum width (dim 1024)** (e.g. same
  cell: e97-lin 0.619, gdn2 0.560 @ T=4096);
- the gap is **positive in every decisive cell** (e97−gdn2 = +0.16 … +0.32 @ T=4096,
  8 seeds) and **grows with the modulus p**;
- **`e97` vs `e97-linear` is byte-identical code on the same fused kernel — the only
  difference is the per-step state map (`tanh` vs `identity`).** This isolates the
  **per-step nonlinearity-in-time** as the causal lever — not capacity, depth, or the
  specific linear architecture.

This corrects a predecessor false-negative: an earlier run (`grok_highp_temporal`)
concluded "width closes the gap → it's capacity, not a class," but it (a) widened only
the linear arms and (b) scored at the *train* length. A high-capacity linear+MLP model
*can memorize any finite instance* — "more width → groks the finite test set" is
capacity buying memorization, which **length-extrapolation defeats**. The symmetric-width
control (all arms widened, scored at far T) flips it: width buys train-length
memorization, not extrapolation.

**The scope limit (honest, load-bearing negative).** Tested on two *other* task
families (`grok_symmetric_width/CONFIRM.md`), the signature **does not replicate**:

- **`iterated_nonlinear_map`** (logistic `h_t = a_t·h_{t-1}(1−h_{t-1})`, binned): 0/24
  grok; all arms plateau flat across T. The map is **contractive (fading memory)** — no
  long-memory to memorize-then-fail, so the per-step nonlinearity is not load-bearing.
  No separation.
- **`anbncn_viability`** (a^n b^n c^n, count comparisons): 24/24 fit; all arms decay
  *together* with length; the e97−gdn2 gap is tiny and **sign-flipping**. Counting is
  additive accumulation where linear cumulative state extrapolates *as well as* bounded
  nonlinear state.

**Mechanism — the separation needs BOTH conditions, which only `modular_quadratic`
satisfies:** (1) a per-step state-nonlinearity the linear arm **cannot represent**
(x² mod p is non-invertible; counting and a fading map are not), **and** (2)
**non-contractive, full-precision long memory** so the failure-to-represent
*compounds* with length instead of washing out. Remove either and the separation
vanishes.

> **Calibrated bottom line:** claim a *robust, mechanism-specific* class separation on
> **iterated-map / modular-arithmetic tasks** (8-seed confirmed, throughput-favorable at
> grok scale), **NOT** a universal "nonlinear-in-time beats linear" law. Overclaiming a
> general law is falsified by the two non-replications above. Note also: the **separator
> battery inside `lb_compare`** (`modular_counter`, etc.) is **grok-suppressed** (LR
> pinned, no weight-decay sweep, short training) and `modular_counter` is *bounded*
> finite-state counting where linear-state is *expected* to win — so that battery does
> **not** test (nor refute) the Emender's claim. Capability proven only at-grok, on
> modular_quadratic, via length-extrapolation.

### 2.3 Throughput → TIE at 1.3B (the grok-scale speedup did NOT survive to scale)

**Provenance:** `experiments/preflight_100b/RESULTS.md` (commit `0e914a3`); grok-scale
microbenchmark in `experiments/grok_symmetric_width/`.

At the literal `emender-mlp` geometry (dim1792/nh216/ns32/dep11/mlp2.26), measured on a
leased RTX 6000 Ada, identical conditions:

| arm | batch | per-GPU tok/s | global tok/s (7-GPU) | peak mem/GPU |
|---|---|---:|---:|---:|
| emender-mlp | bs4 (matched) | 3,211 | 22,474 | 28,942 MB |
| gdn2-mlp | bs4 (matched) | 3,290 | 23,034 | 35,715 MB |
| emender-mlp | bs6 (its DDP max) | 4,470 | 31,291 | 38,497 MB |

- **Matched-conditions per-token throughput ratio emender/gdn2 = 0.976× → a TIE.**
- The earlier **1.26–1.56× speedup** measured at grok scale (small dims, isolated
  1-GPU) **is NOT reproduced at 1.3B — REFUTED**, consistent with the post-mortem warning
  that prior E97 throughput claims were unreliable.
- The Emender's *real* edge at scale is **memory**: 28.9 GB vs 35.7 GB at bs4, so it
  fits bs6 where gdn2 OOMs → a ~1.36× *aggregate* throughput advantage, but that is a
  batch-size/memory effect, **not** a per-token kernel speedup. State the honest version.

---

## 3. Scaleout + frontier

### 3.1 Local seed run — 1.3B `emender-mlp` → 100B tokens

**Goal (provenance: `docs/SCALE_PLAN.md`, commit `c0cdf28`):** stay at **1.3B
parameters** and push the **token count** to **100B** (~77 tokens/param, ~4×
Chinchilla — the *emergence* regime), tracking the capability-eval suite (held-out BPB
+ length-extrapolation + the algorithmic battery) on checkpoints **as the token count
grows**. The central experiment: **does `emender-mlp` diverge from `gdn2-mlp` at high
token count** — i.e. does the modquad-specific capability (§2.2) surface as *measurable
real-LM* capability? The 3B/7B parameter scale-up is a separate, compute-gated, later
phase.

**The hardware reality (provenance: `experiments/preflight_100b/RESULTS.md`).** The
local box is **8× RTX 6000 Ada (49 GB, PCIe, NO NVLink)**. Measured scaling:

| configuration | aggregate tok/s | efficiency vs 1-GPU |
|---|---:|---:|
| 1× GPU (emender bs6) | 8,600 | 100% (baseline) |
| **7× GPU vanilla DDP** (bs6) | **31,291** | **52%** |
| 7× independent processes (no DDP) | ~62,000 | ~103% (near-linear) |

Vanilla per-step DDP **wastes ~48%** of the GPUs: the bottleneck is the per-step
all-reduce of the 1.29B bf16 gradient (2.6 GB) over PCIe with no NVLink — *not* CPU,
NVMe, or power (7 independent procs scale near-linearly). `grad_accum` does not fix it.
Projected wall-clock: **vanilla DDP → ~37 days to 100B**; the independent ceiling → ~19
days.

**DiLoCo periodic-sync — throughput GO, loss-parity NO-GO (the key nuance).**
DiLoCo (Distributed Low-Communication training) trains each rank's replica
independently and averages model **weights** only every K local steps, replacing the
per-step all-reduce. It was implemented in `train.py` (`--diloco --diloco_k ...`,
opt-in; single-GPU path byte-identical) and measured:

- **Throughput: GO.** `experiments/diloco_100b/RESULTS.md` (commit `7997419`): K∈{250,
  500} reaches **57.7–57.9k global tok/s = ~1.85× DDP**, recovering ~98% of the
  independent ceiling; the periodic merge costs only ~2.1 s amortized. Merge correctness
  verified (ScheduleFree y-mode swap; gloo unit test; bit-faithful consensus checkpoint).
  → **~20 days to 100B**, divergence-free.
- **Loss-parity: NO-GO (as a parity path).** `experiments/diloco_100b/longhorizon/RESULTS.md`
  (commit `92a153f`): at **matched tokens**, plain local-SGD DiLoCo (`outer_beta=0`)
  **lags synchronous DDP by a persistent ~0.44–0.47 BPB** in the healthy regime, and the
  gap **does not close** within ~129M tokens. Outer-momentum sweeps (β=0.5/0.9 × lr=0.7/1.0)
  and small-K (K=50) do **not** help. A **2-GPU-island hybrid** (per-step DDP within 3
  islands of 2 GPUs + DiLoCo across) is the only variant that meaningfully closes the
  gap — it **halves** the penalty to **+0.25–0.32 BPB** at **45k tok/s = 1.44× DDP** —
  but still does not reach DDP parity.
- **The dominant blocker is the training RECIPE, not parallelism.** Run to 215M tokens,
  held-out BPB **collapses for BOTH DDP and DiLoCo** (DDP bottoms at 1.571 @ 64.5M then
  climbs to 3.234 @ 215M) under the constant CMA-tuned LR (1.007e-3, `warmup_steps=0`,
  no decay). The apparent β=0 "convergence" at 215M is **mutual collapse, not parity**.
  **No path — DDP or DiLoCo — reaches a usable BPB at 100B with the current recipe.**

**→ Immediate prerequisite before any 100B run: fix the training recipe** (add warmup +
LR decay, and/or lower the constant LR), then **re-evaluate DiLoCo-vs-DDP parity against
a non-degrading baseline.** Filed as follow-up `fix-long-horizon`. Current
recommendation: on this 7-GPU box use **per-step DDP** for the seed run (exact, no
sample-efficiency penalty) once the recipe is fixed, unless the re-evaluation shows
DiLoCo parity; if wall-clock is the hard constraint, the **hybrid** (+0.25–0.32 BPB at
1.44× DDP) is the best trade found. (Hybrid requires `NCCL_P2P_DISABLE=1` + sequential
subgroup-comm warmup on this no-NVLink box — P2P over PCIe deadlocks 2-rank subgroups.)

### 3.2 Frontier — hierarchical DiLoCo-OUTER × 3D/DP-INNER over RCCL/Slingshot

The frontier target is an OLCF-Frontier-class allocation (AMD MI250X, ~20K node-hours;
e.g. 64 nodes × 24 h). **The plan is NOT plain DDP across all ranks** — that forces an
enormous global batch (64 nodes × 8 GCDs × 2048 = ~1.05M tokens/update even at per-GPU
bs=1, risky for learning). Instead use **hierarchical ScheduleFree-DiLoCo**: normal
synchronous training (DDP, ideally 1 node = 8 GCDs) *inside* each island, periodic
weight averaging *between* islands. Even on a fast interconnect, DiLoCo's
straggler/fault-tolerance/communication benefits **compound at ~1000-node scale**.
(Design: `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`; systems notes:
`docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`.)

**Primitives that EXIST (cite, build on — do not reinvent):**

| Primitive | Reference | Role |
|---|---|---|
| DiLoCo (low-communication distributed training) | arXiv:2311.08105 | the outer periodic-averaging method |
| OpenDiLoCo (open reproduction, global scale) | arXiv:2407.07852 | scaled open implementation |
| Decoupled momentum / outer-optimizer variants | arXiv:2604.21428 | outer-optimizer design space |
| Eager updates (reduce DiLoCo comm stalls) | arXiv:2502.12996 | overlapping comm with compute |
| Scaling laws for DiLoCo | arXiv:2503.09799 | how the method scales with model/island count |
| Pier (DiLoCo systems / partitioning) | arXiv:2511.17849 | systems-level DiLoCo |
| Frontier 3D-parallelism + ROCm precedent (Dash et al.) | arXiv:2312.12705 | optimizing distributed LLM training on Frontier/MI250X |
| Low-bandwidth model partitioning | arXiv:2501.04266 | partitioning for slow interconnects |

Supporting systems facts (from `FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`): use
**ROCm/Megatron-LM** (`github.com/ROCm/Megatron-LM`, NOT Microsoft Megatron-DeepSpeed);
the **`aws-ofi-rccl` plugin MUST be built + LD_PRELOAD'd** (default RCCL uses TCP/IP on
Slingshot — catastrophic); pre-build DeepSpeed JIT ops at image time (they fail on
ROCm); checkpoint with PyTorch DCP `SHARDED_STATE_DICT`, tiered NVMe→Orion-Lustre; watch
the known `torch.compile + bf16 + ROCm` NaN gotcha. For non-ParaRNN parallelism, the
recommended inner shape is TP=8 intra-node, PP=4 (interleaved 1F1B), DP cross-node
(ZeRO-1). Optional high-risk axis: **ParaRNN** (arXiv:2510.21450) parallelizes nonlinear
RNNs across the sequence via Newton's method — *if* it converges on the E97 matrix-state
recurrence (untested; matrix state makes the Newton block expensive), sequence
parallelism becomes available; prototype before betting budget on it.

**The 3 GAPS to build/validate (unpublished — this is the research+engineering work):**

- **(a) The composition is unpublished.** DiLoCo-outer over RCCL with a **3D-parallel
  inner** (TP×PP×DP within each island) has not been published. The primitives exist
  separately; composing them — and validating throughput, fault-tolerance, and learning
  parity at multi-node scale — is novel work.
- **(b) Schedule-Free AdamW as the DiLoCo INNER optimizer is unstudied.** Published
  DiLoCo uses AdamW-inner + SGD/Nesterov-outer. This project's inner optimizer is
  **Schedule-Free AdamW**, whose internal train/eval (x/y/z) weight semantics interact
  with the merge in ways not studied in the literature. The local parity test is the
  experiment **currently underway** (`diloco_100b/longhorizon/`); the merge target must
  be the eval (x) weights, with the base sequence z reset to consensus (see
  `SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md` §ScheduleFree Interaction).
- **(c) The E97 fused split-edit kernel → ROCm/MI250X port (biggest eng risk, hard
  prereq).** The fused E97 kernel is currently CUDA/Triton and **bf16-only**, and has had
  chunked fp32-overflow / NaN edge cases on CUDA already (see
  `complex-eig-chunked-overflow`, `fuse-2kernel` notes). It **must** be ported to
  ROCm/MI250X and **parity-verified** (≤ few e-3 in bf16, T∈{128,512,1024,2048}) on a
  single MI250X GCD before any multi-node run. There is **no fp32 safety net** for the
  fused path. This is the gating risk for choosing `emender-mlp` (vs the FLA-mature
  `gdn2-mlp`) as the frontier arm.

### 3.3 Phased, risk-ordered plan

- **P0 — Local seed (now).** Fix the training recipe (`fix-long-horizon`: warmup + LR
  decay), validate Schedule-Free × DiLoCo parity against a non-degrading baseline, and
  produce the **100B `emender-mlp` seed checkpoint** (+ `gdn2-mlp` control) with
  capability evals at a fixed token cadence. This both validates gap (b) and produces the
  seed handed to frontier.
- **P1 — ROCm kernel port (de-risk EARLY).** Port the E97 fused split-edit kernel to
  ROCm/MI250X and parity-verify on a single GCD. This is gap (c) and a hard prereq — do
  it *before* committing HPC allocation. Keep `gdn2-mlp` (mature ROCm via FLA) as the
  fallback arm.
- **P2 — HPC DiLoCo over RCCL + small-multinode validation.** Stand up hierarchical
  DiLoCo-outer × 3D-inner (gap a) on a small node count (e.g. 16 nodes); validate
  throughput, learning parity at matched tokens, and fault-tolerance (straggler/failed-
  island handling) before scaling.
- **P3 — Scale.** Scale node count (64 → larger) and, separately and compute-gated,
  parameters (1.3B → 3B → 7B → 13B) with token scaling, tracking the scaling law and
  capability divergence.

---

## 4. Artifact map

| What | Where | Notes |
|---|---|---|
| **CMA-best 1.3B configs** | `experiments/lb_compare_20260613/REPRODUCTION.md` | exact geometries for all 5 arms; param counts verified |
| **LM leaderboard (measured)** | `experiments/lb_compare_20260613/{LEADERBOARD.md, bpb_results.json, sep_results.json}` | read the CORRECTIONS, not the auto-verdict |
| **Capability separation (modquad)** | `experiments/grok_symmetric_width/{RESULTS.md, CONFIRM.md, *.json}` | 216 + 96/24/24 runs; length-extrapolation to T=4096 |
| **Preflight throughput / DDP scaling** | `experiments/preflight_100b/RESULTS.md`, `run_ddp.sh`, `ckpt_roundtrip.py` | the 0.976× tie + 52% DDP scaling |
| **DiLoCo throughput + merge correctness** | `experiments/diloco_100b/RESULTS.md`, `run_diloco.sh`, `tests/test_diloco_merge.py` | 1.85× DDP, y-mode merge verified |
| **DiLoCo long-horizon loss-parity** | `experiments/diloco_100b/longhorizon/RESULTS.md`, `phase{1,2,2b_hybrid}.sh`, `tests/test_diloco_hybrid.py` | the NO-GO + recipe-collapse finding |
| **Training data** | `/mnt/nvme2n1/erikg/pile.txt` (symlink `/home/erikg/elman/data/pile.txt`); `/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt` | pile = the <1bpb gate corpus; comma-pile = frontier corpus + cross-eval (both verified present) |
| **The cell (source)** | `ndm/models/e88_fla_hybrid.py` (`use_split_edit=True`); ablations via `--e88_raw_write` / `--linear_state` | E97 split-edit delta |
| **Fused E97 kernel** | `ndm/triton/e97_chunked.py` (chunked), `ndm/models/e88_fused.py` (sequential); tests `tests/test_e97_chunked.py` | bf16-only; the ROCm port target |
| **Model wiring (SwiGLU MLP, GDN-2 control)** | `ndm/models/ladder_lm.py` (`SwiGLUMLP`, `MixerMLPWrapper`, `gdn2-mlp`) | the fair MLP counterpart |
| **Trainer (DDP + DiLoCo opt-in)** | `train.py` (`--level`, `--use_triton`, `--diloco`, `--diloco_k/outer_lr/outer_beta`, `--heldout_tensor`, `--data_rank/--data_world_size`) | single-GPU path byte-identical |
| **CMA driver (standard HPO)** | `scripts/cmaes_search_v2.py` (`build_train_command`, search spaces) | use this, full geometry, ≥96 evals — not bespoke searches |
| **Eval harnesses** | separators: `experiments/expressivity_tasks/train_hybrid.py`, `experiments/lb_compare_20260613/run_separators.py`; grok: `experiments/grok_symmetric_width/train_grok.py`; held-out BPB: `experiments/*/run_bpb.py`, `build_heldout_tensor.py` | |
| **Paper** | `paper/main.typ` (build: `bash paper/build.sh`); taxonomy fig anchor `@fig_taxonomy` | |
| **Key docs** | `docs/SCALE_PLAN.md`, `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`, `docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`, `docs/RESEARCH_ASTRAY_POSTMORTEM.md`, `paper/review/EMENDER_TAXONOMY.md` | |
| **GPU lease broker (this box)** | `scripts/gpu_lease.sh` (`eval "$(scripts/gpu_lease.sh N)"`) | ALWAYS lease before touching a GPU; no central allocator, 8 shared GPUs |

Provenance commits (most recent first): `92a153f` diloco-loss-parity · `7997419`
diloco-periodic · `0e914a3` preflight-100b · `c0cdf28` scale-plan lock + postmortem
addendum · `ce7ff39` grok-confirm · `d489955` grok-symmetric-width · `8acd929` /
`ece1b16` lb-compare + corrections · `baf47f2` research-astray post-mortem.

---

## 5. Methodology guardrails

`docs/RESEARCH_ASTRAY_POSTMORTEM.md` is required reading: it documents ~10 days in which
**every "the Emender loses / it's a null" verdict came from a comparison quietly rigged
against the Emender**, each corrected (always by the human PI) toward a positive, with
the final fair test flipping to a win. The errors were *one-directional* — a neutral
error process would scatter. The reflex recurred *one message after* the post-mortem was
written. So the guardrails are not optional:

1. **Trust measured artifacts over synthesis.** No verdict until the committed measured
   data is on screen. Read the **primary metric head-to-head** before stating any
   conclusion; do not inherit a prior agent's verdict without checking the basis it was
   computed on (e.g. averaged vs non-averaged weights — that flips the lb-compare
   ranking).
2. **Test class-separation at CONSTRAINED capacity + LENGTH-EXTRAPOLATION.** Capacity
   buys *memorization*, which masquerades as capability if you score at the train length.
   The real test is whether a model holds as T grows past the trained length. (This is
   exactly what flipped the `grok_highp_temporal` false-negative — §2.2.)
3. **Fair = best-vs-best with documented, symmetric search space, budget, precision,
   MLP, and geometry — fixed *before* drawing any conclusion.** Past failures: searching
   the Emender only over a 2-D mixture axis while giving GDN-2 the full geometry; running
   the Emender in fp32 while controls ran bf16 (a 4.3× token deficit presented as a
   "loss"); comparing GDN-2-with-MLP to Emender-without-MLP; under-searching (64 evals vs
   the leaderboard's 104+); pinning n_state on *all* models when instructed only one.
4. **Test capability AT GROK.** Use AdamW + a weight-decay sweep + train long enough to
   grok (10–100× the steps memorization needs). Grokking is wd-driven; the
   short-schedule, no-wd-sweep, schedule-free batteries (e.g. the `lb_compare`
   separators) are grok-suppressed and do not test the claim. Use the **real separators**
   (iterated maps / unbounded counting at length-extrapolation), not finite-state proxies
   where linear is *expected* to win.
5. **No premature "done."** A task is done only when its run process exits and the result
   is committed — never seconds after launching a multi-hour job (that corrupted runs by
   unblocking concurrent GPU-stacking).
6. **Liveness = generation/step count advancing, not GPU-util snapshots.** A healthy
   search was once killed off a single util snapshot while it was progressing fine at gen
   6/13.
7. **Calibrated honesty — both directions.** Do not overclaim (no "general law" from one
   task family) and do not reflexively reach for the tidy negative or caveat-spray after
   overclaiming. State the tie as a tie, the lead as a lead, the scope limit as a scope
   limit.

---

## 6. Open questions / immediate next

**The central open question.** Does the `modular_quadratic`-specific capability
separation (§2.2) — proven on a synthetic iterated map via length-extrapolation —
surface as **measurable, real-LM capability** at 100B tokens, as a **divergence between
`emender-mlp` and `gdn2-mlp`**? The LM loss is a tie (§2.1) and the capability win is
narrow and task-specific (§2.2); whether it manifests in language at the emergence
regime (~77 tok/param) is the experiment the 100B seed run is designed to answer. Track
held-out BPB + length-extrapolation + the algorithmic battery on checkpoints at a fixed
token cadence throughout.

**Immediate prerequisite (blocks the 100B run).** **Fix the training recipe** — the
constant CMA-tuned LR (1.007e-3, `warmup_steps=0`, no decay) **collapses held-out BPB
for BOTH DDP and DiLoCo past ~64M tokens** (§3.1). No path reaches a usable BPB at 100B
until this is fixed (add warmup + LR decay / lower LR; follow-up `fix-long-horizon`).
Re-evaluate DiLoCo-vs-DDP parity against the non-degrading baseline afterward.

**The 3 scaleout gaps (§3.2), risk-ordered:**
- **(c) ROCm/MI250X port of the bf16-only fused E97 kernel** — the long pole and hard
  prereq; parity-verify on a single GCD before any HPC allocation. `gdn2-mlp` is the
  fallback arm if the port stalls or the throughput edge evaporates.
- **(b) Schedule-Free AdamW as the DiLoCo inner optimizer** — parity test underway
  locally; must be settled (against a fixed recipe) before frontier.
- **(a) The hierarchical composition** (DiLoCo-outer over RCCL × 3D-parallel inner) —
  unpublished; validate throughput/parity/fault-tolerance at small node count (P2) before
  scaling.

**Smaller follow-ups owed:** multi-seed held-out BPB to push the `emender-mlp` vs
`gdn2-mlp` margin out of the ~0.088 noise band (§2.1); a proper grok-protocol separator
battery (AdamW + wd-sweep, unbounded tasks, GDN-2 width control) if the `lb_compare`
separators are to be cited at all (currently grok-suppressed and not load-bearing).

---

*Self-contained handoff authored from committed measured artifacts. Every finding cites
its provenance (path + commit). Calibration is deliberate and stated in both directions:
LM loss is a tie-or-lead, the capability separation is real but `modular_quadratic`-specific,
throughput is a tie. The binding near-term blocker is the training recipe, and the long
engineering pole is the ROCm kernel port.*
