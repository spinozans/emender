# SCALE PLAN — push 1.3B token count to <1 bpb and into capability emergence

> **SCOPE REVISION (PI) — supersedes the "then frontier 3B/7B handoff" framing below.**
> **STAY AT 1.3B.** The plan is to scale up *training* — push the **token count** high on the
> 1.3B models — to (a) reach **< 1 bpb**, and **(b) keep going past the bpb gate into the regime
> where useful, measurable capabilities emerge.** The central deliverable is **capability-emergence
> tracking**: run the capability eval suite (held-out bpb + temporal/length-extrapolation +
> the algorithmic battery: recall, counting, S5, the formal separators) on **checkpoints as the
> token count grows**, and measure **whether `emender-mlp` diverges from `gdn2-mlp` at high token
> counts** — that is the experiment. The **3B/7B parameter scale-up is a SEPARATE, LATER phase,
> gated on securing compute allocations** — keep the §-on-frontier as a brief future option, not
> the immediate plan. Practically: extend the token budget *beyond* the ~16B <1-bpb gate (cap is
> a checkpoint, not the end), and add capability evals at a fixed token cadence throughout.
>
> **LOCKED CONFIG (pending `preflight-100b` throughput):** seed-model run = **`emender-mlp`
> (E97-delta + MLP, dim1792/nh216/ns32/dep11/mlp2.26) on 7-GPU DDP** (leave 1 GPU free for the
> `gdn2-mlp` control + ad-hoc work), **target 100B tokens** (~77 tok/param, ~4× Chinchilla,
> emergence regime). Wall-clock estimate **~18–25 days** (≈3–3.5 wk) depending on whether the
> E97 1.26–1.56× speedup holds at 1.3B — `preflight-100b` measures the real tok/s, max batch, and
> verifies DDP + fused-no-eager + checkpoint-roundtrip + bpb-eval before the box is committed.
> The 100B `emender-mlp` checkpoint **is the seed model** handed to frontier (which re-tunes from
> there: bigger batch via more RAM, re-scaled LR; frontier = its own lean HPO, not a port).
> **Capability caveat (from `grok-confirm`):** the proven temporal class separation is
> **modular_quadratic-specific** (iterated nonlinear maps requiring per-step nonlinearity
> linear-state can't represent + non-contractive memory) — robust and real there, but NOT a
> general law (counting/a^n b^n c^n: linear extrapolates equally; contractive maps: no separation).
> The 100B run's capability suite asks whether that specific capability surfaces as *measurable
> real-LM* capability where emender-mlp diverges from gdn2-mlp at high token count.

**Task:** `scale-plan` (Architect). A concrete plan — **not** a training run — to
**push a few CMA-best 1.3 B configs to high token count** on the local 8-GPU box —
reaching **below 1.0 bpb** and then **continuing into the capability-emergence regime**,
tracking capability evals vs token count. The 3B/7B param scale-up is deferred (frontier
section retained below as a future, compute-gated option, not the next step). Every number here is grounded in committed measured data; provenance is
cited inline, and the one soft number (E97 throughput multiplier) is gated behind
a mandatory pre-flight measurement (§4.0) because prior E97 throughput claims have
been wrong (post-mortem item 13).

> **Framing note (post-mortem discipline).** `docs/RESEARCH_ASTRAY_POSTMORTEM.md`
> documents ~10 days of one-directional false-negatives against the Emender/E97
> line, every one corrected toward a positive by the PI. The fair, committed result
> is that **`emender-mlp` (E97-delta split-edit + SwiGLU MLP) LEADS `gdn2-mlp` on
> both primary metrics** (CMA search avg-loss 5.8606 < 5.8949; non-avg held-out BPB
> 2.091 < 2.101), losing only on the schedule-free *averaged* basis that the run
> itself flags as the inferior/artifact basis (`experiments/lb_compare_20260613/`).
> This plan therefore treats **`emender-mlp` as the lead candidate**, not a foil,
> and does not pre-bake a "gdn2 wins" verdict. The convergence race decides it on
> real long-horizon data.

---

## 0. TL;DR

- **Run 3 arms** (4th optional): **`emender-mlp`** (lead), **`gdn2-mlp`** (control),
  **`pure-E97`** (LM-ceiling / fastest-kernel probe); **`m2rnn`** optional foil.
- **Data:** local convergence race on **`pile.txt`** (the corpus the <1 bpb / E88
  0.973 precedent is anchored on — apples-to-apples gate), with a parallel held-out
  eval on **`commapile_mainmix_v0.1_1tb.txt`** for cross-distribution sanity and as
  the frontier corpus.
- **Per-arm placement:** **1 GPU per arm, all arms concurrent** (3–4 of 8 GPUs),
  via the lease broker. Preserves the exact ~10 K-tokens/update density that the
  E88 single-GPU precedent validated — no global-batch confound, no LR retune.
- **Token budget / gate:** ~**16 B tokens** to reach the gate (E88 precedent:
  0.966 held-out bpb at 15.8 B tokens), **20 B cap**. Gate = **held-out bpb < 1.0**,
  two consecutive evals.
- **Wall-clock (1 GPU/arm, RTX 6000 Ada, measured 7.49 K tok/s E88-class):**
  `gdn2-mlp` ~**25–26 d** to 16 B; `emender-mlp`/`pure-E97` ~**16–20 d** *iff* the
  1.25–1.59× E97 speedup confirms in pre-flight (§4.0). Total wall-clock ≈ slowest
  arm (~26 d), all arms converging in one window.
- **Frontier:** port the winning arm's exact cell+geometry, **scaled** (e.g.
  1.3 B → 7 B), via **hierarchical ScheduleFree-DiLoCo** (`docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md`).
  Top risk = Triton/XMA kernel portability to ROCm.

---

## 1. Configs — recommended arms

All geometries are the **CMA-best 1.3 B winners at their own found geometry**, taken
byte-for-byte from `experiments/lb_compare_20260613/REPRODUCTION.md` (param counts
verified there against each source's recorded `actual_params`). Each is built through
`scripts/cmaes_search_v2.build_train_command` → `train.py` so the long run is
byte-identical to the search that produced it.

| Arm | Cell / level | Geometry | Params | Search avg-loss | LR (CMA) | Run? |
|---|---|---|---|---:|---:|:--:|
| **emender-mlp** | E97 split-edit **DELTA** (`e88_raw_write=0`) + SwiGLU MLP | dim1792 nh216 ns32 dep11 mlp2.26 bs4 | 1286.6 M | **5.8606** | 1.007e-3 | ✅ lead |
| **gdn2-mlp** | GDN-2 mixer + SwiGLU MLP | dim2176 nh30 dep12 mlp3.259 bs4 | 1286.7 M | 5.8949 | 4.74e-4 | ✅ control |
| **pure-E97** | E97 split-edit **raw-write**, no MLP | dim2432 nh416 ns16 dep10 bs3 | 1265.6 M | 5.9511 | 9.85e-4 | ✅ probe |
| m2rnn | M2RNN matrix RNN (XMA fused) | dim3072 nh346 ns16 dep13 bs4 | 1275.0 M | 6.0636 | 1.04e-3 | ⬜ optional |
| ~~Emender-mix~~ | typed mixture f=0.971 (~97% e97_delta) | dim2432 nh212 ns32 dep10 bs2 | 1273.2 M | 6.0756 | 1.144e-3 | ❌ drop |

### Why these three (+1)

- **`emender-mlp` — the lead.** It is the fair MLP-vs-MLP winner on both primary
  metrics, the capability-retaining DELTA cell (not raw-write), and the candidate
  the throughput evidence flags as **1.25–1.59× faster** than gdn2 (most recent
  clean bf16+fused sweep, `grok-symmetric-width`, commit `6c929fe`). It is the only
  arm that is *simultaneously* loss-competitive, capability-bearing, and fast — the
  whole reason to scale. **This is the config we most want to carry to frontier.**
- **`gdn2-mlp` — the control.** The strongest non-E97 baseline; tied-best LM bpb;
  the incumbent the whole convergent-loss-null literature rests on. The race is only
  meaningful with it run under identical conditions. It also de-risks frontier: FLA
  GDN has the most mature ROCm path of the four kernels.
- **`pure-E97` — the probe.** Best **non-avg held-out** of all five (2.0126) → most
  token-efficient on pure LM, and the **fastest, simplest single kernel** (no MLP,
  one Triton call). Running it next to `emender-mlp` isolates *what does the LM work*:
  the raw-E97 cell vs the MLP. It is the LM-ceiling reference. (Caveat: it robustly
  **fails** `modular_counter` — an LM-ceiling foil, not a capability candidate.)
- **`m2rnn` — optional 4th foil** only if a GPU is otherwise idle. It is the weakest
  arm (6.0636) and adds a different substrate (matrix-to-matrix) for breadth, but
  carries no hypothesis we need to settle here.
- **Drop `Emender-mix`.** At f=0.971 it is ~97% the same cell as `pure-E97` and is
  dominated by `emender-mlp` (which adds the MLP and is faster); its bs2 makes it the
  slowest arm. Redundant — `pure-E97` + `emender-mlp` already bracket the mixture axis.

---

## 2. Local sub-1-bpb protocol

### 2.1 Data

- **Convergence race (the gate): `pile.txt`** — `/home/erikg/elman/data/pile.txt`
  → `/mnt/nvme2n1/erikg/pile.txt` (1.308 TB, p50k_base). **This is the corpus the
  sub-1-bpb precedent lives on**: the production 1.273 B E88 reached **0.974 train /
  0.9661 held-out bpb** here (`paper/review/PILE_BPB_MEASURED.md`,
  `paper/review/THROUGHPUT.md`). Using the same corpus makes the <1 bpb gate a
  genuine apples-to-apples target rather than a moving one.
- **Held-out gate slice:** the fixed disjoint **far-tail** slice (offsets ≥ 90 % of
  file), byte-for-byte identical for every arm. Reuse the `lb_compare` builder
  (`experiments/lb_compare_20260613/build_heldout_tensor.py`) but **enlarge to ≥ 1 M
  scored tokens** (lb-compare used 131 072; a long-run gate wants tighter variance).
  Byte denominator fixed (pile.txt tail ≈ **3.878 bytes/token**, p50k_base), so
  `bpb = (CE_nats/ln2) / 3.878` is comparable across arms and to the E88 precedent.
- **Cross-distribution eval (not the gate): `commapile_mainmix_v0.1_1tb.txt`** —
  `/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/` (1.000 TB, license-clean). Score
  the same checkpoints on a comma-pile tail slice each gate eval. Purpose: confirm
  the <1 bpb result is not pile-specific overfit, and pre-validate the frontier
  corpus (frontier trains on comma-pile for licensing).

### 2.2 Optimizer / LR / schedule (long convergence run — NOT the 15-min CMA budget)

**Recommended: schedule-free AdamW at the per-arm CMA-found LR, horizon-free.**
This is the *proven* path — the E88 precedent reached 0.966 held-out with exactly
this optimizer (`schedulefree`, `paper/review/THROUGHPUT.md`,
`run_pile_convergence_3arch.sh`). It needs no pre-committed decay horizon, which is
right for an open-ended *train-until-gate* run where the exact step count is unknown.

- **Optimizer:** `--optimizer schedulefree` (AdamW schedule-free), bf16 uniform.
- **LR:** each arm's own CMA-found LR (table §1). These were tuned at the 15-min
  budget but the E88 precedent shows the schedule-free LR transfers to the long
  horizon without decay. Add an explicit **2–5 k-step linear warmup** for the long
  run (the 15-min search never left warmup-dominated territory).
- **x/y-mode checkpoint discipline (mandatory).** Schedule-free saves **x-mode**
  (eval-extrapolated) weights, which are catastrophic at inference (~17 nats);
  **all eval/gate measurement must apply the y-mode swap** (`optimizer.train()`,
  `generate.load_model` path) — see `paper/review/PILE_BPB_MEASURED.md` UPDATE
  2026-06-01. Resume must restore optimizer state, not just `model_state_dict`.
- **Alternative (if a fixed token budget is locked first):** AdamW + 2 k warmup +
  cosine decay to the 16–20 B horizon. Use *only* if we commit the horizon up front;
  it can edge out schedule-free at a known endpoint but forfeits early-stop-on-gate.

> bf16 + **fused kernels mandatory** (`--use_triton 1` for E97/E88; FLA chunked for
> gdn2; XMA for m2rnn). The fused E97 split-edit kernel is **bf16-only** (no fp16,
> no fp32 fused path — memory `emender-real-cap`); bf16-uniform is the correct and
> only valid precision for the E97 arms. Assert "0 eager fallbacks" at startup.

### 2.3 Token budget and the <1 bpb gate

**Budget anchor (E88 precedent, measured):** the 1.273 B E88 reached **0.9661
held-out bpb** at step **1,542,000** × bs5 × 2049 ≈ **15.8 B tokens**
(`paper/review/PILE_BPB_MEASURED.md`). So:

- **Expected-to-gate budget: ~16 B tokens/arm.** `emender-mlp` and `pure-E97` lead
  on token-efficiency (search avg-loss + non-avg held-out) and may cross **earlier**.
- **Hard cap: 20 B tokens/arm** (~1.95 M steps at bs5/ctx2k) — a safety margin to
  robustly clear 1.0, after which we stop and report wherever the arm landed.
- **GATE = held-out bpb < 1.0** on the fixed pile.txt tail slice (y-mode-swapped
  averaged weights), **two consecutive gate evals** to avoid a noise crossing.
  Report **both** held-out and train-loss bpb (E88 reported 0.974 train / 0.966
  held-out — held-out is the honest gate; train-bpb is the precedent-comparable one).
- **GO criterion for frontier handoff:** the arm clears <1.0 bpb **and** is on or
  inside the E88 budget (≤ 16 B tokens) — i.e. at least as sample-efficient as the
  precedent. `emender-mlp` clearing first / lower is the scale-it signal.

### 2.4 Checkpoint & eval cadence

Mirror the proven `run_pile_convergence_3arch.sh` campaign settings:

- **Checkpoint:** `--save_every 3000` steps (≈ 31 M tokens), `--keep_checkpoints 96`
  (rolling ~96 × 31 M ≈ 3 B-token window retained). **Pin** any checkpoint that
  crosses a gate eval to a stable dir immediately — the live job rotates checkpoints
  out (the E88 eval lost step 1,530,000 to rotation mid-eval).
- **Gate eval:** every **25 k steps** (≈ 256 M tokens) run the held-out bpb gate on
  both pile.txt and commapile tail slices (seconds each — ≥1 M scored tokens).
  Cheap enough to run inline without materially slowing training.
- **Logging:** `--log_every 50` (train loss/bpb + tok/s windows). Watch for
  non-finite / divergence; schedule-free is stable here but the E97 split-edit kernel
  has had chunked-overflow edge cases (memory `complex-eig-chunked-overflow`,
  `fuse-2kernel` NaN bugs) — assert finite loss each log window.

### 2.5 Wall-clock estimate per arm (throughput-based)

Anchor: **measured 7,492 tok/s** sustained for 1.273 B E88 (bs5, ctx2048) on
**this exact GPU model** (RTX 6000 Ada, `paper/review/THROUGHPUT.md`); FLA-GDN
measured 8,248 tok/s. All 8 box GPUs are RTX 6000 Ada (verified), so the anchor
transfers directly (±3 % GPU-to-GPU).

The E97 arms are credited the **1.25–1.59× over gdn2** clean-throughput multiplier
from the most recent fused sweep (`grok-symmetric-width`, commit `6c929fe`; memory
index) — **but this MUST be confirmed at the literal `dim1792/nh216/dep11` geometry
in pre-flight (§4.0) before the long run is trusted.** Estimates below bracket the
1.25× (conservative) and 1.59× (optimistic) ends.

| Arm | Assumed tok/s (1 GPU) | Days to 16 B | Days to 20 B (cap) |
|---|---:|---:|---:|
| **gdn2-mlp** | ~7,000–7,500 (E88-class; MLP-heavy) | **24.7–26.5** | 30.9–33.1 |
| **emender-mlp** | ~9,400–11,900 (1.25–1.59× gdn2) | **15.5–19.7** | 19.4–24.6 |
| **pure-E97** | ~9,400–11,900 (E97 kernel, no MLP) | **15.5–19.7** | 19.4–24.6 |
| m2rnn | ~7,000 (XMA, m2rnn-class) | 26.5 | 33.1 |

- **Per-arm placement: 1 GPU each, all concurrent.** 3 arms use 3 GPUs (4 with
  m2rnn); the box has 8. Acquire via `eval "$(scripts/gpu_lease.sh 1)"` per arm — the
  long heartbeat keeper holds the lease for the whole multi-week run.
- **Total wall-clock ≈ the slowest arm (gdn2-mlp, ~25–26 d to 16 B).** The E97 arms
  finish ~1 week earlier and free their GPUs.
- **Optional 2× acceleration:** DDP across 2 GPUs/arm (~12–13 d for gdn2) — but this
  **doubles the global batch** (20 K tokens/update), departing from the validated
  10 K density and requiring an LR re-scale (√2 or linear). **Not recommended for the
  primary race** (adds a confound to a multi-week investment); reserve DDP for a
  re-run only if a single-GPU arm shows it is genuinely throughput-bound and time
  matters more than a clean apples-to-apples to the E88 precedent.

---

## 3. Frontier handoff

**Target system:** OLCF-Frontier-style allocation (AMD MI250X, 8 GCDs/node, ROCm),
per `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md` and
`docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`.

### 3.1 What scales

- **Winning arm only.** Carry the single arm that (a) cleared <1.0 bpb fastest/lowest
  and (b) brings the capability/throughput story — expected `emender-mlp`. Carry
  `gdn2-mlp` as the frontier control for one matched run (it de-risks the kernel port).
- **Params:** 1.3 B → **7 B** first (then 13 B once the island design is trusted).
  Scale `dim`/`depth`/`n_heads` keeping the arm's found *aspect ratios* (E97
  width-multiprogramming: heads scale with width, ns32 fixed). Do **not** re-CMA at
  7 B blind — seed from the 1.3 B geometry and do a short local-proxy width sweep.
- **Tokens:** follow the loss curve to <1 bpb at the new scale, not a fixed count.
  Order-of-magnitude: a 7 B model at ≥ Chinchilla density wants **≥ 140 B tokens**;
  comma-pile-mainmix (1 TB ≈ 250 B p50k tokens) covers one epoch — plan multi-epoch
  or expand the corpus before 13 B.
- **Data:** **`commapile_mainmix`** (license-clean) sharded by
  `--data_rank/--data_world_size` (the frontier design's required `train.py` flags;
  pre-validated locally per §2.1).

### 3.2 Exact recipe to port

```text
cell/level:        <winning arm level + kwargs, byte-identical from
                   cmaes_search_v2.build_train_command>   # e.g. emender-mlp:
                   E97 split-edit delta (e88_raw_write=0) + SwiGLU MLP, ns32
geometry:          1.3B found shape scaled to target params (aspect-ratio preserved)
precision:         bf16 uniform + fused kernels (E97 split-edit = bf16-ONLY)
inner optimizer:   ScheduleFree AdamW, per-arm LR, 2-5k warmup, y-mode eval swap
parallelism:       hierarchical ScheduleFree-DiLoCo
                     - island = 1 node = 8 GCDs, DDP every local step
                     - per-GCD bs=1-2 (16-32K tokens/local update)
                     - inter-island model-WEIGHT averaging every K=250-1000 steps
                     - outer_beta=0 (local-SGD) first, momentum only once stable
first pilot:       16 nodes x 24h, K=100-250, model-only sync (design Phase 3)
then:              64 nodes, K=500-2000
checkpoint/merge:  every DiLoCo round + periodic local emergency ckpt; merge in
                   fp32 on CPU/GPU, reject non-finite / outlier-norm island deltas
```

### 3.3 Eval & success criteria

- **Primary:** held-out bpb on a comma-pile tail slice (same byte-norm protocol as
  local), evaluated on merged global checkpoint each major DiLoCo round, y-mode swap.
- **Success = the scaling law stays on track:** the 7 B run reaches **strictly below**
  the local 1.3 B gate bpb (a larger model on more tokens must compress better), with
  **throughput parity** to the gdn2 control (≥ ~0.9× tok/GCD) and **no divergence**
  (drift bounded, loss monotone across merges).
- **Secondary:** generation samples from fixed prompts each major round; the formal
  separators re-run *properly* (AdamW + wd-sweep + train-to-grok on unbounded
  counting/Dyck with a GDN-2 width control) — the capability question the lb-compare
  grok-suppressed battery left **UNDETERMINED** (REPRODUCTION corrections §3).
- **Frontier comparison must be matched:** same arch, same tokens, same wallclock,
  vs a single-worker and a DDP baseline (design doc §Evaluation).

### 3.4 Risks (ranked)

1. **Kernel portability (highest).** The E97 split-edit Triton kernel and the m2rnn
   XMA kernel are tuned for NVIDIA/CUDA; **frontier is ROCm/MI250X**. The E97 kernel
   is **bf16-only** and has had chunked-overflow/NaN edge cases on CUDA already
   (memories `complex-eig-chunked-overflow`, `fuse-2kernel`). **Mitigation:** port +
   parity-test (≤ few e-3 bf16, T∈{128,512,1024,2048}) on a single MI250X GCD
   *before* any multi-node run; keep `gdn2-mlp` (FLA, mature ROCm) as the fallback
   arm. This is the gating risk for choosing emender-mlp as the frontier arm.
2. **Precision at scale.** bf16-only E97 + larger dynamic range at 7 B → watch the
   state recurrence for overflow; the adaptive sub-chunking fix
   (`complex-eig-chunked-overflow`) must travel with the kernel. No fp32 safety net
   for the fused path.
3. **DiLoCo drift / global-batch.** K too large → post-sync loss shock (design saw
   drift 0.467 at K=500 for 1.27B); start K=100–250, model-only sync, outer_beta=0.
   ScheduleFree x/y-mode semantics across merges are unverified at scale (merge the
   y-mode/eval weights, design §ScheduleFree Interaction).
4. **Data scaling.** 1 TB comma-pile ≈ 250 B tokens = one 7 B epoch; 13 B wants
   corpus expansion or controlled multi-epoch. Shard correctness
   (`--data_rank/--data_world_size`) must guarantee no duplicate data across islands.
5. **Throughput regression.** If the 1.25–1.59× E97 edge does **not** survive the
   ROCm port (it is the one soft local number), emender-mlp's wall-clock advantage
   over gdn2 evaporates — then the choice rests on loss + capability alone. The local
   pre-flight (§4.0) and the MI250X parity run both feed this decision.

---

## 4. Pre-flight gate (before the multi-week race starts)

### 4.0 Throughput verification (MANDATORY — resolves the one soft number)

Prior E97 throughput claims have been both **too optimistic** (the un-fused
`complex_eig` "kernel" with zero `@triton.jit`, post-mortem item 13) and **too
pessimistic** (the 2-kernel within-layer split measured 0.73–0.88× — memories
`e97delta-1p3b`, `e97-wallclock-cma`). The 1.25–1.59× figure is for the **single
fused E97 kernel** (`emender-mlp`/`pure-E97`), a different and faster path — but it
**must be measured at the literal `dim1792 nh216 ns32 dep11 mlp2.26` geometry** on a
leased RTX 6000 Ada, 5-min sustained window after compile warmup, exactly as
`THROUGHPUT.md` did, **before** committing 16–20 days of GPU time. If the measured
emender-mlp/gdn2-mlp tok/s ratio lands below ~1.2×, update §2.5 wall-clocks and
re-confirm the arm choice — do not carry the 1.25–1.59× assumption into the run.

### 4.1 Other pre-flight checks

- Build every arm via `cmaes_search_v2.build_train_command` and assert param counts
  match §1 (byte-identical to the CMA search).
- Assert bf16 + fused + **0 eager fallbacks** at startup for each arm.
- Verify the y-mode eval swap reproduces a sane held-out bpb on a 5-min smoke
  checkpoint (block-loss sanity gate < 3 nats, per PILE_BPB_MEASURED) — catches the
  x-mode garbage trap before it costs a multi-week eval.
- Enlarge + freeze the held-out gate slice (≥ 1 M tokens, pile.txt + comma-pile);
  commit its descriptor (sha256 + byte range) for reproducibility.

---

## 5. Validation checklist (task)

- [x] **Recommended 2–4 configs with geometries + rationale** — §1 (emender-mlp lead,
      gdn2-mlp control, pure-E97 probe, m2rnn optional; Emender-mix dropped, why given).
- [x] **Local sub-1-bpb protocol** — §2: data (pile.txt gate + comma-pile cross-eval),
      token budget (16 B expected / 20 B cap, E88-anchored), optimizer/schedule
      (schedule-free AdamW + warmup + y-mode discipline; cosine alt), throughput-based
      wall-clock per arm (§2.5 table), checkpoint (save_every 3000/keep 96) & gate-eval
      (every 25 k steps) cadence, **<1.0 bpb gate** (2 consecutive held-out evals).
- [x] **Frontier handoff** — §3: scale targets (1.3 B→7 B→13 B, token scaling),
      exact recipe to port (hierarchical ScheduleFree-DiLoCo), eval + success criteria
      (scaling-law-on-track + throughput parity + no divergence), ranked risks
      (kernel portability, precision, DiLoCo drift, data scaling, throughput).
- [x] **Throughput honesty** — the 1.25–1.59× E97 multiplier is factored (§2.5) **and**
      gated behind a mandatory pre-flight measurement (§4.0), per post-mortem item 13.

### Sources (committed/measured)
- `paper/review/THROUGHPUT.md` — 7,492 tok/s E88, 8,248 FLA-GDN, RTX 6000 Ada, MFU.
- `paper/review/PILE_BPB_MEASURED.md` — E88 0.9661 held-out @15.8 B tokens; y-mode swap.
- `experiments/lb_compare_20260613/{LEADERBOARD,REPRODUCTION}.md` — 1.3 B geometries,
  emender-mlp-leads corrections.
- `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md` — frontier recipe.
- `docs/RESEARCH_ASTRAY_POSTMORTEM.md` — false-negative discipline; throughput-claim hazard.
- `~/elman/run_pile_convergence_3arch.sh` — proven convergence-campaign settings.
- Data: `/mnt/nvme2n1/erikg/pile.txt`, `/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt` (both verified present).
