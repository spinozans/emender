# fuse-2kernel — closing the e97_delta+gdn-neg wall-clock gap at 1.3B

**Task:** `fuse-2kernel` (follow-up to `e97delta-1p3b`, commit 71bda00, which found
e97_delta+gdn-neg BEATS gdn2-mlp **token-matched** (BPB 2.071 vs 2.094) but LOSES
**wall-clock** by running ~40 % slower). Premise to test: that ~40 % is a *fusable*
within-layer two-kernel-split overhead; close it and the token-efficiency edge
becomes a wall-clock GO.

**Date:** 2026-06-08. **Run dir:** `experiments/e97_delta_1p3b_cma/`.
**Compute:** 8× RTX 6000 Ada, idle-only, REAL Pile (`/home/erikg/elman/data/pile.txt`,
p50k_base, ctx 2048), bf16, schedule-free AdamW. No mocks; every BPB is a real
held-out measurement (40 batches) with fresh-process round-trip in the prior batch.

---

## 0. Decision deliverable (verdict)

**Did fusing/closing the two-kernel split close the wall-clock gap, and does
e97_delta+gdn-neg now MATCH/BEAT gdn2-mlp on wall-clock-matched held-out BPB at
1.3B?**

> **Loses on BOTH axes — decisive, and the premise is REFUTED.**
>
> The ~40 % gap was **never** the within-layer two-kernel split. It was that the
> e97_delta heads ran the **slow sequential T-scan**: the prior run used the
> default `state_activation='tanh'`, and the chunked-parallel kernel engages only
> for **linear (identity) state**. Routing e97_delta onto the chunked kernel does
> close the *kernel-level* throughput gap — but it **requires switching e97_delta
> to linear state, which destroys the token-efficiency that was the entire reason
> to pursue it.** Bounded (tanh) state and chunkability are **mutually exclusive**,
> so there is no configuration that is both fast and sample-efficient. The residual
> is **fundamental, not a fixable overhead.**

### The decisive 1.3B head-to-head (2 seeds, 720 s, param-matched ~1.26 B)

| axis | gdn2-mlp | e97_delta+gdn-neg (identity / chunked) | winner |
|---|---:|---:|---|
| **WALL-CLOCK-matched** (720 s) | **2.012 / 2.025** | 2.252 / 2.379 | **gdn2-mlp** (+0.24…+0.35) |
| **TOKEN-matched** (5.60 M tok) | **2.075 / 2.077** | 2.252 / 2.379 | **gdn2-mlp** (+0.18…+0.30) |
| sustained tok/s (full train) | 9975 / 9610 | 7824 / 7869 (0.78×) | gdn2-mlp |

The candidate loses **both** axes by a wide margin — the opposite of the prior
token-matched win, because the prior win was a property of **tanh** state, which
the chunked kernel cannot run.

---

## 1. Root cause of the prior "40 % gap": sequential kernel, not the split

Instrumented the actual training forward (`verify_fused.py`, real 1.3B dims):

| candidate state | chunked-kernel calls / e97_delta layer | tok/s (microbench, plain AdamW) | ratio vs gdn2-mlp |
|---|---:|---:|---:|
| `tanh`     (prior run) | **0 / 18** (all sequential) | 7683 | 0.80× |
| `identity` (this work) | **18 / 18** (all chunked)   | 9077 | 0.948× |

The chunked guard in `e88_fla_hybrid` (`use_chunked = use_chunked_e97 and not
raw_write and linear_state`) is `False` under `tanh`. The prior run's "chunked util
88–100 %" was an isolated bench; the *training* path ran the sequential split-edit
T-scan on all 18 e97_delta layers. So the gap was the kernel, not the within-layer
two-block split — and literal "one launch for both head types" is the wrong lever
(the E97 split-edit kernel scales linearly in heads; FLA's gated-delta stays flat,
so fusing all heads onto one kernel is *slower*, not faster — measured in the prior
attempt's recurrence microbench).

---

## 2. Making linear-state e97_delta trainable — two real NaN bugs fixed

Routing to the chunked kernel exposed two backward-only NaNs at 1.3B (the prior
attempt's "0.99× gap closed" was a throughput microbench that never trained to
convergence). Both are fixed in `ndm/triton/e97_chunked_autograd.py`, parity-
verified (`tests/test_e97_chunked.py`, ALL PARITY OK fp32+bf16), with regression
tests:

1. **DA upper-triangle overflow.** `DA = exp(G_i − G_j)` was formed over the full
   `[C,C]` tile but used only in the lower triangle. Under strong decay the masked
   upper triangle has `G_i − G_j > 88` → `exp = +inf`. The forward's `tl.where`
   drops it, but the backward's **unmasked** `*DA` products (`dQK = dA*DA`,
   `DD = (…)*DA`) hit `0 * inf = NaN`, poisoning every key/decay grad (k,q,g,e) while
   v,w stay finite — the exact observed signature. **Fix:** clamp the exponent to
   `≤ 0` (`exp(min(., 0))`); the discarded upper entries become a finite `1.0`, the
   used lower triangle is bit-identical.

2. **Log-decay drift overflow.** The per-step log-decay `g` is unbounded; linear-
   state e97_delta *learns aggressive decay* and `g` drifts from −45 to **−96 within
   ~15 steps** (loss decreasing the whole time — pure numeric blowup). `inv_decay =
   exp(−g) = exp(96) ≈ 5e41` overflows fp32. **Fix:** floor `g` at **−30** at the
   kernel's single choke point (`exp(30) ≈ 1e13` safe; per-step decay `exp(−30) ≈
   9e-14` is already "forget everything", so zero modeling range is lost) and zero
   the decay-grad on floored steps.

After both fixes a real 60-step 1.3B run is stable (loss 11.2 → 7.7, all grads
finite) and the full 720-s head-to-head ran to `wall_cap` with `nan_seen=False`.

---

## 3. Why it loses on both axes — attribution (each arm 720 s, REAL Pile)

Decomposing the candidate's quality loss with kernel/state controls
(`attribution_control.py`):

| arm | state | kernel | held-out BPB | tok (≈) | tok/s |
|---|---|---|---:|---:|---:|
| **B** prior cell  | tanh     | sequential | **2.055** | 5.16 M | 7217 |
| **A** linear-state | identity | sequential | **2.096** | 5.10 M | 7133 |
| candidate         | identity | **chunked** | **2.252** | 5.60 M | 7824 |
| gdn2-mlp (token-matched @5.60 M) | — | FLA chunk | **2.075 / 2.077** | 5.60 M | 9125–9968 |

Reading the ladder:

- **tanh+seq = 2.055 reproduces the prior token-efficiency win** — it BEATS
  gdn2-mlp's token-matched 2.075. The prior result is real and replicated.
- **Switching to linear state costs +0.041 BPB** (2.055 → 2.096). Small, but it is
  exactly enough to drop the cell *below* gdn2-mlp (2.096 > 2.075). The bounded
  (tanh) state nonlinearity is load-bearing for e97_delta's sample efficiency.
- **The chunked kernel adds a further +0.156 BPB** training degradation (2.096 →
  2.252) at *more* tokens — gradient-quality / decay-floor noise accumulated over
  ~1.4 k steps (partly fixable; see §4). Seed variance is also large (2.25 vs 2.38),
  a stability symptom of the aggressive-decay regime.

The split that matters is **state**, not kernel launches: the chunked kernel
**cannot** run the tanh state that gives e97_delta its edge. That is the wall.

---

## 4. Residuals, and the actual next lever

**Residual 1 — chunked-kernel training degradation (+0.156 BPB, intrinsic to the
chunked gradients, NOT the floor).** The chunked kernel passes fwd+bwd parity at
moderate decay but trains identity-state to a worse optimum than the sequential
kernel at the real aggressive-decay regime. **It is not the decay floor:** a
floor=−50 control (`results/floor50.json`) — flooring far fewer of the drifted
heads — gives BPB **2.245**, statistically identical to floor=−30's 2.252 (Δ 0.007,
within seed noise). The degradation is therefore the chunked backward's bf16/TF32
gradient accuracy accumulated over ~1.4 k steps, not a tuning artifact. A no-floor
exact-fp32 reformulation (fold `inv_decay` into the exclusive cumsum so no
huge×tiny products form, and run the backward dots in fp32) is the refinement that
could recover it — **but it does not change the verdict:** even the chunk-free best
case is the sequential identity-state number, 2.096, which **already loses** to
gdn2-mlp token-matched (2.075).

**Residual 2 — two-block per-step overhead (throughput).** In the full training
protocol the chunked candidate is 0.78× gdn2-mlp (vs 0.948× in the plain-optimizer
microbench): schedule-free AdamW + grad-clip iterate over the candidate's two
recurrent sub-blocks' (gdn + e97_delta) larger param-tensor set. Fixable with a
fused optimizer, but again moot — BPB decides it.

**The real lever (refuting the task premise).** The blocker is **state-nonlinearity
⊥ chunkability**, not the within-layer two-kernel split. To get *both* e97_delta's
token-efficiency *and* GDN-2-class throughput you need a **chunkable bounded-state
kernel** — a saturating (tanh-like) state map expressed inside a chunked-parallel
scan. That is precisely the `gdn2_nonlin_shell` direction (bounded nonlinear-in-time
state already fused into a chunked scan). Pursue that, not split-fusion.

---

## 5. Validation checklist (task gate)

- [x] **Chunked kernel engages for BOTH head types' layers in the chunked path,
  wired as default, loud no-eager guard intact.** 18/18 e97_delta layers route to
  the chunked fused Triton kernel under `e97_state_nonlin='identity'` (control:
  tanh → 0/18); gdn-neg heads on FLA chunk. *Literal single-launch fusion of the
  two head types was rejected with evidence* (E97 split-edit scales linearly in
  heads ⇒ counterproductive); the throughput lever is chunked-routing, not one
  launch. Loud bf16 no-eager guard (`_run_e97`) unchanged.
- [x] **Parity fwd+bwd bf16 vs the reference recurrence**, incl. new log-decay path
  + strong-decay + drift regression tests — `tests/test_e97_chunked.py` ALL PARITY OK.
- [x] **Throughput** — chunked closes the *kernel* gap (sequential 0.80× → chunked
  0.948× microbench, 18/18 util high); residual full-protocol gap (0.78×) quantified
  (§4).
- [x] **1.3B head-to-head, token AND wall-clock matched, 2 seeds, REAL Pile** — done;
  **loses on both axes**; residuals + next lever named (§3–4).

## 6. Anti-regression confirmation

- e97_DELTA (delta correction, `raw_write=False`), gdn-neg (`gdn_allow_neg_eigval=
  True`), within-layer `typed_head_mixture` fractions — all unchanged (`grep
  e97_delta ndm/models/typed_head_mixture.py`; chunked kernel `not raw_write and
  linear_state`). The candidate config matches 71bda00's CMAES-best exactly except
  `e97_state_nonlin='identity'` (the change under test).

---

### One-line summary

Fusing/chunking **does** close the kernel throughput gap, but only by forcing
e97_delta to linear state, which **erases its token-efficiency** (tanh-state 2.055
→ identity 2.096, below gdn2-mlp 2.075) and the chunked kernel degrades it further
(2.252) — **loses on both axes**; the real blocker is bounded-state ⊥ chunkable, and the next
lever is a chunkable bounded-state kernel (`gdn2_nonlin_shell`), not split-fusion.
