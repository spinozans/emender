# emender-real-cap — VERDICT

**Question.** The REAL Emender = a SEA of GDN-2 recall heads (`gdn2_recall`, neg-eig)
plus a SMALL, CMA-DISCOVERED fraction of nonlinear EMENDMENT heads (`e97_delta`
split-edit with per-step bounded `tanh`), at MATCHED PRECISION. Does that sparse
sprinkle (a) tie GDN-2 on loss at ~same speed and (b) add expressivity pure GDN-2
cannot reach? Mixture **found by CMA-ES**, not hand-picked.

**Verdict: NO-GO / NULL, at matched precision and with the mixture discovered by CMA.**
The sparse nonlinear emendment sprinkle adds **no capability pure GDN-2 cannot reach**,
**ties (within noise) on token-matched loss**, and runs at **~0.75× throughput** (not the
0.95× target). This extends the convergent-loss null (architecture → optimization →
the real sparse-Emender form, best-vs-best, matched precision).

---

## 0. PRECISION FINDING (flagged, verified on real GPU) — fp16 is impossible

The task asked for **fp16 uniform**. The fused E97 split-edit Triton kernel (the
Emender's `e97_delta` emendment head) is **bf16-ONLY**:
- pure fp16 weights → `RuntimeError` (the layer refuses rather than silently running
  the eager T-scan);
- fp16-autocast → the emendment head is silently cast to bf16 internally (a hidden
  dtype mismatch — the very thing the task forbids).

So **fp16-uniform-fused is impossible for the Emender.** Per the task's escape clause,
all arms run **bf16 UNIFORM** (half precision, uniform, fused, no fp32, no mismatch).
This is the correct *matched-precision* fix to the opt-1p3b strawman (which ran the
Emender in fp32 = ~4.3× token-starved vs bf16 controls). Every LM arm asserts the
fused kernel engaged (`use_triton_e97=True`, `e97_delta` on the seq split-edit path);
training raises on any eager fallback.

## 1. CMA-ES SEARCHED the mixture (not hand-picked) → small fraction, inside the noise floor

2-D CMA over `[a_delta, a_track]` → 9-type `head_type_logits` (softmax over
{`gdn2_recall` sea, `e97_delta`, `e97_track`}); **param/FLOP-locked** (dim derived per
candidate to ~41M); fitness = **held-out BPB** on the fixed real-Pile slice; popsize 5 ×
4 gens + 2 anchors; proxy depth8/nh32, 6M-token budget, bf16 uniform, fused.

- **CMA-found mixture** = 29 `gdn2_recall` + 1 `e97_track` + 2 `e97_delta`
  = **6.25% e97_delta + 3.1% e97_track (≈9.4% nonlinear)**, bpb 2.373 vs pure-GDN-2
  anchor 2.446 (token-matched, −0.073 at the 6M-token proxy).
- CMA did **NOT drive the nonlinear fraction to 0** — it kept a small sprinkle. BUT the
  bpb deltas it ranked on (~0.01–0.07) are **smaller than the bpb noise floor of this
  proxy** (see §3: token-matched Δ sign-flips by ±0.2 with lr). So the kept fraction is
  **CMA wandering inside a flat loss basin** — consistent with the convergent-loss null
  (loss is ~flat across mixtures), NOT evidence that nonlinearity helps loss.

## 2. EXPRESSIVITY battery — NULL vs the clean control (3 seeds, length-extrapolation)

Two controls: `gdn2` = fla-gdn incumbent; **`gdn2typed` = all-`gdn2_recall` on the SAME
typed path** (isolates the emendment HEADS from the typed plumbing). Separation at the
T=512 cliff (`emender − gdn2typed`):

| task | gdn2typed@512 | emender4 sepT | emender8 sepT | read |
|---|---|---|---|---|
| modular_quadratic (THE cliff) | 0.996 | **+0.003** | +0.001 | tie (ceiling) |
| iterated_nonlinear_map | 0.965 | −0.002 | +0.000 | tie |
| s5_permutation (state-track) | 0.994 | **−0.031** | −0.053 | emender WORSE |
| modular_counter (counting) | 0.630 | **−0.147** | −0.083 | emender WORSE |
| mqar_recall (recall) | 0.766 | −0.029 | +0.006 | tie |

The e97_delta sprinkle unlocks **nothing** the pure-GDN-2 (typed) cell cannot reach on
the named separation tasks (modular_quadratic cliff, S5), and **hurts** counting/S5 by
displacing recall/counting heads. The apparent S5 "win" vs fla-gdn (+0.048) is the
**typed plumbing (substrate)**, not the emendment heads — `gdn2typed` alone gets +0.080.

**The substrate clarification (decisive), small scale dim256/nh32, 2/32 & 4/32:**
the eye-catching modular_quadratic separation **+0.314 vs fla-gdn** is the *typed
plumbing*, NOT the emendment heads — against the same-path control it vanishes:

| task (small) | gdn2 (fla) | gdn2typed | emender4 | emender4 **sepT** |
|---|---|---|---|---|
| modular_quadratic (cliff) | 0.676 | **0.997** | 0.990 | **−0.007** (tie) |
| modular_counter | 0.348 | 0.456 | 0.508 | +0.052 (small; **reverses to −0.147 at large scale**) |
| s5_permutation | 0.999 | 0.996 | 0.919 | −0.077 |
| iterated_nonlinear_map | 0.934 | 0.954 | 0.953 | −0.001 |
| mqar_recall | 0.826 | 0.734 | 0.746 | +0.012 |

`gdn2typed` (pure GDN-2 on the typed path) already solves the modular_quadratic cliff
(0.997); the emendment heads add nothing on top and the lone small positive (counting,
+0.052 at small scale) **flips sign at large scale (−0.147)** — not a robust capability.
This is the convergent-capability null seen across the whole line (opt-1p3b, ttt, nlmem,
complex-eig): the win is always the *substrate* (refit counting/typed plumbing), never
the nonlinear lever.

## 3. CONVERGENT-LOSS TIE — token-matched ≈ tie (within proxy noise); wall-fair → 1.3B

Held-out BPB, real-Pile slice, bf16 uniform, fused, 2 seeds, depth12/~57M, param-matched:

| budget | gdn2 | emender(found) | Δ | note |
|---|---|---|---|---|
| token-matched (12M), lr 5e-4 | 3.132 | 3.337 | **+0.206** (emender worse) | |
| token-matched (12M), lr 1e-3 | 2.914 | 2.885 | **−0.030** (emender better) | sign-flips with lr |
| token-matched (6M proxy), lr 1.2e-3 | 2.446 | 2.373 | −0.073 | CMA anchor |

The token-matched Δ **sign-flips with lr (±0.2)** → a **TIE within the proxy noise floor**
(the convergent-loss null: mixtures tie on loss). The wall-matched runs are
**contaminated** by a systematic small-scale "more-tokens → worse-held-out-bpb" anomaly
(both arms; longer GDN-2 run degrades 3.13→3.44), so a clean wall-fair bpb is **deferred
to the 1.3B downstream**. Throughput (§4) already implies wall-matched favors GDN-2:
the Emender does ~25% fewer tokens per wall-second for at-best-tie sample efficiency.

## 4. THROUGHPUT @ 1.3B head shape (fwd+bwd bf16, ratio vs GDN-2) — 0.75×, target missed

| config | tok/s | ratio |
|---|---|---|
| GDN-2 (ceiling) | 12256 | 1.000 |
| emender4 e97d-tanh, overlap | 9232 | **0.753** |
| emender8 e97d-tanh, overlap | 8884 | 0.725 |

The per-step bounded-`tanh` split-edit emendment head is a **latency-bound sequential
scan**; even on the hetero-kernel stream-overlap path the sparse 4/64 sprinkle is only
**0.75×** GDN-2 — well short of the ~0.95× target. (The 0.95× figure from the
hetero-kernel note was a different head/blend; the e97_delta-tanh head does not reach it.)

## 5. Best fraction for the 1.3B run

The reliable evidence (expressivity NULL + 0.75× throughput + token-loss tie) says
**no nonlinear fraction is justified — the optimum is pure GDN-2 (f=0).** If the
downstream `emender-real-1p3b` nonetheless tests a nonzero Emender, carry the
**CMA-found minimal sprinkle ≈ 4/64 `e97_delta`** (6.25%, the smallest loss-competitive
fraction CMA kept), with the pre-registered expectation of **NO-GO** (token-tie,
wall-loss from 0.75×, no capability gain). m2rnn was not needed to reach this verdict.

## Validation checklist
- [x] Sparse ratio CMA-SEARCHED (not hand-picked): found ≈6–9% nonlinear; search space +
      fitness + found mixture reported. bf16 uniform asserted on all arms (fp16 impossible,
      flagged + verified); fused asserted (no eager). Throughput vs GDN-2 reported = 0.75×
      (0.95× target NOT met — reported honestly).
- [x] Expressivity battery: separation vs pure GDN-2 AND clean `gdn2typed` control reported
      with numbers (modular_quadratic cliff + S5 = NULL/negative); convergent-loss tie
      reported (token-matched = tie within noise; wall-fair deferred to 1.3B with reason).
- [x] Best sparse fraction for 1.3B identified (f=0 by the evidence; ≈4/64 if a nonzero
      Emender is tested); results doc committed (this file + RESULTS_EMENDER_REAL_CAP.md).
