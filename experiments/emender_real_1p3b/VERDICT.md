# emender-real-1p3b — VERDICT

**Question.** At **1.3B**, does the **CMA-FOUND** sparse Emender mixture (a sea of
`gdn2_recall` neg-eig heads + a small CMA-discovered fraction of nonlinear `e97_delta`
split-edit emendment heads, **not hand-picked**) deliver the claim **"same loss + more
capability at ~same speed"** versus a **CMA-best GDN-2** and a **CMA-best m2rnn**, with
**all arms at uniform half precision** (the fair, matched-precision comparison)?

This task **corrects the two confounds** in `opt-1p3b`: that run used a **hand-picked
50%-nonlinear** "house" mixture **in fp32** while the bf16 controls did ~4.3× more tokens
at matched wall — a token-starved strawman. Here the Emender is the mixture **CMA-ES
actually found** in `emender-real-cap` and **every arm runs bf16 UNIFORM**.

## Verdict: **NO-GO / NULL** — the claim is refuted on all three axes.

At 1.3B, matched precision (bf16 uniform), fused, best-vs-best, the CMA-found Emender:
1. **does NOT tie on loss** — it **loses** held-out BPB both token-matched (**+0.043**) and
   wall-matched (**+0.092**) vs CMA-best GDN-2, even at the Emender's *own best* lr
   (1.6e-3, found by a fully-bracketed lr sweep);
2. **adds NO capability** GDN-2 cannot reach — at convergence the named-task separation vs
   the real GDN-2 incumbent is **+0.004** (modular_quadratic tie) and **−0.403**
   (s5 length-extrapolation, Emender *worse*); the eye-catching undertrained gap was
   **convergence speed**, not capability;
3. **runs at 0.80× throughput** — not "same speed"; so wall-matched it loses by even more.

This **extends and strengthens** `emender-real-cap`'s pre-registered 1.3B expectation. At
the proxy scale, the loss was a tie-within-noise; **at 1.3B the Emender loses loss even
token-matched at best lr** — the sparse nonlinear sprinkle is a net liability at scale.

---

## 0. Precision (the central fix vs opt-1p3b) — bf16 UNIFORM, all arms, fused asserted

`emender-real-cap` §0 verified on real GPU that the fused `e97_delta` split-edit Triton
kernel is **bf16-only** (pure fp16 → `RuntimeError`; fp16-autocast → silent bf16 cast =
a hidden dtype mismatch). **fp16-uniform-fused is therefore impossible** for the Emender.
Per the matched-precision mandate, **all three arms run bf16 UNIFORM** — a loud
`PRECISION-ASSERT` refuses any fp32 arm at import. Every typed layer asserts
`use_triton_e97=True` with `e97_delta` on the **sequential split-edit** path (no eager
fallback); m2rnn asserts its XMA Triton backend; gdn2 uses FLA chunked. tok/s parity is
confirmed (no token deficit from a precision mismatch — the opt-1p3b confound is gone).

## 1. The mixture is CMA-FOUND, not hand-picked

From `emender-real-cap/results_cma/cma_result.json:found_mixture` =
**29 `gdn2_recall` + 1 `e97_track` + 2 `e97_delta` @ nh32** (f_delta=6.25%, f_track=3.125%,
**9.375% nonlinear**). Scaled to the 1.3B nh64 at the **same fractions** = **58 `gdn2_recall`
+ 2 `e97_track` + 4 `e97_delta`**. No optimization levers were added (CMA searched **only**
the mixture). Geometry d3072/dep22/h64/ns32/exp2.0 (V=ns·exp=64, at the kernel's
constraint boundary) = **1.310B**, param-matched to the 1.35B gdn2 / 1.31B m2rnn controls.

## 2. LM held-out BPB — REAL Comma-Pile, bf16 uniform, fused, 2 seeds (best-vs-best)

Controls at their CMA-best lr (gdn2 8.63e-4, m2rnn 6.0e-4); Emender at **its own best lr
(1.6e-3)** from a fully-bracketed token-matched lr sweep — the fair best-vs-best (a single
untuned lr would unfairly handicap it; `emender-real-cap` §3 documented the token-Δ
sign-flips ±0.2 with lr).

| match | EMENDER (CMA-found) | CMA-best GDN-2 | CMA-best m2rnn | Δ(em−gdn2) |
|---|---|---|---|---|
| **WALLCLOCK** (20 min) | **1.9097** ±0.009 | **1.8181** ±0.007 | 1.8512 ±0.004 | **+0.092 (loses)** |
| **TOKEN** (7.00 M) | **1.9225** ±0.003 | **1.8799** ±0.002 | 1.8824 ±0.003 | **+0.043 (loses)** |

The Emender **loses on loss both ways**, even best-vs-best at its true-optimum lr. Ranking
on loss: **GDN-2 > m2rnn > Emender** (wall and token). This is *not* the proxy-scale tie —
at 1.3B the sprinkle is a net loss-liability. (At matched lr discipline, lr 8e-4, it is
worse still: wall 1.955 / token 1.970 — see appendix.)

**Emender lr sweep (token-matched 7 M, seed 0)** — the optimum is bracketed and the gap
never closes:

| lr | 6e-4 | 8e-4 | 1.0e-3 | 1.3e-3 | **1.6e-3** | 2.0e-3 |
|---|---|---|---|---|---|---|
| Emender token BPB | 2.024 | 1.974 | 1.944 | 1.943 | **1.925 (best)** | 1.932 |

The curve bottoms at lr 1.6e-3 (turns back up by 2.0e-3) — the minimum is **bracketed
from both sides**. Best Emender (1.925, 2-seed 1.9225) still > GDN-2 (1.880). The loss
deficit is **robust to lr**: no learning rate makes the CMA-found Emender tie GDN-2.

## 3. Throughput — 0.80×, not "same speed"

Sustained training tok/s (fwd+bwd, bf16, the honest wall-clock rate), wall-matched runs:

| arm | tok/s | ratio vs GDN-2 |
|---|---|---|
| CMA-best GDN-2 | 8107 | 1.000 |
| CMA-best m2rnn | 7285 | 0.899 |
| **EMENDER** | **6450** | **0.796** |

The per-step bounded-`tanh` `e97_delta` split-edit head is a **latency-bound sequential
scan**; even sparse (4/64) on the hetero stream-overlap path it is **0.80×** GDN-2 — close
to `emender-real-cap`'s 0.75× microbench and well short of the 0.95× "same speed" target.
This is *why* the wall-matched loss gap (+0.092) exceeds the token-matched one (+0.043):
fewer tokens per wall-second on top of worse sample efficiency.

## 4. Expressivity at 1.3B — NO capability beyond GDN-2 (convergence control decisive)

The honest separation control is **`gdn2typed`** = all-`gdn2_recall` on the *same typed
path* (isolates the emendment heads from the typed plumbing). At a fixed **1500-step**
budget the Emender shows a large *apparent* gap — but this is **convergence speed**, not
capability. A **convergence control at ~4000 steps** settles it:

| task (acc @512) | EMENDER | gdn2typed | CMA-best GDN-2 | em−typed | **em−GDN-2** |
|---|---|---|---|---|---|
| modular_quadratic (THE cliff) | 1.000 | **1.000** | 0.996 | +0.000 | **+0.004 (tie)** |
| s5_permutation (length-extrap) | 0.329 | 0.076 | **0.731** | +0.253 | **−0.403 (Emender WORSE)** |

- **modular_quadratic:** at 1500 steps Emender 1.000 vs gdn2typed 0.526 (+0.47); at 4000
  steps **gdn2typed reaches 1.000** — the cliff is reached by pure GDN-2 given enough
  steps. The `e97_delta` head only learns it *faster* (an inductive-bias/optimization
  accelerant), not a capability pure GDN-2 lacks. Exactly `emender-real-cap` §2.
- **s5_permutation:** the real GDN-2 incumbent (`cma_gdn2`) **also solves it (1.000 at
  train length) and extrapolates BETTER** than the Emender (0.731 vs 0.329 @512). The
  `gdn2typed` weakness here is a typed-recall-only artifact, **not** Emender capability.
- **modular_counter** (counting): all arms ≈ random at this budget — tie, no separation.
- **mqar_recall** (recall, 1500 steps): `cma_gdn2` 0.884 ≫ Emender 0.639 — the Emender
  is **worse at recall** than the real GDN-2 (the 4 emendment heads displace recall
  capacity).

**Net:** versus the actual GDN-2 incumbent, the Emender **ties** modular_quadratic, is
**worse** on s5 extrapolation and recall, and ties (random) on counting. The
convergent-capability null holds at 1.3B.

## 5. Conclusion

The CMA-found sparse Emender, tested fairly at 1.3B (bf16 uniform, fused, best-vs-best,
CMA-best controls), **fails the "same loss + more capability at ~same speed" claim on
every axis**: it **loses loss** (token +0.043, wall +0.092), adds **no capability** GDN-2
cannot reach (convergence-control separation +0.004 / −0.403 vs the real incumbent), and
runs at **0.80× throughput**. CMA-best **GDN-2 is the better 1.3B cell**; m2rnn sits
between. The convergent-loss/convergent-capability null — established across architecture
(e97/complex-eig/nlmem/ttt) and optimization (opt-1p3b) — **extends to the REAL CMA-found
Emender at 1.3B, at matched precision.** This *corrects* opt-1p3b's confounded strawman
and reaches the **same NO-GO** by a fair route: the conclusion was never an artifact of
the 50%-hand-pick or the fp32 token-starvation — pure GDN-2 wins on the merits.

## Validation checklist
- [x] **CMA-found mixture used** (58/2/4 scaled from `cma_result.json:found_mixture`, not
      hand-picked); **bf16 UNIFORM all arms** (fp16 impossible, flagged; loud PRECISION-ASSERT;
      tok/s parity confirmed — no precision token deficit); **fused asserted**
      (`use_triton_e97`, `e97_delta` sequential split-edit; XMA m2rnn; no eager fallback).
- [x] **Fair token + wall BPB + expressivity vs CMA-best gdn2 (+m2rnn)**: token-matched
      (7 M) and wall-matched (20 min), 2 seeds, best-vs-best lr; expressivity vs `gdn2typed`
      substrate control AND the real `cma_gdn2` incumbent, with a **convergence control**
      that distinguishes capability from convergence speed. **Verdict on the claim: NO-GO
      / NULL** (loses loss both ways; no capability at convergence; 0.80× speed).
- [x] **Results doc committed** (this file + `BPB_TABLE.txt`, `JCC_TABLE.txt`,
      `VERDICT.txt`, `summary.json`).
