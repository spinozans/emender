# diloco-loss-parity-longhorizon — RESULTS (measured)

Follow-up to `implement-diloco-periodic`, which proved DiLoCo periodic-sync **throughput**
(~1.85× DDP, ~98% of the independent ceiling → 100B in ~20 d) and **merge correctness**
(y-mode swap), but left the **token sample-efficiency** OPEN: at matched tokens the
local-SGD merge (`outer_beta=0`) lagged synchronous DDP by ~0.45 BPB in early training
and the gap did not close within the **52 M tokens** measured.

**All numbers below are REAL**, measured on the 8×RTX 6000 Ada box (49 GB, PCIe, NO
NVLink), commapile_mainmix 1 TB, p50k_base, ctx 2048, bf16 + FUSED Triton, schedule-free
AdamW, emender-mlp **1,286,589,072 params** (dim1792 nh216 ns32 dep11 mlp2.2623), bs6,
seed 42. Held-out BPB = the **same** agent-1433 preflight tensor used by the predecessor
(`heldout_comma_p50k_2048.pt`, bytes/token 3.938, 65 536 tokens), evaluated on the saved
**consensus** checkpoint (the apples-to-apples metric; per-rank train loss is not
comparable because DDP's effective batch is bs·7 = 42 while each DiLoCo island is bs6).
7-GPU global tokens/step = 6·2048·7 = **86 016**.

Reproduce: `bash experiments/diloco_100b/longhorizon/phase1.sh` (sweep + β=0 long + DDP
long), `bash experiments/diloco_100b/longhorizon/phase2.sh` (fix arms), then
`python experiments/diloco_100b/longhorizon/analyze.py experiments/diloco_100b/longhorizon/results.txt`.

---

## HEADLINE — two findings

**(1) DiLoCo sample-efficiency: no advantage, ~0.45 BPB penalty, does not close.**
In the regime where training is healthy (≤ ~129 M tokens) plain local-SGD DiLoCo
(`outer_beta=0`, K=250) lags synchronous DDP at matched tokens by a **persistent
~0.44–0.47 BPB** — exactly the predecessor's early-training gap, now confirmed over ~2.5×
the horizon. It does **not** close. Outer momentum (4-config sweep) does **not** help (it
overshoots before enough outer steps accumulate).

**(2) The dominant blocker is the TRAINING RECIPE, not parallelism.** Run to 215 M tokens,
the held-out BPB of **BOTH** DDP and DiLoCo **collapses**: DDP bottoms at **1.571 @ 64.5 M**
then climbs monotonically to **3.234 @ 215 M**; DiLoCo β=0 goes 2.03 → **3.19 @ 215 M**.
Per-rank *training* loss keeps falling the whole time (DDP CE → 3.4). The cause is the
recipe shared by both arms — **schedule-free AdamW at a constant CMA-tuned LR=1.007e-3
with `warmup_steps=0` and no decay** — which the short prior runs (≤ 52 M) never exposed.
Because the baseline itself degrades, the DiLoCo−DDP gap *appears* to "close" to ~0 at
215 M (both ≈ 3.2), but that is **mutual collapse, not parity**.

**Consequence:** the matched-token DiLoCo-vs-DDP question cannot be cleanly settled until
the recipe is fixed (warmup + LR decay / lower LR) — filed as follow-up `fix-long-horizon`.
§3 evaluates whether any DiLoCo variant (outer momentum, small-K, or the 2-GPU-island
hybrid) closes the ~0.45 gap in the **healthy regime** (≤ 86 M), where the comparison is
valid.

---

## 1. OUTER-MOMENTUM SWEEP (K=250, matched-token vs DDP)

`--diloco_outer_beta {0.5,0.9} × --diloco_outer_lr {0.7,1.0}` at K=250, to 500 steps
(token points 21.5 M and 43 M). Clean periodic-consensus held-out BPB:

| arm (β, lr) | BPB @21.5 M | BPB @43 M |
|---|---:|---:|
| β=0.5, lr=0.7 | 2.6761 | 2.1909 |
| β=0.5, lr=1.0 | 2.5315 | 2.2052 |
| β=0.9, lr=0.7 | 2.6532 | 2.2854 |
| β=0.9, lr=1.0 | 2.5229 | 2.4042 |
| **β=0 (local-SGD, ref)** | — | **2.0893** |

- At this horizon **β=0 (plain averaging) is the BEST** config; held-out BPB is
  *monotonically worse* with more momentum / higher outer-lr.
- Mechanism: with K=250 and only 1–2 outer steps in 500 local steps, the outer momentum
  buffer barely accumulates; `outer_lr<1` undershoots the consensus. Outer momentum is a
  *long-horizon* stabilizer, not a short-horizon accelerator → tested over the full
  215 M horizon in §3.

(Bug found + fixed while running this sweep: when the last training step lands on a
K-boundary, the unconditional FINAL merge fired a *second* merge at the same step with
`delta=0` but leftover momentum — a spurious outer step that degraded the final
checkpoint, e.g. 2.19 → 2.27. Fixed in `train.py`: skip the final merge when
`step % K == 0` (already consensus). β=0 was unaffected. Tests:
`tests/test_diloco_merge.py`, `tests/test_diloco_hybrid.py`.)

---

## 2. LONGER-HORIZON β=0 vs DDP (to 215 M tokens) — THE GAP WIDENS

K=250, `outer_beta=0`, run to 2500 steps (215 M tokens), held-out BPB of the
periodic-consensus checkpoint every 500 steps, vs vanilla per-step DDP at the same steps:

| tokens (step) | **DDP** BPB | **DiLoCo β=0** BPB | gap (β0−DDP) |
|---|---:|---:|---:|
| 21.5 M (250)  | 1.7702 | — | — |
| 43.0 M (500)  | 1.6185 | 2.0893 | **+0.471** |
| 64.5 M (750)  | **1.5706** (DDP min) | — | — |
| 86.0 M (1000) | 1.5956 | 2.0321 | **+0.436** |
| 107.5 M (1250)| 1.6901 | — | — |
| 129.0 M (1500)| 1.8575 | 2.3054 | **+0.448** |
| 150.5 M (1750)| 2.1122 | — | — |
| 172.0 M (2000)| 2.4557 | 2.7190 | +0.263 |
| 193.5 M (2250)| 2.8435 | — | — |
| 215.0 M (2500)| 3.2342 | 3.1884 | −0.046 |

- **BOTH arms collapse.** DDP held-out BPB bottoms at **1.571 @ 64.5 M** then rises
  monotonically to **3.234 @ 215 M**; DiLoCo β=0 rises 2.03 → 3.19. The DiLoCo penalty is
  a stable **~0.44–0.47 BPB while training is healthy (43–129 M)**, then *shrinks* to ~0
  as DDP collapses to meet it — **mutual collapse, not parity**.
- **Per-rank TRAIN loss stays low/decreasing for both** (β=0: 8.17 → 5.02 CE; DDP → 3.4 CE
  at step 2500). The held-out (schedule-free **eval/x** weights) degrades while the
  **train/y** weights keep improving → the constant-LR-no-warmup schedule-free recipe
  fails to hold out a good *eval* average over a long horizon. This is the dominant effect
  and it is **not DiLoCo-specific** (DDP shows it too). The classic local-SGD consensus
  penalty (averaging drifted replicas) is the *secondary* ~0.45 BPB on top.
- ⇒ A clean long-horizon parity verdict requires fixing the recipe first
  (`fix-long-horizon`). The healthy-regime comparison (§3) is the valid DiLoCo test here.

---

## 3. DOES ANY DiLoCo VARIANT CLOSE THE GAP? (healthy regime, ≤ 86 M)

Because the recipe collapses the baseline past ~64 M (§2), the variants are compared in
the **healthy regime** (to 1000 steps = 86 M) against the clean DDP floor. Held-out BPB:

**7-GPU arms** (tokens/step = 86 016), held-out BPB:

| tokens (step) | **DDP** | β=0 K250 | mom β0.9/lr1.0 | smallK=50 β0 |
|---|---:|---:|---:|---:|
| 21.5 M (250) | **1.770** | — | 2.544 | 2.178 |
| 43.0 M (500) | **1.619** | 2.089 | 2.420 | 2.132 |
| 64.5 M (750) | **1.571** | — | 2.516 | 2.190 |
| 86.0 M (1000)| **1.596** | 2.032 | 2.576 | 2.314 |
| throughput (global tok/s) | ~31 k | 57.7 k | 56.1 k | 55.1 k |

- **None of the 7-GPU periodic-merge variants comes within ~0.4 BPB of DDP at matched
  tokens.** Outer momentum is the **worst** (~2.5, it overshoots before enough outer steps
  accumulate); small-K helps a little vs K=250 early then rises; plain β=0 is the best of
  the family (~2.0). None closes the gap.

**Hybrid** = per-step DDP within 3 islands of 2 GPUs + DiLoCo across islands every K=250
(6-GPU, tokens/step = 73 728), held-out BPB, with DDP interpolated to the SAME token count:

| tokens (step) | **hybrid** | DDP @ same tokens | gap |
|---|---:|---:|---:|
| 18.4 M (250) | 2.3841 | ~1.79 | +0.59 |
| 36.9 M (500) | **1.9147** | ~1.66 | **+0.25** |
| 55.3 M (750) | **1.8604** | ~1.59 | **+0.27** |
| 73.7 M (1000)| 1.9004 | ~1.58 | +0.32 |
| throughput (global tok/s) | **45.0 k** (1.44× DDP) | — | — |

- **The hybrid is the only variant that meaningfully closes the gap: it ~HALVES the pure-
  DiLoCo matched-token penalty** (≈ **+0.25–0.32 BPB** vs DDP, against pure local-SGD's
  ≈ +0.44), while keeping **1.44× DDP throughput** (45.0 k vs 31 k tok/s; the per-step
  intra-island all-reduce + `NCCL_P2P_DISABLE` cost vs pure DiLoCo's 57.7 k). It is a real
  throughput/efficiency compromise — but it still does **not** reach DDP parity (~0.3
  behind) and would also hit the §2 recipe collapse at long horizon.
- The hybrid required two fixes to run on this no-NVLink box: sequential subgroup-comm
  warmup AND `NCCL_P2P_DISABLE=1` (P2P over PCIe deadlocks 2-rank subgroup comm init — a
  2-rank subgroup all-reduce hangs 600 s with P2P on, completes instantly with it off).
  Merge math pre-verified in `tests/test_diloco_hybrid.py` (global all-reduce mean ==
  per-island mean when intra-island ranks are kept identical by DDP).

---

## 4. Accept / reject + days-to-100B

**DiLoCo for the 100 B seed run: not a viable loss-parity path.** The throughput win is
real (1.85× DDP → ~20 d vs 37 d) but it does **not** translate into a matched-token loss
win: plain local-SGD DiLoCo (`outer_beta=0`) carries a persistent **~0.45 BPB** matched-
token penalty in the healthy regime; outer momentum (β=0.5/0.9 × lr=0.7/1.0) and small-K
(K=50) do **not** help. The **2-GPU-island hybrid is the only variant that closes a
meaningful fraction** — it ~halves the penalty to **+0.25–0.32 BPB** at 1.44× DDP
throughput — but still does **not** reach DDP parity. The apparent β=0 convergence at
215 M is **mutual collapse**, not parity.

**The binding blocker is the training recipe, not parallelism.** Held-out BPB collapses
for BOTH DDP and DiLoCo past ~64 M tokens under the constant CMA-tuned LR (1.007e-3,
`warmup_steps=0`, no decay). **No path — DDP or DiLoCo — reaches a usable BPB at 100 B
with the current recipe.** This must be fixed first (follow-up `fix-long-horizon`:
warmup + LR decay / lower LR). The DiLoCo-vs-DDP parity question should be **re-evaluated
against a non-degrading baseline** once the recipe is fixed.

**Recommendation / days-to-100 B (at the dim1792/nh216/dep11 emender-mlp geometry):**

| path | tok/s | nominal days to 100 B | caveat |
|---|---:|---:|---|
| 7-GPU **DDP** | 31.3 k | **37 d** | exact SGD, no token penalty; **needs recipe fix to converge** |
| 6-GPU **hybrid** (3×2 islands) | 45.0 k | **~26 d** | **+0.25–0.32 BPB** matched-token (half of pure DiLoCo), does not fully close; needs recipe fix + `NCCL_P2P_DISABLE=1` |
| 7-GPU **DiLoCo K=250** (β=0) | 57.7 k | 20 d *throughput* | **+0.45 BPB matched-token penalty, does not close → effective days > 20 d**; needs recipe fix |

- **Primary recommendation:** fix the recipe first (`fix-long-horizon`); on this 7-GPU box
  use **per-step DDP** for the seed run (exact, no sample-efficiency penalty) unless the
  recipe-fixed re-evaluation shows DiLoCo parity. The 1.85× throughput of DiLoCo is not
  worth a non-closing ~0.45 BPB matched-token deficit.
- **If wall-clock is the hard constraint** and a modest token-efficiency cost is acceptable
  after the recipe fix, prefer the **hybrid** (per-step DDP within 2-GPU islands + DiLoCo
  across): measured **+0.25–0.32 BPB** (half of pure DiLoCo's penalty) at **45 k tok/s =
  1.44× DDP** (→ ~26 d nominal vs DDP's 37 d) — the best throughput/efficiency trade-off
  found. Pure DiLoCo K=250 is faster still (57.7 k, ~20 d) but its full ~0.45 BPB penalty
  does not close. Requires `NCCL_P2P_DISABLE=1` + sequential subgroup-comm warmup on this
  no-NVLink box.

### Reproduce
```
bash experiments/diloco_100b/longhorizon/phase1.sh          # sweep + β=0 long + DDP long
bash experiments/diloco_100b/longhorizon/phase2.sh          # momentum / small-K (healthy regime)
bash experiments/diloco_100b/longhorizon/phase2b_hybrid.sh  # hybrid (needs NCCL_P2P_DISABLE=1)
python experiments/diloco_100b/longhorizon/analyze.py experiments/diloco_100b/longhorizon/results.txt
python tests/test_diloco_hybrid.py                          # hybrid merge-math (CPU/gloo)
```
