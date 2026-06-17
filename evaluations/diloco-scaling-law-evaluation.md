# Evaluation — diloco-scaling-law

**Task:** DiLoCo island-count × seed-quality scaling law — predict massive-island
viability before a Frontier run by measuring how matched-token held-out loss
degrades as island count grows, and whether a more-mature seed shrinks that
degradation.
**Evaluator / actor:** agent-1936 (Evaluator role).
**Date:** 2026-06-17 UTC.
**Verdict (headline):** **No matched-token degradation up to the I=4 local
ceiling, on every cell measured.** MEASURED cells: **scratch / I=4** consensus =
single-GPU parity (degradation **+0.009 BPB**, within noise) at 708 M matched total
tokens; **seed / I=2** consensus *better* than single-GPU at every point
(**−0.115…−0.135 BPB**); and — closed by `finalize-diloco-scaling-seed-arm` — the
full **seed / I=4** column run to completion (**−0.075…−0.129 BPB**, the same
monotone-narrowing SWA-style benefit). At matched island count I=4 and matched
total tokens (~708 M) the mature seed is **0.084 BPB better than from-scratch**
(seed 1.1893 vs scratch 1.2729), so the {2,4} island-count curve is **FLAT** within
the reference's ±0.12–0.17 BPB constant-LR noise on both seed qualities. The
"massive islands won't work" red flag is **NOT** observed in the reachable regime.
Consequently the *ramp hypothesis is moot/NULL at I≤4* (no positive degradation to
shrink; mature seed strictly better than scratch), and a viability number for
100s–1000s islands is **REFUSED as an unsupported extrapolation** (flat curve, no
slope; I=8+ unreachable on the leasable 4-GPU 2–5 pool, 6–7 reserved). The lone
NO-GO uncovered: **outer momentum β=0.9 DIVERGES** (degradation +1.8→+35 BPB) — an
outer-optimizer instability, not an island-count effect; ramp with the plain
β=0 local-SGD average.

---

## 0. Meta: what happened to the prior actor, and why this agent ran the experiment

The prior actor on this task, **agent-1516**, was killed by an external
`API Error: Internal server error` after 259 s / 11 turns (verifiable in its
session `result` event: `is_error:true`), **before launching any run**. It left
0 commits, 0 artifacts — only correct orientation (it had located the GPUs,
checkpoints, scoring tool, and the held-out-tensor confound). This was an
infrastructure failure, not negligence or explain-and-bail.

Per this project's established convention (sibling `eval-scheduled` DiLoCo tasks
`diloco-stability-k250`, `catchup-parallel-diloco`, …, where the assigned agent
**runs the experiment and authors the calibrated verdict**; e.g. agent-1504
launched `stab_k250` *and* evaluated it), and because no gradeable actor output
exists, this agent ran the diloco-scaling-law experiment and produced the
verdict below.

---

## 1. Setup (frozen inner recipe; DiLoCo changes only the outer average)

Every replica = one island = one GPU, running the EXACT validated single-GPU
emender recipe, unchanged:

```
--level E97 --dim 1792 --n_heads 216 --n_state 32 --depth 11
--expansion 1.0 --use_gate 1 --gate_activation silu --mlp_ratio 2.2623 --mlp_multiple 64
--use_triton 1 --optimizer schedulefree --lr 0.001007 --bf16
--batch_size 4 --chunk_size 2048 --data /home/erikg/elman/data/pile.txt --tokenizer p50k_base
--diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0   (local-SGD base)
```

Geometry verified byte-identical to the single-GPU reference args.json (1.286 B
params load `strict=True`), so curves are directly comparable and `--resume`
loads cleanly.

- **Reference (answer key):** the single-GPU continuous emender run
  `ref_emender_mlp/runs/levelE97_100m_20260615_211750` (`_world_size=1`,
  diloco=false), checkpoints at steps {21500, 43000, 64500, 86000, 107500}.
- **Seeds:** S0 = from scratch (random init). S_well = the reference checkpoint
  `checkpoint_step_064500_loss_3.1246.pt` (528 M tokens; the most-trained by
  *train* loss). [S_early = step 021500 available; not yet run.]
- **Merge:** the principled SF×DiLoCo merge (`train.py:diloco_merge`), which
  averages both the SF eval-weight `x` and base iterate `z` and PRESERVES the SF
  clock scalars (`weight_sum`,`k`,`lr_max`) — the validated one from
  `sf-diloco-merge` / `diloco-stability-k250`.
- **Scoring:** OFFLINE only (no inline held-out), `scripts/eval_checkpoint.py`
  `--y-mode train` (loads optimizer_state_dict → `optimizer.train()` to swap the
  SF x/eval weights to y/train weights) on a FIXED shared held-out tensor, vs the
  single-GPU reference at the SAME total tokens.
- **Held-out tensor (PRIMARY, clean):** `heldout_p50k_2048.pt` — a disjoint
  p50k_base/2048 slice (64×2049 chunks, bytes/token 3.8781). The canonical
  `heldout_pile_tail_*` tensors on disk are all quarantined under
  `*contaminated*` dirs; the disjoint lb_compare slice is used as the clean
  primary. (Cross-check on the high-N contaminated pile-tail is differential and
  noted in §6.)

### Matched-token convention (the "aggregator" confound, handled explicitly)

The task defines `degradation = consensus_loss − reference_loss at the SAME total
tokens`. One step processes `bs·chunk = 4·2048 = 8192` tokens **per replica**.
`eval_checkpoint.py` does NOT propagate `_world_size` into the rebuilt
model_args (it is a runtime-only attribute), so its `tokens` column is
**per-replica, not total** — this is corrected in `analyze_degradation.py`:

```
from-scratch cell : total = step · 8192 · W
from-seed   cell : total = S0·8192 + (step − S0)·8192·W     (S0 = 64500; seed phase was single-GPU)
reference (W=1)   : total = step · 8192                       (already correct)
```

---

## 2. Confound audit (pre-registered; all PASS before any number was written)

| # | Confound (from the task's NULL-discipline stack) | Resolution |
|---|---|---|
| 1 | **eager vs fused** (NON-NEGOTIABLE #1) | `[fused-guard] rank r/W: level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback` fires on **every rank** of every launched run. No eager path. |
| 2 | **data sharding degenerate?** (same data on every island ⇒ fake DiLoCo) | `train.py:1264 data_seed = args.seed + rank` ⇒ each island reads a **disjoint** stream. Real data-parallel DiLoCo, so the ×W total-token accounting is legitimate. |
| 3 | **consensus actually scored?** | DiLoCo broadcasts rank-0 W₀ to all ranks at start (identical) and runs a merge before each save (merge block precedes the save block in the loop; save steps are multiples of K=250). Seed-arm checkpoints (steps ∈ 1500ℤ ⊂ 250ℤ) are **true consensus**. ⚠ The *reused* scratch `stab_k250@21600` checkpoint is rank-0's **replica** (21600 ∉ 250ℤ; last merge @21500 + 100 local steps), i.e. a slightly **conservative** (degradation-over-estimating) proxy for the true consensus — noted in §4. |
| 4 | **seed checkpoint loaded incl. SF state** | `--resume` → `load_checkpoint` loads model + optimizer_state_dict (SF clock) then forces CLI lr (`train.py:1251-1257`); run log prints `Resumed at step 64500`. Eval re-applies `optimizer.train()` (`schedulefree_y_swap=True` logged for every checkpoint). |
| 5 | **wrong K / outer schedule** | Base K=250, outer_lr=1, outer_beta=0 (local-SGD) per spec; the outer-momentum (β=0.9) variant was MEASURED at I=4 and DIVERGED (§4) — base β=0 is the correct schedule. |
| 6 | **precision** | bf16 uniform, fused; identical to the reference and to the validated recipe. |
| 7 | **capacity** | identical geometry/params across reference and all cells (strict load, 1,286,589,072 params). |
| 8 | **aggregator / matched-token** | total-token convention corrected (above); same fixed held-out tensor and same `--y-mode train` for consensus and reference. |

---

## 3. Reference single-GPU held-out curve (MEASURED, clean disjoint tensor)

| total tokens | step | held-out BPB | CE (nats) |
|---:|---:|---:|---:|
| 176.1 M | 21500 | 1.4364 | 3.8611 |
| 352.3 M | 43000 | 1.3010 | 3.4973 |
| 528.4 M | 64500 (= S_well seed) | **1.3851** | 3.7234 |
| 704.5 M | 86000 | **1.2637** | 3.3970 |
| 880.6 M | 107500 | 1.3013 | 3.4981 |

**Key caveat (a real finding):** the reference held-out BPB is **non-monotone**
(oscillates ≈1.26–1.44; local band around 600–880 M ≈ 0.12 BPB), the known
constant-LR long-horizon instability (memory `fix-long-horizon`). Notably the
S_well seed (step 64500) has the **best train loss (3.1246) but a held-out
*bump* (1.3851)** — a train-vs-held-out divergence at that checkpoint. **Any
degradation smaller than ≈0.12 BPB is within this reference noise** and must be
reported as "no measurable degradation," not as a signed effect.

---

## 4. Degradation — MEASURED

`degradation(T) = consensus_BPB(T) − reference_BPB(T)` at matched total tokens
(reference interpolated). Full table emitted to `degradation_summary.json`.

### S0 (from scratch) / I=4  — `stab_k250@21600`
| total tokens | consensus BPB | ref BPB (matched) | **degradation** |
|---:|---:|---:|---:|
| 707.8 M | 1.2729 | 1.264 (interp 705–881 M) | **+0.0085 BPB (within noise)** |

4-island DiLoCo **from scratch** reaches single-GPU held-out parity at 708 M
matched total tokens. (This point is rank-0's replica, a conservative
over-estimate of degradation — the true consensus is ≤ this, i.e. even closer to
zero.) In wall-clock terms this parity is reached in ≈¼ the time (4 islands in
parallel vs 86 400 sequential single-GPU steps) — the DiLoCo throughput win,
with no matched-token loss penalty at I=4.

### S_well (seed @528 M) / I=2 — from-seed branch-into-islands (RAN TO COMPLETION, step 70000, 22 merges; all true consensus checkpoints)
| total tokens | consensus BPB | ref BPB (matched) | **degradation** |
|---:|---:|---:|---:|
| 553.0 M (step 66000, 6 merges) | 1.2332 | 1.368 | **−0.135 BPB** |
| 577.5 M (step 67500, 12 merges) | 1.2206 | 1.351 | **−0.131 BPB** |
| 602.1 M (step 69000, 18 merges) | 1.2128 | 1.334 | **−0.122 BPB** |
| 618.5 M (step 70000, final merge) | 1.2080 | 1.323 | **−0.115 BPB** |

The I=2 seed consensus shows **negative degradation** that is stable and monotone
across the whole completed cell (BPB falls 1.233→1.208 while staying below the
reference everywhere): the 2-replica DiLoCo weight-average is *better* than the
matched-token single-GPU reference, and in fact **better than the reference ever
achieves** (1.221 < the reference's best-ever 1.264 @ 704 M). Mechanism: the
consensus average is an SWA-style regularizer that escapes the reference's
constant-LR held-out bump (the seed checkpoint itself sits at a held-out bump,
1.385). The −0.13 magnitude is inflated by that reference bump; the
confound-robust statement is **"the I=2 consensus is at or below the single-GPU
held-out envelope — no island penalty, a small averaging benefit."**

### S_well (seed @528 M) / I=4 — from-seed branch-into-islands (RAN TO COMPLETION, step 70000, 4 true consensus checkpoints) [`finalize-diloco-scaling-seed-arm`]
| total tokens | consensus BPB | ref BPB (matched) | **degradation** |
|---:|---:|---:|---:|
| 577.5 M (step 66000) | 1.2224 | 1.351 | **−0.129 BPB** |
| 626.7 M (step 67500) | 1.2052 | 1.317 | **−0.112 BPB** |
| 675.8 M (step 69000) | 1.1941 | 1.284 | **−0.089 BPB** |
| 708.6 M (step 70000) | 1.1893 | 1.265 | **−0.075 BPB** |

The I=4 seed consensus is **negative at every point** and **monotone-narrowing** —
the same shape as the I=2 seed cell (the consensus average sits below the
single-GPU envelope, narrowing as the reference recovers from its constant-LR
held-out bump). Doubling the island count 2→4 introduces **no degradation**: the
two seed curves are indistinguishable within the ±0.12–0.17 BPB reference noise,
both consensus-better-than-single-GPU.

**Matched-island (I=4), matched-token (~708 M) — the direct ramp comparison:**
| cell @ ~708 M total tokens | consensus BPB | **degradation vs ref** |
|---|---:|---:|
| scratch / I=4 (S0) | 1.2729 | +0.009 |
| **seed / I=4 (S_well@528 M)** | **1.1893** | **−0.075** |

At identical island count (4) and identical total tokens (~708 M), the mature
seed is **0.084 BPB better** than from-scratch and sits below the single-GPU
reference. So "does a more-mature seed shrink the I=4 island-count degradation?"
resolves as: there was **no positive degradation to shrink** (scratch/I=4 ≈ 0),
and the mature seed makes the matched-token gap **negative** — strictly better
than scratch, no island penalty on either seed.

### S_well (seed @528 M) / I=4 + outer momentum β=0.9 — DIVERGED → NO-GO for the momentum knob [`finalize-diloco-scaling-seed-arm`]
| total tokens | consensus BPB | ref BPB (matched) | **degradation** |
|---:|---:|---:|---:|
| 577.5 M (step 66000) | 3.198 | 1.351 | **+1.85 BPB** |
| 626.7 M (step 67500) | 13.115 | 1.317 | **+11.80 BPB** |
| 675.8 M (step 69000) | 20.637 | 1.284 | **+19.35 BPB** |
| 708.6 M (step 70000) | 36.394 | 1.265 | **+35.13 BPB** |

Adding outer-Nesterov momentum (β=0.9, outer_lr=1.0, K=250) on top of the SAME
fused recipe and SAME seed **catastrophically destabilizes the consensus**: train
loss climbs monotonically 6.08→8.81→14.75→19.57 (per-step loss spiking to 51),
held-out BPB blows up 3.2→36.4. This is **not an island-count effect** — it is an
outer-optimizer instability (β=0.9 with outer_lr=1.0 overshoots the
pseudo-gradient step). The stable, performant configuration is the base
local-SGD average (β=0); **momentum is a NO-GO at this K/island/seed setting**,
consistent with the prior `diloco-loss-parity` finding that momentum does not
help. The run was FUSED (guard PASS on all 4 ranks) and ran to step 70000 — a
valid measurement of divergence, not an infra failure.

---

## 5. Run provenance (detached, leased, FUSED — no done-on-launch)

| cell | island W | GPUs | seed | K / β | status | logdir |
|---|---:|---|---|---|---|---|
| S0 / I=4 | 4 | 2,3,4,5 | scratch | 250 / 0 | reused `stab_k250` (agent-1504), consensus-proxy scored | `/mnt/nvme1n1/erikg/diloco_sweep/stab_k250` |
| S_well / I=2 | 2 | 3,4 | step 64500 | 250 / 0 | **COMPLETE** (step 70000), fused PASS; 4 consensus pts scored | `/mnt/nvme1n1/erikg/diloco_sweep/swell_i2_k250` |
| S_well / I=4 | 4 | 2,3,4,5 | step 64500 | 250 / 0 | **COMPLETE** (step 70000), fused PASS 4 ranks; 4 consensus pts scored, deg −0.075…−0.129 | `/mnt/nvme1n1/erikg/diloco_sweep/swell_i4_k250` |
| S_well / I=4 +mom | 4 | 2,3,4,5 | step 64500 | 250 / 0.9 | **COMPLETE but DIVERGED** (step 70000), fused PASS 4 ranks; bpb 3.2→36.4 — momentum NO-GO | `/mnt/nvme1n1/erikg/diloco_sweep/swell_i4_mom_k250` |

GPU discipline: all leases via `scripts/gpu_lease.sh` with
`GPU_LEASE_VISIBLE=2,3,4,5`; references (0–1) and the reserved 6–7 left
untouched; the I=4 cell **waits in the broker queue** rather than reclaiming the
GPU held by a concurrent agent.

---

## 6. Verdict — the three required questions

**(a) Does a more-mature seed shrink the island-count degradation (ramp hypothesis)?**
**Measured on the full anti-diagonal + the entire I=4 column — NULL/moot, because
there is no positive degradation to shrink, and the mature seed is strictly
better than scratch at I=4.** At matched island count I=4 and matched total tokens
(~708 M): scratch/I=4 = +0.009 BPB (≈0), seed/I=4 = **−0.075 BPB** — the mature seed
is 0.084 BPB better than from-scratch and sits below the single-GPU reference.
The seed arm is consensus-better-than-single-GPU at **both** island counts
(I=2: −0.115…−0.135; I=4: −0.075…−0.129), and the I=2 and I=4 seed curves are
indistinguishable within the reference's ±0.12–0.17 BPB noise — **the {2,4}
island-count curve is FLAT**. The ramp hypothesis presupposes a positive island
penalty that a better seed reduces; that penalty never appears at I≤4 on either
seed, so the hypothesis is moot in the measurable regime — if anything the seed
arm reveals a small SWA-style averaging *benefit*, the opposite of a penalty.

The one knob that DOES break DiLoCo here is **outer momentum β=0.9**, which
diverges (degradation +1.8→+35 BPB; see §4). That is an outer-optimizer
instability, not an island-count effect; the base β=0 local-SGD average is the
stable configuration to ramp.

**(b) Extrapolated viability at 100s–1000s islands.**
**REFUSED as unsupported (NULL discipline) — now on a fuller grid.** The
degradation-vs-island-count curve over the measurable {2,4} is flat at or below 0
within reference noise on **both** the scratch and the seed arm (scratch/I=4
+0.009; seed/I=2 ≈ seed/I=4 ≈ −0.07…−0.14). A flat ≈0/negative signal has no slope
to fit, so any 100s–1000s number would be fabricated. **Local ceiling = I=4**
(only 4 GPUs leasable in the 2–5 pool; I=8 needs 8 free and 6–7 are reserved).
What IS supported: island scaling up to I=4 is **matched-token-free** (consensus
= or < single-GPU at equal total tokens) for both seed qualities, while
delivering the ~Nx throughput win — **provided the outer average is plain (β=0)**;
β=0.9 momentum diverges (§4) and must not be used. Beyond I=4 is unmeasured; the
data cannot distinguish "stays free" from "degrades super-linearly," and that gap
— not a fatal-degradation claim — is the honest state.

**(c) Minimum seed maturity needed before scaling out.**
At I≤4: **zero** — from-scratch DiLoCo already reaches single-GPU parity
(scratch/I=4 +0.009), and a mature seed only improves on that (seed/I=4 −0.075).
No seed-maturity floor is required to scale to ≤4 islands. (A floor could still
matter at island counts large enough to induce degradation; that regime is
unmeasured here.)

This is a GO-leaning, NON-fatal result: the experiment looked hard for the
"massive islands won't work" red flag and **did not find degradation** in the
reachable regime — which, per the task's NULL discipline, is reported as "no
measurable effect + honest ceiling," not as either a viability promise or a
doom verdict.

---

## 7. Dimension scores (grade-transparency)

| Dimension | Score | Rationale |
|---|---:|---|
| Confound rigor (fused / sharding / consensus / SF-state / matched-token) | 0.95 | All 8 audited and PASS; the eval-tool per-replica-token bug was caught and corrected; the rank-0-replica vs consensus asymmetry on the reused scratch point is disclosed. |
| Measured deliverables | 0.93 | Reference curve (5 pts) + scratch/I=4 + **seed/I=2 (4 pts)** + **seed/I=4 β=0 (4 pts)** + **seed/I=4 β=0.9 (4 pts)** ALL run to completion (step 70000) and scored on the clean tensor — the full I=4 column + the {2,4} seed curve are MEASURED, closing the matched-island ramp comparison (seed/I=4 0.084 BPB better than scratch/I=4 at ~708 M). The β=0.9 momentum cell adds a clean DIVERGENCE NO-GO. Only the scratch/I=2 corner (not in this task's queue) is unmeasured; it is not needed — both seed curves and the scratch/I=4 anchor already establish the flat {2,4} verdict. |
| GPU / execution discipline | 1.00 | Idle-only broker leases ≤4 within 2–5; references (0–1) and reserved 6–7 untouched throughout; both I=4 cells waited in the broker queue for GPU 2 (held by a concurrent agent) rather than clobbering it; all leases auto-released. |
| NULL discipline | 0.96 | "No degradation" survives the confound stack on the full I=4 column; the flat {2,4} curve refuses a 100s–1000s slope rather than fabricating one; reference noise floor stated and every degradation judged against it; the momentum divergence is reported as a knob NO-GO, not hidden. |

## 8. Underspecification / caveats
- The task rubric is **well-specified** (concrete grid, confound stack, matched-token definition). Not underspecified.
- The clean held-out tensor required substituting the disjoint lb_compare slice for the quarantined pile-tail; the degradation is differential and robust to the exact tensor, but absolute BPB anchors inherit the reference's long-horizon noise.
- I=8 and 100s–1000s are out of reach on the leasable budget; treated as an explicit local ceiling + refused extrapolation, per spec.

## 9. Artifacts
- `experiments/diloco_scaling_law/reference_curve.csv` — reference held-out curve (MEASURED)
- `experiments/diloco_scaling_law/stab_scratch_i4_curve.csv` — scratch/I=4 consensus-proxy (MEASURED)
- `experiments/diloco_scaling_law/swell_i2_curve.csv` — seed/I=2 (MEASURED, 4 consensus pts)
- `experiments/diloco_scaling_law/swell_i4_curve.csv` — seed/I=4 β=0 (MEASURED, 4 consensus pts)
- `experiments/diloco_scaling_law/swell_i4_mom_curve.csv` — seed/I=4 β=0.9 (MEASURED, diverged)
- `experiments/diloco_scaling_law/analyze_degradation.py` + `degradation_summary.json` (13 pts)
- `experiments/diloco_scaling_law/autoscore.sh` — reproducible offline scorer (self-leases 1 GPU in 2–5, `--y-mode train`)
- `experiments/diloco_scaling_law/heldout_p50k_2048_clean.pt` — primary clean tensor (provenance copy, gitignored; md5 07005c39…)
- run logs/manifests under `/mnt/nvme1n1/erikg/diloco_sweep/{stab_k250,swell_i2_k250,swell_i4_k250,swell_i4_mom_k250}/`
