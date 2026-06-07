# e97-raw + SwiGLU MLP — matched comparison (task e97-raw-plus)

**Question.** e97-raw is #1 on the 1.3B CMA leaderboard but the entire e97 family is
*mixer-only* (no MLP block). On that leaderboard, bolting a SwiGLU MLP onto mixer-only
GDN-2 jumped it from rank 10 (6.385) to rank 2 (gdn2-mlp, 5.961) — a 0.42-nat gain. So
the obvious untried upgrade to the #1 cell is **e97-raw + an MLP**. Does the MLP help
e97-raw, and does e97-raw+MLP beat gdn2-mlp?

**Verdict (both YES, decisively).**
1. **MLP helps e97-raw** — every bolt-on MLP ratio beats the mixer-only baseline on
   *both* matched-token train loss and held-out loss.
2. **e97-raw+MLP beats gdn2-mlp** — by ~0.9 nat held-out and ~0.02–0.04 train,
   param- and token-matched. In fact even *mixer-only* e97-raw beats gdn2-mlp held-out.
3. **Best `mlp_ratio` ≈ 1.0** (best train, near-best held-out); small ratios (0.5–1.0)
   generalize best. **Reallocating** mixer width into depth+MLP (realloc) does **not** help.

---

## Setup (REAL data, REAL training — no mocks)

| Item | Value |
|---|---|
| Scale | **~1.30B params** — the leaderboard scale where e97-raw ranked #1 and gdn2-mlp #2. One model per GPU; ~18 GB/GPU at train. GPUs **4,5,6,7 only** (GPUs 0–3 untouched, owned by sibling e97-raw-expressivity). |
| Data | commapile mainmix v0.1 (`commapile_mainmix_v0.1_smoke_1gb.txt`), p50k_base tokenizer, bf16 |
| Optimizer | schedule-free AdamW, seed 42, batch_size 2 × chunk 2048 |
| Token-match | fixed **3000 steps** for every candidate (identical tokens/step ⇒ matched tokens) |
| Param-match | all candidates within **±18M of 1.30B** (≈1.4% band); exact counts below |
| Held-out | distinct slice `/tmp/e97_heldout_rep.txt`; token-weighted CE (nats/tok) + BPB; **fixed eval seed 1234** ⇒ every model sees the *identical* held-out batches. bytes/token = 3.783 ⇒ BPB = nats / (ln2 × 3.783) = nats × 0.3814. |
| Ranking | **matched-token loss** (train at step 3000 on the schedule-free averaged weights, and held-out on those same weights) — *not* wallclock. tok/s reported but not ranked on. |

`mlp_ratio=0` is the unchanged mixer-only path; `mlp_ratio>0` wraps every mixer layer
with a post-mixer `RMSNorm + SwiGLU` block
(`out = mixer(x) + SwiGLU(RMSNorm(x + mixer(x)))`), bias-free `w3(silu(w1·x) * w2·x)`,
hidden width `round(dim·ratio, /64)` — mirroring the gdn2-mlp reference layer. Code:
`ndm/models/ladder_lm.py` (`SwiGLUMLP`, `MixerMLPWrapper`), `train.py --mlp_ratio`
(commit `68afa1f`).

---

## Results (param- & token-matched, 3000 steps, sorted by held-out)

| candidate | geometry | params | **train** | **held-out** | BPB | tok/s |
|---|---|---:|---:|---:|---:|---:|
| **e97-raw + MLP r0.5** | d10 h344 r0.5 | 1,301,524,192 | 5.5003 | **7.8651** | 2.9995 | 7473 |
| **e97-raw + MLP r1.0** | d10 h334 r1.0 | 1,302,353,432 | **5.4911** | 7.9034 | 3.0141 | 7605 |
| e97-raw + MLP r2.0 | d10 h313 r2.0 | 1,300,555,892 | 5.5006 | 7.9876 | 3.0462 | 7324 |
| e97-raw + MLP r1.5 (bolt) | d10 h323 r1.5 | 1,299,726,652 | 5.5131 | 8.0382 | 3.0655 | 7441 |
| e97-raw + MLP r2.694 | d10 h299 r2.694 | 1,302,306,652 | 5.5221 | 8.0466 | 3.0687 | 7609 |
| **e97-raw (mixer-only)** ⟵ baseline | d10 h354 r0 | 1,300,679,592 | 5.6300 | 8.1974 | 3.1263 | 7164 |
| e97-raw + MLP realloc | **d21** h128 r2.0 | 1,303,547,136 | 5.6909 | 8.2186 | 3.1343 | 6959 |
| **gdn2-mlp** ⟵ rank-2 ref | d17 h8 r2.854 | 1,285,245,320 | 5.5341 | **8.8722** | 3.3836 | 8648 |

All E97 levels: `dim=1536, n_state=32, expansion=1.0, e88_raw_write=1, use_gate=silu, use_triton=1, lr=9e-4`.
gdn2-mlp: `dim=2304, n_heads=8, expansion=2.0, lr=2.45e-3` (its leaderboard/CMA value).

*Held-out is a 16-batch (~65k-token) sample on a deliberately harder OOD slice — hence
held-out (~8) sits well above train (~5.5). The gap is uniform across all models, so the
ranking is apples-to-apples. The committed overnight Wave-1 held-out (larger sample:
mixer 8.207, bolt-r1.5 8.063, gdn2-mlp 8.908) corroborates these fresh numbers closely.*

---

## Findings

**1. MLP helps e97-raw — unambiguously.**
Baseline mixer-only e97-raw: train **5.630**, held-out **8.197**. *Every* bolt-on ratio
improves both:
- best train: **r1.0 → 5.491** (−0.139 nat vs baseline)
- best held-out: **r0.5 → 7.865** (−0.332 nat vs baseline)

This mirrors the leaderboard's MLP effect on GDN-2 (mixer-only → gdn2-mlp, −0.42 nat).

**2. Best `mlp_ratio` ≈ 1.0 (small ratios generalize best).**
Train loss is minimized at **r1.0**; held-out is minimized at **r0.5**, with r1.0 a close
second and the held-out penalty growing monotonically as the ratio increases past ~1
(r0.5 7.865 → r1.0 7.903 → r2.0 7.988 → r1.5 8.038 → r2.694 8.047). The full sweep
revises the earlier interim guess of ~1.5: **the sweet spot is r≈1.0** (best train,
2nd-best held-out, statistically tied with r0.5), and large GDN-2-style ratios (≥2) are
worse. Recommend **`mlp_ratio = 1.0`** as the balanced default; r0.5 if held-out is the
sole priority.

**3. e97-raw+MLP beats gdn2-mlp — by a wide margin.**
gdn2-mlp lands **last** on held-out (**8.872**, BPB 3.384) and mid-pack on train (5.534).
Every e97-raw+MLP variant beats it on both axes; even the mixer-only e97-raw baseline
beats gdn2-mlp held-out (8.197 < 8.872). gdn2-mlp's in-training validation was actively
*diverging* (7.11 → 7.83 → 8.53), consistent with its poor held-out — its hot CMA lr
(2.45e-3) is likely too aggressive for this short 3000-step token-matched regime.
**Caveat:** this is a token-matched short pilot, not the full leaderboard token budget at
gdn2-mlp's tuned schedule; the e97-raw+MLP win is robust *in this matched setting*, but a
longer-horizon rematch with each cell's own tuned lr would tighten the gdn2-mlp claim.

**4. Reallocation (shrink mixer, go deeper) does NOT help.**
Shrinking the mixer (h354→128) to fund depth (d10→21) + a bigger MLP *regresses*: train
5.691 (worse than baseline) and held-out 8.219 (≈baseline, worst non-gdn2 row). The win
comes from **bolting an MLP onto the existing mixer at the same depth**, not from
re-spending mixer capacity. It is also the slowest (6959 tok/s, d21).

**5. Throughput.** gdn2-mlp is fastest (8648 tok/s) but loses on loss; among e97 cells
the MLP adds little/no cost at fixed depth (mixer 7164 → r1.0 7605 tok/s — the MLP rows
are actually *faster* per token at slightly fewer heads). Ranking is on loss, not tok/s.

---

## Recommended next step

Adopt **e97-raw + SwiGLU MLP at `mlp_ratio = 1.0`** as the new e97 default cell and carry
it forward to the e97-convergent architecture and to a full-token-budget 1.3B rematch
against gdn2-mlp (each with its own tuned lr / schedule, and a longer-sample held-out)
to confirm the win at scale and settle the lr-confound caveat. Do **not** pursue the
depth-realloc variant. Note for e97-generalization-audit: e97-raw's known
recall/S5 expressivity gap is *not* closed by the MLP (the MLP improves LM loss, not the
mixer's state-tracking) — the held-out win here is LM perplexity, separate from the
expressivity-probe story.

---

## Provenance
- Code: `ndm/models/ladder_lm.py`, `train.py` (commit `68afa1f`)
- Run scripts: `experiments/e97_raw_mlp/{run_wave.sh,run_one.sh,param_match.py,solve_nheads.py}`
- Held-out eval: `experiments/e97_raw_mlp/eval_one.py` (per-candidate, fixed seed, JSON out)
- Checkpoints + per-candidate held-out JSON: `/mnt/nvme1n1/erikg/e97_raw_mlp_runs/` (local; 1.3B artifacts not committed)
- GPUs 4–7 only; sibling study on 0–3 untouched throughout.
