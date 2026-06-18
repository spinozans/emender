# DiLoCo seed-race I=4 — the read (track / beat / degrade)

**Task:** diloco-seed-race-2 (Frontier dress-rehearsal). Seed a *mature*
single-GPU emender checkpoint (step 150500 ≈ 1.233 B tokens), continue it as a
**4-island DiLoCo** run with **plain averaging**, race ~3 B tokens, and ask
whether I=4 seeded DiLoCo **tracks / beats / degrades** the single-GPU
continuation at **matched total tokens**.

## Verdict: **BEAT** (DiLoCo wins at matched tokens by ≈ 0.10 BPB)

I=4 seeded DiLoCo, plain-averaged, **beats** the single-GPU emender continuation
at matched total tokens. The SWA-style averaging benefit measured at I≤4 from a
528 M seed in `diloco-scaling-law` (degradation −0.075 … −0.135 BPB) **holds and
strengthens** from a mature 1.233 B seed over a long race.

### Effect size (windowed, full bracketed overlap window)

The single-GPU reference run ends at step 244141 = 2.0 B total tokens, so the
matched-token head-to-head lives in the overlap window [1.233 B, 2.0 B]. A
DiLoCo point is "bracketed" (a genuine matched-token comparison, not
extrapolation) once the reference has trained past its token count. With both
runs complete, **all 7 overlap points are bracketed**:

| DiLoCo step | total tokens | DiLoCo BPB | ref BPB (interp) | degradation |
|---|---|---|---|---|
| 153000 | 1.315 B | 1.1453 | 1.2317 | **−0.0864** |
| 156000 | 1.413 B | 1.1309 | 1.2348 | **−0.1039** |
| 159000 | 1.511 B | 1.1215 | 1.2365 | **−0.1150** |
| 162000 | 1.610 B | 1.1147 | 1.2382 | **−0.1236** |
| 165000 | 1.708 B | 1.1088 | 1.2399 | **−0.1311** |
| 168000 | 1.806 B | 1.1049 | 1.2228 | **−0.1179** |
| 171000 | 1.905 B | 1.1014 | 1.1834 | **−0.0820** |

**7 / 7 points favour DiLoCo.** Three independent effect-size views, all
agreeing on BEAT and robust to the reference's (real) checkpoint noise:

- **mean = −0.1086 BPB, median = −0.1150 BPB** (range −0.082 … −0.131).
- **Sign test (distribution-free): 7 / 7 negative, two-sided p = 0.0156.** This
  is the headline statement — it does not depend on any noise-band magnitude.
- **Cohen-d vs the robust mature-region std (0.031) = −3.5**; vs the
  outlier-sensitive mature max−min band (0.071) = −1.5; vs the within-window std
  = −5.8. (The full-curve band, 0.233, is dominated by the 1.40 BPB warmup point
  and is not the right null.)

The verdict is gated to require ≥ 3 *bracketed* points (not a single noisy point,
not extrapolation), per the task's "effect sizes, not single noisy points"
discipline.

### The reference's late descent — and why it makes BEAT *stronger*

The single-GPU reference is not perfectly flat: it plateaus at ~1.23–1.24 BPB
from 1.06 B to 1.76 B, then has a genuine **late-stage descent to ~1.17 BPB**
over its final 0.24 B tokens (236500 → 1.1702 *and* 244141 → 1.1726 — two
checkpoints at ~1.17, so a real trend, not a single noisy point). This *widens*
the mature noise band to 0.071 and *narrows* the matched-token gap at the top of
the overlap (−0.082 at 1.905 B vs −0.131 at 1.708 B). Crucially, **DiLoCo is
still ahead at every one of the 7 points** even against the reference's improved
late checkpoints — the BEAT survives the reference's best showing. And DiLoCo
keeps going: it descends to **1.052 BPB at 4.17 B**, ~0.12 below where the
single-GPU run *ended* (2.0 B → 1.173), i.e. scaling out buys both a
matched-token win and a much deeper reachable floor per wall-clock.

### Mechanism (why BEAT, and why it is real)

The overlay (`seed_race_i4_overlay.png`) shows the story plainly: from the shared
seed at 1.233 B, the **single-GPU reference plateaus at ~1.23 BPB** (it is a
mature constant-LR schedule-free run near its floor), while the **I=4 DiLoCo
consensus descends smoothly and monotonically** to 1.081 BPB at 2.49 B and
falling. At every matched token count past the seed, the DiLoCo consensus is
lower.

Crucially this is a matched-*token* win with **fewer optimizer steps**: at a
given total-token budget T, the single-GPU run has taken T/8192 steps whereas
each DiLoCo island has taken only (T−seed)/32768 + seed-steps. DiLoCo reaches a
lower held-out BPB per token using a 4× larger effective batch (4 islands × bs4)
plus the periodic cross-island averaging — the large-batch + SWA efficiency that
is the entire premise of scaling out. (Per wall-clock the win is far larger
still: 4× the tokens/second.)

## Confound audit — the BEAT is not an artifact

A "DiLoCo degrades" reading would have to survive the confound stack; a "DiLoCo
beats" reading must equally rule out an *unfair-comparison* artifact. All clear:

1. **Seed loaded with full state incl. schedule-free `z`.** `--resume` →
   `load_checkpoint` restored `model_state_dict` + `optimizer_state_dict`; the
   training loss continued seamlessly (3.0442 seed → 2.97 at first save), and the
   offline scorer reports `schedulefree_y_swap=True`. Not a cold re-init.
2. **Plain averaging, β=0.** Manifest: `--diloco_outer_lr 1.0 --diloco_outer_beta
   0.0`. (β=0.9 DIVERGES — `diloco-scaling-law` measured +1.8 … +35 BPB — and was
   deliberately avoided. No divergence observed here; merges are stable.)
3. **Matched TOTAL tokens.** Both runs seed from the *identical* step-150500
   checkpoint. Tokens recomputed with the seed-aware piecewise formula
   `seed_step*8192 + (step−seed_step)*8192*W`, not eval_checkpoint's per-replica
   column (which 4×-overcounts the pre-seed phase).
4. **FUSED, no eager.** Both training (`use_triton 1`, fused-guard asserts on all
   4 ranks, "NO eager fallback") and scoring (E97 fused) run the recurrence
   through the fused Triton kernel. NON-NEGOTIABLE #1 satisfied.
5. **Consensus checkpoints.** `save_every=3000 = 12 × diloco_k(250)`, so every
   saved rank-0 checkpoint lands on a merge step → it is the post-merge
   cross-island average, not one island's local weights.
6. **`gate_activation=silu` passed explicitly** (train.py default is `sigmoid`).
   The offline scorer's `strict load OK` + correct-magnitude BPB confirm the
   architecture is byte-identical to the reference — no silent activation swap.
7. **Same held-out tensor, same scoring basis.** Both curves scored on the
   identical shared pile-tail tensor (md5 `8e1198ab`), `--y-mode train`,
   forward-only, identical `eval_checkpoint.py`. The reference BPB values
   reproduce `offline-eval-references` byte-for-byte (e.g. 21500 → 1.403510,
   43000 → 1.283853), so the pipeline is correct and the two curves are on the
   same axis.

There is no easier-basis, mis-token, or cold-seed artifact: the comparison is
fair, and DiLoCo wins it.

## Reproduce

```
scripts/run_diloco_seed_race_i4.sh    # detached 4-island launch (seeds from newest ckpt)
scripts/score_seed_race_i4.sh         # incremental offline scoring of both run-dirs + analysis
```

Data: `reference_heldout_bpb.csv`, `diloco_i4_heldout_bpb.csv`,
`seed_race_i4_degradation.csv`, `seed_race_i4_verdict.json`,
`seed_race_i4_overlay.png`. Methodology: `README.md`.

## Run accounting (complete)

- **DiLoCo race finished:** step 150500 → 240000 = **2.93 B additional tokens**
  (~3 B), 25.24 h wall, ~35 k global tok/s steady, **358 cross-island merges**,
  fused on all 4 ranks throughout, clean shutdown ("Training complete"; the
  final merge was correctly SKIPPED because step 240000 % diloco_k == 0, so the
  saved checkpoint is already consensus — no spurious double outer-step).
- **Single-GPU emender reference finished** at step 244141 = 2.0 B (its own
  endpoint), scored on the same tensor; it is the comparison baseline.
- Both runs seed from the identical step-150500 checkpoint; 30 DiLoCo consensus
  checkpoints + 12 reference checkpoints scored offline on the shared tensor.

## Bottom line

I=4 seeded DiLoCo with plain averaging **BEATS** the single-GPU emender
continuation at matched tokens (7/7 bracketed points, mean −0.109 BPB, sign-test
p = 0.0156, d=−3.5 vs robust noise), and keeps descending well past where the
single-GPU run ended. The SWA-style averaging benefit measured at I≤4 from a
528 M seed in `diloco-scaling-law` **holds and strengthens** from a mature
1.233 B seed across a full ~3 B race. The Frontier dress-rehearsal — train
locally, then scale out and race ahead — works.
