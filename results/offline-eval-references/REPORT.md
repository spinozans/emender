# Offline held-out BPB: emender vs gdn2 (the trustworthy matched-token comparison)

**Task:** `offline-eval-references`. **Date:** 2026-06-17.
**Verdict discipline:** This is **descriptive (a curve), not an accept/reject verdict.** See "Scope" at the
bottom. Do **not** read an architecture verdict out of LM BPB alone.

> **Fused-kernel re-run (task: `re-run-offline`, 2026-06-17).** The numbers below were
> **regenerated on the fused Triton kernel.** The *original* committed emender curve was
> produced on the **eager** per-step recurrence: the E88/E97 fused paths in
> `ndm/models/e88_fla_hybrid.py` were gated on `self.training`, and the offline scorer runs
> `model.eval()`, so the emender E97 recurrence silently fell back to the eager scan — a
> NON-NEGOTIABLE #1 violation found and fixed in `capability-track-references` (opt-in
> `fused_inference` gate, auto-enabled by `scripts/eval_checkpoint.py:build_model`). This
> task re-scored the **same** checkpoints on the now-fused loader and confirmed the kernel
> path with a **real kernel-invocation guard** (counts actual fused-kernel calls + the eager
> sentinel, *not* the `use_triton` config flag). **Result: the fused-vs-eager difference is
> noise-level and the verdict is unchanged.** emender max |Δ BPB| = **0.00127** (mean
> 0.00038; signs vary), gdn2 |Δ BPB| ≤ **8e-6** (FLA-chunk run-to-run atomic nondeterminism;
> gdn2 was already fused, training-independent). Per-row deltas:
> `fused_vs_eager_delta.csv`; original eager curves archived as
> `archive_eager_*_heldout_bpb.csv`; per-checkpoint guard counts: `fused_guard.json`.

## What was measured

The inline training numbers are self-contradictory because of the schedule-free
`train(y)` vs `eval(x)` weight swap plus per-step noise: emender's train-tail ran ~3.29
while its best checkpoint *filename* loss is 2.997; gdn2's train-tail ran ~2.38 while its
best checkpoint filename loss is 3.21. Neither the train-tail nor the filename loss is a
trustworthy held-out signal. This task resolves it by scoring **every saved checkpoint of
both references on one shared held-out tensor**, forward-only, with the schedule-free
y-mode swap applied and the **fused** kernel.

- **Tool:** `scripts/eval_checkpoint.py` (from `build-offline-eval`). Original eager curve:
  `run_offline_eval.sh`. Fused regeneration: `run_rerun_fused.sh` → `rerun_fused.py`
  (instrumented driver), `apply_fused_results.py` (eager→fused delta + promotion),
  `verify_gdn2_fused.py` (standalone gdn2 kernel-invocation confirmation).
- **Shared held-out tensor:** `heldout_pile_tail_p50k_2048_1m.pt`
  (md5 `8e1198ab0a0a6cc2269c2312c0c8762b`, **identical** across all 3 on-disk copies),
  512 chunks × 2049 tokens = **1,048,576 scored tokens**, `bytes_per_token = 3.945132`.
  Same tensor for both models → apples-to-apples.
- **y-mode:** `train` — every checkpoint logged `schedulefree_y_swap=True` (optimizer state
  loaded, `optimizer.train()` applied to swap the stored x/eval weights to y/train weights).
- **Fused, no eager (NON-NEGOTIABLE #1) — confirmed by a real kernel-invocation guard:**
  The guard wraps the actual fused-kernel entry points + the eager per-step sentinel and
  asserts, **per checkpoint**, fused-calls > 0 AND eager-calls == 0 (this is a kernel-launch
  count, *not* the `use_triton` config flag). All 10 checkpoints PASS (`fused_guard.json`).
  - emender: `level=E97`, `use_triton=1`, `linear_state=0` → the recurrence runs the
    **sequential** fused split-edit Triton kernel `e88_triton_optimized_apply` (per-step tanh
    is non-chunkable, so the chunked path is off). At eval, the opt-in `fused_inference` gate
    (auto-enabled by `build_model`) routes `model.eval()` through this fused kernel instead of
    the eager scan. Guard: **704 fused-kernel calls** (11 E88 layers × 64 batches), **0 eager**
    per checkpoint.
  - gdn2: `level=gdn2-mlp` → `GDN2ExternalMLPLayer`; the recurrence is the external
    GatedDeltaNet-2 fused Triton kernel `chunk_gdn2` (T=2048 > 64, not-training → chunk
    dispatch; `GDN2_PATH=/home/erikg/GatedDeltaNet-2`). This path is training-independent
    (no `self.training` gate), so gdn2 was already fused. Guard: **768 `chunk_gdn2` calls**
    (12 layers × 64 batches), **0 eager**, **0** E88 layers per checkpoint;
    `verify_gdn2_fused.py` independently re-confirms `chunk_gdn2` fires on a live forward.
- **Both references are 1.3B**, matched in params: emender **1,286,589,072**, gdn2
  **1,286,713,448** (the `_100m` in the run-dir name is just the default `--params` label;
  the configs are dim/depth-specified 1.3B). tokens/step = `batch_size 4 × chunk_size 2048
  = 8192`.
- **GPU isolation:** exactly **1 idle GPU** leased via the broker for each run (broker-only
  acquisition; idle util<10% & mem<256 MiB). The reference trainings (GPUs 0/1, pids 522358,
  584124) and unrelated jobs were untouched; the reference run-dirs were read-only (checkpoint
  mtimes unchanged at completion).
- **Coverage:** the **same 5 emender + 5 gdn2 checkpoints** as the original committed curve
  were re-scored (steps emender {21500,43000,64500,86000,107500}, gdn2
  {25000,50000,75000,100000,125000}). The trainings have since produced additional
  checkpoints (now 7 emender / 6 gdn2 on disk); the fused re-run deliberately re-scores only
  the original set so the eager→fused delta is **row-matched** and unconfounded by extra
  training.

## The curve (held-out BPB vs tokens)

See `matched_token_bpb_curve.png`. Raw per-checkpoint numbers (`*_heldout_bpb.csv`,
combined in `matched_token_bpb_curve.csv`):

Values below are the **fused** numbers (canonical `*_heldout_bpb.csv`):

| tokens | emender BPB | gdn2 BPB |
|---:|---:|---:|
| 176 M | 1.4038 | — |
| 205 M | — | 1.2768 |
| 352 M | 1.2840 | — |
| 410 M | — | 1.1800 |
| 528 M | 1.3703 | — |
| 614 M | — | 1.1334 |
| 704 M | 1.2461 | — |
| 819 M | — | 1.1060 |
| 881 M | 1.2858 | — |
| 1024 M | — | 1.0861 |

- **gdn2 is monotone-descending**: 1.2768 → 1.1800 → 1.1334 → 1.1060 → 1.0861.
- **emender is non-monotone/noisy**: 1.4038 → 1.2840 → **1.3703** → 1.2461 → 1.2858. The
  1.3703 spike at 528 M (whose inline *filename* loss 3.1246 was the run's lowest!) and the
  1.2461 dip at 704 M are exactly the schedule-free y/x + noise wobble — the offline curve
  exposes that the filename loss is not a reliable ranking signal.

## Who leads at matched tokens (the answer the train loss cannot give)

Lower held-out BPB wins. Linear interpolation onto a shared token grid inside the measured
overlap **[205 M, 881 M]** (no extrapolation), full table in
`who_leads_at_matched_tokens.csv`:

| matched tokens | emender BPB | gdn2 BPB | emender − gdn2 | leader |
|---:|---:|---:|---:|:--|
| 300 M | 1.3196 | 1.2318 | +0.0878 | **gdn2** |
| 400 M | 1.3074 | 1.1846 | +0.1229 | **gdn2** |
| 500 M | 1.3564 | 1.1594 | +0.1969 | **gdn2** |
| 600 M | 1.3198 | 1.1367 | +0.1831 | **gdn2** |
| 700 M | 1.2492 | 1.1220 | +0.1273 | **gdn2** |
| 800 M | 1.2676 | 1.1086 | +0.1590 | **gdn2** |
| 881 M | 1.2858 | 1.1000 | +0.1857 | **gdn2** |

**gdn2-mlp leads emender at every matched-token point in the overlap, by +0.088 to +0.197
BPB (~0.13–0.19 BPB through most of the range).** At the last matched point (881 M tokens):
gdn2 ≈ 1.100 vs emender 1.286, a **0.186 BPB** gdn2 lead. The gap is far larger than the
noise wobble in either curve (and ≫ the ≤0.0013 BPB fused-vs-eager shift), and gdn2 wins even
against emender's *best* checkpoint in the overlap (emender's lowest overlap point is
1.2461 @704 M, still above gdn2's 1.1220 there).

## Sanity cross-check

- **CE↔BPB conversion is consistent with the tensor's bytes_per_token.** e.g. fused emender
  step 21500: `(3.838857 / ln2) / 3.945132 = 1.40383`, matches the reported `bpb=1.403831`.
  Same holds for every row → the held-out tensor's `bytes_per_token=3.945132` is applied
  correctly.
- **Magnitude/trajectory anchor on the same tensor.** The earlier (later-archived as
  "contaminated") gdn2 run was scored *inline* on this **same** tensor + same y-mode and
  reached **1.4609 BPB at 78 M tokens** while still descending. The clean offline gdn2 here
  is **1.2768 at 205 M** and continues down to 1.0861 — i.e. the clean run lands below the
  earlier curve at more tokens and keeps the same descending shape. Consistent, no scale or
  unit surprise.

## Scope / verdict discipline (important)

This is a **descriptive LM-BPB curve only.** It says gdn2-mlp has the better held-out
language-modeling BPB at matched tokens at this 1.3B/≤1 B-token operating point. It is **not**
an architecture verdict. Emender's claimed value is **capability** — specifically the
modular-quadratic separator — which is a *separate* eval and is **not** captured by LM BPB.
Per the prior 1.3B leaderboard work, LM-BPB ties/losses do not imply capability equivalence.
**Do not write an emender vs gdn2 architecture accept/reject verdict from this curve.** The honest summary
is: *at matched tokens, gdn2-mlp leads emender on held-out pile-tail BPB by ~0.13–0.19 BPB;
emender's separate capability case is untouched by this measurement.*

## Files

**Canonical (fused) outputs:**
- `emender_heldout_bpb.csv`, `gdn2_heldout_bpb.csv` — raw per-checkpoint scores (FUSED).
- `matched_token_bpb_curve.csv` — combined long-form curve.
- `who_leads_at_matched_tokens.csv` — matched-token leader table.
- `matched_token_bpb_curve.png` — overlaid curve.
- `analyze.py` — builds the combined CSV, leader table, and plot from the two raw CSVs.

**Fused re-run (task `re-run-offline`):**
- `run_rerun_fused.sh` — driver (1 idle-GPU lease) → `rerun_fused.py`.
- `rerun_fused.py` — instrumented fused re-scorer (reuses `eval_checkpoint.py`;
  real kernel-invocation guard: fused-call + eager-sentinel counters).
- `rerun_fused.log` — full fused run log (per-checkpoint guard PASS lines).
- `fused_guard.json` — per-checkpoint kernel-invocation counts (fused>0, eager==0 for all 10).
- `verify_gdn2_fused.py` — standalone confirmation that gdn2 fires `chunk_gdn2` on a live forward.
- `apply_fused_results.py` — archives eager, computes the row-matched delta, promotes fused.
- `fused_vs_eager_delta.csv` — per-row eager→fused BPB/CE delta.
- `archive_eager_emender_heldout_bpb.csv`, `archive_eager_gdn2_heldout_bpb.csv` — the original
  EAGER curves (provenance; emender was eager, gdn2 was already fused).

**Original eager run (superseded by the fused outputs above):**
- `run_offline_eval.sh` — original driver.
- `run_offline_eval.log` — original run log.
