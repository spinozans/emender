# E97 WITHIN-LAYER heterogeneous-head study — authoritative results

**Task:** `e97-within-layer` · **Date:** 2026-06-07 · all 8 GPUs · REAL data, REAL
training, no mocks.

## Why / what this is

The within-layer fused-head infra landed on `main` (`c2a7313`): a single
`TypedHeadMixtureLayer` holds many heads of MIXED types **in parallel** inside ONE
layer (head-type fractions), summed into the residual. The two prior hybrid studies
(`e97-gdn-hybrid`, `e97-convergent`) predated this merge and silently regressed to
**interleaved whole-layers** — retired. This is the clean **within-layer**
authoritative study.

`TypedHeadMixtureLayer` `TYPE_NAMES` (canonical order):

```
[gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell, e97_raw, e97_delta]
   idx 0         1        2      3      4          5                  6         7
```

A config is per-layer head-type **fractions** via `head_type_logits` (softmax →
largest-remainder integer head counts). Half/half mixes use equal logits (0) on the
two active type indices and −30 (≈ −∞ → 0 heads) elsewhere; pure arms put all mass
on one index. `n_heads=32` ⇒ half/half = 16/16.

### Architecture is WITHIN-LAYER, not interleaved

- Expressivity: `--layer_pattern typed-gdn2 typed-gdn2 typed-gdn2 typed-gdn2` —
  EVERY layer is a `TypedHeadMixtureLayer`; head-type composition is purely
  within-layer. (Interleaved — the retired approach — would be
  `--layer_pattern E97 gdn E97 gdn`.)
- LM: `LadderLM(level='typed-gdn2-lm')` — every layer a within-layer
  `TypedHeadMixtureLayer` + a SwiGLU MLP (`mlp_ratio 1.0`, per Study B
  `E97_RAW_MLP_RESULTS`).

## The matrix (binding)

`{backbone} × {recall} (+ MLP on the LM axis)`:

- backbone ∈ {`e97_raw` (idx6), `e97_delta` (idx7)}
- recall ∈ {`none`, `gdn` (idx0, `allow_neg_eigval=0`), `gdn-neg` (idx0,
  `allow_neg_eigval=1` — the GDN-2 negative along-key eigenvalue for tracking)}

| config | per-layer head allocation (n_heads=32) | gdn_allow_neg |
|---|---|---|
| `e97_raw (pure)`     | 32 e97_raw                | — |
| `e97_raw + gdn`      | 16 gdn  + 16 e97_raw      | 0 |
| `e97_raw + gdn-neg`  | 16 gdnN + 16 e97_raw      | 1 |
| `e97_delta (pure)`   | 32 e97_delta              | — |
| `e97_delta + gdn`    | 16 gdn  + 16 e97_delta    | 0 |
| `e97_delta + gdn-neg`| 16 gdnN + 16 e97_delta    | 1 |
| `gdn-neg (ref)`      | 32 gdn-neg (recall workhorse bar) | 1 |

The two `*_none` arms are the **pure backbone** configs (first-class). `gdn-neg (ref)`
/ its `+MLP` LM twin `gdn2-mlp` are the established recall and Study-B references.

## Axis 1 — Expressivity battery (within-layer mixer)

Probes (existing battery, `experiments/expressivity_tasks/tasks/`): `s5_permutation`
(TRACK), `anbncn_viability` (COUNT), `iterated_nonlinear_map` (NONLIN),
`flag_hold_recall` K=4 (LATCH), `mqar_recall` (RECALL). 3 seeds {42,123,456}; train
T=128; eval length-extrap T∈{128,256,512,1024}. dim=256 / 32 heads / N=V=32, depth 4
(~6–7.4M). schedule-free AdamW, 5000 steps. bf16 autocast ⇒ E97 heads FUSED. Runner:
`experiments/expressivity_tasks/run_e97_within_layer.py`.

## Axis 2 — LM held-out screens (time-bounded, FUSED)

`train.py --level typed-gdn2-lm` on REAL commapile
(`commapile_mainmix_v0.1_smoke_1gb.txt`, p50k_base tokenizer), dim=768 / depth=12 /
48 heads / N=32, `--mlp_ratio 1.0`, bf16, schedule-free. **Time-bounded**
`--train_minutes 17.5` (leaderboard methodology, NOT slow token-matched — the fused
kernel sustains ~27–40M tok/screen, verified). Held-out BPB on the schedule-free
**averaged** weights over a distinct slice (`/tmp/e97_heldout_rep.txt`) via the
opt-in `train.py --final_heldout_eval` (`FINAL_HELDOUT_BPB`; BPB = CE_nats / ln2 /
3.783 bytes-per-token). Runner: `run_e97_within_layer_lm.py`.

## No-eager-fallback verification (validation item 1)

`experiments/expressivity_tasks/verify_e97_within_layer_heads.py` — **ALL CHECKS
PASSED**:
- **No silent eager fallback**: the fused split-edit Triton kernel is call-counted —
  exactly one call per fused E97 head type; the loud guard raises on fp32-with-cast-off
  rather than degrading to the eager T-scan.
- **Parity** (fused bf16 vs eager ref, fwd+bwd): rel-L2 ≈ 1.1–2.2e-2 at
  T=128/512/1024 for both `e97_raw` and `e97_delta`, + unaligned T=511 (causal
  zero-pad+truncate exact).
- Heterogeneous LM (all 8 head types in one layer) sustains an 18-min fused screen ≈
  41M tok; fusing the E97 heads is **72.5×** faster than eager.

<!-- RESULTS_TABLES -->

<!-- DECISION -->
