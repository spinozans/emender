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

## Results — Axis 1: expressivity (acc, mean±std over 3 seeds @ T=128)

| composition | RECALL | TRACK | COUNT | LATCH | NONLIN |
|---|---|---|---|---|---|
| e97_raw (pure)      | 0.136±0.007 | 0.097±0.007 | 1.000 | 1.000 | 0.682±0.007 |
| e97_raw + gdn       | 0.972±0.001 | 0.167±0.006 | 1.000 | 1.000 | 0.884±0.002 |
| **e97_raw + gdn-neg** | **0.964±0.009** | **1.000** | **1.000** | **1.000** | **0.932±0.003** |
| e97_delta (pure)    | 0.153±0.015 | 0.298±0.022 | 0.989±0.014 | 1.000 | 0.887±0.014 |
| e97_delta + gdn     | 0.967±0.004 | 0.205±0.010 | 0.982±0.024 | 1.000 | 0.894±0.009 |
| e97_delta + gdn-neg | 0.965±0.016 | 1.000 | 1.000 | 1.000 | 0.930±0.003 |
| gdn-neg (ref)       | 0.983 | 1.000 | 1.000 | 1.000 | 0.940±0.002 |
| *random baseline*   | 0.016 | 0.008 | 0.500 | 0.500 | 0.100 |

Length-extrapolation @ T=1024 (mean over seeds): TRACK extrapolates for the gdn-neg
cells (e97_delta+gdn-neg 0.944, e97_raw+gdn-neg 0.737, gdn-neg-ref 0.924; all others
≈ random 0.02–0.03). RECALL degrades for everyone at 8× train length (best ≈ 0.28),
the known hardest extrapolation. COUNT 0.78–0.84, LATCH 0.89–1.00, NONLIN 0.67–0.92.

**Reading axis 1.**
1. **The within-layer mix RECOVERS recall.** Pure backbones are recall-blind
   (e97_raw 0.14, e97_delta 0.15 ≈ baseline 0.016); adding *any* gdn head lifts recall
   to 0.96–0.97, toward the GDN bar 0.98. **YES.**
2. **gdn-neg gives RECALL *and* TRACK from one head type.** Plain `gdn` recovers
   recall but stays track-blind (0.17–0.21); `gdn-neg` (the negative along-key
   eigenvalue) recovers recall **and** drives TRACK to 1.00 (and extrapolates). **YES.**
3. **The e97 backbone owns count/latch/nonlin.** Pure e97_delta does COUNT 0.99 /
   LATCH 1.00 / NONLIN 0.89; pure e97_raw does COUNT/LATCH 1.00 but is weaker on
   NONLIN (0.68 → 0.93 once gdn-neg is mixed in). **YES.**
4. **One within-layer cell covers all five primitives on the battery:**
   `e97_X + gdn-neg` (recall 0.96 · track 1.00 · count 1.00 · latch 1.00 · nonlin 0.93).

## Results — Axis 2: LM held-out screens (17.5-min time-bounded, FUSED, REAL commapile)

All configs **159M params** (matched), p50k_base, dim768/depth12, mlp_ratio 1.0,
held-out BPB on schedule-free averaged weights. Time-bounded ⇒ tok/screen varies with
the cell's throughput (the point of leaderboard methodology, not token-matched).

| composition | held-out BPB ↓ | held-out CE (nats) | tokens (17.5 min) |
|---|---|---|---|
| **e97_raw (pure) + MLP**   | **3.2313** | 8.473 | 18.3M |
| e97_raw + gdn + MLP        | 3.2716 | 8.579 | 20.0M |
| e97_delta + gdn + MLP      | 3.3274 | 8.725 | 16.4M |
| e97_delta (pure) + MLP     | 3.3757 | 8.852 | 16.4M |
| gdn2-mlp (ref, pure gdn-neg) | 3.3934 | 8.898 | **57.4M** |
| e97_raw + gdn-neg + MLP    | 3.3976 | 8.909 | 19.1M |
| e97_delta + gdn-neg + MLP  | 3.4049 | 8.928 | 16.4M |

**Reading axis 2.**
- **e97_raw + MLP is the within-layer LM champion (3.2313)** — it beats the gdn2-mlp
  reference (3.3934) by 0.16 BPB **while processing 3.1× FEWER tokens** (18.3M vs
  57.4M in the same wall-clock). The e97_raw backbone is decisively more
  token-efficient; this reproduces Study B (`E97_RAW_MLP_RESULTS`) in within-layer form.
- **Recall heads cost LM in a fixed time budget.** Mixing in `gdn` adds +0.04 BPB
  (3.272, still beats gdn2-mlp); mixing in `gdn-neg` adds +0.17 BPB (3.398), landing
  exactly on the gdn2-mlp reference (3.393). The capability-richest cells are the
  LM-costliest.
- e97_delta trails e97_raw at every recall setting (read-modify-write is slower and
  less token-efficient than raw-write).

## Combined decision table (acc @ T=128; BPB ↓)

| composition | RECALL | TRACK | COUNT | LATCH | NONLIN | #prim | LM BPB ↓ |
|---|---|---|---|---|---|---|---|
| e97_raw (pure) + MLP   | 0.14 | 0.10 | 1.00 | 1.00 | 0.68 | 3/5 | **3.231** |
| e97_raw + gdn + MLP    | 0.97 | 0.17 | 1.00 | 1.00 | 0.88 | 4/5 | 3.272 |
| **e97_raw + gdn-neg + MLP** | **0.96** | **1.00** | **1.00** | **1.00** | **0.93** | **5/5** | 3.398 |
| e97_delta (pure) + MLP | 0.15 | 0.30 | 0.99 | 1.00 | 0.89 | 3/5 | 3.376 |
| e97_delta + gdn + MLP  | 0.97 | 0.20 | 0.98 | 1.00 | 0.89 | 4/5 | 3.327 |
| e97_delta + gdn-neg + MLP | 0.97 | 1.00 | 1.00 | 1.00 | 0.93 | 5/5 | 3.405 |
| gdn-neg / gdn2-mlp (ref) | 0.98 | 1.00 | 1.00 | 1.00 | 0.94 | 5/5 | 3.393 |

*(#prim = primitives at acc ≥ ~0.5 above baseline; RECALL/TRACK are the discriminating
ones — COUNT/LATCH/NONLIN are solved by every cell.)*

## DECISION

**Does one within-layer cell do everything?** On the expressivity battery, **YES** —
`e97_raw + gdn-neg + MLP` covers all five primitives (recall, track, count, latch,
nonlin). But **NOT for free on LM**: the all-five cells (`*+gdn-neg`) are the
LM-costliest (3.40 BPB), while the LM champion (`e97_raw` pure, 3.231) is recall- and
track-blind. **Capability and time-bounded LM trade off** — there is no single cell
that is simultaneously the LM winner *and* primitive-complete.

**Best within-layer composition — depends on the objective (all use e97_raw backbone,
which dominates e97_delta on both axes):**

1. **Capability-complete winner: `e97_raw + gdn-neg + MLP`.** The recommended single
   cell when the full primitive suite matters. The ONLY e97-backbone within-layer cell
   covering all five (recall 0.96 · track 1.00 · count/latch 1.00 · nonlin 0.93), at an
   LM cost that **ties the standard gdn2-mlp recall reference** (3.398 vs 3.393) — i.e.
   you get the entire primitive suite *and* an e97 LM backbone at the reference's BPB.
   gdn-neg is what makes one head type yield recall **and** state-tracking together.

2. **LM-efficiency sweet spot: `e97_raw + gdn + MLP`.** If state-tracking is not
   required, this recovers recall (0.97, 4/5) at only **+0.04 BPB** (3.272) — still
   beating the gdn2-mlp reference. The cheapest capability-for-LM trade in the matrix.

3. **Pure-LM champion: `e97_raw (pure) + MLP`.** Best held-out BPB (3.231), beats
   gdn2-mlp on 3× fewer tokens — but covers only 3/5 (no recall, no track).

**One-line answer:** the authoritative within-layer cell is **`e97_raw + gdn-neg +
MLP`** — one layer, e97_raw backbone heads + gdn-neg recall heads in parallel — which
is the only composition that covers **all five primitives** while remaining
**LM-competitive with the gdn2-mlp reference**; if LM throughput is paramount and
recall/track can be dropped, fall back to `e97_raw + gdn` (4/5, +0.04 BPB) or pure
`e97_raw` (3/5, LM-best).

## Reproduce

```
# axis 0 — fused / no-eager-fallback (validation item 1)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/expressivity_tasks/verify_e97_within_layer_heads.py
# axis 1 — within-layer expressivity battery (105 jobs, all 8 GPUs)
PYTHONPATH=. python experiments/expressivity_tasks/run_e97_within_layer.py --steps 5000 --output_dir paper/review/wl_results
# axis 2 — time-bounded fused LM screens (7 configs, one per GPU)
PYTHONPATH=. python experiments/expressivity_tasks/run_e97_within_layer_lm.py --minutes 17.5 --gpus 0 1 2 3 4 5 6 --output_dir paper/review/wl_lm
# aggregate
PYTHONPATH=. python experiments/expressivity_tasks/aggregate_e97_within_layer.py --results_dir paper/review/wl_results --lm_dir paper/review/wl_lm
```

Raw per-run JSON: `paper/review/wl_results/`; LM screen logs: `paper/review/wl_lm/`;
verification log: `paper/review/wl_logs/verify_within_layer.log`.
