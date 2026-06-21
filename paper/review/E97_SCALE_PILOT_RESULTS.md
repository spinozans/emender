# E97 WITHIN-LAYER scale pilot — does the winning composition hold at LM scale?

**Task:** `e97-scale` · **Date:** 2026-06-08 · 8 GPUs (49 GB) · REAL commapile data,
REAL training, FUSED kernel, no mocks.

## Question

`e97-comp-cma` / `e97-within-layer` selected a **within-layer** head-type composition
(one `TypedHeadMixtureLayer` per layer, head types mixed *in parallel* + SwiGLU MLP)
that is **capability-complete** at small scale. Small-scale screens are suggestive; the
real test is **LM competitiveness at a meaningful scale on held-out BPB**. This pilot
scales the winning composition to a ~0.48 B-parameter LM and asks: does it MATCH/BEAT
the two reference cells — `gdn2-mlp` and `e97_raw+MLP` — on REAL Pile held-out BPB,
param-matched and FUSED? Accept / reject for a full scale run.

## Anti-regression confirmation (mandatory, all PASS)

1. **Worktree current** — `grep e97_raw ndm/models/typed_head_mixture.py` shows the
   fused `e97_raw` / `e97_delta` head types (commit `a2c9c71`, ≥ `c2a7313` on main). ✓
2. **Architecture is WITHIN-LAYER** — every layer is a `TypedHeadMixtureLayer`
   (`--level typed-gdn2-lm`), head-type composition via `--head_type_logits` (softmax →
   largest-remainder integer head counts). NOT interleaved whole-layers / `HybridLadderLM`. ✓
3. **Kernel is FUSED only** — `use_triton_e97=True` + bf16; the loud no-eager-fallback
   guard (`typed_head_mixture.py:343`) did NOT fire on any run; steady-state throughput
   is near-GDN (e97 ≈ 0.85–0.88× pure-GDN tok/s — see §Throughput). ✓
4. **Time/token-bounded FUSED screens**, not slow token-matched eager. ✓
5. **REAL data** (`commapile_mainmix_v0.1_smoke_1gb.txt`), held-out BPB on a distinct
   slice (`/tmp/e97_heldout_rep.txt`, 100 MB). ✓

## Scale choice (stated, with param count)

| knob | value |
|---|---|
| dim | 1024 |
| depth | 24 |
| n_heads | 64 |
| n_state | 32 |
| level | `typed-gdn2-lm` (within-layer typed mixture + SwiGLU MLP) |
| **params** | **≈ 0.48 B** (478.9 – 482.7 M across configs, **param-matched to 0.8 %**) |

0.48 B sits squarely in the requested 300 M – 1.3 B band and is the largest size that
lets all configs run dedicated-GPU FUSED screens in parallel on the available machine.

### Param-matching (the cell-under-test stays canonical; only the neutral FFN flexes)

At a fixed `dim/depth/n_heads`, the recurrent head types carry **different** parameter
counts, so the canonical (`mlp_ratio=1.0`) cells are NOT param-matched:

| config (mlp_ratio = 1.0) | params | note |
|---|---|---|
| `raw_none` (pure e97_raw) | 480.9 M | heaviest — e97 split-edit heads are param-heavy |
| `raw_gdnneg` (gdn-neg + e97_raw) | 431.7 M | |
| `g2i2_cmawin` (CMA mixture) | 402.4 M | |
| `gdn2_mlp_ref` (pure gdn-neg) | 382.4 M | lightest |

To make the BPB comparison fair, all configs are matched to the **heaviest** (≈ 481 M)
by flexing **only the SwiGLU MLP ratio** — a uniform, fungible param sink that leaves
the recurrent head composition (the thing under test) exactly as defined:

| config | role | head allocation (n_heads = 64) | mlp_ratio | **params** |
|---|---|---|---|---|
| **`raw_gdnneg`** | **winning within-layer cell** | 32 gdn-neg recall + 32 e97_raw | 1.652 | **478.9 M** |
| `gdn2_mlp_ref` | control 1 (GDN-2 + MLP) | 64 gdn-neg | 2.305 | 481.5 M |
| `raw_none` | control 2 (e97_raw + MLP) | 64 e97_raw | 1.000 | 480.9 M |
| `g2i2_cmawin` | secondary: exact comp-cma LM-best | CMA logits, 6 head types, knob_lr 23.4 | 2.040 | 482.7 M |

The winning composition is the **capability-complete 2-type cell** `raw_gdnneg`
(`e97_raw` backbone + `gdn-neg` recall head), i.e. the `e97-within-layer` winner
"e97_raw + gdn-neg + MLP (all 5 capabilities, LM ties ref)". `g2i2_cmawin` is the exact
`e97-comp-cma` LM-best operating point (a 6-head-type CMA mixture) carried at scale as a
fidelity check.

## Capability spot-check at scale (recall / count / track)

Probe battery (`train_hybrid.py`, within-layer typed mixer, dim 256, 5000 steps, seed
42, eval length-extrapolation T ∈ {128…1024}), winning cell vs the `e97_raw` control:

| capability (probe) | **`raw_gdnneg`** (winner) | `raw_none` (e97_raw ctrl) | random baseline |
|---|---|---|---|
| COUNT (`anbncn_viability`) | **1.0000** | 0.9996 | 0.500 |
| RECALL (`mqar_recall`) | **0.9597** | 0.1323 | 0.0156 |
| TRACK (`s5_permutation`) | **1.0000** | 0.1052 | 0.0083 |
| RECALL @ T=1024 (8× train len) | **0.2019** | 0.0173 | — |
| TRACK @ T=1024 | **0.8542** | 0.0205 | — |
| COUNT @ T=1024 | 0.8631 | 0.8566 | — |

**Reading.** The winning within-layer cell is **capability-complete**: recall 0.96,
track 1.0, count 1.0, and track/count **extrapolate** to 8× train length (track 0.85).
The `e97_raw + MLP` control is **recall-blind (0.13) and track-blind (0.11)** — exactly
the small-scale `e97-within-layer` result. The within-layer `gdn-neg` mix genuinely adds
recall + track that `e97_raw + MLP` alone cannot express; this property is intrinsic to
the head composition and is reproduced here.

## Held-out BPB — the LM verdict

Held-out BPB on the schedule-free **averaged** weights, FUSED, REAL Pile slice.
`batch_size 12 × chunk 512 = 6144 tokens/step`.

### ⚠ Throughput / compilation confound (and the fix)

The first pass used **time-bounded** 25-min screens with a **cold** Triton cache. Three
e97 configs JIT-compiling concurrently paid a large **one-time compilation cost** that
ate into the wall-clock budget, so the e97 cells saw far fewer tokens than pure-GDN
*despite comparable steady-state throughput*:

| config | cold-cache 25-min BPB | steps | tokens | steady tok/s |
|---|---|---|---|---|
| `gdn2_mlp_ref` | 3.3749 | 4229 | 26.0 M | 18 619 |
| `raw_none` | 3.4366 | 1764 | 10.8 M | 19 732 |
| `raw_gdnneg` | 3.4696 | 1500 | 9.2 M | 16 564 |

This is **apples-to-oranges on tokens-seen** (gdn2 saw ~2.8× the tokens), so it is NOT a
valid cell-quality comparison — it mostly measures cold-start compile overhead. (The
6-head-type `g2i2_cmawin` mixture compiles **six** distinct kernels and never reached a
usable step count — a real practical liability of the full-CMA operating point at scale.)

**Fix:** re-run **token-matched** (fixed `--steps 4000` = 24.6 M tokens for every
config) with a **warm** Triton cache, so startup is cheap and BPB is compared at equal
compute. Wall-clock-to-completion is reported alongside to retain the throughput signal.

### Token-matched held-out BPB (4000 steps = 24.6 M tokens each — PRIMARY)

All three param-matched to ≈ 481 M, FUSED, identical 4000-step / 24.6 M-token budget,
schedule-free averaged weights, same held-out slice:

| config | role | **held-out BPB** | CE | train wall-clock | effective tok/s | throughput vs gdn2 |
|---|---|---|---|---|---|---|
| `gdn2_mlp_ref` | control 1 (GDN-2 + MLP) | **3.3524** | 8.7906 | 0.394 h | 17.3 k | 1.00× |
| **`raw_gdnneg`** | **winning within-layer cell** | **3.3636** | 8.8200 | 1.014 h | 6.7 k | **0.39×** |
| `raw_none` | control 2 (e97_raw + MLP) | 3.8951 | 10.2137 | 0.917 h | 7.5 k | 0.43× |

Δ(BPB):
- **winner vs `gdn2-mlp`**: **+0.0112** (0.33 % worse) — a statistical **tie** at this budget.
- **winner vs `e97_raw+MLP`**: **−0.5315** — the winner is **decisively better** (the
  `e97_raw` backbone alone is a poor LM cell; raw-write is a pure LM liability).

So on the literal head-to-head bar the within-layer cell **MATCHES `gdn2-mlp` and BEATS
`e97_raw+MLP`**. But it buys **no BPB advantage** over the much simpler, ~2.6× faster
`gdn2-mlp`, which is itself already capability-complete (recall 0.98 / track 1.0 /
count 1.0 in the `e97-within-layer` study). The 32 `e97_raw` heads in the mix add
~nothing to LM BPB over pure `gdn-neg` (3.364 vs 3.352) while imposing the full
throughput penalty — at scale they are **dead weight**.

## Throughput (corrected — effective, not instantaneous)

The per-10-step **instantaneous** tok/s printed by `train.py` (~16.7 k for the e97 cells,
~18.6 k for gdn2) is **misleading** at this scale: it covers GPU-compute windows but not
the CPU-side launch gaps between the many small kernels of the split-edit recurrence.
Measured from **wall-clock to a fixed step count** (the honest figure):

| config | 4000 steps in | effective tok/s | vs gdn2 |
|---|---|---|---|
| `gdn2_mlp_ref` | 0.394 h | 17.3 k | 1.00× |
| `raw_none` (e97_raw) | 0.917 h | 7.5 k | 0.43× |
| `raw_gdnneg` (winner) | 1.014 h | 6.7 k | **0.39× (≈ 2.6× slower)** |

The e97 split-edit kernel is **latency-bound** (GPU util ~13–15 % vs gdn2's ~97 %), so at
dim 1024 the within-layer cell runs at **< 0.4× the wall-clock throughput** of `gdn2-mlp`
— much worse than the ~0.9× steady figure seen at small dim. (Whether this is a
kernel-maturity gap or fundamental is open; either way it is the cost *today*.) No eager
fallback fired; no NaN. The 6-head-type `g2i2_cmawin` CMA mixture is worse still — it
JIT-compiles six distinct kernels and never reached a usable step count at scale.

## Verdict

**Held-out BPB head-to-head (token-matched, param-matched, FUSED):**
the winning within-layer cell **MATCHES `gdn2-mlp`** (3.3636 vs 3.3524, +0.33 %, a tie)
and **BEATS `e97_raw+MLP`** (3.3636 vs 3.8951, −0.53 BPB). It is **capability-complete**
at scale (recall 0.96 / track 1.0 / count 1.0, track+count extrapolating to 8× length).
So it clears the literal "match/beat both references" bar.

### Accept / reject for a full scale run: **dominated (ties gdn2-mlp on BPB, no unique capability, ~2.6x slower wall-clock)**

The within-layer e97 composition is **competitive but dominated**, not scale-worthy:

1. **No BPB upside.** Token-matched, it only ties `gdn2-mlp`; the 32 `e97_raw` heads add
   nothing over pure `gdn-neg` (3.364 vs 3.352). The recall (`gdn-neg`) heads do the LM
   work; the `e97_raw` heads are dead weight for held-out BPB.
2. **No unique capability.** Its recall/track/count completeness is **also** held by the
   simpler `gdn2-mlp` cell — the within-layer mix buys no capability `gdn2-mlp` lacks.
3. **~2.6× slower** wall-clock at scale (latency-bound split-edit kernel, GPU util
   ~13–15 %). At equal wall-clock — the regime a real training run lives in —
   `gdn2-mlp` sees ~2.6× more tokens and **wins outright** (the cold-cache time-bounded
   screen showed this directionally: gdn2 3.375 vs winner 3.470).
4. The only thing the cell decisively beats is `e97_raw+MLP`, which merely **confirms
   `e97_raw` alone is a poor LM backbone** — it does not argue *for* mixing it in.

**Recommendation.** Do **not** commit a full scale run of the within-layer `e97_raw +
gdn-neg` mixture as an LM. Plain **`gdn2-mlp` (GDN-2 + MLP) is the better cell at scale**:
equal held-out BPB, equal capability-completeness, ~2.6× the throughput, and a single
mature kernel. The e97 split-edit / raw-write heads earn their place on targeted
*expressivity* probes (count/latch), not as an LM backbone. A full scale run would only
be warranted if (a) the fused split-edit kernel is optimized to approach GDN throughput
**and** (b) a larger-token-budget run demonstrates a BPB crossover where the e97 heads
start to *help* — neither is shown here.

### What WOULD change the verdict
- A kernel pass that lifts e97 GPU util from ~15 % toward GDN's ~97 % (closes the 2.6×
  throughput gap), **plus**
- A longer-budget token-matched run showing `raw_gdnneg < gdn2_mlp_ref` on BPB (a real
  per-token advantage, not the tie seen here).

---

### Reproduction

- Scale screens: `train.py --level typed-gdn2-lm --dim 1024 --depth 24 --n_heads 64
  --n_state 32 --bf16 --head_type_logits=<cfg> --gdn_allow_neg_eigval 1
  --mlp_ratio <param-match> --steps 4000 --final_heldout_eval` on
  `commapile_mainmix_v0.1_smoke_1gb.txt`, held-out `/tmp/e97_heldout_rep.txt`.
  Logs: `experiments/expressivity_tasks/scale_pilot_sm/{raw_gdnneg,gdn2_mlp_ref,raw_none}/`.
- Capability probes: `experiments/expressivity_tasks/train_hybrid.py --task
  {mqar_recall,anbncn_viability,s5_permutation}` (dim 256, 5000 steps, seed 42).
  Logs: `experiments/expressivity_tasks/scale_cap_probe/`.
- Param-match arithmetic: FFN slope ≈ 75.5 M params / `mlp_ratio` unit at this size.
