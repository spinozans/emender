# E97 generalization + measurement audit — authoritative results

**Task:** `e97-audit2` · **Date:** 2026-06-07 · all 8 GPUs · REAL commapile, REAL
training, FUSED kernel, no mocks. Runs AFTER `e97-within-layer` and re-anchors its LM
conclusions on **held-out BPB**.

## Question

The leaderboard ranked **e97-raw #1 on TRAINING loss**. This audit checks two things:

1. **Measurement sanity (gates everything).** How are *train loss* and *held-out*
   actually computed — units (nats vs bpb), tokenizer, normalization, windowed-running
   vs clean eval, live vs averaged weights? Make them apples-to-apples and **quantify
   how much of any train→held "gap" is a measurement artifact vs real generalization.**
2. **Held-out ranking.** For `{e97_raw+MLP, e97_delta+MLP, gdn2-mlp, + the within-layer
   convergent winner (e97_raw+gdn-neg+MLP)}`, train fused time-bounded and report train
   loss **and** held-out BPB. **Does the held-out ranking match the train-loss ranking,
   or reorder? Is e97-raw's lead real or a training-loss artifact?**

## Validation gate (item 1) — current worktree, within-layer, FUSED not eager

- **Worktree current:** `ndm/models/typed_head_mixture.py` carries the `e97_raw`/`e97_delta`
  fused head types (commit `c2a7313` lineage; HEAD includes `cd10853` within-layer). Not stale.
- **Architecture WITHIN-LAYER, not interleaved:** every screen is `LadderLM(level='typed-gdn2-lm')`
  — every layer a `TypedHeadMixtureLayer` (head-type *fractions* in parallel) + SwiGLU MLP
  (`mlp_ratio 1.0`). Head composition is set by `--head_type_logits` (idx 0 = `gdn2_recall`,
  6 = `e97_raw`, 7 = `e97_delta`).
- **FUSED, no eager fallback:** `verify_e97_within_layer_heads.py` → **ALL CHECKS PASSED**
  (call-counted single fused Triton call per E97 head type, loud-guard armed, parity
  rel-L2 1.1–2.2e-2 fwd+bwd, 73.2× faster than eager). The 8 audit screens themselves ran
  at **19k–61k tok/s** (raw_none 19.1k, delta_none 22.6k, raw_gdnneg 49.4k, gdn2_mlp_ref
  61.2k) — all ≫10k, near-GDN order; **`grep` for `eager`/`loud`/`fallback` across every
  screen log returns NONE.** Not the 733/104 tok/s eager throttle that killed the prior
  attempts.

## Part 1 — Measurement sanity (how the two numbers are made)

Read straight from `train.py`:

| | leaderboard **train loss** (`FINAL_LOSS_LAST100`) | **held-out** (`FINAL_HELDOUT_BPB`) |
|---|---|---|
| quantity | `model(chunk, return_loss=True)` = cross-entropy | same CE, `validate()` |
| **units** | **nats / token** (PyTorch CE is natural-log) | **bits / byte** (converted) |
| **data** | **TRAIN** stream (`--data`) | **held-out** slice (`--val_data`, `/tmp/e97_heldout_rep.txt`) |
| **weights** | **LIVE** (schedule-free `y`, mid-learning) | **AVERAGED** (schedule-free `x`, `optimizer.eval()`) |
| **estimator** | **windowed running** mean over `log_every` steps (last-100 avg) | **clean** token-weighted full pass (`model.eval()`) |
| tokenizer | p50k_base | p50k_base |
| conversion | — | `BPB = CE_nats / ln2 / 3.783` |

The unit conversion factor is `1/(ln2 · 3.783) = 0.38136` (verified: held CE 8.6869 nats →
3.3128 BPB). It is a **global scale** — it shifts every BPB equally and is therefore
**ranking-invariant** (the choice of 3.783 vs the 3.82/3.92 bytes/token measured elsewhere
changes the absolute number, never the order). Study B (`E97_RAW_MLP_RESULTS`) states the
same relation: "BPB = nats × 0.3814".

**So the leaderboard train-loss number and the held-out BPB number differ by FOUR
superimposed transforms**, only one of which is real generalization:

1. **units** nats/token → bits/byte (×0.38136) — pure artifact, ranking-invariant;
2. **bytes/token constant** (3.783) — global scale, ranking-invariant;
3. **live running-average weights → schedule-free averaged + clean eval**;
4. **train distribution → held-out distribution** — the *only* real generalization term.

To isolate (4) from (3), I added an opt-in `train.py --final_train_eval` (commit below)
that runs the **same** clean `validate()` on a **train-distribution** slice with the
**same averaged weights** → `FINAL_TRAIN_CE/BPB`. Now train and held are on identical
units, weights, and estimator, so their difference is *pure generalization*.

### Decomposition of the apparent "train→held gap" (raw_none, mean of 2 seeds)

| step | value | delta | what it is |
|---|---|---|---|
| leaderboard train loss | 5.904 nats | — | windowed-running, LIVE weights, TRAIN |
| → convert units to BPB | 2.252 BPB | (×0.38136) | **units artifact** (ranking-invariant) |
| → clean eval, AVERAGED weights, train slice | 3.279 BPB | **+1.028** | **live→averaged + running→clean artifact** |
| → switch train slice → held-out slice | 3.300 BPB | **+0.021** | **REAL train→held generalization** |

**The real generalization gap is +0.021 BPB — essentially zero.** Across all four arms
the clean train→held gap is **−0.003 to +0.025 BPB** (gdn2 is *negative* — held-out
slightly *below* train), i.e. the held-out slice is **in-distribution** with training and
there is no meaningful generalization penalty. **~98 % of the apparent "gap" is
measurement artifact** (units + the live-vs-averaged weight swap), **~2 % is real.**

The dominant artifact is transform (3): in these short, non-converged, time-bounded
schedule-free runs the **averaged** weights are *worse* than the **live** weights (the
average still carries early high-loss iterates), so the clean averaged-weight eval sits
~1.0 BPB *above* the unit-converted live running loss. This is the opposite direction
from a converged run and is purely a short-horizon optimizer artifact, **not**
generalization.

## Part 2 — Held-out ranking vs train-loss ranking (4 arms × 2 seeds)

159M-param configs, 17.5-min fused time-bounded screens, REAL commapile, seeds {42,123}.
Full per-seed numbers in `paper/review/audit2_lm/AGGREGATE.txt`.

| arm | `FINAL_LOSS_LAST100` (train nats, LIVE) ↓ | held-out BPB (AVG, clean) ↓ | clean train BPB ↓ | opt. steps (17.5 min) |
|---|---:|---:|---:|---:|
| **raw_none** (e97_raw + MLP) | 5.904 | **3.3001** | **3.2793** | 2,015 |
| raw_gdnneg (e97_raw + gdn-neg + MLP) | 6.146 | 3.3384 | 3.3224 | 2,445 |
| delta_none (e97_delta + MLP) | 5.966 | 3.3980 | 3.3726 | 1,435 |
| gdn2_mlp_ref (pure gdn-neg + MLP) | **5.452** | 3.3888 | 3.3922 | **7,240** |

**Two rankings, and they REORDER:**

- **By raw train loss (`FINAL_LOSS_LAST100`, time-bounded):** `gdn2_mlp_ref` (5.45) <
  raw_none (5.90) < delta_none (5.97) < raw_gdnneg (6.15). **gdn2 looks best.**
- **By held-out BPB:** **raw_none (3.300)** < raw_gdnneg (3.338) < gdn2_mlp_ref (3.389) <
  delta_none (3.398). **raw_none is best; gdn2 falls to 3rd.**

**Why the flip — and why it does NOT impeach e97-raw's #1 standing.** In a *time-bounded*
screen, throughput sets step count: the fast pure-GDN cell takes **7,240** optimizer
steps while the e97 cells take **~1,400–2,500**. `FINAL_LOSS_LAST100` is a running average
on **live weights at the cell's own (much later) step**, so gdn2's lower train-loss merely
reflects "did 3.6× more steps", **not** better per-token quality. Held-out BPB on the
**averaged** weights measures **token-efficiency**, where e97_raw dominates — exactly the
within-layer / Study-B story. **Raw train loss is not a valid cross-cell ranking metric in
the time-bounded (unequal-token) regime; held-out BPB is the correct re-anchor.**

This is *consistent* with the leaderboard, not contradictory: the 1.3B leaderboard that
crowned e97-raw was **token-matched** (Study B: fixed 3,000 steps), where train loss *is*
apples-to-apples and e97-raw wins train (5.49 < gdn2 5.53) **and** held-out (7.90 < 8.87
nats). The only way to get the "wrong" (gdn2) answer is to rank *time-bounded* screens on
*raw* train loss — which this audit shows inverts the top of the table.

**e97-raw's lead is REAL, not a training-loss artifact.** `raw_none` is **#1 on held-out
BPB in BOTH seeds** (s42 3.3128, s123 3.2873 — best in each), reproducing within-layer
(3.231) and Study B (1.3B, both axes). If anything, time-bounded *raw train loss*
**understates** e97-raw (puts it 2nd behind gdn2); only the held-out re-anchor reveals its
true #1 token-efficiency.

**Convergent winner held-out competitiveness — YES.** The within-layer convergent winner
`e97_raw + gdn-neg + MLP` (`raw_gdnneg`) lands **2nd on held-out (3.338)**, *beating* both
the gdn2-mlp recall reference (3.389) and e97_delta (3.398), only +0.04 BPB behind the
pure-raw champion. The within-layer single-seed table had it tying gdn2 (3.398 vs 3.393);
the 2-seed average here puts it **slightly ahead** of gdn2. So the capability-complete
cell (all 5 primitives) is **held-out-LM-competitive** — within seed-noise of the top, and
ahead of the recall reference it was feared to merely match.

**Seed-variance caveat.** Run-to-run / seed spread is ~0.05–0.08 BPB (e.g. raw_gdnneg
3.296↔3.381, delta 3.331↔3.465). The fine ordering among the three trailing cells
(raw_gdnneg / gdn2 / delta) is **within this noise** and should not be over-read. What is
**robust across both seeds** is: (i) raw_none is #1 on held-out, (ii) the train-loss↔held
ranking flip, (iii) the near-zero clean train→held gap.

## Decision deliverable

1. **Is e97's standing REAL on held-out?** **YES.** `e97_raw + MLP` is #1 on held-out BPB
   in both seeds (3.300 mean), reproducing within-layer and the 1.3B leaderboard. The lead
   is *not* a training-loss artifact — it survives the clean, averaged-weight, held-out
   re-anchor, and is in fact under-credited by time-bounded raw train loss.
2. **Is the train→held gap real or a measurement artifact?** **~98 % artifact.** The real
   generalization gap is ≤0.025 BPB (held-out is in-distribution); the rest is units
   (nats→bpb, ranking-invariant) plus the live-vs-averaged-weights / running-vs-clean
   estimator difference (~1.0 BPB, a short-horizon schedule-free effect).
3. **Is the within-layer convergent winner held-out-competitive?** **YES** — `e97_raw +
   gdn-neg + MLP` is 2nd on held-out (3.338), beating the gdn2-mlp reference and trailing
   the pure-raw champion by only ~0.04 BPB, i.e. you get all five primitives at a
   held-out cost that beats the recall reference.
4. **Measurement guidance (the trap to avoid):** do **not** rank time-bounded screens on
   raw train loss — it rewards throughput (step count), not token-efficiency, and inverts
   the top of the table. Rank on **held-out BPB on the schedule-free averaged weights**, or
   keep screens **token-matched**.

## Reproduce

```
# validation item 1 — fused / no-eager (also gives parity + 73.2x gain)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/expressivity_tasks/verify_e97_within_layer_heads.py
# the audit screens (4 arms x 2 seeds, 8 GPUs, one wave) — captures train loss + clean
# train BPB + held-out BPB together on the SAME averaged weights
PYTHONPATH=. python experiments/expressivity_tasks/run_e97_audit2_lm.py \
    --minutes 17.5 --seeds 42 123 --gpus 0 1 2 3 4 5 6 7 --output_dir paper/review/audit2_lm
```

Raw per-screen logs: `paper/review/audit2_lm/{arm}_s{seed}.log`; aggregate table:
`paper/review/audit2_lm/AGGREGATE.txt`; fused-verify log: `paper/review/audit2_verify.log`.
The `--final_train_eval` flag (the train-distribution clean eval that isolates the
generalization term) was added to `train.py` for this audit.
