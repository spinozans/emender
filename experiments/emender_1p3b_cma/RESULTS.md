# emender-1p3b-cma — MEASURED RESULTS

The typed-gdn2 **Emender** (gdn2_recall sea + e97_delta nonlinear split-edit
emendment heads) run through the **STANDARD** `scripts/cmaes_search_v2.py` driver
at 1.3B — the SAME driver and the SAME symmetric budget that GDN-2 (`fla-gdn`)
and `m2rnn` were run through. This replaces the prior `emender_real_cap` search,
which was a bespoke 2-D mixture-only proxy (41M params, n_heads crippled to 32
while the m2rnn baseline correctly got 370).

Every number below is **measured on real GPUs** (8× idle, bf16 uniform, fused
split-edit asserted; run date 2026-06-11). No paper numbers. Data files committed:
`CONVERGENCE.json`, `search/<run>/generations.jsonl`, `search/<run>/results.json`,
`search/<run>/eval_*/` (per-candidate params/stdout/.done), `heldout/heldout_run.log`.
Reproduce with `python experiments/emender_1p3b_cma/aggregate.py`.

## Driver + budget (symmetric to gdn2 / m2rnn)

Source of the symmetric budget: `scripts/repro_cmaes_1300m/launch_gdn2.sh` and
`scripts/repro_cmaes_1300m/launch_missing_cma_20260531.sh`.

| knob | value |
|---|---|
| driver | `scripts/cmaes_search_v2.py --model emender` |
| target params | 1300M, `--param_tolerance 0.03` |
| popsize | 8 |
| train_minutes / candidate | 15 |
| chunk_size | 2048 |
| tokenizer | p50k_base (vocab 50281) |
| sigma0 | 0.8 (refinement 0.32) |
| min_generations | 8 |
| warm start | 1 central anchor (dim1280/dep22/nh256/f0.15), `--anchor_only_cmaes` |
| GPUs | 8 (one candidate / GPU) |
| precision | bf16 uniform, fused e97 split-edit (`use_triton_e97=True`) |
| fitness | average train CE over all steps (identical to the gdn2/m2rnn searches) |
| wall | 3.60 h, 64 evaluations (8 generations) |

## Search space

dim 1024-4096 (int_mult128) · **n_heads 32-2000 (int, the real
width-multiprogramming regime — NOT 32)** · depth 10-50 · **mixture_nonlin
0.0-0.5 (float, gdn2_recall ↔ e97_delta fraction)** · lr 1e-4..3e-3 (log) ·
batch_size 1-128 (int_log, memory-clamped). **n_state PINNED 32**, **expansion
PINNED 1.0** (the 32×32 tile, like m2rnn).

## Convergence curve (best fitness per generation — `generations.jsonl`)

| gen | best so far | gen best | sigma |
|----:|------------:|---------:|------:|
| 1 | 6.64037 | 6.64037 | 0.2909 |
| 2 | 6.64037 | 6.72039 | 0.2727 |
| 3 | 6.47827 | 6.47827 | 0.2524 |
| 4 | **6.15245** | 6.15245 | 0.2429 |
| 5 | 6.15245 | 6.31798 | 0.2400 |
| 6 | 6.15245 | 6.38004 | 0.2427 |
| 7 | 6.15245 | 6.37455 | 0.2198 |
| 8 | 6.15245 | 6.35993 | 0.1980 |

Converged at gen 4; ran the full min 8 generations. Gen 3's per-generation best
visited the **pure-GDN-2 null** (mixture_nonlin → 0.000, loss 6.478); gen 4's
**f=0.233** beat it at 6.152 — i.e. CMA kept a substantial nonlinear-emendment
fraction on the search fitness.

## Found best (geometry + mixture — `results.json`)

```
dim = 1792   n_heads = 378   depth = 10   n_state = 32   expansion = 1.0
mixture_nonlin = 0.2335   lr = 1.108e-3   batch_size(probed) = 2
head allocation @ nh=378 :  290 gdn2_recall  +  88 e97_delta   (23.3% nonlinear)
parameters = 1.2879B   (measured 1,287,942,344;  target 1.300B, dev -0.93%)
CMA fitness (avg train CE, p50k) = 6.15245
```

- **n_heads = 378** lands squarely in the width-multiprogramming regime
  (comparable to the m2rnn CMA-best's 370) — the standard driver is NOT crippled
  to 32 heads.

## Held-out BPB of the CMA-best (`heldout/heldout_run.log`)

Measured on a disjoint held-out slice (48 MB carved from the training pile at a
300 GB byte offset — provably disjoint from the dataloader's two batch_size=2
streams near byte 0 and ~654 GB), schedule-free averaged weights, 200 val
batches, 15-min train matched to the search budget.

```
FINAL_HELDOUT_CE  = 6.2866 nats
FINAL_HELDOUT_BPB = 2.4079    (bytes_per_token = 3.7666, measured on the held-out set)
model parameters  = 1,287,942,344  (= the found-best geometry, fused bf16, verified)
```
