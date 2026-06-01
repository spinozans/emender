# E88 / GDN / M²RNN-CMA — Held-out Pile BPB inside the LIVE TRAINING HARNESS

**Task:** `e88-heldout-live` · **Run date:** 2026-05-31 → 2026-06-01 · **GPU 0 only**

Measure held-out Pile bits-per-byte for the three v0.3 architectures using the
**real training forward** (not the in-tree `ndm` scaffold stubs, which produce
worse-than-random ~17.6 nats because they are structurally incomplete). Every
number below comes from a strict checkpoint load through the live elman harness,
gated on a block-loss sanity check **before** any BPB is trusted.

---

## Results (canonical held-out slice, ctx 2048, stride 1024)

| Model | Step | Params | block-loss (nats) | Sanity | mean nats/token | Tokens scored | **Held-out BPB** |
|-------|------|--------|-------------------|--------|-----------------|---------------|------------------|
| **E88** (mamba-decay hybrid) | 1,542,000 | 1.273 B | 1.8092 | **PASS** | 2.5598 | 2,616,009 | **0.9661** |
| **GDN** (gated-delta, FLA) | 2,031,000 | 1.352 B | 1.6022 | **PASS** | 2.5597 | 2,616,009 | **0.9661** |
| **M²RNN-CMA** | 1,491,000 | 1.307 B | 1.7106 | **PASS** | 2.5470 | 2,616,009 | **0.9613** |

- **BPB definition** (no 3.92 constant): `BPB = total_NLL_nats / (total_UTF8_bytes · ln2)`
  with `total_UTF8_bytes = 9,999,511`. Verified: `6,696,445.36 / (9,999,511 · ln2) = 0.96614` for E88.
- **Sanity gate:** a correct forward yields ~2.6 nats/token (sub-1-bpb). All three
  land at 2.55–2.56 nats/token with a first-block loss of 1.6–1.8 nats — **PASS**.
  None approached worse-than-random (~10.8 nats) or absurd (~17.6 nats); the
  ~17.6-nat stub failure is **not** reproduced here.
- Held-out byte denominator: 9,999,511 bytes / 2,616,010 tokens = **3.822 B/token**
  (the run's own `p50k_base` tokenizer; the byte slice is identical across models).

### Sanity gate: PASS / FAIL per model
| Model | block-loss nats | gate [1.5, 4.0] | nats/token | verdict |
|-------|-----------------|-----------------|-----------|---------|
| E88   | 1.8092 | in range | 2.5598 | **PASS** |
| GDN   | 1.6022 | in range | 2.5597 | **PASS** |
| M²RNN-CMA | 1.7106 | in range | 2.5470 | **PASS** |

---

## E88 held-out vs train-loss 0.974

| Quantity | Value |
|----------|-------|
| E88 held-out BPB (this measurement) | **0.9661** |
| E88 paper train-loss BPB | **0.9738** (0.973765 → rounds to 0.974) |
| **Delta (held-out − train-loss)** | **−0.0076 BPB** |

The held-out BPB is **0.008 BPB *below*** the paper's headline train-loss 0.974.

**Why the two are not a like-for-like delta (honest caveat).** They use different
normalizations, and the −0.0076 net is the sum of two opposing effects:

| | per-token loss | B/token denominator | BPB |
|---|---|---|---|
| Paper train-loss (trailing-100k) | 2.644925 nats | fixed 3.918625 | 0.9738 |
| This held-out (sliding-window) | 2.5598 nats | real 3.822 | 0.9661 |

- **Protocol lowers per-token loss:** the held-out sliding-window forward gives
  every scored token up to 2047 tokens of left context, so per-token NLL drops to
  2.5598 nats (vs the 2.6449-nat trailing *training* loss accounting). Holding the
  fixed 3.918625 denominator, that protocol alone gives **0.9424 BPB**.
- **Real byte denominator raises BPB:** this slice tokenizes at 3.822 B/token under
  `p50k_base`, *below* the paper's fixed 3.918625, which pushes BPB back up. Holding
  the train-loss per-token loss but using 3.822 gives **0.9983 BPB**.
- Net of the two: 0.9661, i.e. −0.0076 vs the published 0.974.

**Bottom line:** E88's held-out generalization is healthy — the true-byte held-out
BPB (0.9661) is essentially equal to or slightly better than the train-loss
headline, with no held-out blow-up.

---

## Held-out panel ordering vs train-loss ordering

- **Train-loss (paper):** E88 0.974 < GDN 0.977 < M²RNN-CMA 0.980.
- **Held-out (this measurement):** M²RNN-CMA 0.9613 < E88 0.9661 ≈ GDN 0.9661.

On the true-byte held-out slice all three sit in the **same ~0.005-BPB band**; E88
and GDN are within 0.00002 BPB of each other (a tie at 0.966), and M²RNN-CMA is
marginally lowest. The train-loss ordering does **not** strictly survive to
held-out — the three are statistically indistinguishable at this slice size.
This is consistent with the within-band caveats already recorded in
`paper/review/V3_NUMBERS.md` and `paper/review/BPB_FULL_TABLE.md`.

---

## Harness / config path used (the real forward, not the stubs)

- **Training repo (forward source):** `/home/erikg/elman` — model construction via
  `elman.models.LadderLM` (E88, fla-gdn) and `elman.models.m2rnn_baseline.M2RNNLM`
  (m2rnn), built **exactly** as `/home/erikg/elman/train.py` does, with each run's
  own `args.json` flags (`use_triton`, `gate_activation`, `n_heads`, `expansion`,
  auto `r_h_mode`, XMA Triton backend for M²RNN).
- **Measurement script:** `scripts/measure_pile_bpb_elman.py`
  (authored in worktree `/home/erikg/ndm/.wg-worktrees/agent-740`).
- **The critical fix vs the in-tree `ndm` stubs:** these runs use the
  **schedule-free** optimizer, which saves the eval-extrapolated *x-mode* weights.
  Those are catastrophic at inference (~17.6 nats). The usable *y-mode* (training)
  weights are recovered by loading `optimizer_state_dict` and calling
  `optimizer.train()` (mirrors `generate.load_model`). The script loads
  `model_state_dict` **strict** (0 missing / 0 unexpected) and then applies this
  y-mode swap. This is why the forward here is sane and the in-tree `ndm` package
  is not.
- **dtype:** bf16 (training dtype). **GPU:** hard-pinned `CUDA_VISIBLE_DEVICES=0`
  before importing torch; training racers on GPUs 1–7 were never touched.

### Checkpoints scored (provenance — copied from the live convergence runs)

| Model | Scored copy | Live training source |
|-------|-------------|----------------------|
| E88 | `/tmp/e88gdn_bpb/ckpts/e88/checkpoint_step_1542000_loss_2.5970.pt` | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/` |
| GDN | `/tmp/e88gdn_bpb/ckpts/gdn/checkpoint_step_2031000_loss_2.7303.pt` | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/` |
| M²RNN-CMA | `/tmp/e88gdn_bpb/ckpts/m2rnn/checkpoint_step_1491000_loss_2.7347.pt` | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/` |

The scored steps (E88 1,542,000 / GDN 2,031,000 / M²RNN 1,491,000) are the latest
available checkpoints from those live runs — close to but not identical to the
nominal task references (E88 1,524,000 / GDN 1,998,000 / M²RNN-CMA 1,467,000). They
are the same architectures/configs; the small step difference does not affect the
sub-1-BPB verdict. The v0.3 weights also exist on HF (`poietic-pbc/*@v0.3`) if a
re-run on the exact nominal steps is later required.

### Canonical slice (verified byte-identical to the rest of the panel)

- source `/mnt/nvme2n1/erikg/pile.txt`, byte_offset 1,000,000,001,956, byte_length 9,999,511
- sha256 `3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a` — **verified**
  on the cached copy `/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt`
  (the script asserts the sha before scoring; UTF-8 round-trip also asserted).
- Same bytes the rest of the held-out panel used (`paper/review/heldout_slice.json`).

---

## Raw result JSON

`/tmp/e88gdn_bpb/{e88,fla-gdn,m2rnn}.json` (full per-model output, including
`nll_nats_sum`, `bytes_per_token`, `ppl_token`, wall-clock seconds, batch size).
Preserved in this branch under `paper/review/heldout_harness_json/` and the
measurement script under `scripts/measure_pile_bpb_elman.py`.

```
E88        : bpb=0.96614  nats/tok=2.5598  tokens=2616009  block=1.8092 (PASS)  2977.4s
GDN        : bpb=0.96612  nats/tok=2.5597  tokens=2616009  block=1.6022 (PASS)   454.3s
M2RNN-CMA  : bpb=0.96132  nats/tok=2.5470  tokens=2616009  block=1.7106 (PASS)   486.0s
```

---

## Status: NOT a blocker — all three measured and sane

The forward was made to run correctly inside the live training harness. All three
models pass the sanity gate and yield real sub-1-BPB held-out numbers. No values
were fabricated. `paper/main.typ` was **not** modified.
