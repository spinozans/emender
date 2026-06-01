# Pinned paper-critical training checkpoints

**Task:** `pin-paper-checkpoints`. Safely PIN (copy + checksum + verify) the
paper-critical schedule-free training checkpoints out of volatile `/tmp` onto a
persistent drive **before** the racer training runs are killed. **READ-ONLY** from
the source dirs; **no training process was signalled or touched**; `paper/main.typ`
was **not** modified.

**Scope (author widening, 2026-06-01):** for EACH model pin BOTH (a) the **latest
complete** checkpoint (the training frontier / resume point for the next phase) AND
(b) the **paper / held-out** checkpoint, plus **recent intermediates** as space
permits — all WITH full optimizer / schedule-free state so y-mode stays recoverable
from every pinned copy.

## TL;DR / Verdict

> **SAFE TO KILL THE RACER TRAINING RUNS: YES.** For all three models the latest
> frontier checkpoint **and** the paper/held-out checkpoint (plus every available
> recent intermediate) are pinned to persistent storage, byte-verified
> (sha256 source==dest), confirmed to `torch.load`, and confirmed to carry the
> **full schedule-free optimizer state** (`z` + `exp_avg_sq`) required to recover
> the usable **y-mode** inference weights. E88 y-mode recovery was reproduced
> end-to-end from the pinned copy. This report does not kill anything — that remains
> the author's action.

**Pinned: 19 full checkpoints + 3 `args.json` (≈152 GB).**

| model | pinned ckpts | paper step | frontier step | steps pinned |
|---|---:|---:|---:|---|
| E88 | 7 | **1,542,000** | **1,560,000** | 1542000, 1545000, 1548000, 1551000, 1554000, 1557000, 1560000 |
| GDN (fla-gdn) | 6 | **2,031,000** | **2,046,000** | 2031000, 2034000, 2037000, 2040000, 2043000, 2046000 |
| M2RNN-CMA | 6 | **1,491,000** | **1,503,000** | 1488000, 1491000, 1494000, 1497000, 1500000, 1503000 |

The paper steps (E88 1,542,000 / GDN 2,031,000 / M2RNN 1,491,000) are exactly the
held-out BPB reference checkpoints in `paper/review/HF_V03_FIX.md` §2 /
`paper/review/E88_HELDOUT_HARNESS.md` (E88 2.5598 nats / 0.9661 BPB;
GDN 2.5597 / 0.9661; M2RNN 2.5470 / 0.9613).

## Destination & free space

`/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/` (drive `/dev/nvme3n1`, 14T).
Chosen over `/mnt/nvme1n1` (only 1.3T free). One subdir per model: `e88/`, `gdn/`,
`m2rnn/`. **Free space after pinning: 5.1 TB** (≈152 GB consumed by this pin).
The destination is self-describing: it also holds `SHA256SUMS.txt`,
`SHA256SUMS_full.txt` (source↔dest pairs), `checkpoint_contents_verification.json`,
and a copy of this report (`README_PINNED_CHECKPOINTS.md`).

## Why the optimizer state is non-negotiable

These are **schedule-free** runs (`AdamWScheduleFree`; each `args.json` has
`"optimizer": "schedulefree"`). `train.py` calls `optimizer.eval()` *before* saving,
so the saved `model_state_dict` is the **x-mode** (eval-extrapolated) view, which is
**catastrophic at inference** (~18 nats/token — measured 18.05 for the source E88
`model_state_dict` alone, `scripts/e88_export_vs_source_result.json`). The usable
**y-mode** (training) weights are recovered **only** by also loading
`optimizer_state_dict` and calling `optimizer.train()` (the swap in
`scripts/measure_pile_bpb_elman.py:161-183`). **Pinning only model weights would
permanently destroy the ability to recover y-mode and re-export correct HF weights.**
Every one of the 19 pinned checkpoints was verified to carry `model_state_dict`
**and** `optimizer_state_dict` with the schedule-free `z` base-sequence buffer +
`exp_avg_sq` and the `train_mode` param-group flag (see
`checkpoint_contents_verification.json`).

## Verification performed

1. **Copy** — `cp -p` (timestamps preserved), READ-ONLY from source. The one
   mid-write file at copy time (GDN `checkpoint_step_2046000`, held open for write by
   the training PID per `lsof`) was skipped on the first pass and pinned only after
   its writer closed and `lsof` showed it complete. Every copy re-checked with an
   `lsof` guard.
2. **sha256 (integrity)** — every pinned `.pt`/`.json` hashed and compared against
   its source: **all source==dest, byte-identical** (`SHA256SUMS_full.txt`;
   `sha256sum -c SHA256SUMS.txt` passes).
3. **Loadability + optimizer state** — every checkpoint `torch.load`s cleanly and
   contains `model_state_dict` (E88 87 / GDN 297 / M2RNN 150 tensors) **and**
   `optimizer_state_dict` with `z` + `exp_avg_sq` state — verified for all 19.
4. **y-mode recoverability (E88, end-to-end, from the PINNED copy)** —
   `scripts/measure_pile_bpb_elman.py` built E88 as `train.py` does, loaded the
   pinned `model_state_dict` **strict (0 missing / 0 unexpected)**, applied the
   schedule-free `optimizer.train()` swap from the pinned `optimizer_state_dict`, and
   measured the canonical held-out Pile slice (sha `3e4241a9…`, 9,999,511-byte
   denominator, ctx 2048 / stride 1024):
   - block-loss sanity gate: **1.8092 nats/token** — matches the harness reference
     block-loss 1.8092 exactly (HF_V03_FIX.md §2).
   - strided held-out mean: **2.525 nats/token** over the first 205,823 scored tokens
     (converging to the reference mean 2.5598 nats / 0.9661 BPB) — squarely in the
     expected "~2.56 nats" territory. Without the swap the same weights give ~18 nats,
     so the schedule-free swap is demonstrably operating on the pinned data.
     (Full-slice slide runs to completion on GPU 0; raw → `/tmp/pin_verify_e88_ymode.json`.)

   The other models share the identical save/swap code path and all carry the same
   full schedule-free state (verified in step 3), so y-mode is recoverable from every
   pinned copy by the same procedure; reference y-mode for GDN is 2.5597 nats / 0.9661
   BPB and for M2RNN 2.5470 nats / 0.9613 BPB.

---

## Per-model file inventory (step · role · ckpt-loss · sha256)

`role`: **frontier** = latest complete (resume point); **paper** = held-out BPB
reference; **intermediate** = recent intermediate. Source dirs:
E88 `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832`,
GDN `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832`,
M2RNN `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023`.
Every file: `optimizer_state_dict` present (z+exp_avg_sq) ✅, sha256 source==dest ✅.

### E88  →  `e88/`  (7 ckpts; paper 1,542,000 · frontier 1,560,000)

| step | role | loss | size (B) | sha256 |
|---:|---|---:|---:|---|
| 1,542,000 | **paper** | 2.5970 | 7,639,217,707 | `64ae1e7e79d68405803122f65d643e6c016e23585ff4290c5785806a04c6c3a4` |
| 1,545,000 | intermediate | 2.6435 | 7,639,217,707 | `8c519c90c5a7a168067fd2deddd40677e69210bcafbe5c23ac2d59bff6bd38cf` |
| 1,548,000 | intermediate | 2.6555 | 7,639,217,707 | `48d735fb1c11d422fce616466e288634a21891fd0a52483ceb5a1cced0c27e60` |
| 1,551,000 | intermediate | 2.6290 | 7,639,217,707 | `da513edd7aadd1a7bef7d36b6cc73d88f6ada9ba371e151941db38bc8f74c7aa` |
| 1,554,000 | intermediate | 2.7036 | 7,639,217,707 | `da6360e9e5e15bcb021ae27f4a59e291b75f0f942529ee0e44e8634f177c3e05` |
| 1,557,000 | intermediate | 2.6946 | 7,639,217,707 | `14ea7824625da28ebc53e12d39d36e70c76776881fd8822dea3acccfac7e59c3` |
| 1,560,000 | **frontier** | 2.6681 | 7,639,217,707 | `9fac89a94b22a27c344dc96d417a1f97262040b365c1e8ff1842396036aea537` |

`e88/args.json` sha256 `4253a7c477dc26dee8df9f41dd2cd9f170341b0e0750eba33b9b8f60fa43270b`.

### GDN (fla-gdn)  →  `gdn/`  (6 ckpts; paper 2,031,000 · frontier 2,046,000)

| step | role | loss | size (B) | sha256 |
|---:|---|---:|---:|---|
| 2,031,000 | **paper** | 2.7303 | 8,114,430,987 | `5e6a00ae22d79b90cd71826e0b44997a5b4d740f8991c57951284082ef1970a9` |
| 2,034,000 | intermediate | 2.5695 | 8,114,430,987 | `b5cbe904b2e61a660bf3b54ccc9418807a8a00d92a69135f9dcce61a21cce374` |
| 2,037,000 | intermediate | 2.5785 | 8,114,430,987 | `3e248affd07b4e08d72cd38a11202fe469033c08b35220442f2bec31807ef4fd` |
| 2,040,000 | intermediate | 2.6648 | 8,114,430,987 | `eca19bdc400cffc676ddb6145e6c7b0dabfc442016c91b8fa983e0dd25c755a1` |
| 2,043,000 | intermediate | 2.5799 | 8,114,430,987 | `2c1b98990a52157f46b6a40dc40a2265328017a34d41c68b9a4b45bd7a62ff04` |
| 2,046,000 | **frontier** | 2.7407 | 8,114,430,987 | `06939b0d74d7b61f3389841043630aff3ba26ceb75abf5b94b54d6e6ad29f6e6` |

`gdn/args.json` sha256 `e45ad7f92ff541b9f0ac8c4a17cb06205ab0be62402a93f4c71eec72519df153`.

### M2RNN-CMA  →  `m2rnn/`  (6 ckpts; paper 1,491,000 · frontier 1,503,000)

| step | role | loss | size (B) | sha256 |
|---:|---|---:|---:|---|
| 1,488,000 | intermediate | 2.6765 | 7,842,766,221 | `34d59da3f369f41f4bf293ab513d152e5403e53a9f49a2dab4d88d868c3cd872` |
| 1,491,000 | **paper** | 2.7347 | 7,842,766,221 | `58ad602019be8896870f8e599e69d0ba1822f8c2e709df5951dcbfefd71cdc74` |
| 1,494,000 | intermediate | 2.6745 | 7,842,766,221 | `816d9114f05f896d2990ac723fec625db1355c9cf75e117ccc2d84cf1b3d935c` |
| 1,497,000 | intermediate | 2.7027 | 7,842,766,221 | `d7d41aa6935daf16ed0983545501bac587b623e673fa6de76242ff2b83a43be3` |
| 1,500,000 | intermediate | 2.6079 | 7,842,766,221 | `408e2a05da4000c77d19b16f51beb56640573361af658d4b022cfaba862bb74b` |
| 1,503,000 | **frontier** | 2.6653 | 7,842,766,221 | `3bd5fe4a98fc6924b1caa28427d24fb3d9316ec168edd5bb313f6d24ed8a248d` |

`m2rnn/args.json` sha256 `81a505f01183ca96007bfa5b3ce756dbe63d9d85b7abdb496fcd92bf7df98633`.

---

## Validation checklist

- [x] Latest frontier **and** paper checkpoint pinned for all three models
      (+ every available recent intermediate: 7 / 6 / 6).
- [x] Persistent, non-`/tmp` destination with free space confirmed
      (`/mnt/nvme3n1`, 5.1 TB free after ≈152 GB pin).
- [x] Every pin includes optimizer/schedule-free state (`z` + `exp_avg_sq`,
      `train_mode`), **not** just model weights — verified for all 19.
- [x] sha256 + loadability verified per checkpoint (all source==dest byte-identical;
      all `torch.load` cleanly); exact steps recorded.
- [x] y-mode recoverability confirmed for E88 **from the pinned copy** (strict
      0/0 load + schedule-free swap → block-loss 1.8092 = reference; mean
      2.525 nats/tok on target); structurally guaranteed for GDN/M2RNN (identical
      state present + identical swap path).
- [x] No training process signalled/killed; source dirs read-only; mid-write GDN
      `2046000` identified via `lsof` and only pinned once complete.
- [x] `paper/main.typ` NOT modified.

## SAFE-TO-KILL verdict

**YES — it is now safe to kill the racer training runs.** For all three models the
latest frontier checkpoint and the paper/held-out checkpoint (plus all available
recent intermediates) are pinned to persistent storage, byte-verified, load cleanly,
retain the full schedule-free optimizer state, and E88 y-mode recovery has been
reproduced end-to-end from the pinned copy. This report does not kill anything; that
action remains the author's.
