# Disk reclaim receipt — 2026-11-09/10

Operator-authorized cleanup after the disk audit (nvme1n1 80%, nvme2n1 91%, `/` 93%).
Every removal below is either verified-regenerable, a duplicate, a cache, or a
declared-busted experiment whose evidence was preserved first.

## Stage 1 — verified-regenerable / duplicate / cache

| Removed | Size | Justification | Recovery |
|---|---:|---|---|
| `comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt` | 1.0 TB | decompressed form of the adjacent `.zst`; the live training copy is on Frontier Lustre (`/lustre/orion/bif148/proj-shared/commapile/`) | `zstd -d` in 25m22s; **integrity verified before deletion**: `zstd -t` decompressed to exactly `1,000,000,725,401` bytes = manifest `actual_file_size`, XXH64 check enabled; manifest sha256 `44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9` retained |
| `comma_v0.1_training_dataset/.git/lfs` | 485 GB | LFS cache duplicate of the working tree (link count 1, not hardlinked) | `git lfs pull` from `huggingface.co/datasets/common-pile/comma_v0.1_training_dataset` |
| `~/.triton` | 155 GB | Triton JIT compile cache | regenerates on next kernel compile |

## Stage 2 — e97-rl-loop-v1 (declared total failure by the operator)

`/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1`: **4.24 TB → 54 GB**.

Removed: `bank/lanes` (2.3 TB, 8 lanes of dead experience), `bank/merges/merge-{0001..0004,0006..0009,0010..0032 minus kept}` (506 GB),
`wash-v1` (296 GB), `w3-matcheddose` (187 GB), `training` (136 GB), `bank-pilot2` (136 GB),
`bank-heldout-reads` (94 GB), `bank-singlelearner-v1` (92 GB), `w4-chillwash` (84 GB),
`bank-singlelearner-B` (73 GB), `wash-v2` (23 GB), `smoke` (23 GB).

Preserved (in place): `STATUS.md`, `README.md`, `RECEIPT_SPEC.md`, `F7-probe`,
`docnll-gpu-probe-v1`, `bank-merge-shakeout`, `episodes`, `scripts`, `receipts`, `packs`,
`cycles`, `queue`, `state`, `bank/{report,state,pool,unit-test-v2}`, and
**`bank/merges/merge-0005` + `merge-0033`** (the evaluated merge from STATUS.md and the final
merge state — 46 GB of weights kept as the null result's reproducibility artifact).

The recorded outcome is preserved verbatim in `e97-rl-loop-v1/STATUS.md`: after ~20 h of RL
(1,154 updates, 8-lane weighted soup) the anchor held to within **+3.4e-4 nats (0.013 %)**
— a null result. The bank was dead experience, not evidence.

## Result

| Disk | Before | After |
|---|---|---|
| `/mnt/nvme2n1` | 1.4 T free (91 %) | **5.0 T free (65 %)** |
| `/mnt/nvme1n1` | 3.3 T free (77 %) | **4.7 T free (67 %)** |
| `/` | 968 G free (93 %) | **1.1 T free (92 %)** |

Total reclaimed: **≈ 5.2 TB**.

## Not touched (deliberately)

- `e97-lake-expansion-v1` — live daemon (`standing_supply.py`), 1.6 G.
- The 54 E97 `checkpoints/` dirs (1.98 TB) and the September screens (~1.2 TB): these belong to
  the chat-agent post-training arcs, not the RL loop. Awaiting an operator ruling on whether
  those arcs are also retired before any deletion.
- `~/emender/experiments` old-ladder checkpoints (~700 GB of `.pt`): needs a keep-list pass
  against the PROGRAM ledger (C14 frozen checkpoints are still referenced).
- `diloco_8gpu` (2.1 TB) and `commapile.txt.zst` (545 GB): operator calls.
- All genomics / sxgc data on nvme3n1 and nvme2n1.
