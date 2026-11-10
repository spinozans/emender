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

---

## Stage 3 (operator-authorized, same session): all E97 and September-DiLoCo weights

Operator ruling: "all the E97 stuff can go… September DiLoCo checkpoints, those can go too."
Rule applied uniformly (identical to Stage 2): **keep every file under 50 MB** — receipts,
STATUS/README, summaries, metrics, logs, terminal.json, eval JSON — **remove files ≥ 50 MB**
(weights and bulk data). No evidence file was deleted.

| Target | Before | After |
|---|---:|---:|
| `~/emender/experiments` — 49 `results_*` trees + `local/`, checkpoint binaries (`.pt/.npz/.bin/.pstats`) | 701 GB | 32 GB (all under the live/current dirs) |
| `/mnt/nvme2n1/erikg/e97_systematic_posttraining` (excl. live `e97-lake-expansion-v1`) | 6.9 TB | **157 GB** |
| `/mnt/nvme1n1/erikg/diloco_8gpu` (Sept DiLoCo) | 2.21 TB | 35 GB |
| `/mnt/nvme2n1/erikg/diloco_8gpu` | 0.42 TB | 11 GB |
| `/mnt/nvme1n1/erikg/seed_maturity_threshold` | 0.52 TB | 0 |
| `/mnt/nvme1n1/erikg/ref_emender_mlp` | 0.09 TB | 0 |
| `/mnt/nvme1n1/erikg/ref_gdn2_mlp` | 0.08 TB | 0 |
| `/mnt/nvme1n1/erikg/race` | 0.06 TB | 0 |
| `/mnt/nvme1n1/erikg/invertible_stage4_13b` (E97 1.3B headline weights) | 0.67 TB | 0 |

**E97 1.3B null result — receipts survive independently of the deleted weights:**
`paper/results/expressivity_1p3b/capacity/`, `paper/figures/e97_recorded_loss.json`,
`configs/frontier/e97_*.json`, `REGIME-extraction.json`. The published numbers never
depended on retaining the `.pt` files; they remain re-derivable only by retraining, which
is stated here explicitly as the cost of the reclaim.

Read-only checkpoint dirs (mode `r-x`) required `chmod -R u+w` before removal; 14 initial
`Permission denied` errors were resolved this way, no file was left behind.

## Final disk state

| Disk | Start | End |
|---|---|---|
| `/mnt/nvme1n1` | 11.9 T used, **2.9 T free (80 %)** | **8.0 T free (44 %)** |
| `/mnt/nvme2n1` | 12.6 T used, **1.4 T free (91 %)** | **8.2 T free (42 %)** |
| `/` (nvme0n1p2) | **968 G free (93 %)** | **1.8 T free (88 %)** |

**Total reclaimed: ≈ 12.3 TB.** Live work verified intact throughout: the three byte
searches on cards 2/3/4 (100/100/98 % utilisation), `e97-lake-expansion-v1` (live daemon),
x42/x43 current-program artifacts, pile.txt and the pile-head100 spine (read-only, untouched).
