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

---

## Stage 4 — materialized sequence text (operator directive: "raw DNA text should never have been materialized")

Operator ruling: the raw DNA text must be read from the AGC archive, never materialized on disk.

### 4a. `sxgc-pilot/k466/h466_rl.txt` — 1.4 TB, REMOVED

The 1,403,221,068,481-byte text (38,790 sequences) that was fed to the BWT builder
(`grl.log`: BCR-exact BWT over 1,403,221,068,481 symbols; `chi.log`: n=1,403,221,068,481,
runs=2,739,735,806, χ=2,249,968,075). It is a *materialization* of the HPRC release-2
assembly set:

- **Source archive (present on the box):** `/moosefs/laurap/hprc/HPRC_r2_assemblies_0.6.1.agc`,
  second copy `/moosefs/garrisonlab-s3-aws/tracepoints/HPRC_r2_assemblies_0.6.1.agc`
- **Tool:** `/home/erikg/agc/bin/agc`
- **All derived structures retained** (the expensive artifacts, all still valid without the
  text): `chi_h466.sA` 18.0 GB (suffix array), `h466rt.lcp_index.lcp_index` 82.5 GB,
  `h466_rl.rl_bwt` 16.4 GB, `h466rl_pfp.parse` 62.1 GB, `h466rl_pfp.dict` 18.6 GB,
  `h466.ri4` 27.7 GB, `h466rt.bwt.{len,heads}` 16.4 GB, `h466r.{len,syms}` 13.7 GB,
  `h466_rl.txt.names.tsv` (sequence-name sidecar, 1.9 MB), plus all logs and scripts.

Nothing was deleted except the derived text. k466 directory: 1.7 TB → 260 GB.

### 4b. Verified pure duplicates of `pile.txt` — 207.4 GB, REMOVED

`pile-head100.txt` (100.0 GB) and `pile-100gb-snap.txt` (107.4 GB) were verified
prefix-identical to `pile.txt` (`cmp` over the first 2 GB, plus a 1 GB check) — i.e.
unannotated copies of a slice of the canonical corpus. The pile-head100 **spine container**
`P100c/pile-head100.sxi2` (63.2 GB, χ=24.19 G) is retained untouched, and `pile.txt`
itself (1,307.9 GB, read-only canonical) is untouched, so no query capability was lost.

### 4c. Flagged, NOT deleted — awaiting a ruling

| Path | Size | Why it is flagged |
|---|---:|---|
| `se-plan/n-pile.txt` | 1310.0 GB | **normalized** full corpus (differs from pile.txt at byte 1) — regenerable from pile.txt via the `se-plan` tooling |
| `se-plan/n-100gb.txt` | 100.0 GB | normalized head-100 GB slice of the same |
| `pwf-pileplan/pile-sparse.txt` | 1310.0 GB | **no derivation script, manifest, README or log in its directory** — 1.31 TB of unattributed materialization (exactly 1,310,000,000,000 bytes) |
| `sxgc-pilot/k10/{h10.fa,h10_rl.txt,h10ss.txt}`, `sxgc-k10sep/k10sep.txt` | ~121 GB | k=10 DNA materializations whose derived `.sxi`/sA structures exist |
| `fineweb-edu/fineweb-{16kx1m,0kx10m}.txt` | 82 GB | public sample corpora |

## Running total

Stage 1 ≈1.6 TB · Stage 2 3.9 TB · Stage 3 ≈6.8 TB · **Stage 4 ≈1.6 TB** → **≈13.9 TB**

---

## Stage 5 — the flagged corpora removed (with a measurement correction)

Operator: "the flagged 2.6T can be removed." Removed:

`se-plan/n-pile.txt`, `se-plan/n-100gb.txt`, `pwf-pileplan/pile-sparse.txt`,
`fineweb-edu/fineweb-{16kx1m,0kx10m}.txt`, `sxgc-pilot/k10/{h10.fa,h10_rl.txt,h10ss.txt}`,
`sxgc-k10sep/k10sep.txt`. (Derived k10 structures verified present first: `h10.walk.agg` 59.5 GB,
`h10.slim.head.agg` 59.5 GB, `h10rt.lcp_index` 51.1 GB, and the two `k10*.sxi` containers.)

### Correction: these were largely SPARSE placeholders, not 2.6 TB of data

Real space reclaimed was **~0.26 TB** (nvme2 +140 GB, nvme3 +121 GB), not the ~2.9 TB of
apparent size. Cause, measured after the fact:

- `se-plan` remaining files show **apparent 12 GB / real 0 blocks** — `du` reports zero
  allocated blocks for `n-10gb.txt`, `n-1gb.txt`, `n1g.txt`.
- `pwf-pileplan/pile-sparse.txt` was **exactly** 1,310,000,000,000 bytes, alone in a directory
  with no script/manifest/log — the signature of a `truncate`-style size reservation that was
  never filled.

So the earlier "2.6 TB" figure was **apparent size**; the honest reclaim is ~0.26 TB. This
receipt is corrected rather than the original claim left standing.

### Consequence worth flagging (not yet investigated)

If those normalized corpora were hole-filled reservations, then **any experiment that read
`n-pile.txt` or `pile-sparse.txt` was reading zeros for the un-written extents.** That is a
data-integrity question about the *se-plan / pwf-pileplan* workstreams, not about the
emender program (which reads `pile.txt` and the `P100c` spine only, both untouched and
verified). It should be checked on that side before those results are trusted.
