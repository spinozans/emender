# SAL-2 protocol: the 512:1 ratio point

Registered before any natural membership access. SAL-2 extends the SAL-1
machinery unchanged (model `model.py`, driver `run.py`, gate `gate.py`,
extraction `data.py`); every registered difference is listed here. The
standing mandate (autonomous operation, ~one day or until intervention) and
`AMENDMENT.md` (A1: ORDER2 deferred) carry over.

## Question

SAL-1 measured the count skeleton at a 4:1 archive-to-training ratio and
found nothing (`SKELETON − LOCAL = −0.00426` bits/original byte, CI
[−0.01633, +0.00412], both seeds negative). X43 measured suffix-count
consultation over a 100 GB index at ~10^5:1 and won by ~0.45 bits/byte.
The autopsy of SAL-1's frozen weights showed the mechanism was not
suppressed — the learned corpus prior *rose* to 0.52 mass and the depth
weights escalated 131× toward depth 8 — but the corpus carried only 0.8% of
attention mass because matched depth classes contain ~1–3 rows in an 8.4 MB
archive. The live lever is the archive:host knowledge ratio, not the
machinery.

SAL-2 tests the same law at one decisive intermediate point: **512:1**.

## Design point

- BUILD (archive): **262,144 fresh TRAIN documents** (>= 4,097 bytes each,
  distinct by sha256), scanned in source order from byte 0 of the indexed
  region of the Pile (`/mnt/nvme2n1/erikg/pile.txt`, bytes [0, 100 GB) —
  the 100 GB XSA index territory). No chain experiment has ever drawn a
  role from this region. One registered 4097-byte hash-window per document,
  same rule as the chain's `select()`: `start = int(sha[16:32],16) %
  ((length-1)//4096) * 4096`. 262,144 x 4,096 = **2^30 archive rows**.
- FIT: 512 fresh documents (2,097,152 training targets) — the next rows of
  the contiguous rank chain after SAL-1's spent set, which is first
  reproduced exactly from the sealed receipts.
- PROBE: 128 fresh documents (524,416 scored bytes), same role size as
  SAL-1. Ratio: archive 1,073,741,824 rows to 2,097,152 fit targets =
  **512:1**.
- Host: unchanged `SalLM` (633,291 parameters). Dose: 2,048 updates, batch
  16, warmup 32, cosine 2e-3 → 2e-4, seeds 5751/5752.
- Arms: **LOCAL** and **SKELETON** only. No bank (EndpointIndex /
  WindowArchive / cores / covers are not built; order-2 stays deferred per
  A1 with its measured 23.7 s/update and 24.2 GiB baseline).

## Static count cache

At 2^30 rows the in-process CountIndex build is ~45 minutes and dominated
by sorts. It is built **off the leased card** in parallel CPU workers
(<= 4 concurrent, each one depth), written to
`output/retrieved-span-effects/sal-2/data/counts-cache/` with per-level
receipts binding `archive_sha` to the sealed `build.json` `array_sha`, and
loaded by the stage only when the binding matches; any mismatch fails
closed. Row ids are not built (skeleton-only stages never read them).

## Campaign

- Ledger `output/retrieved-span-effects/sal-2/campaign.jsonl`; cap
  **4,000 card-seconds**; end `1795845600` (the operator's one-day mandate).
- No automatic retry. Failed stages are results: continuation only under a
  new stage name with a retained note. Stage-suffixed artifacts, stage and
  module in separate files, `leased2.sh` for this campaign's paths.
- Qualification: `receipts/sal2-qualification.json` binds the published
  source, the CPU test log, and zero natural memberships at qualification
  time; data roles seal through `rows.json`/`reservation.json`/`ready.json`
  exactly as SAL-1.

## Decision rules (registered before the probe opens)

Primary contrast: mean `SKELETON − LOCAL` bits/original byte over the
probe bytes, per seed and pooled, with the same bootstrap CI family as
SAL-1.

1. **Positive** (pooled CI excludes 0 and both seeds positive): the count
   law pays at feasible in-RAM scale. Next registrations: a downward ratio
   sweep to locate the crossover, and the XSA-service merge at the true
   100 GB index (the system baseline).
2. **Not positive**: the in-RAM route is exhausted; the crossover, if any,
   lives above 512:1 and below ~10^5:1. Next registration: the XSA
   transport at 100 GB scale; no further numpy-scale ratio points.
3. ORDER2's successor (fused moment kernels or two-timescale bank) remains
   a separately registered follow-up in both cases.

zstd-3 plain and zstd-3 with the 1 GB raw archive as a shared dictionary
are reported as codec baselines; they measure the archive's literal-copy
channel, which a depth-8 count law cannot see, and are not the acceptance
criterion.

## Engineering notes retained from SAL-1

`torch.full` without `device=` is invisible to CPU oracles; `leased.sh`
reads `$stage` before guarding; observers use stage-suffixed files and
distinct path/bool names; `str.replace` asserts its pattern; data receipts
bind internal consistency across reseals; seal stages re-derive reports
from sealed losses without a second evaluation.
