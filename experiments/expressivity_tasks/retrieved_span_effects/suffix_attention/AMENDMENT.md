# SAL-1 amendment A1: split the ORDER2 question out of the first probe

Stage `device_check3` (charge 154 upper card-seconds, rc=0, witnessed) measured
the arms at the full 8,388,608-row geometry on the leased card:

| Arm | Warm complete update | Peak allocated |
|---|---:|---:|
| LOCAL | 12 ms | 351 MiB |
| SKELETON | 13 ms | 354 MiB |
| ORDER2 | **23.7 s** | **24.2 GiB** |

CPU/GPU parity passed for all arms first (losses agree to ~1e-6). The ORDER2
cost is the current-weight bank rebuild: roughly forty memory-bound passes over
2.1 GiB tensors for encode, statistics, gather and their adjoints. At the
registered 2,048 updates that is ~13.4 hours per fit — beyond any honest
single-card campaign. This is the §7 wall of
[XSA_ATTENTION_LAW.md](../XSA_ATTENTION_LAW.md), measured rather than assumed.

## What changes

* **Stage 2 runs LOCAL and SKELETON only** (seeds 5751/5752, the full
  registered 2,048-update equal dose, all remaining panel discipline
  unchanged). The **primary comparison — does the exact count skeleton beat
  LOCAL at a 4:1 archive ratio — is answerable in minutes of card time.**
* **ORDER2 is deferred**, not cancelled. Its successor requires a separate
  registration: either Triton-fused moment kernels built from the §8 adjoints
  (one pass for leaf statistics, one for the adjoint), or a two-timescale bank
  variant with a measured staleness bias. The 23.7 s/update and 24.2 GiB
  figures are the engineering baseline any successor must beat.
* The freeze witness and scorer accept the fit set named in the stage's
  admission; every other gate, seal and the single PROBE opening are unchanged.
* Campaign ledger: 159 upper card-seconds spent to date (154 for stage 1 plus
  the two documented failed handshakes). Stage 2 is admitted under its own cap
  within the 15,000-card-second campaign ceiling.

## What does not change

Fresh document-disjoint roles, the locked PROBE, one scoring pass from frozen
weights, paired starts, the bootstrap and seed criteria, zstd baselines, and
the no-retry rule. No arm's law changed; the ORDER2 implementation remains
qualified and its oracles stay in the suite.
