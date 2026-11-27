# SAL-1: the exact count skeleton did not beat LOCAL at a 4:1 archive ratio

**The primary comparison failed its registered criterion.** On fresh
document-disjoint Pile text, the order-0 count skeleton — the exact suffix law,
provably equal to depth-scored attention over all 8,388,608 archive rows —
needed **0.00426 more bits per original byte** than the same host without any
archive. The interval includes zero; both seeds are slightly negative. At a
4:1 archive-to-training ratio, a co-trained byte host absorbs everything the
count law can offer. Per the registered decision rule, the next lever is
**ratio, not machinery**.

## Measurement

Fresh TRAIN-suffix roles after every prior cohort: **BUILD 2,048 documents**
(8,388,608 archive rows — a complete 2²³ dyadic tree), **FIT 512 documents**
(2,097,152 shifted bytes, each seen exactly once per fit in 2,048 complete
updates), **PROBE 128 documents locked**, opened once after all fits were
frozen and independently witnessed. Paired seeds 5751/5752 with identical
initializations, schedules and target order. Host: 633,291-parameter causal
byte transformer (width 128, four blocks). Ideal bits/original byte over all
524,416 probe bytes, first byte supplied at 8 bits.

| Co-trained model | Bits/original byte ↓ | Seed 5751 | Seed 5752 | Fit seconds |
|---|---:|---:|---:|---:|
| LOCAL (no archive) | **3.16190** | 3.15171 | 3.17210 | 27–30 |
| SKELETON (exact counts in the softmax) | 3.16616 | 3.15334 | 3.17899 | 29–31 |

| Contrast (positive saving is better) | Gain | Paired document 95% interval | Seeds | Positive |
|---|---:|---:|---|---|
| SKELETON vs LOCAL (primary) | −0.00426 | [−0.01633, +0.00412] | −0.00163, −0.00689 | no |

Both models beat plain zstd-3 (3.41971 bits/byte) by ~0.26 — they are real
language models. But zstd-3 **with the raw 8.4 MB BUILD archive as a shared
dictionary reached 3.06593**, beating both neural arms by ~0.10: at this scale
the archive's value is literal copy material, which a generic codec extracts
and a count law feeding a small host does not.

The skeleton's own costs validated the theory exactly: static counts meant
**zero per-update index cost** (12 ms CPU binding per 1,024-query batch,
index build 20.7 s once), equal-dose fits within ~10% of LOCAL's wall time,
and CPU/GPU parity to ~1e-6.

## The ORDER2 deferral (amendment A1)

The learned residual (exact matched core + order-2 moment nodes with the
value tilt) passed every oracle and device parity, but its **current-weight
bank rebuild cost 23.7 s and 24.2 GiB per update** at full geometry — ~13.4
hours per fit at the registered dose. This is the §7 wall of
[XSA_ATTENTION_LAW.md](../XSA_ATTENTION_LAW.md), now measured. ORDER2 needs a
separately registered successor: Triton-fused moment kernels from the §8
adjoints, or a two-timescale bank with a measured staleness bias. Its
question — does the learned residual beat the skeleton — remains open.

## Campaign accounting

Seven staged no-wait leases, every one released with verified identity;
**600 upper card-seconds of the 15,000 cap**. The ledger retains three
failed stages with full notes (a lease-script ordering bug, a CUDA device
placement bug caught by the parity gate, a stale-progress observer race, a
reservation-seal binding too strict for a resumed campaign, and a terminal
metadata crash after all scoring had sealed). No stage was silently retried;
each continuation ran under a new stage name with its own admission. The
probe was opened and scored exactly once; this report re-derives everything
from sealed per-document losses and receipts.

## Reading and next step

The X43 consultation won at an archive-to-training ratio near 10⁵:1. SAL-1
tested 4:1 and found nothing for the count law to add. The skeleton arm
scales for free — its costs are one-time and static — so the registered next
experiment is a **ratio sweep**: BUILD at 32–128 MB (16:1 to 64:1), SKELETON
vs LOCAL, with the archive's literal-dictionary value (zstd) as the honest
codec baseline that any neural reader must beat.

[Protocol](PROTOCOL.md) · [Amendment A1](AMENDMENT.md) · [Law](../XSA_ATTENTION_LAW.md) · [Compact result and hashes](receipts/result.json)
