# SAL-2 result: the 512:1 ratio point is null — and the mechanism is now measured

Registered in `PROTOCOL-SAL2.md` before any membership access. Frozen fits,
witnessed freeze, probe opened and scored exactly once, report re-derived
from sealed losses by `close2.py` (no second evaluation). Receipt:
`receipts/sal2-result.json`.

## Result

On 524,416 fresh probe bytes (128 document-disjoint documents, out of
archive, beyond the 100 GB index region):

| arm | bits/original byte (mean) | per seed |
|---|---|---|
| LOCAL | **3.09697** | 3.11302 / 3.08091 |
| SKELETON | **3.09935** | 3.11414 / 3.08455 |

**SKELETON − LOCAL = −0.00238** bits/original byte, bootstrap CI
[−0.00586, +0.00124], seeds [−0.00112, −0.00364], positive = **False**.

zstd-3 plain: 3.44168. zstd-3 with the 1 GB raw archive as a shared
dictionary: **3.06750** — again beating both neural arms.

## The two ratio points, side by side

| | SAL-1 (4:1) | SAL-2 (512:1) |
|---|---|---|
| archive | 8,388,608 rows (8.4 MB) | 1,073,741,824 rows (1 GiB) |
| fit / probe | 2,097,152 / 524,416 bytes | same |
| host | SalLM 633,291 params | same |
| SKELETON − LOCAL | −0.00426 [−0.01633, +0.00412] | −0.00238 [−0.00586, +0.00124] |
| zstd with raw dictionary | 3.06593 (vs LOCAL 3.16190) | 3.06750 (vs LOCAL 3.09697) |

A **128× larger archive moved the effect by less than its own CI**. The
in-RAM ratio lever is dead at these scales, and the registered decision
rule fires: no further numpy-scale ratio points; the successor is the
XSA transport at the true 100 GB index (SAL-3).

## Why: the fitted parameters complete the story

The SAL-2 skeleton fit (seed 5751) learned:

- corpus prior mass `e^{log_pi} = 0.527` — half the prior budget, again
  (SAL-1: 0.516). The optimizer did **not** gate the archive off.
- depth weights escalating monotonically, depth 8 at **125×** depth 0
  (SAL-1: 131×). The model, again, begs for deep matches.
- corpus attention share: **mean 2.24%, median 0.06%** (SAL-1: 0.77%).
- matched class sizes on real fit contexts: m₈ median **478**, m₆ median
  862, m₄ median 22,088 — the 1 GB archive's deep counts are *dense for
  typical text*.

That last row is the decisive measurement. At 512:1 the archive finally
backs its confident predictions with hundreds of examples — and the model
still refuses to use them, because **where the counts are dense, the local
host is already good** (common n-grams are exactly what a 633k-param byte
model learns from 2 MB of fresh Pile). The archive can only add value
where the host is weak — rare, long, specific contexts — and there the
1 GB backing is still single-digit (the *average* m₈ over all archive
rows is 8.2; only the Zipf head reaches hundreds). Redundant where
reliable, unreliable where needed: both ratio points measured the same
empty intersection, which is why 4:1 and 512:1 agree.

Two further structural facts, both now empirical:

1. **The softmax share is a choice, not a constraint.** `log_pi` is a free
   learned scalar; raising it reproduces any multiplicity-style prior.
   The optimizer tested that space and chose a ~1–2% share at both
   scales — the share is small because more corpus mass does not reduce
   fit loss, not because the placement forbids mass.
2. **The crossover is set by backing, not by bytes.** For the value-bearing
   rare-context regime to have m_ℓ ≳ 50–500, the archive needs ~50–500×
   more text than 1 GB — i.e. the 100 GB class. X43's 100 GB consultation
   (1.657 vs 2.109 bits/byte) sits exactly at that endpoint of this curve.

And the one channel that keeps paying at small scale is the one a depth-8
count law cannot see: zstd's dictionary gain (~0.03–0.10 bpb across both
campaigns) comes from long literal matches — Pile duplicate passages —
which is copy material, not population statistics.

## Engineering record

- Campaign: 261 card-s of the 4,000 cap; both stages closed with verified
  release, renewed heartbeat, and witnessed progress. No retries: the
  failed stages continue under new names with retained notes
  (`attempt-run-note.txt`, `attempt-run2-note.txt`).
- Stage `run` (197 card-s, rc=1): all four fits closed and the freeze was
  witnessed; the quarantined probe subprocess ran the default campaign
  paths and failed closed before touching the probe. Fix: the probe entry
  is the `PROBE_ENTRY` constant, rebound to the campaign CLI (published
  before continuation).
- Stage `run2` (64 card-s, rc=1): resumed from the witnessed freeze — no
  fits retrained — opened and scored the probe once, sealed
  `complete.json` and `document-losses.npz`, and died on a hardcoded
  terminal ack name (`watch-ack-run` instead of `watch-ack-run2`). The
  report is re-derived from sealed losses only (`close2.py`).
- Known recorded defect: `complete.json`'s `archive.documents` field read
  the un-rebound constant (2048); the true geometry — 262,144 documents,
  2^30 rows — is in `rows.json`/`build.json` and re-checked by the close
  audit; `archive.rows` in the same file carries the correct measured
  value.
- The off-card count cache (131,084,704 distinct 8-grams over 2^30 rows,
  built in 14.5 CPU-minutes) and the load-only stage path (13 s load,
  52 s bind) are the engineering template for any successor archive that
  fits in RAM.

## What follows (registered decision rule 2)

**SAL-3: the 100 GB XSA transport** — the actual working-system
configuration: the same count law over the real 100 GB index via the XSA
service (the transport X43's consultation already ran at 32.8 steps/s),
co-trained host, fresh out-of-index fit/probe slice, LOCAL control and
zstd-dictionary baselines. The SAL-2 measurement predicts this is the
first scale where the count source's rare-context backing crosses the
usability threshold.
