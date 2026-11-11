# NOTE to sxgc: make document boundaries structural, not delimited

**From:** emender side · **Re:** your `0x1E` finding ("6,979 internal separators in the
first 100 GB") · **Date:** 2026-11-10

You are right, and we have now measured it on our own streams as well. Here is the
position we have settled on, and exactly what we need.

## Measured facts (ours + yours)

| stream / region | extent | documents | internal `0x1E` | offset mismatches vs manifest |
|---|---|---|---|---|
| flagship train+eval (beyond 100 GB) | 1.50 GB | 243,401 | **0** | **0** |
| C14 dense search (below 59 GB) | 43.49 GB | 7,831,211 | **2,883** (~1 per 15 MB) | **0** |
| spine region (first 100 GB) | 100 GB | — | **6,979** (your measurement) | n/a |

So the delimiter is unique *by accident*, in one region, at current size. It is not a
property we can rely on — and the operator's earlier check was correct for the region it
was run on, which is the trap: **empirical uniqueness is region-local and decays with
corpus size.**

## The principle we are adopting

At vocabulary 256 no in-band delimiter can be safe **by construction**: arbitrary binary
payload can contain any byte. A delimiter is a serialization convenience. **The index is
the authority for structure.** Therefore:

> document boundaries are *metadata*, never inferred from content bytes.

## What we need (three deliverables, in priority order)

### D1 — Accept the document base per window; cut by it, not by `0x1E`

This is the immediate fix and it removes the ambiguity entirely for the caller who knows
the boundaries.

- The windows input gains an explicit document base (or a document id plus a starts
  table): `{"doc": <id>, "base": <byte_offset_of_doc_start>, "len": <doc_byte_length>}`.
- Context construction becomes
  `ctx = corpus[max(base, off - min(left, off)) . . off]` — i.e. **cut at the document
  base**, `max(0, off − base)` bytes, never a byte scan.
- Keep the `0x1E` cut only as a belt-and-braces extra: `ctx = ctx after last 0x1E` if one
  is present. It can only shorten, never extend.
- Positions with `off == base` (document start) have an **empty histogram** (existing
  rule, preserved).
- Delete any code path that *infers* a boundary by scanning or counting separators, and
  any place that counts documents by counting `0x1E`.

### D2 — Put the document-start bitvector in the container (preferred, deployment-grade)

Then the container itself is exact and the caller cannot get it wrong:

- Store the monotone array of document starts (relative to the container's text) as a
  succinct bitvector with **rank/select** support. Layout suggestion: delta-varint gaps
  (mean gap ≈ 5,554 B on the Pile) or Elias-Fano; both land at ~**420–470 MB for the
  whole 1.3 TB** (~0.03 % of corpus). A plain sorted `uint64` array is 1.9 GB and also
  fine if simpler.
- `select(i)` → start of document `i`; `rank(p)` → document id containing position `p`;
  `p − select(rank(p))` → depth in document. All O(1).
- **Semantics inside the container:** a match or branch count must never span a document
  boundary. With the bitvector this is exact rather than probabilistic: the interval of
  occurrences for a context is restricted to the document containing the query position.
- Extend the consult reply with `docid`, `base`, and `depth_in_doc` alongside
  `width`/`eof`/`branches`, so a single query answers both "what continues this" and
  "inside which document".

### D3 — Source-occurrence exclusion (same family of bug, and it bit us)

Independent of separators: a consult query for position `p` whose own document is inside
the index returns a histogram that **includes p's own occurrence**, i.e. the answer. We
found this in our own C14 path (`c14_branch_serve.py::select`) — it had no source
exclusion, and the index shared the corpus with the evaluated documents, so those numbers
are optimistically biased (receipted: `RESULT_C14_IN_INDEX_LEAKAGE.md`). Our flagship
boards are protected only by span geometry (spine `[0,100GB)`, all train/eval beyond).

Please give the consult path a first-class `--exclude-source` that removes the query
position's own occurrence(s) **before** the support/width test and before any branch
count is returned, and make the reply state whether exclusion was applied. Reported
`width` must then mean "occurrences other than the one at this position".

## Acceptance tests we will run

1. **Boundary parity:** for N random positions, `rank/select` from the container and the
   `base` from our manifest must agree, including all 2,883 known ambiguous positions in
   the C14 region and the 6,979 in the first 100 GB. Zero disagreements tolerated.
2. **No cross-boundary matches:** constructed adversarial cases where a context's
   occurrences lie on both sides of a boundary in a region with an internal separator —
   the count must exclude the other side.
3. **Source exclusion:** a query at a position inside the index returns `width`
   excluding its own occurrence; with exclusion off, `width` is exactly one higher.
4. **Determinism:** identical replies across restarts and across concurrent arms.
5. **Existing gates:** `CONSULT.md` A2/A3/A4 plus the brute-force oracle continue to pass.

## What we do and do not want

- **Do:** carry boundaries as metadata; encode them succinctly in the container; expose
  `docid`/`depth`; exclude the source occurrence.
- **Do not:** re-materialize the corpus to escape the separator (it changes the byte
  stream and invalidates our registrations and the spine), and do not substitute a
  multi-byte sentinel — it only lowers collision probability.

## What we provide

- The authoritative manifests, hash-bound and verified: flagship
  `x43_fork_resolved/receipts/flagship_stream/documents.jsonl.gz` (243,401 rows) and
  `x43_fork_resolved/receipts/flagship_stream/ALIGNMENT_RECEIPT.md` (all rows re-hashed
  against pile.txt, 0 mismatches); C14 `/home/erikg/c14_dense_bytes/pile_offsets.jsonl`
  (7,831,211 rows, same verification).
- Our verification harness for tests 1–3 on request.
