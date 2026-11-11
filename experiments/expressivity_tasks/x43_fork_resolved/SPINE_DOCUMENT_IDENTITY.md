# The spine's document identity, and how it relates to our manifest

**Question (operator):** we made document separation an out-of-system definition —
documents are assigned independently of a character in the stream. Does that hold for
the XSA build too, and how does the container's notion of a document relate to ours?

## Answer in one line

The container is a **flat text of `pile.txt[0:100e9]` with the separators indexed as
ordinary bytes**, and its *document* notion is defined **out of band** by a sidecar map
built from the Pile's own shard records. Our boards are defined by our own manifest. The
two derivations are independent, and they agree **byte for byte**.

## The two authorities

| | derivation | result |
|---|---|---|
| **sxgc pilemap** (`/mnt/nvme2n1/erikg/pilemap100`, 891 MB) | `xsa corpus --pile-map`: for shard 00..29 in filename order, each record in file order, the JSON-unescaped `text` followed by one `0x1E`; offsets from **record boundaries**, then every document checked byte-for-byte against the pile | 16,119,767 docs over 100e9 bytes; 16,119,766 own a separator; **6,979 internal `0x1E`**; identity `separators == owning + internal` holds |
| **our manifest** (`pile_offsets.jsonl`, `documents.jsonl.gz`) | per-document `(byte_offset, byte_length, sha256)`, hash-verified at consumption | 7,831,211 rows / 43.49 GB (below 59 GB) and 243,401 rows / 1.50 GB (beyond 100 GB), **0 mismatches** |

Their byte exactness gate: `pile[offset, offset+length)` SHA-256 == the shard record
text SHA-256, **and** `pile[offset+length] == 0x1E`, for all 16,119,767 documents
(`DOCHASH mismatches=0`), plus 400/400 re-extracted from the shards, 200 of them chosen
*because* they contain a literal `0x1E` (`DOCHASH`/`DOCVERIFY` in
`~/sxgc/RESULTS_PILE_DOCMAP.md`).

So the container and our boards are two independent readings of the same bytes, and both
report zero disagreement. That is the sanity check.

## Where the internal separators live (not a uniform hazard)

The 6,979 internal `0x1E` in the first 100 GB are concentrated:

| Pile component | internal `0x1E` |
|---|---:|
| FreeLaw | 4,733 |
| PhilPapers | 2,006 |
| Ubuntu IRC | 221 |
| StackExchange | 18 |
| OpenSubtitles | 1 |

i.e. legal filings and pseudonymous chat logs — text that legitimately contains control
characters. That is why no byte can be a boundary oracle, and why deriving boundaries
from the *records* is the only sound method.

## Do we use the separator to break documents during training?

**No.** Training and evaluation boundaries come from our manifest; every document slice
is SHA-256 verified at consumption, and the loader raises on any mismatch
(`byte_stream.py`). The separator is used in exactly one place — the consult query's
context cut — and that use is *conservative*: it can only shorten a context, never extend
it across a boundary.

## The 100 GB seam, examined

From their map: the last *complete* document of the 100 GB range ends at
**99,999,999,398**; the remaining **602 bytes** are a head-cut partial tail
(`partial_tail=1`, owns no separator). Our first flagship document starts at
**100,000,000,071**.

Therefore the one document that **straddles** the boundary spans
`[99,999,999,398, 100,000,000,070]` (602 B below the line + 70 B above it, separator at
100,000,000,070) and is **excluded from both training and evaluation** — our stream
begins at the next true document boundary. The span-disjointness argument therefore holds
exactly, not approximately.

Residual, stated: the straddling document's 602-byte **prefix** is inside the container's
indexed text (it is part of `pile.txt[0:100e9]`). Nothing is trained or evaluated on it,
so it is not leakage; it is simply index content that is never a target.

## New capability this unlocks

Their `xsa query <container> --docs <pilemap>` already returns, per hit:
`pattern_id, doc_id, offset_in_doc, pile_set_name, global_offset` — verified live on our
container:

```
QUERY_DOCS source=docs docs=16119767 corpus_bytes=100000000000
QLITE_OPEN ... n=100000000001 runs=31381597694 ...
```

That is exactly deliverable D2 of `SXGC_DOCUMENT_BOUNDARIES_NOTE.md`, already built. It
also gives us the mechanism for **source-occurrence exclusion** (D3): a hit whose
`doc_id` equals the query position's own document can be filtered, because document
identity is now explicit rather than inferred from bytes.

## What would be insane (none of which we are doing)

- Trusting a delimiter as a boundary oracle. *(Not doing it: training uses the manifest;
  the only delimiter use is a conservative context cut.)*
- Re-materializing the corpus to escape the separator. *(Not doing it: it would change
  the byte stream and invalidate both registrations and the spine.)*
- Carrying two documents of truth and not reconciling them. *(Doing the opposite: their
  16,119,767-document map and our 8.07M-row manifest both verify byte-exactly, and the
  separator census accounts for every byte of the difference.)*

## The container's text is provably pile.txt's first 100 GB

Their manifest records `pile_sha256_file = de79fe96057b604a71513770111f246ab0e936d3cf56415bcc5a43bba60bbbfd`
for `pile-head100.txt` — the file our disk reclaim deleted as a verified
prefix-duplicate. Re-derived from the canonical source instead:

```
sha256(pile.txt[0 : 100,000,000,000]) = de79fe96057b604a71513770111f246ab0e936d3cf56415bcc5a43bba60bbbfd
                                         100,000,000,000 bytes in 122 s
```

**Identical.** So the spine's indexed text is exactly `pile.txt[0:100e9]` — the
span-disjointness geometry of the boards is not an assumption but a hash equality — and
the deleted copy was provably redundant (re-derivable in two minutes).
