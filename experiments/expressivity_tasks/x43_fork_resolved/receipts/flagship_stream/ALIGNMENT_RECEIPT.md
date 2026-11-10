# Stream alignment receipt — every registered document re-hashed against pile.txt

**Date:** 2026-11-10 · **Trigger:** sxgc's finding that `0x1E` is not unique in the
Pile (6,979 internal occurrences in the first 100 GB). We tested whether that
compromises our boundaries.

## Method

For every record in `documents.jsonl.gz` (the frozen flagship stream manifest):
`pread(pile.txt, byte_offset, byte_length)` and compare SHA-256 against the record.
Read-only; no materialization; offsets as registered.

## Result

```
documents verified : 243,401
hashed bytes       : 1.500 GB
mismatches         : 0
splits             : train 238,461 · final_test 2,462 · calibration 2,478
internal 0x1E inside document bodies : 0
```

## What this settles

1. **Our offsets are exactly aligned with pile.txt.** Zero mismatches over the whole
   registered stream, all three splits. Document boundaries come from the manifest,
   never from scanning bytes.
2. **In the training/evaluation stream the record separator IS unique** — zero `0x1E`
   bytes inside any of the 243,401 document bodies. The 6,979 internal separators sxgc
   measured are in the **first 100 GB**, which is the *spine* region: index content,
   not our training text.
3. **Therefore queries never contain the separator** (contexts are cut at it and none
   exist inside training documents), so a consultation match cannot cross a document
   boundary via a separator. The residual ambiguity is index-side only, and comes to
   ~7.0e-8 of bytes.

## Still to do (unchanged)

- Cut consult contexts at the **true document base** carried in the window index rather
  than at the last `0x1E` — makes the guarantee exact instead of probabilistic and
  stops truncating contexts inside the spine region's 6,979 ambiguous places.
- Repeat this receipt for the C14 offsets (7,831,211 rows) so the dense-search stream
  has the same explicit evidence.

## Second stream: the C14 dense-search offsets (7,831,211 rows)

Same method, same source, read-only:

```
rows verified  : 7,831,211
hashed bytes   : 43.49 GB
elapsed        : 94 s
mismatches     : 0
internal 0x1E inside document bodies : 2,883
```

**Offsets: exactly aligned (0 mismatches).** So both streams are immune to the
separator ambiguity where it matters — document boundaries come from the manifest and
are hash-verified at consumption, not inferred from bytes.

**But the two streams differ in separator uniqueness:**

| stream | region | internal `0x1E` |
|---|---|---|
| flagship (train + eval) | beyond 100 GB | **0** in 243,401 documents / 1.50 GB |
| C14 dense search | below 59 GB | **2,883** in 7,831,211 rows / 43.49 GB (~1 per 15 MB, ~1 per 2,715 documents) |

## Consequence, precisely

- **Training is unaffected in both streams**: boundaries come from the manifest, each
  document is SHA-verified at consumption, and the running searches have been doing
  exactly that for 123+ candidates without a single mismatch.
- **Consult contexts are the only affected path.** Cutting at the last `0x1E`
  (a) is always *conservative* — it can shorten a context, never extend it across a
  boundary — and (b) truncates early in the ~1-in-2,715 C14-region documents that
  contain an internal separator. Since the consulted arms draw their contexts from the
  **beyond-100 GB** stream (zero internal separators), they are unaffected today.
- **The fix is still worth making**, because it converts a probabilistic guarantee into
  an exact one: carry the document `base` in the window index and cut at
  `max(0, off − base)`.

## Verdict

`0x1E` is not a valid boundary *oracle* — sxgc is right about that. It is not a data
integrity problem for this program: both registered streams verify byte-exactly against
their manifests, and the one stream we consult from has no internal separators at all.
