# Consult boundary hardening — base cut, source exclusion, short-spine amendment

**Date:** 2026-11-11. **Status:** implemented and unit-tested; **no registered board's
behaviour or number changed**. This is a prerequisite build for any consulted arm, not a
measurement and not a training authorization. Budget: CPU only, no GPU, no Slurm.

Motivation: [`RESULT_C14_IN_INDEX_LEAKAGE.md`](RESULT_C14_IN_INDEX_LEAKAGE.md) found that
the consult path never excluded the evaluated position's own occurrence, so every
in-region consulted reading is optimistically biased; and
[`SXGC_DOCUMENT_BOUNDARIES_NOTE.md`](SXGC_DOCUMENT_BOUNDARIES_NOTE.md) established that a
document boundary is metadata, never a byte scan (`0x1E` is content: 0 internal in the
flagship stream, 2,883 in the C14 region, 6,979 in the first 100 GB — see
[`SPINE_DOCUMENT_IDENTITY.md`](SPINE_DOCUMENT_IDENTITY.md)).

## 1. What changed

`c14_branch_serve.py` (the registered consult path module) gains four pure functions and
two opt-in keyword arguments; nothing else in the repository changed behaviour.

| Name | Role |
|---|---|
| `consult_context(text, base, off, left=None)` | the document-base context cut, `corpus[max(base, off-left) .. off]`, with the `0x1E` cut kept as a belt-and-braces extra |
| `exclude_occurrence(reply, next_byte, count=1)` | pure arithmetic: the query position's own occurrence removed from `width`/`eof`/`branches` |
| `is_source_occurrence(hit, doc_id, offset_in_doc)` | the exact filter rule that attributes an `xsa query --docs` hit to the query position |
| `spine_left(policy=None)` | the spine-policy config flag (registered default; the amendment is a flag only) |
| `END_BRANCH` | witness sentinel: the position's own occurrence ends the corpus (the END branch) |
| `select(..., spans=…, excludes=…, left=…)` | the registered selector, now able to (a) cut contexts at the document base and (b) apply exclusion before every support/width test |

## 2. Exact new semantics

### 2.1 Base cut (`consult_context`)

```
ctx = corpus[max(base, off - left) .. off]      # never crosses `base`
ctx = ctx after its last 0x1E, if one is present # belt-and-braces, can only shorten
```

- `base` is the document's `byte_offset` from the window manifest; `text` is that
  document's bytes from `base`. A caller that has already trimmed at a stricter
  in-document floor (a window start, as `DocumentWindows.causal_context` does) passes that
  floor as `base`; the function never reads outside `text`, so it can only shorten a
  caller's admissible prefix.
- A **document shorter than `left`** contributes only its own bytes; no padding, no
  neighbour bytes.
- `off == base` (the document's first byte) yields the **empty context**, whose histogram
  is empty **by rule** — preserved: `select` never queries it and the row keeps
  `context_bytes = 0`, `context_count = 0`, `coverage = False`, all 260 features zero.
- The last-`0x1E` cut is retained exactly as an extra, and is now documented as what it
  is: a conservative loss inside documents that carry an internal separator (FreeLaw,
  PhilPapers, IRC logs), never a boundary. Nothing in the path *infers* a boundary from a
  byte.
- Geometry is validated: `off < base`, `base < 0`, `left < 1`, or document bytes shorter
  than the depth to `off` all raise rather than silently clamping.

`select` routes through it only when given spans: `spans` is a parallel sequence of
`(base, off)` pairs, and then `lefts[i]` is the document's bytes from `base`. Without
`spans`, `select` applies the historical `[-left:]` + `0x1E` cut so every registered
caller and receipt keeps its numbers (§4).

### 2.2 Source-occurrence exclusion (`exclude_occurrence` + `excludes=`)

- `excludes` is a parallel sequence of the query position's own continuation: an int byte
  `0..255`, `END_BRANCH` when that occurrence ends the corpus, or `None` when the
  position's document is **known** (by span geometry) to be outside the index.
- With `excludes` given, the excluded counts are applied **before every support and width
  test** — the binary search itself, the exact-`k` support rates, and the final row all use
  `histogram(i, k)`, which subtracts the position's own occurrence. Therefore
  `context_bytes` becomes "the longest depth whose POST-exclusion width is nonzero", i.e.
  a censored maximal context whose only occurrence is the source **collapses to depth 0**
  instead of reading the answer back.
- `width` (`context_count`) then means **"occurrences other than the one at this
  position"**, and `support_size`, `distinct_branches`, `entropy`, `coverage` and the 260
  features are all derived from the excluded histogram.
- The reply states it per row: **`exclude_source: True`** is present on every row produced
  in exclusion mode. The key is **absent** in default mode, where the counts are the raw
  histogram — absence means raw, and registered rows stay byte-identical (§4).
- A witness that is not in the raw counts is an **error**, never a silent no-op: a skipped
  exclusion is leakage, not a smaller count. Emitted branch order (count descending, byte
  ascending) is preserved; a branch whose count reaches zero is dropped.

**The filter rule for attributing a hit to the query position** (documented in
`is_source_occurrence`): `xsa query <container> <context> --docs DIR` returns one row per
occurrence carrying `doc_id` (0-based container document id) and `offset_in_doc` (the
occurrence's byte offset inside that document). A row **is** the query position's own
occurrence iff **both** equal the query position's own pair:

```
hit['doc_id'] == doc_id  and  hit['offset_in_doc'] == offset_in_doc
```

A container position is unique, so the pair identifies at most one row. Other occurrences
**inside the same document** are *not* the position's own occurrence and stay in the
counts; removing those is `--doc-restrict` (histogram restricted to the containing
document), a different question that this change does not answer. A hit missing either
field raises `KeyError` rather than quietly failing to exclude.

Exclusion is **per position**, not per pattern: the same context can be supported for one
position and source-only for another, so the shared reply cache stays raw and the
subtraction happens in the per-position view.

### 2.3 What the arithmetic assumes, stated plainly

The witness is the continuation byte the source occurrence contributed — which the caller
knows, because it is the target byte of the position being evaluated. Subtraction uses it
only to decrement; it cannot add information. `count` is 1 per container position (a
suffix-array container yields positions, not multiplicity).

## 3. Test evidence

`experiments/expressivity_tasks/x43_fork_resolved/test_c14_consult_boundaries.py` — 19
tests, all pure and deterministic (no binary, no corpus, no network):

- base cut: context never crosses the document base; a document shorter than `left`; a
  position exactly at `base` (empty context → empty histogram → all-zero features);
- the known-ambiguous document class (a legal-filing/chat-log shape with many internal
  `0x1E`): every position's context stays inside its document and the extra cut only ever
  shortens; an internal separator is shown to be content, with the base cut doing the
  boundary work;
- geometry rejection and determinism (repeated calls, identical outputs);
- exclusion: the source occurrence as the **only** one ⇒ `width` 0 and empty branches; a
  shared histogram decremented once with order preserved; the END branch; absent witnesses
  (missing byte, empty reply, `count` > available, bad types) all raise; determinism;
- exclusion applied **before** the support test, end-to-end through `select` (source-only
  context ⇒ depth 0 and all-zero features; other-position support retained); the
  `exclude_source` reply flag; asserted-absent positions keep raw counts; cardinality
  mismatches raise;
- a brute-force property check that the excluded histogram equals the raw histogram minus
  exactly one position (60 of 60 positions over 12 randomized repetitive corpora; all of
  them keep a maximal context after exclusion, so every one is comparable);
- spans: `select(..., spans=…)` equals `select(consult_context(…))` and censoring follows
  the cut context.

Commands and results (this session, `/home/erikg/emender/.venv/bin/python`, CPU only):

```
$ .venv/bin/python -m pytest experiments/expressivity_tasks/x43_fork_resolved/test_c14_consult_boundaries.py -q
19 passed
$ .venv/bin/python -m pytest experiments/expressivity_tasks/x43_fork_resolved/ -q \
    --ignore=…/test_c14_transformer_stacking.py --ignore=…/test_c14_stacking.py
124 passed, 23 skipped   (the two ignored files fail collection on pre-existing
                         relative-import errors unrelated to this change)
```

Default-path parity with the pre-change selector (randomized, 80 corpora including
`0x1E`-bearing ones, comparing `(rows, rates, cache size)` of the pinned
`e437d7a2` implementation against the new one): **0 mismatches**. This is the evidence that
no registered number moved.

## 4. Registered boards: unchanged, and why the defaults are off

Every registered consultation number was produced by `select(lefts, query)` with no
spans, no exclusions and a 1024-byte cap. That call is byte-identical to before, verified
above and by the pre-existing tests (`test_c14_branch_serve`, `test_c14_cache_scale`,
`test_c14_registered_cache`, `test_flagship_union_service`). Enabling either hardening for
a registered board would change its feature stream and therefore its numbers, so both are
**opt-in** and no registered runner (`flagship_union_service.py`, `c14_registered_cache.py`,
`c14_assembly.py`, `p100_heldout.py`, `c14_branch_serve.main`) passes them.

Consequence of touching the module at all: `PROGRAM.md`'s claim-bindings ledger pins
`c14_branch_serve.py` at `e437d7a2`; the pin is re-pinned to this change's commit with a
`supersedes` note (the established practice — see `f1cf6c8a`, `86147195`), which keeps the
fail-closed audit green while recording that the *file* no longer reproduces the exact
bytes that produced the C14 serving-preflight receipts. No receipt file, measurement,
registration table or claim status is modified. Scripts that assert
`sha256(c14_branch_serve.py) == <historical selector_sha256>`
(`receipts/c14_union_first_hours/verify.py`, `c14_transformer_reconcile.py`) will now
refuse — that is the identity binding working, not a regression: they no longer attest a
reproduction.

Audit state after the re-pin: **all 461 `emender` bindings in the ledger hash-match the
working tree**. `test_validate_program::test_valid_ledger` is nevertheless still red, on a
**pre-existing and unrelated** binding: `sxgc/xsa/src/branches.inc.rs` is pinned at
`88d4e074`/`badad614…` and the read-only `~/sxgc` tree has since advanced (working
`f293d1c0…`, HEAD `6cf19f2`). It was red before this change as well — the validator raises
on the first mismatch in ledger order, and the `emender` pin happened to come first. That
pin belongs to the sxgc-side registration and `~/sxgc` is read-only for this task, so it is
reported, not re-pinned.

## 5. Short-spine amendment (dated annotation; config flag; no behaviour change)

**Annotation S-1, dated 2026-11-10/11, status: proposed, NOT active, NOT implemented.**

> **Registered text being amended.** "Maximal supported causal suffix cap 1024, clipped at
> window reset." ([`PREREGISTRATION_C14_BYTE.md`](PREREGISTRATION_C14_BYTE.md));
> "the registered 1024-byte causal consultation suffix cap"
> ([`PREREGISTRATION_150M_LATTICE.md`](PREREGISTRATION_150M_LATTICE.md));
> "use a causal maximal-supported-Wikipedia context selector capped at 1024 bytes"
> ([`PREREGISTRATION_C14.md`](PREREGISTRATION_C14.md)).
>
> **Proposed replacement.** The consult feature is a **variable-length spine of 8–16
> bytes with support-weighted relax**: start at the 8-byte floor, ascend one byte at a time
> until the context has enough occurrences (support-weighted, not merely nonzero), and stop
> at the 16-byte ceiling; the emitted `context_bytes` is then the length actually used, and
> the registered 256-probability geometry is unchanged.
>
> **Why — measured depth.** Support is concentrated in the first handful of bytes and
> collapses with depth: C14 training-interior support is 36.59 % / 2.36 % / 0.59 % /
> 0.09 % at 16 / 64 / 256 / 1024 bytes, with a **median supported length of 12 bytes**
> ([`PREREGISTRATION_C14.md`](PREREGISTRATION_C14.md),
> [`PREREGISTRATION_C14_BYTE.md`](PREREGISTRATION_C14_BYTE.md)), and sxgc measures 1.4 % of
> positions with a 1024-byte context occurring elsewhere
> ([`RESULT_C14_IN_INDEX_LEAKAGE.md`](RESULT_C14_IN_INDEX_LEAKAGE.md)). A 1024-byte budget
> is ~100× deeper than the data supports; it spends the whole budget to arrive, almost
> always, at a censored maximal context.
>
> **Why — the leakage reason.** A censored maximal context at that depth is very often a
> **unique** occurrence, and the unique occurrence is usually the evaluated position
> itself: before §2.2, `width` was frequently exactly 1 and the histogram `{target: 1}`, so
> the deep spine read the answer instead of the corpus's general behaviour. With source
> exclusion the deep spine collapses to zero support far more often than a shallow one, so
> depth 1024 buys nothing in-region and short depth is what remains after exclusion
> out-of-region as well. A short spine also keeps the feature in the regime where
> occurrences are *shared* — legitimate transfer — rather than unique — memorisation.
>
> **Scope of this annotation.** It changes no registered board. It is a proposal for the
> next registration of a consulted arm, and it must be re-measured (support at 8/12/16
> bytes with source exclusion, in-region and out-of-region) before it is nominated.

**Config flag** (`c14_branch_serve.py`): `SPINE_POLICY = SPINE_POLICY_REGISTERED`
(`'registered-1024'`, the default) versus `SPINE_POLICY_SHORT` (`'amended-short-8-16'`);
`SHORT_SPINE_BYTES = (8, 16)`. `spine_left()` resolves the flag and is the single source of
the context cap for both `consult_context` and `select`. The amended policy **raises
`NotImplementedError`**: the support-weighted relax is unimplemented and no registered
board may silently consume a different feature definition. `spine_left()` also raises on an
unknown policy string.

## 6. What is still NOT done

1. **The live native path has no exclusion of its own.** Our running path is `xsa branches
   --serve -k 256`, whose framed reply carries only `(interval, width, eof, branches)` — no
   document id, no offset, no per-row exclusion flag. It therefore cannot exclude anything
   natively, and this change does not pretend it can: the arithmetic (§2.2) and the filter
   rule are ours, and a caller must supply the witness. sxgc has since built exactly the
   native capability requested in
   [`SXGC_DOCUMENT_BOUNDARIES_NOTE.md`](SXGC_DOCUMENT_BOUNDARIES_NOTE.md):
   `xsa consult --container … --windows … [--docs DIR] [--exclude-source]
   [--doc-restrict]`, with a versioned row carrying `docid`, `base`, `depth_in_doc` and a
   per-row `source-excluded` flag (`~/sxgc/CONSULT.md` §§1–3). **Adopting that native path
   in place of `xsa branches --serve` is a separate build task** (window set emission,
   row-code mapping into `FLAGSHIPSPARSE2`, `--strict` parity, R2/R4 determinism and replay
   receipts per [`LIVE_CONSULT_IDENTITY.md`](LIVE_CONSULT_IDENTITY.md)) and is not done
   here.
2. **No caller is switched on.** No runner emits `(base, off)` spans or witnesses, because
   doing so changes a registered board's feature stream. A consulted arm that wants honest
   in-region numbers must pass `spans=` (with document bases from the manifest, exactly what
   `DocumentWindows` already knows) and `excludes=` (witnesses from `xsa query --docs`
   filtered by `is_source_occurrence`, or `None` only where span geometry proves the
   position's document is outside the index).
3. **No real-data boundary parity run.** The 2,883 (C14) and 6,979 (spine) internal-`0x1E`
   offsets were **not** re-run here: this task was forbidden from touching `pile.txt` and
   the containers, and its tests use constructed analogues of the FreeLaw/PhilPapers class.
   The census numbers above are the cited receipts' measurements, not new evidence.
4. **No consulted number was re-measured or re-reported.** In-region consulted and
   array-alone numbers remain as annotated, optimistically biased and not held-out clean.
   Nothing here converts any of them into a clean number.
5. **`--doc-restrict` semantics are not implemented.** The exclusion filter removes the
   position's own occurrence only; same-document *other* occurrences remain in the counts.

## 7. Residual risks

- A caller can pass `excludes=[None, …]` and thereby assert "outside the index" without
  proof. `None` is documented as an assertion of span geometry, and the only safe use is a
  document whose span lies entirely outside the spine; the code cannot check that for the
  caller.
- The belt-and-braces `0x1E` cut is a real, if small, feature distortion inside the
  ambiguous documents (those positions lose pre-separator context). Truncating at the
  document base removes the *boundary* question but not this conservative loss; the byte
  census bounds it at 2,883 + 6,979 offsets over 43.49 GB + 100 GB.
- The exclusion arithmetic is exact only if the container's occurrence set is complete for
  the queried context; a truncated or `--limit`ed occurrence walk would make the witness
  check fail loudly (by design) rather than silently.
