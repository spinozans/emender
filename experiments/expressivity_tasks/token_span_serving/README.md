# Serve a learned token-span read from actual observations

`ReadSession.observe(token_id)` updates the two host memories with **one** actual token and returns next-token logits over the native vocabulary (no padded classes). No scored targets, predicted alternatives or retrieved passage tokens enter that transcript. This is an FP32 CPU reference using the small [co-trained token architecture](../retrieved_span_effects/NATURAL_PROTOCOL.md), not a production-checkpoint adapter or a speed claim about GPU decoding.

```python
# Authorized independent reference documents in bytes, native tokenizer and one TokenLM.
archive = Archive(reference_docs, encode_bytes, one_token_spelling,
                  tokenizer_sha256=tokenizer_hash, vocab=model.sh.vocab, page=model.sh.page)
host = FrozenHost(model.cpu().float(), archive, tokenizer_sha256=tokenizer_hash,
                  cache_bytes=2 * 1024**2)  # zero selects explicit on-demand replay
stream = host.session(live_source_identity)
logits = stream.observe(observed_id)
# Only observe another ID if it was selected/observed; logits do not mutate the stream.
stream.reset(next_source_identity)  # before reading a different document
```

`Archive` verifies byte-for-byte tokenization, hashes complete reference documents and attaches `(archive version, ordinal, source hash, token [lo,hi), byte [lo,hi))` to each actual hit. Identical pages share their first canonical occurrence. An excluded document stays excluded even when **all** its pages were duplicates. Live identity comes from trusted external role metadata. It is used only to reject archive membership, not to rank hits or compute logits. This does not detect paraphrases, guarantee a dishonest ID is correct, or license a data source. Never mount an unobserved live continuation. The natural training panel uses the same page rule, not this new provenance image; no old text is reopened to retrofit its addresses.

The session keeps two live recurrent matrices, the unfinished query prefix and a window-local weighted passage memory. At actual token 16 the **lower** state produces the seed and the learned reranking query. Only at that point does it search once, fetch and weight up to four document-local pages. A later token asks that saved memory a *new* question from its own lower-host state; the upper host still advances causally. After 64 real inputs, the next real token starts both host memories and the read schedule afresh, as in training. `reset` drops everything live immediately. A miss supplies no hint including in LOCAL. No repeated full-prefix replay to produce each prediction.

### Cache keys and what can be reused

A mount takes a private frozen copy of the host. Its binding covers the weights and named geometry, precision/runtime convention, exact archive and tokenizer, and window/reset semantics. A new training generation requires a **new mount**; a live seed or a pooled result must never be cached against passage ID alone. The optional serial, byte-bounded LRU holds only the page key and its seed-independent packet for this mount:

| Reader | Saved packet | Operation at a new frontier |
|---|---|---|
| STATE / NEUTRAL, either family | `F_i(0)` | weighted read; add current seed only for NEUTRAL |
| GDN‑2 EFFECT | `(A_i, B_i)` | apply to **current** seed, then pool |
| E97 EFFECT | passage controls | explicit nonlinear replay on **current** seed, then pool |
| MIX-EFFECT | GDN‑2 pair for half the heads, E97 controls for the other half | independent applications in their own spaces |

With `cache_bytes=0`, encode and replay each selected page each lookup. All variants reuse the resulting memory only **within** their live window. `cache_payload_bytes` counts retained tensor payload, not peak scratch, Python overhead, encoder weights or the page index. Compiled packets can be much larger than raw token IDs. Mount/session mutation from concurrent threads is not supported by this reference. Don't use it as a new optimizer path or serve a graph from changing weights.

### Qualification and price

`test_session.py` compares each real serial forecast against the unchanged batched training forward in the Triton CPU interpreter, with independent literal enumeration of routed pages. Covers 16 host/reader cells, both cache modes, different live seeds using common pages, misses, short documents, exact boundary resets, stale handles, duplicate pages, rejected admissions, eviction, runtime precision and weight changes. No corpus, panel membership, or fitted array is opened. A first test-collection failure (`attempted relative import beyond top-level package`) was retained and repaired by using absolute imports across experiment packages.

`meter.py` is a separate **synthetic CPU serving clock**: same 50,281-token vocabulary and width-128 geometry; randomly initialized weights, 512 invented 32-token pages. One stream at a time, actual next-logit API, 4 × 64 observed tokens/pass; first-16 prefix has support on four pages. Price 5 alternating-order blocks of two passes, cold (empty learned LRU) then repeat (same four topics and same weights but *fresh* live sessions). Report full ms/token, lookup-position and ordinary ms, builds/reuses and retained payload. The repeat is intentionally reuse-friendly, not a natural page reuse estimate. Excludes mount/build, tokenizer, sampling, network and disk I/O; both cold and repeated passes include the actual search. Never infer GPU/token-generation throughput or natural quality from this clock.
