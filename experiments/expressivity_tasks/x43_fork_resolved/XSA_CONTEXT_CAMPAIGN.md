# XSA-backed controller: campaign start

Goal: replace re-feeding long context with a growing, causally queryable index of visible bytes. Keep the neural controller and its programmable scratch memory distinct from that index. Freeze the LM baseline. This campaign does not depend on K24 or authorize board training.

## First contact: existing sealed XSA, not a mock

Read-only CPU smoke, on the existing local release executable. Emender base `e784d327`; SXGC checkout `ea90a637`. The binary identity is the SHA below, not an assertion that it was built from the current checkout. This is a pinned regression witness, not independent verification of the histogram.

```
sha256sum ~/sxgc/xsa/target/release/xsa /tmp/nxc/x1/x1a/x1a-trie.sxi2
28e6d41415482f3d2d6ae9692be9d391752bec5e8931e4c39561dcb315da962b  xsa
aa8f3cc1fff6c72035055ffebbf5906b9a14adb8f0408772145bb20f2c967f7f  x1a-trie.sxi2

~/sxgc/xsa/target/release/xsa branch-of /tmp/nxc/x1/x1a/x1a-trie.sxi2 the
present count=3 depth=3 children=3
child 0x06 count=1 node=-
child 0x12 count=1 node=-
child 0x7f count=1 node=-
```

Negative controls: `branch-of` refused `/tmp/nxc/x5gate/map/pile150.sxi2` (v10 without m20) and `/tmp/sw-fx/t150m-v10.sxi2` (boundary-unaware m20). Do not bypass refusals with `--cross-doc`. The host fixture is not vendored or a portable test dependency. Replace it with a reproducible tiny build before CI.

## Delivery order and stop conditions

1. **One read contract, two backends.** Start with XSA's supported `seek`/context lookup and branch histogram, including structural END and distinction between missing materialized nodes and absent text. Use the same caller for a sealed snapshot and a live session. Append is a writer operation; implementation details of base + tail stay below this API. Keep context identity logical; physical node IDs are generation-local. Do not silently approximate unsupported ops. See `~/sxgc/MEMORY_OPS.md`, `xsa/src/ctx.rs`, and `DESIGN_SUFFIX_ATTENTION.md` in this directory.
2. **Conformance before speed.** Build a deterministic binary, multi-document fixture. Replay it byte by byte; at selected cutoffs compare live and freshly sealed reads against a brute-force oracle. Include overlap, patterns spanning flush boundaries, END vs literal 0x1e, and contexts whose continuation is not yet visible. Distinguish open-prefix EOF from a confirmed document end; neither is a predicted content byte. Seal/compact at alternate boundaries: observable answers must not change. Pin corpus, manifest, binary and container identities. Any disagreement blocks controller training.
3. **Measure the product budget.** At 100K, 500K, 1M visible bytes, measure append + real query p50/p99, worst compaction stall, RSS, and batched causal prefill. Include transport and compare to the measured decoder step on the same host, not offline build throughput. Do not put a CLI spawn in the decode loop.
4. **First controller experiment.** Same held-out documents and target positions; bounded local neural input. Compare Emender index-off/on against fixed lookup/backoff and GDN2 with equal interface and query budget. Include a longer-context reference. Register thresholds before training. Require benefit at stated latency and a causal intervention (mask or swap answers); index access alone is not evidence for learned memory management. Test programmable writes separately.

Milestone reached: sealed, boundary-aware `branch-of` smoke only. Online append, compaction equivalence, decode latency and controller gain are not yet demonstrated.
