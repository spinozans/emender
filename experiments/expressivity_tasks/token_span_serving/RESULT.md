# Fixed-weight serial read: cached operators help at refresh, not necessarily over the whole stream

**Correctness.** On invented documents, 18 CPU tests passed. Every real next-token logit was compared to the unchanged training forward running in the Triton CPU interpreter with candidate pages enumerated independently: 2 hosts × 8 reader/controls, with explicit replay and with bounded compiled packets. This covered 23 frontiers/cell including different seeds querying overlapping pages, zero support, window rollover and a short final window, then a seven-token document reset. Raw ranges and native-token ranges resolved to the same bytes; a document represented only by duplicates was still refused. Weight replacement in the caller left the old mount unchanged. These are FP32 CPU checks, not native BF16 checkpoint parity or an additional run on natural text.

**Latency.** One thread of an AMD EPYC 7713, full 50,281-way next-token logits, width 128, 6,767,112 allocated model parameters. Four new live streams × 64 observations; one search each at token 16. The repeated pass revisited all 16 selected pages **in the archive**, with no live state carried over from priming. In the table, “cold”/“warm” refer to the learned page cache, not a cold machine. Five alternating-order blocks; units ms. [Full aggregate](receipts/serial_cpu.json) includes the ordinary-position clocks, uncached first pass and LOCAL controls.

| Host / reader | Refresh: replay / cold compile / warm cache | Whole ms/token: replay / warm cache | Learned LRU tensor payload (16 pages) |
|---|---:|---:|---:|
| GDN‑2 / GDN‑2 STATE | 6.955 / 6.998 / 2.413 | 1.931 / 1.731 | 65 KiB |
| GDN‑2 / GDN‑2 EFFECT | 6.917 / 10.334 / 2.618 | 1.826 / 1.940 | 129 KiB |
| E97 / E97 STATE | 7.705 / 7.802 / 2.420 | 2.028 / 1.760 | 65 KiB |
| E97 / E97 EFFECT | 7.691 / 7.781 / 6.543 | 1.942 / 1.872 | 513 KiB |
| GDN‑2 / mixed EFFECT | 10.711 / 14.126 / 6.237 | 1.984 / 1.834 | 321 KiB |

For GDN‑2 EFFECT, reusing `(A,B)` cut the measured refresh from **6.92 to 2.62 ms** on this favorable workload. Building those maps was slower than replaying the pages once. The warm refresh reduction **did not yield a full-stream speedup for this cell**: only one in 64 tokens did a lookup, and all needed both recurrent blocks and the vocabulary head. A cold E97 EFFECT packet saves controls, not a nonlinear terminal map; there is still a seeded replay at every refresh. Blank-start E97 states can be cached directly. The largest cache-vs-replay logit difference at full geometry on the untimed fixture window across the seven metered cells was `3.58e-7`. No natural reuse rate, model-quality result or GPU serving price follows from these CPU clocks.
