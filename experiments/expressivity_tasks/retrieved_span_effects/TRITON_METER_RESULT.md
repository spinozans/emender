# GPU result: the reverse read saves work, but not every step is a reverse read

On one broker-leased RTX 6000 Ada, in FP32, the single-query GDN‑2 read at 256 groups × 4 hits × 64 passage tokens × 32×32 state took **17.498 ms** for PyTorch matrix replay plus backward, **0.910 ms** for Triton matrix replay plus backward, and **0.475 ms** for the Triton reverse-vector read plus backward (4 rows/program). Incremental peak allocations were 532.4, 297.1 and 41.4 MiB, respectively. The reverse implementation is algebraically equivalent for this read; it does not approximate or freeze the seed gradient. This is not an index lookup or a whole-model speedup.

| Groups × hits × tokens × K × V | Dense replay + backward | Triton matrix | Triton reverse, fastest measured rows/program |
|---|---:|---:|---:|
| 64 × 4 × 16 × 16 × 16 | 5.653 ms | .888 ms | .470 ms (8) |
| 256 × 4 × 64 × 32 × 32 | 17.498 ms | .910 ms | .475 ms (4) |
| 512 × 4 × 32 × 32 × 32 | 9.234 ms | .917 ms | .474 ms (8) |

A full 256-example invented-manual optimizer step, including embeddings, trainable routing, reader, live recurrence, decoder and AdamW, took **9.376 / 6.033 / 6.047 ms** for dense / matrix / reverse. That's **1.55×**, not 37×. For many future queries on one hit set, materializing a pooled state once is a sensible alternative. Nonlinear E97 still needs explicit per-hit state replay. See [source and timing contract](TRITON_WINDOW.md) and [aggregate](receipts/triton_meter.json). No claim these are hardware-independent optimal tile shapes.

19 CPU checks and 11 leased-device tests passed, including FP64-reference VJPs, missing/empty/tail spans, E97's separate nonlinear recurrence, and dense-fixture logits, gradients and optimizer steps. No natural text or results opened. Detached owner, independent advancing-work/heartbeat witness, rc=0 positive release, ≤29 card-s. Next meter uses the actual token-LM geometry, mixed precision and full CUDA-graph optimizer replay, including comparison against graphed dense replay.
