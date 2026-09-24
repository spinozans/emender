# E97 throughput qualification standard (operator ruling 2026-09-24)

## Context
The E3-gap throughput probes (`e97-throughput-fix-v1/probe-decision-table.md`)
measured armA control bit-identical to the sfparam arm1 reference (patch
neutrality), armB `--projection-chunk-size 2048` at 1.13x (6,885 tok/s) with a
max per-update loss delta of 1.81e-4, and armC (CUDA graph capture)
deterministically unqualified on the autocast `cache_enabled=True` interaction.

## Ruling
Training volume is the binding constraint on the post-training program
(operator, 2026-09-24). Accordingly the operator adopts a two-class loss-match
standard:

1. **Semantic-class changes** (anything altering what is computed: loss
   chunking, masking, recurrence order, boundary resets) retain the strict
   5-6 digit loss-match bar.
2. **GEMM-tiling-class changes** (chunk size, CUDA graph capture: identical
   computation in exact arithmetic, different bf16 accumulation order) are
   qualified on: exact-arithmetic equivalence + update-1 pure-tiling signature
   + zero drift across updates + final-loss agreement within 1e-5.

armB qualifies under the tiling-class standard (update-1 signature 6.0e-5, no
drift over 32 updates, final losses within 3.1e-6). Qualified config for
segments 3+ of E3 and all future arcs pending the armD stacked probe:
chunk-2048 first; graph capture only if armD passes the tiling-class standard.

The scan-kernel fusion program (serial-walk batching across packs/layers) is
authorized as the beyond-flag lever, gated on its own numerics qualification.
