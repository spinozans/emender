# X40 part 12 — union-pooled read adjoints

**Performance gate: FAILED.** Implementation and requested measurement work is complete.

Fresh initialization/journal gate: passed, 80 B32 fixtures. See [support receipt](receipts/part12/support_gate.json) and [source audit](receipts/part12/source_audit.json).

The ever-touched union is a conservative support envelope. Popped and threshold-decayed rows stay in it. Logical-S reduction operands and cold native-gradient backing remain intact.

The candidate extends fresh-origin pooling through T2048 behind `X40_LONG_POOL=1` plus `X40_FRESH_POOL=1`. Scratch uses packed indices; the prefix-width dispatch is disabled for pooled replay. Production BF16 arithmetic is unchanged.

All 40 pooled native byte cases passed, as did eight split-reverse and eight combined core/read cases. The paired full 1.3B TinyStories slice ran 2,097,152 tokens per arm at T2048: every update loss, gradient norm and full gradient hash matched; initial/final parameter and input-data hashes matched. See [curve receipts](receipts/part12/curve_candidate_T2048.json).

Measured support (actual full-stack diagnostic, when available):

| T | S/head | Mean touched | Union/S | Reserved rows |
|---:|---:|---:|---:|---:|
| 512 | 64 | 64.00 | 100.000% | 64.00 |
| 512 | 256 | 256.00 | 100.000% | 256.00 |
| 512 | 736 | 256.00 | 34.783% | 256.00 |
| 512 | 1472 | 256.00 | 17.391% | 256.00 |
| 2048 | 64 | 64.00 | 100.000% | 64.00 |
| 2048 | 256 | 256.00 | 100.000% | 256.00 |
| 2048 | 736 | 736.00 | 100.000% | 736.00 |
| 2048 | 1472 | 1024.00 | 69.565% | 1024.00 |

Weighted touched/S over all four heads: T512: 32.911%, T2048: 82.278%. These are workload-specific journal unions, not current live occupancy. Full-S reductions remain, so their reciprocals are not measured adjoint speedups. T512 already used the fresh union before part 12.

Matched pure/transformer elapsed-time ratios:

Envelope: B64 effective/B32 physical, checkpointing for both arms, packed journals for pure only, no offload. Dense blocks use BF16; pure native stores and bounded vocabulary projection remain FP32, while the control uses fused BF16 vocabulary loss. Synthetic nonpadding inputs share independent seed 3402, matching the part10 protocol.

- T512 profile: 2.5335x (passed); physical cards ['1', '1'].
- T512 sustained: 2.2085x (passed); physical cards ['5', '5'].
- T2048 profile: 3.5544x (passed); physical cards ['1', '1'].
- T2048 sustained: 3.1564x (passed); physical cards ['5', '5'].

Production fused measurements (seconds):

| T | Full update | Sequential core | Separate reads | Fused reverse |
|---:|---:|---:|---:|---:|
| 512 | 11.791857 | 2.233044 | 0.234797 | 4.809204 |
| 2048 | 69.192618 | 15.360480 | 1.843583 | 33.860060 |

The exact part10 shared checkpoint/B32 sustained receipts give 2.2128x at T512 and 3.2073x at T2048, rather than the task-supplied 1.6x prior. [Historical source hashes](receipts/part12/matched_history.json) retain the comparable evidence; the new measurements above remain separate.

Exclusive synchronized split-reverse regions (seconds; extra launch/spill costs are explicit):

| T | Undo | Strength/store events | Score/softmax | Read adjoints | Wrapper overhead |
|---:|---:|---:|---:|---:|---:|
| 512 | 8.527274 | 8.775189 | 8.310946 | 11.637808 | 2.477080 |
| 2048 | 33.697999 | 40.502053 | 34.947922 | 57.653939 | 9.523138 |

These are additive wall regions of the separately gated split implementation. The unchanged fused kernel retains its own authoritative wall measurement; split fractions are not applied to that wall.

The initial diagnostic split omitted the original compact-replay assertion. Restoring its byte occupancy operand restored parity; a compiler layout effect is inferred. Three rejected attempts are retained, and restoring the original assertion passed all eight B32 shapes. Production union replay was not changed by these scoped diagnostic patches.

Sequential core/read launch experiment:

- T512: core 384; batched read 384; reverse 192. The combined core/read path leaves 0 separate read calls; wall ratio 1.1112x, profile loss equal=True. Compiler resources and grids are in the profile receipts; they are not achieved-occupancy measurements.
- T2048: core 384; batched read 384; reverse 192. The combined core/read path leaves 0 separate read calls; wall ratio 1.0970x, profile loss equal=True. Compiler resources and grids are in the profile receipts; they are not achieved-occupancy measurements.

The combined core/read candidate is not promoted: it is slower at both T values. This losing experiment does not establish a hardware or algorithmic floor.

Compiler launch resources (reported `n_regs`/`n_spills`, not measured achieved occupancy):

| T | Region | S | Registers | n_spills |
|---:|---|---:|---:|---:|
| 512 | sequential_forward_core | 64 | 168 | 32 |
| 512 | sequential_forward_core | 256 | 168 | 28 |
| 512 | sequential_forward_core | 736 | 168 | 20 |
| 512 | sequential_forward_core | 1472 | 168 | 20 |
| 512 | reverse_fused_read_math_and_adjoints | 1472 | 147 | 0 |
| 512 | reverse_fused_read_math_and_adjoints | 736 | 142 | 0 |
| 512 | reverse_fused_read_math_and_adjoints | 256 | 107 | 0 |
| 512 | reverse_fused_read_math_and_adjoints | 64 | 93 | 0 |
| 2048 | sequential_forward_core | 64 | 168 | 32 |
| 2048 | sequential_forward_core | 256 | 168 | 28 |
| 2048 | sequential_forward_core | 736 | 168 | 26 |
| 2048 | sequential_forward_core | 1472 | 255 | 24 |
| 2048 | reverse_fused_read_math_and_adjoints | 1472 | 202 | 0 |
| 2048 | reverse_fused_read_math_and_adjoints | 736 | 220 | 0 |
| 2048 | reverse_fused_read_math_and_adjoints | 256 | 138 | 0 |
| 2048 | reverse_fused_read_math_and_adjoints | 64 | 111 | 0 |

These B32 store-owner kernels launch 32 CTAs on a 142-SM RTX 6000 Ada. A single such launch can place CTAs on at most 32 SMs; this is a grid bound, not an achieved-occupancy measurement. The core reports spills while these reverse shapes report zero spills.

Pilot: 55,380.14 tok/s; 100k gate **FAILED**. Three warmups, twenty full B128 updates. The unchanged T512 pilot curve is inherited explicitly; this is not a new pilot quality run.

Residual risks and open gates:

- The pilot 100k gate remains open/failed.
- Synchronized split-kernel times include spill/reload and extra launches; they cannot be presented as exclusive intervals inside the production fused binary.

Single attended session; approved Python and run_leased.sh only, physical cards 1 and 5. PID 634691 was alive before and after work and was not disturbed. Supervisor contact tool was not exposed. No WG service, subagents, Frontier/Slurm, resilient DiLoCo work, or .pt artifacts. Production BF16 and unrelated workspace changes are untouched.

No resilient DiLoCo requirement IDs apply. AGENTS.md and CLAUDE.md are unchanged. Unrelated working-tree changes are excluded from this work.
