# R5 register-tile composition obligation

R4's COMPOSITION_PROOF.md remains the state/token induction authority. R5 changes
READ only. The unchanged normalization produces p_i=round32(weight_i*strength_i)
once; each row product is round32(p_i*V_i), with fusion disabled.

The new tile loads32 adjacent physical rows and128 columns. The products reshape
to [16,2,128], then permute to [2,128,16]. Thus index [parity,column,j] is exactly
row 32*tile+2*j+parity of the page. Layout conversion and four splits move bits
only. Repeated splitting removes the low bit first; named n4_0..n4_15 reorder the
leaves back into increasing j. Explicit conversions normalize each leaf layout.
No reduction or arithmetic occurs in the split/reshape operations.

For page0/tile0, leaf0 initializes the two parity accumulators by assignment,
without adding a zero. Leaves1..15 then append one rounded addition each.
Every later tile/page starts with the exact preceding FP32 accumulators and
appends leaves0..15. Pages store/load FP32 without arithmetic. No additional
padding pairs are appended: tile count=min(PAGE,BS)/32 with these power-of-two
BS>=32; all selected long fixtures have BS>=1024 and PAGE1024. Row masking beyond
S loads the same positive-zero value as the reference. Only the final page adds
the parity rows using unchanged fixed_sum. Invalid tokens emit zeros as before.

Induction on j, tiles and pages yields the same rounded addition sequence for
each column. Per-token control and persistent physical state are unchanged,
including pre-read CORRECT/expiry/array writes and post-read POP/MERGE; no earlier
KV snapshot can replace them. The single CUDA stream orders all these stages.
This is a composition argument, not a formal proof of compiler correctness.

Independent full-matrix primitives through BS4096 exercise random values,
cancellation, signed zeros and NaN byte payloads. Ten monolithic native/state/
raw-trace comparisons anchor the shared semantics through T1024. T2048 uses
PAGE256 decomposition plus these anchors; it never claims repaired unchanged
monolithic LLVM compilation. Masked tails and distinct-document fresh-state
reset/replay/gradients test native boundary composition. Maximum-live operation-
rich long cases and full trained-model/resume/throughput gates remain additional
promotion obligations, not implied by this local proof.
