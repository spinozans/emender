# X36 baseline decomposition — before fixes

Times ms/step; medians of five samples after two warmups. Actual X34 LM,
fp32 AdamW, T512/H4/S2944. Event ranges do not add host sync boundaries.

| Component | B8 | B64 |
|---|---:|---:|
| pure_cell_forward_ablation | 69.307 | 68.496 |
| journal_capture_increment_estimate | 10.204 | 2.910 |
| replay_backward | 516.547 | 13256.643 |
| trace_metrics | 2.768 | 17.276 |
| read_out_forward | 0.131 | 0.409 |
| everything_else_including_projection_backward_optimizer_and_host_launch_gaps | 24.216 | 143.774 |
| **Total measured wall** | **623.173** | **13489.507** |

Forward ablation includes its CPU launch gaps. Journal increment is estimated,
not a direct kernel attribution. Read_out backward is in everything else.
Launch overhead is included in these rows, not an additional additive cost.
A separate calibrated launch-only cost has NOT been established.

Lane token kernels: 1536 at B8 and 1536 at B64 (3/token).
Replay already uses one reverse kernel per chunk, plus rebase/copies.

This is partial attribution: no claim of direct in-kernel journal timing or
independently isolated launch overhead. Those remain named work orders.

## Part 3 measured scale floor

The current decomposition is in [REPORT.md](REPORT.md) and
[part3_decomposition.json](receipts/part3_decomposition.json). The aligned
8192->8224 boundary isolates the BS8192->16384 register/spill cliff at fixed
live=256; shared memory stays 8 KiB. T4096 logical payload traffic is measured
from a separate routing trace, with a constant-live/four-warp control. CPU
allocator self time and journal ablation estimates are not additive GPU rows.
Capacity, long-context ratio and inclusive-launch gates remain FAILED.
