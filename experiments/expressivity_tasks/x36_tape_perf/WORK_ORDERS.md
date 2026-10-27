# Remaining X36 work after part 2

Part 2 delivers persistent forward, zero clipping sync, a passing matched B64
ratio (2.655x), 123 tests and both 2M-token unchanged-loss curves.
The registered target set remains **FAILED**.

1. **Inclusive launches <=10.** The recurrent forward is one launch and the
   custom phase count is six, but scan/replay still contain 100 device
   activities including setup/fills/transposes/copies. Fuse or remove the
   native setup and tensor materializations; do not relabel six custom kernels
   as the inclusive gate passing. Projections are already excluded here.
2. **T4096 <=2x.** Matched B64 ratio is 3.311x. The bounded vocabulary
   projection and in-place/restored private address log now fit memory. Reduce
   the remaining persistent replay and forward cost without changing the
   oracle reduction tree. Global 8-warps/imports and 16-warps/S2944 failed byte
   comparisons; only the independently tested cumulative-prefix <=512
   specialization may use eight warps.
3. **Capacity performance.** S2944->11776 (4x capacity) costs 9.206x wall
   and 16.426x replay. The AoS large-capacity fallback avoids the 9.58 s
   transposed regression but still has compiler n_spills=626. Preserve exact
   full-S reductions while fixing the layout/resource cliff. The remaining
   cost is implementation debt; no irreducible read-GEMM FLOP floor is proved.
4. **Asymptotic and attribution evidence.** Complete a fixed-occupancy T/S
   study that separates reachability, occupancy and warp specialization. The
   current T128/512/1024/4096 sweep is recorded, with bounded loss at T>=1024,
   but does not certify flat per-token time. Independently calibrate host
   launch overhead and capture-only cost; retain the distinction between
   overlapping CPU API time, GPU busy time and additive wall ranges.
5. **Preserve the journal and semantics gates.** Keep publication ordered
   across warps, audit zero-initialized logs for holes, restore private
   addresses after backward, and retain all mutation/rollback/retained-state
   tests plus production-width imports and B64/T512 byte checks. Require the
   matched 512-step/2,097,152-token-per-arm curve check before certification.

No additional scientific LM pilot/report work is authorized by this slice.
