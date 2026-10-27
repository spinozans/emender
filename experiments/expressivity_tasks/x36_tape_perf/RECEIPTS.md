# X36 part 1 preregistration

Registered before any fix measurement, 2026-10-27.

At B64/T512/H4/S2944, complete training step tape/base ratio <=3;
at T4096 ratio <=2. Lane launches <=10 per step and zero host syncs in
steady training. Measure S736,2944,11776; test attention-shaped O(S) read
cost, distinguishing sequential token dependence from launch overhead. No
claim that a causal recurrent core can eliminate its device token dependency.

Parity precedes performance. X31's unchanged 109-test oracle suite must pass;
changes live only in X36. GPU broker visible set 0,1,5,6 excludes certified
2/3/4 and forbidden 7. Exact project Python only. No .pt output.

Decomposition precedes fixes. Synchronized wall boundaries isolate cell,
telemetry, replay and read_out within a real X34 LM AdamW training step.
Pure cell forward and journal increment use a matched no-grad ablation on
identical cell inputs (the increment is an estimate, not direct in-kernel
attribution). Everything else is the measured step remainder. Separate
profiler collection counts kernel launches and host synchronization APIs;
profiling overhead is excluded from wall results. Raw timings retained.
Historical reference: X34 receipts/smoke_summary.json (the requested
summary_console.txt is absent): 26.6774 vs 624.8954 ms, ratio 23.4242.
X31 summary_part8.json is forward-only at T16; not an LM training estimate.

Unmet gates remain failures. Persistent forward and replay fusion, S sweep,
T4096 and matched 2M-token validation require their own receipts.

Baseline table persisted in DECOMPOSITION.md before implementation.
Next fix hypothesis: replay dominates (B64 13256.6/13489.5 ms). Keep its
existing single-kernel token loop but bound fresh empty-origin read adjoints
to next_power_of_2(min(S,T)). No state import/continuation reduction. Logical
strides remain S and inactive returned adjoints remain unchanged. Expect >=2x
replay speedup at T512/S2944; bit-exact X31 outputs and gradients required.
Telemetry training branch omitted; eval telemetry remains identical.

Prefix attempt REJECTED before performance: 3 bit-exact gradient failures,
110 passes; retained log/XML/patch. Reduction-tree changes alter roundoff.
Next hypothesis: keep full-capacity reductions unchanged; mask gradient
payload reads/writes to restored-live slots, leaving dead adjoints untouched.
This removes no-op dead-row memory traffic without changing reduction order.
Same >=2x replay target; exact finite gradient parity required before timing.

Live-mask candidate passes 114 tests: 109 unchanged X31 oracle cases routed
through X36, four exact fresh-state output/gradient/routing cases, and one
exact train/eval telemetry case. B64 replay 13256.6 -> 407.94 ms (~32.5x),
complete step 13489.5 -> 628.51 ms (~21.5x); B8 623.17 ->328.26 ms.
These do not establish the <=3x model/base gate. Separate matched model/base
capacity sweep and 512-step/2,097,152-token per-arm slice follow this record.
Original clipping/AdamW loop is retained for matched semantics; global zero
host-sync gate remains FAILED. T4096 and persistent forward remain NOT RUN.

## Part 2 final receipts

The registered thresholds are unchanged. `part2_final_parity.txt/.xml`: 123
passed, including independent B64/T512 and production-import byte comparisons.
`part2_shape_gates.json`: B64/T512 2.655x PASS; B64/T4096
3.311x FAIL. `part2_profile_b64.json` and
`part2_optimized_b64.json`: six custom phases but 100 inclusive scan/replay
activities FAIL; no steady stream sync or DtoH PASS (two boundary device syncs).
Capacity scaling FAIL. `part2_training_slice.json`: 2,097,152 tokens per arm,
all 512 historical loss samples checked, max error 9.53674316e-07 PASS.

Rejected tiled-adjoint, 16-warp and global-8-warp parity attempts are retained.
The zero-address journal publication failure was fixed with ordered emission;
dense-loss/address-clone OOM attempts and the large-S SoA regression remain
raw failed receipts. Final measurements use the ordered journal, restored
private addresses, bounded-prefix warp guard and large-S AoS fallback.
