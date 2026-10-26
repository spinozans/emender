# X31 preregistration

Registered 2026-10-26 before executing X31 tests or timing measurements.

Acceptance gates, in order:

1. Exact integral routing/death traces; outputs, keys, values, strength and
   first-order gradients against X30 naive_reference at atol=rtol=1e-5.
   Seeded randomized imported stores must include age ties, popped/dead rows,
   threshold-boundary deaths, all routing cases and continuation.
2. MQAR validation >=0.99 by update 5000; slots=736, heads=4, batch=64,
   AdamW lr=1e-4. No inherited X30 result counts as an X31 result.
3. Ladder a^k b^k, k<=128, 2500 updates, slots=736, heads=4.
   Success signal balanced accuracy >0.9; report both class recalls regardless.
4. Full forward latency at batch=64, heads=4, slots in {736,11776},
   against X30 and X28E eager. Target >=10x X30 at 11776.
   Heads=16 / heads=1 latency ratio <=2 at equal batch and slot budget.
5. Sparse checkpoint overhead <=5% of forward at slots=11776;
   demonstrate events scaling independently of S. Report base state separately.
6. Append-phase parallel prototype faster than sequential formulation;
   no numerical speedup threshold beyond >1x, include launch/synchronization.

Timing protocol: GPU lease restricted to physical cards 0..6. Warm up 5,
measure 20 repetitions with CUDA events and synchronize before reading results;
report medians, dimensions, software, physical GPU and dtype. Training and
performance claims are forbidden until the complete parity gate passes.
CPU interpreter checks are correctness diagnostics, never GPU performance.

## Results

Pending. Unrun gates must remain explicitly unrun, never inferred from v1.

### Recorded validation (2026-10-26)

The initial small-store GPU suite passed 20 tests. The final expanded suite
passed **31 tests in 3.38 s**, on lambda01 physical GPU 1, acquired/released
through the broker. Full output: `validation.txt`. This elapsed pytest time
includes compilation and is **not a throughput measurement**.

Forward oracle comparisons: seeds 0..4, S={1,7,16,736,11776}, batch=2,
heads=2, T=9, K=4, V=5; plus exact-death, age-tie and pressure-window fixtures.
Outputs/state satisfy atol=rtol=1e-5 and discrete traces match exactly in these
cases. First-order gradients are unsupported, so gate 1 remains **UNMET**.

| Preregistered target | Measured / status |
|---|---|
| Complete parity including gradients | Forward checks pass; gradients NOT IMPLEMENTED |
| MQAR >=0.99 by update 5000 | NOT RUN |
| Ladder balanced accuracy >0.9 by update 2500 | NOT RUN; class recalls unavailable |
| >=10x X30 at S=11776, batch64/H4 | NOT MEASURED |
| H16/H1 latency <=2x | NOT MEASURED |
| Checkpoint overhead <=5% at S=11776 | NOT MEASURED; sparse journal not integrated |
| Append-phase speedup >1x | NOT MEASURED |

Sparse journal storage fixture: 3 IDs (int64) + 3x3 old fields (fp32) =
60 bytes, unchanged between S=7 and S=11776. This verifies the primitive's
storage shape only; it does not establish sparse training checkpoint costs.
All performance/training measurements remain withheld under the parity-first
rule. No v1 training result is substituted for X31 evidence.

The first expanded 31-test run took 20.00 s including new large-budget kernel
compilation; the final rerun after input-validation checks took 3.38 s with
cached kernels. Neither number is a performance receipt.
