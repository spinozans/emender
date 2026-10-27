# Remaining X36 work after part 3

The registered target set remains FAILED. Final evidence is in REPORT.md and
receipts/summary.json; completed measurements are indexed in
receipts/part3_measurements.json. Part 2 is archived.

1. **Remove the padded reverse register cliff with exact reduction trees.**
   At B64/T512/live256, aligned S8192->8224 grows capacity 0.391% but doubles
   BS8192->16384, costs 1.551x step / 1.755x replay and increases compiler
   n_spills166->388. Shared memory stays 8 KiB. Per-double replay growth still
   fails at both registered capacity doublings. Final SoA/128-register policy
   improves S11776 step to 1208.622 ms from fresh 4078.613 ms, but the gate
   remains failed. Replace padded register residency with an explicitly exact
   tiled layout/reduction schedule. Naive large-S staged temporaries regressed;
   the 32-warp SoA fresh-prefix variant failed bytes by 1.907e-5.
2. **B64/T4096 <=2x.** Final matched ratio is 2.883x FAIL
   (6170.573 / 2140.605 ms). Verified final live/head=2048 and 1,073,741,824
   live-slot/token pairs demand 0.825 TB forward / 2.199 TB replay logical
   payload accesses per step. Reduce repeated passes or improve reuse while
   preserving oracle reductions. Logical volumes are not DRAM counters or a
   hardware lower-bound proof. Constant-live64/four-warp control has
   scan/token 1.071x and adjoint/token 0.595x across 8x T. Keep this distinct
   from the growing-live production sweep. No nonsequential-T claim.
   Extended eight-warps failed dense T4096 bytes and remains rejected.
3. **Inclusive launches <=10.** Replay setup is fused; unused key adjoints
   are removed. Narrow custom count 7 PASS; inclusive 59(T512)/61(T4096)
   FAIL. Remaining 52-54 native activities are chiefly forward initialization,
   input/output materialization and mapping transfers. Fuse/remove them
   without hiding work outside the registered scan/replay ranges.
4. **Preserve proof.** Require the 138-test exact suite, including frozen-key,
   imported/retained/consuming state, dense long context, B64/large-S bytes,
   zeroed typed-log ownership and address restoration. Required key gradients
   remain enabled. Keep zero steady host syncs with pinned ingestion and the
   two 512-step / 2,097,152-token historical-curve checks (max 9.536743e-7).

All GPU execution remains through run_leased.sh with project Python and
physical cards 0/1/5/6. No scientific LM pilot is resumed by this engineering
slice. No WG, dispatcher, worker, delegation or completion/review machinery.
