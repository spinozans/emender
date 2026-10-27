# X36 part 2 — persistent forward and B64 target delivered; scale gates remain failed

**Partial engineering delivery.** Matched B64/T512 is **2.655x**
(435.582 / 164.045 ms), passing <=3x.
The recurrent forward drops from **1536 token launches to one**. The full
custom tape path has **6 phase kernels**. The stricter registered count
including setup/fills/transposes/copies is **100 activities: FAIL**.
B64/T4096 is **3.311x: FAIL** against <=2x. Capacity scaling also
fails. These failures are not waived by the B64 improvement.

The final unchanged-oracle suite plus strengthened exact-byte and loop tests
is **123 passed**. Both checkpoint-free 2,097,152-token arms match all 512
X34 smoke loss samples with max absolute error **9.53674316e-07** (<=1e-5).
This is unchanged-semantics evidence, not a learning-quality improvement.

Part 1 is archived in [REPORT_PART1.md](REPORT_PART1.md) and
[summary_part1.json](receipts/summary_part1.json). Current machine-readable
gates are in [summary.json](receipts/summary.json).

## Implementation and correctness

`forward.py` and `kernels.py` integrate the original 32-slot tiled read,
8-value reduction, lazy decay, birth index, heap/bitmap mutation and typed
journal inside one store-owner device token loop. The full logical-S reverse
reduction remains intact. Historical X31/X34/X35 sources are unchanged.

Replay payloads and adjoints use transposed backing at smaller capacities,
with typed address rebasing and exact return-layout restoration. Eight warps
are restricted to the tested S2944/DK64/DV128 empty-origin prefix with
host-known cumulative reach <=512. Unknown imports and longer continuations
retain the oracle warp count. At S>=4096 the oracle AoS backing is retained:
the transposed prototype regressed S11776 to 9576.5 ms/step; the final fallback
returns it to 4010.1 ms. It still fails the scaling gate.

Mixed-shape runs exposed published journal entries whose addresses were zero.
The persistent core had reused the original count-before-payload publication
without sufficient cross-warp ordering. X36 now fences group emission before
publishing its count and fences mutation visibility. Local mutation helpers
and materialization use this publication rule. Failed ownership assertions
are retained in `part2_journal_race_failed.*` and
`part2_payload_profile_failed.txt`; final mixed-shape and 512-step runs finish.
Exact fixtures zero the reserved address backing, audit every active address,
and verify the entire private address log is restored after backward.

Journal capacity is bounded by mutation counts, with the bound documented in
`journal.py`; it removes the obsolete T*reachable dense-write reservation.
Backward rebases its private address log in place and restores it on the same
stream in a `finally` block. This avoids the 4.5 GiB address clone that caused
the initial T4096 OOM. Sequential retained backward, consuming continuation,
rollback, mutation rejection and imported-state checks remain green. Unused
native-output gradients are initialized only when replay needs them.

An eight-column adjoint tile failed byte parity (max error 2.3842e-7), and a
16-warp S2944 reverse failed the new B64/T512 byte fixture (max error
3.4332e-5). Global eight-warps also failed a production-width imported
continuation. They were reverted or restricted before final measurements;
logs/XML and rejected patches remain in receipts. The prefix-reduction attempt
rejected in part 1 remains rejected. The original reference calls now bypass
the pytest adapter so they cannot share X36's forward accidentally.

## Zero-sync gate

The native finite clipping arithmetic is retained with
`error_if_nonfinite=False`; a device flag accumulates finiteness and is checked
at loop boundaries. Tests verify byte-exact clipping/AdamW updates and that
NaN/Inf norms fail the boundary check. Training telemetry remains eval-only.

The separate profiler covers nonblocking pinned-batch ingestion and the whole
steady training step. It reports **zero stream/event synchronizations and
zero DtoH activities**. Its two `cudaDeviceSynchronize` events are the explicit
measurement boundary and profiler exit. The 174 CPU optimizer scalar events
are not GPU synchronizations. The actual last clipping DtoH/stream sync is gone.

Profiler correlation links every activity to `x36::scan_forward` or
`x36::replay_backward`: **42 + 58 = 100** activities, already excluding
cell projections. Custom kernels are initialize-free (1), scan (1),
materialize (1), rebase/restore (2), reverse (1). Thus the persistent-phase
subtarget passes, while the inclusive <=10 launch gate fails honestly.
Raw trace: `receipts/part2_profile_b64.json`.

## Matched cost and decomposition

Two warmups and five raw wall samples per point; profiler runs are separate.
Stream-event component samples add no intermediate host synchronization.
The B64/T512 end-to-end ratio uses uninstrumented matched steps.

| Instrumented B64/T512 component, ms | Part 1 | Part 2 |
|---|---:|---:|
| Pure cell forward ablation | 69.985 | 44.204 |
| Journal increment estimate | 0.416 | 5.080 |
| Replay backward | 407.937 | 244.650 |
| Training telemetry | 0.000 | 0.000 |
| Read-out forward | 0.333 | 0.342 |
| Everything else, including read-out backward and optimizer | 149.842 | 149.505 |
| Total instrumented wall | 628.513 | 443.781 |

Replay improves **1.667x** from part 1;
complete instrumented wall improves **1.416x**.
The new breakdown further measures read-out backward at
0.949 ms and optimizer at 1.852 ms; the
remaining transformer/projection/head/host-gap row is
146.704 ms.
The journal increment is a matched no-grad ablation estimate, not a direct
in-kernel clock. CPU launch API time overlaps GPU work and is never added to
the wall table. A separately calibrated launch-only cost is still absent.

| T512 capacity S | Complete tape ms | Capture ms | Replay ms | Step ms/token |
|---|---:|---:|---:|---:|
| 736 | 286.660 | 49.346 | 89.328 | 0.5599 |
| 2944 | 435.582 | 49.484 | 236.646 | 0.8507 |
| 11776 | 4010.135 | 54.107 | 3887.088 | 7.8323 |

Four times S from 2944 to 11776 costs **9.206x** complete wall and
**16.426x** replay. This is a failed linear-S performance gate.
The remaining scale cost is concentrated in replay, with a strong layout/
resource cliff. It is not an established irreducible forward-GEMM FLOP floor.

| T/S | Replay backing | Warps | Registers/thread | Compiler n_spills |
|---|---|---:|---:|---:|
| 512/736 | transposed | 4 | 228 | 0 |
| 512/2944 | transposed | 8 | 255 | 0 |
| 512/11776 | oracle_aos | 16 | 64 | 626 |
| 1024/2944 | transposed | 4 | 255 | 126 |
| 4096/2944 | transposed | 4 | 255 | 204 |

| T, S2944 | Base ms | Tape ms | Ratio | Capture ms/token | Replay ms/token | Final live/head |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 39.872 | 113.145 | 2.838x | 0.0790 | 0.5078 | 64 |
| 512 | 164.045 | 435.582 | 2.655x | 0.0966 | 0.4622 | 256 |
| 1024 | 454.818 | 1054.031 | 2.317x | 0.1210 | 0.4909 | 512 |
| 4096 | 2337.486 | 7739.483 | 3.311x | 0.3134 | 1.0083 | 2048 |

T>=1024 uses the same chunked-1024 vocabulary projection/loss in both arms.
The original dense T1024 attempts OOMed and are retained in
`part2_pre_memory_fix_gates.*`. The bounded projection is tolerance-validated;
T512 and the 2M curve proof keep the original dense projection. All final
registered shapes finish with five raw samples. T4096 tape peak allocation is
44.991 GiB on the leased 48 GB card.
Occupancy and the exactness-preserving warp specialization change across the
T sweep. The table does not establish flat per-token time, nor removal of the
causal device token dependency. Launch count is independent of T; measured
scale efficiency remains open. Component samples in the sweep are a separate
five-step phase and should not be treated as a closed additive wall table.

## 2M-token unchanged-semantics proof

| Arm | Tokens | Median ms/step | Max loss error vs X34 smoke |
|---|---:|---:|---:|
| base_lm | 2,097,152 | 23.930 | 9.53674316e-07 |
| tape_lm | 2,097,152 | 160.926 | 9.53674316e-07 |

Both arms complete 512 steps with no checkpoints. End-to-end B8 tape/base is
6.725x, versus part 1's 14.296x; this is distinct from the
registered B64 gate. Cross-run loss curves are not bit-identical; even the
unchanged base has the same sub-micro difference. In-process cell output,
gradient, routing, zero-sign, imported continuation and full B64/T512 byte
comparisons supply the separate exactness evidence.

## Operational evidence and remaining work

All GPU work used `run_leased.sh`, physical card 0, exact project Python and
whitelist 0/1/5/6. Certified 2/3/4 and forbidden 7 were excluded. No .pt,
WG service, delegation, Frontier/Slurm or resilient-training work was used;
Partition/QOS and resilient requirement IDs are inapplicable. AGENTS.md and
CLAUDE.md remain synchronized. Unrelated workspace output was preserved.
No callable `contact_supervisor`/Intercom tool was present in the catalog;
that limitation is recorded in `part2_coordination.txt`.

[WORK_ORDERS.md](WORK_ORDERS.md) retains the inclusive launch, long-context,
capacity/layout and complete-attribution work. No scientific LM pilot is
resumed by this engineering slice. Final source hashes are in
`receipts/part2_sources.json`; raw failures remain failures.
