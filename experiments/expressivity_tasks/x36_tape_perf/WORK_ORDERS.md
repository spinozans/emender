# Remaining X36 work after part 4

The overall target set remains FAILED. REPORT.md and receipts/summary.json
separate passed fresh-capacity gates from incomplete representation work and
failed long-context/inclusive-launch gates. Parts 1–3 are archived.

1. **Complete arbitrary-slot occupancy-proportional backing.** Fresh T512 now
   bounds principal resident row adjoints at 512 lanes for every measured S
   (live=256), with exact logical-S reductions. Both capacity doublings pass
   <=1.5x step and replay. S8192->8224 is 1.022x step / 1.092x replay; compiler
   spills 0->2, shared memory fixed at 2048 bytes. This closes the measured
   spill cliff, not every S-dependent operation. Unknown sparse imports,
   long-lived holes, native owners/cold rows and required native gradients
   retain dense backing. Reuse X31 part-9/X35 pool scoping: stable logical
   ordering, cold-row storage and ownership/rollback support for every journal
   address. The identity-prefix bound is not the complete arbitrary-slot pool.
   Full logical-S reduction operands remain, and exactness is layout-sensitive.
2. **B64/T4096 <=2x.** Final seven-phase source measures 6121.207 / 2222.592 =
   2.754x FAIL. Device-only allocation reach dispatch selects 2048 resident
   rows, but the dense fallback remains in the compiled program (255 registers,
   26 compiler spills). Reverse takes 2757.794 GPU ms; scan takes 1047.658 ms.
   Measured occupied pairs grow 16,777,216->1,073,741,824 (64x for 8x tokens),
   with zero frees and final live/head=2048. Logical 0.825/2.199 TB accesses are
   not DRAM counters or a hardware/GEMM lower-bound proof. The matched <=2 wall
   target needs 1676.024 ms improvement. Reduce repeated streaming and exact
   reduction work without changing the oracle's arithmetic or waiving the gate.
   Split fallback, eight-warps compact, and eight-warps/128-register variants
   passed affected parity but still failed <=2 at 2.726x, 2.700x and 2.763x.
   They were reverted to preserve the requested seven custom phases.
3. **Inclusive launches <=10.** T512 has 42+17=59 activities; long has 44+17=61.
   Both retain seven custom kernels. Narrow <=10 and seven-phase shape pass;
   inclusive <=10 FAILS. Remaining 52/54 native activities include initialization,
   fills, input/output copies/transposes and mapping transfers. Fuse them inside
   the existing phases without moving work outside the registered ranges.
   Global reduction scratch is already eliminated and is not this bottleneck.
4. **Retain proof.** Require all 141 tests (138 existing plus dense S736/T512
   and aligned S8192/S8224 T512 byte fixtures). Preserve B64/large-S bytes,
   dense/random/frozen-key T4096, expanded/strided adjoints, imported/retained/
   consuming state, signed zeros, zeroed-log ownership and full private address
   restoration. Preserve required native cold-row gradients. Both checkpoint-free
   512-step/2,097,152-token arms retain maximum historical loss error
   9.5367431640625e-7. Keep zero steady host syncs/DtoH with pinned ingestion.
   Do not performance-time exactness-failed variants. Source patches, failure
   logs and parity-cleared performance failures are indexed in
   receipts/part4_attempts.json. Reproduce the ledger with the exact project
   interpreter on receipts/part4_analysis.py.

All GPU work remains through run_leased.sh and the exact project interpreter,
with physical whitelist 0/1/5/6. No card 7, certified 2/3/4, .pt checkpoints,
WG service/dispatcher/workers, delegation or completion/review machinery.
