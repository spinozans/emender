# X40 part 24: ownership analysis before repair

FAIL / incomplete prerequisite. No kernel edit has been made. This receipt
contains a complete index of the 175 mapped endpoint occurrences (73 distinct
PCs), conditional thread/address ownership sets, and four concrete failures of
unique scratch writer ownership. It does **not** contain a complete dynamic
lifetime proof or a mechanically justified source repair. The requested
ownership-only milestone is therefore not claimed complete.

Read part23/diagnosis.md at b44020f2 first. Its isolated cubin, rather than its
30 displayed clusters, is the scope here. `ownership_audit.py` checks all four
captured artifact hashes, the retained SASS hash, every mapped instruction,
175 occurrences and 73 distinct PCs. `ownership.json` preserves every original
group/occurrence as well as the deduplicated sites. `endpoint_sites.md` gives
the per-site address table. The five certified-run groups remain X31 evidence;
they cannot be assigned ownership using this X40 cubin.

## Proven scope and memory semantics

`scan_kernel` uses `n = program_id(0)`, with one program owning one store.
The captured TTGIR has four warps per CTA and 32 threads per warp. Below,
`t = tid.x`, `w = t >> 5`, and `l = t & 31`, with `0 <= t < 128`.
Shared addresses are CTA-relative bytes. Two programs using offset zero do
not thereby alias. Conversely, one program owning a store does **not** mean
one lane owns its compiler-generated shared scratch.

Ordinary shared loads/stores need inter-thread synchronization when they
conflict; identical intended values do not create unique writer ownership.
A CTA barrier can order a producer phase before a consumer phase, but cannot
serialize several writers in the same phase. A subsequent reuse also needs
the previous readers to finish. These are requirements of the
[PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)
and [CTA barrier semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar).
`LDSM.16.M88` is a collective layout read: its address operand must not be
treated as an ordinary four-byte per-thread read. The ledger intentionally
does not pretend to enumerate its footprint; see
[ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix).

## Ownership table and required lifetimes

Every range in this table is **within a dynamic invocation** of that
operation: token, tile/dimension tile, branch and loop iteration matter.
The PC table indexes all occurrences, including repeated uses of one PC in
different race clusters. The lifetimes below are algorithmic requirements,
not claims that the current control-flow graph establishes them.

| Site family | Writers and readers in one CTA | Required lifetime/order | Repair implication |
| --- | --- | --- | --- |
| Scalar sum/max/min reductions, `[R154.X4]`, `[R15.X4]`, `[RZ]` | First stage: lane 0 of warp w writes word 4w. Collector threads 0..3 read words 0,4,8,12 and use shuffles. Thread 0 writes the final word at 0; all threads read it. | Partial writes precede collector reads; collector/shuffle precedes final write; final write precedes broadcast reads; every broadcast read precedes reuse by a different operation. | Genuinely cross-thread within an operation. Privatizing the scalar per lane would discard partials unless the same reduction tree is explicitly preserved. No blanket barrier repair follows from this table. |
| Query broadcast, 0x1bc40/0x1c1c0, kernels.py:99 | Each thread writes four bytes at `16*(l&7)+((l&24)>>1)`. Warp id is absent. All four warps write the same 32 words. Readers load 16 bytes from `t&112` to expand query into key-row layout. | Each query coordinate must be defined before consumers read it; concurrent redundant writers must be eliminated or given distinct storage. Release only after all consumers finish. | Logical query replicas need privatization or a compiler layout with one writer per byte. A barrier between these concurrent writers and their readers cannot fix the WAW. |
| Strength/weight broadcast, 0x1c320/0x1c350/0x1c360, kernels.py:119 | Each thread writes 16 bytes at `t&112`. Sixteen threads write each vector. Readers load vectors from `(t&96)>>1` and that offset +64. | Broadcast strength/weight must remain defined through weighted payload arithmetic, before reduction scratch reuse. | Replicated logical inputs, with nonunique physical writers. Privatize replica storage or use a verified layout conversion; simply electing a writer is not yet proved equivalent for all coordinates. |
| Tile value reduction exchange, 0x1c740..0x1ca50, standard.py:291 | Threads write four partial words at `4w+64l+{0,16,32,48}`. These writes are injective. All threads read four words at `4t+{0,512,1024,1536}`. Threads satisfying `(t&3)==0` write reduced words back to those four banks. Consumers read `64l+{0,16,32,48}`. | Partial writes -> cross-warp reads -> reduced writes -> final reads -> next use. Different addresses within one stage do not prove separation from other stages or future tokens. | Genuinely cross-thread reduction. Preserve its exact tree. If its lifecycle cannot be expressed reliably by source-level Triton operations, externalize partials and split launches. |
| Tile output layout, 0x1caa0/0x1cb00, kernels.py:120 | Threads write 16 bytes at `16l` (four warp replicas). Matrix read address is `((128w) OR (16t)) & 496`. | Reduced tile values precede layout stores, which precede collective reads and global partial output. The same-phase four-writer overlap must be removed. | Privatize logical output replicas or establish unique producer layout before any phase split is accepted. |
| Read-pass denom broadcast, 0x1d780/0x1d7b0/0x1d7c0, kernels.py:149 | Thread t writes one word at `4t` (injective). Each thread reads at `4*(t&126)` and that address +256. | Denom conversion must finish before payload masks use it; readers must finish before scale overwrites the same storage. | Cross-thread layout conversion, not lane-private: e.g. t=1 reads t=0's word. Requires ordered exchange, even though writes are injective. |
| Read-pass scale broadcast, 0x1d860/0x1d890, kernels.py:150 | Same writer and reader address mapping as denom. The +256 read at 0x1d880 is also visible in the cubin, although it is not one of the mapped endpoint PCs. | Scale stores -> both scale reads -> arithmetic -> reuse. | Same cross-thread exchange requirement; endpoint-only enumeration must not omit the other consumer in a repair proof. |
| Read-pass value reduction, 0x1db70..0x1dd40, standard.py:291 | Lanes 0 and 1 of each warp write at `4w+64*(l&1)+{0,16,32,48}`; others skip. Threads 0..31 collect words at `4t`, shuffle and write the reduced word at `4t`. All threads read at `64*(t&1)+{0,16,32,48}`. | Partial writes -> collector reads/shuffles -> reduced writes -> broadcast reads -> layout output. | Genuine cross-thread reduction. Lane-private replacement needs the same arithmetic tree, not arbitrary recomputation. |
| Read-pass output layout, 0x1de90/0x1deb0, kernels.py:151 | Every thread writes 16 bytes at `16*(t&1)`. Sixty-four writers per vector. Collective read address is `16*(t&1)`. | Output arithmetic precedes stores; collective read must consume a uniquely defined vector before global output and reuse. | Replica privatization/verified conversion required. This cubin contains a concrete same-instruction WAW ownership violation, independent of any cross-operation reuse argument. |
| Residual, history requests and history caps reductions, 0x20ad0..0x240b0 | Same scalar reduction exchange pattern as first row, occurring in mutation/residual and pressure phases. | All final readers must finish before another reduction/token reuses offset 0. Scalar/global metadata ownership does not extend the lifetime of shared reduction bytes. | Preserve genuine cross-thread reduction; no inter-program synchronization requirement is established. |

## Four conditional nonunique-writer witnesses

These are address-set proofs, conditional on the listed lanes reaching the
same dynamic operation. They are stronger than inferring ownership from a
source line, and weaker than a full path-sensitive execution proof. The PTX
stores at these four sites are unpredicated. No equality or value neutrality
assumption is used to excuse their overlap.

| PC | Two writers | Overlapping shared bytes | Multiplicity |
| --- | --- | --- | --- |
| 0x1bc40 | t=0, t=32 | 0..3 | 4 writers per word |
| 0x1c320 | t=0, t=1 | 0..15 | 16 writers per vector |
| 0x1caa0 | t=0, t=32 | 0..15 | 4 writers per vector |
| 0x1de90 | t=0, t=2 | 0..15 | 64 writers per vector |

SASS setup provenance: tid is loaded at 0x178d0; lane at 0x1a090;
query addressing at 0x17e60, 0x17ed0, 0x18510 and 0x1a100;
strength/weight address at 0x1a460; output address at 0x1a340/0x1a5c0.
PTX spells the corresponding stores as `st.shared.b32 [%r48]`,
`st.shared.v4.b32 [%r49]`, `st.shared.v4.b32 [%r60]`, and
`st.shared.v4.b32 [%r70]`. The offline verifier enumerates all 128 thread ids.
Part23 also reports a self-conflict at 0x1de90. This analysis does not dismiss
that report because the intended vector values might be equal.

## Why the repair remains gated

The cubin already has barriers at, for example, 0x1d770 (before denom store),
0x1d790 (after it), 0x1d850 (before scale store), 0x1d870 (after it), and
0x1dea0 (before the collective output read). Their presence in linear text
does not prove convergence, matching dynamic barrier epochs or scratch
lifetime across the nested loops and assertion paths. It also does not
resolve the four same-phase duplicate-writer sets. The assertion failure in
the captured run remains a real terminal failure, not a successful sweep.

Still needed before kernel edits: pair every producer and **all** consumers
by dynamic token/tile/branch epoch; prove the lane participation of enclosing
paths; establish the complete collective-read footprints; and prove a
source-level transformation preserves each logical coordinate and reduction
tree. This ledger provides the address mapping needed for that work but does
not certify it. In particular, source-level `tl.sum` expresses a reduction;
the existence of cross-thread traffic alone is not proof that a split is
necessary. Separate launches become required where the needed lifecycle
cannot be expressed/proved in-kernel. Their intermediate arrays must be
program-private and preserve the existing read/mutation/token order.

No repair batch, paired-shape racecheck/parity gate, full battery, 100-repeat
9x bin, or budget-scaled probe was run. Zero endpoints and battery green are
not claimed. Promoted uninitialized-access masks and production remain
unchanged. No GPU was touched or leased, no .pt file was created or staged,
and no WG service/worker was used. The runtime exposes no contact_supervisor
tool (searched by tool name and description), so no intercom update could be
sent. Work remained single-threaded.
