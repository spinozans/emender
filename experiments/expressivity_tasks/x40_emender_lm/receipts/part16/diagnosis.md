# X40 part 16: race-class evidence, causal reduction remains unresolved

Bounded unknown. The unchanged production path is intermittent at the identical
B8/T2286/S2944 batch, and compute-sanitizer reports shared-memory hazards in
scan_kernel. This is stronger evidence for an ordering/race-class failure than
for a deterministic arithmetic overflow. It does not prove which hazard causes
the int32-add assertion, or a safe minimal repair. No production fix is made.

## Sanitizers first

All GPU work used sequential exclusive broker leases on physical card 2, with
an allowlist of 2/3/4. broker_before.txt, lease*.txt, and broker_after.txt retain
the evidence; card 2 was released and idle afterward. Cards 0/1/5/6 and forbidden
card 7 were never selected. No search, anchor, service, worker or dispatcher was
started, stopped or preempted. Host: lambda01; approved .venv Python only.

run.sh exercised the unchanged part15/single.py batch before any new GPU probe:

- memcheck: complete batch, 24 scans, exit 0, **0 errors**.
- racecheck: exit 86, **35 error groups**, int32-add assertion already at scan 1,
  CUDA block 7. Reports involve generated shared reduction scratch around
  kernels.py:99/119/120 and :149/150/151, plus standard.py reductions. The
  complete raw report is racecheck.txt; this is a failure, not a clean check.
- initcheck: uninitialized global reads at kernels.py:102, ReadStrengthCache.
  Stopped after >5 minutes during scan 4 (exit 143); see initcheck_stop_note.txt.
  No terminal clean result or complete-batch coverage is claimed.

Filtered attempts are retained, including their limitations:
memcheck_scan10.txt completes with 0 errors; racecheck_scan10.txt fails at scan 1
before the intended scan 10, with an internal sanitizer hardware-exception error
and one reported error group. initcheck_scan10.txt reaches and completes scan 10
(prints scan 11), then hits the 600-second timeout (exit 124). Its printed
host-copy uninitialized errors are not reliable attribution: filtering out
producer kernels removes their write tracking. Neither failed filtered attempt
is a clean scan-10 certification.

To finish a fully instrumented check of the captured failing shape without
that filtering limitation, phase3.sh runs unchanged part15/isolated.py original:

- memcheck_isolated.txt: PASSED, exit 0, **0 errors**.
- racecheck_isolated.txt: **29 error groups**, exit 86, int32-add assertion in
  block 6. Isolation therefore does not remove the sanitizer-visible failure.
- initcheck_isolated.txt: application prints PASSED, but sanitizer exits 86 with
  **8,224 errors**, 100 printed and 8,124 suppressed by its default print limit.
  All 100 printed sites are kernels.py:102. Every emitted message is retained.

The cache is allocated with empty_like and loaded for newly live slots before
selecting r_reference with tl.where (:102-108). An unselected uninitialized
operand can explain these reports; its causal role in the assertion is **not
proven**. Racecheck reports concern shared memory, not proof that History's
global-memory ordering is the sole problem. Tool perturbation remains relevant.

## Repeatability and layout

Ten independent processes run the exact part15/single.py reproduction, same
checkpoint, batch start 80 and flags. **6 pass; 4 fail**, at calls **10, 10, 22,
18** (repeat_2/4/7/9.txt). The launch is not fixed. Even the two call-10 failures
select different CUDA blocks (5 and 6). This satisfies the work order's varying
launch/race-class discriminator, without identifying a specific race.

The fixed captured scan-10 inputs survive **100/100** ordinary replays with
addresses logged and another **100/100** without address logging, in two fresh
processes. These are repeated calls within each process, not 200 fresh-process
samples; output parity was not certified. The racecheck failure above prevents
calling isolation universally safe. environment.txt hashes the uncommitted
/tmp capture; no .pt is committed.

Thirty new-process layout probes use stable mapping 0/1 and retained padding
allocations of 0, 512, 65,536, 1,048,576 or 16,777,216 bytes before each scan,
three runs per combination. **19 pass, 11 fail**. Stable=0 fails 4/15;
stable=1 fails 7/15. Every padding below 16 MiB has mixed outcomes across its
repeats/settings; 16 MiB passes 6/6. That is a candidate perturbation effect,
not a workaround or statistical proof. Failures move across calls 1-23.

Each launch logs all **62 tensor arguments** of scan_kernel, including internal
scratch, with data/storage bases, offsets, strides, sizes and 256/4096-byte
residues. Raw layout_*.txt and isolated100.txt retain all launches. Summaries:
run_summary.json, layout_addresses.json and address_comparison.json.
All failing launches' tensor arguments are 256-byte aligned. Page residues
sometimes differ, but a stronger control exists: failing layout_0_0_r3 call 14
and passing layout_0_0 call 14 have **identical relative addresses for all 62
arguments, identical address order, and identical page residues**, differing
by a uniform absolute address translation. Simple relative address order or
alignment alone cannot discriminate this pair. Absolute-address effects,
code/shared-scratch layout, timing and prior device execution remain open.
No claim that the failure follows padding deterministically is justified.

## X31 connection and bounded exclusions

source_findings.md identifies the exact part-4 history-store/reload/reduce site
at kernels.py:287-292, in the same persistent kernel. The diagnostic barrier
from part 4 is still absent there; the earlier barrier precedes the history
store. This is a concrete shared risk site. The reported racecheck groups also
cover earlier read reductions. No causal equivalence between the two symptoms
has been established, and no diagnostic barrier is installed this session.

Three bounded exclusions are justified:

1. The proposed allocation-product overflow: part 15's 53,839,872 freed bytes,
   1,507,328 key floats and 3,014,656 value floats are below int32 bounds; counted
   live/history lanes are bounded. This excludes that explanation, not every
   possible generated integer error.
2. X40_STABLE_MAPPING as a direct forward-path repair: its only branch is in
   backward.rebase_owned, bypassed by this no_grad path. Both settings fail.
3. Relative input address ordering or 256/4096-byte alignment as a sufficient
   deterministic explanation: the matched call-14 layout pair above disagrees
   in outcome despite matching those properties. Absolute addresses are open.

The exact failing reduction and the relationship among shared-memory hazards,
the pressure-history barrier, and unused cache loads remain unresolved.
Full production 9x depth acceptance remains failed/missing. No kernel edit was
made, so X31 bitwise gates, 109 oracles, graph/eager and curve-match recertification
were not run or claimed. production_hashes.txt verifies unchanged production
sources. Only evidence/scaffolding under part16 is included in this commit.

Coordination limitation: contact_supervisor is not exposed in ALL_TOOLS (names
and descriptions searched at task start). No alternative messaging channel
was used. This receipt claims no approval for a compiler workaround or weakened
gates. No resilient training behavior or Slurm submission was changed/exercised.
