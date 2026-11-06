# Phase 5 experimental live-range attack

This is an opt-in, uncertified internal-kernel experiment. No production runner
or historical registration is amended. Read phase2/3/4 reports and X31 part8
requirements before implementation. Python is the repository .venv interpreter;
only broker physical cards4 (probe) and5 (sequential validation) are used.

The reference materializes BS*128 values, plus BS*64 key score intermediates.
At BS2048 the former alone represents 1024 FP32 elements/thread before
metadata. Candidate stages values through global memory in 16-column chunks,
keys in 128-row chunks, with the existing explicit reference reduction layouts.
FinalS is private transient score scratch; FinalMeta is transient float-bit
scratch for previous-value vectors, overwritten by complete final state.
The small-S dispatch retains the reference when 7*S<128. No cross-document
state or cross-sequence scratch is permitted. CTA barriers guard changed
producer/consumer thread layouts; the first missing-barrier failure is retained.

The probe compares complete nonzero read/state/trace byte fingerprints and
repeats, then reports five synchronized native-head samples and compiler
register/spill metadata for the ten phase4 shapes. This workload has nonzero
constant values (phase4 timing used zero values), so timing ratios are diagnostic
rather than matched-input speedup evidence. Compiler spills are not hardware
transaction counters. Nsight permission limitation is inherited, not retested.
A speed regression or any parity failure rejects this candidate for promotion.

Run all 109 X31 tests and all 105 C14/operator tests with candidate explicitly
installed, plus the existing nonzero shape/gradient probes. These are necessary
and insufficient: full trained-model fingerprints, >=2M paired real masked
targets with optimizer/RNG/checkpoint resume, long-context operation-rich
fixtures including T2048, and full-scale training/memory remain required before
promotion. Do not run the expensive paired training gate for a rejected speed
candidate. Required class remains 2B raw-byte scenario targets in 86400 seconds:
23148.148 targets/s. At 22 layers, two heads and two passes per target window,
T1024 requires <=0.50269 ms/head even if all other work were free.

Supervisor/intercom tools were searched by name and description; none is
available. No supervisor coordination is claimed. No WG, delegation, training,
protected cards1/2, sxgc or /mnt/nvme2n1 operation is authorized here.
