# Phase4 long-context tractability qualification

Before measurement: use broker card4, certified phase2 sources unchanged.
Native-forward controlled schedule peak min(T,257), zero values, no pop,
B/T/cap grids (1,128,257),(16,128,257),(1,257,515),(1,512,1025),
(16,512,1025),(1,1024,2049). BS=nextpow2(min(cap,2T)). Three warmups,
repeat bit fingerprints,10 synchronized durations, compiler registers/spills.
A repeat check is not reference/gradient qualification. No training here.

Locate the long-context extent at which spills return; compare B1/B16 at fixed
T/BS to expose sequence-grid effects. Evaluate any wrapper-only adjustment with
reference/gradient gates before claiming semantic equivalence. If insufficient,
name an internal kernel design change and its X31 recertification bill rather
than invent a tractable forecast. Budget class: historical final_pure width2048,
L22,N2,B1/T1024,2e9 raw-byte target scenario in86400s (token corpus differs).
Report measured native head time *22*2*2/1024 seconds/byte, plus omitted costs,
required23148.148148 targets/s, FP32 params/grad/Adam minimum20.8GB at1.3B.
No full1.3B training or memory qualification inferred from small model timing.
