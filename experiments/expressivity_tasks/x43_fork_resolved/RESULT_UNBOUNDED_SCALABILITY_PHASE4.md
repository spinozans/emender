# X40/X43 scalability phase4 — 1.3B tractability verdict

**The registered1.3B/24h class is not qualified. The named blocker requires kernel-internals work; no wrapper-only fix is promoted.**

Physical card4 via detached broker wrappers. No model training, checkpoint, frozen arm, union service, card1, sxgc or mounted corpus changes. Every row below is native-forward only; repeated fingerprints and controlled telemetry pass, without long-context reference/gradient certification.

| B | T | cap | BS | median ms | registers | compiler spills |
|---:|---:|---:|---:|---:|---:|---:|
|1|32|65|64|0.635626|255|32|
|1|64|129|128|1.414367|255|298|
|1|96|193|256|3.581210|255|858|
|1|128|257|256|4.758182|255|862|
|16|128|257|256|5.814233|255|862|
|1|129|259|512|21.508696|40|3312|
|1|257|515|1024|79.485388|40|8068|
|1|512|1025|1024|148.331667|40|7696|
|16|512|1025|1024|182.929147|40|7696|
|1|1024|2049|2048|718.648853|40|13802|

Residual spills already exist at C14 T128/BS256 (862); “return” is an amplification of an existing spill wall. The lower-context sweep quantifies residual spills and the first BS512 grid step at T129. Counts are compiler metadata, not hardware spill transactions. Nsight hardware counters remain unavailable per phase2 ERR_NVGPUCTRPERM.

Terminal B1/T1024 native head median is 0.718648853s. Math: 0.718648853 *22 layers *2 heads *2 native passes /1024 = 0.061758886s/byte; 16.192002 targets/s;2e9 targets => 1429.604 native-only days. Required2e9/86400 = 23148.148148 targets/s, a 1429.6x scenario gap. This is a controlled peak257 workload and a raw-byte budget-count scenario; the historical terminal corpus was GPT2 tokens. No token/byte equivalence is claimed.

The single-sequence grid launches one CTA on142 SMs. At8 warps/CTA, analytic global warp occupancy is at most8/(142*48)=0.1174%, versus1.878% at B16. Measured B16/B1 rows compare throughput at fixed shape; batching changes the B1 protocol and is not a free solution.

The reachable wrapper already scans nextpow2(min(cap,2T)); reducing cap would violate unbounded capacity. The kernel hard-codes8-warp Gluon layouts; increasing num_warps alone cannot change those layouts legally. At BS2048 the full value vector has2048*128/256=1024 FP32 elements/thread before keys and metadata, versus at most255 registers/thread. Global staging between tokens still reloads a full BSx128 value tensor within each token. A spill-free long-context path needs tiled value reads/updates with an exact reduction tree and bounded intermediate lifetime; distributing a sequence across CTAs also needs deterministic inter-CTA state/update synchronization. These are internal design changes, not certified wrapper knobs.

X31 bill: implement and retain the new kernel separately; rerun the109-oracle/105-C14-operator regression gate; exact nonzero read/state/trace and replay/gradient fingerprints including tie/POP/CORRECT/MERGE/DECAY cases; add B1/T512/1024/2048 with cap2T+1, high-live fixtures, masked tails and document resets; rerun >=2M real masked targets with every loss/gradient/optimizer/RNG/cursor/checkpoint-resume fingerprint; then measure150M and1.3B-width full training throughput and peak memory. Existing20.8GB FP32 parameter/gradient/Adam arithmetic excludes activations/store/replay. Engineering time and GPU-hour cost for that rewrite are unmeasured, so no invented numeric fix price is reported.

Attempt1 is retained as failure: B16 fixture had B1 logits, rejected before valid measurement. Own process group stopped, wrapper exit143. Corrected fixture was pushed before rerun. Lower-context supplement was independently registered before measurement. No kernel internals were edited or certified here. Fresh document/branch tests:9 passed. No resilient DiLoCo behavior changes; gap-matrix IDs inapplicable.

Residual risks: full1.3B training and natural-scale occupancy/memory remain unmeasured; repeated native fingerprints do not certify long-context gradients; the controlled schedule may differ from training. Kernel rewrite cost is a named validation bill, not a measured duration estimate.
