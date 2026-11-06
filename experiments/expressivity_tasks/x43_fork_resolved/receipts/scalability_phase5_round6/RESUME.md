# Exact resume: measured traffic prerequisite still blocked

First audit commit bb3997ac was pushed before component instrumentation.
No candidate, training process or checkpoint exists. No GPU process remains
from this work. R5 remains the unchanged diagnostic implementation.

1. Read TRAFFIC_AUDIT.md, audit.json, traffic_terminal.json/traffic_console.txt,
   component_fixed.json and RESULT_UNBOUNDED_SCALABILITY_PHASE5_ROUND6.md.
2. Obtain operator/admin resolution of ERR_NVGPUCTRPERM or explicit operator
   amendment of the physical-traffic-first prerequisite. No privilege or
   driver changes are authorized/inferred here. No contact_supervisor tool
   was exposed in this session; no unanswered tool request is pending.
3. Physical-counter probe launch-count1 was only a permission preflight. After
   resolution collect the ENTIRE warmed R5 store cycle, both read/write DRAM
   and preferably L2 traffic, with no concurrent torch.profiler subscriber.
   Separate physical traffic from logical requests and cache hits. Use a NEW
   wrapper/receipt label; existing traffic wrapper refuses completed output.
4. Finish unresolved component measurements: backward and its retained tensors,
   gather/carry separation where valid, metadata versus content update costs,
   enqueue gaps with matched instrumentation. Do not add overlapping intervals.
5. Only then receipt design before implementing the fused op-cycle. Preserve
   exact normalization tree, FP32 parity chains, masked tails, document reset,
   mutable CORRECT and mid-flight POP/MERGE. Reuse R4/R5 compositional fixtures.
6. New backward choice must follow measured stored intermediates/recompute cost.
   Run ten shapes, full byte/replay/gradient gates and1.3B projection. Standing
   >=2M optimizer/RNG/cursor/resume and full-width throughput gates remain.

Fresh timing-only replication (does not unblock physical traffic):

```
bash experiments/expressivity_tasks/x43_fork_resolved/run_c14_phase5r6_profile_leased.sh component_recheck1
```

The label must be new. Only /home/erikg/emender/.venv/bin/python, broker cards
3/4/5/6; supplied wrapper restricts physical4 and fails on occupied/unavailable.
No WG/delegation. Push each scoped commit; never stage unrelated changes or .pt.
