# Exact round2 resume: unqualified speed candidate

No training, optimizer checkpoint or training process exists. Best experimental
source is c14_phase5r2_selected_kernel.py, installed only by the explicitly named
selected validation module. Production kernels and historical bindings are
unchanged. Do not promote it. Final evidence is validation_summary.json and
selected/sweep_fixed.json; sweep.json is the failed compile attempt, whose old
source bytes are retained in selected/attempt1_kernel.py.

Both candidates pass ten native/repeat pairs,109 X31,108 C14/operator,8 short
shape,4 short gradient/replay, T512/T1024 peak257 masked reset, and24 adversarial
primitive pairs. T2048 has no result: v1 stopped143 after >15min (agent-selected
diagnostic budget), instrumented retry124 after180s, selected retry124 after180s.
Faulthandler proves the unchanged reference is in Triton make_llir ->
llvm.optimize_module(OPTIMIZE_O3), before candidate execution. Do not call this
a byte mismatch or a green gate. gdb attach was denied; its failure is retained.

Next task is the T2048 reference compiler bottleneck. Preserve the original
S4097/BS4096 gate and arithmetic; do not shrink the required context or silently
swap in a weaker reference. After diagnosing compilation, rerun with a fresh
output name under an exclusive broker lease on physical3/4/5/6 only:

/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r2_selected_long_parity --tokens 2048 --output experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round2/selected/long_2048_retry2.json

The existing launchers deliberately refuse completed receipt paths. Copy a
launcher to a new explicitly registered path and use a fresh receipt directory
for a changed candidate/battery. Rebind new source bytes in the same commit,
check index hashes/pre-commit, push, and check HEAD bindings. Preserve failures.
Full-model fingerprints,>=2M paired real-target optimizer/RNG/cursor/checkpoint
resume, maximum-live operation-rich long cases, and150M/1.3B throughput/memory
remain open. The best native-only projection is557.274949 days; training is not
warranted as a24h qualification attempt. Scalar metadata and global single-CTA
underutilization remain structural limitations after the dominant matrix live
ranges were removed. No unmeasured eventual speedup is promised.

The work was performed serially. Supervisor/intercom transport was unavailable.
Final own-process and cards4/5 lease/GPU checks are in completion_check.json.
No protected-card operation or corpus change is needed to resume this work.
