# X26b launches and graph disposition

Registration/code/gates commit **7251c265** pushed origin/main BEFORE launch;
receipts/launch.json binds full commit/remote ref and registration SHA256.
Guard PID1369463 persists at60s; supervisor-owned native notification arming is
not claimed. `receipts/watcher_wiring.json` gives probe/terminal paths.

- GRU native IMM physical card4, worker1369527/trainer1369677.
- DeltaNet native IMM physical card5, worker1369528/trainer1369696.

Both fresh seed0, no weights restored. Real N300/batch128 backward+eval smokes
pass at zero updates. `receipts/launch_verified.json` independently checks
worker+trainer starttime/host/aliveness, registered configs, actual physical GPU
UUID-to-trainer PID binding. At verification GRU step3121/val3000=55.525%;
Delta step3000/val2500=55.23%. These are interim observations, NOT verdicts.
A later `--once` probe returns2 (both running); expected while terminal absent.
Free disk1.45TB; 400GiB minimum/40GiB per-rung/7day identity-safe guard active.

## Eager graph is censored, no graph relaunch

X26 eager graph latest153@59890.76s, recent12.3549updates/h, GPU2 active10%,
CPU100%. Update0 validation alone13898.28s. Not wedged; estimated60k cost202days.
Supervisor ruled INFRA-RESOURCE-INFEASIBLE, identity-verified TERM of the owned
worker group; old standard terminal receipt is CENSORED exit143. Card2 released;
old worker/trainer verified dead. No scientific inability claim.

An experimental persistent batch-fused Triton scan retained source ParameterLists,
clamp6/256/layer1/pooling/head and tested the eager oracle before any updates.
Six normal full-model short-sequence forward/parameter-gradient/eval cases pass
(B1/T11,B4/T33,B17/T65,fp32+bf16); normal BF16 worst parameter-gradient relative
error.015625. FP32 recurrence saturation+padding passes too. BUT BF16 saturation
input-gradient error.2734375 EXCEEDS pre-set.08; qualification FAILS. Therefore
no graph relaunch, no unqualified replacement installed. Rejected candidate and
test source retained in `/home/erikg/emender-scratch/fix-fanout/X26b-rejected-scan/`,
hashes and exact failure receipts in `receipts/graph_scan_deferred.json`,
`scan_tests.txt`, `scan_match.json` (the latter covers ordinary cases ONLY).
Do not mistake ordinary-case pass for overall qualification. Relaunch waits for
an improved implementation and a fully passing pre-update equivalence gate.

Field comparison and frozen decision tree are in COMPARISON.md/REGISTRATION.md.
No single-field causal attribution or exact Figure2 reproduction claim licensed.
Independent acceptance review required; controls continue under persistent guard.
