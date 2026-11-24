# W2: a saved cue and two branch reads change the current byte

**The initialized workbench resolved all four cases before observing the target.** At the decision point the visible eight bytes were always `789;key=` and the parent branch was always `{a:1, b:1}`. Changing either the earlier cue or the archive binding flipped the emitted code. This adds something W1 didn't demonstrate: the second read changes *this* prediction, not just the next one. A fixed CPU join got exactly the same answers for less time. This is a constructed program on a fixture, not learned language understanding or an all-byte LM result.

## What ran

The two-doc archive contains `key=aM, key=bS` in world 0 and `key=aS, key=bM` in world 1. The 22-byte input is `unit=<M or S>;0123456789;key=`. No code label is fed to the model. At visible position 6, the device recognizes `unit=<byte>` and writes a vector representing that last observed byte. Sixteen more observed bytes pass. The only way the policy receives the old cue is through its session-bound record; it sees neither the whole transcript nor a host-provided cue or world ID.

At position 22 the [model](w2_model.py) issues a suffix read at `key=`, ranks the two actual outgoing byte IDs, and sends both to the [runtime](w2_join.py). Two read-only replicas of the same sealed native fixture answer the two independent child queries concurrently. Their answers are rebound to logical handles, not whichever worker finished first. The model takes the dot product of each child's byte counts with its stored vector, assigns the resulting mass to the corresponding code, and normalizes. The byte distribution keeps a positive local fallback (1:4 local:archive). **Only then is the emitted code appended.** Empty byte evidence backs off; END is not a candidate, and parent fanout above two is refused rather than clipped.

| Archive after `key=` | Earlier cue | Emitted code | Full P(correct) | P(correct) after parent alone |
|---|---|---|---:|---:|
| `a→M, b→S` | M | a | 0.80078 | 0.40078 |
| `a→M, b→S` | S | b | 0.80078 | 0.40078 |
| `a→S, b→M` | M | b | 0.80078 | 0.40078 |
| `a→S, b→M` | S | a | 0.80078 | 0.40078 |

**Interventions.** Mask only the saved payload at recall, or replace both child histograms with their average (same tag totals, no edge-to-tag association). Each intervention restored the parent-only distribution on **all four** cases: 0.40078 on each code, 2/4 greedy accuracy by deterministic tie break. Both ran the same 25 Transformer passes (22 prompt + 3 current), opcodes and 3 native queries as the unablated model. The fixed non-neural join parsed the same online input and read the same three histograms; it matched the full distribution (max error within 3e−5). Mean NLL **on these four designated code bytes only**, full / fixed / erase / blind: **0.22217 / 0.22217 / 0.91434 / 0.91434**.

**Checks.** [16 CPU tests](receipts/w2/cpu/tests.txt) passed, including W1 regressions, stale record/handle refusal, END-only/missing/overcap, other binary codes/tags and reversed reply order. A later loss had a gradient on the earlier tensor writer without an optimizer step. The 609,848-parameter model reused W1's 20-slot, width-288 attention/MLP block. Marker recognition, record layout and join rule are explicitly initialized, not discovered. GPU reloaded the CPU state (`ea836b1a70e0487f44354398d3b61181d386c33227924914d78ba68b38e4eded`). CPU/GPU actions matched and max discrepancy in the checked writes and byte probabilities was 0. An [offline audit](receipts/w2/verified.json) reconstructed both archive identities, native answers, causal traces, selected-byte results and timing populations from [CPU](receipts/w2/cpu/cpu.json) and [device](receipts/w2/gpu/gpu.json) receipts. Max analytic probability error 5.97e−9 or less.

## Cost vs doing the same join directly

Single RTX 6000 Ada, PyTorch 2.9.1+cu128, FP32 eager, OMP=4; native branch services resident on CPU. Four warmup decisions per arm, then eight repeats of all four cases per arm, with paired arm order rotated. The primary decision wall includes all three real reads, rebuilt views, device work, returning the greedy byte and appending it. **Prefill is separate, not free:** both arms processed all 22 observed bytes in order, including the original write. Repeated synthetic keys, not novel archive latency or general LM throughput.

| Arm | Median current-byte wall | p95 current | Mean current | Median 22-byte prompt processing |
|---|---:|---:|---:|---:|
| Workbench, GPU | 6.587 ms | 6.923 ms | 6.455 ms | 43.23 ms |
| Fixed join, CPU | 0.990 ms | 1.254 ms | 0.981 ms | 1.98 ms |

The workbench spent **+5.474 ms per decision** on average vs the fixed join (ratio of medians 6.65×). The fixed baseline needs neither Transformer passes nor a dense tensor to store a known one-byte tag. In the separate fenced four-case diagnostic, workbench mean time per current decision: parent RPC 0.121 ms, native child pair wall 0.641 ms, packing 0.197 ms, H2D 0.098 ms, three forwards together 4.877 ms, decision returns 0.084 ms, append 0.055 ms. The two child RPC intervals overlapped in all four diagnostic workbench cases (mean 0.201 ms, **inside** the child pair wall). These are host wall times including dispatch, not kernel-only times. Replica construction/seal per archive world 4.60 / 5.49 ms outside clocks. [Raw samples](receipts/w2/gpu/samples.jsonl) · [summary](receipts/w2/gpu/timing.json).

**Decision:** the current-byte dependency is real, and the mechanistic gate is complete. More fitting on this example would not distinguish the model from the known rule. The remaining question is whether such joins occur often enough on natural bytes and whether a learned policy can improve on equally informed fixed reads. No such panel was opened or trained here.

## Reproduction and bounds

Protocol `bbb8b6bbc` ([W2_PROTOCOL](W2_PROTOCOL.md)); runnable code/tests frozen before outcomes at `98b1a9dc8`. From the repo root:

```bash
D=experiments/expressivity_tasks/x47_xsa_attention
M=experiments.expressivity_tasks.x47_xsa_attention
P=/home/erikg/emender/.venv/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$P" -m "$M.w2_verify"  # receipts only, stdlib
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$P" -m pytest -q "$D/test_workbench.py" "$D/test_w2.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$P" -m "$M.w2_run" cpu output/w2-new-cpu
# Only with remaining authorization/budget, and a new directory:
# bash "$D/workbench_leased.sh" output/w2-new-cpu output/w2-new-gpu w2
```

Local checkpoint `output/w2-cpu1/run/initial.pt`, bound by file and tensor hashes in the CPU receipt (not staged). One lease on physical 0, owner PID/starttime 113832/553644204; heartbeat advanced from 1795470454 to 1795470459, [released](receipts/w2/gpu/released) at 1795470459 rc=0. ≤6 new card-seconds, X47 cumulative ≤66/1800. No push, no Pile reads, no training, no remaining W2 job.
