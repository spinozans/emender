# T7 stopped before the first fixed-route update

**The sampled route matched. The complete predictive distribution did not.** Do not use T7 as a matched frozen-route/value-learning control. The run stopped at its initial comparison rather than substituting an argmax, NLL or tolerance check. No fixed-route result on the natural batch and no post-update route-sensitivity estimate.

Same checked 32 TRAIN windows × 128 targets, T4 retained FP32 packet, step-1 P2 policy, indexed seed 7809 and strict four-thread CPU stack. Pass 1 sampled from a non-updating selector with gradients disabled; pass 2 performed the normal gradient-bearing sampled AdamW step. All **32 per-document event hashes, the dispatch hash, joint-score hash and scalar NLL** matched. Each pass executed **14,518 instructions**, submitted 2,854 logical keys in 2,165 physical dispatches and incurred 99,152 native LF steps. Each included 489 saves and 595 nonempty prior-byte recalls. The normal pass reproduced the entire T5 retained-strict-A check, including first derivatives, optimizer and updated weights. However, the SHA of the **4,096 × 256** output log probabilities was `0ee33c48…` in capture vs `dc99080a…` in normal training. NLL was 1.9111803770065308 in both. Hashes alone do not locate the difference, give its size, or establish which kernel caused it.

The [private in-memory capture](../../../ndm/memory_program/frozen_routes.py) retained discrete instructions/operands and ordered request/result-layout digests, not a table of teacher logits or native answers. Its 555,104 bytes of SHA digests exclude instructions and Python object overhead. Synthetic CPU tests verified later writer credit and refusal on changed scope, ordered/multiplied rows, input SHA and pre-dispatch request, but do **not** stand in for replay on these natural inputs.

| Pass actually completed | Wall after setup | Setup-inclusive lifetime | Main+child core-s in measured region |
|---|---:|---:|---:|
| Capture, real reads, no step | 8.587 s | 13.573 s | 28.89 |
| Full fresh sampled update | 12.422 s | 16.901 s | 43.99 |

**8,192 target computations, 4,096 distinct TRAIN positions, one independent full-policy update, zero fixed or host updates.** Four planned passes were not attempted. `proc_dd06` rc=1 after ~40 s, not a timeout or device failure. No rerun with altered seeds/tolerances. Zero additional card-seconds; X47 still **≤1324/1800**, 476 s remaining. No held-out scoring, corpus release or training promotion.

[Protocol](FROZEN_ROUTE_CONTROL_PROTOCOL.md) `4c7f2b20c`; gated `6a1413404`, 108 CPU tests. [Audit](receipts/memory-program-t7/verified.json), [terminal error](receipts/memory-program-t7/run.log), [capture](receipts/memory-program-t7/capture.json), [full step](receipts/memory-program-t7/full-update.json). Recheck with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python -m experiments.expressivity_tasks.x47_xsa_attention.frozen_route_verify`. No private feature files, route tapes, tensors or weights in git.
