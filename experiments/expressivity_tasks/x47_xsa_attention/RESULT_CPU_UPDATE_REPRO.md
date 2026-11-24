# Strict four-thread CPU learners reproduced complete program updates from frozen host features

**Qualified for these offline CPU workers, not a retrofit to T4.** Two independent CUDA-invisible workers per frozen host packet, with `torch.use_deterministic_algorithms(True, warn_only=False)`, produced bit-identical **first derivatives, updated policy weights and AdamW moments** on both the real 150M and dense replicated 2.22B serving features. In all six runs the fresh sampled program issued the same logical instructions and native requests as the corresponding prior coupled forward. Every byte-vector and joint selection score matched. Nothing trained the backbone; no replayed instruction sequence or held-out inputs.

| Retained packet comparison | Unequal gradient elements / max absolute | Unequal post-step parameter elements / max absolute |
|---|---:|---:|
| Ordinary A vs B, both standalone | 403 / 2.10e−9 (`embed.weight` only) | 3 / 1.49e−8 |
| Strict A vs B, both standalone | **0 / 0** | **0 / 0** |
| Ordinary A vs strict A | 28,757 / 4.77e−7 | 2,544 / 2.73e−6 |

Strict A vs B on the replicated host packet likewise had **zero** unequal elements, including the two moment arrays and step counters. All 288,987 policy parameters were checked in each post-step snapshot. Ordinary vs strict is a change of numerical implementation, not a different cost-to-go estimator: all gradients met the separately registered T1-scale characterization on the *new* pair. This doesn't reveal the magnitude of T4's original two hash differences, identify a particular atomic operation as the cause, or qualify an ordinary coupled update as exactly equal to a strict one.

Retained ordinary update wall 10.957 / 11.282 s; strict 11.385 / 11.380 s (**1.024× mean**, passed ≤1.25× gate). Replicated strict 11.518 / 11.284 s. Each measured dense policy step used ~38–40 CPU core-s, ~2.16 GiB learner RSS plus separate native processes; all performed 99,152 LF steps. Timed interval includes native work, backward, clip/update and ~4–5 ms pre-clip snapshot checks, not worker startup or publication of the comparison files. Complete sequential diagnostic **125.7 s**, six disposable AdamW steps: 24,576 target computations on **4,096 distinct TRAIN positions**, not a training curve or pricing of six independently trained 2B models. No extra GPU lease/card-seconds. X47 remains ≤1284/1800, **516 s available**.

[Protocol](CPU_UPDATE_REPRO_PROTOCOL.md) `4daceb7c5`, gated `880fba381`, 101 CPU tests; [audit](receipts/memory-program-t5/verified.json) reruns from hashes, per-tensor element/error counts and timing receipts. Private tensors were compared locally, not committed. For an exclusive worker requiring reproducible program updates at this shape, enable strict algorithms for the entire step and pin CPU threads, framework, inputs, mounts and sample coordinates. It is a process-wide setting, not a per-client serving option or cross-machine guarantee.
