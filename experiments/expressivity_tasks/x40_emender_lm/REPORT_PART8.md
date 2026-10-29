# X40 part 8 — three-architecture throughput surface v2

Sustained surface: **36/36** arms. 100k acceptance gate: **FAILED**.

Synthetic capacity-matched training: scale labels name pure trainable count; comparators match pure **total** parameters within 0.1%. This is a compute experiment, not a scale quality comparison.

Baselines use bf16 dense compute and fused bf16 vocabulary projection with fp32 master parameters/AdamW. Pure uses bf16 dense blocks, its parity-certified **fp32 native store** and bounded fp32 projection. This precision asymmetry is intentional; ratios compare against best-practice baselines.

Every rate requires three warmups and 20 synchronized forward/backward/clip/AdamW updates. Each arm starts bare. Only a measured CUDA OOM permits envelope escalation. OOM peak is a partial-run lower bound, never required capacity. First fitting means minimal in the declared policy ordering, not a globally fastest combination.

Failed envelopes remain in attempts/. The missing T512 checkpoint-gate filename caused a preflight failure, preserved in preflight_failures/ and retried using the existing strict B32 gate. Initial GDN2 timings with the forced pure backbone are rejected as best-practice evidence and retained in rejected_forced_gdn_backbone/. Accepted GDN2 configurations instead tune width/depth near the registered recipe MLP ratio; configs.json receipts their exact differences from the paper configuration. A partial packed B32 pure trial was interrupted when policy review found the missing checkpoint-only B32 candidate; its unqualified log remains in policy_corrections/. Each smaller microbatch now tests checkpoint-only before adding packing.

| Scale | T | B | Architecture | Status | Sustained tok/s | Peak allocated GiB | Envelope |
|---|---:|---:|---|---|---:|---:|---|
| 22M | 512 | 64 | transformer | passed | 147,881.8 | 16.281 | bare |
| 22M | 512 | 64 | gdn2 | passed | 117,092.3 | 17.127 | bare |
| 22M | 512 | 64 | pure | passed | 27,741.2 | 12.332 | bare |
| 22M | 512 | 128 | transformer | passed | 131,120.8 | 31.571 | bare |
| 22M | 512 | 128 | gdn2 | passed | 107,291.0 | 33.266 | bare |
| 22M | 512 | 128 | pure | passed | 45,817.1 | 24.038 | bare |
| 22M | 2048 | 64 | transformer | passed | 90,286.1 | 15.741 | checkpoint |
| 22M | 2048 | 64 | gdn2 | passed | 84,556.4 | 9.433 | checkpoint |
| 22M | 2048 | 64 | pure | passed | 14,820.5 | 13.618 | checkpoint |
| 22M | 2048 | 128 | transformer | passed | 89,734.3 | 30.492 | checkpoint |
| 22M | 2048 | 128 | gdn2 | passed | 81,561.8 | 17.977 | checkpoint |
| 22M | 2048 | 128 | pure | passed | 22,194.4 | 26.754 | checkpoint |
| 150M | 512 | 64 | transformer | passed | 50,540.7 | 25.395 | bare |
| 150M | 512 | 64 | gdn2 | passed | 46,986.1 | 26.591 | bare |
| 150M | 512 | 64 | pure | passed | 15,822.4 | 34.332 | bare |
| 150M | 512 | 128 | transformer | passed | 34,711.1 | 10.425 | checkpoint |
| 150M | 512 | 128 | gdn2 | passed | 40,224.6 | 12.692 | checkpoint |
| 150M | 512 | 128 | pure | passed | 18,683.8 | 10.815 | checkpoint |
| 150M | 2048 | 64 | transformer | passed | 30,172.1 | 17.787 | checkpoint |
| 150M | 2048 | 64 | gdn2 | passed | 36,256.5 | 22.357 | checkpoint |
| 150M | 2048 | 64 | pure | passed | 7,479.0 | 19.924 | checkpoint |
| 150M | 2048 | 128 | transformer | passed | 29,405.4 | 32.508 | checkpoint |
| 150M | 2048 | 128 | gdn2 | passed | 36,288.2 | 41.685 | checkpoint |
| 150M | 2048 | 128 | pure | passed | 10,459.4 | 37.685 | checkpoint |
| 1.3B | 512 | 64 | transformer | passed | 5,753.4 | 27.384 | checkpoint |
| 1.3B | 512 | 64 | gdn2 | passed | 5,936.3 | 29.359 | checkpoint |
| 1.3B | 512 | 64 | pure | passed | 3,628.9 | 27.573 | checkpoint |
| 1.3B | 512 | 128 | transformer | passed | 6,039.7 | 37.399 | checkpoint |
| 1.3B | 512 | 128 | gdn2 | passed | 5,809.9 | 41.520 | checkpoint |
| 1.3B | 512 | 128 | pure | passed | 4,584.5 | 38.323 | checkpoint |
| 1.3B | 2048 | 64 | transformer | passed | 6,233.1 | 42.747 | checkpoint_B32 |
| 1.3B | 2048 | 64 | gdn2 | passed | 6,363.6 | 34.652 | checkpoint_B16 |
| 1.3B | 2048 | 64 | pure | passed | 1,823.2 | 42.728 | checkpoint_B32 |
| 1.3B | 2048 | 128 | transformer | passed | 6,243.5 | 42.748 | checkpoint_B32 |
| 1.3B | 2048 | 128 | gdn2 | passed | 5,961.7 | 34.653 | checkpoint_B16 |
| 1.3B | 2048 | 128 | pure | passed | 1,789.5 | 42.729 | checkpoint_B32 |

| Scale | T | B | Pure / transformer | Pure / GDN2 | GDN2 / transformer |
|---|---:|---:|---:|---:|---:|
| 22M | 512 | 64 | 0.1876 | 0.2369 | 0.7918 |
| 22M | 512 | 128 | 0.3494 | 0.4270 | 0.8183 |
| 22M | 2048 | 64 | 0.1642 | 0.1753 | 0.9365 |
| 22M | 2048 | 128 | 0.2473 | 0.2721 | 0.9089 |
| 150M | 512 | 64 | 0.3131 | 0.3367 | 0.9297 |
| 150M | 512 | 128 | 0.5383 | 0.4645 | 1.1588 |
| 150M | 2048 | 64 | 0.2479 | 0.2063 | 1.2017 |
| 150M | 2048 | 128 | 0.3557 | 0.2882 | 1.2341 |
| 1.3B | 512 | 64 | 0.6308 | 0.6113 | 1.0318 |
| 1.3B | 512 | 128 | 0.7591 | 0.7891 | 0.9620 |
| 1.3B | 2048 | 64 | 0.2925 | 0.2865 | 1.0209 |
| 1.3B | 2048 | 128 | 0.2866 | 0.3002 | 0.9549 |

The 100k gate requires fresh sustained bare pilot throughput, pressure-inclusive production parity and a matched curve. It uses the original v1 pilot (about 21.719M trainable), whereas the surface uses the parameter-targeted scale family (21.999M at 22M). Their bare rates are separate results; the historical 54.8k pilot rate is not substituted for a surface cell. No historical measurement substitutes for a missing new gate. See gate.json for qualification.

Residual risks: synthetic capacity matching does not establish model quality across scale; B1 native parity does not establish byte equality of every full scale model. Envelope ordering does not test every combination. Sparse backward union pooling and event suppression retain exact replay; no unsupported skipping of value-consuming events is accepted. A failed 100k measurement establishes no hardware floor.

No WG, subagents, .pt artifacts, Frontier, Slurm or resilient DiLoCo changes. AGENTS.md and CLAUDE.md are unchanged. contact_supervisor was unavailable. Protected GPU cards and the depth probe were untouched.

Measured scale trends (22M → 150M → 1.3B):
- T512/B64, pure_over_transformer: 0.1876 → 0.3131 → 0.6308; monotonic increase: True.
- T512/B64, pure_over_gdn2: 0.2369 → 0.3367 → 0.6113; monotonic increase: True.
- T512/B128, pure_over_transformer: 0.3494 → 0.5383 → 0.7591; monotonic increase: True.
- T512/B128, pure_over_gdn2: 0.4270 → 0.4645 → 0.7891; monotonic increase: True.
- T2048/B64, pure_over_transformer: 0.1642 → 0.2479 → 0.2925; monotonic increase: True.
- T2048/B64, pure_over_gdn2: 0.1753 → 0.2063 → 0.2865; monotonic increase: True.
- T2048/B128, pure_over_transformer: 0.2473 → 0.3557 → 0.2866; monotonic increase: False.
- T2048/B128, pure_over_gdn2: 0.2721 → 0.2882 → 0.3002; monotonic increase: True.

Uniform monotonic improvement against both comparators is established only if every corresponding trend above passes. A reversal is retained as an experimental result, not averaged away. These are measured throughput ratios, not statistical significance or model-quality claims.

Fresh 100k gate: **FAILED**. Sustained tok/s: 54836.098334469425. Qualified parity/curve measurement: True.
Matched curve: `{"batch": 128, "candidate_physical_batch": 128, "candidate_receipt": "part8_pilot_B128_bare_fused_sparse.json", "context": 512, "max_loss_difference": 1.9073486328125e-06, "reference_microbatch": 32, "scope": "Identical initialization/data/effective batch, weighted microbatch means, one clip and AdamW per update. Reference uses recorded checkpoint and exact journal packing, original stream/register policy.", "status": "passed", "steps": 31, "tokens": 2031616, "tolerance": 1e-05}`.

Ratio curves: [SVG](receipts/part8/ratio_curves.svg). Full configurations and external kernel provenance: [configs.json](receipts/part8/configs.json). Each surface row links its complete ordered attempt receipts in the corresponding JSON under receipts/part8/surface/.
