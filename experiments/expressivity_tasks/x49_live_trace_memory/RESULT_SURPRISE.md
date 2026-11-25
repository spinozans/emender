# X49-U: remembering how surprising the old follower was improved the next prediction

**Remembering the old forecast helped on the out-of-fit diagnostic.** B already knows which earlier byte to retrieve. When that byte first arrived, U also stored **one byte encoding how improbable it had been under the preceding host forecast**. Later, with the address, retrieved byte and 32-D neural payload unchanged, the correct mark reduced loss by **.002671 nats per real byte** over current-confidence calibration, and **.002488** over an equal-size calibrator trained and tested on a displaced mark. Both paired intervals exclude zero; both gains are positive at every seed. **Carry this exact fitted readout to a separate native held-out confirmation.** This is not yet a final-test result or a scale nomination.

[Prospective protocol](PAST_SURPRISE.md) · [implementation](SURPRISE_IMPLEMENTATION.md) · [aggregate, per-seed coefficients, work and clocks](receipts/surprise_result.json) · [closed-receipt audit](surprise_report.py).

## What got better, and relative to what?

For each of three fixed W continued-B predictors, fit only scalar coefficients, once, on 256 new native TRAIN documents. Freeze **every** fit before opening any of the 128 disjoint PROBE documents, also native TRAIN. Score all 524,288 shifted targets per seed and cell, including misses; no neural weights updated. At a hit the calibration changes only the odds of the retrieved byte, not the relative probabilities of the other 255.

| Predictor | Information used by the calibrator | Probe NLL (nats/byte) |
|---|---|---:|
| B | No calibrator | 1.111451 |
| N | Current B and bare-host confidence | 1.111192 |
| S | N plus a surprise mark for the same observed byte from a rotated age-matched record | 1.111009 |
| **U** | **N plus the past surprise of the actual retrieved occurrence** | **1.108521** |

Positive gain = control minus candidate. These 95% percentile intervals resample whole documents jointly across all four predictors and fixed seeds (10,000 draws, RNG 4907). They do not treat bytes or repeated seed passes as independent samples.

| Candidate : control | All-byte gain [paired 95% CI] | By seed 4901 / 4902 / 4903 |
|---|---:|---:|
| **U:N** — information beyond current confidence | **+.002671 [+.002312,+.003061]** | +.002686 / +.002578 / +.002749 |
| **U:S** — does the occurrence matter? | **+.002488 [+.002168,+.002830]** | +.002543 / +.002388 / +.002533 |
| **U:B** — drop-in gain | **+.002931 [+.002575,+.003301]** | +.002932 / +.002794 / +.003065 |
| N:B — current calibration alone | +.000260 [+.000162,+.000347] | +.000246 / +.000217 / +.000316 |
| S:N — displaced history vs current only | +.000183 [+.000110,+.000263] | +.000143 / +.000189 / +.000216 |

**Interpretation.** N's gain is calibration, not evidence of recall. The displaced mark also helped somewhat; S is not a zero-information control. U and S each had four fitted scalars and the same current inputs; S was fitted *with* its rotation, not surprised by a test-only intervention. Only the source of the one-byte field changed, never the source of B's 32-D vector. U's further advantage over S supports useful **occurrence-specific** forecast information here. The rotation preserves the follower and log-age bin, not all context. It does not prove a causal explanation of *why* some copies are useful.

The fitted coefficients on U's historical field were +.859/+.843/+.863 (S: +.120/+.158/+.180). At fixed current features, a larger old surprise raised the fitted log odds of copying: about +.21 per nat of old surprise on U's feature scale, before quantization. This conditional interpretation is not a gain estimate. Effects were measured with complete frozen predictors, not inferred from coefficients. PROBE had 512,670 hits and 11,618 misses per seed; the retrieved byte was correct on 242,928 hits (47.38%). The sham displaced 436,048 slots and changed 405,892 / 406,233 / 406,226 codes. Every probe document had both hits and misses and some changed codes. No written mark saturated in FIT or PROBE. The distinct coefficients mean U:S is not confined to positions with a changed code.

## What it cost

- After 41,445 prior full-document SHA exclusions, 24,339 eligible native TRAIN documents remained before reservation. CPU prep read 5,056,758 FIT source bytes, verified 1,048,832 window bytes and both shifts, and opened zero PROBE documents (20.34 s wall / 20.25 CPU s). Each seed got 1,048,576 FIT targets. PROBE later read 3,065,988 source bytes. Neither split reopens historical finals, and full-SHA exclusions do not imply passage-level deduplication. Do not compare absolute NLL on this diagnostic with other populations as if paired.
- U and S each need **one additional uint8 slot per input**, 4,096 bytes for a full window, including a never-read first slot. The host/B, byte index, temporary scores and other model state are not included in this payload size. Each U hit reads one mark alongside B's one value. No field becomes an observed text byte.
- Mean measured PROBE times per 4K window: shared host prefill **112.81 ms**, ordinary B head **87.88 ms**, shared current fields **4.04 ms**, shared U/S mark construction **0.52 ms**, U mark read/feature **0.07 ms** (S rotation + read **9.83 ms**), N/U/S score offsets **3.99/3.70/3.58 ms**. These are measured components of the batched diagnostic, not a streaming decode benchmark or an independent end-to-end speedup. Mean FIT moment accumulation across the three readouts was 0.71 ms/window after constructing features.
- One RTX 6000 Ada lease released rc=0, **≤287 card-s (4 min 47 s)** for all fits and probes, 304.37 s inclusive job wall, 551.11 owner + waited child CPU s. Peak device allocation/reservation 715,992,064/746,586,112 bytes; process peak RSS 2,603,440 KiB. Detached owner and driver, heartbeat renewal, terminal 1,152 real units and the 768-unit pre-probe fit boundary were independently witnessed. Earlier training of B is a common sunk cost. No retry or remote dispatch.

Local qualified source `bcff5643047683b575f984504620a59dae6b17a2`, public source/CPU evidence `31f8e868142e53c8c2d819d6c91e8b8eb7dbdb1f`. The pre-probe fitted map SHA is `67656fe3d1053a30519574c70e0c6258c4bced91089be171ffb1d832075156b9`. The receipt audit independently solved all nine FIT systems by Cholesky, reproduced all five paired bootstraps by document multiplicity, rehashed the three W-B checkpoint files and checked the terminal charge without accessing text or running a predictor. Aggregate SHA-256 `2acc7cba0b6e63de8c6db705a74499da04b08138ec3e0777f5a02b99903615a6`.

**Decision:** retain B as the local baseline; U is the candidate for confirmation with these exact coefficients. No re-fit, threshold tuning, reopening either U subset, or unconditional replacement of B on this TRAIN diagnostic.
