# Changing client membership worked; long positional reads still stalled the synchronous tick

[`ByteService`](../../../ndm/memory_program/byte_service.py) is an in-process, bounded, first-party serving interface, not a network daemon. One owner thread, private live readers/cells per client, one admitted immutable P100 domain, one 150M FP32 graph store. Open/step/cancel/release use opaque tickets. A mixed tick prefills only new clients, advances already served clients by their **last actual byte**, then runs direct retrieval or the small CPU workbench. Ambiguous native/host failures discard the owner rather than continuing from uncertain state. A prelease CPU test caught a real bug where clearing a consumed prompt changed the continuation list mid-tick; grouping now snapshots membership first. [Registered S12](ARRIVAL_SERVICE_PROTOCOL.md) `bff5ac958`, gated `838e938a9`, 88 CPU tests, including partial-append fault cleanup.

**Four fixed offered-load trials passed the lifecycle gate.** Independent producer offered 32 descriptors/trial: either 4 every 60 ms, or two bursts of 16 separated by 50 ms. The owner didn't make the next offer wait for a response. Peak offer jitter <0.8 ms. All trials reached 16 live bindings, max observed waiting 20 (cap 32); 16–24 offers landed *inside* an ongoing prediction tick per trial. One new native process per admission, all **128** reaped; both immutable workers exited after each domain shut down. Across four trials 112 clients finished 12 bytes, 16 were canceled after exactly three, all stale cancellation tickets refused and all host slots scrubbed. No retained raw prompt/trace in benchmark receipts. Logical erasure only, not physical GPU sanitization or unrelated-user isolation.

**768 independent closed oracle bytes + 1,392 actual arrival bytes.** Every complete reference stream matched S11 at 12 bytes. All 128 dynamic stream **prefixes**, including canceled programs, matched the corresponding oracle bytes and normalized instruction/reader/event digest. Per trial exactly 1,024 prompt tokens + 316 new-byte host steps for 348 responses; no fabricated placeholder steps. Sampled here means stochastic instructions, argmax emitted bytes.

| Offered schedule / predictor | Bytes/s, first offer to last release | p50 / p95 offer→first byte ms | p50 / p95 interresponse ms | p95 sojourn ms |
|---|---:|---:|---:|---:|
| Spaced / direct | 271.5 | 387.7 / 775.0 | 14.5 / 104.4 | 982.0 |
| Burst / sampled | 129.7 | 1643.8 / 2163.4 | 38.1 / 1146.6 | 2633.3 |
| Spaced / sampled | 118.2 | 1524.8 / 2182.0 | 37.4 / 1177.4 | 2644.7 |
| Burst / direct | 363.2 | 402.6 / 723.0 | 13.5 / 95.9 | 908.3 |

Rates include queue, admission, host, native, delivery and retirement, exclude prebuilt store and 0.73–0.94 s common domain setup per trial. Whole trial+setup/teardown 1.85–3.97 s. Peak framework allocation through post-release checks 1,307 MiB, not just steady model payload. One short trial per schedule/predictor is a **functional open-offer qualification**, not sustainable arrival capacity, network QPS, deadline compliance or a latency SLA. Each sampled client's tick still waits behind the slowest chosen native lookup in that batch. See [full timing distributions, queue delays and counts](receipts/memory-program-s12/verified.json).

GPU 0 owner `1432985/556967151`, `proc_e5ac` witnessed actual work/renewal, released rc=0, **≤33 card-s**. X47 ≤1153/1800, **647 s remain**. No learning, holdout or publication.
