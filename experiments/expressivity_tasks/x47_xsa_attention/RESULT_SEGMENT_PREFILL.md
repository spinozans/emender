# Long prompts worked by carrying native FP32 state across short segments

**For the retained 150M streaming GDN2, use segmented prefill rather than ingesting the prompt one graph byte at a time.** Same model/precision, bounded arena, no external changes. With `segmented_prefill=True`, a prompt is ingested in consecutive ≤64-byte pieces; all 27 recurrent matrices and all three conv histories continue across each cut. The external short recurrence still sees at most 64 bytes per invocation. This qualifies a streaming function up to the tested **2048-byte frontier**. It does not make the failed S9 monolithic FP32 chunk kernel runnable, measure long-context quality, or move the memory-program policy's recent-byte pointer horizon.

[Protocol](SEGMENT_PREFILL_PROTOCOL.md) `f2fb9506f`, source `a5b2df37f`, 82 CPU gates. On the real host, compared a complete segmented 2048-byte prompt to first-byte prefill + 2047 actual graph steps in another slot. Separate no-arena native singleton cache checked through 129 observations and the 130th byte; also ragged 65/513, swapped order, reuse/survivor. Seven complete state/logit snapshots; maximum errors recurrence / q,k,v ≤7.63e-6, hidden 3.76e-6, byte log-probability 5.73e-6. All argmax matched.

**512 observed prompt bytes/client + eight actual responses.** 24 closed-load cells, 1,632 emitted bytes, both TRAIN prompt blocks, no future inputs. Every per-document byte and logical memory event digest matched its serial-input arm. Each arm encoded 519 observed tokens/client: serial 1 initial byte + 518 graph steps, segmented 8 real 64-byte pieces + 7 graph steps. Captured graph state/workspaces aren't recaptured or grown on a segment; persistent public+private arena unchanged at 229.61 MiB. CPU greedy/sampled workbench on the S8 packet; direct uses fixed archive/live branch backoff.

| Predictor | B1 serial / segment bytes/s | B16 serial / segment bytes/s | B16 serial / segment first-response median ms | B16 serial / segment median **after** first ms |
|---|---:|---:|---:|---:|
| Direct | 5.2 / 10.5 | 49.1 / 69.5 | 2508 / 2746 | 13.7 / 12.8 |
| Greedy program | 6.0 / 14.4 | 53.0 / 199.2 | 2270 / 500 | 21.0 / 21.2 |
| Sampled program | 5.0 / 9.8 | 45.9 / 115.3 | 2541 / 773 | 33.9 / 34.5 |

All six pooled rate gates passed (1.41× to 3.76×). This improves **prompt ingestion**, not the basic recurrent decode step; the seven subsequent intervals are similar. Tiny quantiles, including cold first-use B16 direct in block 0, are not SLAs. B16 segmented peak allocator ~864 MiB vs serial ~834 MiB. Rates include first host compute and complete selected reader/controller service; session/mount/prompt-observation setup is separately priced, not folded into generated-byte wall. No projection to other prompt lengths or an independently trained wide host.

[Offline metrics](receipts/memory-program-s10/verified.json). GPU 0 owner `1395172/556828827`, `proc_5140` witnessed actual advances and renewal, released rc=0, ≤90 card-s. X47 **≤1017/1800**, **783 s remain**. No updates, dev access or push.
