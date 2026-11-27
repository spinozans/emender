# Whole-corpus GPU meters: larger frozen read won; all-frontier clock blocked

These were **invented FP32 workloads** on one broker-leased RTX 6000 Ada at a time, after the [four-page natural panel](NATURAL_RESULT.md) had closed. They did not access or score language data. [Compact results, per-block timings, memory and terminal receipts](receipts/corpus_device_meters.json).

## Corpus attention at independent prediction frontiers

At public source `4a6b5d4b3ddd76f9ebed10c0e36009dc3e70fb9f`, **8/8 real-device cases passed**, including search/extension alignment, ragged reads, first derivatives and successive optimizer updates. The indexed Triton/eager implementations target the same mixed law; the exact GEMM reference instead computes exhaustive attention. Width/rank 32, 1,024-token vocabulary, 128-row leaves; immutable routes already bound. Warm, synchronized wall-clock medians, five blocks; include dispatch. Smaller is faster.

| Work | Exact GEMM | Indexed PyTorch | Indexed Triton |
|---|---:|---:|---:|
| Fixed bank, 16,384 corpus rows × 64 queries | **1.179 ms** | 1.950 ms | 1.312 ms |
| Fixed bank, 131,072 × 128 | 1.635 ms | 1.934 ms | **1.266 ms** |
| Complete current-weight optimizer update, 16,384 × 64 | **8.510 ms** | 14.992 ms | 13.338 ms |

- The larger batched read was **1.29× faster** than exact GEMM, with **6.24 vs 192.00 MiB** incremental peak allocated for the read. At the smaller size it used 0.69 vs 12.00 MiB, but GEMM was faster. These peaks exclude the allocated bank and are not whole-process footprints; the receipt includes allocator baselines. This is not a per-token generation clock.
- The indexed small-geometry training update was **1.57× the elapsed time** of GEMM (31.20 vs 24.77 MiB incremental peak). The complete step includes current-weight row/summary construction as applicable, encoding/decoding, loss, backward, clipping and AdamW. Don't use the larger frozen-read improvement as an end-to-end training claim.
- Refined 8,704 / 1,048,576 and 139,264 / 16,777,216 query-row pairs, respectively. Unrefined rows still contributed through the positive background. Mean absolute corpus-share difference vs exact: 0.02266 / 0.02318. Weighted read RMS difference: 0.001237 / 0.001213. Fidelity to an exhaustive read is not LM utility.
- Resident interval lookup was 0.034 ms per batch at each size (device IDs already resident); not the full CPU plan/packet pipeline. CPU endpoint index build 0.019 / 0.206 s; one-time query packet and plan using the device locator 0.493 / 0.095 s, not steady-state lookup latency. Independently witnessed, rc=0, positive release, ≤21 card-seconds.

## Packed document-window training: failed real-device gate

At public source `1debd28c4504f907d98a879db735514396f743c6`, **11/12** real-device tests passed; `test_leaf_statistics_and_independent_vjps[205-33-9-201]` failed during `_bvalue` launch for 205 rows, 33 feature channels, value width 9, leaf size 201. The compiled value-gradient program requested **102,912 bytes shared memory**, above the **101,376-byte** hardware limit by 1,536 bytes. This is a resource failure, not a numerical mismatch on that shape. CPU interpretation and offline assembly had not checked hardware launchability. The other fixtures, including actual model updates on smaller geometries, do not waive that gate.

**No all-frontier GPU timings were taken.** The [80.29 ms / 4,633 targets/s streamed update](corpus_windows/TRITON.md) remains a CPU result. The single-use meter stopped before its first witnessed completed work unit; no `watch-ack`, no successful completion, no automatic retry. Owner release verified with rc=1; charge ≤23 card-seconds, including the failed work. Do not infer a GPU speedup, peak footprint, or a passing hardware configuration from the missing clock.

Across the original meters (29 + 39), natural panel (36,959) and these two meters (21 + 23), terminal upper bound **37,071 card-seconds**; synthetic portion **112 card-seconds**. No further GPU run on this failed admission. A [subsequent CPU/offline correction](corpus_windows/SUBROW_RESOURCE.md) reduces the offending program's compiled shared memory to 29,184 bytes without changing which rows it differentiates. The later, separately admitted [raw-language pilot](language_compression/RESULT.md) passed this fixture on hardware and measured a complete-step speedup over same-law eager PyTorch at its own geometry; this does not fill in the missing times above.
