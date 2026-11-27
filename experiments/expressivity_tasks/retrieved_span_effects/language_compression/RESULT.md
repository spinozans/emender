# Faster indexed training; worse language compression on the fresh probe

**Keep LOCAL over either learned archive reader at this size and dose.** The Triton implementation made a full indexed optimizer update **1.60× faster than eager PyTorch** on the registered device meter. That made the experiment practical, not the reader useful: on fresh Pile text the indexed model needed **0.15294 more bits per original byte** than LOCAL. Both seeds lost. The exact reader also lost, so the feature approximation is not the whole problem.

## What was compressed

A 64-wide two-block causal byte Transformer trained from scratch. The BUILD archive was **128 × 129 raw bytes**, not the entire Pile: 16,384 positions with a real within-document successor. FIT and PROBE used separate documents, also distinct from BUILD and from earlier reserved cohorts. Each of eight fits saw all **1,048,576 shifted FIT bytes once**, in 1,024 updates. Seeds 5751/5752 had paired starts, optimizer schedules and target order across arms. Every fit was frozen and independently acknowledged before opening PROBE once. No checkpoint selection, tuning or filtering on lexical support.

The denominator is **524,416 original bytes** (128 fresh 4,097-byte excerpts): 524,288 next-byte predictions plus 128 first bytes at eight bits each. One excerpt can have repeated passages; whole-document hashing is not passage deduplication. No domain breakdown: the selection did not bind reliable domain labels. Results are for this small, once-through regime, not all language, archive sizes, model capacities or retrieval laws.

| Co-trained model | Ideal bits/original byte ↓ | Seed 5751 | Seed 5752 | Train ms/update¹ | Score s/M shifted bytes¹ |
|---|---:|---:|---:|---:|---:|
| LOCAL (causal local rows, no archive) | 3.59961 | 3.59739 | 3.60182 | 8.45 | 2.41 |
| COUNT (LOCAL + suffix distribution) | **3.59429** | 3.59641 | 3.59217 | 9.12 | 2.58 |
| EXACT (all archive rows, exact scores) | 3.73405 | 3.75160 | 3.71649 | 10.62 | 2.94 |
| INDEXED (all rows, exact matching intervals) | 3.75254 | 3.76121 | 3.74388 | 13.52 | 4.14 |

¹ Two-seed mean, complete fit loop or batched scoring on prepared packets; **not** serial decoding, matched-time quality or an arithmetic-coder benchmark. All four models contain 124,802 parameter scalars; only COUNT uses the additional 65-parameter mixture gate. They must share the trained model with the decoder; the non-LOCAL arms also share BUILD. No neural compressed files were emitted. The 16,734,208-byte materialized count table can be rebuilt from the raw archive.

| Comparison (positive saving is better) | Bits/original byte saved | Paired document 95% interval |
|---|---:|---:|
| COUNT vs LOCAL | +0.00531 | [−0.01177, +0.01940] |
| EXACT vs LOCAL | −0.13444 | [−0.14017, −0.12833] |
| **INDEXED vs LOCAL (primary)** | **−0.15294** | **[−0.17173, −0.14042]** |
| INDEXED vs COUNT | −0.15825 | [−0.16746, −0.14944] |
| INDEXED vs EXACT | −0.01850 | [−0.03521, −0.00955] |

Saving = comparator rate minus candidate rate. Resample whole probe documents together across arms and seeds (10,000 draws); these intervals do not quantify variation across a population of model seeds. COUNT's two point estimates beat LOCAL but its interval includes zero. A three-byte miss is not an empty read: **every query reads the entire archive**. On 400,728/524,288 positions (76.43%) there was a three-byte exact interval; 10,128,196/8,589,934,592 query-row pairs (0.118%) got exact scores. The remaining rows contributed through positive rank-32 feature scores. The result compares separately co-trained systems. The 0.01850-bit difference versus EXACT is not a fixed-weight estimate of approximation error, and the comparison doesn't identify which component of the joint corpus/local/no-read normalization caused the loss.

## Actual coded bytes for context

zstd level 3, one independent frame per excerpt, every frame decoded and byte-checked. Without a dictionary: **240,721 bytes (3.67221 bits/original byte)**. With the raw 16,512-byte BUILD archive as a shared dictionary: **229,196 bytes (3.49640 bits/original byte)**. The archive saved 11,525 frame bytes. If it had to be sent verbatim with this one probe, the total would instead be **245,708 bytes (3.74829 bits/byte)**. Thus a conventional codec could use this particular small archive even though the learned readers did not earn their cost. The zstd rates are actual frame sizes; the LM rates above are probability-based payload estimates excluding delivery overhead.

## What Triton accelerated

All 93 pre-access CPU/interpreter checks passed; 13 offline SM89 cases compiled within the shared-memory bound. On the leased RTX 6000 Ada, **19/19 device cases passed**, including the previously blocked 201-row backward launch. The new parent and final-normalization kernels differentiate through the current archive; no learned bank is detached or reused across updates. [Implementation and usage](README.md).

At 16,384 archive rows × 1,024 live predictions, width 64, rank 32 on invented bytes, median warm **complete indexed update** fell from **20.477 to 12.822 ms**. Incremental peak allocation fell from **129.51 to 60.37 MiB**. Max logit and clipped-gradient differences vs the eager implementation of the **same law** were `5.96e−7` and `3.73e−9`. The five timing blocks included forward, full backward through the rebuilt archive, clipping and fused AdamW, with the device synchronized. The invented input had 310 exact pairs per batch; don't extrapolate its speedup to different hit densities. This is not a claim of beating exact vendor GEMM. In the natural run, INDEXED took **1.27× as long as EXACT** and **1.60× as long as LOCAL** per equal-dose fit.

FIT/PROBE route preparation took 1.387/0.651 s once (excluding source extraction and archive construction). Absolute fit peaks, seeds 5751/5752, in MiB: LOCAL 581/815; COUNT 582/816; EXACT 838/1,028; INDEXED 866/867. These include resident packets. Route caches fill lazily and are shared across fits; use the paired warm meter above for the same-law incremental comparison. The pilot retained only final weights. No matched-wall-time quality checkpoints can be inferred from the timing columns.

One new no-wait lease, independently witnessed renewed heartbeat and advancing work; release verified, rc=0, **≤123 card-seconds**. This did not reopen the old failed meter or its budget. The terminal audit recomputed the paired results from sealed document loss sums, not a second model evaluation. [Protocol](PROTOCOL.md) · [Qualification](receipts/qualification.json) · [Result, seeds, prices and hashes](receipts/result.json).
