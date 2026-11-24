# Direct retrieval beat optional learned execution at matched Muon continuation dose

**Keep direct retrieval as the incumbent.** On the 3,670,016 fresh TRAIN targets after learning-rate selection, the direct retriever scored **1.26357** versus **1.27646 nats/byte** for DIRECT/ENTER (lower is better). The gap narrowed but did not reverse on the final 1,048,576 targets. These are next-byte losses scored before each update on exactly paired, previously unconsumed windows. They are **not** held-out scores of the final checkpoints.

| Registered measure | Direct | DIRECT/ENTER |
|---|---:|---:|
| Post-selection NLL (steps 129–1,024) | **1.26357** | 1.27646 |
| Last 256 updates, NLL | **1.25073** | 1.26085 |
| Last 256, NLL plus registered work price | **1.25206** | 1.26218 |
| Charged GPU seconds, incl. both LR candidates and checkpoints | **≤7,066** | ≤9,167 |
| Main + native CPU core-seconds in measured training segments | **10,801** | 15,856 |
| Whole-panel unique targets per charged second, by arm | **593.6** | 457.5 |

**What trained:** the retained 150M GDN2 and the compatible retrieval head in both arms; in the second arm, the entry gate and the conditional program/value network as well. Both selected the higher registered continuation LR, restored that step-128 checkpoint **with both optimizer states**, and continued for 896 updates on new documents. Each selected trajectory consumed **4,194,304 distinct targets across 32,768 document windows**; the losing LR candidate added 524,288 repeat target computations per arm. Total 1,152 physical updates/arm, 1,024 on the selected trajectory. The two arms started from common host/direct weights, but their sampled predictors were not identical at initialization. This compares training recipes, not isolated versus shared gradients.

**What the gate actually did:** in the selected H candidate it entered 7,535/131,072 times over the first 32 updates and 3,134/131,072 over the last 32 of LR selection. On the fresh post-selection stream it entered 10,936/3,670,016 times. In the **last 256 updates it entered just 75/1,048,576 times (0.0072%)**. DIRECT still queries memory: it learns trust over host prediction, archived suffix followers, document-local followers and latest local continuation. ENTER instead runs the sampled workbench; it does not first obtain the direct answer for free.

One descriptive check on the saved step logs, **not a separate decision criterion**: on 189 of the final 256 updates the program arm chose DIRECT at *every* position. On those same 774,144 targets the direct arm averaged 1.24908 and the program arm 1.25905. Thus even on batches with no executed ENTER, the two trained direct predictions differed in quality. Rare bad ENTER predictions on the final stream cannot by themselves explain the whole late gap. This conditional slice does not identify which training gradient caused the difference.

**Interpretation:** optional program execution did not buy an improvement over a working retriever, in raw loss, priced loss, or complete elapsed cost on this run. The learned gate mostly returned to direct retrieval. This experiment does not estimate the gain over a host-only continuation, determine the best optimizer, or rule out better program initializations or credit assignment. No held-out split was opened. No further arm was launched.

[Locked protocol](PROTOCOL.md) `0f0f257cb` · trained source `ee157fe25` · [comparison receipt](receipts/comparison.json), SHA-256 `16c190e90d556f7eefbf0a372a9aca1feee8fc04ae67e77bb65fd482dcd30dfe`. Both GPU-0 leases released rc=0, inclusive total **≤16,233 s (4 h 30 min 33 s)** of the separate six-hour allocation. `proc_bd13` witnessed PPid 1, advancing optimizer steps, renewed leases and terminal release. The post-release comparison checked both endpoint checkpoint digests and all post-selection batch identities. Private checkpoints and step logs remain under `output/x48-muon/{direct,program}/`; no tensors or raw text in Git. With the registered local artifacts present, rerun the CPU audit with `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .venv/bin/python -m experiments.expressivity_tasks.x48_muon_continuation.compare`.
