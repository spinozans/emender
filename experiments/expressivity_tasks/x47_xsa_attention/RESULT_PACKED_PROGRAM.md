# Packed controller: same memory program, 162 fresh training targets/s at B32

**Keep the packed path; do not start the quality comparison.** On the same four documents, a fresh sampled update fell from **10.480 to 6.653 s (1.58×)**. With 32 documents it reached **161.7 targets/s**, 3.23× the historical [P1](RESULT_MEMORY_PROGRAM_PROFILE.md) rate. At the best measured rate, one 1,048,576-target arm projects to **6,483 s**, against **1,620 s** left. Those numbers exclude controls and evaluation. This is an engineering improvement, not evidence that a trained join improves prediction.

## What changed and what didn't

[`PackedMemoryPolicy`](../../../ndm/memory_program/packed.py) accepts the reference `MemoryPolicy` state dict. It flattens rows and fields from the active documents, reduces by page/row/column index, and scores the legal operands together at each dependency depth. A masked batch can contain, for example, a `JOIN` field choice in one document and a `SLICE` length in another. It reuses differentiable features of unchanged saved cells within the rollout and batches the `EMIT` readout. No permission, key equality, join multiplicity, row cap or action prior is decided by these temporary feature indices. The original `Frame` still checks budgets and executes only the chosen operations. An unavailable reader response is still different from zero results. There is **no new address-space mask**, preselected join, or cache of future requests. Subsequence choices, native lookup, and observation across bytes are still dependent.

## Equivalent at fixed programs

- [47 CPU tests passed](receipts/memory-program-p2/qualification.txt) before the lease. A missing `Instruction` import in an early CPU-only smoke was fixed before the source freeze. On CPU and CUDA we replayed mixed programs through a scalar reference with nonzero scoring weights, including empty/refused reads, duplicate bindings, a delayed write/read and a many-to-many join. On [CUDA](receipts/memory-program-p2/attempt1/device-parity.json) all **103 conditional distributions** (opcodes and dependent operands) agreed within `1.44e−6` max absolute log probability. The later-byte loss had adjoint norm `0.146117` into an earlier saved value; max error `7.46e−9`.
- On the actual 4×128 TRAIN batch we recorded the reference's **sampled** instructions and operands, then reissued them against fresh native sessions at the original weights. We compared the ordered logical traces (including refusals), actual coalesced native-facing reader calls, and **all** returned register schemas and ordered bindings. Only actual live-session nonces were renamed, bijectively by document and window. Exact match: **512 decisions, 297 provider dispatches, 1,128 intermediate pages plus 16 endpoint cells**. All 15,766 log masses matched exactly; 756,768 numeric value components differed by at most `6.56e−7`.
- Across all 131,072 byte log probabilities, max error `3.82e−6`; across all 512 joint policy log probabilities, `3.82e−6`. Mean NLL and the score-function surrogate agreed; largest pre-clip gradient error `3.82e−6` across 35 parameter tensors, identical unused-parameter masks. [Natural parity receipt](receipts/memory-program-p2/attempt1/natural-parity.json). Forced replay took 6.447 s including backward, **no optimizer step**. It is not counted as fresh throughput. These checks are numerical, not bitwise equivalence for every possible session or precision.

## What fresh sampling cost

Frozen C14 and the same P100/tape edge, occurrence, contents and following readers as P1. B4 used W3 TRAIN rows 4–7; B32 used rows 0–31, including B4. Full document/window hashes and source identities checked, first 129 bytes per registered window, no padding or cross-document state. No W3 join-plan inputs. Each cell started from the P1 weights and sampled fresh programs with its own seed-7412 generator. All real shifted targets were scored before append. Each fresh cell took **one** price-aware delayed-credit AdamW update, not three successive updates of the same model. The two fresh B4 cells happened to produce identical full event digests, not merely similar opcode counts. All activities below occurred before each cell's update.

| Fresh teacher-forced batch | Ref B4 | Packed B4 | Packed B32 |
|---|---:|---:|---:|
| Targets | 512 | 512 | 4,096 |
| Setup outside batch, s | 0.772 | 0.642 | 0.554 |
| Forward incl. backbone, native requests, objective, s | 6.168 | 3.621 | 19.473 |
| Backward incl. finite-objective check, s | 4.212 | 3.014 | 5.829 |
| Pre-clip gradient snapshot, s | 0.013 | 0.010 | 0.019 |
| Clip + update, s | 0.087 | 0.008 | 0.004 |
| **Synchronized batch through update, s** | **10.480** | **6.653** | **25.325** |
| **Targets/s** | **48.9** | **77.0** | **161.7** |
| Peak PyTorch allocation, MiB | 115.9 | 149.8 | 1,111.3 |
| Executed instructions / controller rounds | 1,719 / 641 | 1,719 / 641 | 14,348 / 768 |
| TRAIN NLL, nats/byte | 1.49946 | 1.49946 | 1.58601 |

One timed batch per geometry, in registered order. Fresh mounts/models, **not** cold OS/CUDA caches. The reference went first (backbone 0.159 vs 0.003 s in packed B4); that is part of the observed contrast. No steady-state error bars, held-out NLL, or single-document decode rate. B32 vs B4 changes the document mix and number of reads, not just the kernel width. Do not add the three exposures as distinct training data.

**Where the time went:** at B4, rebuild/choose host walls went from 2.768/2.269 s to 1.267/1.330 s; provider callbacks stayed near 0.4 s. At B32, 8,941 newly encoded pages (129,378 rows) and 218,987 legal operand candidates were processed. Rebuild/choose/emit took 2.931/3.041/0.630 s. Page encoding (1.301 s) is *inside* rebuild, operand descriptors (0.273 s) *inside* choice. All provider callbacks took **11.198 s**, of which **10.610 s** was `P100/occurrences`. The worker reported 10.142 s of locate work over 15 LF rows / 57,031 LF steps; that time is *within*, not in addition to, the callback wall. Appends took 0.146 s. These are nested host clocks including waits, not isolated resident GPU kernel times.

**What B32 actually used:** 933 distinct opcode sequences, 2,865 `LOOKUP` attempts, 2,878 logically submitted keys before cross-document coalescing. Of 485 exact joins, 372 nonempty, 106 empty, seven refused over cap. Of 471 `RELATE`s, 444 nonempty, 14 empty, 13 refused. There were 481 writes and 529 nonempty prior-byte recalls (median lag 15, max 111). One successful live `contents` instruction actually used a **locator-produced** address; the other 91 successful live `contents` and all 61 successful `following` instructions used slice addresses. P100 occurrence callbacks processed 140 keys: 112 m20-certified over cap, 18 native over cap, six empty, four positive (15 spans). **No archive span value entered this batch**: all 181 submitted archive `contents`/`following` keys were foreign. Returning a span, reading a value and improving a prediction remain three different claims.

## Decision and evidence

Physical RTX 6000 Ada **0**; detached owner `884471/555541964`, PPid 1, trainer `884569`. The independent [watcher](receipts/memory-program-p2/attempt1/survival.jsonl) observed renewed heartbeats, advancing scored targets, and the last B32 update. [Lease](receipts/memory-program-p2/attempt1/initial) → [release](receipts/memory-program-p2/attempt1/released) `1795489431..1795489496`, rc=0: **≤66 card-seconds** including setup, fixture, replay, all updates and teardown. X47 cumulative **≤180/1800**. The fresh checkpoints were checked offline; neither weights nor observed text are in the Git receipts.

The remaining budget requires **647.3 targets/s for one arm alone**, about **4.0×** the measured B32 rate. As a scale check, even subtracting the *entire* 11.198 s provider wall from this batch gives only ~290 targets/s, not a measured alternative implementation. Similarly, speeding the controller alone would leave the real positional service bill. Further engineering should price faster verified locate or overlap of **already chosen** requests across independent documents, alongside the 5.829 s backward and remaining controller cost. Do not switch to guessed positions, truncation or a static address plan under an equivalence claim. No quality run, W3 confirmation, new panel, or second seed.

[Registered protocol](PACKED_PROGRAM_PROTOCOL.md) `ec1b91d88`; [measured source](receipts/memory-program-p2/gated-ref) `e2a8c0f4c`; [B4 reference](receipts/memory-program-p2/attempt1/reference-b4.json), [B4 packed](receipts/memory-program-p2/attempt1/packed-b4.json), [B32 identity](receipts/memory-program-p2/attempt1/packed-b32-identity.json), [B32 packed](receipts/memory-program-p2/attempt1/packed-b32.json), [verified costs](receipts/memory-program-p2/verified.json). Local offline audit:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python \
  -m experiments.expressivity_tasks.x47_xsa_attention.packed_program_verify
```

Add `--checkpoints output/memory-program-p2/attempt1` to verify all three step-1 checkpoint files and optimizer states on CPU.
