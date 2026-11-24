# Real span reads work; sampled program training is not yet cheap enough

**Decision: keep the readers and do not launch the quality comparison at this rate.** The generic controller sampled its own instructions, register operands, addresses and writes on real Pile bytes, and took one optimizer step. Nothing in the sampler selected a join for the model. **512 targets took 10.237 s** from training batch entry through the update on one RTX 6000 Ada (50.0 targets/s). At that measured rate, even one 1,048,576-target arm projects to **20,966 card-seconds** against **1,686 s** left in the X47 allocation. No development or confirmation windows were opened. This is a cost decision, not a negative finding about learned memory programs.

## What the model could read

The same `LOOKUP(register, field, reader)` now accepts four capabilities on either an admitted P100 archive or the session's own visible window: byte edges, complete bounded occurrences, exact span contents, and the following portion of the same document (up to 16 bytes). Returned locations are ordinary joinable `(source, version/session, document, start, stop)` keys. Contents/following return actual bytes, with a differentiable mean of their byte embeddings as the row value; following also supplies the first byte and a new span. No positional pseudo-value, separate opcode for a particular domain, or table of preselected W3 addresses. Only the observed transcript advances, after scoring.

- Archive occurrences use the existing resident B2 positional worker. Its raw interval limit is eight, **not** a top-eight sampler. A count >8 for the *full key* from the same m20 mount proves even the raw interval is too large, avoiding a locator call. A missing m20 node does not prove that. Otherwise the worker checks actual bytes and document bounds. A raw overcap can conservatively refuse even where some candidates would cross a boundary. No partial document ACL is implemented. Span reads validate the mount-bound fd identity and the span's source, version and document-relative extent.
- Live occurrences come from a separate native session per window. Values use only the observed prefix. Reading after an occurrence at the live frontier returns no row, not a byte or structural END. The live adapter proves a query fits the native response cap before issuing a potentially terminal overcap request. Invalid/foreign addresses are explicit refusals, not misses.
- Conditional sampling now packs legal opcode/operand alternatives for the active documents at each dependent choice. Masked Gumbel-max sampling returns chosen indices together per choice depth and retains exact conditional log probabilities for the gradient. Only chosen lookups reach a provider. This is **partial batching**: register/row views and candidate features still involve Python and many small tensor operations; dependent rounds and observations remain serial.

## What actually ran

Four SHA-verified W3 **training** documents, rows 4–7, first 129 bytes of each registered window; 128 real shifted targets per document. All four full document hashes and full archived raw windows were rechecked. Frozen C14 Transformer; no W3 plan features, answer-dependent filtering or forced program. One sampled AdamW update of the workbench; remaining-window losses and declared work prices provide policy credit, ordinary gradients train the value path. All action counts below are **before the first update**.

| Trace statistic | Observed |
|---|---:|
| Emitted predictions / distinct opcode sequences | 512 / 158 |
| Instructions / batched controller rounds | 1,719 / 641 |
| `LOOKUP` attempts / logically submitted keys | 327 / 326 |
| `JOIN` (exact equality) nonempty / empty | 31 / 9 |
| `RELATE` (learned compatibility) nonempty / empty | 56 / 3 |
| Writes / explicit recalls | 56 / 87 |
| Nonempty recalls from earlier bytes | 85; median lag 15, max 89; 42 at lag ≥16 |
| Emits selecting a register descended from an earlier-byte recall / exact join / `RELATE` | 30 / 11 / 23 (overlap possible) |

**Located vs used:** 22 live occurrence lookups returned 56 spans. Of 16 P100 occurrence keys, 14 were certified overcap by m20; the native positional worker saw two, refused one, and returned one verified span (1,994 LF steps). There were **nine successful span-value reads**, all live: seven contents, two following. All nine used addresses from an observed slice, **none used a locator-produced address**. All 19 requests to the P100 contents/following readers used foreign spans and were refused; zero P100 passage values entered this batch. The separate CPU positive checks read the actual contents and following bytes at all three known P100 hits; that is interface qualification, not sampled usage. A result register appearing in an emit's ancestry does not establish a beneficial join, a useful write, or a necessary path. Saved cells also enter the controller view without an explicit recall.

Sampled all-byte TRAIN NLL **1.49946** at initialization (individual documents in the receipt); not a held-out estimate, post-update score, or comparison to W3. The update was finite, pre-clip gradient norm 103.40. Opcode, operand pointer, pair query and writer received gradients. No post-update route-change or counterfactual benefit is claimed.

## Time and resource decision

| Synchronized batch wall, 4 documents × 128 targets | Seconds |
|---|---:|
| Forward, sampled native requests, scoring and objective construction | 6.287 |
| Backward including finite-objective check | 3.832 |
| Clip and AdamW update | 0.118 |
| **Total** | **10.237** |

Within forward: frozen backbone 0.252 s; host walls around workbench rebuild **2.569 s**, instruction/operand choice **2.010 s**, emit 0.143 s. All reader callbacks together **0.937 s**, of which P100 occurrences took 0.855 s. These diagnostics include waits/packing and are **nested host clocks**, not isolated additive CUDA kernel times. Live appends added 0.027 s. Peak PyTorch allocation **114.3 MiB** (not total system memory). Setup after device check and before the batch was another 0.857 s; imports, process startup, checkpointing and release are outside that clock. This is one teacher-forced batch with real reads and gradients, not a warmed steady-state throughput distribution or a serial decode timing.

The registered floor for **one** trainable arm alone would need ~622 targets/s with the remaining allocation (~12.4× this rate); controls and evaluation cost extra. Larger or better-packed batches might change that. The measured work points to packing register features and operand choices into shared device tensors and batching the backward graph across more independent documents, rather than optimizing just substring search. Mask known-incompatible address spaces before dispatch, without conflating the origin of a retrieved column with the address it contains. Test any faster path at fixed chosen traces, including gradients across bytes, before using its rate to budget a learning comparison.

**Receipts.** [Protocol](MEMORY_PROGRAM_PROFILE.md) `e75d3e2f2`, measured source `ab3fecf80`, [41 passing CPU gates](receipts/memory-program-p1/qualification.txt), [source/data/device identities](receipts/memory-program-p1/attempt1/identity.json), [observed costs and action counts](receipts/memory-program-p1/attempt1/profile.json), [offline budget check](receipts/memory-program-p1/verified.json). Physical GPU 0, detached owner `363454/554797312` (PPID 1), trainer `363540`; [watcher evidence](receipts/memory-program-p1/attempt1/survival.jsonl) observed 0→512 scored targets, renewed heartbeat and completed update. Released rc=0, ≤21 card-seconds; cumulative X47 ≤114/1800. Scratch checkpoint `output/memory-program-p1/attempt1/latest.pt`, SHA `9ece76090e3362326ef3d6ad08f5da86616f4c39aaea023692e6db5cdd6ff654`, not in Git. Raw inputs, per-query text, and corpus values are not in the result receipts.

Recheck the receipts without indexing, acquiring a GPU, or running a new experiment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x47_xsa_attention.memory_program_profile_verify
```

Add `--checkpoint output/memory-program-p1/attempt1/latest.pt` to check the retained checkpoint digest, model state, data identity and step on CPU.
