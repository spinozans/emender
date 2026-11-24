# A small Transformer over a rebuilt working context

**Architecture design; first loop implemented and one natural-byte join policy tested.** For the reusable controller that chooses its own operations and operands, see [learned memory programs](LEARNED_MEMORY_PROGRAMS.md). A fixed experiment should not define the platform's addressing rules. Give the Transformer a small workbench, not a 100 GB prompt. At each prediction it sees recent bytes and a few things it has chosen to bring into view: branches of an index, passages, bookmarks, or its own saved computations. It can rearrange that view, ask another question, and look again before predicting a byte. The same Transformer runs on each view. Discarding the view does not discard the history.

**First implementation:** [W1 runnable trace and measurements](../experiments/expressivity_tasks/x47_xsa_attention/RESULT_W1_WORKBENCH.md). Before training, the same small Transformer on CPU and GPU followed different branches when the archive changed and used a conditional write on the next byte. In the four-document fixture it produced the same byte distributions as direct suffix backoff, at higher cost. The [W2 current-byte join](../experiments/expressivity_tasks/x47_xsa_attention/RESULT_W2_CURRENT_JOIN.md) went one step further: two child answers combined with an older saved cue changed the byte being predicted *now*. Masking the cue or hiding which edge had which answer removed the advantage. A fixed CPU join matched all four answers at lower cost. Neither fixture is a claim of learned navigation or natural-language gains. [W3](../experiments/expressivity_tasks/x47_xsa_attention/RESULT_W3_NATURAL.md) tested a natural analogue on 4.19M Pile targets per arm with a pretrained Transformer and the real 100 GB XSA. The direct mixture was better on all-byte development NLL than the trainable two-child join; confirmation stayed closed. That is a result about this write/address policy, not about all rebuilt workspaces.

## What lives where?

| Object | Example | What survives a workspace rebuild? |
|---|---|---|
| Observed history | The actual document bytes up through `retry_`, with a growing XSA | All observed bytes. Append only after prediction. Never put a hypothetical byte here. |
| Mounted archives | A read-only Pile XSA, or a separately admitted project archive | The indexes stay open. Adding a compatible source changes the mount, not the model weights. |
| Bookmarks | “Project archive, pattern `retry_d`” | A logical reference to **all eligible occurrences**, not one file offset. Bind it to source, version, scope, and live session/watermark where applicable. |
| Computed records | “If the next byte is `d`, recall this vector” | An addressable, session-local value the model **wrote**. It is not a fact inserted into the archive. |
| Workbench | Recent byte embeddings, two branch responses, one recalled vector, a prediction slot | Nothing implicitly. An item survives because it is still in a source, has a bookmark, or was explicitly written. |

The service knows how to answer an index request. The **model's controller** decides which request to issue, which answer to keep, and when to stop. A host script that supplies both the first and second queries in advance tests the reader, not that controller.

## One prediction, in order

1. **Assemble a view.** The most recent observed bytes keep their order. Other slots are typed: a span has a source and position, a branch has a pattern and counts, a recalled record has an address. Moving a passage to slot 1 makes it visible; it does not make it the first passage in the document. Slot position and document position are different.
2. **Run the small Transformer.** Its prediction slot can attend to the whole view. Scratch slots may attend to each other in both directions: every item is already available *before this target*. This does not permit attention across different target times into future views.
3. **Emit or read.** To read, select a mount and bookmark or construct a pattern from visible bytes. `BRANCH(P)` asks *which bytes follow `P`, and how often?* A branch answer can occupy one slot as a projected 257-bin histogram; retain the exact labeled counts for choosing edges. To follow byte `b`, request `BRANCH(P+b)`. Use positional search only when the computation needs a particular occurrence, its surroundings, or a value stored there.
4. **Rebuild and repeat.** Replace an obsolete slot, insert the answer, or recall a different record. Run the same weights on the new view. Bound the number of slots, requests per round, rounds per byte, and persistent records. After the last permitted read, the model must predict; no unbounded search. An overflow or unavailable operation is a status, not license to silently truncate an exact answer.
5. **Score, then observe.** Produce a distribution over 256 bytes. In training, score the actual byte *before* appending it; in generation, append the emitted byte. A write based on a speculative read can persist, but stays labeled with the condition under which it applies. Reset session state at independent document/window boundaries.

All reads obey the source's scope **before** counting or ranking. Unsupported exclusions must be refused, not approximated by filtering returned examples. No match or continuation crosses a document boundary. `END` is a structural outcome, not a byte (`0x00`, `0x1e`, and `0xff` are all data); an unfinished live prefix has no confirmed END. An unmaterialized archive node is not proof that its string never occurs. If the controller backs off to a shorter suffix, the answer names the suffix actually queried.

## A program that should work before training

Here is a small, falsifiable starting behavior. It is a reference policy, not a claim that prefetching is optimal.

- **Read the current suffix** in the live source and each admitted archive. Start with the last eight observed bytes (or the whole prefix if shorter); try four and then two only if shorter and needed, within the read budget. Keep each source's evidence separate. For counts `h` with positive byte mass, its prediction for byte `b` is `h[b] / sum(h[j] for j in bytes)`; preserve END separately. Mix the available predictions with fixed source weights and a positive full-support local fallback. Do not pool raw counts and let the largest corpus win by size. With no byte evidence, use the fallback alone.
- **Expose alternatives, then follow one.** For the first witness, retain the two best supported outgoing byte handles of one archive locus. Initialize the action scores from counts (fixed byte-order tie break); use the selected byte, not a host-supplied constant, to form the next request. If the child has byte evidence, write its normalized vector with the parent handle and selected byte as its condition. On a matching byte next step, the controller can select that child locus and use the record **in place of**, not in addition to, a fresh read at the same archive locus. On a mismatch, it is not evidence about the observed suffix. Other bookmarks need not be destroyed.
- **Make this a model initialization.** For example, score each supported edge by `log(count) + learned_correction(workbench, edge)`, with the correction zero at birth. Softmax gives the exact relative counts; argmax chooses a handle to extend. Give the controller typed request/write outputs and a nonzero path from the read vector to the byte distribution. The record writer can initially copy the normalized vector; learning can change what it writes from the rest of the workbench. Native code executes the selected operations; it does not choose the answer on the model's behalf. Initializing a generic `READ` bias is not enough. Hard address selection also needs an explicit credit mechanism if it is later to be learned; next-byte loss does not differentiate an integer address.

### Worked trace: two reads, then one new byte

Take four fixture documents: `retry_de`, `retry_de`, `retry_da`, `retry_c`. The live document so far is exactly `retry_`. Here is the archive part of each rebuilt view; probabilities shown are the archive contribution, before mixing with the local predictor.

| Observed bytes | What the small Transformer sees | What it does |
|---|---|---|
| `retry_` | Recent bytes; no archive answer yet | Ask `BRANCH("retry_")`. |
| `retry_` | `d:3, c:1` | Keep the parent distribution and both handles. Select `d`; ask `BRANCH("retry_d")`. |
| `retry_` | Parent plus child answer `e:2, a:1` | Write “after this `d`: ⅔ `e`, ⅓ `a`.” Predict the **current** byte from the parent: ¾ `d`, ¼ `c`. |
| `retry_d` **if `d` was observed** | Updated recent bytes and the recalled vector | Use ⅔ `e`, ⅓ `a` for the **following** byte without reading that child again. |

Both reads happened while the transcript still ended in `_`. If the actual byte is `c`, the `d` record stays off the prediction path. Exploring `d` wasn't emitting `d`, and observing the child answer wasn't grounds to predict `e` one byte early.

Prefetching is just the easy-to-check first program. In a longer document the workbench could put an earlier mention of *milliseconds* next to the `retry_` fork. A learned controller could inspect a project passage to distinguish *count* from *delay* and use that evidence to revise the **current** probability of `d`. Or it could write that interpretation for a later question. Neither requires copying the passage into the transcript. The initializer need not understand the word *milliseconds*.

## What is cached, and what is parallel?

This is **not** dense attention over all history with a clever KV cache. Older material matters only if it is represented in a visible slot or affects an index answer. The workbench can put retrieved material at the front, replace its middle, or drop most of yesterday's view: recompute the whole small view rather than reusing stale contextualized K/V. What remains warm is the open index, its native caches, and any explicitly saved records. Reuse an immutable read only under matching source/version/scope/value semantics. A saved live result is evidence about its old prefix, not a fresh query after an append.

Slots in a known view run in parallel within each layer. Independent requests to several mounts can be batched. But if the answer about `retry_` determines the query `retry_d`, those reads are **serial**, as are successive generated bytes. With at most `R` read rounds there are at most `R+1` Transformer passes per byte (initial view plus one per answer batch). A small view bounds neural work per pass; it does not bound native lookup time or guarantee faster decoding.

## Next decision: prove the loop, then price it

1. **Trace gate, no optimizer.** Use a tiny live prefix and an authorized fixture archive whose branch counts can be counted by hand. Log visible prefix, views, controller choices, requests, writes, and byte distributions. Swap the parent counts at identical visible bytes: the second request must change. Hold bytes, handles, and read schedule fixed but erase the saved vector: the later prediction on the matching path must change; the mismatch path must not use it. Check empty/END-only, unavailable, source-removed, and reset cases. This proves a functioning dependency, **not** an LM gain.
2. **Price the right operation.** If the trace passes, compare rebuilt vs static/append-only views on the same mounted sources and request budgets. Measure warm **BRANCH**, packing/transfer, each small-model pass, and synchronized total latency per emitted byte; distinguish serial decode from independent batched throughput. A positional LF walk and a device-only attention kernel time are neither of them this measurement.
3. **Only then decide on training.** Keep direct suffix backoff and the frozen initialized program as controls. Re-reading the child on the next byte gives the same archive evidence as the toy conditional write; the model must earn its extra machinery. A subsequent preregistered comparison should use ordinary all-byte NLL and measured compute on matched, document-disjoint data, with windows long enough for a write to affect a later loss. For the first natural training comparison and its limitations, see the linked W3 result.

For the native capabilities that exist today, the measured positional and device paths, and the distinction between branch counts and per-occurrence neural values, see the [XSA memory system map](XSA_MEMORY_SYSTEM.md).
