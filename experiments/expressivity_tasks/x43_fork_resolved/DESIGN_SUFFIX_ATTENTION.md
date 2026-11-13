# The suffix array of the model's own context as an accessible attention object

Design and feasibility proposal, 2026-11-13. Operator's idea; no implementation,
trained result, transcript benchmark, or quality gain is claimed here.

## Revision 2 — operator-corrected core (2026-11-13; supersedes the framing below)

The operator corrected the framing on 2026-11-13. Where Revision 2 and the sections
below disagree, **Revision 2 governs**.

### R2.1 MEM is a string kernel, not a next-byte lookup

The operator is a **maximal-exact-match string kernel** in the classical sense used by
read-mapping systems (BWA-MEM, minimap2): a long MEM between the current context and a
window is *evidence that the window is a similar context*, not a promise about the next
byte. The kernel returns a **weighted neighbourhood**; the neural block decides what to
extract from it (contents, regions, continuations, embeddings).

Consequence: the frequency of exact repetition in any particular corpus is **not** the
argument, and the depth-support statistics of §5 must not be used as one. They describe
which corpora are repetitive; they do not bound what the operator can find, and they say
nothing about the value of a neighbourhood once found.

### R2.2 The operator chain: seeds → chain → correspondence → fork

| stage | mechanism | status |
|---|---|---|
| seeds | `xsa seeds`, adaptive length solved for a target multiplicity | exists |
| **colinear chaining** | sparse DP over anchors, ordered/gap-consistent, chain score | **prototype built** (`chaining.py`, 16 tests); measured in `RESULT_CHAIN_COVERAGE.md` — the DP is instant, the limit is the anchor-set SHAPE the CLI returns |
| correspondence | the chain yields context position ↔ corpus position alignment | follows from chaining |
| **fork / variable site** | branch histogram at the aligned site with multiplicities | exists (`xsa branches`; `xsa forks` for the fork matrix) |

Without chaining the operator is a bag-of-seeds ranker: it learns "this document shares a
string with me". With chaining it is a **positional pointer** — it knows where the
correspondence is, hence where the variable site is, hence which corpus position answers
for the current one. That is the difference between retrieval and attention.

### R2.2a Measured (2026-11-13): the signal exists; the anchor feed is the blocker

150 byte-uniform positions per panel, 32-byte seed band (`RESULT_CHAIN_COVERAGE.md`):

| metric | transcript | Pile |
|---|---|---|
| L=12 context suffix occurs >=2x | **36.7%** | 82.7% |
| abutting correspondence (d16/d32) | **37.3%** | 82.7% |
| **fork top-1 = actual next byte** | **58.9%** (chance 15.0%) | 75.2% (chance 14.4%) |
| chains with >1 anchor (d32) | **0/150** | 11/150 |

Two things follow. The fork carries real signal when a correspondence exists —
3.9x chance on transcripts, 5.2x on the Pile — so the operator is not vacuous on
conversational text. But multi-anchor chains are essentially absent, and the cause is
not the DP: `xsa seeds --limit N` returns the first N rows of the seed's suffix-array
interval, which share a query offset and therefore **can never chain to each other**.
The fix is an anchor-table output shape (diverse anchors, distinct query offsets,
positions), requested in `SXGC_COLINEAR_CHAINING_ASK.md`.

### R2.3 The guarantee is completeness, not statistical support

Seed-based search is **complete by construction**: for any true correspondence containing
at least one exact seed of length ≥ L within the accepted transformation class, chaining
cannot miss it. Precision is then supplied by the chain/support filter. The only losses
are enumerable by design: (a) no exact stretch ≥ L, or (b) the correspondence lies outside
the accepted class. **L and the transformation class are the two knobs, and both are
registered parameters, not heuristics.**

### R2.4 One operator was wrong: it is an additive algebra

`ATTEND_SELF` alone is a design error. The general object is a measure-valued operator,
and operators compose by summing measures:

```
O(x) -> { (j, w_j, v_j) }   bounded candidates, nonnegative weights, value vectors
      + A_O                  exact aggregates (mass, distinct-region support, entropy),
                             independent of any truncation

a_o(j) = alpha_o * w_o(j) / sum_{j' in cand_o} w_o(j')      per-operator normalisation
out    = sum_o sum_j a_o(j) * v_j                          additive readout
```

Siblings, all on the same interface:

- **ATTEND_SELF** — universe: visible context regions; weights from local rarity and
  region proximity/recency.
- **ATTEND_WORLD** — universe: corpus documents; weights `ln(D/df)` over *distinct*
  seeds, region-clustered.
- **ATTEND_STORE** — universe: the slot store; weights from slot strength (the existing
  Emender memory branch expressed in the same algebra).

Three rules make the algebra safe to extend:

1. **Per-operator normalisation, not joint.** A joint softmax lets one strong operator
   starve the others. With per-operator normalisation plus a learned `alpha_o`, each
   contribution stays bounded and comparable, and truncation only drops candidates — it
   can never make an operator look *sharper*. `A_O` carries the exact mass, so truncation
   is always disclosed.
2. **`alpha_o = 0` is exactly a no-op.** Every operator is ablatable by construction, so
   contribution is measured rather than argued.
3. **Empty is a value.** An operator with no support contributes exactly zero mass; the
   neural path still predicts. Empty support must never renormalise into noise.

A new operator is a four-item contract: position space + value source; a weight rule
defined on the *eligible* universe with source-document exclusion; exact aggregates
independent of truncation; a nonnegative learned scale initialised small (so `alpha=0` is
an exact ablation). Ties sort by (layer, document, position) for determinism.

### R2.5 The accepted transformation class is a registered parameter

"Transformations of it that you would accept" defines the kernel's equivalence class:
byte-exact admits everything losslessly but misses case/whitespace/format variants;
case-folded or normalised classes generalise but can merge distinct contexts. Because it
is a *class* and not a heuristic, it is declared, hash-bound, and ablated like every other
registered choice.

### R2.6 Linear in state: chunk-scan is the performance path

The step is affine with a diagonal linear part and rank-1 injections:

```
s_{t+1} = D_t * s_t + e_{i_t} (x) delta_t
```

Composition is associative and **factorises over slots** (the linear part is diagonal in
the slot basis), so the sequence chunk-scans exactly as chunked linear attention / the
GDN2 delta rule does: intra-chunk sequential (hard routing preserved, exact), inter-chunk
parallel prefix over the composed diagonal products and injected values. Serial depth
drops from O(T) to O(T/C) with no semantic change.

**What it buys and what it does not**, stated so the measurement decides:

- It removes *serialisation*. If the deficit is latency-bound small-matrix work — which
  the 2.13M regime (13 steps/s) suggests — chunking wins materially.
- It does **not** reduce per-token read/write volume (match over B slots, store traffic;
  compulsory traffic measured at 180.08 MB/head). If we are bandwidth-bound — which the
  T1024 profile (718.65 ms/head, 13,802 spills) suggests — only a smaller state or higher
  arithmetic intensity helps.
- The match still needs the state, so intra-chunk work stays sequential. C is the knob.

---

## Architectural decision

Make the model's own input an addressable exact-match object. The primary memory
is a dynamic XSA of the visible context bytes. The secondary memory is the
immutable pile-head100 corpus XSA. A single ATTEND_SELF operator exposes both
through suffix intervals, positions, continuation counts, rarity, and regions.
The neural block learns what to do with those answers. It need not learn an
induction circuit merely to discover that an earlier input span matches.

Start as an additional residual branch alongside the existing Emender memory
branch and MLP. Replacement of learned attention is an experimental arm, not a
prerequisite. Exact byte matching provides no semantic-similarity guarantee.
The proposal is a dynamic instantiation of the weighted interval-set model in
[THEORY_INTERVAL_OPERATORS.md](THEORY_INTERVAL_OPERATORS.md), Part I.

## 1. Object, ownership, and causal semantics

Let C[0:t) be exactly the bytes visible before predicting byte C[t]. In this
program tokens are bytes, vocabulary 256: a 1M-token context means 1,000,000
bytes. A million subword tokens is a different byte budget and must be measured
as such. Store the original bytes, not token IDs, normalized text, or embeddings.

The context object consists of immutable indexed segments, an append delta,
and an out-of-band manifest of logical documents, messages, regions, visibility,
and provenance. A conversation is one explicitly declared logical document;
message/tool boundaries are regions. Independent training documents remain
separate: no match, shifted target, or carried model state crosses their
boundaries. An API may explicitly mark additional hard boundaries inside a
conversation. No byte, including literal 0x1E, defines identity.

For a requested depth d, use forward-order P=C[t-d:t], clipped at the logical
document start. Backward search consumes these bytes right to left. For layer
L, I_L(P)=[lo,hi) contains suffix starts whose prefix is P. Its raw width is
hi-lo. Physical indexes may contain boundary-crossing matches or future bytes;
those are never automatically eligible evidence.

For the local layer define the eligible set

    E_local(P,t) = {p in I_local(P): p+d < t,
                    [p,p+d+1) lies in one visible logical document,
                    p is live and allowed by the visibility/provenance policy}.

The strict inequality means the candidate's continuation byte is already
observed. Overlapping occurrences are allowed if this condition holds. The
current suffix ending at t is excluded. A prefix index still contains that
suffix; a full-transcript index additionally contains future occurrences. Both
need the same explicit causal filter. A query at depth zero returns EMPTY
rather than a universal interval.

For the static layer, remove **every occurrence belonging to the evaluated
source document**, identified by the manifest's corpus document ID, before any
support, depth selection, maximality, or fan-out test. Also exclude matches or
continuations crossing document ends. Unknown source identity in an in-index
held-out evaluation is an error, not permission to retain self support. A
geometrically out-of-index document can carry a verified disjointness receipt.
Past local occurrences in the same conversation remain legal causal memory;
excluding the entire conversation locally would defeat ATTEND_SELF. These are
different exclusion policies, both applied before deciding support.

Return both raw width and eligible width m=|E|. The exact branch histogram is
H[a]=sum(p in E) 1[text[p+d]=a] for a in 0..255; sum H=m. End-of-document counts
may be separately reported, but never treated as a byte or a known target.
Any maximal supported suffix is selected using these **filtered** counts.
The failure in [RESULT_C14_IN_INDEX_LEAKAGE.md](RESULT_C14_IN_INDEX_LEAKAGE.md)
was specifically selecting on unfiltered source-inclusive support.

The static and local arrays are a **logical merged search object**, not a
per-turn physical rewrite of the 100 GB spine. An interval handle is a tagged
pair/list of layer-segment intervals; the same P is searched in each. Counts
combine only after eligibility and provenance deduplication. Copying a corpus
span into chat must not count twice as independent evidence when its origin is
known. Preserve layer-separated counts even when forming a combined answer.
Physical cross-LCP merge is for local context segments; it is unnecessary to
merge the spine with each turn.

Store bytes, XSA/BWT rank/locate structures, boundary starts, live ranges,
provenance, segment hashes, and index generation. Recompute query intervals,
eligible histograms, depth choices, and bounded candidate summaries; cache only
under a complete generation/policy/query key. Exact filtered counts need either
full enumeration or a document/position range-count augmentation. Document
rank/select alone does not provide filtered suffix-interval aggregation. The
initial exact reference can enumerate; the fast augmentation is engineering
work, and its cost must appear in benchmarks.

Document identity is grounded in
[SPINE_DOCUMENT_IDENTITY.md](SPINE_DOCUMENT_IDENTITY.md) and
[sxgc RESULTS_PILE_DOCMAP.md](../../../../sxgc/RESULTS_PILE_DOCMAP.md): measured
16,119,767 documents cover 100 GB, including a partial tail. There are 6,979
internal 0x1E bytes; separator count equals documents owning a separator plus
internal separators. `xsa corpus --pile-map` uses shard records, not delimiter
inference. SXI2 v10 member m15 can embed starts; `xsa docstarts verify` checks
rank/select parity. Preserve source/index hashes and document-map identity.

## 2. ATTEND_SELF primitive and block integration

Proposed interface (not an existing xsa command):

    ATTEND_SELF(context_generation, t, source_identity,
                depths_or_adaptive_policy, min_support, target_fanout,
                max_candidates, max_regions, visibility_policy)
      -> {selected_depth, layer_intervals, raw_widths, eligible_widths,
          exact_branch_histograms, entropy, distinct_document_support,
          candidates[(layer, doc, byte_start, continuation, region, score)],
          regions, truncation_flags, provenance, status}

Status distinguishes EMPTY, OK, and ERROR. EMPTY is normal; corruption,
identity mismatch, and inability to compute required exact counts are errors.
Candidate lists are bounded; widths and histograms are not estimates derived
from that bounded list. Expose truncation and retain exact count metadata.

First query the local layer at depths 16/8/4/2 (proposed defaults, not an optimum).
Choose the deepest suffix meeting min_support after filtering; adaptive mode
solves length for a target multiplicity. If the local result is too thin under
a registered threshold, query the static layer with the same operator. Preserve
an explicit local-empty/static-empty distinction. Fixed-depth diagnostic mode
never backs off, so coverage remains interpretable. Deep traversal remains
available for identifiers and long exact repeats; do not pay depth 1024 at
every position by default.

Existing substrate, verified by the local `xsa --help` interface:

- `mem` returns maximal exact matches, widths, and bounded positions. Its `-k`
  is an output cap, **not** a support threshold; it cannot alone implement
  supported-depth search.
- `seeds` solves seed length to target multiplicity. This is a discrete attention
  temperature: longer patterns usually narrow fan-out. It is not a softmax.
- `fanout` uses weight(seed)=ln(D/mult), sums over DISTINCT shared seeds per
  document, and clusters hits into regions. Retain the tool's denominator and
  multiplicity conventions in receipts rather than relabeling occurrence count
  as document frequency.

For a new document-frequency scorer define explicitly
w(P)=ln(D/df(P)), with df counting eligible documents and D the eligible document
universe. For local attention, use visible regions as the analogous universe
and count distinct regions. This proposed variant avoids interpreting repeated
occurrences in one document as independent rarity evidence. Score a candidate
region by distinct-seed weights plus a bounded distance/recency term; freeze
coefficients and deterministic tie breaks in the protocol. Expose raw occurrence
multiplicity as well. Locality and rarity can rank candidates without changing
the exact intervals. Never silently equate this proposed scorer with measured
`xsa fanout` performance.

Project a bounded answer into the block: continuation-byte embeddings, histogram
features, interval metadata, and bounded causal span payloads form z_t. Then

    h'_t = h_t + W_suffix z_t + MemoryRead(h_t, slots)
    h_next = h'_t + MLP(h'_t).

Coordinates/readout and bounded hypothesis weights are trainable; interval
membership is discrete and detached. Span payloads may be byte-derived features
or already computed past hidden states, with their storage charged separately.
The latter retains a bounded value cache, not a claim that suffix arrays contain
hidden representations. A store-door variant APPENDs interval answers using the
native keying described in [ARCHITECTURE.md](ARCHITECTURE.md). Train and serve use
the same causal operator and policy; no future-populated feature cache.

Learned attention compares learned keys by dot products and usually touches all
visible positions. Here eligibility is exact substring equality, and weighting
uses explicit corpus/region statistics plus a bounded learned readout. It can
recover verbatim instructions, identifiers, and copies directly. It cannot
retrieve a paraphrase solely because its meaning is similar. Keep a neural
prediction path for empty support and smooth the final distribution: a forced
index branch is evidence, not permission to assign every other byte zero mass.

## 3. Updates, invalidation, and the five operations

At turn boundaries freeze the visible prefix; construct the new local generation
in scratch and publish it atomically only after hash and parity checks. Queries
pin a generation. Old readers finish before reclamation. Crash recovery uses
the last complete generation plus the replayable append log.

Three policies should share identical query semantics:

1. **Rebuild:** build one local index over all retained bytes each turn. Simplest
   correctness baseline; linear repeated work as the conversation grows.
2. **Segment build + cross-LCP merge:** index new turn bytes and merge local
   segments on a deterministic size-tiered tree. The proven route recomputes
   cross-boundary order. A merge costs proportional to bytes processed; a
   growing tree revisits bytes at multiple levels. Do not call repeated full
   merges constant-cost appends.
3. **Delta append:** between turns keep new bytes in a small causal delta with
   an exact short-pattern lookup or bounded scan. Search base and delta, dedup
   by global byte position, and cover patterns spanning the base/delta seam
   using up to d_max-1 preceding bytes with original ownership. For unrestricted
   depths, a fixed seam buffer is insufficient: fall back to exact search of
   retained bytes or rebuild. Publish merged generations at a size/latency
   threshold. Per-token updates affect the delta and visibility, not the full
   compressed index.

Do not stable-interleave chunk BWTs. The sxgc RESEARCH entries “MERGE OBSTRUCTION”,
“CHUNK-MERGE V3”, and “CROSS-LCP MERGE IS DONE” record three falsified merge
families before cross-LCP succeeded: cyclic BWT(A++B) is not a stable interleave
of the two input BWTs. Boundary order, periodic inputs, and wraparound need the
proven construction and differential checks. Correct physical cyclic order also
does not authorize cyclic matches as language-model evidence.

The five-op analogy is exact at the hypothesis/set layer, with limits at the
physical index layer:

| Op | Context/interval action | Physical consequence |
|---|---|---|
| APPEND | Add visible bytes and new interval hypotheses | Delta grows; old final suffix gains an observed continuation |
| CORRECT | Reweight hypotheses on contradiction; explicitly version an edited input | Editing bytes invalidates affected structures; rebuild/version, never patch counts casually |
| POP | Expire owned hypotheses or evict a visible range | Tombstone first; filter before support; compact later |
| DECAY | Reduce stale region/hypothesis weight | Weights change, exact raw counts do not |
| MERGE | Consolidate equivalent weighted hypotheses | Optional physical cross-LCP consolidation is a separate, semantics-preserving operation |

Evict at an explicit byte/RSS budget or remove revoked tool output. Demoting to
an archive makes it secondary, no longer local; retain provenance and separate
counts. Masked bytes are immediately inaccessible even before compaction.
Eviction is an emendation of the accessible index and interval set, not merely
KV allocation housekeeping. Privacy deletion additionally requires physical
reclamation of old generations, logs, and caches.

## 4. Cost ledger and conditional crossover

All measurements below belong to their cited workloads, not this proposal.

| Quantity | Status and source |
|---|---|
| 1,082,130,213-position, 16-chunk cross-LCP tree merge: 1,331.4 s, 27.5 GB peak RSS | Measured, sxgc RESEARCH “CROSS-LCP MERGE IS DONE”; 14.7x versus banked 19,596.8 s batch-BCR |
| Exactly 1.00 comparisons/position; 17.9 verified symbols/symbol; identical four output files under serial/tree schedules; chi 306,164,765 exact | Measured, same receipt; amortized O(1) per processed position, not O(runs) |
| Monolithic fragment construction 4,789 s | Measured, sxgc RESEARCH “CHUNK-MERGE V3”; full-container finalization costs must be accounted separately |
| Merge ~1.23 microseconds/byte; monolithic ~4.43 microseconds/byte | Derived by dividing those times by fragment positions |
| 1M-byte context: ~1.2 s merge or ~4.4 s monolithic construction | Projected linear scaling of fragment measurements; excludes startup, delta build, filtering structures, and publication; no transcript RSS measured |
| Pile query p50: 472.471 us at depth 1; 7,047.950 us at 16; 51,351.993 us at 1024 | Measured serial persistent CLI round trips, shared host, sxgc CONTEXT_DEPTH.md; not pure rank cost |
| Raw branches 22,768 contexts/s; end-to-end ~4,600–5,884/s; assembly 121,430/s | Measured different serving workload, SXGC_EXTRACTION_PERFORMANCE_ASK.md; not interchangeable with depth timings |

A one-second-scale maintenance pause per conversational turn may be viable.
Rebuilding or merging a million-byte base per emitted byte costs seconds per
byte and is not viable. The linear merge receipt proves neither online append
latency nor a fully dynamic XSA. Size-tiered compaction charges roughly O(N log N)
processed bytes over growth, unless a stronger maintained structure is built.
The earlier fragment peak RSS must not be scaled into a certified chat RSS.

For fixed feature width, dense attention visits O(n) keys per decoded position
and O(n^2) pairs over a prefix. At one million positions that is about 5e11
causal pairs (derived count); even without storing a quadratic matrix, dense
prefill computation is prohibitive for the intended deployment budget.

Fresh FM-style search costs O(d) rank operations, each often O(log r) in a
compressed implementation. A stated O(log n) rank-step budget **per position**
requires bounded/logarithmic d or maintained navigation; it is not a general
suffix-array theorem for arbitrarily long patterns. A maintained locus with
right-extension/suffix links could target O(1) amortized navigation updates,
plus rank, reporting, filtering, and maintenance costs. Those links are a
proposed implementation requirement, not an existing measured capability.
Even a fixed ladder costs O(d_max) extensions per position, as the extraction
ask itself specifies. Count aggregation and locating B bounded hits add costs;
full enumeration is O(m) and destroys a sublinear bound on common strings.

The long-context argument is conditional but concrete: bounded-depth,
bounded-output indexed search can avoid scanning a million keys, using O(N)
retained context/index space and roughly O(N d log r) fresh-search work over a
prefix, rather than quadratic pair work. Static index size is a separately
charged constant. Exact post-exclusion aggregation must also avoid enumeration
before the fast bound is realized.

A crossover estimate must name constants. Let dense attention's incremental
cost be a*n seconds and indexed retrieval cost q seconds/position, with M
maintenance seconds over G emitted bytes. Then n_cross=(q+M/G)/a. As an
**illustrative estimate only**, assume a=1 ns/key, q=10 us, M=1.2 s per turn,
G=1,000 bytes: n_cross≈1.21M positions. Excluding maintenance gives 10k positions.
At n=1M the same assumption yields 1 ms dense attention versus 1.21 ms indexed
amortized cost. If q instead equals the measured Pile depth-16 CLI latency of
~7 ms, the crossover exceeds 8M positions under these assumptions. These are
sensitivity calculations, not GPU timings or a speedup claim. Measure a and q
on the target hardware, including host/device transfers and synchronization.

The operator-provided matched-time receipts report ~226M versus 972M byte
targets at 2.13M parameters, and ~19.8M versus 103.5M at 150M in 2700 s: roughly
a 4–6x throughput deficit. The first pair is independently linked in
[RESULT_C14_EMENDER_TRAINED_V2.md](RESULT_C14_EMENDER_TRAINED_V2.md); the 150M
pair and operation-mix diagnosis are supplied task context, not a new benchmark
in this document. Sequential small-matrix store work has low arithmetic
intensity and leaves the GPU idle. Moving matching into an exact index may
change that operation mix; adding CPU queries may also worsen it. It does not
repair the store's arithmetic-intensity floor by assertion.

## 5. Failure envelope and evidence corrections

*(Superseded as an argument by R2.1/R2.3: corpus repetition statistics are not the basis
on which this operator is judged. Retained for the record as evidence about which corpora
are repetitive, and for the practical fan-out constraint in R2.4.)*

The natural-text prior is sparse at depth and broad at shallow lengths. The
primary sxgc [CONTEXT_DEPTH.md](../../../../sxgc/CONTEXT_DEPTH.md) is more precise
than its companion [note](../../../../sxgc/NOTE_TO_EMENDER_CONTEXT_DEPTH.md):

- Measured 14/1,000=1.4% at depth 1024 have **at least two other occurrences**.
  With **at least one other occurrence**, it is 75/1,000=7.5%. The companion
  note's “anywhere else = 1.4%” wording conflates thresholds. Neither figure
  is whole-source-document-excluded, boundary-filtered causal chat coverage.
- Information concentrates in the first ~16 bytes. At depth 32 the repeated
  subset's **median** entropy is zero, not every repeating context's entropy:
  the full table has nonzero entropy tails. At depth 1024 all sampled branch
  histograms are deterministic, but include source occurrences. Determinism
  does not imply predictive correctness on a new position.
- Measured total-width medians at depths 2/4/8/16 are
  341,562,604 / 5,473,331 / 24,873 / 11. Common strings can have enormous
  fan-out. Never locate every hit in the production path; adapt depth, cap
  candidates, and use exact aggregate structures or disclose the slow path.

A chat transcript contains repeated instructions, tool outputs, quotes, code,
and identifiers. Its repeat structure can differ sharply from the Pile, but
that is a hypothesis to measure. Exact matches miss conceptual references,
renamed identifiers, and paraphrases. Repetition can amplify obsolete or
malicious tool text; eligibility/provenance and learned judgment remain
necessary. Empty support must preserve a usable neural prediction path.

Dynamic indexing adds reproducibility hazards: turn segmentation, compaction
order, byte serialization, edited messages, eviction, and tie order. Bind exact
raw-byte and manifest hashes, binary/source revision, merge schedule, seeds,
query policy, scorer coefficients, and generation ID. Sort output ties by
layer/document/position. Compare rebuilt and merged indexes both structurally
where canonical serialization permits and semantically against exhaustive
reference queries. Byte-identical merge artifacts on one fragment are evidence
for that route, not proof for every future chat workload. Error means stop and
retain a failure receipt, not silently substitute a different feature stream.

## 6. Two measurements that decide whether to implement the model branch

These are executable experiment specifications, not measurements performed here.
Use a consented real transcript with approximately one million **byte tokens**;
record any subword-token count separately. Preserve tool bytes and message
metadata. Hash the immutable input and out-of-band logical-document/region map;
never write transcript contents into public receipts without authorization.
Use scratch under Emender, read-only access to sxgc and the static spine, and
`/home/erikg/emender/.venv/bin/python` for orchestration. No training is needed.

### A. Build and merge wall time and peak RSS

Record host, cores, thread cap, available RAM, storage, tool commit/binary hash,
compiler/build provenance, input byte count, and cache-state caveats. Use the
existing `xsa build --tape --input FILE --scratch DIR --threads J --memory-gb M`
for the monolithic baseline where supported. Cross-LCP uses the proven
`bit6/cross_lcp_merge.cpp` route and its recorded recipe; `xsa --help` does not
advertise a merge subcommand. Pin that recipe and dependencies before execution;
if unavailable, record merge as NOT RUN rather than inventing CLI flags.

Run monolithic construction, actual-turn incremental segment construction plus
size-tiered merges, and a final balanced 16-chunk merge for comparison to the
fragment receipt. Repeat each schedule three times (proposed protocol), with
fixed threads and order recorded. Wrap each build/merge stage in `/usr/bin/time
-v`; preserve stderr, exit status, elapsed time, maximum resident set, bytes
processed, and output hashes. For parallel child trees sample aggregate RSS as
well as each process's maximum: a child's peak alone is not tree peak memory.
Separate new-segment construction, merge, finalization, boundary augmentation,
and publication; report their sum as user-visible per-turn latency, with p50,
p95, maximum, and total processed bytes. Label startup and page-cache effects.

Require parity against monolithic results and a brute-force byte oracle on
boundary/seam, periodic, all-equal, edited/evicted, and ordinary sampled cases.
Assert no cyclic-wrap or hard-boundary candidates. Report per-token delta
append/query cost separately. A projected 1.2 s merge is not a passing result.
The deliverable is a hashed manifest plus per-stage JSONL timing/RSS receipts
and explicit failures; do not scale a partially completed run into success.

### B. Causal retrieval coverage and fan-out

For every eligible prediction position t in that transcript, query exact fixed
depths 2,4,8,16 without backoff, separately for local, static, and union. Positions
with fewer than d preceding bytes in their logical document are excluded from
that depth's denominator and counted explicitly. Require p+d<t locally; exclude
the full static source document before counting. Record m, raw width, distinct
region/document support, entropy, selected candidate count and truncation.

For each depth/layer report fraction m>=1 and m>=2, with exact numerators and
denominators; full fan-out histogram or lossless per-position counts; zero
fraction, p50/p90/p95/p99/max both including zeros and conditional on m>0.
Stratify by user/assistant/tool, code/quotation regions where annotated, distance
bins, and turn age. Report conditional fallback frequency and static rescue
fraction under a frozen min_support policy. Never present capped returned-hit
counts as fan-out. Cache counts only by prefix generation and exclusion policy.

An offline reference can scan visible bytes or use depth-specific hash tables
of exact byte strings, updating continuation counts only when that byte becomes
visible. Audit all depths at 1,000 seeded, saved positions plus every boundary
and delta seam against exhaustive scans; this checks index results independently.
Record query latency distribution, rank/locate work, filtering work, and RSS
alongside coverage, so an O(m) reference is not mistaken for production speed.

Compare with the Pile's same depths: measured repeated-subset m>=2 counts in
the 1,000-position study are 1000/1000, 1000/1000, 966/1000, 644/1000 at
2/4/8/16. Its widths include the source and its elsewhere counts exclude only
one occurrence, not the whole document; it also allows separator crossings.
Show these as a differently defined baseline. For an apples-to-apples second
panel, requery the saved Pile positions using the now-available document map,
boundary checks, and whole-source-document exclusion; label those results NEW
when actually measured. No numerical source-excluded Pile coverage is asserted
here. A third local-prefix Pile control can use equal-sized single-document
windows, resetting state at each boundary, to separate corpus size from chat
structure. Do not carry state across unrelated Pile documents.

Decision rule: if causal local support is sparse at all useful fan-out budgets,
the object remains an exact-copy niche and static fallback, not a replacement
for learned attention. If coverage is substantial and total measured update +
filtered-query cost beats the measured dense-attention cost at target length,
implement the bounded residual branch and register a later trained comparison.
Neither coverage nor construction speed alone establishes a quality gain.

## 7. Interval algebra, implementation boundary, and claims

The maintained model state is S_t={(layer, I(P), w, owner, generation)}. EXTEND
and BRANCH propagate exact hypotheses, SUBSUME backs off, SELECT/PRUNE allocates
a bounded frontier, UPDATE reweights evidence, and EMIT mixes branch evidence
with the neural readout. APPEND/CORRECT/POP/DECAY/MERGE provide that set's lifecycle.
The dynamic context changes the ground text as well as the weighted set; every
handle therefore includes a generation and byte-pattern identity for rebind.
A raw interval rank is not stable across merges. This is the interval theory's
particle-filter interpretation instantiated over the model's own input, with
a static corpus prior available through the same operator.

The first implementation boundary is the object and two measurements above:
versioned context snapshots, causal/exclusion-aware query adapter, exact
reference counter, cross-LCP maintenance adapter, bounded answer format, and
replay receipts. No learned controller, training launch, or spine modification
is needed to establish feasibility. No services were started for this design.

Do not claim a trained suffix-attention model exists; semantic attention has
been replaced; all deep repeats are deterministic; 1.4% means any-repeat
coverage; merge is O(runs); dynamic updates or filtered aggregation are already
constant time; CLI round trips predict in-process local latency; a million
subword tokens equals a million bytes; or the projected seconds/RSS are measured
chat results. The engineering case is explicit causal self-access with bounded
candidate work at long context. Its practical value remains conditional on the
two receipts, followed by a separately registered quality experiment.
