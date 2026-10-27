# Chunk-parallel tape: formalization and a structural obstruction

Authority: read-only X31 PARALLEL_DECOMP.md, REPORT.md part 8 residuals and
post-part-9 adjudication, summary_part5/6/7/8.json; X30 naive_reference.py;
X28E REGISTRATION.md F12/F12g/F12h. X31's 109 tests establish 1e-5 numerical
parity, not universal bit identity. No resilient DiLoCo behavior is involved.

## Semantics and dependency graph

Let X_t be the state immediately before token t. READ(X_t,q_t) precedes POP,
post-pop death, WRITE, rounded decay, age increment and decay death. A routing
quantity being detached asserts its derivative is zero, not its independence
from X_t. In particular F12g addresses the lowest-index live age-1 row; it does
not match keys against preexisting content. Correction does not renew birth.

| Op | Inputs/dependencies | Parallel opportunity / exact restriction |
|---|---|---|
| Write enable | detached beta, token mask | independent per token |
| Age-1 selection | live and birth, including earlier appends | shifted append IDs only when survival and tie ordering known |
| Residual and append gate | current target value, current has-live | intra-chunk appended value is needed even for ZERO_RESIDUAL |
| Case | age-1 existence, residual threshold, free availability | cannot evaluate from entry state alone |
| POP | all currently live keys, query argmax, strength | new keys can win; repeated strength changes/deaths affect later tokens |
| Correction | target value + beta*(v-target) | disjoint targets scatter; same-target fp32 recurrence ordered |
| CAP | current full status and youngest-live target | competing frees/appends and repeated writes ordered |
| First-free | current free set | rank scan only without intervening frees |
| Decay | rounded repeated multiply, touches | exact X31 jump arithmetic; analytic powers not equivalent |
| Death | exact deadline, POP touches, token phase | untouched deadlines known; touches invalidate predictions |
| Read | all causal live keys, values, strengths | causal batched scores possible AFTER payload dependencies resolved |
| Pressure | request/CAP bits, rolling 128 window | integer prefix counts + shifted differences |
| Backward | recorded routes, ordered value/strength history | preserve replay ownership and accumulation order |

## Counterexample to the proposed first pass

Empty state, beta=1, pop=0, append logit=0, coefficient=0, value v=1 at
both tokens. First token stores sigmoid(0)*1 = 0.5, then its age becomes 1.
Second token's squared residual is 0.25 >= 1e-6, so it CORRECTS slot 0.
Evaluating both tokens against the empty chunk-entry state produces EMPTY,
EMPTY and two appends. This uses detached quantities throughout. In an a-run
with constant projected value it repeats append/correct, because the correction
does not reset age. Token spelling cannot certify pure appends.

Even a genuine pure append run is insufficient: v0=1, v1=0.5 and coefficient=1
make token 1 ZERO_RESIDUAL, matched=0, with append gate sigmoid(0)=0.5.
Entry-only evaluation uses no match, EMPTY instead of ZERO_RESIDUAL, and
has-live=false; a nonzero residual/gate example follows by perturbing v1
within sqrt(eps). A zero coefficient removes that gate dependency but leaves
case/matched differences. These are trace differences before reduction rounding.

Finally, two zero-valued appends have no corrections or pops, but token 1's
read includes token 0's newly live row in the softmax denominator. With
nonzero payloads this changes the read. Batched reads must use causal masks
and time-indexed payloads, never only the entry store.

## Classifier: exact implemented conservative condition list

There are two distinct labels. MUTATION_CANDIDATE means every token case is
NO_WRITE/EMPTY/NOMATCH/ZERO_RESIDUAL, no POP, no death, and all allocation
IDs are distinct. It permits intra-chunk matched-value dependencies and is
NOT an execution certificate. ENTRY_PURE is a conservative sufficient
structural routing certificate (with time-projected entry ages, strengths and
liveness, not frozen age fields), requiring additionally no write-enabled token after any
prior append in the segment. This excludes occupancy changes, age-1 chains,
residual dependencies and new-key POP argmax changes. All POP, correction,
CAP, death, and repeated allocation tokens are event boundaries. The
instrumentation reports intra-chunk matched/POP references separately.

This is a retrospective trace classifier, not a predictive GPU dispatcher.
A trace cannot prove that the alternative entry-only pass would have made
the same decision: the stronger conditions above are deliberately restrictive.
The label alone cannot certify arbitrary imports: projected age-1 ties, initial
values, live status, and exact death schedules must also be known
before using any future runtime certificate. Classification does not assert
parallel READ or gradient bit identity. Empty/dead payload preservation and
usage (excluded by X31) require separate scope declarations.

## Conditional parallel algorithm (work order, not implemented)

1. Validate a predictive certificate before writes. Derive exact survival and
births. For entry rows compute per-time strength using X31 exact jumps.
2. Build free-bit prefix ranks r(s)=sum_{j<=s} free(j). For token append
indicators a_t, let p_t=sum_{u<=t} a_u. Destination is the unique free slot
with r(s)=p_t. Integer scans are exact; fp32 free-list order is not the issue.
Reject on insufficient capacity. No mid-segment frees are allowed.
3. Resolve appended values and gates. If token t reads the value from a
previous append u, add edge u->t even when case is ZERO_RESIDUAL. Resolve
these edges in dependency waves; a chain can have depth C. Do not call this
a two-pass independent decision evaluation. Scatter only disjoint writes.
4. Score all queries against entry plus new keys. Apply causal birth/death
masks, per-time strength and value versions. Reuse the exact X31 per-query
slot tiling, reduction tree, fp32 rounding and disabled fusion for bit identity.
An arbitrary GEMM with a different reduction tree offers tolerance parity only.
5. Materialize state and preserve trace and journal/replay semantics. Gradients
must replay recorded decisions, including gate's detached residual, and retain
identical reduction/accumulation order if bit identity is required.

## Sparse segmentation and equivalence

Walk tokens in order. Before an event or dependency on a pending write, flush
the preceding certified segment, run the event with X31, and rebase against
the resulting state. Maximal segments can be discovered retrospectively by
`segments`; actual routing still needs a predictive certificate. In the worst
case every token is an event. Splitting X31 calls also adds materialization
and can alter gradient accumulation order: composition requires explicit
bitwise tests, not an assumed associativity of fp32. Keep projections outside
segmentation to avoid changing GEMM rounding or token context.

An induction proves equivalence only if each segment receives identical
entry state, routes, gates, rounded arithmetic and causal reads, and produces
identical boundary state and replay derivatives. Disjoint destinations prove
write independence, not routing or read independence. General delta-rule
recurrences are affine only over real arithmetic; fp32 reassociation changes
rounding. Associative composition of arbitrary rounded maps exists but no
compact efficient scan representation is established here. Thus the requested
unconditional two-pass theorem is false, not a GPU implementation gap.

## Timer wheel and occupancy work orders

A bounded wheel must use absolute integer deadlines and rounds/overflow for
long horizons (decay=1 has no finite death). Bucket members need cancellable
links/generations for POP rescheduling, deterministic tie processing, and the
separate post-POP/end-decay phases. Scanning empty time intervals is O(elapsed
tokens), not O(deaths); an adversarial bucket costs O(bucket size). O(1)
amortized insertion/cancellation is achievable with intrusive lists; exact
terminal strengths still require X31 jump arithmetic. No wheel is integrated
here: adding one before a valid parallel path would not rescue the theorem.

Part 2 backing work: live-row pool + cold-row storage (including dead keys,
values, ages/strengths), stable logical-slot map, first-free logical ordering,
materialize-on-access public views, and ownership/rollback support for every
journal address. Measure total allocations and continuation copies, not just
retained logs. Part-9 scoping correctly identifies a representation change;
it cannot be replaced by dropping dead payloads.

## Consequences (trained-density receipt)

`density_trained.json` records exactly 0.5 event-token fraction in every
registered trained ladder bin and distinct-k band, including all-a/all-b full
windows. The case prefix is EMPTY, CONFLICT, NOMATCH, CONFLICT, ...: after
the first empty-store append, subsequent appends are NOMATCH, not EMPTY.
This is the trained F12g append-then-correct mechanism, with correction every
two tokens; it is not initialization noise. The MQAR hybrid is a separate
contrast and does not change this ladder result.

For this instruction set, hard age-1 architectural addressing entails an
irreducible sequential decision chain inside a chunk: each append creates the
next token's age-1 target, whose payload determines residual/correction routing.
Detaching that decision does not remove the dependency. Conversely, retaining
this exact mechanism requires retaining those ordered decisions; the measured
speculation depth is two tokens. This is a mechanism-specific trade-off, not
an impossibility theorem for all hard addressing or all parallel algorithms.
DeltaStack's soft-pointer/WY formulation gains parallel training by relaxing
exactly this hard age-1 append/correct mechanism; it is a different recurrence,
not an equivalent scheduling of F12g. Sequential routing remains authoritative.
Batched READs still require causal masks and time-indexed payload versions,
including mid-chunk appends/corrections, with recorded-decision replay adjoints.
