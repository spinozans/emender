# Round5 attention import: register-tiled exact READ

Pre-implementation decision. Read all four result documents and R1 execution
notes/R2-R4 resume receipts. No supervisor/intercom callable tool is exposed
(discovery by name and description returned zero); no contact is claimed.

## Exact algebra, and why direct online softmax is not byte compatible

c14_b_store.read and the native score/normalize implement softmax, not a tuned
non-softmax match kernel. For each currently live slot i:
s_i = fixed_sum_64(K_i * q) * (2 / sqrt(64)); m=max_live(s);
n_i=exp(s_i-m); d=max(fixed_sum_BS(n),1e-30);
p_i=round32(round32(n_i/d)*strength_i).
READ accumulates round32(p_i*V_i) in two ascending parity chains, then adds the
two chains using fixed_sum. No live slots gives zero. Scores use scale 1/4.
match_coefs and match_wire affect the post-read stream write-enable gate via
match_coef*(match_wire*max_score); they do not change softmax addressing.
Search in velocity_candidate ranges match_wire .25..4, separate occupancy and
error terms. This distinction rules out deriving a different read algebra from
the search parameter names.

FlashAttention (Dao et al., https://arxiv.org/html/2205.14135v2, Algorithm 1)
imports tiles and online m,d,U: m'=max(m,max(tile)); alpha=exp(m-m');
d'=alpha*d+sum(exp(s_tile-m')); U'=alpha*U+sum(exp(s_tile-m')*strength*V).
In real arithmetic U/d is this store's read; strength enters numerator ONLY.
The empty-state initialization needs a guard against exp(-inf-(-inf)).
Finite FP32 rescaling, division placement, and partial sums reorder the native
operations, so a near-verbatim online accumulator is mathematically exact but
not the existing byte semantics. We import register-sized tiling and latency
hiding, retaining the native denominator tree and normalized parity chains.
This is an internal kernel candidate, not a semantic/registration amendment.

## Profile-driven choice

profile_fixed.json retains completed CUDA-event stage measurements even though
the enclosing harness later failed an attribute lookup. A fresh corrected run
is also required. T1024: READ 218.75 ms; control 73.22; normalization 38.25;
score 25.07. Instrumentation adds event/enqueue gaps: these are not additive
attribution of the uninstrumented 256.59ms nor hardware counters. Nonetheless
ordered READ dominates the measured stream intervals (~62%). Each READ page
currently issues 512 dependent pair-load iterations. Target load latency with
32-row register tiles / static 16-pair scheduling. Keep PAGE1024 carries.
Tile is 32x128 values, not BSx128 or TxBS. Compiler regs/spills and wall time
will decide whether this import actually helps. No speedup predicted as fact.

PagedAttention (Kwon et al., https://arxiv.org/abs/2309.06180) additionally uses
logical/physical page tables. R4 pages are launch partitions, not that memory
mapping. No measured allocator/fragmentation bottleneck warrants adding page
tables to this contiguous scratch fixture. Context cannot be processed in
parallel like immutable causal KV: pre-read CORRECT/expiry/array updates and
post-read POP/MERGE mutate old slots. Token-order state transitions remain.
No claim of O(1) total metadata: scalar control and normalization remain BS-wide.

## Sparse-read evidence and decision

Existing PB3 PB_RESULT reports all array effective weights nonzero (rate1.0),
not their concentration. It cannot answer top16/32 mass. Synthetic diagnostic
snapshots at t256/512/1023 have top16 .688/.667/.661 and top32 .791/.840/.810,
with 256/257/257 nonzero slots. This does not certify trained-model concentration;
trained PB3 top-k masses remain unmeasured. It provides no byte-exact pruning
claim. Do not implement top-k or train. Any subsequent sparse read needs a
variant registration and quality predictions; exact top-k selection still scans
occupancy unless an index is introduced. No automatic O(k) addressing claim.

## Carry proof and gates

New READ must leave each rounded product unchanged and append each to the
same parity chain in increasing slot order, first pair without extra zero.
A tile is a scheduling/storage unit, never a sum. FP32 global page carries
retain the bits, and the final two-chain reduction is unchanged. Induction on
pairs/tiles/pages, then tokens, is R4's composition proof; all other JIT functions
are imported unchanged. Disabled fusion, NaN payloads and signed zeros tested
against independent matrix primitives through BS4096. Ten-shape monolithic
native/full-state/raw-trace/repeat gate; operation-rich short shape/replay/
gradient fixtures; T512/T1024 monolithic and T2048 PAGE256 compositional masked
fresh-document fixtures. Preserve failures and source snapshots, no overwrite.
Measure five synchronized allocation-inclusive wall samples, matched inputs,
then reproject 22 layers*2heads*2passes, 2B targets/T1024. <=.502691ms/head is
necessary for24h, not sufficient; full training gates remain before promotion.
No training, DiLoCo changes, WG/delegation, protected-card operations, corpus or
.pt commits. Only broker card4/5, approved venv; unrelated worktree preserved.
