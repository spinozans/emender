# K7 design — the high-rank sequential decision walk (written BEFORE the build)

**Status: the design fork of K7 Part B, resolved by argument and receipt before any
kernel was written.** The companion build document is [`K7_HIGH_RANK.md`](K7_HIGH_RANK.md);
Part A (the slot-blocked chunked value scan) has no fork and is specified there, §1.
Everything here is at the boundary the lane's receipts use: one (layer, window) scan,
fp32, forward only, probe scale.

## 0. The constraint, receipted

The K6 per-token walk (`k6_sequential_matrix.py`, receipts `k6_gates.json`) holds the
ENTIRE store — `rank x d_v` fp32 — in ONE program's registers, because every decision
and every read needs full rows:

* at the registered shape (rank 128, `d_v` 128 = 16 384 floats) it fits exactly:
  255 registers, **0 spills**, 8 warps (`K6_SEQUENTIAL_MATRIX.md` §2);
* the first K6 design that also held the normalised rows spilled 222 registers and ran
  83 ms — the receipted proof that the register tile is the binding constraint, not a
  guess (`K6_SEQUENTIAL_MATRIX.md` §6.1);
* `sequential_available` therefore refuses by name beyond 16 384 floats, and the K6
  wrapper falls back to the **eager per-token oracle** — the same semantics, the parity
  fixed point, slow. At rank 1024 the store is 8x the budget (131 072 floats); at
  rank 2048 it is 16x (262 144 floats). The same ceiling, at a different constant
  (`MAX_PERSISTENT_TILE` = 8 192), refuses the persistent chunk-granular kernel
  (`hostloop_free.value_block`, `NO_HOST_LOOP.md` §4) and the K4 backward.

So the growth snap set {1024, 2048} (`keys.ORTHONORMAL_RANKS`) is unreachable by every
one-program-owns-the-store kernel in the lane. The walk must be redesigned so the
program no longer owns the store. Two designs were on the table; the second is refused
here, from the receipts.

## 1. Option (i) — global-tiled state walk (CHOSEN)

The state lives in **global memory** for the whole window. One program per
`(batch*head)` row walks the tokens serially; per token it sweeps the store **once or
twice in slot tiles of `BR x d_v` floats held in registers**, computes each tile's
contributions to the per-token decision quantities, and writes only what changed back.

Per token, in the registered order (`k6_sequential_matrix.py`'s forward):

1. **sweep A (read-only)** over slot tiles: load the tile (pre-decay — the state as
   this token's read and decisions see it), compute per row the DEFINED value-axis
   tree folds: `nr_j = sqrt.rn(tree(S_j . S_j))`, the read score
   `score_j = div.rn(mul.rn(scale, tree(uq . S_j)), max(nr_j, EPS))` and the decision
   cosine `sim_j = div.rn(tree(pv . S_j), max(strength_j, EPS))`, writing the three
   `(rank,)` vectors to a per-token scratch and folding the tile's max into the
   running softmax max (max is exact — order-independent).
2. **the decisions and the read weights**, in registers, on whole-rank vectors —
   exactly K6's code: the softmax denominator is the defined balanced tree over the
   padded rank axis (in ONE program — no split), the selection chain
   (best/second/worst with the first-index rule), the five thresholds, the MERGE
   blend from two gathered rows, the weights `w = mul.rn(probs, strength)`.
3. **sweep B** over slot tiles: load the tile again (still pre-decay), accumulate the
   read numerator partial `P_b = sum_j w_j S_j` (a native per-tile sum — see §3),
   apply DECAY in place (`mul.rn` per element), store the tile back.
4. **the lanes** (after the decay, the registered order): the addressed row is one
   `d_v`-vector read-modify-write; the MERGE source is zeroed; the liveness vector
   (registers, whole rank) is updated.

**The bandwidth arithmetic, stated (and measured).** Per token per row the store
is read twice and written once: `3 * rank * d_v * 4` bytes. At rank 2048,
`d_v` 128 that is **3.0 MiB per token per row**; over a 2 048-token window at
16 rows (the registered block's `batch 8 x heads 2`): **103 GiB of DRAM traffic
per scan**. Measured (receipt `k7_high_rank.json`, `refusal_removed.walk_rows`):
**374 ms per scan at rank 2048** -- an effective ~275 GB/s, i.e. the walk is
FOLD-bound, not bandwidth-saturated (the sweeps are made of dependent defined-
order tree folds, the same latency class that makes K6's register walk 13
us/token), so the pure-bandwidth estimate of ~150 ms at ~700 GB/s is a floor,
not the wall. The honest comparisons: the walk is **29 ms at rank 128** against
the K6 register-resident kernel's 26.5 ms (1.1x, and it works where K6's kernel
refuses), **191 ms at rank 1024**, **374 ms at rank 2048**. This is the price of
NOT changing the registered semantics; it is stated, not optimised away.

Sweep A cannot be merged into sweep B for free: the read numerator needs `w`, which
needs the softmax denominator, which needs the global max, which is only known after
every slot's score exists. The alternative — dropping the max subtraction (scores are
bounded in `[-scale, scale]`, so `exp` cannot overflow) — would make one sweep
possible (2.0 MiB/token, a 33 % traffic saving) but changes the softmax's fp
expression away from the registered/K6 form (which subtracts the max); refused at
probe level, recorded here as the available lever.

## 2. Option (ii) — compressed decision state (REFUSED)

The walk maintains only decision-relevant quantities instead of the store: exact
scalar recurrences for the strengths/norms "where the algebra allows", plus a
registered sketch for the query-alignment term. Three receipts refuse it:

1. **The strength recurrence is not exact in the defined order.** In real arithmetic
   DECAY scales a norm multiplicatively, but the registered arithmetic RECOMPUTES
   `nr_j` as the defined tree fold of the decayed squares each token (the K6 spec's
   own definition, `PREREGISTRATION_SEQUENTIAL_MATRIX.md`, "every reduction is a
   balanced binary tree"). A carried `nr'_j = mul.rn(nr_j, lam)` is a different fp
   chain — it drifts from the defined value at ~1 ulp per token, and a decision at a
   threshold (`tau_match` 0.45 on `sim = fold/max(strength, EPS)`) can flip on that
   drift. "Exact" scalar recurrences are exact only in real arithmetic; the gate is
   bit-exact DECISIONS, and the recurrence breaks it by construction.
2. **APPEND breaks the recurrence outright.** `||S_j + p||` is not a function of
   `(||S_j||, ||p||)` — it depends on the angle between them — so a strength channel
   maintained without the rows is wrong at every APPEND, the most common live op.
3. **The query-alignment term has no recurrence at all.** `q_t . S_j` changes with
   the query at every token; only a fixed low-rank sketch of the rows (a NEW
   registered semantic, its own draft registration, operator sign-off) could
   approximate it — and approximate is the operative word: the K5A probe measured what
   non-reproducible decision inputs cost (~48 % of similarity entries bit-equal,
   `K5A_ROUTING_DIFF.md`), and K6's gate A (0 flips in 17.0 M lanes) exists precisely
   because the kernel computed the decision inputs in the defined order from the
   actual rows. A sketch fails that bar by construction.

The compressed option is therefore not a cheaper implementation of the registered
walk — it is a different architecture whose decision inputs are not the registered
ones. K7 does not have the authority to register it ("registered semantics
untouchable"); it is recorded here as the refused fork with its reasons.

## 3. What is bit-exact and what is at the floor, stated once

* **Bit-exact (gate: `torch.equal`)** — every decision channel (slot, op, erase,
  live_after, the MERGE-blend payload lane, pre-decision liveness) and the read
  WEIGHTS channel, against the K7 oracle (`k7_high_rank_store.sequential_oracle`,
  written to the kernel's reduction order — the K6 precedent). The value-axis folds
  are per-row defined trees, identical to the whole-row trees (a tile never splits a
  row's value axis); the rank-axis max/min and the first-index tie rules are exact;
  the softmax denominator is the defined whole-axis tree in one program, unsplit.
* **At the measured floor** — the read OUTPUT, `sum_j w_j S_j`: a native per-tile sum
  plus sequential cross-tile adds, a reduction order no torch expression reproduces.
  This is exactly the K1/K23/K6 corner ("the read feeds no decision, so no flip can
  come of it" — `K6_SEQUENTIAL_MATRIX.md` §1.3); the floor is measured on the same
  fixture first and the allowance is the lane's `NOISE_FACTOR` (8x).

## 4. Growth readiness (design requirement, not wired)

Nothing in the chosen design precludes mid-sequence rank change: the walk reads its
shapes from the tensors at call time (`state.shape[2]` is the rank; `RP` is the only
rank-derived `constexpr`, so each snap {64, ..., 2048} is its own cached kernel
binary), the live/scratch channels are allocated per call at the snap's width, and a
growth event that widens the rank at a window boundary re-dispatches to the next
snap's kernel (or its pre-captured graph) with the new rows exact zeros — the same
prefix-reveal invariant the chunked arm's growth already relies on
(`keys.py`: the active table is a prefix, growth only reveals rows). Wiring a growth
wave is explicitly out of scope for K7; both parts of this document only guarantee
the shapes are schedule-recorded, never baked into a module global.
