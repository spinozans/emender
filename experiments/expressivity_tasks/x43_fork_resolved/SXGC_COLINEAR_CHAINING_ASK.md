# Ask to sxgc: a native `xsa chain` (colinear chaining as one operator)

**From:** emender side · **Date:** 2026-11-13 · **Measured against:**
`/mnt/nvme2n1/erikg/P100c/pile-head100.sxi2` (63.16 GB), binary
`/home/erikg/sxgc/xsa/target/release/xsa`

Companion to `SXGC_SEEDS_COST_NOTE.md` (the batch/resident `seeds` request) and to
`RESULT_CHAIN_COVERAGE.md` in this repo (the measurement that motivates this one).
This is a **document only**: nothing here was implemented in `~/sxgc`, and nothing
in `~/sxgc` was modified.

## Why

Stages 1, 3 and 4 of the operator chain in
`experiments/expressivity_tasks/x43_fork_resolved/DESIGN_SUFFIX_ATTENTION.md`
(R2.2) already exist as CLI primitives: `xsa seeds` (anchors), `xsa branches`
(fork), `xsa forks` (fork matrix). Stage 2, **colinear chaining**, exists only as
the Python prototype in this repo (`chaining.py`). The measurement in
`RESULT_CHAIN_COVERAGE.md` says the prototype's binding constraint is not the DP —
the DP is O(n²) on a few dozen anchors and instant — it is the **shape of the
anchor set the CLI can hand over**:

* `xsa seeds --limit N` returns the first `N` rows of the seed's suffix-array
  interval. Those rows are adjacent in SA order, they share a query offset, so
  **they can never be chained to each other** (a chain needs distinct query
  offsets on one diagonal). With `N = 8` (the default) and a 32-byte query band,
  chains with more than one anchor occurred in 0/150 sampled transcript positions
  and 11/150 sampled Pile positions.
* The correspondence stage needs the mirror image of the seed scan: matches whose
  **right edge is the prediction position**, not matches that start at every
  offset. `xsa seeds` emits one seed per window *start* and suppresses a seed
  contained in an earlier one, which removes exactly those right-edge seeds when
  a longer seed covers them.

So the missing object is not "a chaining library". It is **one call that returns a
chain-ready anchor table for a query window, and the chain/correspondence derived
from it**, so that stage 2 does not require 10⁴ single-position locate walks per
transcript.

## The proposed operator

```
xsa chain <container> --windows FILE [--chain-opts] [--docs DIR | --corpus FILE]
xsa chain <container> --serve -j N            # framed: u32 len + bytes per window
```

### Inputs

| input | semantics |
|---|---|
| window | the query context, arbitrary bytes, forward order, **no** next byte. Framed in serve mode (`u32` length + bytes), or one window per record in a `--windows` file with declared boundaries (not newline-inferred) |
| `--min-len L` | shortest seed counted (default 12, as `seeds`) |
| `--max-len` | longest seed (default 64) |
| `--target-mult T` | multiplicity the seed length is solved for (default 50, as `seeds`) |
| `--anchor-cap K` | positions returned **per region**, not per seed (see the next section) |
| `--region-gap G` | corpus gap that separates two regions (default 2000, matching `fanout`) |
| `--site` | the query offset that must be answered (usually `len(window)`); the correspondence target |
| `--exclude-source` | drop anchors whose corpus span overlaps the query's own source region (required to make a corpus-queried-from-itself measurement honest) |
| `--max-gap`, `--diag-penalty`, `--gap-extend` | the DP's registered constants, echoed back in the receipt header |

### The one design requirement that matters: region-first anchor enumeration

A chain is a set of anchors in **one corpus region**. Enumerating all positions of
a seed is exactly the wrong currency (a 12-byte seed has 10⁴–10⁷ positions); the
useful currency is **which regions share a long string with me, and a bounded
anchor set inside each**. Concretely:

1. find the distinct seeds of the window (already `seeds`' job);
2. cluster their intervals into corpus regions (`fanout --region-gap` already does
   a version of this over distinct seeds);
3. return, per region, up to `K` anchors with their query offsets, so a chain can
   be built inside a region without materialising the interval.

That also fixes the fan-out number the design needs (`A_O`'s distinct-region
support), which today must be reconstructed from a bounded SA-edge sample.

### Outputs

Machine-readable, one row per window, no byte echo required (the caller already
has the window):

* **anchors**: `(query_offset, corpus_pos, length, seed_mult, region_id)` — a
  complete set up to `(L, transformation class)`, or an explicit `truncated=true`
  with the truncation rule named. Never mix "no evidence" with "capped output".
* **chains**: `(chain_id, score, n_anchors, query_span, corpus_span, diagonal,
  region_id)` with the score defined exactly as
  `covered_query_bytes - diag_penalty*|Δdiagonal| - gap_extend*(query_gap+corpus_gap)`
  (the prototype's rule, `chaining.py`), so the two implementations are comparable.
* **correspondence** for `--site`: `(corpus_pos | null, kind, extrapolation_bytes,
  chain_id)` with `kind ∈ {matched, abutting, extrapolated}`:
  * `matched` — the site lies inside an anchor, the corpus byte is verified equal;
  * `abutting` — the chain's right edge is exactly the site (extrapolation 0);
  * `extrapolated` — the chain's diagonal is continued `extrapolation_bytes` past
    its last anchor; **the caller sets the budget** and gets the distance back.
* **exact aggregates, independent of any cap** (the `A_O` contract of R2.4):
  number of distinct seeds, number of distinct regions, the exact seed widths, and
  the widest region's mass. These must not be derivable only from the truncated
  rows.
* **boundary honesty**: chains that cross a document boundary must be marked
  (`crosses_document=true`) or refused. `branches` already refuses to conflate
  boundaries silently on a container without member m15; `chain` should not
  reintroduce the conflation by chaining across a separator.

### Complexity and cost the caller must be able to predict

Measured on this container, warm, serial:

| operation | measured | what it means |
|---|---|---|
| `xsa seeds … --limit 0` on a 32-byte window | **80 ms / call** (scan only) | fine for thousands of windows |
| `xsa query --patterns … --count`, 10³ patterns | **0.09 ms / pattern** | length/multiplicity solving is cheap; the support ladder is nearly free |
| `xsa query --patterns … --limit 8` | **≈ 33–37 ms per located position** | the LF locate walk (≈3.5 k rank steps) dominates everything |

`--limit 8` on 308 distinct seeds for 150 transcript positions cost **63 s**
(2 331 positions). The Pile panel cost **202 s** (5 386 positions). Locate is
≈10⁵× more expensive per answer than an interval width, so:

* the chain operator should be **O(distinct regions × K)** locates per window, not
  O(occurrences);
* an interval-only mode (widths, no positions) should stay available, because it
  answers the L-ladder question — "does this context still have a counterpart at
  length L" — at 1/1000 of the cost;
* the DP itself may be quadratic in the anchors of one window; nothing measured
  here came close to that being the bottleneck.

### Determinism

Ties resolve by `(region_id, corpus_pos)`, never by hash order; the same window and
the same options must produce byte-identical output, and the receipt header must
carry the option values (the `seeds`/`branches` habit of echoing
`target_mult`/`cap`/`mode` is exactly right).

### What this operator would NOT fix

Chaining cannot manufacture evidence that is not there. On a real agent
transcript, the 12-byte context suffix has a counterpart elsewhere in a 100 GB
corpus at only **36.7%** of byte-uniform positions (15.3% at 16 bytes, 0.7% at
32 bytes; Pile: 82.7% / 69.3% / 19.3%). A native `chain` makes the operator
cheaper and the anchor set honest; it does not raise that number. `L`, the
transformation class, and the corpus are the only knobs that do.
