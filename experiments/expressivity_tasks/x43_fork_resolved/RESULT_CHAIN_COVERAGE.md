# Colinear chaining over `xsa seeds` and retrieval coverage on a real transcript

2026-11-13. Stage-2 build and first measurement for
`DESIGN_SUFFIX_ATTENTION.md` Revision 2 (§R2.2 `seeds → chain → correspondence →
fork`, §R2.3 completeness-not-support, §R2.5 the registered L). This receipt covers
the chaining prototype, the coverage/fan-out/fork measurement on a real agent
transcript and on Pile text, and nothing else: **no claim about model quality,
about the value of a neighbourhood once found, or about training.**

The short version, with numbers that are explained below:

* Chaining works and is exact where it is exercised (synthetic known-correspondence
  test), but on these sources **the DP is not what limits the operator** — the
  anchor set the CLI can hand over is. With `xsa seeds`' default 8 SA-adjacent
  positions per seed, chains containing more than one anchor appeared in **0/150**
  sampled transcript positions and **11/150** sampled Pile positions (depth 32).
  Widening the per-seed position sample to 64 raises that to 2/30 and 13/30.
* A usable (site-abutting) correspondence exists for **37.3%** of byte-uniform
  transcript positions at depth 16 and 32 (registered `L = 12`), versus **82.7%**
  for Pile. The measured abutting fraction equals the measured `L = 12` support
  fraction exactly, which is what §R2.3 predicts.
* Where a correspondence exists, the fork at the aligned site names the model's
  actual next byte at rank 1 for **58.9%** (depth 16) / **50.0%** (depth 32) of
  transcript positions, against a **15%** chance rate — but the sites are short
  exact repetitions (fork width median 4 occurrences, 1 distinct continuation), so
  this is a deterministic-continuation effect, not retrieval of the same content.
* Below `L = 12` the operator has no evidence at all at depths 2/4/8 by
  construction. That is the dominant loss, and it is the registered knob, not a bug.

## 1. What was built

| file | role |
|---|---|
| `chaining.py` | stage 2: anchors → colinear chains (sparse DP) → correspondence; region clustering; fork-outcome scoring. Pure geometry, no I/O |
| `test_chaining.py` | 16 tests, including a synthetic case where the true correspondence is known by construction |
| `chain_coverage.py` | the measurement driver: sources, sampling, `xsa seeds`/`query`/`branches` stages, chaining, statistics, markdown tables |
| `receipts/chain_coverage/*.json` | derived data only: positions, counts, hashes. **No transcript bytes** |
| `receipts/chain_coverage/report.md` | the tables below, regenerated from the JSON by `chain_coverage.py report` |
| `SXGC_COLINEAR_CHAINING_ASK.md` | the document-only ask for a native `xsa chain` |

The chaining rule is minimap2-family sparse DP with two simplifications, both
registered in code and tested:

* a chain's score is **covered query bytes minus gap penalties**
  (`diag_penalty = 1.0` per byte of unexplained diagonal drift, `gap_extend = 0.25`
  per byte of gap on each axis), so overlapping anchors cannot inflate it;
* anchors may overlap by up to 8 bytes and only the uncredited part of the later
  anchor counts, because `xsa seeds` does emit partially overlapping seeds.

Correspondence is the part that makes a chain a positional pointer rather than a
bag of seeds. `correspondence_at(chain, query_pos)` returns one of

* `matched` — the query position lies inside an anchor, so the corpus byte is
  verified equal (this kind can never occur *at* the prediction site, whose byte is
  not in the context);
* `abutting` — the chain's right edge is exactly the prediction site;
* `extrapolated` — the chain's diagonal is continued past its last anchor, with the
  extrapolation distance returned to the caller and a caller-set budget.

`best_correspondence` prefers `matched` > `abutting` > `extrapolated` and ties by
gap distance, score, then corpus position, so the answer is deterministic.

## 2. Sources and sampling

**Transcript.** One real agent conversation, first 1,000,000 bytes of its textual
content in file order:

| field | value |
|---|---|
| file | `/home/erikg/.pi/agent/sessions/--home-erikg-emender--/2026-08-21T16-33-44-921Z_01a0252b-d5d9-7ec9-8dba-564a2c69de4a.jsonl` (150,170,061 bytes) |
| file SHA-256 | `55f038dfa6934faf5f45e23a916e99077a6b5b36d24e1cf01ca52a3c355732b4` |
| derived stream | 1,000,000 bytes, SHA-256 `3b3ead83b2c79ba91644f85bc1d12f1f0a55135f8c39d500f7a10969efc7071b` |
| message spans in the first 1 MB | 363 (assistant 155, toolResult 192, user 22) |
| blocks used | text 311, thinking 150, toolCall arguments 193 |
| excluded from sampling | 11,728 bytes (each side of a message boundary within 32 bytes) |
| eligible bytes | 988,272 |

The stream is the concatenation, in file order, of block text, thinking text,
tool-call arguments (JSON) and tool-result output, UTF-8 encoded. **Message
boundaries are kept out of band**, never inserted into the stream, so a sampled
position can be required to sit at least 32 bytes inside one message; positions
that do not are excluded and counted (1.2% of the prefix). Images, roles,
timestamps and JSON framing are not in the stream. No transcript content appears in
this receipt or in the JSON: only hashes, sizes, counts, offsets and derived
statistics.

**Pile contrast.** 150 byte positions sampled uniformly (without replacement) in
`[1024, 100000000000 − 32)` of `/mnt/nvme2n1/erikg/pile.txt`, read-only by offset;
the first 1 KiB excluded exactly as `CONTEXT_DEPTH.md` does. Contexts are read from
the corpus, not from the container, and every query is against
`/mnt/nvme2n1/erikg/P100c/pile-head100.sxi2` (63,163,489,696 bytes, mtime 2026-11-05).

Both panels use Python `Random(20261113)` and 150 positions (30 for the
wide-anchor panel). Positions are stored as integers in the JSON receipts.

## 3. Method: the four stages, per sampled position

Band = the 32 bytes immediately before the sampled position `p` (so `p` is the
prediction site and `p` itself never enters a query). Depths 2/4/8/16/32 are
suffixes of that one band. Per sampled position:

1. **seeds** — `xsa seeds --target-mult 50 --limit 0` on the 32-byte band. One
   invocation per band (`--limit 0` does no locate walk). The output rows give the
   seed's band offset, solved length and multiplicity; the seed *bytes* are taken
   from our own band, never from the CLI's echoed string (which is lossy for
   non-UTF-8 corpus bytes).
2. **site seeding** (a second, deliberately generous seeding mode) — for each
   depth, the longest suffix of the context ending exactly at `p` whose
   multiplicity is ≥ 2, found from a length ladder `L ∈ {2,4,6,8,10,12,16,20,24,32}`
   solved with `xsa query --count`. This is the seeding that makes an *abutting*
   correspondence possible at all; `xsa seeds` alone almost never produces it.
   Two panels are reported: `scan` (stage 1 only) and `scan+site` (stage 1 ∪
   stage 2).
3. **anchors + chain** — all distinct seeds are located in one batched
   `xsa query --patterns … --limit 8` call (8 = `xsa seeds`' own default sample).
   Anchors are `(query_position, corpus_position, length)`; `chaining.py` runs the
   DP over the anchors of each (position, depth, mode).
4. **fork** — for each correspondence, the corpus context of the same depth ending
   at the aligned corpus position is sent to one `xsa branches --serve -j 8 -k 256`
   session, giving the exact next-byte histogram at the site. The model's actual
   next byte is then located in it: present/absent, 1-based rank among distinct
   bytes, mass, and the histogram's distinct-byte count and top-1 mass.

Three honesty rules are applied inside the measurement:

* **The source occurrence is removed where it is in the histogram.** For the Pile
  panel the query text *is* corpus text, so the source occurrence at `p` sits
  inside the index and contributes its own next byte to any histogram taken at a
  site with the same context. That one occurrence is identified exactly (the two
  contexts are byte-compared) and subtracted (`source occ. removed` column). 97 of
  118 Pile depth-16 sites needed this.
* **Self sites are excluded from the fork panel.** A correspondence that lands
  within the query's own band is the source answering for itself, not evidence
  from elsewhere; the `cross-site` column is that honest denominator. No such
  exclusion applies to the transcript (its bytes are not in the index).
* **`branches --cross-doc`.** The pile-head100 container carries no m15
  document-start member, so the boundary-aware histogram is unavailable; the
  conflated answer is used, the same convention `CONTEXT_DEPTH.md` used. END
  branches and boundary-crossing matches are not part of the reported histogram.

## 4. Commands

```sh
# unit tests
cd experiments/expressivity_tasks/x43_fork_resolved
/home/erikg/emender/.venv/bin/python -m pytest test_chaining.py -q      # 16 passed

# main panels (150 positions each, xsa seeds' default 8 positions per seed)
/home/erikg/emender/.venv/bin/python chain_coverage.py measure \
    --source transcript --positions 150 --out receipts/chain_coverage/transcript.json
/home/erikg/emender/.venv/bin/python chain_coverage.py measure \
    --source pile       --positions 150 --out receipts/chain_coverage/pile.json

# wide-anchor panel (30 positions, 64 positions per seed)
/home/erikg/emender/.venv/bin/python chain_coverage.py measure \
    --source transcript --positions 30 --limit 64 --out receipts/chain_coverage/transcript_wide.json
/home/erikg/emender/.venv/bin/python chain_coverage.py measure \
    --source pile       --positions 30 --limit 64 --out receipts/chain_coverage/pile_wide.json

# tables (recomputed from the stored per-cell rows; no measurement, no xsa call)
/home/erikg/emender/.venv/bin/python chain_coverage.py report \
    --transcript receipts/chain_coverage/transcript.json \
    --pile receipts/chain_coverage/pile.json > receipts/chain_coverage/report.md
```

Raw CLI invocations issued by the driver (exact argv, `xsa` =
`/home/erikg/sxgc/xsa/target/release/xsa`, binary SHA-256
`eb3317d9c7ba78387bdbb2c34c549b40a28478128963116f76c649782da2a084`):

```sh
xsa seeds "$CONTAINER" - --target-mult 50 --limit 0 --jsonl           # band on stdin
xsa query "$CONTAINER" --patterns /tmp/chain_coverage_patterns.bin --limit 8 --output jsonl
xsa query "$CONTAINER" --patterns /tmp/chain_coverage_counts.bin --count
xsa branches "$CONTAINER" --serve -j 8 -k 256 --cross-doc            # framed stdin/stdout, u32 length + bytes
```

## 5. Cost (first-class result: per-query cost decides per-turn viability)

Warm page cache, serial, CPU only, no GPU; no other process held the container open
during these runs (`fuser` empty afterwards), so these are not contention-contaminated.

| operation | measured | note |
|---|---|---|
| `xsa seeds --limit 0`, 32-byte band | **80 ms per invocation** (150 calls: 12.2 s transcript, 12.6 s Pile) | scan only, no locate; cost is process start-up + the interval solve |
| `xsa seeds` default `--limit 8`, 400-byte text | **142.6 s** | locate-bound: 512 positions, 4.08 M LF steps, `scan_ms = 281` |
| first (cold) `xsa seeds` invocation, 200-byte text | **>5 min, 3.25 GB read** | recorded in `~/sxgc/NOTE_FROM_EMENDER_XSA_SEEDS_COST.md` |
| `xsa query --count`, 1,004 patterns | **0.09 ms per pattern** (0.21 s wall) | widths only; this is why the L-ladder is nearly free |
| `xsa query --limit 8` | **38 ms per located position** | 84 positions in 3.2 s |
| `xsa query --limit 64` | **34 ms per located position** | 3,697 positions in 125 s |
| `xsa branches --serve -j 8 -k 256` | **2.6 ms per context** | 426 contexts in 1.08 s |
| transcript panel, total | **≈ 74 s** | 12.2 s seeds + 0.18 s widths + 61.3 s locate (2,331 positions) + 0.6 s branches |
| Pile panel, total | **≈ 214 s** | 12.6 s + 0.23 s + 200.5 s locate (5,386 positions) + 1.1 s branches |

The locate (LF) walk is ~400× more expensive per answer than an interval width and
dominates everything. `seeds` with positions enabled is the same command that a
per-position loop must not use: its own `--limit` default of 8 turns a 400-byte
query into 2.3 minutes.

## 6. Results

### 6.1 Support-depth ladder (no chaining; the R2.3 loss condition made explicit)

Fraction of sampled positions whose **context suffix of length L occurs at least
twice** in the 100 GB corpus (`xsa query --count`; no locate). This is the upper
bound on correspondence coverage for any chaining method at that L.

| suffix length L | transcript mult≥2 | transcript mult≥50 | Pile mult≥2 | Pile mult≥50 |
|---:|---:|---:|---:|---:|
| 2 | 142/150 = 94.7% | 142/150 | 147/150 = 98.0% | 147/150 |
| 4 | 138/150 = 92.0% | 135/150 | 145/150 = 96.7% | 145/150 |
| 6 | 120/150 = 80.0% | 110/150 | 143/150 = 95.3% | 139/150 |
| 8 | 103/150 = 68.7% | 73/150 | 137/150 = 91.3% | 127/150 |
| 10 | 73/150 = 48.7% | 48/150 | 130/150 = 86.7% | 111/150 |
| **12** (`seeds` default `--min-len`) | **55/150 = 36.7%** | 30/150 | **124/150 = 82.7%** | 99/150 |
| 16 | 23/150 = 15.3% | 12/150 | 104/150 = 69.3% | 64/150 |
| 20 | 11/150 = 7.3% | 5/150 | 85/150 = 56.7% | 37/150 |
| 24 | 3/150 = 2.0% | 2/150 | 60/150 = 40.0% | 11/150 |
| 32 | 1/150 = 0.7% | 1/150 | 29/150 = 19.3% | 3/150 |

Support is monotone in the sense that a supported length implies support at every
shorter length, so the union of support over `L ∈ [12, d]` equals support at
`L = 12`; that is why every abutting-correspondence fraction below (37.3%, 82.7%)
reproduces this row exactly.

### 6.2 Chaining, correspondence and fan-out

Each cell is one (sampled position, depth, seeding mode). `joint chains` counts
cells whose best chain has more than one anchor. `abutting site` counts cells with a
chain edge exactly at the prediction site; `any site` allows up to 16 bytes of
diagonal extrapolation. Percentiles are nearest-rank over cells, zeros included.

#### transcript source (n = 150 positions)

**seeding mode `scan`** (only the `xsa seeds` output)

| depth | anchors p50/p90/p99/max | chains p50/p90/p99 | regions p50/p90/p99 | best chain score p50/p90/p99 | joint chains (>1 anchor) | abutting site | any site (≤16 B gap) | cross-site sites |
|---:|---|---|---|---|---:|---:|---:|---:|
| 2 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 4 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 8 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 16 | 0/8/16/16 | 0/8/16 | 0/8/15 | 0.0/12.0/14.0 | 0/150 | 1/150 | 20/150 = 13.3% | 20 |
| 32 | 8/32/48/64 | 8/32/48 | 7/30/48 | 12.0/19.0/23.0 | 0/150 | 1/150 | 75/150 = 50.0% | 75 |

**seeding mode `scan+site`** (stage 1 ∪ the site-ending suffix at the longest
supported length ≥ 12)

| depth | anchors p50/p90/p99/max | chains p50/p90/p99 | regions p50/p90/p99 | best chain score p50/p90/p99 | joint chains (>1 anchor) | abutting site | any site (≤16 B gap) | cross-site sites |
|---:|---|---|---|---|---:|---:|---:|---:|
| 2 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 4 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 8 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 16 | 0/10/21/24 | 0/10/21 | 0/10/18 | 0.0/16.0/16.0 | 0/150 | 56/150 | 60/150 = 40.0% | 60 |
| 32 | 8/40/56/69 | 8/40/56 | 8/32/52 | 13.0/21.0/30.0 | 0/150 | 56/150 | 90/150 = 60.0% | 90 |

Fan-out (`regions`, chains clustered by corpus gap ≤ 2000, the `xsa fanout`
region-gap convention): at depth 32, p50/p90 = 8/32 competing regions under
`scan+site`, 7/30 under `scan`. Anchors are spread across dozens of regions while
chains essentially never span two of them: with 8 SA-adjacent positions per seed and
a handful of seeds per band, the anchor set is a bag of regions, not a colinear
lattice.

#### Pile contrast panel (n = 150 positions)

**seeding mode `scan`**

| depth | anchors p50/p90/p99/max | chains p50/p90/p99 | regions p50/p90/p99 | best chain score p50/p90/p99 | joint chains (>1 anchor) | abutting site | any site (≤16 B gap) | cross-site sites |
|---:|---|---|---|---|---:|---:|---:|---:|
| 2 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 4 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 8 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 16 | 0/8/16/16 | 0/8/16 | 0/8/15 | 0.0/12.0/15.0 | 0/150 | 0/150 | 23/150 = 15.3% | 23 |
| 32 | 32/48/64/72 | 32/48/64 | 24/45/61 | 18.0/24.0/27.0 | 4/150 | 0/150 | 129/150 = 86.0% | 127 |

**seeding mode `scan+site`**

| depth | anchors p50/p90/p99/max | chains p50/p90/p99 | regions p50/p90/p99 | best chain score p50/p90/p99 | joint chains (>1 anchor) | abutting site | any site (≤16 B gap) | cross-site sites |
|---:|---|---|---|---|---:|---:|---:|---:|
| 2 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 4 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 8 | 0/0/0/0 | 0/0/0 | 0/0/0 | 0.0/0.0/0.0 | 0/150 | 0/150 | 0/150 = 0.0% | 0 |
| 16 | 8/11/22/24 | 8/11/22 | 8/9/17 | 16.0/16.0/16.0 | 0/150 | 124/150 | 129/150 = 86.0% | 118 |
| 32 | 32/53/64/75 | 32/53/64 | 29/48/61 | 23.0/32.0/32.0 | 11/150 | 124/150 | 145/150 = 96.7% | 102 |

The Pile panel's own-copy answers dominate its `abutting site` column: of the
145 sites at depth 32, 43 land inside the query's own band. That column is the
`xsa seeds`-plus-ladder number; only the `cross-site` column is comparable to the
transcript.

### 6.3 The fork at the aligned site

`actual byte present` = the model's true next byte occurs as a branch of the
histogram at the aligned corpus site; `top-1` = it is the most frequent branch.
`source occ. removed` counts sites where the source occurrence had to be subtracted
(Pile only). `chance` = the same byte-equality question answered by the stream's
majority next byte over the same cross-site positions.

#### transcript

| mode | depth | sites | source occ. removed | fork width p50/p90/p99/max | distinct next bytes p50/p90/p99 | actual byte present | top-1 branch is the actual byte | rank when present p50/p90 | mass when present p50/p90 | abutting only: present | abutting only: top-1 | abutting only: fork width p50/p90/p99 | chance: actual = stream majority |
|---|---:|---:|---:|---|---|---:|---:|---|---|---:|---:|---|---:|
| scan | 16 | 20 | 0 | 1/11/58/58 | 1/1/2 | 3/20 | 3/20 | 1/1 | 1.0/1.0 | 1/1 | 1/1 | 2/2/2 | 1/20 |
| scan | 32 | 75 | 0 | 1/2/14/21 | 1/1/1 | 9/75 | 9/75 | 1/1 | 1.0/1.0 | 1/1 | 1/1 | 1/1/1 | 13/75 |
| scan+site | 16 | 60 | 0 | 4/1336/65496/32490771 | 1/9/49 | 37/60 | 34/60 | 1/1 | 1.0/1.0 | 36/56 | 33/56 | 5/2202/65496 | 9/60 |
| scan+site | 32 | 90 | 0 | 1/2/9/254 | 1/1/1 | 32/90 | 32/90 | 1/1 | 1.0/1.0 | 28/56 | 28/56 | 1/2/9 | 16/90 |

#### Pile

| mode | depth | sites | source occ. removed | fork width p50/p90/p99/max | distinct next bytes p50/p90/p99 | actual byte present | top-1 branch is the actual byte | rank when present p50/p90 | mass when present p50/p90 | abutting only: present | abutting only: top-1 | abutting only: fork width p50/p90/p99 | chance: actual = stream majority |
|---|---:|---:|---:|---|---|---:|---:|---|---|---:|---:|---|---:|
| scan | 16 | 23 | 2 | 3/25/67/67 | 1/4/8 | 3/23 | 2/23 | 1/4 | 1.0/1.0 | 0/0 | 0/0 | – | 2/23 |
| scan | 32 | 127 | 1 | 1/2/37/43 | 1/1/2 | 10/127 | 10/127 | 1/1 | 1.0/1.0 | 0/0 | 0/0 | – | 21/127 |
| scan+site | 16 | 118 | 97 | 105/15675/2297712/762642499 | 2/25/83 | 106/118 | 85/118 | 1/3 | 0.98/1.0 | 106/113 | 85/113 | 166/21457/2297712 | 17/118 |
| scan+site | 32 | 102 | 23 | 1/7/794133/425515414 | 1/1/16 | 64/102 | 60/102 | 1/1 | 1.0/1.0 | 64/81 | 60/81 | 1/9/794133 | 19/102 |

(The top-1 branch mass percentiles, and every other per-cell field, are in
`receipts/chain_coverage/report.md` and the JSON receipts.)

Reading these honestly:

* the transcript's depth-16 `scan+site` sites are **short exact repetitions**
  (fork width median 4 occurrences of a 12–16-byte context, median 1 distinct
  continuation). 33/56 of those single-continuation sites match the transcript's
  actual next byte. This is phrase-level determinism in a 100 GB corpus, i.e. exactly
  the R2.1 "similarity evidence, not a next-byte promise" situation. It is not
  retrieval of a shared *conversation*.
* the scan-only mode (what the design's `seeds` stage alone can deliver) produces a
  usable site at only 1/150 transcript positions; its extrapolated sites are near
  chance (9/75 = 12%, chance 13/75).
* the Pile's high rates (106/118 present at depth 16) ride on contexts that occur
  ~10²–10⁸ times. They say the corpus is repetitive; by §R2.1 they must not be used
  as an argument for anything.

### 6.4 Transcript vs Pile, side by side

| metric | transcript | Pile (cross-site only) |
|---|---|---|
| positions sampled | 150 | 150 |
| `L = 12` context suffix occurs ≥ 2× | 55/150 = 36.7% | 124/150 = 82.7% |
| `L = 16` | 23/150 = 15.3% | 104/150 = 69.3% |
| `L = 32` | 1/150 = 0.7% | 29/150 = 19.3% |
| abutting correspondence, depth 16 (scan+site) | 56/150 = 37.3% | 124/150 = 82.7% |
| abutting correspondence, depth 32 (scan+site) | 56/150 = 37.3% | 124/150 = 82.7% |
| any site ≤ 16 B extrapolation, depth 32 | 90/150 = 60.0% | 145/150 = 96.7% |
| competing chain regions at depth 32, p50/p90 | 8/32 | 29/48 |
| chains joined (>1 anchor), depth 32, limit 8 | 0/150 = 0% | 11/150 = 7.3% |
| chains joined, depth 32, limit 64 (n = 30) | 2/30 = 6.7% | 13/30 = 43.3% |
| fork top-1 = actual byte, abutting, depth 16 | 33/56 = 58.9% | 85/113 = 75.2% |
| fork top-1 = actual byte, abutting, depth 32 | 28/56 = 50.0% | 60/81 = 74.1% |
| chance (actual = stream majority byte) | 9/60 = 15.0% | 17/118 = 14.4% |

Wide-anchor panel (30 positions, `--limit 64`, same depths): transcript anchors
p50/p90 = 64/362 at depth 32 with 45/292 regions and 2/30 joined chains; Pile
249/378 anchors, 188/342 regions, 13/30 joined chains. Widening the sample
multiplies the anchor count and the region count but does not turn the transcript's
anchor sets into colinear lattices.

## 7. Validation and consistency checks

* **`seeds` multiplicity vs `xsa query --count` width**: 195/195 agreeing rows
  (transcript) and 475/475 (Pile). The remaining 38/233 and 44/519 rows are seeds
  whose bytes contain a control byte; `--count` emits TSV with no escaping, so those
  rows were skipped rather than guessed (counted in the JSON as
  `distinct_patterns_scanned` minus `mult_parity_cells_checked`).
* **Support identity**: the abutting-correspondence fraction equals the `L = 12`
  support fraction plus the scan-mode abutting cases — 55 + 1 = 56 (transcript),
  124 + 0 = 124 (Pile). The chaining layer therefore loses nothing that §R2.3 says
  it cannot lose: at depth ≥ 12 the only reason for "no site" is "no supported
  suffix of length ≥ 12".
* **Determinism**: the DP is order-independent (unit test shuffles the anchor list
  and compares chains); `report` recomputes every table from the stored per-cell
  rows without touching the index, so the tables cannot drift from the receipt.
* **Chaining correctness where ground truth exists**: `test_chaining.py` builds a
  query that shares a 40-byte run with a known corpus offset plus three distractor
  anchors on other diagonals; the chain recovers the run and the correspondence
  names exactly `corpus_offset + query_length`. Further tests cover crossing
  anchors, an inconsistent diagonal, overlap not inflating the score, the
  extrapolation budget, determinism under shuffle, and fork rank/mass.
* **No transcript content in the receipt**: cells carry only integers, booleans and
  mode labels (keys: `anchors, best_anchors, best_chain_has_site_anchor, best_score,
  chains, depth, extrapolation, fork, fork_drops_source, fork_width, hit_marginal,
  kind, mode, regions, sample, seeds, site, site_is_source, site_level,
  site_seed_length`). Seed bytes exist only in process memory and in a temporary
  patterns file under `/tmp`, which is removed by the next run.

## 8. What this does not establish

* **No claim that chaining improves prediction.** The fork's rank-1 hit rate was not
  compared against a language model on the same positions. It is compared against a
  single degenerate baseline (the stream's majority next byte).
* **One transcript.** One agent session, first 1 MB, in one workspace. Byte-uniform
  positions over that prefix, not conversation turns, not tokens, not documents. A
  different transcript could move the 36.7% materially; 150 positions give wide
  binomial intervals (±7–8 percentage points at the 95% level for a 37% rate).
* **The transcript stream is a reconstruction.** It is the textual block content in
  file order, without role markers, JSON framing, images or timestamps, and it
  includes thinking blocks. It is close to, not identical with, the bytes a model
  consumed.
* **Byte prediction is not the task.** "The top-1 branch equals the actual next byte
  at a 12–16-byte site" says nothing about token-level or semantic quality, and a
  corpus that repeats a phrase is not thereby answering a conversation's question.
  Per §R2.1 the repetition frequency of these corpora is not an argument in either
  direction.
* **Document structure is absent.** The container has no m15 document-start member,
  so `branches --cross-doc` conflates document boundaries and no document-level
  support, document-disjointness or cross-document claim is made. Region clustering
  here is a corpus-gap heuristic (`gap ≤ 2000`), not document identity.
* **Positions are not complete.** `xsa query --limit 8/64` returns the first N rows
  of the SA interval, which are SA-adjacent, so anchors inside one region are biased
  to that region's lexicographic edge; the fan-out counts are lower bounds on the
  true number of regions and the joined-chain counts are lower bounds on what a
  complete anchor set would produce. This is the single largest measurement
  limitation and the reason `SXGC_COLINEAR_CHAINING_ASK.md` asks for region-first
  enumeration.
* **No GPU, no model, no training, no service.** Reader-only: nothing was modified
  in `~/sxgc`, `pile.txt`, the P100c container, `receipts/`, the preregistrations or
  `runs_150m`.
* **Not a queue, not a scheduler, not a Slurm receipt.** No partition/QoS evidence
  applies: everything ran on this host, CPU only.

## 9. Decision this supports

Under §6's decision rule, the measurements say the operator is **not** dead: at the
registered `L = 12`, 37% of byte-uniform positions in a real agent transcript have
a site-abutting correspondence to a 100 GB corpus, and where one exists the corpus
continuation names the actual next byte at rank 1 half to two-thirds of the time
(the sites are short, near-deterministic repetitions). It is also **not** yet a
retrieval operator: the anchor sets the current CLI can hand over almost never
produce a chain with more than one anchor (0/150 transcript, 11/150 Pile at depth
32; 2/30 and 13/30 with an eight-times wider position sample), so the "positional
pointer" is usually one seed's coincidence rather than a corroborated alignment.
The path to real colinear chains is the region-first anchor enumeration in the ask
document. What remains unmeasured is whether a chain that *does* span a
region changes the fork's rank — that is the next measurement once a native anchor
table exists (prototype: raise `--limit` and locate per region).
