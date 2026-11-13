# NOTE to sxgc: `xsa seeds` has no warm path, and the fast path already exists

**From:** emender side · **Date:** 2026-11-13 · **Measured against:**
`/mnt/nvme2n1/erikg/P100c/pile-head100.sxi2` (63.16 GB), binary
`/home/erikg/sxgc/xsa/target/release/xsa`

## What we hit

One invocation, for coverage measurement over a real transcript:

```
xsa seeds /mnt/nvme2n1/erikg/P100c/pile-head100.sxi2 /tmp/q1.txt --jsonl
```

with `q1.txt` = **200 bytes** of query text. Observed:

- **3.25 GB read** from the container before completion (`/proc/<pid>/io` rchar,
  advancing ~240 MB/s, process in `D` state, `/tmp` output 428 bytes)
- wall time **> 5 minutes per invocation**
- 3.3 M `read(2)` calls (`syscr`) for one 200-byte question

So a per-position measurement loop is impossible: our design needs on the order of
100–500 sampled positions per transcript, and at minutes per call that is hours-to-days of
disk touch for a first feasibility number.

## Diagnosis, from your own receipts

`seeds` is a **cold, locate-bound** command. It has no resident/warm variant:
`mem --serve`, `branches --serve`, `serve` (bind), and `consult` have one; `seeds` does not
(`xsa/src/seeds.inc.rs::seeds_command` parses no `--serve`; only `mem.inc.rs` and
`branches.inc.rs` do).

And you have already measured exactly how much that costs, and already solved it once:

- `RESULTS_CHI_CONSULT.md` **G4_RATE**: `positions_per_s=6173.0`,
  `baseline=13.7`, **`multiple=450.6x`**, `peak_rss_gb=7.999 cap_gb=8`,
  99.08 % of rows answered with **zero locates** under the separator-framing +
  chi-proportional layers (trie 11.6 µs p50; width-1 3.0 ms p50; forced/rawfork 5.9 ms).
- `e94a600` — *"ctx: batch mode — the only way a <=50 us target can be measured at all."*
  The same lesson applies to `seeds`.

So this is not a request for new capability so much as a request to apply the
already-proven pattern to one more command.

## What we ask for

### A. A batch or resident mode for `seeds`

Either form is fine:

- a framed stdin/serve mode mirroring `xsa mem <container> --serve -j N`
  (u32 length + bytes per window), keeping the index and the framing verification warm;
- or an explicit **windowed input file** where window boundaries are declared
  (not inferred), with `--stride` semantics documented, so one process answers all
  windows of a transcript.

Record the cold-vs-warm rate the way G4 does, so the cost is on the record.

### B. Chain-ready anchors in one pass

Our operator chain is `seeds → colinear chain → correspondence → fork`. The chaining
is ours (we confirmed `xsa/src/chain.rs` is the *construction* chain, not colinear
chaining, so nothing conflicts). What we need from `seeds` is the anchor table in one
pass, per window:

```
window_id, query_offset, corpus_doc, corpus_offset, match_len, multiplicity
```

That is the input to the colinear DP. Two things matter beyond convenience:
multiplicities must be **exact aggregates**, not derived from a truncated hit list
(`-k`/`--limit` are output caps, not support thresholds — worth stating in the help,
because it reads as a support threshold today), and reproducibility requires a
documented **tie order** so two runs produce byte-identical anchor tables.

### C. A cost table for `seeds`, like the consult layer table

Which layer serves which query class, at what p50, cold and warm. The consult table is
what let us reason about per-turn budget; we need the same object for `seeds`.

## Why it matters to us

The suffix-kernel design evaluates, per position, the anchors and the **fork at the
aligned site** over a ~1 MB context. If the per-call cost is minutes, then (a) the
feasibility study cannot be run, and (b) per-turn dynamic indexing is dead on arrival.
Your 450.6× warm result says the fix is known: the answer should be index-proportional,
not locate-bound.

## What we are doing meanwhile

- Batching all windows of a transcript into a single invocation per batch, and recording
  the per-query cost as a first-class result in our design receipt.
- **Not** modifying anything in this repo. All XSA-side work is requested here as a
  document, per the operator's instruction.

## Our measurement receipt

`experiments/expressivity_tasks/x43_fork_resolved/RESULT_XSA_SEEDS_COST.md`
(commands, `/proc` evidence, and the wall/read numbers).
