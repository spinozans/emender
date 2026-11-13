# Measured cost of a cold `xsa seeds` invocation (2026-11-13)

Small receipt backing `SXGC_SEEDS_COST_NOTE.md` (delivered to `~/sxgc`).

## Command

```
xsa seeds /mnt/nvme2n1/erikg/P100c/pile-head100.sxi2 /tmp/q1.txt --jsonl
```
`/tmp/q1.txt` = 200 bytes of query text (a single short window of real prose).

## Observation

| quantity | value | source |
|---|---|---|
| bytes read from the container | **3.25 GB** and still rising at cutoff | `/proc/<pid>/io` `rchar` (3,192,680,927 → 3,252,485,729 over 5 s ⇒ ~240 MB/s) |
| read syscalls | 3,316,516 → 3,377,557 | `/proc/<pid>/io` `syscr` |
| process state | `D` (uninterruptible IO) | `ps -o stat` |
| RSS | 440 MB | `ps -o rss` |
| bytes written | 428 | `/proc/<pid>/io` `wchar` |
| wall time | **> 5 min**, not complete at observation | `ps -o etimes` = 284 s |

## Interpretation

`seeds` is a cold, locate-bound command with no resident/warm path (`seeds.inc.rs`
parses no `--serve`; `mem.inc.rs` and `branches.inc.rs` do). Cost scales with
container touch, not with query size — 3.25 GB of reads for a 200-byte question.

The already-measured warm path on the same container is `xsa consult` with the
separator-framing zero-locate layers: **6,173 positions/s warm, 450.6× the 13.7/s
baseline** at the same 8 GB cap (`~/sxgc/RESULTS_CHI_CONSULT.md` G4_RATE), with 99.08 %
of rows answered with zero locates.

## Consequence for our design work

Per-position coverage measurement (100–500 sampled positions over a ~1 MB transcript)
cannot be done by invoking `seeds` per position. Our chaining lane therefore batches all
windows into one invocation per batch and records the cold-vs-warm cost as part of its
result. The request for a batch/resident `seeds` mode is in `SXGC_SEEDS_COST_NOTE.md`.

This is a measurement of the existing binary, not a modification of it: no file in
`~/sxgc` was changed to produce it.
