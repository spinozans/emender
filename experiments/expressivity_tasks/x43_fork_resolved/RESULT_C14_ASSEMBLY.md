# C14 assembly: exact binary shards; end-to-end target not achieved

The premise that the entire 5.7x difference is feature assembly is contradicted
by the measured pipeline. Serving already uses binary frames, not JSON. A
selected feature requires about 6.8 native queries, not one. Removing output
assembly alone cannot make selected features/s equal native queries/s.

Checked the continuation's committed receipt at `92f7544f` before doing work:
its sparse-cache parity covers the same 1,034,056 positions, with no complete
train/valid cache. Its measured rate was 6,088.688 positions/s. Its working-tree
files and caches were left untouched. No full cache extraction was duplicated.

New `c14_assembly.py` supplies `assemble(rows)` and bounded `Shards.write` for
serving and extraction callers, plus direct native extraction with `--extract`.
It scatters sparse probabilities into contiguous float64 numpy batches and
writes independently memory-mappable `.npy` feature and identity shards, without
per-position dense Python lists or JSON output. Python integer division and
math.log1p are retained to preserve bits; numpy log/reduction changes are avoided.
The existing selector, branch decoder, terminal policy and geometry are unchanged.
No multiprocessing is introduced: measured assembly is already faster than
serving, and attended implementation/debugging remains single-threaded.

## Instrumented profile

Receipts are in `receipts/c14_assembly/`. At j128, 512 source-order windows
contained 64,250 positions and required 438,542 queries (6.825556/position).
Disjoint wall stages, seconds:

| Stage | Seconds |
| --- | ---: |
| Source identity/hash checks and windows | 1.985619 |
| Query calls, including pipe waits, framing and decode | 17.775975 |
| Selector excluding query calls | 1.089416 |
| Old JSON + gzip output | 2.397547 |
| New dense feature assembly | 0.243131 |
| New npy features + identity output | 0.091840 |
| Server shutdown | 13.547459 |

Decode is a nested 2.297614 seconds within query calls, not an extra stage.
The residual query time includes native work, pipe wait and Python framing;
it is not a measurement of pure native CPU time. Total 37.133482 seconds
includes both old and new output paths and startup/bookkeeping. This small
cold/warm-dependent profile is not a full-size throughput benchmark. The j32
128-window profile is retained separately; it also spends most query-path wall
time outside decoding. Neither profile supports blaming all cost on assembly.

Within the j128 profile, even eliminating the entire measured decoder would
leave 15.478361 seconds of query/pipe work, about 4,151 positions/s before
selector, source and output costs. This is an observed workload ceiling for
that run, not a universal performance bound. Query count/rate normalization is
necessary before comparing against the claimed 22,768 contexts/s. The old
committed preflight itself reported 31,530.865 native queries/s and 6.802295
queries/position, not a one-query feature operation.

## Exact oracle replay and assembly rate

All 1,034,056 oracle positions were converted and verified, including persisted
feature bits and document/start/position order. Both feature streams have SHA256
`70971034f5cf8ddf79e0f91c8bb578d09ced536e1f0a3d8a0a89557dc93b4c73`.
The input gzip hash matches the continuation's committed parity receipt:
`2eaa4a4ec590171ceef69e89f7c734687e21074eb5ff06e73d3146861cd11bba`.

Replay stage timings: gzip + JSON read 8.869376 seconds; old dense list feature
construction plus array conversion 16.506783 seconds; new assembly, shard
serialization and file hashing 8.515626 seconds (**121,430 features/s**).
The latter is assembly-only, not extraction throughput. Full validation replay
including both implementations, comparisons and hashing took 53.115034 seconds.
There is no claim of a 5.7x end-to-end gain; measured assembly/serialization is
1.94x faster than the old feature-list conversion stage, despite doing more I/O.

The native 261-fixture independent bytes.find oracle was rerun against the
existing fixture index; counts, intervals, EOF and new selected feature bits
pass. Nine unit tests pass, including terminal-only zeros, uint64 division,
all bytes, persisted replay and direct extraction across a shard boundary.
The xsa binary hash is unchanged and its isolated clone is at 88d4e07.
An exploratory `branches --help` returned exit 1; protocol documentation confirms
binary --serve, so no unsupported transport change was made.

## Continuation handoff and limitations

The replayed shard directory is
`/home/erikg/c14_consultation_features/assembly_oracle_8192`.
Its manifest and per-file hashes are copied into the receipts. Consumers use
`np.load(path, mmap_mode='r')` for features and matching uint64 identity triples.
The directory is about 2.1 GiB: dense float64 is much larger than sparse gzip.
Use bounded shards for repeated evaluation/training reads; do not allocate a
full-corpus dense cache without a storage budget. A manifest is written only
on successful completion, and existing directories are rejected.

To run a new bounded native pass (new output directory required):

```bash
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_assembly --extract --windows 8192 --jobs 128 --output /path/to/new/shards
```

This direct mode shares the exercised selector and writer; its chunk wiring is
unit-tested, but a full 8,192-window direct-native shard pass was not run.
No held-out features, complete training cache, training integration or
>10,000 selected features/s end-to-end result is claimed. Sparse continuation
storage/reuse remains complementary. Further end-to-end gains require reducing
selector queries or native/pipe time; assembly mechanics alone cannot establish
the requested result. The requested success commit title would be inaccurate.

contact_supervisor is absent from the available tool catalog; no intercom
coordination was possible. This receipt is the continuation handoff. No worker,
service, subagent, GPU lease, .pt commit or change to ~/sxgc was used.
