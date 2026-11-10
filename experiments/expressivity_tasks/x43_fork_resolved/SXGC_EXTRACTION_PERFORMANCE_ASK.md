# NOTE (from the emender side): extraction performance — bounded-memory streaming, training-rate consultation

**For:** the sxgc agents · **From:** the Emender/X43 program · **Re:** `xsa` serving-path
performance against pile-head100 · **Date:** 2026-11-09

## What we run today (all numbers receipted in our repo)

Per-position 260-scalar-equivalent consultation features against
`/mnt/nvme2n1/erikg/P100c/pile-head100.sxi2` (63.16 GB container, χ = 24.19 G,
first 100e9 bytes of pile.txt), for training our byte-level LLM arms.

The measured chain:

| Stage | Rate | Note |
|---|---|---|
| `xsa branches --serve` raw query | **22,768 contexts/s** | verified exact vs brute force (17/17 oracle contexts) |
| C-kernel assembly only | **121,430/s** | feature-scalar assembly is cheap once rows are in hand |
| End-to-end (our python orchestration) | **~4,600–5,884/s** | the glue is the bottleneck, ~26× off assembly |
| Memory of the end-to-end service | **~400 GB RSS at 822M positions** | ≈ linear, ~490 B/position; **we stopped it; we will not restart this configuration** |

The service writes ~29 B/position to disk incrementally (24 GB banked, fsync'd,
journaled, resumable) — the disk/output design is fine. The RAM accumulation is in
the live path: our orchestration python (`flagship_union_service.py`) spawns the
serve binary with `jobs=128`, holds a 250k-entry query-reuse cache, and streams
16-window chunks. One of these is retaining ~490 B/position across positions —
our leading suspects are the serve child (per-connection/per-query allocations
with 128 jobs) or result retention on our side of the pipe. We can hand over the
service source, the journal, and console logs on request.

## The two asks

### A. Bounded-memory streaming extraction (cache-build mode)

A native path (CLI or daemon) that emits our per-position feature rows with a
**hard RSS budget (≤ 8–16 GB per process)**, streaming to disk with an append-only
journal and a resumable cursor, backpressure instead of buffering. Minimum bar:
match the current ~5k positions/s so we can finish the remaining 44% of our
flagship cache (644M positions). Target: see B.

### B. Training-rate live consultation (the real prize)

Sustain **≥ 150,000 positions/s end-to-end on one node** with bounded memory.
Why this number: a 150M–1.3B byte-level transformer consumes ~131k positions per
optimizer step (T2048 × batch 64, ≥ 1 step/s), so *training-time* consultation
(no pre-extracted cache at all) needs ≥131k/s continuous. At that rate:

- the remaining 44% of our cache finishes in **~1.2 hours** instead of ~39;
- consultation becomes a live training-time input (a reader swap in our arms —
  the consultation door is already interface-identical to the cached reader);
- generation-time consultation (a few queries per emitted token) is trivially
  covered — that regime already works at your 22.8k/s.

The gaps to close are both in scope: 22.8k → 150k (batching API amortizing
per-query IPC, multi-process/multi-socket serve, or whatever you see), and
121k assembly → end-to-end (fold our scalar assembly into the native path; the
assembly math is receipted cheap).

## Interface contract (we adapt on our side once, receipted)

- **Input:** a stream of windows — (document id, byte offset, left-context
  positions). We can emit JSONL or a binary batch protocol; our windows are
  pre-frozen and hash-bound, so cursor/resume is easy on both sides.
- **Output:** per-position rows. Either our compact layout (FLAGSHIPSPARSE2 —
  ~29 B/position on disk; spec in `flagship_cache.py`, we ship the reader) or
  your own documented row layout; we adapt the training-side reader once.
- **Two modes:** (a) cache-build (append-only + journal + resume cursor +
  fsync per chunk — our arms require crash-resumability), (b) live-serve
  (batch query/response over a socket; bounded queues; sustained ≥150k rows/s).
- **Determinism:** outputs must be byte-identical to the current verified
  chain. Before adopting anything we re-run the 17-context brute-force oracle
  parity and full compact-row parity (our standing discipline; failures are
  receipted as failures).

## Context and stakes

Our flagship experiment is six 1.3B-parameter byte-level arms (dense
transformer / GDN2 / emender variants ± consultation features from this
container), pre-registered and hash-bound; the consultation arms drink exactly
these features. The cache-build is 56% done and paused pending the
bounded-memory fix. The 150k/s live path also unlocks the deployment story we
want for the paper: consultation at training speed, not just generation speed.

Everything above is cross-referenced in
`/home/erikg/emender/experiments/expressivity_tasks/x43_fork_resolved/` — see
`SXGC_HANDOFF.md`, `receipts/flagship_stream/`, `RESULT_SPINE_DECISION.md`, and
the `C14_*` receipt series for the rate-chain history (15.7 → 3,972 → 121,430 →
5,884 end-to-end). We're happy to do a live session with profiling artifacts.

— the emender side (via the operator's attended session)

---

## ADDENDUM (same day): the algorithmic shape that hits the target — maintained match-state ladders

We checked our cache format: each position needs exactly **one query** — the next-byte
branch histogram of the spine after the last ≤1024 bytes (one context length, no
backoff; absent context = empty histogram). That makes consecutive positions
*incrementally related*:

- **Ramp phase** (context growing 1→1024): position t+1's context is position t's
  context **plus one byte**. A maintained match-state takes one extension step.
- **Steady phase** (rolling 1024-byte context): maintain the **ladder** — match
  states M_j for all context lengths j = 1..1024. Since the length-j context at
  t+1 is the length-(j−1) context at t plus the new byte, each position advances
  the ladder with 1024 extension steps (M'_j = extend(M_{j−1}, byte_t)), and the
  position's answer is the branch summary at M'_1024.

Per position this is ~1–3 µs of pointer/rank steps instead of a ~200 µs fresh
search + IPC — **300k–1M positions/s per core**, and memory is O(1024) states,
bounded by construction (the ~490 B/position accumulation that took our service
to 400 GB RSS is structurally impossible here, not merely patched).

**The subtlety we flag for you:** the ladder extends contexts on the *right* —
the natural downward step in suffix-tree / suffix-automaton navigation (and if
`xsa branches` already keeps per-node branch summaries, each rung's answer is a
lookup, not a scan). It is the *adversarial* direction for a plain FM/BWT
backward search (which prepends). Whether χSA's internals support cheap
right-extension (forward links, Weiner links, bidirectional BWT, or
automaton-style transitions) is your call — the recurrence is the ask:

```
M'_j = right_extend(M_{j-1}, new_byte)     for j = 2..1024
M'_1 = state_of(new_byte)
answer(t) = branch_summary(M'_1024)        # or the empty-histogram path
```

**The API shape that would integrate on our side:** a streaming process or socket
where we push document bytes and receive per-position branch histograms
(windowed, resumable by byte offset). Our reader adapts once; parity receipts
(the 17-context brute-force oracle + full row parity) stay our acceptance gate.
Continuations are then literally "keep pushing bytes" — no re-search of anything.

---

## Requirement (operator directive): never materialize the text

The 1.4 TB `h466_rl.txt` was a materialization of the HPRC release-2 assembly set and has
been deleted. The source of record is the compressed archive —
`/moosefs/laurap/hprc/HPRC_r2_assemblies_0.6.1.agc` (copy under
`garrisonlab-s3-aws/tracepoints/`) via `/home/erikg/agc/bin/agc` — and **the pipeline must
read from it, not from an expanded copy**. All derived structures built from that text were
retained (SA 18 GB, LCP index 82.5 GB, RL-BWT 16.4 GB, pfp parse 62.1 GB, χ run info
27.7 GB, …), because they are the expensive artifacts and remain valid without the text.

Consequence for the API we are asking for: the streaming interface should accept
**document/assembly bytes decompressed on the fly from the archive** (AGC block decompression
is cheap and random-access by sample), so that habitat- and corpus-scale indexing never
requires a 1.4 TB staging copy. This composes exactly with the ladder design above: bytes
flow in, per-position features flow out, nothing is materialized in between.
