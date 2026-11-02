# C14 phase 1: Pile stream against Wikipedia

**Query-health gate failed:** two impossible branch counts were quarantined.
Coverage below is a provisional measurement against this binary, not an
end-to-end correctness certificate. C14 training remains blocked.

10,000 deterministic eligible Pile training-token boundaries, seed 1401. No
training, GPU allocation, or held-out scoring. Wikipedia MANIFEST SHA-256:
`853ee5ea723004bbacda9fa88088a3461101863928d6ebfb218106b981e3c7cc`. Rebuilt xsa at
`139926a09db84d02c9877943d31fc8dcff52cb26`; binary SHA in provenance.json.

| Left context bytes | Wikipedia support rate |
|---|---:|
| 16 | 36.5900% |
| 64 | 2.3600% |
| 256 | 0.5900% |
| 1024 | 0.0900% |

Maximal context quantiles (bytes): `{"0": 0.0, "0.1": 5.0, "0.25": 7.0, "0.5": 12.0, "0.75": 19.0, "0.9": 26.0, "0.95": 35.0, "0.99": 150.0, "1": 1024.0}`.
At the 1024-byte cap: 9 / 10,000 (right censored).
Observed next-token first-byte agreement at the maximal supported context:
69.3333%, conditional on nonempty context support. This is a measured
first-byte statistic, not a full-token accuracy or NLL prediction.

256-byte spans (128 bytes before / after each of the first 1000 sampled
boundaries): 0.5000% exact Wikipedia support. Bounded locate is deterministic
suffix-rank sampling of at most one hit per span, not unbiased occurrence or
document sampling. `validation.json` verifies every returned offset against
immutable corpus.bin bytes. The rate measures exact span overlap, not Wikipedia
source attribution. Wikipedia markup and Pile preprocessing differ.

This world has shared short English contexts but sparse long verbatim overlap;
near-disjoint and substantially overlapping are length dependent descriptions.
It is not legitimate to call these corpora disjoint, infer a document component
fraction, or treat C12's no-whole-task-copy property as established here.
`intervals.json` gives conditional binomial Wilson 95% intervals. Excluding short
documents/near-edge positions changes the population; no all-token estimate is
claimed. Rejected draws: 2114.

## Machinery and budget

AGC FFI cannot query SXI2. The batch Python bridge emits token-boundary-aligned
summary geometry. See C14_FEATURE_BRIDGE.md for the exact upstream handoff and
byte-30 omission. Features never consume Pile target/future bytes. Measurement
queries for observed next bytes and verbatim spans are separate diagnostics.

Branch enumeration: 10000 positions, 2524500 searches;
30.648s process wall, 0.594s journaled query kernel;
326.22 features/s whole process, 16842.99/s kernel on this host.
40M-position linear extrapolation: 34.06h
whole-process branch work, 0.66h
kernel; context search/decoding/projection/transfer add cost. The training budget
needs 14,814.815 features/s. Neither CLI opening nor a cached persistent service
is free. Full measured workflow wall is 609.42s
for this sample, including coverage and locate diagnostics (not all needed for
production features). No 40M run was performed; cache effects and repetition
could change scaling. 260 float32 scalars need 41.6 GB/40M positions, while full
50,257-token vectors need 8.041 TB.

A performance trap: `-j 1` skips Rayon global-pool initialization in the current
CLI, so the actual default pool can use the whole 256-logical-CPU host. This is
host throughput, not single-core throughput. Materialized run columns are about
266 GB. CPU model/topology and affinity are receipted. No single-core benchmark
or leased GPU benchmark is claimed; the present machinery does not establish
a one-core online budget. The persistent batched API remains an upstream handoff.

## Residual risks and readiness

No complete BPE candidate distribution, EOS continuation model, document-seam
filter, held-out coverage measurement, isolated one-core benchmark, or 40M
feature cache is delivered. The 255-byte outgoing total omits separator byte 30;
context support can also include terminal occurrences. 1024 is a cap, not a
proof that no longer suffix matches. Training is blocked on freezing a complete
causal cache geometry and receipting its generation/transfer costs, not on any
unmeasured promise of speed. This phase registers an experiment, not an outcome.

## Confirmed query failure: training blocked

Two extension queries returned UINT64_MAX (samples 2769 and 9057, appended byte 194). Their outgoing totals exceed the parent context counts. The receipt audit caught this; raw outputs and initial failure are preserved. The two feature rows are explicitly invalid, with no entropy or probability distribution. Aggregate coarse support queries and observed-next queries had no out-of-range counts, but this does not prove all remaining index answers correct. The source subtracts r-l without a guard; the underlying interval/rank cause remains unisolated. No upstream code was modified. The bridge now fails closed on impossible counts. Repair and independent upstream regression evidence are mandatory before any C14 training.

## Content versus formatting overlap

All five supported centered 256-byte spans are spaces only: 0/1000 non-whitespace centered-span matches. This is not evidence of a 0.5% Wikipedia content component.

| Context bytes | Non-whitespace supported / non-whitespace eligible | Whitespace-only supported |
|---|---:|---:|
| 16 | 3264 / 9605 | 395 |
| 64 | 118 / 9882 | 118 |
| 256 | 23 / 9962 | 36 |
| 1024 | 2 / 9990 | 7 |

Supplemental bounded locate covers all 59 supported 256-byte contexts and all nine supported 1024-byte contexts, including two non-whitespace 1024-byte contexts. One is four-byte-alphabet encoding noise; the other is a basketball statistics table. Thus rare substantive long overlap exists, but formatting/repetition dominates many long matches. These selected hits are not an unbiased document-component estimate. Raw patterns, references, offsets and direct-byte checks are retained.
