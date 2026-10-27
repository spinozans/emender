# X35 part 1 result: the entry-state two-pass thesis is false as stated

The delivered result is a negative structural receipt, with executable X31
counterexamples, conservative trace instrumentation and an integer allocation
subproblem. It is not a new training cell. Detached routing is state-dependent:
a prior append changes age-1 selection, residual, case and possibly gate.
Even a certified append-only mutation run can have these dependencies.
See CHUNK_PARALLEL.md for the dependency table, restricted algorithm,
segmentation conditions, proof obligations, timer-wheel design and backing work.

## Measured density (seed 0 initialization only)

C=64, S=2944, K=64, V=128, H=4, d_model=256; 8 positive k=128 ladder
words and 8 MQAR T128 validation-prefix examples, using read-only X31 TapeCell.
Density counts individual (example, head, chunk) stores; a whole batched GPU
chunk requires every participating store to qualify. No trained weights were
reconstructed from the historical summary JSONs. Repeated ladder examples
measure identical-token runs and head variability, not independent samples.

| Data | Store chunks | Append mutation candidates | Entry-pure structural candidates |
|---|---:|---:|---:|
| Ladder, all full chunks | 128 | 0% | 0% |
| Ladder, all-a full windows | 32 | 0% | 0% |
| Ladder, all-b full windows | 32 | 0% | 0% |
| MQAR, all full chunks | 64 | 50% | 50% |
| MQAR, all-padding windows | 32 | 100% | 100% |
| MQAR, non-padding windows | 32 | 0% | 0% |

Each ladder store excludes 2 trailing tokens from the full-chunk density;
case histograms include the complete word. All-a and all-b full windows each
contain 1024 intra-chunk matched-value references across their 2048 decisions.
There are no POPs at this initialized closed gate. The >90% a-run purity
prediction FAILS on this stated initialization protocol. This does not establish
its failure after learning. MQAR's 50% is entirely trailing padding, not useful
parallel append work. `density.json` retains event-token fractions, source/token
hashes and exact numerators; `density.txt` retains broker card evidence.

## Validation and scope

`run_leased.sh -q`: **12 passed in 10.74 s**, on physical GPU 0, lambda01,
Python 3.12.3, with only the mandated project interpreter. Three fixtures compare
X31 and X30 outputs, five functional fields and every trace field bit-for-bit.
One additional fixture checks the read gradient with respect to values bitwise.
Other tests falsify chunk-entry routing, expose a true append run's dependency,
check nonzero error-coefficient gating and causal read/gradient dependence,
verify fail-closed missing-death evidence and segment coverage, and check exact
integer rank allocation with holes, no-writes and capacity overflow.

These are counterexample and component tests. They are not full v3 pure/mixed
state/gradient parity, and do not replace the 109-test X31 suite. X31 usage EMA
is excluded consistently with its public functional scope. The integer scan
preserves lowest-free order; the obstacle is not an fp32 free-list ordering.
No defect in X31 is alleged and no reference source was modified.

## Registered gates and staged implementation

- (a) >=3x complete pure forward: NOT RUN; no valid general pure path built.
- (b) trace density: MEASURED on initialization; a-run >90% prediction FAIL.
- (c) >=2x mixed training loop: NOT RUN; no mixed training implementation.
- (d) zero full pure/mixed bitwise parity failures: NOT ESTABLISHED. The 12
  delivered tests pass, but the required implementation/suite does not exist.

Stage 1 instrumentation and the formalization are delivered. Stage 2 is limited
to the integer allocation subproblem; no complete batched decision/read/write
path is delivered. `segments` is retrospective analysis only, not stage 3
runtime composition. Stages 3–4 remain unimplemented. Stage 5 is receipted as
the part-2 work order. No mutation-only timing is substituted for cell speed.

A corrected research direction is to batch append/correction PAIRS or compute
dependency waves and causal row versions. It may work well even though the
proposed independent first pass fails. The negative result does not prove
chunk parallelization impossible, nor that a two-token event rate imposes an
absolute throughput bound. A trained trace study remains necessary before
claiming sparse events for deployment.

## Operational receipt and residual risks

Only X35 files are included in the commit. Existing unrelated workspace edits
are preserved. No .pt files, WG service, workers, agent delegation, Frontier,
Slurm submissions or resilient training changes were used; resilient requirement
IDs and scheduler Partition/QOS requirements are inapplicable. GPU broker
wrappers restrict cards to 0–6; GPU7 was not requested or used. The short runs
used cards 0/1, not the long-run card reservation policy.

`contact_supervisor` was searched for first in the callable tool catalog and
was unavailable. No supervisor decision/progress contact could be sent.

Residual risks: initialization-only and small data sample; retrospective
classification does not predict routes; arbitrary imported states require
additional time-projected age/liveness checks; general bitwise GEMM and replay
accumulation equivalence is unproved; timer-wheel and compact ownership-aware
backing are work orders. There is no speedup or learning-quality claim.
