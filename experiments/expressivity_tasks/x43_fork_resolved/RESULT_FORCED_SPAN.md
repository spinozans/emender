# X43 B — forced-span verbatim law: recall vs branch entropy

Registered in 5da0c320 before training. PB-L1 is **refuted** in the
registered finite-context, fixed-budget diagnostic. This is training-corpus
recall, not unseen-document generalization. The exact self index supplies
entropy labels only; the neural emender receives zero consultation features.

The 2,133,700-parameter tier-1 arm trained for
4,068 steps / 8,120,887 targets in 180.032
active seconds, 205.353 equivalent corpus target exposures.
Seed4301, AdamW 3e-4 wd.01 clip1, BF16 forward and float32 CE. Compilation
inside the training loop counts toward the wall budget. No tuning by recall.

Source: first16 complete nontruncated Pile train documents, full raw-document
hashes matched to documents.jsonl and attested offsets. Natural bytes, vocab256;
128-byte independent DocumentWindows; masked tails; zero cross-document windows.
The receipts bind source index, offset index, exact context-count index, runner,
and protocol hashes. The first8 and last8 documents are disjoint calibration
and final-test recall panels; BOTH panels were included in training.

| Panel | Branch entropy bin | Targets | Byte accuracy | NLL | Mean entropy |
|---|---|---:|---:|---:|---:|
| calibration | forced | 11409 | 0.816987 | 0.557957 | 0.000000 |
| calibration | low | 6830 | 0.795315 | 0.620110 | 0.513541 |
| calibration | high | 9973 | 0.678031 | 0.993052 | 1.570699 |
| final_test | forced | 4902 | 0.809874 | 0.594905 | 0.000000 |
| final_test | low | 2518 | 0.791501 | 0.663710 | 0.558580 |
| final_test | high | 3914 | 0.689320 | 0.950781 | 1.539523 |

| Panel | Forced runs >=4 bytes | Exactly recalled | Span accuracy |
|---|---:|---:|---:|
| calibration | 862 | 483 | 0.560325 |
| final_test | 388 | 185 | 0.476804 |

Forced is one exact continuation; low is multiple branches with H<=log2; high
is H>log2. Contexts are at most4 bytes and truncated at each window boundary.
Whole-span exact match is stricter than byte accuracy. Under deterministic
causal decoding with the same true prefix, an entire forced run is reproduced
iff every teacher-forced top1 prediction in that run is correct. No claim is
made about arbitrary open-ended generation.

C11's exact self-containing teacher distributions motivate measuring this
relationship; its documented shortcut/OOD failures do not demonstrate the law.
These results bound the PartII candidate in THEORY_INTERVAL_OPERATORS.md.
The outcome is retained without changing entropy bins, training budget,
seed, corpus selection, or recall threshold.

Residual risks: one seed and corpus subset; training-corpus recall, with no
unseen-document inference; finite4-byte contexts rather than maximal suffix
intervals; uniform window sampling gives unequal document exposures; 180-second
training may underfit; entropy and natural-text difficulty are confounded.
A negative outcome refutes the registered near-perfect finite-budget prediction,
not the possibility of recall after more training or with exact consultation.
No resilient DiLoCo behavior changed; gap-matrix IDs inapplicable.

Manual notes: only GPU5/6 broker allowlist used for this arm; actual card is
recorded in results.json and lease.txt. No checkpoint was created or committed.
Frozen arms, union service, ~/sxgc, and read-only raw source were unchanged.
contact_supervisor was unavailable in the active catalog; no alternate
orchestration or subagents used.

The byte-accuracy ordering is consistent with entropy degradation on both
panels, but near-perfect forced recall fails: final-panel forced bytes are
80.9874% correct and entire forced runs only47.6804%, far below99%. Do not
collapse these two findings into a claim that the whole registered law passes.

Validation: all117 existing/new X43 tests passed on broker GPU3 before training;
the separate CPU run passed100 with17 CUDA skips. Independent post-run checks
verify all39,546 corpus targets are counted once, the reporting document sets
are disjoint, all entropy strata are nonempty, the180-second budget is met,
and PB-L1's refutation follows from the registered rule. Physical training
GPU5; terminal exit0. The8,120,887 sampled targets correspond to205.353 target
exposures, not205 complete source-ordered epochs. Completed before15:01 UTC,
well before the16:20 harvest. No checkpoint created.
