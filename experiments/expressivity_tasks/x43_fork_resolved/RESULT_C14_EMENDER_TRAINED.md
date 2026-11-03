# C14 Emender consultation trained at registered active-time budget

Decision receipt commit: `8cd81972`. The supervisor's attended work order resolves
stopping-rule scope: PREREGISTRATION_C14_BYTE.md:47 governs, with tier-1 time-matched
precedent and the program-wide matched-budget doctrine. Frozen dense prefixes are
consumption records. The dense controls were not retrained or changed.

| Arm | Steps | Actual target bytes | Active seconds | Training-process wall seconds | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense (frozen) | 111596 | 225880702 | 2700.022528 | unavailable | 41.3 |
| Consultation | 27401 | 55447993 | 2700.021958 | 2726.838463 | 10.148436 |

Throughput ratio to the dense reference: 0.245725.
Different consumption at the same budget is the measured deconfound; no paired
quality delta is claimed. Routine sparse decoding, feature assembly, sampling,
transfers and training count toward active time. Checkpoint save/hash stalls
(2.053944s) are separately itemized.
The wrapper acquired card 2 through gpu_lease.sh with --no-wait; lease acquisition
preceded the process wall timer. Setup/preflight and checkpoint cost are included
in process wall time. This process wall measurement does not include the earlier
cache extraction or lease acquisition overhead.

The run uses seed/dropout seed 3401, batch16/context128, BF16, AdamW 3e-4/.01,
clip1, 2,133,700 trainable parameters. An independent keep=.5 coin per position
zeros all 260 feature scalars together, without rescaling. Every consumed cache
window was identity/length matched to frozen source offsets; padding was masked.
Launch rehashed all 13,950 cache shards and the journal; original cache provenance
retains its historical pre-decision audit hash. Frozen dense receipts pass again.
Checkpoints are outside git: `/home/erikg/c14_dense_bytes/emender_consultation_resume.pt`, SHA256
`5499d02420774829fa23ee17b232f06789d292ab815f12ae039e671793485261`. Model/optimizer tensors are finite; exact optimizer,
RNG and next-window state are saved. Curves/config/seeds/cache identity, itemized
cost and validation are in `receipts/c14_consultation_emender/`. Eleven existing
boundary/sparse/batching tests pass. No resilient DiLoCo behavior was changed,
tested or run; DiLoCo gap-matrix requirements do not apply.

Paired eval waits for the live registered_eval phase. At 2026-11-03T18:24:40.979807+00:00,
178688 / 3,431,242 windows and 22617975 / 434,204,084 targets were committed.
The phase-average scenario (6426.93 targets/s) projects 2026-11-04T12:12:01.807804+00:00;
the accepted baseline scenario projects 2026-11-04T13:55:53.520072+00:00.
These are projections, not guarantees. The service was not touched.

Residual risks: paired quality evaluation, P1/P2/P3 and C13 remain unmeasured;
transformer consultation remains pending. Existing native-service OOM/restart
risk remains as documented in RESULT_C14_EMENDER_CLOSED.md. Cache parity here
uses audited extraction attestations rather than a fresh native-oracle rerun.
The runner refuses an existing output directory; saved state is available for
manual recovery, with no automatic resume path claimed.
