# Round5 exact resume: attention register tiling, real1.3B NO-GO

No production selection, training or training checkpoint exists. Do not start
the1.3B scenario from this result. Best internal diagnostic now uses
c14_phase5r5_tiled_pages.py:32x128 register tiles, PAGE1024, exact normalized
parity carries. Matched191.888684ms/head versus R4 255.924206; projection381.723005
native-only days. All ten native/repeat pairs,109+108 suites,8 short state pairs,
4 replay/gradient pairs,24 primitive adversarial pairs and T512/T1024/T2048
masked-reset fixtures pass. T2048 is compositional, not repaired monolithic.

Pre-implementation design a8fb1ac5; implementation4b9aa02d. Read DESIGN.md,
COMPOSITION_PROOF.md, result report and validation_summary.json first.
profile_final.json is event attribution; cuda_r4/r5.json are CUPTI actual kernel
durations, NOT spill counters. READ189.907448->27.859426ms device time; control
56.66ms now dominates device time. Uninstrumented wall191.89ms includes host
submission and gaps; profile/unprofiled difference is not an exact subtraction.
Next structural candidate should separately measure/control those gaps and
control, retaining mutable-store order. Do not blindly re-import online softmax:
its real-arithmetic equivalence does not retain the native FP32 reduction tree.
Do not assume trained reads are top-k sparse: PB3 lacks weight masses, synthetic
top32 is only .79-.84. No top-k implementation/registration/training exists.

Only diagnostic install is c14_phase5r5_validate.install. It is serial-only;
shared READ temporarily patched in try/finally. Fresh runs require NEW output
paths/wrapper receipt directory, never overwrite existing completed receipts.
Use only /home/erikg/emender/.venv/bin/python and a broker lease restricted to
physical3/4/5/6 (this round used4/5); never occupied/protected cards. The source
wrappers deliberately refuse their completed terminals. New candidate needs a
new committed/pushed design and explicit source bindings before validation.
Push every commit. Preserve historical guards and failed attempt evidence.

Mandatory before promotion: operation-rich maximum-live long/full trained-model
fingerprints, >=2M paired real-target optimizer/RNG/cursor/resume identity,
150M/1.3B sustained throughput/memory. No other training/evaluator was touched.
No DiLoCo changes; IDs inapplicable. No supervisor transport exposed/contact
claimed. Unrelated worktree changes preserved. Fresh process/device/lease checks
and final source hashes are retained separately. No own GPU process remains.
