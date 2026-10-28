# X42 part 3 — scale GPU checks and resume; corpus still building

All six registered baseline/pure/hybrid pilot/final configurations passed the
single-card engineering checks on broker-leased physical card **1**, NVIDIA RTX
6000 Ada Generation (48 GiB class). Card 0, X40, certified cards 2/3/4, forbidden
card 7 and detached corpus PID 793830 were left untouched. No pilot/search or
WorksGood service was launched. This is local single-card work, not resilient
DiLoCo; the DiLoCo gap-matrix requirements do not apply.

## GPU evidence

`receipts/part3_gpu_launch_accum2/terminal.json` and `receipts/part3_gpu/*.json`
record the completed checks. Each config uses its registered width, depth,
1024-token context, microbatch 1, fp32 parameters/master/gradients, bf16 autocast,
activation checkpointing and fused AdamW. Two optimizer updates accumulate
**2 microbatches each**, rather than the registered scientific protocol's 256.
All logits, losses, trainable gradients and updated parameters are finite.
Each adapter graph asserts exactly four X36 `ReplayBackward` nodes; baseline
asserts zero. No upstream parity or trained-quality verdict follows from this.

Second-update measurements (first update warms kernels and allocates moments):

| Configuration | Parameters | Peak allocated GiB | Peak reserved GiB | Analytical GiB | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| pilot baseline | 148,075,008 | 3.015 | 3.361 | 9.550 | 17,154.8 |
| pilot pure | 147,519,764 | 2.903 | 3.332 | 10.642 | 1,059.0 |
| pilot hybrid | 147,544,340 | 2.904 | 3.252 | 10.642 | 1,039.2 |
| final baseline | 1,313,738,752 | 22.513 | 23.270 | 30.286 | 5,343.8 |
| final pure | 1,328,187,924 | 22.676 | 23.000 | 31.653 | 903.7 |
| final hybrid | 1,328,368,148 | 22.681 | 23.412 | 31.656 | 896.5 |

The ~1.328B adapter measured peak is below the analytical **31.66 GiB** estimate.
Allocator peaks include existing parameters, gradients and initialized Adam
moments; CUDA driver/context allocations are outside PyTorch's allocator
statistics. Parameter counts include frozen geometry/features. Throughput is
single-card synthetic repeated-input throughput, including finite/gradient
checking overhead; it is not corpus-training throughput or a scientific
architecture comparison. The measured run does not establish full 256-way
adapter accumulation throughput or long-run memory stability.

The pilot baseline also completed accumulation 256: **18,076.0 tokens/s**, peak
allocated **3.015 GiB**, in `pilot_accumulation256.json`. The first GPU invocation
failed on a config filename typo after its baseline smoke check; its traceback
and exit 1 remain in `part3_gpu_launch`. After correcting the names, the
accumulation-256 invocation completed the baseline but was explicitly terminated
(SIGTERM/exit 143) during the costly recurrent adapter check. See
`part3_gpu_launch_full/interruption.json`. No adapter completion is claimed for
that interrupted invocation. The subsequent accumulation-2 launch passed all
six configs; failed/interrupted attempts have not been erased.

## Checkpoint/resume

`checkpoint.py` atomically writes/fsyncs trusted local `.ckpt` files and retains
model, optimizer, CPU/CUDA/Python/NumPy RNG, completed step, data cursor, curve
prefix and elapsed budget. `train.py` now checkpoints periodically and at the
last update; `--resume` verifies the config/data/verdict registration, restores
the exact update boundary, discards any uncheckpointed curve suffix and verifies
step/cursor consistency. Resume records its checkpoint hash and current lease.
Deterministic CUDA algorithms and the cuBLAS workspace setting are explicit.
Checkpoints are ignored by Git; no `.pt` artifacts are produced or committed.

`resume_probe.py` runs independent uninterrupted, killed and resumed processes
using the production checkpoint helpers. At each full pilot architecture shape
(width/depth/context unchanged, accumulation 2), it SIGKILLs the worker after
checkpoint 1 of 3 and compares every update's losses, model/optimizer hashes,
CPU/CUDA RNG hashes, Python/NumPy samples and cursor. The probe is engineering
validation; it does not bypass the corpus/X40 gates to run scientific training.
See `receipts/part3_resume/` and `part3_resume_launch/` for actual outcomes.

## Corpus and remaining scope

The original detached CPU builder remains in progress. `receipts/part3_corpus.json`
is a timestamped **not_complete** snapshot, not final corpus attestation. The
part-2 pinned source/tokenizer, exact dedup, document order/boundary receipts and
24 GiB token target remain authoritative. The corpus is an explicitly substituted
uncopyrighted Pile subset, not the original complete Pile. No restart was made.

Monitor and then attest without changing the builder:

```bash
ps -p 793830 -o pid,ppid,etime,pcpu,rss,args
cat experiments/expressivity_tasks/x42_pile_scale/data/pile24/progress.json
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.attest_corpus \
  --data experiments/expressivity_tasks/x42_pile_scale/data/pile24 \
  --output experiments/expressivity_tasks/x42_pile_scale/receipts/part3_corpus.json
```

Only a completed manifest plus successful attestor establishes final tokenizer,
index and split hashes, token/document counts, strict stored source order,
boundaries and contiguous per-split token offsets. A running PID, advancing
snapshot or declared source interval alone cannot prove completion. Stored split
streams exclude duplicates and documents belonging to the other split; the
registered character cap and final-document token truncation still apply.

Pilot commands were not prepared: the corpus is incomplete and X40 v1 lacks a
completed verdict receipt. The heads-spectrum protocol amendment remains open
before search. Scaling transfer, scientific accuracy and long-run performance
remain unvalidated. The SIGKILL regression tests checkpoint helpers at the pilot
shapes, rather than a full corpus/verdict-gated training invocation.

`receipts/part3_tests.txt` records the CPU suite. Supervisor coordination was
unavailable because this session exposes no `contact_supervisor` tool. All
implementation, GPU work and debugging remained single-threaded with no agents.
