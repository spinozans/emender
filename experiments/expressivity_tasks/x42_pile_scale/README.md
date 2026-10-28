# X42 Pile scale infrastructure

Use `/home/erikg/emender/.venv/bin/python` exclusively. Read RECEIPTS.md first.
No pilot/search has run. Current local corpus is an access sample, explicitly
incomplete; scientific training rejects it. Full corpus construction is CPU/network
work and does not depend on X40. For a 24 GiB token budget:

Part 2 has started a detached build at `data/pile24`; see `REPORT.md` and
`receipts/part2_corpus_launch.json` before launching another build. Monitor
`data/pile24/progress.json`. The build is incomplete until its final manifest
attests completion. `documents.jsonl` records source-order boundaries and split
token offsets; shuffling remains in the batch sampler.

```bash
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.data --output experiments/expressivity_tasks/x42_pile_scale/data/pile24 --gib 24
```

Dependencies: existing torch/numpy/tokenizers/datasets/huggingface_hub, plus
zstandard (installation receipted). Train/valid binaries, dedup DB, tokenizer
and manifest remain under ignored data/. No .pt files are emitted.

After X40, record a verdict JSON with `verdict` pure/hybrid,
`approved_for_x42: true`, `source_receipt_path`, `source_receipt_sha256`.
Set config architecture accordingly. Bundled `configs/*_pure.json` and `*_hybrid.json` use emender.py, which reuses
the X39 semantic wire with X36 per-head replay scans through width projections.
This new adapter has meta construction checks only; GPU parity and numerical
smoke remain required after the verdict. Pure uses FFN/memory blocks without
attention; hybrid uses X34-style attention plus memory. A pure arm requires `model_factory` in
`module:function` format; it must return logits from token IDs. The default
scaled X34 transformer is suitable for a transformer control; hybrid uses
`lane_factory` and `lane_positions`, returning a width-preserving residual.
A whole-model factory may also implement a hybrid. Factories consume config,
support meta-device construction and accept GPT-2 vocabulary 50257.

Search factories additionally attest `consumed_search_constants` and implement
`auxiliary_loss()` for gate cost and `occupancy_profile(stream, lengths)` returning
`occupancy_length_slope`, `pressure`, `rich_capacity`. The profile must reset
state per length/window and include uncapped/rich-capacity pressure evidence.
Checkpoint replay must not double-count gate statistics. Recompute memory
receipts including `extra_workspace_bytes` for recurrent state/journals.

```bash
experiments/expressivity_tasks/x42_pile_scale/run_leased.sh -m experiments.expressivity_tasks.x42_pile_scale.train --config CONFIG.json --data DATA --output NEW_RUN_DIR --verdict VERDICT.json
experiments/expressivity_tasks/x42_pile_scale/run_leased.sh -m experiments.expressivity_tasks.x42_pile_scale.search --config SEARCH_CONFIG.json --output NEW_SEARCH_DIR --verdict VERDICT.json --evaluator experiments.expressivity_tasks.x42_pile_scale.evaluator:evaluate
```

For search place `data` in SEARCH_CONFIG.json. Set proxy tokens and
shape from pilot config; evaluator fixes exposure to the registration. A failed
or censored candidate aborts search; no partial objective is scored. Broker
wrapper retains launch command, base commit, lease, GPU snapshot, console and
terminal exit code, using the established broker heartbeat/release mechanism.

CPU checks:

```bash
/home/erikg/emender/.venv/bin/python -m pytest experiments/expressivity_tasks/x42_pile_scale/tests -q
```

Residual work: full token corpus; X40 selection and validation of the bundled scale adapter;
measured device memory and numerical smoke after that integration; pilot/search
runs. No checkpoint/resume or crash recovery is claimed by this initial harness.

For matched arm pairs, run `paired` through run_leased.sh with `--configs A.json B.json --data DATA --output NEW_PAIR_DIR --verdict VERDICT.json`.
It writes matched_protocol.json and rejects differences in seed, context,
microbatch, accumulation, token budget or evaluation schedule before training.

## Part 3 engineering validation and resume

Run synthetic scale checks (no X40 performance verdict or corpus needed):

```bash
X42_LAUNCH_RECEIPTS=experiments/expressivity_tasks/x42_pile_scale/receipts/part3_gpu_launch_accum2 \
  bash experiments/expressivity_tasks/x42_pile_scale/run_leased.sh \
  -m experiments.expressivity_tasks.x42_pile_scale.validate_gpu --accumulation 2 \
  --output experiments/expressivity_tasks/x42_pile_scale/receipts/part3_gpu
```

This checks all six registered baseline/pure/hybrid pilot/final configurations.
It uses synthetic repeated input, two optimizer updates, bf16 autocast and
activation checkpointing. The first update warms kernels and allocates moments;
the second reports steady throughput, including validation overhead. It checks
finite logits, loss, all trainable gradients and updated weights, and asserts
four X36 ReplayBackward nodes per lane. These are engineering receipts, not
training quality or upstream numerical parity claims.

Training writes atomic `latest.ckpt` at `checkpoint_every` updates (default:
`eval_every`) and at the last update. Add `--resume` to the original training
command, using the same config/data/verdict/output. Registration changes fail
closed. Model, optimizer, CPU/CUDA/Python/NumPy RNG, completed update, sampler
cursor, curve prefix and elapsed budget are restored. Partial updates after the
latest checkpoint are replayed; time spent offline is excluded. Checkpoints
are trusted local pickle files, ignored by Git; never load untrusted files.
CUDA resume uses deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The process-level regression uses production save/restore helpers at each pilot
architecture's full width, depth and context, with accumulation 2 to bound its
cost. It SIGKILLs a worker after update 1 of 3, starts a fresh resume process,
and compares update losses and model/optimizer/RNG/cursor hashes with a fresh
uninterrupted reference. Temporary `.ckpt` files are removed. It does not run
the corpus/verdict-gated scientific training entry point.

```bash
X42_LAUNCH_RECEIPTS=experiments/expressivity_tasks/x42_pile_scale/receipts/part3_resume_launch \
  bash experiments/expressivity_tasks/x42_pile_scale/run_leased.sh \
  -m experiments.expressivity_tasks.x42_pile_scale.resume_probe \
  --output experiments/expressivity_tasks/x42_pile_scale/receipts/part3_resume

/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.attest_corpus \
  --data experiments/expressivity_tasks/x42_pile_scale/data/pile24 \
  --output experiments/expressivity_tasks/x42_pile_scale/receipts/part3_corpus.json
```

The attestor reads the existing builder output without restarting or modifying
it. `not_complete` records a monitoring snapshot; only `attested` verifies final
hashes, counts, source order, boundaries and contiguous per-split token offsets.
