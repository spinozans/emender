# X43 part 1 result: model legs passed; index/decomposition blocked

Part 1b retried the correct crate path: the release build passed, but exact-corpus
index construction failed at endpoint cyclic seam repair. See
[RESULT_PART1B.md](RESULT_PART1B.md) and `receipts/part1b/` for the new receipts.
The decomposition remains blocked.

| Arm | Trainable parameters | Steps / 180 seconds | NLL nats/byte | Accuracy | Fork NLL share | Fork accuracy |
|---|---:|---:|---:|---:|---|---|
| emender | 2,000,324 | 5,552 | 2.201701 | 35.199% | blocked | blocked |
| transformer | 2,000,372 | 35,853 | 1.829293 | 52.884% | blocked | blocked |

Both arms evaluated 111,539 identical held-out targets with context reset every
128 bytes. These are wall-time shakedown results; token budgets and training
feature precision differ. They do not establish an architecture ranking.

The prescribed xsa build exited 101: Cargo.toml missing in ~/sxgc. Verbatim
failure: receipts/xsa_build.log. No index or fork labels were produced; no fork
density, histogram, automaton statistics, forced/fork loss, or branch accuracy
is claimed. Those central X43 measurements remain blocked. No changes to sxgc.

The exact-count interpolation and fork-write-gate follow-ups were preregistered
before training and were not run. The deferred decomposition utility passed
synthetic checks only. CPU byte/alignment/causality checks and local artifact
validation passed. Checkpoints remain local and ignored by Git. Compressed per-target arrays are
retained as evidence; receipts/artifact_manifest.json retains all artifact hashes.

Residual risks: missing index and unresolved chi/orientation convention;
one seed and a short run; reset context; unmatched token budgets; bf16 feature
autocast in Emender training versus FP32 control; no small-model kernel parity
recertification. Upstream certified core files were reused unchanged.
