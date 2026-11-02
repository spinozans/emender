> Current program, claim statuses and exact receipt bindings: [PROGRAM.md](PROGRAM.md).
> Run `/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/validate_program.py`
> before calling a result documented. The part-1 account below is historical.

# X43 part 1: byte-level fork shakedown

The index leg is **blocked**, not passed. The prescribed build in `~/sxgc`
failed with exit 101; see `receipts/xsa_build.log` (verbatim) and
`receipts/index_status.json`. The operator's project was not patched.
Consequently there is no `.sxi`, chi density, fork histogram, automaton receipt,
or forced/fork NLL or accuracy measurement. No substitute classification is
claimed. The fork program's first measurement remains outstanding.

The corpus is canonical TinyShakespeare, 1,115,394 raw bytes, vocab 256, no
encoding/tokenizer transform. Provenance and SHA256 are in `receipts/corpus.json`.
The contiguous first 90% is training data; remaining 10% is held out. The first
held-out byte seeds evaluation, so targets begin one byte after the split.

`run.py` reuses the X42 velocity memory+MLP Candidate with two blocks and two
memory heads as the pure EmenderLM foundation. Its upstream scan/backward and
certified core are unchanged. This small configuration is an engineering
shakedown; it does not extend an upstream certification claim. The control
uses causal pre-norm transformer blocks with GELU and learned positions.
Counts match within 0.1% in trainable parameters; frozen memory features are
reported separately in total counts. Both use FP32 compute (Emender's loss
wrapper retains its upstream bf16 feature autocast / FP32 store and loss).
Each arm gets 180 seconds including first-step initialization; this is a
wall-time shakedown, not a matched-token scientific architecture comparison.

Run from repository root:

```bash
bash experiments/expressivity_tasks/x43_fork_resolved/run_leased.sh
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.validate
```

The wrapper atomically leases only card 2 or 4, releases it at exit, and receipts
exit status. Models run sequentially. It does not start any worker service.
`receipts/results.json` contains actual parameter counts, duration, learning
curves, and held-out NLL and accuracy. Local `.pt` checkpoints are ignored by Git; compressed per-target `.npz`
predictions are retained as evidence. Positions are zero-based corpus offsets.
`validate.py` checks all byte values, once-only target alignment including the
short final window, and future-perturbation invariance for the transformer.

`PREREGISTRATION.md` binds follow-ups before execution. Exact counts and oracle
labels must be distinguished from training-only causal inference. BWT boundary
orientation and suffix-to-byte mapping require resolution when xsa is available;
run boundaries alone do not establish a natural-order next-byte classification.

The session exposes no `contact_supervisor` tool, so that requested coordination
channel could not be invoked. Work stayed in this attended session.

`decompose.py` is the deferred join: it requires a corpus-order Boolean fork
array plus successful xsa provenance, corpus/index/binary/label hashes, explicit
chi definition, orientation, and label derivation. It rejects incomplete target
coverage and reports NLL mass and accuracy in each class. Its synthetic utility
validation is receipted separately and is **not** a Shakespeare fork result.
