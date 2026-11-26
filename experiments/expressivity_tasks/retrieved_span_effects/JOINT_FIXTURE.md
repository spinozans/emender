# Joint token training: invented manual fixture

Question: can a next-token loss train the live recurrent model, a soft search over passages, the passage reader and the merge together? This is an implementation check, not a test of natural-language utility or a hurdle for running a larger trial.

One episode has four documents with distinct titles sampled from eight symbols. Each contains a day setting and a night setting. For each field independently, assign four distinct randomly chosen value tokens from eight possibilities to the four rows. Prompt = `[title, field]`; predict the one real answer token over the full 18-token vocabulary. Given just the prompt, the answer is uniform over eight values. The four documents are the intended evidence; their order and assignments are random. Model inputs are the prompt and archive, never the answer label or supporting-row index. There is no auxiliary routing label, positional hint to the relevant row or evidence in the live transcript. This is a synthetic conditional token problem, not generic text pretraining.

Train two small models from the **same random initialization** on the **same episodes**. In both, the host is a trainable single-head gated-delta recurrence, and a softmax scorer uses the host hidden state and four title embeddings to weight all four rows. A second trainable gated-delta recurrence runs on each passage; the live prompt supplies its read query. The gated read reaches a trainable token decoder. These are dense recurrence cores, not production GDN‑2 or E97 checkpoints. No trainable component is frozen. In STATE the passage starts from zero; in EFFECT every passage starts from a differentiable copy of the same live head state. No materialized maps or corpus rebuild. STATE can solve the problem; the generator does not force EFFECT to win.

Fixed reproduction recipe (not a schedule recommendation): CPU FP32/single thread, width 24, head 8, 320 AdamW updates per arm, batch 64, lr .003, weight decay .01, clip 1. Init seed 5401, FIT generator 5402. Check finite nonzero gradients and a real first-step parameter change in each module. After both fits, freeze parameters and score the same 64 × 64 independent episodes (generator 5403). Independent draws are not claimed to have globally unique assignments. No choosing checkpoints, tuning on evaluation or retry based on the result.

Report token NLL and number correct, not only gate activity. At the same weights (1) disable the hint; (2) rotate the two values as a pair to a different title, keep routing unchanged, and score both the old answer and the new manual's answer. They always differ. For EFFECT also (3) keep the true blank-start writes `B_i` but rotate the seeded contributions `F_i(S) − B_i` across rows on the same seed. An archive swap tests use of archive contents, not specific use of `A`. Two separately learned arms need not develop the same representations. Report all results, including failure. One seed on this invented task gives no ranking of architectures.

Run locally with no corpus, weights download or GPU:

```sh
env -u GPU_LEASE_HELD -u PYTHONOPTIMIZE CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m experiments.expressivity_tasks.retrieved_span_effects.joint_fixture
```
