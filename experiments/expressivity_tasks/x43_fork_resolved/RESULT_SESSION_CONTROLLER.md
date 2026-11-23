# A trained controller for a live suffix-index read

We trained a 53,700-parameter local next-byte predictor and read gate jointly. The model receives 16 bytes of neural input; the native session returns exact continuation counts for the last 8 bytes in the visible document. On a hit the gate interpolates two distributions (local prediction and observed continuation frequencies); otherwise the model predicts locally. The gate is trained by ordinary next-byte loss, not an oracle switch label. Query address and append schedule are fixed. There is no programmable write store in this pilot.

One seed, 128 training documents, 229,376 distinct targets, three epochs. Native XSA answers were checked against independent overlapping counts before training; minibatches reuse the equal counts. Test: 114,688 targets in 64 different hash-checked documents, no final-test hyperparameter selection. Same bytes for all comparisons. Results (nats/byte, lower is better):

| Predictor | NLL | top-1 accuracy |
|---|---:|---:|
| Learned local + conditional read gate | **2.20124** | 42.967% |
| Same local logits, train-selected constant 0.80 | 2.21565 | 42.965% |
| Independently trained fixed 0.50 mixture | 2.24085 | 43.008% |
| Controller with reply masked | 2.50835 | 34.699% |
| Independently trained local-only model | 2.52123 | 34.601% |
| Controller with byte labels rotated | 2.85221 | 29.459% |

**Most of the gain is the read.** Masking it costs 0.30712 nats/byte. Learning how much to trust it adds **0.01441** over a global scalar on the same logits (95% paired document bootstrap interval 0.01194–0.01689). The conditional gate improves probability assignment rather than top-1 accuracy over scalar mixing. Supported reads have 10th/50th/90th percentile weights 0.451/0.870/0.972. These are one-seed pilot results, not a comparison with a competitive full LM. The controller cannot yet decide which suffix to query or learn a write.

Reloaded checkpoint and performed one test-document replay, predicting before appending each true byte. Across 1792 targets the largest absolute difference in any of the 256 probabilities between live and batched evaluation was 9.54e-7. `receipts/xsa_controller_v1/` holds the small trained checkpoint, ordered training manifest, and scores including all per-document NLLs. Regenerate the larger training-count cache using `xsa_controller_data.py`. See `PREREGISTRATION_SESSION_CONTROLLER.md` for constraints and claims.

From repository root, score the next byte of a raw input file (one document, >=16 observed bytes):

```bash
D=experiments/expressivity_tasks/x43_fork_resolved
OMP_NUM_THREADS=4 .venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.xsa_controller_predict \
  /home/erikg/sxgc/xsa/target/release/xsa "$D/receipts/xsa_controller_v1" /path/to/prefix
```

This CLI checks checkpoint and executable digests. It starts a fresh session per invocation; it is a demonstration rather than a latency benchmark. SXGC implementation: `b75a38f`. Example input `My favorite color: blue. My favorite color: ` yields one observed continuation, gate 0.268 and top prediction `b` (probability 0.282).
