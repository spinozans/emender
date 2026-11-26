# Train every real token frontier against one current-weight corpus bank

This is the many-position training path for the [whole-corpus attention head](../corpus_kernel/README.md), not a new retrieval law. With `S` document windows there are `Q = Σ max(length−1,0)` queries/targets. Build the differentiable corpus bank **once** for those `Q` queries at the current weights. Backpropagate their token-sum NLL divided by `Q`, then update the LM and all learned memory encoders together. No detached/stale learned state is cached between updates.

```python
from experiments.expressivity_tasks.retrieved_span_effects.corpus_windows.packet import WindowArchive

archive = WindowArchive(model, index)   # mounted document-reset corpus, immutable integer inputs
windows = archive.windows([('live-a', [3, 8, 0, 7]), ('live-b', [5, 4])],
                          chunk_targets=128, min_match=1, lookback=0)
optimizer.zero_grad(set_to_none=True)
windows.loss(model, mode='indexed', backend='torch', checkpoint_blocks=True).backward()
optimizer.step()                        # four real targets, one update
```

Only real input tokens (`window[:-1]`) become live states. The host computes its causal prefix means independently inside each document. A single shared table holds local K/V: for query after token `t`, only rows with successors **already observed by t** are addressed. The window's last target is never in that table or used for its routes. No padded row is encoded. Search may include earlier MEM anchors with `lookback=k` or `None` as before; each query's route uses only the observed prefix. `windows.predict()` returns all joint reads, shares, partitions and token probabilities. `mode='dense'` is the exact GEMM comparison on identical rows, priors, targets and local masks.

Query blocks share one live encoding, one set of local rows, and the same graph for archive rows and feature summaries. Blocking bounds forward workspace. By default autograd retains block intermediates until backward; `checkpoint_blocks=True` recomputes the **query-side** work to bound saved activations without rebuilding/detaching the bank. Permanent costs still include the corpus, feature/value tree and integer route descriptors. This is not sublinear current-weight bank construction. The previous prefix-packet implementation already shared a bank across prefixes in **one** packet; the changes here also avoid re-encoding prefixes and duplicating local rows.

## CPU before new Triton code

Eight invented CPU cases passed (FP64 predictions and first derivatives for dense/indexed, pointer/vector, missing sources; strict local causality and reset; odd query blocks, sum/denominator and optimizer equivalence; single bank build including checkpointed backward; summary graph vs the original implementation). No natural data or accelerator. To reproduce: hide CUDA, unset `GPU_LEASE_HELD`, `PYTHONOPTIMIZE` and `TRITON_INTERPRET`, use one CPU thread, then `python -m pytest -q experiments/expressivity_tasks/retrieved_span_effects/corpus_windows/test_windows.py`.
