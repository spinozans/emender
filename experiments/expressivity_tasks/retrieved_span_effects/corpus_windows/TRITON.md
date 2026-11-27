# All-frontier training: Triton path and measured update cost

The [CPU reference](README.md) passed before porting (public `29b571b4982207048a8537dcbbf6a4992b272a2f`). Select `pkt = archive.windows(docs, locator='triton', ...)` on an admitted accelerator and `backend='triton'` for training. Integer routes can alternatively be prepared on CPU; neither locator sees future tokens. This is the small token-native static-value head, not a modification to either recurrent host in the natural four-page panel.

**Hardware status:** the [single-use GPU meter](../CORPUS_DEVICE_RESULT.md) stopped on excessive shared memory for the 201-row leaf value gradient (11/12 passed; no timings). The [subrow correction](SUBROW_RESOURCE.md) was subsequently **requalified on device** under the separate [raw-language pilot](../language_compression/RESULT.md): 19 device checks passed and the fused indexed training step took 12.822 ms vs 20.477 ms for the same law in eager PyTorch at the new geometry. This does not supply a timing for the failed meter. The original pre-hardware qualification and clocks below are retained.

## What changed on the training path

- **Packed prefix scan.** `triton_prefix.py` scans real input embeddings by document/feature tile. The live encoder and archive encoder use the same causal `mean_t = sum_{j≤t} e_j/(t+1)`. Backward scans the other way, accumulating `dmean_t/(t+1)` for each contributing input. No padded states, reads across a document boundary, or full prefix repetition. Standard linear projections remain GEMMs.
- **Current-weight corpus summaries.** `triton_leaf.py` builds the per-leaf log masses and value means for positive feature channels in FP32. It saves maxima and reciprocal normalizers, **not** a row × feature × value product or per-row softmax tape. For feature `r` and row `i` with normalized within-leaf weight `p_ir`, the score gradient is `p_ir (dlogZ_r + dmean_r · (v_i − mean_r))`; the value gradient is `Σ_r p_ir dmean_r`. Tiles recompute weights and own their output gradients; these two backward kernels do not use atomics. Tested leaf size is not a corpus fanout cap. This first port accepts actual leaves of 1–256 rows; use the torch backend for larger leaf geometry. In the first port, upper tree levels, the random-feature projection and source normalization used PyTorch. The [language pilot](../language_compression/README.md) added Triton parent merges and a fused final normalization for the value-only training path; the diagnostic API still exposes source shares. Indexed row reads and their gradients reuse the [Triton corpus tiles](../corpus_kernel/TRITON.md).
- **Token loss.** For vector values, `triton_nll.py` computes exact full-vocabulary cross-entropy from each query block's logits, including odd widths and multiple vocabulary tiles. Use `(max−target_logit)+log(den)` and `exp(logit−max)/den` in backward: a large common logit shift must not erase the normalizer. Pointer mode uses the predicted successor law. Precision is FP32; speed or parity under quantization/BF16 is not inferred.
- **Streamed exact first derivative.** `pkt.backward_nll(model, backend='triton')` replaces `pkt.loss(...).backward()`. It accumulates parameter gradients without updating or clearing them. Within this one call each shared tensor gets a temporary gradient buffer: accumulate direct query-block partials once, then send their sum through a **single multi-output backward** of the current live/archive graph. This includes a key's direct exact-read adjoint *and* the gradients arriving via its features and learned tree. No bank gradients discarded, no recomputed query forwards, no learned sidecar reused at a later weight version. Activation lifetime is one query block plus the shared graph/adjoints, **not** a guarantee of lower peak memory at every size. Use the ordinary autograd path if this cut is not wanted.

```python
optimizer.zero_grad(set_to_none=True)
loss = pkt.backward_nll(model, backend='triton')  # already backpropagated
# optionally clip here; do NOT call loss.backward() again
optimizer.step()
```

## Results before real-device admission

**27 tests passed** with CUDA hidden: 15 ordinary CPU cases and 12 Triton-interpreter cases. Checks include shifted labels/uneven blocks, isolation of documents and future tokens, shared vs expanded local rows, all parameter gradients and successive updates, dense and indexed reads, checkpointed and streamed backward, odd/tail tiles, FP32 large common shifts, and a 50,281-class NLL. **24 new kernel variants assembled offline for SM89**: six packed scan variants, twelve leaf forward/score-backward/value-backward variants and six loss variants. The earlier interval and row-attention kernels have their own qualification. [Receipt](receipts/qualification.json). Interpreter time is not a GPU benchmark.

Single-thread CPU FP32 wall time on invented IDs. Same 16,384 archive boundaries, 12 windows, **372 real shifted targets**, one optimizer update, same indexed law where indicated. 21,307 of 6,094,848 query–archive pairs refined; non-hit rows still contribute. Each index-bank build re-encodes all corpus rows using current weights. Includes token loss, backward, clipping, AdamW; excludes one-time integer preparation. Median of five warm complete updates, not dispatch times.

| Training path | ms / update | real targets / s |
|---|---:|---:|
| Expanded prefixes, rebuild per query block (6 builds) | 281.89 | 1,320 |
| Expanded prefixes, one bank for all positions | 129.24 | 2,878 |
| Packed, indexed, ordinary backward | 93.00 | 4,000 |
| Packed, indexed, checkpoint query blocks | 115.81 | 3,212 |
| **Packed, indexed, streamed backward** | **80.29** | **4,633** |
| Packed, exact dense GEMM, ordinary backward | 88.83 | 4,188 |
| Packed, exact dense GEMM, streamed backward | 89.10 | 4,175 |

The measured gain is mostly from avoiding repeated encoding/builds and sharing local rows (360 rows vs 5,584), **not** a claim that all indexed updates beat all dense updates. Ordinary indexed was slower than ordinary dense here. At the streamed indexed first update the loss difference vs the six-build reference was zero at FP32 precision; maximum post-clipping parameter-gradient difference `3.73e−9`, post-AdamW parameter difference `4.25e−7`. Dense and indexed compute different corpus laws; parity is checked within each, not demanded between them. [Full compact clock](receipts/cpu-clock.json). No natural quality comparison and no hardware speedup from these CPU clocks.

## Reproduction and hardware boundary

```sh
x=experiments/expressivity_tasks/retrieved_span_effects/corpus_windows
mod=experiments.expressivity_tasks.retrieved_span_effects.corpus_windows
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GPU_LEASE_HELD PYTHONOPTIMIZE TRITON_INTERPRET CORPUS_WINDOWS_TEST_DEVICE
python -m pytest -q "$x" --ignore="$x/test_kernels.py"
TRITON_INTERPRET=1 python -m pytest -q "$x"
python -m "$mod.codegen" /new/path/to/offline.json
python -m "$mod.clock" /new/path/to/update-clock.json
```

The optional local `device_gate` is single-use and **does not queue work**. It requires positive terminal releases for the two initial meters, the closed natural campaign, and the earlier static corpus-head meter. Debit actual prior charges from both the 43,200-card-second total and the 900-card-second synthetic ceiling; also cap to remaining wall time before 16:58 UTC and 150 card-seconds for this pass. Require at least 120 seconds on admission, renewed broker heartbeat, advancing independently witnessed units, timeout and positive release. Real-device fixtures run first; only then do the same-target clocks, including dense/streamed counterparts and peak allocation. No success condition on speed, automatic retry, natural data selection, or claim the hardware pass has run.
