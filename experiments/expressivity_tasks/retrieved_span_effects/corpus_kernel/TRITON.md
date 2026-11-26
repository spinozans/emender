# Port and qualification: whole-corpus head

The [CPU kernel and fidelity fixture](README.md) were committed and published first (`10b0a11570a27084c41e21282f5ee2b4e583e98d`). No language data, natural model or GPU was used for this qualification.

The subsequent [real-device result](../CORPUS_DEVICE_RESULT.md) reports eight passing hardware tests and both fixed-bank and full-update clocks. The separate packed-window training meter stopped before timing.

## What runs where

- `triton_frontier.ResidentIndex` keeps the document-bounded reverse endpoint array on the device. A batched kernel narrows intervals by two binary searches at each observed suffix depth, with 64-bit addresses and explicit document starts. It can return the interval at a depth threshold or the nested intervals at **all** depths. It has no fixed hit count. `TokenArchive.batch(..., locator='triton')` groups identical threshold substrings, returns compact interval boundaries to the CPU, aligns internal matches if asked, and unions/covers disjoint ranges there. This is **not** a GPU suffix-array builder or a zero-transfer routing path. CPU construction and host planning have a cost.
- `triton_tiles` fuses the expensive ragged score and value reductions. A program handles one row tile and one slice of value lanes: exact dot-product rows, explicit background-feature rows at a partial leaf, or channels of a summarized subtree. It returns a detached maximum `m`, `z=Σ exp(s−m)` and `u=Σ exp(s−m)v`. Group by query and source, rescale by the maxima, and normalize all sources together. No query-by-corpus array is required on the indexed path. Partial tiles are masked. Backward recomputes tile scores; for upstream `(dz,du)`, each row gets `ds=exp(s−m)(dz+du·v)` and `dv=exp(s−m)du`. For the explicit feature law, `ds` is distributed by the feature softmax. FP32 atomics accumulate contributions when queries reuse rows. This is a first-order VJP, not bitwise deterministic parallel accumulation.
- The compiler target for the kernels is CUDA; GPU execution is opt-in. Document-bucketed host/row encoding, random feature projections, construction/merge of learned summaries, joint source normalization, full-vocabulary head and optimizer use PyTorch. `batch.py` lets training reuse immutable integer IDs and work descriptors **without caching learned states**. Serving with fixed weights can amortize learned summaries as well. Comparing an indexed frozen read alone against a whole training update would be misleading.

```python
# m and idx as in README.md; keep m and the integer packet on the same device.
from experiments.expressivity_tasks.retrieved_span_effects.corpus_kernel.batch import TokenArchive

archive = TokenArchive(m, idx)
packet = archive.batch([('live-doc', [7, 3, 9])], min_match=2, lookback=None,
                       locator='cpu')   # or 'triton' on the admitted accelerator
prediction = packet.predict(m, backend='torch')  # use 'triton' for FP32 tile kernels
(-prediction.probabilities[0, 2].log()).backward()
# packet.predict(m, mode='dense') uses exact attention for comparison.
```

## Checks and clocks

19 ordinary CPU checks include a literal all-internal-MEM occurrence oracle, predictions from aligned extensions, and ragged/padded document packet gradient checks. **27 tests passed** with CUDA hidden, including **8** Triton CPU-interpreter cases. They compare the resident locator, reads, source mass, first derivatives, changing training packets and successive optimizer updates to the CPU law. Independent offline compilation assembled **20 SM89 cubins** (three attention geometries × three score modes × forward/backward, and two locator variants). Assembly and CPU interpretation do not establish device parity or speed. [Qualification receipt](receipts/triton_qualified.json).

Invented native IDs, one CPU thread, FP32, default frontier proposal, rank/key/value width 32, leaf size 128. Wall time is the median of five warm blocks of three. Each fixed-bank read includes local and no-read competition. Search/packet/learned-bank construction is *not* timed inside that fixed-bank read. Training times **do** include encoding the live prefix and complete corpus with current weights, rebuilding learned summaries where applicable, native-token loss, backward, clipping and AdamW. Immutable packets are already bound for both methods.

| Corpus rows × live queries | Exact GEMM read | Indexed read (PyTorch) | Exact GEMM optimizer step | Indexed optimizer step |
|---|---:|---:|---:|---:|
| 16,384 × 64 | 12.39 ms | 2.82 ms | 43.57 ms | 56.80 ms |
| 131,072 × 128 | 267.55 ms | 54.16 ms | not measured | not measured |

At the larger size, 139,264 of 16,777,216 pairs were refined (0.83%). Binding the query packet took 10.3 ms; encoding the rows and building their learned summaries took 123.1 and 88.3 ms once for the fixed bank. These clocks demonstrate **amortization for batched fixed-weight reads**, not a single-token latency or a universal advantage. At the measured training geometry the indexed **full update was slower**, despite its cheaper read. [Clock receipt](receipts/cpu-clock.json). The fidelity measurements are a different fixture; neither is a natural LM result.

To reproduce CPU qualification in an activated environment with Torch and Triton installed:

```sh
x=experiments/expressivity_tasks/retrieved_span_effects/corpus_kernel
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GPU_LEASE_HELD PYTHONOPTIMIZE TRITON_INTERPRET CORPUS_KERNEL_DEVICE
python -m pytest -q "$x" --ignore="$x/test_triton.py"
TRITON_INTERPRET=1 python -m pytest -q "$x"
python -m experiments.expressivity_tasks.retrieved_span_effects.corpus_kernel.compile_check /new/path/to/cubins-receipt.json
```

Real-device clocks require a separate positive broker release. The local, single-use `device_gate` refuses admission before successful closure and accounting of the earlier panel and meters. Its cap is the minimum of 180 card-seconds, remaining actual aggregate budget, remaining synthetic engineering allowance and remaining wall time; no launching on a second card, queueing around a failed gate, or retry. `device_meter` compares the same indexed law in eager and Triton, measures it alongside the dense GEMM reference at both sizes, and includes complete current-weight updates at the smaller size. It neither selects natural data nor certifies end-to-end speed in advance.
