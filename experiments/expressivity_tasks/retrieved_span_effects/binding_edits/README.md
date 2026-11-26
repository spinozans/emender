# Reproducing the synthetic row-edit checks

Both edit studies exercise `CorpusTokenLM` with its learned vector value and normal vocabulary head. They never mix live documents or expose their final targets as inputs. The [permutation protocol](../semantic_tail/PROTOCOL.md) and [repeated-value protocol](PROTOCOL.md) specify different questions. For the registered fits use FP32, Python 3.12, PyTorch 2.9.1 and a single CPU thread. Do **not** acquire a GPU.

From the repository root (`PY` is the project Python):

```sh
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GPU_LEASE_HELD PYTHONOPTIMIZE TRITON_INTERPRET
x=experiments/expressivity_tasks/retrieved_span_effects
m=experiments.expressivity_tasks.retrieved_span_effects
"$PY" -m pytest -q "$x/semantic_tail" "$x/binding_edits"
"$PY" -m "$m.semantic_tail.audit" "$x/semantic_tail/receipts/result.json"
"$PY" -m "$m.binding_edits.audit" "$x/binding_edits/receipts/result.json"
```

The tests include a literal normalized row read, independent logit-difference calculation, changed/unchanged target alignment, a counterexample that solves permutations by excluding the desired row, and checks for disjoint, balanced splits. The last two commands verify the public aggregates **without weights or new fits**.

**Training and identity.** `python -m <package>.run fit NEW_DIR --public-ref REF` runs all nine final fits into a *new*, private directory. Use a clean checkout of the respective qualified ref: `1d0f5a4615d26da1aa51d9b31774e49a7d7037d1` for `semantic_tail`, `dbf6e19af0172e1e742e42bee1c4a5596971e933` for `binding_edits`. Use the environment above; keep outputs **outside** the checkout. A fit caches only immutable tokens/routes. Check all weights, orders, starts and exposure receipts before supplying `probe NEW_DIR --public-ref REF --freeze-sha SHA256_OF_VERIFIED_FROZEN_JSON`. The repeated-value runner additionally requires `--original DIR` for the **original nine permutation fits**; it checks their registered manifest hash and reads no original probe scores. Do not pass different weights off as the closed transfer panel. A fresh training replay may reproduce the model-state hashes but will not recreate a byte-identical timed freeze. Distributed artifacts contain the hashes and aggregates, not the private weights.
