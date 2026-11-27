# What the whole-corpus Triton path accelerates

[Pilot results: speed, compression, actual zstd bytes and costs](RESULT.md) · [Registered comparison](PROTOCOL.md) · [Sealed aggregate](receipts/result.json)

The normalizer is over **all** exposed corpus, causal local and no-read rows. A three-byte match proposes an exact score, not eligibility. Outside that interval the reader uses positive feature scores for every row. No hit still leaves a full corpus read. The language pilot has a two-block causal Transformer on the live window; the cheaper document-prefix key encoder and successor vectors in the archive are rebuilt with current weights on every update.

- `corpus_windows/triton_parent.py`: merge the leaf statistics up the dyadic tree in FP32, including an odd last child. Save each child's actual rescaled probability. For child `i`, `dlogZ_i = p_i (dlogZ_parent + dot(dmean_parent, mean_i − mean_parent))`, `dmean_i = p_i dmean_parent`. No row×rank×value activation and no subtracting a huge background to recover a small residual.
- `corpus_windows/triton_mix.py`: the final training read groups partials by live frontier using cached integer offsets. A tile contributes its detached scaling peak `p`, scaled denominator `z` and vector numerator `u`. Stream the tiles to compute `r = Σ exp(p) u / Σ exp(p) z` in stable local-maximum units. With `a = exp(p−max)/scaled_den`, the first VJP is `du = a dr`, `dz = −a dot(dr,r)`. The score gradients pass through the tile's numerator and denominator, not through its detached numerical peak. Descriptor count is not limited to one fixed tile. This value-only path does not replace the API that also returns per-source log masses and shares.
- `language_compression/routes.py`: prebind the complete exact interval and disjoint cheap prefix/suffix covers for regular byte windows. CPU arrays and plans are immutable across weights. All subintervals of a 64-row bank were checked against `Plan.build`; per-byte counts and suffix intervals were also checked against independent literal scans. The specialized planner requires a complete power-of-two leaf tree; the existing irregular planner remains available.
- `language_compression/lm.py`: exact GEMM or indexed Triton tiles, shared local rows, gated lexical count comparator, per-target Triton NLL and causal reset windows. Embedding/linear projections, exact GEMM, Transformer SDPA and AdamW remain PyTorch/vendor operations. Price complete steps, not just reduction kernels.

For an already admitted training worker, the integration is:

```python
model = LanguageLM(seed=5751).to(device)
pack = Pack(build_bytes, model)          # uint8 [128,129], document-local reference
batch = pack.bind(real_windows)          # uint8 [16,65]; label never enters its query
optimizer.zero_grad(set_to_none=True)
loss = model(batch, 'indexed', backend='triton').mean()
loss.backward()                          # includes key/value encoders and all summaries
optimizer.step()                         # next call builds a current-weight archive
```

Only `pack` and `batch` integers/raw byte tables may persist between updates, not a detached learned bank. The same batch and model support `'exact'`, `'count'` and `'local'`. Fixed geometry in this pilot is intentional; use the ragged `WindowArchive.windows` API for other document lengths.

To verify the public implementation **without natural data or a GPU**:

```sh
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONOPTIMIZE GPU_LEASE_HELD TRITON_INTERPRET
x=experiments/expressivity_tasks/retrieved_span_effects
TRITON_INTERPRET=1 "$PY" -m pytest -q "$x/language_compression" "$x/corpus_windows" "$x/corpus_kernel" "$x/semantic_tail" "$x/binding_edits" "$x/test_corpus_attention_law.py"
"$PY" -m experiments.expressivity_tasks.retrieved_span_effects.language_compression.codegen /tmp/new-language-sm89.json
```

Interpreted tests and offline compilation do not establish real-device throughput. See the [terminal result](RESULT.md) for device evidence. No command above opens, reserves or reuses a natural role, and none renews a GPU authorization.
