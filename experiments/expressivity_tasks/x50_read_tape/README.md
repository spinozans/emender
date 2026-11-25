# X50: can the model learn what to look up next?

**[Completed natural comparison](RESULT.md) · [aggregate](receipts/result.json).** The program can inspect a real XSA reply and decide which extension to read. Learning that choice did not improve prediction here. On 256 fresh TRAIN-probe documents, ADAPT was worse than ONE at all three seeds (+.002731 nats/byte loss) and took 2.38× the batched evaluation time. Following the most common branch beat ADAPT, but did not reliably beat one read. Keep the tape; prefer ONE in this prototype.

At each real byte the program asks a mounted string index about the visible suffix. **ONE** attends to the observed bytes and those answers. **ADAPT** reads the first answer, chooses a hypothetical next byte, and asks the index about that extension before predicting. The extension and its answer stay on the work tape; the extension never becomes observed text. **BLIND** and **GREEDY** get two reads as well. BLIND cannot see answers when choosing its question. GREEDY always follows the most common continuation. See the [prospective comparison and scope](PROTOCOL.md).

- [`kernel.py`](kernel.py): `Kernel.read([b'New '])` returns a 256-bin byte-valued suffix-kernel reduction. `follow(parent, byte)` is the same read at the extended query and evaluates only the chosen extension. It reuses logical paths from the parent and gathers whole, scoped, immutable P100 m20 pages. The query result can be a single row instead of one token per occurrence because identical byte values have additive weights. This does not reduce arbitrary occurrence-specific neural values to counts.
- [`program.py`](program.py): `execute` runs without labels. ADAPT appends typed observations, answers and chosen operands and computes each next action against its causal prefix. `predict` attends over the resulting sequence to produce **all 256** logits. `score` introduces external labels only after execution and includes the downstream return in the gradient for a hard address choice. The first read is compulsory and the arm fixes the number of reads; this is not an unrestricted planner.
- [`data.py`](data.py), [`study.py`](study.py), [`run.py`](run.py): new document-disjoint TRAIN fit/probe, one native archive transport, paired CPU fitting, complete freeze before probe, spent reservation and receipt-only close. No old windows, GPU or final split needed. Do not replay a spent natural job; run the invented tests instead.

### Try the loop on your own prefix

[`ReadSession`](online/stream.py) accepts one real byte at a time, with no future input buffer. Each `observe(b)` performs the XSA read(s), appends typed rows to the attention cache and returns the 256 next-byte logits and the separate hypothetical extension, if any. `reset()` starts a new window. The trained position range is 64 external bytes. On the project machine with the pinned P100 mount and private FIT states:

```sh
env -u GPU_LEASE_HELD -u PYTHONOPTIMIZE CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m experiments.expressivity_tasks.x50_read_tape.online --arm ADAPT --text 'New '
```

Use `--arm ONE` for the cheaper direct read; `--hex` accepts binary input. `--birth` explicitly uses the unfitted initializer if you only have the mount; it does **not** reproduce trained quality. With no mount, the fixture tests below run independently. The streaming API was added **after** experiment closure, not substituted for the scored batch implementation. All 288 issued follow-up bytes on two invented streams agreed with the registered runner at all applicable frozen states; across all 12 models/384 frontiers, worst absolute discrepancy in any logit was `1.91e-6` ([parity receipt](receipts/online_parity.json)). No extra natural evaluation.

### Tests

From the repository root, on the project Python environment:

```sh
env -u PYTHONOPTIMIZE -u GPU_LEASE_HELD CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q experiments/expressivity_tasks/x50_read_tape tests/test_x50_read_session.py
```

These tests use invented sources. They compare the kernel with explicit position enumeration, check no unchosen child page is fetched, compare incremental decisions and full-tape replay at every prefix, intervene on first and past second answers, and compare delayed action gradients with exact expected-loss differentiation. [`mount.py`](mount.py) is the separate native check: it compares invented queries into the actual whole P100 mount with the independent stock `branch-of` executable before natural reservation.

**Retained qualification issue.** Thirteen mechanism tests passed. The first full gate then stopped during pytest collection: `attempted relative import beyond top-level package` when pytest loaded `x50_read_tape` as a top-level test package. The failed test log SHA-256 is `820a58874ba63018e6553e343ef054301ee5bab49c1994d2f2403a39bf78f9bf`. Sibling experiment imports now use their absolute namespace, preserving both normal CLI and pytest invocation. No model, selection, exposure or decision changed; no natural document had been reserved or opened.
