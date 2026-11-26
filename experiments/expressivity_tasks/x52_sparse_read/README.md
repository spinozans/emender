# X52 — one archive read per four bytes

[Registration](PROTOCOL.md) · [previous every-byte study](../x51_read_retention/RESULT.md)

Question: when no new answer is available, does the reader use a saved answer to predict the next **real byte** better?

- `reader.py`: `acquire` sends only the 16 scheduled suffixes to `Kernel.read` (EVERY uses the efficient full scan). `predict` exposes archive content only on deposited rows. During gaps the direct prior is the FIT unigram for both sparse arms. The query, retained answers, and loss never become observations.
- `study.py`: same four matched models per seed, actual archive/reader clocks, per-age document sums and paired intervals. ERASE4 reuses the exact same acquired packet and weights; its diagnostic time is outside KEEP4's chosen-path cost.
- `data.py` and `run.py`: published qualification before allocation, complete spent-rank replay, one opening per data role, hash/seal all fits and obtain the independent process witness before the first probe byte. No GPU or historical weights are needed.
- `test_sparse.py`: invented suffix request trace, explicit one-frontier tapes, dense parity, gradients, membership/freeze refusals, gap denominators, bootstrap checks.

No quality result follows from the read count. The sparse resolver can require **more native requests** than an every-byte streaming scan, even though it reads fewer answer distributions. Clocks are batched with a shared OS cache, not cold-disk single-user latency.
