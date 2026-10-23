# X26 prelaunch validation

lambda01 local CUDA, .venv Python3.12.3/torch2.9.1+cu128, not Frontier.
32 tests passed (7.18s): `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python
-m pytest -q tests/test_merrill26_replication.py tests/test_merrill26_separators.py
tests/test_recipe_controlled_separators.py`; raw receipts/tests.txt and tests_exit.txt.
New tests cover source-class weight identity, dimensions and DeltaNet recurrence,
scalar GRU causality/padding, BCE/CE equivalence, exact auxiliary positivity and
microbatch normalization, curriculum sampling, fixed cards, dataset symlink,
disk preflight and terminal receipt hashes/markers. Existing reviewed task tests
still pass; generators/oracles unchanged. Shell syntax and compileall passed.

Four real max-bin zero-update GPU forward/backward/eval gates passed using a
standalone lease on physical card6 (released afterward): graph90,904 unary tokens,
IMM2,700 scalar tokens. RNNgraph232450 parameters/519414272B peak/88.63s;
Deltagraph430146params/1295851520B/64.30s; RNNinteger791297params/57696256B/.83s;
Deltainteger267841params/319969280B/77.13s (includes first Triton compile).
Raw JSON/text receipts retained. No optimizer update before registration.
Two optional RWKV probes failed upstream v_first=None (exit1), raw tracebacks
retained; no recurrence patches, trained result or success claim for these.

Source snapshots independently compared byte-for-byte to upstream git HEAD blobs.
PDF and layout SHA receipts preserved; Table1 LR really3e-4, not markdown damage.
Independent acceptance review required. Source-provenance snapshots intentionally
retain upstream whitespace; executable edits checked separately. No claim that
our scalar IMM adaptation is an exact Figure2 model/data reproduction.
Resilient requirement IDs R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07:
not applicable; no resilient behavior, Frontier queue or WorksGood service changed.
