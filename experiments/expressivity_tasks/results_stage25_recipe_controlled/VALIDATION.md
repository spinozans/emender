# X25 prelaunch validation

Host lambda01, local NVIDIA CUDA; project .venv Python3.12.3 (not Frontier).
No WorksGood service/dispatcher machinery; standalone shell GPU leases only.

`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q
 tests/test_merrill26_separators.py tests/test_e97_softplus_graph.py
 tests/test_recipe_controlled_separators.py`: **28 passed in8.87s**, exit0.
Raw receipt: receipts/tests.txt, tests_exit_code.txt. Covers exact task oracles,
model shapes/weights, unchanged hot defaults, constant paper LR/WD configuration,
five fixed card bindings, refusal of out-of-scope rungs, actual shared dataset
link, disk preflight before launch mutation, identity-checked TERM/KILL guard,
and terminal hash/marker validation. No optimizer training runs in these tests.

`bash -n scripts/launch_recipe_controlled_separators.sh`: passed.
Task-scoped `git diff --check`: passed. Repository-wide check also found pre-existing
unrelated mutable growth-report trailing blank lines; no unrelated files edited.

Historical X23E max-bin tanh/GDN2 graph/integer zero-update forward/backward/eval
smokes and X24 softplus smoke/kernel-match gate are deliberately reused, not new
instrumented runs. Their raw source hashes and copies are bound by registration.
No model, engine, precision, data, or input-shape changes; registration builder
checks X24 model/kernel and dependency bytes still match that qualified engine.
The trained GDN2 nonfinite diagnosis only extracts existing stream/traceback;
state-magnitude evidence is unavailable, recorded without inventing telemetry.

Hot graph kills verified PID/starttime/host and owned process groups before TERM;
workers/trainers dead afterward, existing best.pt SHA256 verified/preserved.
RECIPE-CONFOUNDED/SUPERSEDED receipts are right-censored, not training verdicts.

Before launch trainer checks all source/dependency bytes and committed HEAD blobs;
watcher checks remote commit equality, 400GiB disk preflight, existing smoke gates,
then starts only the five authorized fresh rungs. Config records actual AdamW
WD and LR at updates1/5000/60000 plus source commit, registration/data identities.
Independent resource guard runs60s; native arming remains supervisor-owned.

Resilient requirement IDs R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07:
not applicable, local standalone research. No resilient behavior or Frontier
Partition/QOS binding changed/claimed. Acceptance review required; no training
outcome is established by launch or these prelaunch tests.
