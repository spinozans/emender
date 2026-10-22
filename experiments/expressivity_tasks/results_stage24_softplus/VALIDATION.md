# X24 pre-update validation

Host: lambda01 (NVIDIA local CUDA, not Frontier; project .venv Python3.12).
GPU7 physically idle and exclusively shell-leased during CUDA tests/smokes.
No WorksGood service/dispatcher/worker machinery used.

Commands (from repo root; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2):

```
export GPU_LEASE_VISIBLE=7
eval "$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"
X24_KERNEL_RECEIPT=experiments/expressivity_tasks/results_stage24_softplus/receipts/kernel_match.json \
.venv/bin/python -m pytest -q tests/test_e97_softplus.py tests/test_e97_softplus_graph.py \
 tests/test_e88_state_activation.py tests/test_e97_facade.py tests/test_e97_recurrent_precision.py \
 tests/test_e88_triton.py tests/test_e88_triton_optimized.py tests/test_merrill26_separators.py
.venv/bin/python -m experiments.expressivity_tasks.run_e97_softplus_graph \
 --smoke experiments/expressivity_tasks/results_stage24_softplus/receipts/smoke_e97_graph.json
bash -n scripts/launch_e97_softplus_graph.sh
git diff --check -- ndm
```

Final tests: **87 passed**, no skips (51.26s), exit0; raw receipts/tests.txt.
Kernel gate explicitly instruments fused facade dispatch, train and eval,
production nonchunked and projection-chunked paths. Exact case metrics/tolerances
in receipts/kernel_match.json; all gate calls and smoke are zero optimizer updates.
Smoke: N300, 90,904 tokens, peak allocation10,290,562,560 bytes, finite backward,
fused evaluation, 8,950,272 parameters; raw runtime logs name state=softplus.

Regression coverage includes unchanged tanh/linear, fixed FP32 policy, legacy
checkpoint dtype, masking/reset gradients, gate/no-gate, extreme softplus states,
unaligned eval, exact initial weights/shapes, actual X23E shared data link, disk
preflight, and identity-verified group signalling. This is a launch qualification,
not a training/convergence or length-generalization result. All three scientific
predictions remain unadjudicated. No automatic instability retry is authorized.

Resilient requirement IDs R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07:
not applicable, standalone local research; no resilient conformance or Frontier
Partition/QOS binding is claimed. Independent acceptance review remains required.
