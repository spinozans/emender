# Corrected E99 1.3B Mixture-Aware CMA - Launch Gate

**Date:** 2026-06-06  
**Task:** `corrected-e99-1-3b`  
**Status:** BLOCKED before launch, by design.

This is a launch-gate record, not an experiment report. No corrected
mixture-aware CMA, LM screen, capability screen, checkpoint publication, or
multi-day run was launched from this task instance.

## Gate Verdict

The two technical prerequisites are complete:

| prerequisite | status | evidence |
|---|---:|---|
| `implement-triton-fused` | done | WG log reports commit `e234cc8`; targeted shell tests and full pytest passed. |
| `fix-e99-mixture` | done | WG log reports commit `3e7ab3e`; per-mixture sizing smoke passed and typed-head tests passed. |

The human-approval prerequisite is **not** complete. I checked the current task
messages and WG records for `corrected-e99-1-3b`; there is no human message that
explicitly approves launching this corrected 1.3B run. The most relevant user
message in the graph is the earlier `redo-e99-1-3b` stop instruction from
2026-06-06T08:11:39Z: stop additional evaluation launches, audit kernel/sizing,
and wait for revised instructions.

Because the task text says the run remains gated behind explicit human go, the
correct action is to stop before any GPU launch.

## Safeguard Added

The corrected launch drivers now fail closed:

- `experiments/e99_mixture_aware_lm_cma/run_mixture_cma.py`
- `experiments/e99_mixture_aware_lm_cma/run_capability.py`

Any GPU-launch path now requires:

```bash
--approved-human-go \
--approval-note "<human approval text naming corrected-e99-1-3b and launch approval>"
```

The CPU-only `--smoke_param_check` path remains available without approval so
future agents can validate parameter sizing without starting a run.

When approved, the approval note is recorded into the generated run summary:

- `all_results.json` for the LM screen/CMA driver
- `capability_summary.json` for the capability driver

## Validation Against Task Criteria

| criterion | status | note |
|---|---:|---|
| Both prerequisites complete before any run | pass | Both upstream tasks are WG `done`; no run was launched before checking. |
| Explicit human approval recorded before launch | blocked | No approval was present, so launch was refused. |
| Every candidate param-matched to target +/-2% | launch-ready | The driver derives `dim` per deterministic head allocation and asserts the exact counted parameter target before launch. CPU smoke remains available. |
| No head ranked by wallclock-loss unless all compared heads are fused | launch-ready | The driver can run the fused shell path from `implement-triton-fused`; shell controls remain explicitly separable in results by head counts and role. |
| Report separates computational mechanism from implementation artifact | launch-ready | The prior audit already identifies the shell implementation artifact; the corrected run has not produced mechanism results yet. |
| Idle-GPU-only / no-preempt; GPU set + parallelism logged | launch-ready | The driver uses the `<2GB used memory` idle test, avoids occupied GPUs, and records per-eval GPU assignments. Not exercised here because approval is absent. |

## Non-Launch Validation Performed

Safe checks that do not start GPU training:

```bash
PYTHONPATH=. python experiments/e99_mixture_aware_lm_cma/run_mixture_cma.py --smoke_param_check
python -m py_compile experiments/e99_mixture_aware_lm_cma/run_mixture_cma.py experiments/e99_mixture_aware_lm_cma/run_capability.py
python experiments/e99_mixture_aware_lm_cma/run_mixture_cma.py --skip_cma 1
python experiments/e99_mixture_aware_lm_cma/run_capability.py --steps 1
```

The last two commands are expected to fail before launch with the new approval
gate if `--approved-human-go` and a qualifying `--approval-note` are absent.

## Approved Launch Shape

After a human approval message exists, the corrected run should be started with a
command in this shape:

```bash
PYTHONPATH=. python experiments/e99_mixture_aware_lm_cma/run_mixture_cma.py \
  --approved-human-go \
  --approval-note "HUMAN APPROVAL: corrected-e99-1-3b launch approved by <name> on <date>; budget/GPU scope: <scope>" \
  --output experiments/e99_mixture_aware_lm_cma/results/run1 \
  --gpus <idle_gpu_list> \
  --wall_minutes 15 \
  --anchor_roundtrip 1
```

Capability runs should carry the same approval note:

```bash
PYTHONPATH=. python experiments/e99_mixture_aware_lm_cma/run_capability.py \
  --approved-human-go \
  --approval-note "HUMAN APPROVAL: corrected-e99-1-3b launch approved by <name> on <date>; budget/GPU scope: <scope>" \
  --output experiments/e99_mixture_aware_lm_cma/results/capability \
  --gpus <idle_gpu_list>
```

Selection remains token-matched LM loss plus MQAR/S5/nonlinear-state/mixed
capability. Wallclock may be reported with implementation-tax context, but must
not select across non-fused paths unless the compared heads are all fused.

## Next Required Event

A human must post an explicit approval message for `corrected-e99-1-3b` before
any agent retries the corrected 1.3B launch. Without that message, the correct
state is blocked, not partially run.
