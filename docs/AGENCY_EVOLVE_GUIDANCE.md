# Agency evolution guidance (for `wg evolve`)

## The problem to fix
Role assignment has **collapsed to a single grader role** ("Default Evaluator",
role `fb055c2d`, ~1193 tasks). Nearly every task — kernel implementation, 1.3B CMA
searches, grokking sweeps, training harnesses — is being stamped with the *Evaluator*
(grader) role regardless of task type. A grader role doing correctness-critical kernel
and experiment work is wrong and has caused real failures (premature "done", inert
kernels shipped, mis-read verdicts). We need **role differentiation that matches the
work this project actually does**: frontier ML-architecture research — Triton/CUDA
kernels, recurrent-cell design, CMA-ES geometry search, grokking/expressivity sweeps,
length-extrapolation experiments.

## Roles to create / evolve toward (gap-analysis)

1. **Scientific Programmer** — correctness-critical numerical & kernel implementation.
   Triton/CUDA fused kernels, recurrent-cell math, parity vs eager reference (fwd+bwd),
   bf16/precision discipline, `@triton.jit` real kernels (NEVER pure-torch in a
   `triton/` path), no mocks/stubs/fallbacks, assert-fused-no-eager. High-integrity math.

2. **Research Programmer** — builds the experimental apparatus. Training harnesses,
   CMA-ES drivers, sweep orchestrators, task/data generators, aggregators, length-extrap
   eval, GPU-lease integration. Reuses the standard protocol (`cmaes_search_v2.py`, full
   geometry, ≥96 evals, same data slice) rather than inventing bespoke reduced searches.

3. **Research Executor** — runs experiments to completion and reports MEASURED data.
   Leases GPUs via the broker, runs CMA/sweeps, **stays to completion** (does NOT mark a
   task done on launch — only after the search exits and results are committed), monitors
   liveness by *generation/step count* not GPU snapshots, commits curves/results, reports
   numbers with seeds — never a framing doc in place of a run.

4. **Evaluator** — CONSTRAIN to actual agency/evaluation work only: `.evaluate-*`,
   `.flip-*`, scoring, validation/grading, calibrated verdicts. **Must NOT be assigned to
   implementation, kernel, CMA, or experiment-execution tasks.**

## Routing intent (role × task type)
- kernel / fused-Triton / cell implementation → **Scientific Programmer**
- harness / driver / sweep-builder / aggregator → **Research Programmer**
- run CMA / sweep / training / capability battery → **Research Executor**
- `.evaluate-*` / `.flip-*` / scoring / validation → **Evaluator** (only)

## Guardrail (project-specific bias to encode in every role)
Trust measured artifacts over synthesis. No "stop / null" verdict without
constrained-capacity, extrapolation-controlled, fair-comparison data on screen.
Class-separation claims are tested at *constrained capacity + length-extrapolation*
(capacity buys memorization, which the extrapolation regime defeats).
