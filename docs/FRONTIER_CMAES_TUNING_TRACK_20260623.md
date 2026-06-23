# Frontier CMAES Tuning Track for E97-MLP and GDN2-MLP

Date: 2026-06-23
Task: `design-frontier-cmaes-tuning-track`

## Scope and Priority Boundary

This is a design-only, small-budget experiment track. It is explicitly
non-blocking for immediate E97-MLP checkpoint finalization, walltime-aware final
checkpointing, live Frontier resume validation, and the E97-first DiLoCo canary.
Do not assign workers or Frontier allocation to this track until those immediate
E97 checkpoint/DiLoCo gates have either completed or a human explicitly records
that a small CMAES smoke can run without delaying them.

E97-MLP is the primary research arm. GDN2-MLP is the high-quality control arm.
The purpose here is not to optimize GDN2 ahead of E97; it is to test whether
Frontier-specific MI250X/ROCm/runtime conditions change which CMAES/model
configurations are plausible for E97-MLP, while using GDN2-MLP as a clean
reference for systems noise and search-protocol calibration.

This track should not implement checkpointing, launch DiLoCo scaleout, edit the
CMAES driver, or submit production sweeps. It should produce evidence only when
the budget is small enough to be a diagnostic and when it does not compete with
the active E97 checkpoint/finalization and E97-first DiLoCo canary work.

## Current Evidence to Treat as Priors

- `docs/FRONTIER_E97_CHECKPOINT_DILOCO_QUALITY_PASS_20260623.md` states the
  project priority: E97-MLP is primary, GDN2-MLP is control, and scaleout is
  gated behind live E97 checkpoint/resume evidence.
- `docs/FRONTIER_POST_CACHE_DEBUG_20260621.md` shows the shared `p50k_base`
  cache works on compute nodes. E97-MLP reached finite loss, throughput, memory,
  and checkpoint evidence, but the recorded job predates the rank-mapping fix and
  ended failed after producing evidence. GDN2-MLP completed a one-node debug run
  with finite loss, throughput, memory, fused-path guard, and checkpoint.
- `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md` gates multi-node work on
  training metrics, fused/no-eager guards, checkpoint/manifest evidence, ledger
  accounting, and human approval. This CMAES track inherits those guardrails.
- `docs/repro/lb_gdn2_mlp_20260612/REPRODUCTION.md` provides a clean 1.3B
  GDN2-MLP control under the standard driver: 104 evals, popsize 8,
  `min_generations=13`, `train_minutes=15`, `chunk_size=2048`, `bf16`, fused
  official GDN-2 path, best avg-loss 5.8949.
- `docs/repro/lb_emender_mix_20260612/REPRODUCTION.md` shows a full-range
  Emender mixture search moving strongly toward the nonlinear/E97 side, but it
  is a local protocol result, not a Frontier runtime verdict.

## Frontier-Measurable Knobs

Only knobs that are both plausible and measurable on Frontier should enter this
track. Each candidate eval must emit enough metadata to compare loss, throughput,
memory, fused path, restartability, and cost per useful candidate.

### Shared CMAES Protocol Knobs

| Knob | Plausible range | Why measurable on Frontier | Keep fixed initially? |
| --- | --- | --- | --- |
| `train_minutes` per candidate | 5, 10, 15 minutes | Directly trades allocation for fitness fidelity; measurable via final avg loss, final-window loss, and candidate runtime. | Use 5 minutes for smoke, 10 minutes for mini-track, do not exceed 15 before go. |
| `popsize` | 4 or 8 | Controls parallel candidate pressure and scheduler shape; measurable by valid evals/hour and best-so-far variance. | Start 4 for cheap Frontier diagnostics; compare to local popsize 8 only after gate. |
| `min_generations` / max evals | 2-4 generations for smoke, 6 generations for mini-track | Determines whether search is real or just runtime probing; measurable by best-so-far curve and sigma movement. | Cap tightly before checkpoint/DiLoCo completes. |
| `sigma` | 0.35, 0.5, 0.8 | Frontier runtime may make memory/probe failures uneven across wide candidates; measurable by invalid-candidate rate and parameter feasibility. | Use 0.5 for smoke; only test 0.8 after stability. |
| Warm start / anchors | Pinned known-good E97 and GDN2 anchors | Avoids wasting allocation on known impossible corners; measurable by anchor reproducibility and candidate feasibility. | Required for every small-budget run. |
| `param_tolerance` | 0.03, optionally 0.02 | Parameter-count accuracy affects fairness; measurable by accepted candidate params. | Keep 0.03 to match standard protocol until Frontier feasibility is proven. |
| Candidate timeout / max valid attempts | Small bounded retry budget | Frontier queue/runtime issues can inflate invalid attempts; measurable by invalid cause counts. | Required; stop on repeated invalids. |
| Seed/data slice/tokenizer | Same p50k and staged commapile/pile-compatible slice | Controls comparability; measurable through metadata and tokenizer probe. | Fixed. |

### Shared Training/Runtime Knobs

| Knob | Plausible values | Measurement |
| --- | --- | --- |
| `batch_size` | CMA axis, clamped by memory probe; candidate values likely 1-8 for 1.3B | Peak memory, OOM rate, tokens/sec, loss per token, valid eval count. |
| `chunk_size` | 2048 fixed first; optional 1024 or 4096 diagnostic only | Fused-kernel path, memory, tokens/sec, loss comparability. |
| `bf16` | Required on Frontier for production-like kernels | Fused/no-eager guard, finite loss, nonfinite grad/loss rate. |
| `grad_clip` | Keep default 1.0 initially | Existing docs show clipping is load-bearing; changing it would confound architecture and runtime tuning. |
| `optimizer` / LR schedule | Keep current CMA candidate `lr` with ScheduleFree unless explicitly testing schedule sensitivity | Loss slope, grad norm, checkpoint resume behavior. |
| compile/autotune behavior | Record startup/compile separately from timed training | Candidate wallclock, steady-state tokens/sec, invalid timeout attribution. |
| output/checkpoint cadence | At least candidate summary; checkpoint only if the current E97 checkpoint implementation is known safe | Do not add checkpoint load to this track before E97 checkpoint tasks finish. |

### E97-Primary Model Knobs

These are the research knobs. They should be explored only after an E97 one-node
post-rank-fix checkpoint/resume validation exists.

| Knob | Plausible range | Hypothesis |
| --- | --- | --- |
| `dim` | around prior E97/CMA anchors; 128-multiple values near 1.3B target | Frontier memory and MI250X kernel occupancy may prefer different width/depth balance than local GPUs. |
| `depth` | shallow-to-mid range around existing E97 winners | Frontier runtime may penalize deeper recurrent stacks through launch overhead and activation memory. |
| `n_heads` | E97 prior regimes around hundreds of heads, constrained by state/head dims | MI250X occupancy/register pressure may favor fewer/larger or more/smaller heads differently than local CUDA results. |
| `n_state` | keep 32 initially unless testing typed/mix variants already support it | Changing state size changes recurrence memory and should not be mixed into the first Frontier smoke. |
| `lr` | CMA log axis around existing E97 tuned LR near 1e-3 | Frontier batch/throughput and bf16 numerics may shift stable LR. |
| E97 raw/delta/typed mixture | E97-MLP primary first; raw/mix variants only as follow-up | Do not let ablations displace the main E97-MLP checkpoint/DiLoCo work. |
| `use_chunked_e97` / `e97_chunk_size` | Only after chunked-E97 ROCm quarantine is resolved; values 16/32/64 are code-valid | Current quarantine means this is not a first-track knob. |

### GDN2-Control Model Knobs

These knobs calibrate whether the Frontier CMAES protocol can reproduce the
known high-quality control. They must not become the dominant optimization path.

| Knob | Plausible range | Hypothesis |
| --- | --- | --- |
| `dim` | 1536-3072, multiples of 128 | The clean control should remain near the known 2176-ish regime if the protocol is sound. |
| `depth` | 10-32 | Frontier may favor shallower GDN2 if fused kernel throughput dominates fitness at short budgets. |
| `n_heads` | 8-40 | Existing control prefers small head counts; Frontier should not need a radically different head regime unless runtime changes dominate. |
| `expansion` | 1-3 | Keep in the standard GDN2-MLP search space. |
| `gdn2_mlp_ratio` | 2.0-4.0 | Standard control axis; measurable by loss and params. |
| `lr` | 1e-4 to 3e-3 log scale | Check whether Frontier batch/runtime shifts the best LR from the local control near 4.7e-4. |
| `use_conv` / `d_conv` | Keep official-style `use_conv=1`, `d_conv=4` | Do not mutate the control definition in the first pass. |

## Hypotheses

### E97-Primary Hypotheses

1. Frontier MI250X/ROCm conditions may make the E97-MLP loss-vs-wallclock
   fitness prefer a different width/depth/head-count balance than the local
   CUDA CMAES runs, even if matched-token loss is similar.
2. E97-MLP candidate feasibility may be limited by fused Triton path behavior,
   compile/autotune overhead, or memory pressure rather than by optimizer
   stability. If so, a Frontier-specific search should optimize valid
   tokens/node-hour and not only short-run loss.
3. If the E97 primary anchor cannot complete repeated short candidate evals with
   finite loss, fused/no-eager guard, clean rank behavior, and checkpoint-safe
   metadata, then CMAES is premature and allocation should remain on
   checkpoint/resume and DiLoCo readiness.
4. If a tiny E97 mini-track produces multiple valid candidates and improves
   best-so-far loss or tokens/node-hour over the pinned E97 anchor by a
   practically visible margin, a larger E97-only Frontier CMAES sweep may be
   justified after DiLoCo canary gating.

### GDN2-Control Hypotheses

1. GDN2-MLP should reproduce a stable, high-quality control regime on Frontier
   under the same fused/no-eager constraints. If it does not, the problem is
   likely the Frontier CMAES protocol, data path, or runtime accounting rather
   than E97 model quality.
2. The control should not require broad retuning to beat E97 operationally. If a
   small GDN2 mini-track finds a much better Frontier-specific regime but E97
   checkpoint work remains incomplete, record the result as a control finding
   and do not divert primary workers from E97.
3. GDN2 runtime metrics are useful for separating search noise from E97-specific
   kernel/runtime issues because the one-node GDN2 smoke already completed with
   finite loss and checkpoint evidence.

## Proposed Small-Budget Matrix

This matrix is intentionally smaller than the DiLoCo canary ladder. It should be
run only after the immediate E97 checkpoint/finalization work has a clean live
Frontier result or after human approval states that the budget will not delay
that path.

Node-hour accounting uses requested node-hours, not optimistic consumed time.
Each row uses one Frontier node unless explicitly stated. A row is complete only
if candidate summaries include args, params, rank mapping, fused guard, first
loss, final-window loss, grad norm/nonfinite status, peak memory, compile/startup
time, steady-state tokens/sec, valid/invalid eval count, and requested/consumed
node-hours.

| ID | Arm | Purpose | Shape | Requested node-hours | Stop / continue rule |
| --- | --- | --- | --- | ---: | --- |
| `CMA-FRONTIER-PREFLIGHT-E97` | E97-MLP primary | No-allocation design gate using existing artifacts plus optional syntax/metadata dry run only | No Frontier job | 0 | Continue only if the active checkpoint/finalization tasks have not reserved the same worker focus and the needed E97 anchor/config metadata is available. |
| `CMA-FRONTIER-SMOKE-E97` | E97-MLP primary | Cheap candidate-run smoke from one pinned E97 anchor plus 3 nearby candidates, no broad search | 1 node x 0.5 h | 0.5 | Stop if any fused/no-eager guard is missing, any nonfinite loss/grad occurs, invalid candidates exceed 1/4, no first training metric appears by 10 minutes, checkpoint/metadata races reappear, or throughput is below the prior one-node E97 debug by >2x after compile without explanation. |
| `CMA-FRONTIER-SMOKE-GDN2` | GDN2-MLP control | Same smoke shape from known GDN2 control anchor to calibrate runtime/search noise | 1 node x 0.5 h | 0.5 | Stop if GDN2 fails to reproduce finite fused-path training and plausible throughput; if E97 failed but GDN2 passes, return to E97 kernel/checkpoint diagnosis instead of launching sweeps. |
| `CMA-FRONTIER-MINI-E97` | E97-MLP primary | Bounded CMAES mini-track: popsize 4, 4 generations, 5-10 min candidates, warm-started | 1 node x 4 h | 4 | Continue only if at least 14/16 candidates are valid, best candidate improves pinned-anchor short-run fitness or tokens/node-hour by >=3%, no checkpoint/metadata issue appears, and loss ranking is not dominated by startup artifacts. |
| `CMA-FRONTIER-MINI-GDN2` | GDN2-MLP control | Optional control mini-track with same popsize/generation budget | 1 node x 4 h | 4 | Run only if E97 mini-track passes or if E97 fails and a control is needed to isolate runtime. Do not run solely to optimize GDN2. Stop if it merely confirms the known control regime within noise. |
| `CMA-FRONTIER-REPLAY-TOP2-E97` | E97-MLP primary | Replay anchor plus top 2 E97 candidates for longer 15-minute evals | 1 node x 1 h | 1 | Promote only if longer replay preserves ranking and improves either loss or tokens/node-hour without new stability issues. |
| `CMA-FRONTIER-REPLAY-TOP2-GDN2` | GDN2-MLP control | Optional matched replay for calibration | 1 node x 1 h | 1 | Use only to interpret E97; do not let this become a GDN2-first sweep. |

Recommended maximum before a human reviews: 1 node-hour for the two smokes, or
5-6 node-hours if the E97 mini-track and one replay are approved. The full
optional matrix is 12 node-hours if both arms and both replays run, but that
should require explicit approval because it is already larger than a diagnostic.

No obvious zero-risk Frontier smoke exists beyond dry-run/syntax/metadata checks:
even a single short candidate consumes allocation and may compete with active
E97 checkpoint/DiLoCo workers. Therefore the default action from this design is
to wait for the checkpoint/finalization tasks, not to submit a job now.

## Stopping Rules

Stop the track immediately and return focus to E97 checkpoint/DiLoCo if any of
the following occur:

- Active E97 checkpoint/finalization, walltime-aware checkpoint, or E97-first
  DiLoCo canary work needs the same worker focus, repository state, or
  allocation window.
- E97 lacks a clean live Frontier checkpoint/resume validation after the current
  checkpoint tasks finish.
- Any E97 candidate shows nonfinite loss/grad, missing fused/no-eager guard,
  rank mapping regression, checkpoint/latest metadata race, or repeated invalid
  parameter/memory probes.
- Candidate startup/compile time consumes more than half of the candidate budget
  for 2 consecutive candidates, making fitness mostly a compile artifact.
- Valid candidate rate is below 85% after the first 8 attempted candidates.
- GDN2 control fails the same smoke that previously completed, indicating a
  runtime/protocol regression rather than useful model evidence.
- Requested allocation for the track would exceed 6 node-hours before a human
  reviews the smoke/mini-track artifact.

## Go / No-Go for Spending Beyond Diagnostics

Current decision: no-go for CMAES sweeps. The project should continue immediate
E97 checkpoint/finalization and E97-first DiLoCo canary work first.

Move to a diagnostic smoke only when all are true:

- The active E97 checkpoint/finalization tasks have recorded live Frontier
  checkpoint and resume evidence, or human approval explicitly permits a
  non-blocking smoke before that completes.
- The run can be capped at 1 requested node-hour total for E97+GDN2 smokes.
- The artifact path, ledger debit, and stop rules are recorded before
  submission.
- The smoke uses pinned known-good anchors and cannot silently launch a broad
  sweep.

Move from smoke to E97 mini-track only when all are true:

- E97 smoke completes cleanly with finite loss/grad, fused/no-eager guard,
  plausible throughput, and no checkpoint/metadata regression.
- GDN2 either completes its control smoke or a documented reason explains why
  the control is not needed.
- The E97 mini-track consumes no more than 4 requested node-hours and uses
  popsize 4 with a hard generation cap.
- The DiLoCo canary path is not waiting on the same person, worktree, or queue
  slot.

Justify a larger Frontier CMAES sweep only if all are true:

- The E97 mini-track has a reproducible candidate or configuration trend that
  improves pinned-anchor short-run fitness or tokens/node-hour by at least 3%
  after replay, with stable loss and no runtime regressions.
- The GDN2 control confirms that the observed improvement is not merely a
  Frontier timing artifact or data/tokenizer mismatch.
- E97 checkpoint/resume and the E97-first DiLoCo canary have produced clean
  evidence, so a sweep will not delay higher-priority scaleout readiness.
- A human approves a new matrix with ledger impact, reserve protection, expected
  eval count, and explicit opportunity cost versus DiLoCo scaleout.

Continue E97 checkpoint/DiLoCo instead of CMAES when:

- The only CMAES evidence is feasibility, not improvement.
- GDN2 looks cleaner than E97 because E97 checkpoint/runtime issues remain
  unresolved.
- Search ranking changes are within short-run noise or driven by compile time,
  invalid-candidate filtering, or different checkpoint behavior.
- The next E97 DiLoCo canary can answer a higher-priority research/systems
  question for comparable or lower allocation.

## Artifact Requirements for Future Execution

Any future execution task created from this design should write a single summary
document with:

- exact command/environment and git commit;
- allocation ledger row before submission and consumed node-hours afterward;
- per-candidate JSONL with args, params, status, invalid reason, loss metrics,
  grad norm, memory, tokens/sec, compile/startup time, fused guard, checkpoint
  metadata status, and seed/data/tokenizer identifiers;
- separate E97-primary and GDN2-control sections;
- a decision table mapping observed evidence to the go/no-go rules above;
- an explicit note that the track remained non-blocking for immediate E97
  checkpoint/finalization and DiLoCo canary work.

## Validation Checklist

- Plausible Frontier-measurable CMAES/model knobs are listed for shared
  protocol/runtime, E97-primary, and GDN2-control cases.
- The experiment matrix is capped to 0, 1, 4, 6, or 12 requested node-hour
  decision points, with stopping rules at every step.
- E97-primary hypotheses and GDN2-control hypotheses are separated.
- Spending beyond diagnostics is justified only by replayed E97 improvement,
  GDN2 calibration, clean checkpoint/resume evidence, and explicit human
  approval.
- The track is explicitly non-blocking for immediate E97 checkpoint/finalization
  and E97-first DiLoCo canary work.
