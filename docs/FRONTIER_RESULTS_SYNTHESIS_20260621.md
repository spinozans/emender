# Frontier Results Synthesis and Next Launch Decision - 2026-06-21

Task: `frontier-results-synthesis`

## Decision Memo

Recommendation: **submit no extended-queue Frontier job now**. The next
concrete Frontier allocation burn should be a post-cache debug retry, followed
by an 8-node canary only if that debug retry records first training metrics,
checkpoint/resume evidence, ledger accounting, and human approval.

This is a conditional launch decision, not an inherited no-go. The observed
Frontier evidence shows that the integrated tree can submit debug jobs, import
the staged ROCm runtime, see MI250X GPUs, run the `e97-MLP` fused split-edit
kernel smoke, and run the `gdn2-MLP` external GDN2 bf16 preflight. It does not
yet show Frontier training loss, throughput, peak memory, checkpoint creation,
resume behavior, multi-node communication, DiLoCo merge behavior, or 64-node
readiness for any arm.

Immediate next spend should therefore be limited to **6 requested node-hours**
for debug follow-up: one 2-hour `e97-MLP` post-cache run, one 2-hour `gdn2-MLP`
post-cache run, and one 2-hour checkpoint/resume run for the selected arm. If
those pass and a human approves, the next scaleout submission should be one
**8-node x 4-hour canary**, adding **32 requested node-hours**. The combined
debug-plus-canary burn is **38 requested node-hours**, leaving
**19,961.802776 node-hours** from the current ledger baseline of
**19,999.802776**, with the **4,928 node-hour reserve intact**.

The prepared 64-node x 24-hour e97 package remains useful as an auditable
artifact, but it should stay blocked. A direct 64x24h launch would request
**1,536 node-hours** and would spend **10.2%** of the current non-reserve
spendable balance before the project has first-loss or checkpoint evidence.

## Completed Evidence and Artifacts

| WG task | Job ids / commits / artifacts | Observed evidence | Boundary |
| --- | --- | --- | --- |
| `frontier-orient-handoff-paper` | `docs/FRONTIER_ORIENTATION_BRIEF_20260621.md` | Separates paper claims, handoff interpretations, and Frontier evidence requirements. Identifies `emender-mlp`/`e97`, `e97-linear`, `gdn2-mlp`, DiLoCo, commapile, and ROCm kernel portability as the relevant workstream. | Orientation evidence only. It does not authorize a Frontier launch and explicitly treats architecture winner/no-go claims as hypotheses until Frontier evidence exists. |
| `rocm-kernel-port-audit` | `docs/ROCM_TRITON_KERNEL_PORT_AUDIT_20260621.md` | Audits E97/e97-linear/GDN2 ROCm risks and smallest tests. Notes E97 fused paths, chunked linear-state path, GDN2 external import path, and need for no-eager fused guards. | Static audit and implementation plan, not Frontier runtime proof. |
| `rocm-e97-mlp-port` integrated by `frontier-merge-rocm-debug-main` | branch tip `36531c6`, main integration commit `4d87cdd`, `tests/test_rocm_e97_runtime_config.py` per `docs/ROCM_DEBUG_MAIN_INTEGRATION_20260621.md` | Adds E97/e97-linear ROCm runtime controls and smoke hook updates into integrated main. | Code integration evidence. It does not prove training loss, throughput, or checkpoint behavior. |
| `rocm-gdn2-mlp-port` | `docs/ROCM_GDN2_MLP_PORT_20260621.md`, commit `3c856b1`; integrated main content via `2dee6ce` | Adds structured GDN2 dependency probe and Frontier preflight script. Local py_compile passed; local no-torch environment produced structured failure. | Preflight implementation evidence. HIP compile/runtime behavior required Frontier debug proof. |
| `frontier-merge-rocm-debug-main` | `docs/ROCM_DEBUG_MAIN_INTEGRATION_20260621.md`; branch `wg/agent-34/frontier-merge-rocm-debug-main` | Confirms allocation ledger, debug recipe, ROCm audit, E97 port, GDN2 port, and smoke helper were integrated into mainline before downstream debug work. | No `sbatch`, training, benchmark, or extended allocation job launched by this task. |
| `rocm-kernel-integration-smoke` | `docs/ROCM_KERNEL_INTEGRATION_SMOKE_20260621.md`; `scripts/frontier/debug_smoke_one_node.slurm`; `scripts/frontier/submit_kernel_integration_smokes.sh` | Prepared reusable one-node smoke commands and artifact capture. Observed pre-merge/debug-slot/environment blockers. | Pre-merge blocker evidence only; no variant job was submitted. |
| `frontier-debug-benchmark-matrix` | `docs/FRONTIER_DEBUG_BENCHMARK_MATRIX_20260621.md`; `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/{env.txt,manifest.json}` | Confirmed `sbatch`, `squeue`, `sacct`, and canonical commapile source were visible, but ROCm Python env, smoke data, and GDN2 checkout were missing. No jobs submitted. | Zero node-hours consumed; this is environment blocker evidence, not negative model/kernel evidence. |
| `retry-frontier-debug` | Jobs `4880725`, `4880730`, `4880747`, `4880875`; `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`; `frontier_runs/debug/20260621/{e97-MLP,e97-linear-MLP,gdn2-MLP}/...`; `logs/frontier/debug/emender-smoke-4880*.out` | Staged runtime and smoke data enough to run debug jobs sequentially. `e97-MLP` job `4880875` passed one-rank fused split-edit Triton smoke, selected fused path, then failed before first training metric on compute-node `p50k_base.tiktoken` download. `e97-linear-MLP` job `4880730` failed chunked-E97 parity/finiteness smoke. `gdn2-MLP` job `4880747` passed external GDN2 bf16 forward/backward preflight with finite loss, then hit the same tokenizer download before first training metric. Setup job `4880725` failed before kernel execution because `pytest` was not staged. | Scheduler/runtime/GPU visibility and kernel/preflight evidence only. No training throughput, training loss, peak memory, checkpoint, resume, validation, multi-node, or DiLoCo evidence. |
| `frontier-allocation-ledger` | `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` | Records 20,000 node-hour allocation, 4,928 node-hour reserve, retry debug elapsed consumption of 0.197224 node-hours, current remaining 19,999.802776, and blocked extended candidates. | Ledger is accounting source, not approval. Extended rows remain `BLOCKED_PENDING_APPROVAL`. |
| `frontier-extended-e97-readiness` | `docs/FRONTIER_EXTENDED_E97_READINESS_20260621.md`; `scripts/frontier/e97_extended_64x24.sbatch`; commit `b09000b` | Prepared guarded 64-node x 24-hour e97 launch script with tokenizer/cache, checkpoint, resume, logging, accounting, and human approval gates. Recommendation is no-go for immediate 64x24h. | Launch package only. No extended job submitted. Requires debug first-loss, throughput, checkpoint, resume, ledger confirmation, and explicit approval. |
| `frontier-diloco-scaleout-readiness` | `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`; `scripts/frontier/diloco_scaleout_readiness.sbatch`; commit `baeeb60`; ledger updates | Designs 8/16/32/64-node DiLoCo ladder. Recommends 8-node canary only after debug gates pass and approval is recorded. | Scaleout design only. No scaleout job submitted. GPU-island, communication, optimizer, and merge behavior remain hypotheses. |

## Evidence Separation

Observed Frontier facts:

- Debug QOS accepted and ran sequential one-node jobs under the one-submitted
  job per user limit.
- The staged runtime imported ROCm PyTorch, Triton, ScheduleFree, `tiktoken`,
  NumPy, pytest, and the external GDN2 dependency path after the retry staging.
- The jobs observed MI250X GPUs, and rank-local `srun` visibility showed one
  GPU under `--gpus-per-task=1 --gpu-bind=closest`.
- `e97-MLP` passed the one-rank fused split-edit Triton smoke on MI250X.
- `gdn2-MLP` passed the external GDN2 bf16 forward/backward preflight with
  finite output, finite loss, finite input gradients, and finite parameter
  gradients.
- `e97-linear-MLP` failed chunked-E97 parity/finiteness on MI250X before
  training was launched.
- `e97-MLP` and `gdn2-MLP` both failed before first training metrics because
  compute nodes attempted to download `p50k_base.tiktoken`.
- Accepted retry debug jobs requested 2.000000 node-hours and consumed
  0.197224 elapsed node-hours.

Paper and handoff claims:

- The paper and handoff claim recurrent token-mixer architectures can occupy a
  narrow loss/perplexity band at the 1.3B class, so short LM loss alone should
  not be treated as an architecture verdict.
- The handoff reports local nonlinear state-tracking separation for nonlinear
  E97 versus e97-linear/GDN2 on selected synthetic tasks, plus negative controls
  where that separation does not appear.
- The handoff reports local DiLoCo evidence only at small island counts and
  frames Frontier 512-GCD/large-island behavior as untested.
- The handoff's 64-node token/day and scheduler details are planning inputs
  that require Frontier verification before allocation decisions.

Prior-agent interpretations:

- The orientation brief recommends treating "E97 is the launch arm" and "GDN2
  is the launch arm" as hypotheses until Frontier ROCm correctness, loss,
  throughput, and checkpoint evidence exist.
- The retry-debug report interprets the current next gate as tokenizer cache
  staging plus chunked-E97 ROCm parity work.
- The extended e97 readiness report interprets the guarded 64x24h package as
  prepared but not launchable.
- The DiLoCo readiness report interprets the 8-node canary as the smallest
  useful scaleout test after debug gates, while 16/32/64 nodes remain deferred.

Hypotheses still open:

- `e97-MLP` may reach finite first loss and useful throughput once a shared
  `p50k_base` tokenizer cache is pre-populated for compute nodes.
- `gdn2-MLP` may be the first scaleout arm if the tokenizer fix reaches
  training metrics and the fused GDN2 path remains active.
- One Frontier node may serve as one DiLoCo island of eight GCDs, but current
  evidence only verifies one-node GPU binding, not island training or merge
  behavior.
- The plain-average DiLoCo recipe may stay stable at 8+ one-node islands, but
  this must be tested by merge/loss/checkpoint evidence.

Risks:

- Compute-node network restrictions or cache path mistakes can fail jobs before
  useful training evidence, as already observed.
- The debug runtime is a repaired user-site/base environment, not a clean named
  production conda environment.
- `e97-linear-MLP` currently has a real ROCm chunked-kernel correctness blocker
  and should not be used as a readiness arm until fixed or intentionally
  disabled.
- One-node success may not predict RCCL/NCCL, filesystem, checkpoint, or
  merge-latency behavior at 8+ nodes.
- A 64x24h extended job can fit in the allocation but would burn 1,536 requested
  node-hours without the evidence currently missing.

## Next-Job Table

Baseline for the table is the ledger after retry debug:
`allocation_remaining=19,999.802776`, `reserve_held=4,928`, spendable
non-reserve balance `15,071.802776`.

| Approval status | Job ID / action | Queue | Nodes | Walltime | Expected node-hours | Remaining after requested debit | Prerequisites | Success criteria |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| `N/A - no sbatch` | `CACHE-P50K-SHARED` | none | 0 | 0 | 0 | 19,999.802776 | Login/service-node access to shared Frontier path; select one canonical `TIKTOKEN_CACHE_DIR`. | `p50k_base.tiktoken` exists in shared storage, is readable from compute-node job context, and no compute-node log attempts network download. |
| `RECOMMENDED_AFTER_CACHE` | `DEBUG-E97-POSTCACHE-1N2H` | debug | 1 | 02:00:00 | 2 | 19,997.802776 | Cache action complete; `e97-MLP` fused smoke remains enabled; smoke data paths present; ledger row added before submit. | First finite training loss, global tokens/sec, peak memory or memory sample, fused E97 guard/no eager fallback, manifest/env/train log/sacct artifacts. |
| `RECOMMENDED_AFTER_CACHE` | `DEBUG-GDN2-POSTCACHE-1N2H` | debug | 1 | 02:00:00 | 2 | 19,995.802776 if run after e97 debug | Cache action complete; GDN2 checkout and preflight path present; smoke data paths present; ledger row added before submit. | First finite training loss, global tokens/sec, peak memory or memory sample, fused GDN2 guard/no eager fallback, manifest/env/train log/sacct artifacts. |
| `RECOMMENDED_IF_ONE_ARM_PASSES` | `DEBUG-SELECTED-RESUME-1N2H` | debug | 1 | 02:00:00 | 2 | 19,993.802776 if run after both post-cache jobs | Select `e97-MLP` or `gdn2-MLP` based on finite first-loss/throughput evidence; configure checkpoint cadence short enough for debug; ledger row added. | At least one checkpoint and `latest.pt` are written, resume from that checkpoint reaches a later finite loss, and checkpoint/resume paths are recorded as artifacts. |
| `BLOCKED_PENDING_DEBUG_AND_HUMAN_APPROVAL` | `DILOCO-8x4H-CANARY` | regular or eligible short queue | 8 | 04:00:00 | 32 | 19,961.802776 if run after the three debug jobs | Cache, one-node first-loss/throughput, selected arm, checkpoint/resume evidence, ledger row, reviewed `scripts/frontier/diloco_scaleout_readiness.sbatch`, explicit WG human approval. | Launch reaches first training step on all ranks, completes at least one K=250 DiLoCo merge if configured, finite loss/grad, global tok/s above 4x one-node baseline after warmup, checkpoint/manifest/sacct artifacts, no RCCL/NCCL systemic failure. |
| `BLOCKED_NOT_RECOMMENDED_NOW` | `E97-64x24-01` | extended | 64 | 24:00:00 | 1,536 | 18,457.802776 if run after the three debug jobs; 18,463.802776 from current baseline without them | Successful e97 post-cache debug, checkpoint/resume test, optional 8-node canary, ledger confirmation, and explicit approval naming 64 nodes, 24h, and 1,536 node-hours. | First finite loss/throughput, fused E97 path, stable training, checkpoints, resume plan, scheduler accounting, and post-run artifact package. Do not submit from current evidence. |
| `BLOCKED_NOT_RECOMMENDED_NOW` | `DILOCO-64x24-01` | extended | 64 | 24:00:00 | 1,536 | 17,791.802776 after the planned 8/16/32 ladder from DiLoCo readiness | Successful 8/16/32-node scaleout ladder, updated budget, explicit 64-node human approval. | Stable launch, multiple clean merges, acceptable communication overhead, finite loss/validation, restartable checkpoint, and cost model validation. Do not submit from current evidence. |

Budget implication:

- Immediate recommended debug burn: **6 requested node-hours**.
- Recommended canary burn after debug gates and approval: **32 requested
  node-hours**.
- Total recommended next-stage burn if all gates pass: **38 requested
  node-hours**, leaving **19,961.802776** node-hours and preserving the **4,928**
  node-hour reserve.
- Extended-queue burn remains **0 approved node-hours**. Both 64-node rows are
  planning entries only.

## Follow-Up Tasks Created

The following WG tasks were created from blockers or missing evidence:

- `stage-p50k-cache`: stage shared `p50k_base` cache and rerun
  `e97-MLP`/`gdn2-MLP` one-node debug to first loss, throughput, memory, and
  artifacts.
- `verify-selected-frontier`: after post-cache debug, run a
  one-node checkpoint/resume verification for the selected arm.
- `fix-or-quarantine`: fix or quarantine the chunked e97-linear ROCm
  parity/finiteness failure before using that arm in readiness decisions.
- `review-frontier-evidence`: after post-cache debug and resume evidence,
  review the evidence and either approve or reject the 8-node x 4-hour canary
  with explicit ledger accounting.

WG depth note: direct child dependencies from this synthesis task would exceed
the configured `max_task_depth: 8`, so these follow-ups were placed at the
current graph level with the same readiness predecessors. Their descriptions
record the evidence-order dependency where WG could not represent a deeper edge.

## Final Recommendation

1. Do `CACHE-P50K-SHARED` immediately; this costs no node-hours.
2. Spend up to 6 requested debug node-hours on post-cache one-node evidence for
   `e97-MLP`, `gdn2-MLP`, and selected-arm checkpoint/resume.
3. If those debug gates pass, request human approval for exactly one 8-node x
   4-hour canary, expected burn 32 node-hours.
4. Keep all 64-node extended submissions blocked until the debug and canary
   evidence is observed, summarized, and approved with explicit ledger rows.
