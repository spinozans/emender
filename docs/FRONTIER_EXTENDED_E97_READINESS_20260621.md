# Frontier Extended e97 Launch Readiness - 2026-06-21

Task: `frontier-extended-e97-readiness`

## Recommendation

**No-go for immediate 64-node x 24-hour extended submission.**

The launch package is prepared as a guarded artifact, but the present debug
evidence does not justify spending 1,536 node-hours yet. The selected variant is
`e97-MLP` because it is the only e97-family variant in the retry matrix whose
Frontier ROCm kernel smoke passed. However, `e97-MLP` has not reached a first
training loss, throughput line, checkpoint write, or resume check on Frontier.
`e97-linear-MLP` is explicitly quarantined from extended readiness decisions by
`docs/FRONTIER_E97_LINEAR_ROCM_QUARANTINE_20260621.md` because debug job
`4880730` failed the chunked-E97 ROCm parity/finiteness smoke before training.

Recommended next spend before any extended job:

1. Fix and pre-populate the shared `p50k_base` tiktoken cache.
2. Rerun one-node `e97-MLP` debug until it records first training loss,
   throughput, peak memory, checkpoint creation, and a short resume check.
3. If the one-node run passes, run an 8-node x 4-hour canary before 64x24h.
   This costs 32 requested node-hours and preserves 1,504 node-hours versus a
   failed direct 64x24h launch.

The prepared 64x24h script is:

```text
scripts/frontier/e97_extended_64x24.sbatch
```

It fails fast unless both a WG human approval record and the explicit
`CONFIRM_EXTENDED_SUBMISSION=I_ACCEPT_1536_NODE_HOURS` acknowledgement are
provided at submission time.

## Observed Evidence

Evidence comes from committed WG artifacts, not prior claims alone:

- `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`
- `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md`
- `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z/artifacts/manifest.json`
- `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z/summaries/summary.md`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/artifacts/manifest.json`
- `frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z/artifacts/manifest.json`

Observed matrix outcomes:

| Variant | Debug job | Result relevant to extended readiness |
| --- | ---: | --- |
| `e97-MLP` | 4880875 | One-rank fused split-edit Triton kernel smoke passed. Eight-rank training launched the fused E97 path, then all ranks failed before first metric while `tiktoken` tried to download `p50k_base.tiktoken` from compute nodes. No throughput, memory peak, training loss, checkpoint, or resume evidence was produced. |
| `e97-linear-MLP` | 4880730 | Chunked-E97 ROCm smoke failed before training. Reported parity/finiteness failures include very large forward errors and non-finite strong-decay/g-drift cases. This variant is not eligible for extended readiness. |
| `gdn2-MLP` | 4880747 | External GDN2 import and bf16 forward/backward preflight passed with finite one-rank loss. Eight-rank training hit the same tokenizer download failure before first training metric. This is useful contrast evidence but is not e97 launch evidence. |

## e97-linear Quarantine

`e97-linear-MLP` is not a selectable extended launch arm in the prepared
readiness package. It may not be used as the `E97-64x24-01` variant, as a
fallback if `e97-MLP` remains blocked, or as supporting evidence for any
e97-family extended allocation until the quarantine report's exit criteria are
met.

The active quarantine source is
`docs/FRONTIER_E97_LINEAR_ROCM_QUARANTINE_20260621.md`. The cited blocker is
job `4880730`, which ran under debug QOS on 1 node with 00:30:00 requested
walltime and failed all seven selected `tests/test_e97_chunked.py` ROCm smoke
checks before training.

Observed scheduler/runtime facts:

- Frontier debug jobs were accepted and ran sequentially under the debug QOS.
- The staged runtime imported ROCm PyTorch, Triton, schedulefree, tiktoken,
  numpy, and pytest from the Frontier miniforge/user-site environment.
- The one-node jobs saw eight MI250X GPUs, and the rank-local `srun` visibility
  probe saw one GPU under `--gpus-per-task=1 --gpu-bind=closest`.
- The retry consumed 0.197224 elapsed node-hours across four accepted debug
  jobs, with 2.000000 requested node-hours.

## Interpretation Boundary

Observed:

- `e97-MLP` kernel smoke passed on one MI250X rank.
- The selected training command reached the fused E97 runtime path.
- Training stopped before first optimizer metric because tokenizer cache
  staging was incomplete for compute nodes.

Hypotheses:

- `e97-MLP` may be viable once `p50k_base` is pre-cached in a shared path.
- One-node debug success after the cache fix is likely cheaper and more
  informative than submitting directly to the extended queue.
- An 8-node canary is the smallest useful scaleout check for 64-node launch
  mechanics because it exercises multi-node RCCL/NCCL, logging, and checkpoint
  I/O without risking a full 1,536 node-hour debit.

Risks:

- The 64-node command may expose RCCL/NCCL or filesystem behavior not covered
  by one-node debug.
- The staged Python runtime is a repaired user-site/base environment, not a
  clean named production conda environment.
- `train.py` checkpointing is rank-0 only under distributed training; this
  avoids checkpoint races, but the resume path has not yet been tested on
  Frontier for `e97-MLP`.
- Without a fixed held-out tensor, held-out curve/final eval must be disabled
  or supplied explicitly; the launch script accepts `HELDOUT_TENSOR` but does
  not invent one.

Prior-agent interpretations retained as interpretations:

- The retry task recommended fixing the tokenizer cache and rerunning
  `e97-MLP`/`gdn2-MLP` to first loss and throughput before extended readiness.
- The ledger reserved four 64x24h e97/e97-linear production slots, but marked
  `E97-64x24-01` as `BLOCKED_PENDING_APPROVAL`, not approved.

## Budget Impact

Allocation source: `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md`.

| Item | Node-hours |
| --- | ---: |
| Total Frontier allocation through 2026-09-01 | 20,000.000000 |
| Reserve held | 4,928.000000 |
| Debug retry elapsed consumption recorded in ledger | 0.197224 |
| Current allocation remaining after retry | 19,999.802776 |
| Spendable while preserving reserve | 15,071.802776 |

Candidate and canary costs:

| Shape | Queue | Nodes | Walltime | Requested node-hours | Remaining after, from current 19,999.802776 | Reserve status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| One-node debug rerun | debug | 1 | 2h | 2 | 19,997.802776 | Reserve intact |
| 8-node canary | batch/regular or smallest approved scaleout queue | 8 | 4h | 32 | 19,967.802776 | Reserve intact |
| Direct `E97-64x24-01` | extended | 64 | 24h | 1,536 | 18,463.802776 | Reserve intact but 10.2% of spendable non-reserve balance used |

The ledger's planned candidate row uses a conservative post-debug baseline of
19,000 remaining, so it lists `E97-64x24-01` as:

```text
requested_node_hours=1,536
allocation_remaining_before=19,000
reserve_held=4,928
allocation_remaining_after=17,464
approval_status=BLOCKED_PENDING_APPROVAL
```

Using the newer elapsed accounting from the retry row, a direct submission would
leave 18,463.802776 node-hours before any scheduler charge adjustments. The
readiness recommendation remains no-go because the gating evidence is missing,
not because the allocation cannot fit the job.

## Prepared 64x24h Launch Package

Do not run this until all human checklist items below are complete.

```bash
mkdir -p logs/frontier/extended

# Pre-populate from a login/service node where outbound access is allowed.
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
mkdir -p "$TIKTOKEN_CACHE_DIR"
python - <<'PY'
import os
import tiktoken
os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/lustre/orion/bif148/proj-shared/tiktoken_cache")
enc = tiktoken.get_encoding("p50k_base")
print(f"cached p50k_base vocab={enc.n_vocab} at {os.environ['TIKTOKEN_CACHE_DIR']}")
PY

# Submission gate. Replace HUMAN_APPROVAL_RECORD with the exact WG message or
# committed approval artifact. This command requests 64 * 24 = 1,536 node-hours.
HUMAN_APPROVAL_RECORD='<WG approval message/artifact id>' \
CONFIRM_EXTENDED_SUBMISSION='I_ACCEPT_1536_NODE_HOURS' \
EMENDER_CONDA_ENV=base \
TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache \
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt \
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/extended \
sbatch scripts/frontier/e97_extended_64x24.sbatch
```

Default 64x24h model/run configuration in the script:

| Field | Value |
| --- | --- |
| Variant | `e97-MLP` |
| Nodes/ranks | 64 nodes, 8 ranks per node, 512 ranks total |
| GPU binding | `--gpus-per-task=1 --gpu-bind=closest` |
| Distributed env | Script maps `SLURM_PROCID`, `SLURM_NTASKS`, and `SLURM_LOCALID` to `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` before `train.py` starts |
| Data | `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt` |
| Tokenizer | `p50k_base`, with required shared `TIKTOKEN_CACHE_DIR` |
| Precision/kernel | bf16, `--use_triton 1`, fused split-edit E97 |
| Geometry | `--n_state 32 --n_heads 323 --dim 1536 --depth 10 --mlp_ratio 1.5` |
| Optimizer | schedulefree, `--lr 9e-4`, `--seed 42` |
| Context/batch | `--chunk_size 2048 --batch_size 1` per rank |
| Walltime budget | `--train_minutes 1410` with 30 minutes held for launch/checkpoint drain |

## Checkpoint, Resume, and Log Plan

Checkpoint cadence:

- Default `SAVE_EVERY=500` optimizer steps.
- Default `KEEP_CHECKPOINTS=12`.
- `train.py` writes checkpoints from rank 0 only under distributed training,
  named `checkpoint_step_<step>_loss_<loss>.pt`, plus `latest.pt`.
- With `batch_size=1`, `chunk_size=2048`, and 512 ranks, each optimizer step
  covers `1 * (2048 + 1) * 512 = 1,049,088` tokens, so the default checkpoint
  interval is about 524.544 million tokens.

Resume behavior:

- To resume, submit the same script with `RESUME_CHECKPOINT=/path/to/latest.pt`
  or a concrete `checkpoint_step_*.pt`.
- The script validates that `RESUME_CHECKPOINT` is readable before launch and
  passes `--resume` to `train.py`.
- A resume approval must debit the requested node-hours separately in the
  ledger before submission.
- Before any 64x24h approval, require a debug or canary resume test that writes
  at least one checkpoint, resumes from it, and logs a subsequent finite loss.

Logs and artifacts:

- Scheduler stdout/stderr: `logs/frontier/extended/%x-%j.out` and `.err`.
- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/extended/<date>/<run-id>-<job>-<stamp>/`.
- Environment capture: `<run_root>/artifacts/env.txt`.
- Manifest: `<run_root>/artifacts/manifest.json`.
- Training log: `<run_root>/logs/train.log`.
- Optional held-out curve: `<run_root>/heldout_curve.csv` if `HELDOUT_TENSOR`
  is supplied.
- Post-run accounting must collect `sacct -j <jobid> --format=JobID,State,Elapsed,AllocNodes,AllocTRES,ExitCode`.

Monitoring cadence:

- First 15 minutes: watch stdout/stderr for module import, tokenizer cache,
  fused guard, rank binding, and first training step.
- Every 30 minutes for the first 2 hours: check `squeue`, `tail` the train log,
  confirm finite loss and no repeated non-finite skip messages.
- Every 2 hours after that: record latest step/loss/tokens, checkpoint age,
  and filesystem free space under `OUTPUT_ROOT`.
- At 23 hours: confirm a recent checkpoint exists and decide whether to let the
  job drain naturally or cancel after checkpoint if loss is non-finite or logs
  are stalled.

Failure criteria:

- Any tokenizer network/download attempt on compute nodes.
- Missing fused E97 guard or fallback away from `--use_triton 1`.
- No first loss line within the expected compile/autotune window.
- Non-finite loss that stops training, or repeated non-finite gradient skips
  without recovery.
- No checkpoint after `SAVE_EVERY` steps.
- Filesystem/quota errors under the run root.
- RCCL/NCCL process-group failure or rank divergence.

Success criteria:

- First finite training loss and throughput lines are present.
- Fused E97 guard appears for ranks and no eager fallback is logged.
- At least one checkpoint is written and `latest.pt` points to it.
- If this is a canary/debug gate, a resume from that checkpoint reaches a later
  finite loss.
- Post-run manifest, env, train log, scheduler stdout/stderr, checkpoint list,
  and `sacct` accounting are recorded as WG artifacts.

## Human Approval Checklist

Before any extended-queue submission, a human must confirm all items:

- [ ] WG contains an explicit approval message or committed approval artifact
      naming `E97-64x24-01`, `64` nodes, `24h`, and `1,536` requested
      node-hours.
- [ ] `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` is updated or confirmed
      current with remaining allocation before/after and reserve held.
- [ ] The approval cites successful post-cache debug evidence for `e97-MLP`:
      first finite loss, throughput, memory, checkpoint creation, and log paths.
- [ ] A checkpoint resume test has passed in debug or canary scale.
- [ ] The shared `TIKTOKEN_CACHE_DIR` is populated and verified without
      compute-node download.
- [ ] Data, output, and checkpoint directories are readable/writable from
      Frontier compute nodes.
- [ ] The exact `sbatch` command and environment variables are recorded in WG.
- [ ] `HUMAN_APPROVAL_RECORD` is set to the WG approval message/artifact id.
- [ ] `CONFIRM_EXTENDED_SUBMISSION=I_ACCEPT_1536_NODE_HOURS` is set only after
      the above checks pass.

## Current Go/No-Go

Current status: **NO-GO for extended 64x24h submission**.

Allowed immediate work: tokenizer-cache fix, one-node debug rerun, checkpoint
and resume validation, and then an 8-node x 4-hour canary if one-node evidence
passes.

Blocked work: `sbatch scripts/frontier/e97_extended_64x24.sbatch` in the
extended queue without the human approval record and explicit node-hour
acknowledgement.
