# e97-rl-loop-v1 — RL first-loop prototype

**Status:** WORKSPACE PROTOTYPE (operator GO directive, tranche 2). Nothing
frozen, nothing committed to main; every artifact lives in this workspace.
Single-threaded, single-GPU. **Tranche 2 closed both directed gaps**: (1) the
on-policy SPLICE — teacher corrections are now same-transcript continuations
of the policy's own partial trajectory with only the teacher suffix
supervised (proved by `scripts/rl_splice_test.py`); (2) the sealed GYM
HOOKUP — the loop pulls sealed first-party task-lake bundles through the
task-lake validation layer and grades with the actual sealed first-party
validator program. **The smoke test PASSED** — one full cycle (serve → attempt →
grade → live-teacher-correct → verified receipts → pack → single-GPU train
step → re-serve probe) completed on ONE leased GPU; full receipt in
`STATUS.md` ("SMOKE TEST RECEIPT"), artifacts under `smoke/cycle-0001/`,
receipts under `receipts/cycle-0001/`.

## RL V2 — THE POLICY-GRADIENT CHANNEL (2026-09-27, operator RL-regime ruling)

The bank now has its first true reinforcement gradients. A SECOND train step
(`scripts/rl_policy_gradient_step.py`) sits alongside the receipts-SFT step,
flag-gated per lane (`--lane-train-step policy-gradient|sft-receipts`, default
unchanged). **Per the operator's regime ruling ALL 8 LANES run the
policy-gradient channel**, with the three load-bearing guards kept:
verified-only receipts, fail-closed NaN/KL fallback to the SFT step per lane
per cycle, and disk GC. What is implemented (full detail + measured evidence
in STATUS.md "RL V2 BUILD" through "FULL-RL BANK"):

- **The gradient path (end-to-end, verified):** a digest-verified trajectory
  batch (`scripts/rl_pg_batch.py`) of the lane's own train-split on-policy
  attempts (passes AND failures — the reward contrast; failed attempts are
  the RL signal receipts never carried), GRPO group advantages from the
  sealed-validator outcomes via the audited candidate
  (`ndm/e97_outcome_rl_candidate.py` — implemented in the E1 era, first-ever
  update taken by this build), per-token policy logprobs captured through
  the qualified logprob machinery's layout and chunked-head FP32 path, the
  candidate's clipped-surrogate + k3-KL loss summed per row, and one real
  AdamW step per cycle. Honest deviation (measured + recorded): the
  chunked-CE checkpoint replay is disabled for the grad-mode capture (the
  non-reentrant checkpoint rejects escaped hook tensors); numerics identical.
- **Importance ratio, honestly:** the bank's greedy episodes carry no
  generation-time logprob trace and generating checkpoints are reclaimed, so
  behavior logprobs cannot be recomputed; old_logp = the current weights'
  detached logprobs at each step start (ratio ≡ 1; single-use trajectories;
  first-update gradient = REINFORCE-with-group-baseline; the k3 anchor is
  the parent == the trajectory-generating policy — the textbook GRPO
  reference. The fixed E3-u256 substrate anchor was MEASURED at ~3555
  nats/token k3 distance and would dominate the loss ~10^6x — recorded as a
  negative result, kept only as an explicit screen variant).
- **Parameter screen** (`scripts/rl_pg_screen.py`, pre-registered,
  register-before-run): lr 1e-5 DISQUALIFIED (realized k3-KL 4.2e8, tokens
  crushed ~27 nats — the guard tripped); lr {2e-6, 1e-6, 5e-7} sane.
  Production: **lr 2e-6, kl_beta 0.01, kl_guard 0.056** (S5: 3× measured),
  1 update/cycle, batch = 8 most-recent episodes (the SCREENED geometry —
  measured: 21-24-episode batches at lr 2e-6 moved 0.17-0.99 KL/token and
  guard-tripped; production matches the qualified batch scale), GRPO groups
  keyed by sealed template family (lakeexp tasks are one-shot instances;
  per-task groups would be singletons).
- **Fail-closed:** any guard trip (nonfinite loss/grad, CE-consistency
  mismatch, realized-KL blowout) or no-contrast batch → NO checkpoint saved,
  the receipts-SFT step runs that cycle (`LANE_PG_FALLBACK`) — the bank
  never loses a lane. First hours: 17 PG updates (realized KL 0.0086-0.0464,
  all under the guard; grad norms ~0.13-0.19), 47 guard-trip fallbacks.
- **Evidence:** every PG update appends a digest-linked train-event receipt
  to the lane's `training/pg-stream.jsonl` (parameter set + loss + KL +
  advantage stats + parent/batch/checkpoint shas; hash chain, both
  serialization forms verified). The first completed guarded update:
  `smoke/pg-v2/qualification-final/` — checkpoint bff4adec…, receipt
  dd6495fd….
- Unit validation 52/52 (10 v2 tests); `run_bank.sh` preflight requires 52
  and supports `LANES_ON_POLICY_GRADIENT` for future launches.

This v1 proves **the CYCLE** on a handful of tasks. It does NOT include the full
5,181-task gym, the full teacher role-play episode design, or DiLoCo multi-worker
sync — those come after the cycle is proven. Read the gap list below for exactly
what is stubbed vs real in each stage.

## The loop (one program: `scripts/rl_loop_driver.py`)

```
            ┌──────────────────────────────────────────────────────┐
            │  (a) SERVE: load current policy checkpoint            │
            │      (E3-u256 3770621c…, agent-server engine path,   │
            │       weight_mode='saved')                             │
            ▼                                                      │
  queue/pending ──claim──► (b) ATTEMPT: one real Pi-native tool      │
            │               episode per task (bridge + real pi      │
            │               child executes the 11 frozen tools)     │
            ▼                                                      │
      (c) GRADE: mechanical run_check verifier (pilot machinery)    │
            │ pass                                                 │
            ├────────────► (e) QUEUE: verified receipt ────► digest │
            │ fail                     (on-policy-success)   chain   │
            ▼                                                jsonl  │
      (d) TEACHER-CORRECT: GLM-5.3 continues the SAME TRANSCRIPT    │
          from the policy's own partial trajectory (byte-exact      │
          prefix replay into the correction bridge; live lunaroute  │
          transport; ONLY the teacher suffix is supervised)         │
            │ grade again — teacher output must ALSO pass          │
            ▼                                                      │
      (e) QUEUE: verified receipt (teacher-corrected)  ─────────────┤
                                                                   │
  receipts ≥ threshold ──► (f) TRAIN: receipts -> masked-SFT         │
                            authority -> boundary-aware packs       │
                            -> single-GPU masked-SFT step           │
                            (chunk-2048 geometry)                   │
                            ──► new checkpoint                      │
                                                                   │
            (g) RE-SERVE: probe the new checkpoint; loop ◄──────────┘
```

The loop idles when `queue/pending` is empty — the work-driven contract. There is
**no database anywhere**: per the
[no-database amendment](/home/erikg/emender/docs/RESILIENT_DILOCO_COMPUTE_POOL.md)
(2026-07-25) of the resilient DiLoCo design, coordination is durable
filesystem evidence only (atomic-rename queue claims, digest-linked receipt
streams, immutable 0400 receipt copies). `state/current_checkpoint.json` is
operator convenience, never authority — every receipt binds the exact
checkpoint sha256 it was produced against.

### Queue storage decision (operator steering, sibling-review concurrence)

`ndm/e97_task_leases.py` is SQLite-backed and **MUST NOT be the queue's
storage** — exactly the conflict the no-database amendment forbids and the
sibling review flagged. The v1 queue keeps the lease *semantics* on a pure
filesystem basis:

- **claim** = one atomic `os.rename` pending → active (single filesystem, no
  lock, no database; loser of the rename simply tries the next task);
- **deadline** = the episode budget (`episode_seconds` / turn and token
  budgets enforced by the bridge and the generate contract) — a crashed
  claimant leaves the task in `active/`, and `status --sweep-orphans` (or the
  driver's startup sweep) returns it to `pending/` as crash recovery from
  durable state;
- **receipt** = the verified training receipt stream (below) replaces the
  completion row; `queue/done/` holds the retired task file as evidence.

### Train-step decision (operator steering, RL optimizer audit)

The RL optimizer audit found the GRPO-style candidate machinery has never
taken an update (`rl_optimizer_ready: false`; end-to-end gradients, DDP
normalization, and optimizer integration all remaining). V1's train step
therefore uses the **proven** path: verified receipts → cohort-format packs
(the prep-builder machinery's authority/pack layout) → a masked-SFT training
step with the chunk-2048 config on ONE GPU. The policy-gradient channel is v2,
a separate lane once the cycle is proven. Every stage of the first loop runs
on machinery that has actually run before.

## Layout

```
scripts/
  rl_common.py            workspace plumbing: queue (rename-claim), state, digests
  rl_seed_tasks.py        freeze deterministic seed tasks (pilot terminal_case)
  rl_loop_driver.py       the cycle driver: init | collect | probe | status | verify
  rl_receipts.py          receipt format, digest chain, verification (re-encode)
  rl_teacher_fixture.py   authored deterministic teacher STUB (explicit, not GLM-5.3)
  rl_build_pack.py        receipts -> tulu3-masked-sft-v1 authority -> packs
  rl_train_step.py        single-GPU masked-SFT step (canonical machinery, world=1)
  rl_gym_tasks.py         GAP #2: pull sealed task-lake gym tasks into the queue
  rl_splice_test.py       GAP #1 acceptance test: one spliced cycle (CPU, real
                         Pi transport, real sealed validator, teacher-only
                         target-span proof)
  rl_cpu_dryrun.py        CPU integration dry-run (no policy, no GPU)
  smoke.sh                ONE full cycle on ONE leased GPU
  run_cycles.sh           multi-cycle continuous run on the sealed gym (GPU)
queue/{pending,active,done}/   atomic-rename task queue
receipts/stream.jsonl          digest-linked receipt chain (the durable evidence)
receipts/cycle-NNNN/*.json      immutable per-receipt copies
episodes/cycle-NNNN/<task>/{attempt,correction}/   full episode evidence
packs/cycle-NNNN/{authority,packs}/                 receipts->pack outputs
training/cycle-NNNN/                               train-step outputs
smoke/                                             smoke-test artifacts + receipts
```

## Receipt format (digest-linked, no database)

See `RECEIPT_SPEC.md` for the full normative fields. Shape:

- `receipt_sha256` — SHA-256 over the canonical JSON of the receipt body
  (this field excluded): content-binding, tamper-evident.
- `prev_receipt_sha256` — digest of the previous receipt in
  `receipts/stream.jsonl`; the stream is a hash chain.
- `kind` — `on-policy-success` (policy episode passed the mechanical grader)
  or `teacher-corrected` (teacher continuation passed the same grader;
  `attempt_link` carries the failed on-policy attempt's evidence digests).
- `episode` — full Pi-native episode text + per-turn generation token IDs +
  `supervise_from` (supervision boundary) + verified token/target counts.
- `grade` — the mechanical grade receipt (per-check pass/fail + trace).
- `teacher` — teacher identity (live model or fixture-stub marker).
- `policy_checkpoint` — exact checkpoint path + sha256 the policy served.

Verified-only discipline: `verify` (stream walk) re-encodes every episode with
the same `encode_candidate` machinery the pack builder uses and fails closed on
any mismatch, un-passed grade, or broken chain link.

## Seed tasks and the sealed gym tranche

Seed tasks: deterministic terminal-bench-STYLE tasks authored by the teacher
pilot's `terminal_case` generator (seeded synthetic names/values, no benchmark
instances), graded by the pilot's mechanical workspace checks.

Sealed gym tranche (tranche 2, `scripts/rl_gym_tasks.py`): tasks are PULLED
from the admitted sealed first-party task-lake authority
(`/mnt/nvme2n1/erikg/task_lake/e97-firstparty-cpu-phase-bc-v1-admitted`) through
the task-lake validation layer — registry + collection validation
(`ndm.e97_task_lake`), admission-receipt digest binding, per-bundle private
validator spec digest binding (canonical JSON), sealed program sha binding,
fixture archive sha + safe extraction with expanded-tree-digest verification —
and frozen into the filesystem queue with per-cycle task ids (the loop
re-attempts the sealed tasks each cycle against the CURRENT policy
checkpoint). No SQLite anywhere. Development-split bundles are measured but
never receipted (`receipt_eligible=false`; sealed train/development split
isolation). The gym-recovery clusters were NOT adopted into main (no adoption
commits; the recovery review recommended them), but every piece this tranche
needs is already in main from commit 2002eded, so the recovery worktree is
not required.

## What is REAL vs STUBBED in v1 (the gap list)

| Stage | Real | Stubbed / deviated |
|---|---|---|
| SERVE | full agent-server **engine** path (`ndm.e97_agent_server.TorchE97AgentEngine`, `load_e97_checkpoint`, sha-verified, weight_mode='saved') | the HTTP surface of `serve_e97_agent_openai.py` is NOT used: its service layer parses the dense `Action:/Arguments:/Final:` protocol and rejects the five-line Pi-native frames the bridge/teacher/pack machinery uses. Rollouts drive the engine in-process exactly like `eval_e97_native_execution.py` did for the 96 episodes. |
| ATTEMPT | real Pi tool transport (real `pi` child executes the eleven frozen tools in a real per-task workspace); strict canonical five-line frame validation; greedy on-policy decoding | episode budgets retuned for single-GPU decode speed (12 turns / 1024 tokens per turn / 1500 s) vs the pilot teacher panel; no policy sampling (temperature 0) — sampling/log-prob receipt fields are the RL-leg follow-up |
| GRADE | seed tasks: real mechanical grader (pilot `run_check`). Sealed gym tasks (tranche 2): the ACTUAL sealed first-party validator program (sha-verified against every bundle's private spec `program_sha256`) executed in focused + regression modes over a faithful dense-format terminal projection of the real Pi-native episode; the task-lake validation layer (`ndm/e97_task_lake.py` registry + collection validation) gates task admission at pull time | honest deviations, recorded in every gym grade receipt: the sealed validator is dense-terminal-bound so the loop derives a format-only projection of real actions/observations; the spec's hermetic pinned interpreter is absent on this host (the byte-verified sealed program runs under the project interpreter, recorded per grade); sealed `limits.seconds`/`completion_tokens` are dense-controller accounting replaced by the Pi-native lane's own budgets (`limits.turns`=12 is honored across prefix+suffix); the frozen 11-tool surface has no listing tool so the two opaque discovery tasks cannot pass the sealed minefield (list_files/read) in this lane — they fail honestly. No guide-teacher qualitative review (deliberately out of scope: "mechanically graded outcomes only") |
| TEACHER-CORRECT | **live GLM-5.3-flash-background** through lunaroute using the pilot's Pi-transport frame contract; the correction bridge REPLAYS the policy's retained partial trajectory into its own canonical transcript byte-exact (`replay_prefix`; fail-closed prefix check) and the teacher generates given that exact prefix as conversation context — the true on-policy corrective demonstration. Receipts are same-transcript records: policy prefix frames are masked context, ONLY the teacher suffix frames are supervised targets (`supervise_from` partial-target masking, the canonical loopbreak/pointerchase family machinery) | trailing policy finish frames / unresolved trailing tool calls are dropped from the replayed prefix (the codec forbids content after finish; an unanswered call has no observation) and remain immutable attempt evidence; the authored deterministic fixture (`rl_teacher_fixture.py`) is the explicit test/fallback stub and is NOT GLM-5.3 output |
| QUEUE | digest-linked JSONL hash chain; immutable 0400 copies; fail-closed stream verification with full re-encode; NO SQLite anywhere | receipt→task completion is single-consumer v1 (one loop process per GPU); no multi-worker receipt attribution yet |
| TRAIN | receipts re-encoded by the pilot's `encode_candidate`; authority in the canonical `emender-e97-tulu3-masked-sft-v1` layout; packs by the canonical `build_e97_sft_packs.py` (boundary-aware v2, epoch-permutation); trainer reuses the canonical loader, precision policy (FP32 logits + 2048-token chunked CE), the qualified `--projection-chunk-size 2048` scan geometry, CPU-offloaded AdamWScheduleFree, and the canonical checkpoint schema | **the canonical `train_e97_4b_pi_sft.py` hard-requires 8/64-rank worlds in 8-rank islands and cannot run on ONE GPU**; `rl_train_step.py` runs the same objective (target-token-normalized masked CE) as a world-of-one single-process step, no DDP/DiLoCo merge. The production cycle must return to the canonical 8-rank trainer (or per-GPU DiLoCo workers) after the prototype |
| RE-SERVE | probe reloads the trained checkpoint through the same engine path and emits one policy frame (validated) | the probe is one frame, not a full re-attempt run; next-cycle re-attempts are the loop's continuing stage |
| Receipts→training admission | authority manifest records purpose, chain head, per-record receipt digests | the full proposal/admission/audit machinery that gated E1/E2/E3 segments is out of scope; the prototype authority is workspace-local and claims no admission |
| DiLoCo / multi-GPU | — | entirely out of v1 scope (single-GPU prototype by directive) |

## Running

```bash
export PYTHONPATH=/home/erikg/emender:$PWD/scripts
PY=/home/erikg/emender/.venv/bin/python

# CPU integration dry-run (fixture teacher, real Pi transport, receipts, packs)
$PY scripts/rl_cpu_dryrun.py --scratch smoke/cpu-dryrun --indices 0 1 2 \
    --teacher fixture --build-pack

# Full smoke: one cycle on ONE leased GPU (serve->attempt->grade->correct->
# receipt->pack->train-step->re-serve probe)
bash scripts/smoke.sh
```

Environment bindings: the Pi runtime + frozen tool manifest
(`configs/pi/e97-active-tool-surface-v1.json`, sha
`55421905438806223414d96a4e64f1a6ae64772afdb1fbc67fbbfb77e8d5d90`), lunaroute
credentials (`~/.pi/agent/auth.json`, `lunaroute` key), starting checkpoint
E3-u256 `3770621c…` and its pretraining `args.json`.

## Measured results

### Tranche 2: the first real continuous run (cycles 2-6, sealed gym tasks)

Five consecutive cycles, one leased GPU, live GLM-5.3 splice corrections,
sealed-validator grading. Per cycle: 4 sealed task attempts (2 train, 2
development split), receipts only from the train split:

| cycle | policy pass | teacher pass | receipts | targets | step loss |
|---|---|---|---|---|---|
| 2 | 0/4 | 2/4 | 1 teacher-corrected, SPLICED (supervise_from=1) | 75 | 1.4178 |
| 3 | 0/4 | 2/4 | 1 teacher-corrected, SPLICED (supervise_from=1) | 90 | 0.9512 |
| 4 | 1/4 | 1/4 | 1 on-policy-success | 149 | 0.1214 |
| 5 | 1/4 | 1/4 | 1 on-policy-success | 142 | 0.0159 |
| 6 | 1/4 | 1/4 | 1 on-policy-success | 121 | 0.0044 |

The train direct-read task failed on-policy in cycles 2-3 (premature wrong
finish; the splice dropped the trailing finish, the teacher continued from
the policy's own read observation and finished with the correct token), and
passed ON-POLICY from cycle 4 onward — the loop's first visible on-policy
learning progression. The two opaque discovery tasks fail every cycle: the
frozen 11-tool surface has no listing tool, so the sealed minefield
(list_files/read) is unsatisfiable for them in this lane (honest recorded
failures). The dev-split tasks are measured but never receipted. Receipt
chain head after cycle 6 (8 receipts, all re-encode verified, spliced target
spans proven teacher-only):
`0a42683cc9c2127e4088fdd8ea354b16a8e07646ebe62fd3dfb577a9ce43a5b6`.
Checkpoint lineage: 1e059464… -> 9a3b49ff… -> c647fc60… -> 9a0a241f… ->
1cf9ecfb… -> bec77602… (current).

### Tranche 3: the 8-GPU DiLoCo-synced bank — pilot + merge shakeout (measured)

Bank machinery (rl_bank.py, rl_bank_lane.py, rl_bank_merge.py,
rl_bank_coordinator.py, run_bank.sh): CPU unit validation 33/33 (28 original
+ 5 merge-adoption/window-reset tests), all on the real admitted lake and
real lease machinery.

**Two-lane GPU pilot (bank-pilot2/, 16 min, K=32):** 6 pool rounds, 24
retired claims, 6 verified receipts (2 teacher-corrected spliced +
4 on-policy-success), 46 teacher calls, 6 train updates (3/lane), step
losses collapsing per lineage (1.58 -> 0.002 / 1.70 -> 0.007), all probes
valid, clean STOP exit, every GPU lease released. NO merge fired — expected
physics (see honest limits below).

**Merge shakeout (bank-merge-shakeout/, 2 lanes, K=2, pilot-scale ONLY):**
the live merge path proven end-to-end — TWO published merges (trigger ->
weighted soup by window pass-rate -> R07 atomic publish -> sha-verified
lane adoption with the window counters RESET -> post-merge training ->
second merge ranked on the RESET windows). Merge receipts digest-linked;
9 rounds / 36 retired claims / 9 receipts / 71 teacher calls / 9 updates;
LANE_IDLE (yield-when-idle) observed when the pool drained between rounds.
Production geometry stays K=32.

Honest limits of the bank as measured (the operator's gym-scaling
evidence):
- The gym authority is the 4-bundle admitted lake. The pool recycles those
  4 sealed bundles round-scoped; only ONE task per round can produce a
  receipt (train direct-read; the opaque train bundle fails honestly in
  this lane, dev-split tasks are measured but never receipted), so
  bank-wide update production is ~1 update/pool-round ~= 22/hour (measured
  on both pilots). Identical re-attempts on the passed bundle produce
  diminishing signal — step loss on the mastered direct-read task
  collapses toward 0 within a few updates per lineage; the round refresh
  keeps re-attempting it anyway.
- At the locked K=32 per LANE the first production merge is therefore
  HOURS away (a lane must bank +32 updates against ~22 updates/hour
  bank-wide shared across lanes), and lane idle time between claims
  engages yield-when-idle (observed: LANE_IDLE, leases released — the bank
  does not squat GPUs while the pool is drained).
- In the 4-bundle regime the weighted soup DEGENERATES to winner-take-all
  in practice: at most one lane can hold a positive window pass rate at a
  time. The soup machinery is proven (merge math unit-tested to exact fp32
  reference; live wiring shaken out at K=2); a multi-lane-weighted merge
  needs a bigger sealed gym to be meaningful.
- RECLAIM-LOG discipline (supervisor-approved retention): per-cycle lane
  checkpoints are superseded lineage weights (CACHE, not evidence — every
  receipt and merge receipt already binds the exact sha). After a lineage
  advance a lane reclaims its own superseded checkpoints once past a
  15-minute grace window; fail-closed refusals keep every merged
  checkpoint, every lane's current lineage, anything outside the lane's
  training tree, and any identity-drifted file. Every deletion is logged
  (path + sha + reason) in bank/reclaim-log.jsonl. Steady-state disk is
  thereby capped at roughly 8 lanes x ~2 checkpoints + merged checkpoints
  ~= 500-600GB. A low-disk fail-closed guard (the run.sh 100GiB-guard
  precedent) pauses lane cycles (status disk-low, no claim) and defers
  coordinator merges (trigger stays armed) instead of dying on ENOSPC.
  Unit validation covers every refusal path (42/42).

### Measured smoke results (cycle 1)

- E3-u256 policy: 100% canonical five-line frame compliance, 100% episode
  completion, 0/3 task success — the loop's honest starting point.
- Live GLM-5.3 teacher-correct: 3/3 verified continuations (including one
  recovery from two real mid-episode tool errors).
- Receipt stream: 3 receipts / 2,769 assistant target tokens, chain head
  `ff3267974b1cd921…`, all re-encode verified.
- Train step: 1 update on 9,657 tokens / 1,179 targets, loss 2.1819,
  20.1 GB HBM peak on ONE GPU; new checkpoint
  `1e0594646189544609f7dcff106acbd8431eac8b98721a015356891425da1e12`.
- Re-serve probe: the updated checkpoint emits a valid canonical frame
  (competence unchanged after one 1,179-target update, as expected).

## Honest scope statement

Tranche 2 proves the loop can run CONTINUOUSLY on sealed gym tasks: the
policy attempts sealed task-lake bundles in its own Pi-native protocol, is
graded by the actual sealed first-party validator, is corrected by live
GLM-5.3 as a same-transcript splice of its OWN partial trajectory (only the
teacher suffix supervised), takes a real optimizer step per cycle, and —
measured, not claimed — passes the same sealed task on-policy after two
spliced correction updates. The sealed-lane deviations (dense-terminal
projection grading, host interpreter for the pinned-validator program,
Pi-native budget accounting, the unsolvable-in-this-lane opaque tasks) are
recorded in every grade receipt and in STATUS.md's residual gap list.

v1 proved the cycle itself: a policy checkpoint can be served, can attempt
mechanically graded tasks through real tool execution, can be
teacher-corrected by live GLM-5.3 under verified-only discipline, can have
those verified episodes converted to the canonical pack format, can take a
real optimizer step on one GPU, and can be re-served from the resulting
checkpoint. Neither tranche claims: gym scale (the admitted sealed lake is 4
bundles), multi-worker DiLoCo sync, role-play episode design, RL-policy
sampling/log-likelihood receipts, or any admission/promotion. The E3-u256
checkpoint's Pi-native five-line frame compliance is itself a measured
output of the smoke test, not an assumption.
