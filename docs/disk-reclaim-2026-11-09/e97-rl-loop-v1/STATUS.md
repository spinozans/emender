# STATUS — e97-rl-loop-v1

Single-GPU continuous RL cycle prototype (operator GO directive).
Workspace-only. No proposal frozen, nothing committed, no main-branch edits.

## Log (append at every step)

- 2026-09-27 (tranche 2 start) GAP #1 + GAP #2 implementation. Grounding:
  encode_candidate's supervise_from partial-target masking confirmed against the
  canonical multi-turn precedent (loopbreak/pointerchase/extracterror families
  supervise from turn 1-2 with error-result prefixes; scripts/build_e97_pi_native_
  curriculum.py:388-399). Gym adoption check: NO gym-recovery adoption commits in
  main's git log (the recovery review workspace e97-gym-machinery-recovery-075bead0
  recommends adoption but none landed); however every piece this tranche needs IS
  in main from commit 2002eded "Add sealed E97 first-party task authority":
  ndm/e97_task_lake.py (validation layer, byte-identical to the recovery worktree
  copy), ndm/e97_first_party_read_observe.safe_extract_fixture_archive, and the
  sealed validator program scripts/e97_first_party_validator.py whose bytes hash
  to the program_sha256 pinned in every admitted bundle's private spec
  (344a1209ed32e44f58ce6b2be2723de9f60cc9f137dfd169c8480e49a54efaf3) -- so the
  recovery worktree/PYTHONPATH discipline is NOT needed for this tranche.
- Sealed lake in use: /mnt/nvme2n1/erikg/task_lake/e97-firstparty-cpu-phase-bc-v1-
  admitted (registry sha 380539d5..., tasks sha a745e8b0..., admission receipt
  verified; validate_source_registry + validate_task_collection pass on main
  machinery; 4 bundles: 2 train [direct-read, opaque-read], 2 development).
- GAP #1 implemented (same-transcript splice): rl_loop_driver.replay_prefix +
  retained_prefix replay the policy's own partial trajectory into the
  correction bridge's PiNativeEpisode BYTE-EXACT (native_turn(source_message)
  reproduces the policy's canonical frame; accept_generated_turn re-validates;
  fail-closed startswith check against the policy episode text). The teacher
  (live GLM-5.3 via the pilot Pi-transport frame contract) generates given the
  spliced transcript as conversation context; receipts carry merged
  generations with supervise_from = retained policy prefix frames, so ONLY the
  teacher suffix tokens are masked-SFT targets. Trailing finish frames and
  trailing unresolved tool calls are dropped from the replay (codec forbids
  content after finish; unresolved calls have no observation) and remain
  immutable attempt evidence via attempt_link. Episode records now publish
  source_messages (owner-side splice evidence).
- GAP #2 implemented (sealed gym hookup): scripts/rl_gym_tasks.py pulls tasks
  from the admitted lake through the task-lake validation layer (registry +
  collection + admission-receipt digests + per-bundle validator spec_digest
  canonical-JSON binding + program sha + fixture archive sha + safe extraction
  with tree-digest verification) and freezes them into the filesystem queue
  (no SQLite). Grading executes the ACTUAL sealed first-party validator program
  (focused + regression argv) over a faithful dense-format terminal projection
  of the real Pi-native episode (grade receipt records projection + program +
  spec digests). Honest deviations documented: (a) the sealed validator is
  dense-terminal-bound, the projection is format-only translation of real
  actions/observations; (b) the spec pins a hermetic interpreter (1643dacd...)
  absent from this host -- the byte-verified sealed program runs under the
  project venv interpreter, recorded per grade; (c) sealed limits.turns=12 is
  honored across prefix+suffix, while limits.seconds/completion_tokens are
  dense-controller accounting replaced by the Pi-native lane's own budgets;
  (d) the frozen 11-tool Pi-native surface has no listing tool, so the two
  opaque discovery tasks cannot pass the sealed minefield (list_files/read)
  in this lane -- they are attempted, graded, and fail honestly;
  (e) development-split tasks are measured but never receipted (sealed split
  isolation: receipt_eligible=false).
- SPLICE TEST PASSED (CPU, real Pi transport, real sealed validator):
  scripts/rl_splice_test.py --scratch smoke/splice-test: authored failing
  policy prefix (real read of a stale path -> real error observation -> stop),
  attempt grade FAILED by the sealed validator, splice-correct continued the
  byte-exact policy prefix (fixture teacher read the sealed required path for
  real and finished with the observed token), sealed validator PASSED the
  spliced episode, receipt 3c8a79b2e47c3068... supervise_from=1 (6066 tokens /
  118 targets), TARGET SPAN PROVEN TEACHER-ONLY (policy frame span [5665,5724)
  unsupervised; teacher spans [5872,5955)+[6031,6066) supervised; 118/118
  teacher tokens = all targets; encode_candidate mask == independent span
  recomputation; canonical boundary-aware pack built and the packed authority
  assistant_mask == the teacher-only span). Artifacts: smoke/splice-test/.
- Next: multi-cycle continuous run on the sealed gym tasks (cycles 2+, live
  GLM-5.3 teacher, 1 leased GPU) via scripts/run_cycles.sh.

- 2026-09-27 (t0) Workspace created. Grounding read: teacher pilot
  (scripts/collect_e97_teacher_generation_pilot.py + e97-teacher-generation-pilot-v1),
  docs/RESILIENT_DILOCO_COMPUTE_POOL.md no-database amendment (2026-07-25),
  ndm/e97_task_lake.py + ndm/e97_task_leases.py (NOTE: leases module is SQLite —
  the no-database amendment forbids SQLite on loop paths, so the v1 queue/receipts
  design below uses filesystem digest-linked streams only),
  scripts/serve_e97_agent_openai.py + ndm/e97_agent_server.py,
  scripts/eval_e97_native_execution.py, scripts/train_e97_4b_pi_sft.py
  (incl. commit 3cf6439e P=2/chunk flags),
  scripts/prepare_e97_pi_native_repair_training.py + build_e97_sft_packs.py
  (receipts->authority->packs precedent), native-actor/onpolicy-canary repair
  precedent (scripts/e97_native_onpolicy_canary.py).
- Sibling lane draft e97-rl-leg-proposal-v1/DRAFT.md does NOT exist yet (dir is
  empty). Operating model taken from the task directive + teacher pilot.
- KEY FINDING: scripts/train_e97_4b_pi_sft.py hard-requires world in {8,64}
  ranks in 8-rank islands ("qualified Pi SFT worlds are 8 or 64 ranks"). A
  one-GPU train step with the canonical trainer is therefore impossible;
  the smoke constraint (ONE GPU, never 8) wins. v1 train stage uses a
  workspace-local single-GPU masked-SFT step (scripts/rl_train_step.py) that
  reuses the canonical loader/precision policy/optimizer/pack machinery
  verbatim and mirrors the canonical checkpoint schema. Deviation is
  documented in README gap list.
- KEY FINDING: serve_e97_agent_openai.py's HTTP service layer parses turns as
  the dense-agent "Action:/Arguments:/Final:" protocol and rejects the
  five-line Pi-native frames ("Analysis:/Commentary:/Think:/Action:/Arguments:")
  that the teacher-pilot bridge, the real Pi tool transport, and the
  receipts->packs encode path (encode_candidate) all use. v1 SERVE stage
  therefore uses the same agent-server ENGINE (ndm.e97_agent_server.TorchE97AgentEngine)
  + load path (ndm.e97.load_e97_checkpoint, weight_mode='saved') in-process,
  driving rollouts exactly like scripts/eval_e97_native_execution.py did for the
  96 native-execution episodes. The HTTP surface is unused in v1 (gap list).
- Building: queue/seed tasks (pilot terminal_case machinery, mechanical
  verify checks), digest-linked receipt stream (no SQLite anywhere),
  receipts->authority->packs converter, single-GPU train step, loop driver,
  smoke test on ONE leased GPU.
- CPU DRY-RUN PASSED (fixture teacher): 3/3 seed tasks (merge-shards,
  rename-manifest, config-migration) completed through the REAL Pi tool
  transport (real pi child executing the eleven frozen tools), mechanical
  grades passed, 3 verified receipts appended to the digest chain and
  re-verified (full encode_candidate re-encode), authority built
  (3 records / 19,326 tokens / 1,174 targets), canonical boundary-aware
  packs built via scripts/build_e97_sft_packs.py (3 packs, context 8192,
  epoch-permutation). Artifacts: smoke/cpu-dryrun/.
- CPU DRY-RUN PASSED (LIVE teacher): GLM-5.3-flash-background through lunaroute
  + the pilot's Pi-transport frame contract, 3/3 episodes verified
  (wall 164s/26s/68s), 2,460 assistant target tokens, receipts+chain verified.
  Artifacts: smoke/cpu-dryrun-live/. The live teacher IS reachable from this
  workspace, so the smoke test uses the live teacher (fixture kept as
  documented fallback stub).
- REMAINING for smoke: GPU stages (policy engine load + rollouts, single-GPU
  train step, re-serve probe). Running next via the lease broker (1 GPU).
Sat Sep 26 10:15:21 PM UTC 2026
- OPERATOR STEERING folded in (mid-smoke, no restart needed — both decisions
  were already the implemented design):
  (1) QUEUE STORAGE: ndm/e97_task_leases.py is SQLite-backed and MUST NOT be
      the queue storage (no-database amendment; sibling review flagged the
      same conflict). v1 queue keeps lease SEMANTICS on a pure filesystem
      basis: claim = one atomic rename pending->active; deadline = episode
      budgets + crash-recovery sweep of active/ back to pending/; receipt =
      the verified digest-linked receipt stream. README now documents this.
  (2) V1 TRAIN STEP: the RL optimizer audit found the GRPO-style candidate
      machinery has never taken an update (rl_optimizer_ready: false). v1
      trains on the PROVEN path: verified receipts -> cohort-format packs
      (prep-builder authority/pack layout) -> masked-SFT step with chunk-2048
      on ONE GPU. Policy-gradient channel is v2. README now documents this.
  (3) GRADER: the task-lake validation layer (ndm/e97_task_lake.py + sealed
      first-party validator argv) is the designated grader hookup for the
      sealed-bundle task tranche (fits the grading role cleanly per the gym
      machinery recovery review); it is spec-digest-bound to sealed bundle
      machinery outside v1 scope, so v1 seeds use the pilot run_check
      mechanical checks in the same role. Noted in the README gap list.
- SMOKE PROGRESS: GPU collect stage PASSED — E3-u256 policy served, emitted
  valid canonical five-line Pi-native frames on every task, finished its
  episodes, and FAILED all 3 mechanical grades (honest on-policy result: the
  u256 checkpoint can speak the protocol but cannot complete the seeded
  tasks). Live GLM-5.3 teacher-correct produced 3/3 VERIFIED continuations in
  the SAME workspaces; 3 teacher-corrected receipts appended to the digest
  chain (ddb7d384..., ff326797..., one more) and the stream verified.
  Authority built (sha 5fe76e5ded8c2fba...), canonical packs built
  (sha 8ffa7f15552cd89f..., 3 packs, context 10240). Train step (7/8)
  running on leased GPU 1.

## SMOKE TEST RECEIPT (cycle 1, 2026-09-26/27) — PASS

One full cycle on ONE leased GPU (gpu_lease broker granted GPU 1; CUDA_VISIBLE_DEVICES
single-GPU; released cleanly; `smoke/cycle-0001/terminal.json`).

Stages, all REAL machinery:
- SERVE: E3-u256 checkpoint 3770621c… sha-verified, loaded via the agent-server
  engine path (TorchE97AgentEngine, load_e97_checkpoint, weight_mode='saved',
  bf16+triton, GPU).
- ATTEMPT (3/3 queued seed tasks): the u256 policy emitted VALID canonical
  five-line Pi-native frames on every task and FINISHED every episode
  (2/2/3 turns, 150-192 tokens), executing real tools through the real Pi
  child — but FAILED all 3 mechanical grades (honest measured result: the
  checkpoint speaks the protocol and cannot complete the seeded tasks).
  Evidence: episodes/cycle-0001/<task>/attempt/{episode-private.json,grade.json}.
- GRADE: pilot run_check mechanical verifier over the real workspaces; grade
  receipts published (immutable).
- TEACHER-CORRECT (3/3): LIVE GLM-5.3-flash-background via lunaroute through
  the pilot's Pi-transport frame contract continued the SAME workspaces from
  the policy partial trajectories; every teacher continuation PASSED the same
  mechanical grader (task-0001's teacher even recovered from 2 real tool
  errors mid-episode). No fixture stub was needed — the live teacher IS
  reachable and was used.
- QUEUE: 3 verified teacher-corrected receipts appended to the digest-linked
  stream (NO SQLite anywhere):
    ee6313cbddb506cf  722 targets, 3 turns
    ddb7d3846566d5c3 1179 targets, 5 turns
    ff3267974b1cd921  868 targets, 5 turns  (chain head)
  Total 2,769 assistant target tokens / 27,283 episode tokens. Stream walked
  and re-verified (full encode_candidate re-encode of every episode).
- TRAIN: receipts -> authority (emender-e97-tulu3-masked-sft-v1, manifest
  sha 5fe76e5ded8c2fbae8f3bf80c0966034153802a2050ae05328f0dc5edb220881,
  3 records) -> canonical boundary-aware packs via build_e97_sft_packs.py
  (manifest sha 8ffa7f15552cd89f20c89abc55789e55688792f2bfa7121ec81b55861727a7ac,
  3 packs, context 10240) -> single-GPU masked-SFT step (rl_train_step.py,
  chunk-2048 geometry, FP32 logits + 2048-token chunked CE, CPU-offloaded
  AdamWScheduleFree): 1 update, 9,657 tokens / 1,179 targets, loss 2.1819,
  grad_norm 18.9, 22.8 s step, 20.1 GB HBM peak — fits one GPU.
  New checkpoint (canonical schema, 24.3 GB):
    training/cycle-0001/checkpoints/checkpoint_agent_sft_u000001_loss_2.1819.pt
    sha256 1e0594646189544609f7dcff106acbd8431eac8b98721a015356891425da1e12
- RE-SERVE: the updated checkpoint re-loaded through the same engine path and
  emitted a syntactically VALID canonical frame (124 tokens, greedy, 10 s) —
  probe evidence smoke/cycle-0001/re-serve-probe.json. (Competence is
  unchanged after a single 1,179-target update, as expected; the probe's
  frame proposes an unverified bash command — honest, recorded verbatim.)
- Loop state now points at the new checkpoint; queue is empty (work-driven
  idle); the loop would continue cycle 2 from 1e059464… on new tasks.

Earlier crash artifacts retained as durable evidence:
episodes/cycle-0001-crashed-1790462836 (first run hit a missing
correction-dir mkdir; fixed; the crashed attempt evidence was archived and
orphans swept back to pending before the successful rerun).

## MULTI-CYCLE RUN RECEIPT (cycles 2-6, 2026-09-27) — the first real
## continuous run — PASS

Five consecutive cycles on the SEALED first-party gym tasks with spliced
live-teacher corrections, one leased GPU for the whole run (released cleanly;
`cycles/run-cycles-3-6.out` terminal `original_exit:0`). Per cycle:
gym-freeze (task-lake validation layer, 4 sealed bundles) -> collect
(policy attempts -> sealed-validator grades -> spliced GLM-5.3 corrections ->
verified receipts) -> stream verify -> authority/packs -> single-GPU
masked-SFT step -> re-serve probe -> state update.

Honest per-cycle metrics (full detail: cycles/cycle-NNNN/metrics.json,
cycles/cycle-NNNN/… per-cycle logs; assembled from collect-summary +
collect stream + train log + probe):

| cycle | attempts | policy pass | teacher pass | receipts (kind) | targets queued | step loss | new checkpoint | probe |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 0 | 2/4 | 1 teacher-corrected (SPLICED, supervise_from=1) | 75 | 1.4178 | 9a3b49ff… | valid |
| 3 | 4 | 0 | 2/4 | 1 teacher-corrected (SPLICED, supervise_from=1) | 90 | 0.9512 | c647fc60… | valid |
| 4 | 4 | 1 | 1/4 | 1 on-policy-success | 149 | 0.1214 | 9a0a241f… | valid |
| 5 | 4 | 1 | 1/4 | 1 on-policy-success | 142 | 0.0159 | 1cf9ecfb… | valid |
| 6 | 4 | 1 | 1/4 | 1 on-policy-success | 121 | 0.0044 | bec77602… | valid |

The on-policy learning progression is visible and honest: the train
direct-read task (9f036dbb…) was failed by the policy in cycles 2-3 (it
finished prematurely with the wrong final; the splice DROPPED that trailing
finish frame, the teacher continued from the policy's own read observation
and finished with the correct token — receipts dbff49eb… and 77a93754…,
target spans PROVEN teacher-only by re-encode), and from cycle 4 onward the
policy PASSES the same sealed task on-policy (receipts 6558ee15…,
d998206c…, 0a42683c…). Step loss collapses toward zero on this tiny
direct-read distribution once the policy masters it (cycles 4-6 train on
the policy's own correct episodes) — a real signal, not a claim of general
competence. The two opaque discovery tasks (one train, one dev) fail every
cycle: the frozen 11-tool Pi-native surface has no listing tool, so the
sealed minefield (list_files/read only) is unsatisfiable for them in this
lane (grade detail: "minefield tool violation" when the teacher uses
fffind/bash to discover, or no grounded read at all) — recorded, not
hidden. The dev direct-read task (fa889a72…) is teacher-passed every cycle
but never receipted (sealed train/development split isolation;
receipt_eligible=false).

Receipt-chain head after cycle 6 (all 8 receipts re-encode verified, the 2
spliced receipts re-verified with supervise_from=1 partial-target masking,
target spans == teacher suffix tokens exactly):

    0a42683cc9c2127e4088fdd8ea354b16a8e07646ebe62fd3dfb577a9ce43a5b6

Checkpoint lineage across the run: 1e059464… (cycle-1 smoke output)
-> 9a3b49ff… -> c647fc60… -> 9a0a241f… -> 1cf9ecfb… -> bec77602…
(current state; every receipt binds the exact checkpoint sha it was
produced against).

## Residual gaps (next lanes)
1. ~~Same-transcript teacher-correct records~~ CLOSED (tranche 2): corrections
   are byte-exact same-transcript splices; only the teacher suffix is
   supervised. Remaining slice: policy-gradient (RL) channel on top of the
   splice (v2, rl_optimizer_ready: false today); sampling + log-prob
   receipts for on-policy turns.
2. Live-teacher role-play episode design + GLM-5.3 guide review (pilot has
   both; deliberately out of scope).
3. ~~Sealed first-party task bundles + task-lake validation layer as the
   grader hookup~~ CLOSED (tranche 2): sealed bundles are pulled through the
   task-lake validation layer and graded by the actual sealed validator
   program over faithful dense-terminal projections. Remaining: (a) the gym
   is only 4 admitted bundles — a bigger sealed tranche (fresh first-party
   generation + admission, or swesmith/swegym/r2egym sources once admitted)
   is the next authority decision; (b) the two opaque discovery tasks need a
   Pi-native listing tool (or a task-scoped tool surface) before they can
   pass the sealed minefield; (c) the dense-controller lane itself (worktree
   Cluster A-E adoption) remains un-adopted in main.
4. Sealed-lane deviations to close later: hermetic pinned interpreter
   (1643dacd…) absent on this host (project interpreter used, recorded per
   grade); sealed seconds/completion_tokens accounting replaced by
   Pi-native budgets (turns=12 honored across prefix+suffix).
5. Canonical 8-rank trainer (or DiLoCo workers) once multi-GPU is authorized;
   v1/tranche-2 train step is a world-of-one deviation.
6. Full 5,181-record gym + dynamic task authoring + multi-queue attribution.
- CYCLE 2 (sealed gym, spliced corrections): attempts=4 policy_pass=0 teacher_pass=2 receipts=1 ({"teacher-corrected": 1}) spliced=1 targets_queued=75 step_loss=1.4177668253580729 new_ckpt=9a3b49ffc4793650 probe_valid=True chain_head=dbff49ebdb0ce9aa (metrics: cycles/cycle-0002/metrics.json)
- CYCLE 3 (sealed gym, spliced corrections): attempts=4 policy_pass=0 teacher_pass=2 receipts=1 ({"teacher-corrected": 1}) spliced=1 targets_queued=90 step_loss=0.9512216356065538 new_ckpt=c647fc60e974954b probe_valid=True chain_head=77a937548df3269a (metrics: cycles/cycle-0003/metrics.json)
- CYCLE 4 (sealed gym, spliced corrections): attempts=4 policy_pass=1 teacher_pass=1 receipts=1 ({"on-policy-success": 1}) spliced=0 targets_queued=149 step_loss=0.1213838846091456 new_ckpt=9a0a241fc4c3a40e probe_valid=True chain_head=6558ee15b727a452 (metrics: cycles/cycle-0004/metrics.json)
- CYCLE 5 (sealed gym, spliced corrections): attempts=4 policy_pass=1 teacher_pass=1 receipts=1 ({"on-policy-success": 1}) spliced=0 targets_queued=142 step_loss=0.015885987751920457 new_ckpt=1cf9ecfb4dabd07e probe_valid=True chain_head=d998206cdf528ad1 (metrics: cycles/cycle-0005/metrics.json)
- CYCLE 6 (sealed gym, spliced corrections): attempts=4 policy_pass=1 teacher_pass=1 receipts=1 ({"on-policy-success": 1}) spliced=0 targets_queued=121 step_loss=0.00443632198759347 new_ckpt=bec776022fc96d72 probe_valid=True chain_head=0a42683cc9c2127e (metrics: cycles/cycle-0006/metrics.json)

## RL BANK BUILD (2026-09-27, tranche 3) — the 8-GPU DiLoCo-synced bank

- GROUNDING COMPLETE. D re-pin tranche lane landed on main BEFORE this build
  started: dc8fffd7 (re-pin artifacts + cluster-D controller code) + 9a92bc49
  (deferred identity/snapshot tests) — both on main, working tree clean, nothing
  to coordinate around; no commits clobbered (this workspace commits nothing).
- KEY FINDING (blocking, escalated, supervisor DECIDED): the "full 5,181
  replay-proven first-party gym" DOES NOT EXIST as a task-lake authority.
  Verified: only ONE admitted first-party collection exists anywhere (the 4
  bundles this loop already uses, e97-firstparty-cpu-phase-bc-v1-admitted);
  the 5,181 figure traces to the OpenHands T1 translation funnel (5,181
  replay-PASSED trajectory records — an EXTERNAL source, and episode records
  not task bundles with sealed validators); the first-party generator
  (ndm/e97_first_party_read_observe.generate) emits exactly 4 read-observe
  bundles per seed; every NEW collection admission fails closed without an
  operator authorization entry in the checked-in allowlist
  (configs/pi/e97-firstparty-collection-authorizations-v1.json — currently
  EMPTY/candidate). SUPERVISOR DECISION: build the bank on the EXISTING
  admitted 4-bundle lake (machinery lake-agnostic; any bigger gym authority
  drops in when the operator creates it — the ~5,181 first-party generation
  path is an operator data-authority decision, NOT authorized here). The
  supervisor is briefing the operator that the 5,181 claim in the build
  brief was factually wrong (supervisor's error, recorded here and in the
  README honest-limits).
- KEY FINDING (sealed validator drift, escalated, supervisor APPROVED the
  fail-closed fix): the D re-pin tranche changed
  scripts/e97_first_party_validator.py bytes 344a1209 -> 72820d63, but every
  admitted bundle's private spec pins program_sha256=344a1209 — the tranche-2
  gym grading path fails closed against current main as-is. Fix (APPROVED):
  bind the byte-verified pinned program (344a1209, recovered byte-exact from
  git 2002eded:scripts/e97_first_party_validator.py, which uses exactly the
  --mode/--spec/--terminal argv the loop's grader invokes) into the
  workspace as the gym program_path; sha-verified per bundle at pull time
  AND per grade; recorded in every grade receipt as before. The re-pinned
  bytes (72820d63) are for FUTURE collections whose specs pin them — both
  program versions coexist under their own shas; no bytes are unverified.
- LOCKED SUPERVISOR DEFAULTS implemented: receipts->SFT channel primary
  (GRPO stays v2 — rl_optimizer_ready:false unchanged); sync K=32 updates
  (DiLoCo geometry); merge = weighted soup by cycle pass-rate with
  winner-take-all fallback; ONE global task pool with work-stealing +
  filesystem lease semantics (deadline-bounded claims, stale claims
  re-queueable — no SQLite anywhere); first-party tasks ONLY; teacher on
  background slots throttled to queue consumption.
- DESIGN (built per docs/RESILIENT_DILOCO_COMPUTE_POOL.md + ADR-003; cites:
  R07/NDP15 atomic checkpoint publication — temp + os.replace, pointer only
  from a readable complete file; R12 stable lineage restart; R14/NDP13
  bounded termination per stage; R16 evidence discipline — every artifact
  digest-bound; R01/R10 no-database clause — filesystem lease semantics
  only, no SQLite, no shared locks. NOT claimed: R02-R06/R08-R11, NDP02,
  NDP17 elastic/native semantics — fixed 8-lane bank, no dynamic membership).
  Bank layout: bank/pool/{pending,claims,done} (global work-stealing pool),
  bank/lanes/lane-00..07 (per-lane state + per-lane digest-linked receipt
  chains + per-lane packs/training/cycles in the v1 metrics format),
  bank/merges (atomic merged checkpoints + digest-linked merge receipt
  chain + current.json convenience pointer — never authority),
  bank/coordinator.json (K watermarks, pool rounds), bank/report (session
  metrics). Merge trigger: any lane's updates_total - its counter snapshot
  at last merge >= K=32 (race-free: lanes keep monotonic counters; the
  coordinator owns the watermark). Merge math mirrors train.py:diloco_merge
  tensor semantics: model x + optimizer z weighted-souped by per-lane
  window pass-rate, exp_avg_sq + scalar clocks taken from the winner
  (diloco_merge keeps exp_avg_sq per-rank; across 8 independent file-based
  lineages the winner's is the defensible pick — recorded per merge
  receipt); winner-take-all fallback when the soup has no positive weight;
  merge SKIPPED (receipted, no publish) when zero passes bank-wide (no
  signal to weight or rank by). GPU lanes via the gpu_lease broker: a lane
  holds a lease only while it has claimed work + unconsumed receipts
  (yield-when-idle — the bank must not squat 8 GPUs while idle; shared box
  with user shuoc).
- BUILT (CPU, all in this workspace, nothing committed): scripts/rl_bank.py
  (bank plumbing: ONE global pool with lease-claim semantics — atomic-rename
  claim, deadline-bounded claim block, heartbeats, stale sweep, voluntary
  requeue, fail-closed double-retirement refusal, pinned-validator binding,
  round-scoped pool refresh, lane state, merge current pointer, coordinator
  state), scripts/rl_bank_lane.py (per-lane orchestrator + GPU collect stage:
  the v1 stage functions verbatim, bank claim/heartbeat/retire, per-lane
  digest-linked receipt streams, teacher throttled to queue consumption,
  crashed-cycle evidence archival, subprocess GPU stages with hard deadlines,
  GPU lease per cycle with a heartbeat keeper bound to the lane pid,
  yield-when-idle release at every cycle boundary), scripts/rl_bank_merge.py
  (the K-boundary DiLoCo merge: weighted soup by window pass-rate over lane
  lineages, diloco_merge tensor semantics x+z weighted, winner's exp_avg_sq +
  scalar clocks, winner-take-all ranking, zero-signal SKIP with receipt,
  R07 atomic publish, digest-linked merge receipt chain),
  scripts/rl_bank_coordinator.py (K=32 trigger via per-lane monotonic
  update counters + merge-time snapshot watermarks — race-free; stale-claim
  sweep; pool round refresh when drained; session metrics assembly;
  bounded-termination loop), scripts/rl_bank_unit_test.py (28 CPU tests),
  scripts/run_bank.sh (preflight -> lane init -> coordinator + N lanes ->
  bounded session -> STOP -> final report).
- TWO MORE D re-pin compatibility breaks found and fixed in the workspace gym
  path (both would have broken even the v1 single-loop path against current
  main; both are the adopted cluster-D code being era-incompatible with the
  ADMITTED lake's immutable pins — the lake's pins govern, per the supervisor
  decision): (a) safe_extract_fixture_archive moved to a retained-BYTES
  first argument -> the gym extractor now snapshots the archive bytes once
  and passes them (the adopted hardening, kept); (b) _tree_digest widened to
  include directory entries -> the admitted lake's fixture_tree_digest pins
  reproduce ONLY under the 2002eded-era FILES-ONLY tree digest semantics
  (verified empirically: files-only digests MATCH the pinned values; the
  adopted widened digest does not), so the extractor keeps every adopted
  safety property (member safety, count/disk bounds, non-regular rejection,
  bytes-first) and verifies the expanded tree with the era-correct
  files-only semantics that produced the sealed pin. Both recorded per grade
  receipt and in the README honest-limits.
- UNIT VALIDATION PASSED (CPU, 28/28, scripts/rl_bank_unit_test.py, real
  lake + real lease machinery): pool exactly-one-winner claims under 4-way
  REAL process concurrency, heartbeat lease extension, stale-claim requeue +
  cross-lane reclaim, retirement evidence; merge weighted-soup math exact
  against fp32 reference (x AND z), winner exp_avg_sq/clock selection, R07
  atomic publish + sha binding, digest-linked merge chain, zero-signal SKIP
  (nothing published); K-trigger snapshot window semantics; pinned validator
  bytes == lake pin (344a1209…); 4 sealed bundles admitted through the task-
  lake validation layer into the pool with round-scoped ids bound to the
  pinned program bytes; sealed split isolation (train receipt-eligible only);
  lane state round-trip.
- Next: 2-lane GPU pilot session (shake out live orchestration end to end),
  then the first full 8-lane bank session.

## TWO-LANE GPU PILOT RECEIPT (2026-09-27 05:38-06:07 UTC) — PASS

run_bank.sh with LANES=2 SESSION_MINUTES=16, coordinator K=32, bank dir
bank-pilot2/, seeded from the loop's current checkpoint bec77602… (the
proven v1 cycle-6 output). Both lanes exited cleanly at the STOP boundary
AFTER completing their in-flight cycles (no state lost: the two last trains
finished post-STOP and their receipts/lineages are intact); coordinator
COORD_EXIT clean; every GPU lease released (verified: lease table empty,
box idle, yield-when-idle at every cycle boundary).

Live end-to-end evidence (bank-pilot2/report/session-metrics-final.json,
launch.log, lane-*.log, lanes/*/cycles/*/metrics.json):
- 6 pool rounds (round-scoped refresh from the 4-bundle admitted lake),
  24 retired task claims (lane-00: 11, lane-01: 13; retire stages:
  teacher 8, honest-failed 12, policy 4).
- 6 verified receipts (both per-lane digest chains walked + re-encode
  verified): 2 teacher-corrected (SPLICED, supervise_from=12, targets
  75/91) + 4 on-policy-success. Teacher calls: 46.
- 6 train updates (3 per lane), step-loss trajectories 1.5844 -> 0.0087 ->
  0.0021 (lane-00 lineage) and 1.6950 -> 0.0089 -> 0.0071 (lane-01 lineage);
  every trained cycle's re-serve probe frame valid.
- NO merge fired: expected physics, not a fault — the 4-bundle lake yields
  exactly ONE receipt-producing task per pool round (train direct-read
  9f036dbb…; the opaque train bundle fails honestly in this lane, dev-split
  tasks are measured but never receipted), so measured bank-wide update
  production is ~1 update/round ≈ 22/hour. K=32 per LANE therefore puts the
  first merge HOURS out (a full merge cycle was NOT observable in the pilot
  window at the locked geometry — escalated, see below).
- Work-stealing + lease semantics behaved: sorted-claim races with no
  double-retirement, heartbeats extended mid-collect, claims peak 5 during
  multi-task steals then retire to 0; round refresh re-attempted the sealed
  tasks against CURRENT lane lineages each round exactly as designed.

Live-surface findings (shake-out outcome):
(a) run_bank.sh stop-wait bound (300s) was shorter than in-flight cycle
    completion -> cosmetic "still alive" WARNING + the final report was
    assembled while lanes were finishing their last cycles. FIXED: the
    graceful-exit wait is now bounded at 1800s and the final report is
    assembled after the wait (matches the observed in-flight-cycle span).
(b) run_bank.sh had no way to leave a healthy bank RUNNING (operator wants
    indefinite operation). FIXED: SESSION_MINUTES=0 now means INDEFINITE —
    the monitor loop runs until $BANK/STOP appears (never touches it),
    lanes/coordinator get a 30-day R14 bound instead of the session window,
    and the clean-stop procedure is documented in the script header:
    `touch $BANK/STOP` (lanes exit at cycle boundaries; coordinator at its
    next poll).
(c) MERGE SEMANTICS GAP (escalated, not silently patched): the merge
    docstrings define soup weights as the pass rate "since the last merge",
    but window_attempts/window_passes are never reset anywhere — from
    merge #2 onward the implemented weights would be LIFETIME pass rates,
    silently diverging from the documented locked rule. Also escalated:
    merge observability at locked K=32 vs 4-bundle-lake physics (see
    coordinator decision once recorded).
- Next: supervisor decision on (c), then the first full 8-lane bank session.

- 2026-09-27 (live phase, supervisor decisions + fixes landed): Q1 DECIDED
  option C (production 8-lane session stays at locked K=32 INDEFINITE;
  live merge wiring proven separately in a throwaway 2-lane K=2 shakeout
  bank — pilot-scale evidence, recorded in the README honest-limits).
  Q2 DECIDED option (i): soup weights are the pass rate SINCE THE LAST
  MERGE as documented — implemented as a lane-owned window-counter reset
  on merge ADOPTION (adopt_merge_current in rl_bank_lane.py; race-free:
  only the lane writes its state; updates_total stays monotonic for the
  K trigger). New unit tests (adoption: no-pointer no-adoption, sha-verified
  adoption, window reset, updates_total preserved, no re-adoption) —
  33/33 PASS; run_bank.sh preflight now requires 33.

## MERGE SHAKEOUT RECEIPT (2026-09-27 06:12-07:13 UTC, bank-merge-shakeout) — PASS

The supervisor-approved throwaway 2-lane K=2 bank (pilot-scale evidence
ONLY — production geometry stays K=32; session window 60m, stopped early
via `touch $BANK/STOP` after the second merge). THE LIVE MERGE PATH IS NOW
PROVEN END-TO-END, not just unit-validated:
- MERGE #1 (index 1, epoch 1): trigger lane-1 at +2 updates; weights from
  pre-fix window semantics lane0 0/3 policy passes -> 0.0, lane1 1/3 ->
  0.3333 (only positive -> normalized 1.0); winner lane-1; 237 model
  tensors souped, 236 z tensors, 236 winner exp_avg_sq; merged checkpoint
  3961c27b… (24.66 GB) published atomically; receipt 9789503f… (chain
  genesis). BOTH lanes ADOPTED at their next cycle boundary with the NEW
  window-reset semantics verified LIVE (window counters 0/0/0 on adoption,
  updates_total preserved 2/3 — monotonic K geometry intact), then trained
  post-adoption cycles from the merged lineage.
- MERGE #2 (index 2, epoch 2): trigger lane-0 at +2 updates past the merge-1
  snapshot; weights from the RESET windows (lane0 1/2 -> 0.5, lane1 0
  attempts since adoption -> 0.0); winner lane-0; merged bd2f9064…, receipt
  a1c9386e… chaining onto 9789503f… (digest-linked merge chain verified,
  canonical-JSON receipt digests recomputed and matched for BOTH receipts).
  The adaptive window weighting works as decided: each merge ranks CURRENT
  lineages, not lifetime luck.
- Honest observation for the operator: in the 4-bundle lake regime the
  soup DEGENERATES to winner-take-all in practice — only one lane can
  hold a positive window pass rate at a time (the lake yields exactly one
  receipt-producing task per round). The weighted-soup machinery is proven;
  its multi-lane weighting needs a bigger gym to be meaningful.
- Session totals: 9 pool rounds, 36 retired claims (17 lane-00 / 19
  lane-01), 9 receipts (7 on-policy-success + 2 teacher-corrected spliced),
  71 teacher calls, 9 train updates (4+5), step losses collapsing per
  lineage (1.51 -> 0.0004 / 1.53 -> 0.0004), all probes valid, LANE_IDLE
  observed (yield-when-idle engaged when the pool drained between rounds),
  clean STOP exit — the improved run_bank.sh reported "all bank processes
  exited cleanly" and assembled the final report AFTER exit; every GPU
  lease released (verified: lease table empty).
- Artifacts: bank-merge-shakeout/ (merges/merge-000{1,2}, merges/stream.jsonl,
  report/session-metrics-final.json, launch.log, lanes/*/…).
- Next: the first full 8-lane bank session (K=32, INDEFINITE, all 8 leased
  GPUs), left running per the operator directive.

- 2026-09-27 (live phase, RECLAIM decision executed): supervisor APPROVED
  lane-side GC of superseded checkpoints (option i) + shakeout GC + a
  low-disk fail-closed guard. Motivation (measured): every train update
  writes a ~24.6GB checkpoint; at ~22 updates/hour bank-wide an unGC'd
  indefinite 8-lane session hits ENOSPC on the workspace volume in ~11h.
  Implemented: rl_bank.reclaim_superseded_lineages (RECLAIM-LOG discipline
  — ledger in lane state; deletes ONLY this lane's own grace-elapsed
  superseded training checkpoints after sha verification; fail-closed
  refusals: merged checkpoints kept forever, any lane's current lineage,
  anything outside the lane's training tree, identity drift; every
  deletion logged with sha+reason in bank/reclaim-log.jsonl); lanes append
  to the ledger on lineage advance AND merge adoption, and run the reclaim
  at every cycle start; MIN_FREE_BYTES=100GiB guard (the run.sh precedent)
  pauses lane cycles (LANE_DISK_LOW, status disk-low) and defers
  coordinator merges without disarming the K trigger. Unit validation now
  42/42 (9 new: reclaim happy path + log + ledger + 5 refusal paths +
  disk-guard predicate); run_bank.sh preflight requires 42. The throwaway
  shakeout bank's checkpoint bytes were GC'd per the decision (20 files,
  268.2GB reclaimed; all receipts/merge receipts/reports/logs kept; the one
  file lost to an aborted first sweep attempt is retroactively logged with
  its receipt-bound sha). Free space 5.9T -> 6.2T.

## FIRST FULL 8-LANE BANK SESSION — LIVE (2026-09-27 07:31 UTC launch, RUNNING)

run_bank.sh LANES=8 K=32 SESSION_MINUTES=0 (INDEFINITE, operator directive)
  bank dir: bank/ (the production bank), console: bank-console.log
  clean stop (any time): `touch bank/STOP` — lanes exit at cycle boundaries
  after finishing in-flight cycles, the coordinator at its next poll, and
  run_bank.sh then assembles bank/report/session-metrics-final.json.
Preflight 42/42 on the real lake; all 8 lanes seeded from bec77602…
(the proven v1 current checkpoint) at 07:36:12Z; coordinator K=32.

Opening telemetry (first ~46 min, launch.log MONITOR lines + report):
- ALL 8 LANES LIVE and cycling; 45 pool rounds in 46 min (~1 round/min —
  faster than the 2-lane pilot because 4 tasks/round are claimed in
  parallel by 4 lanes); 177 retired claims; 44 receipts (19
  teacher-corrected spliced + 25 on-policy-success); 121+ teacher calls
  (~5/min peak, no teacher failures); 30 train updates spread EVENLY
  across lanes [2-4 per lane] — step losses per lineage collapse fast
  (e.g. 1.69 -> 0.012 -> 0.006…), the expected mastered-bundle saturation.
- YIELD-WHEN-IDLE ENGAGES EXACTLY AS DESIGNED: with 4 claimable tasks per
  round against 8 lanes, the un-claimed lanes park (LANE_IDLE, GPU lease
  RELEASED at the cycle boundary) — the bank never squats more GPUs than
  it has work; leases rotate across GPUs 0-7 with fresh heartbeats.
- RECLAIM-LOG DISCIPLINE LIVE-VERIFIED in production: after the 15-min
  grace, lanes reclaim their own superseded checkpoints (LANE_RECLAIMED
  lines; bank/reclaim-log.jsonl — 7 x 24.3GB in the first hour), refusing
  the shared v1 seed (outside the lane's training tree, dropped from the
  ledger without deletion). bank/ disk growth bounded at the predicted
  ~500-600GB steady-state envelope; 5.6T free.
- No merge yet (expected: K=32 per lane; at the measured ~40
  updates/hour bank-wide spread across 8 lanes, the first merge is ~6-7h
  out — the live merge path itself is already proven end-to-end by the
  K=2 shakeout above).
- No LANE_FAILURE, no COORD_FAILURE, no LANE_DISK_LOW, no lease-wait
  contention with shuoc in the first hour.
- Session left RUNNING (indefinite) per the operator directive.

## RL V2 BUILD — THE POLICY-GRADIENT CHANNEL (2026-09-27, operator GO "when do we RL = now")

- GROUNDING COMPLETE. Read: this workspace (README/STATUS/RECEIPT_SPEC/scripts — the running
  8-lane bank), the RL-leg proposal audit (e97-rl-leg-proposal-v1/DRAFT.md §6: the GRPO-style
  candidate ndm/e97_outcome_rl_candidate.py is implemented but has NEVER taken an update;
  rl_optimizer_ready:false in all four native-RL qualification workspaces),
  the qualified logprob machinery (scripts/qualify_e97_native_rl_logprobs.py — the
  teacher-forced training-layout capture: turn_layout alignment-128 rows, chunked head
  hook, FP32 log_softmax; measured cached-replay bit-exactness in worker capture mode,
  CE-head consistency 9.1e-8), and ndm/e97_outcome_rl_candidate.py verbatim.
- KEY DESIGN DECISIONS (all recorded honestly; none change the running bank's behavior):
  (1) TRAIN STEP: scripts/rl_policy_gradient_step.py — a SECOND train step alongside
      rl_train_step.py. Rows = (episode, assistant turn) in the qualified turn_layout
      (prefix = full episode tokens before the turn, generated = the turn's tokens);
      per-token policy logprobs captured through the SAME chunked-head FP32 log_softmax
      path the logprob assay qualified (head forward-hook; checkpoint_loss_chunks stays
      ON — the hook is removed before backward so the checkpoint replay can never
      double-capture; the internal chunked-CE sum is kept as a per-row consistency
      check mirroring the assay's ce_head_consistency gate).
      Advantages = ndm.e97_outcome_rl_candidate.group_advantages over binary
      sealed-validator outcomes (pass=1/fail=0) grouped by LAKE TASK ID across
      same-task attempts in the batch window — the reward contrast. Loss =
      the candidate's policy_loss verbatim (token-clipped surrogate, clip 0.2,
      + k3-KL against a FIXED anchor) summed per row (the candidate docstring's
      "chunk contributions summed" rule; denominator = whole-episode tokens ×
      distinct episode count). Optimizer = the same CPU-offloaded
      AdamWScheduleFree/build_optimizer path as the SFT step, same canonical
      checkpoint schema, one GPU.
  (2) IMPORTANCE RATIO, honestly: the bank's episodes are greedy-decoded and carry NO
      generation-time logprob trace (the recorded v1 gap), and superseded generating
      checkpoints are reclaimed after the grace window, so behavior-policy logprobs
      cannot be recomputed. v2 therefore treats every batch as on-policy-fresh:
      old_logp = the CURRENT weights' detached logprobs at each step start
      (ratio ≡ 1 by construction; single-use trajectories; the clip machinery is
      exercised but inactive at ratio 1 — the first-update gradient is exactly
      REINFORCE-with-group-baseline + k3-KL tether). Recorded in every step
      receipt as importance_ratio_mode="recomputed-on-policy". The measured
      training-layout vs generation-time logprob deviation (abs max 0.1198,
      p99 7.1e-4, the assay's failed teacher_max gate) is thereby never compared
      across layouts in v2.
  (3) k3-KL ANCHOR = the E3-u256 behavioral substrate 3770621c… (the DRAFT §1/§7.1
      substrate; checkpoint_agent_sft_u000128_loss_1.3049.pt, sha-verified on disk,
      weight_mode='saved'), FIXED for all lanes — a self-anchor (the trajectory-
      generating lineage) would give a zero-gradient KL term within a single step
      (k3 is stationary at new==ref), so the anchor must differ from the trained
      weights. The anchor is loaded ONLY for a no-grad reference-logprob pass and
      freed before the training forward (peak HBM stays ~SFT-step envelope + one
      row of activations).
  (4) FAIL-CLOSED: guards trip (nonfinite loss/grad/rewards, CE-consistency
      mismatch, or realized post-step k3-KL vs the pre-step weights above
      --kl-guard) => NO checkpoint is saved, the step exits nonzero, and the lane
      falls back to the receipts-SFT step for that cycle — the bank never loses a
      lane to the experimental train step.
  (5) A/B: flag-gated per-lane train-step selection (--lane-train-step
      policy-gradient|sft-receipts), DEFAULT sft-receipts (the seven untouched
      lanes' behavior is byte-identical); lane-05 (the lowest window pass-rate
      lane, 4/15 at build time) is the PG lane on relaunch.
- BATCH PHYSICS, honest: verified on the live bank — in the 4-bundle admitted lake
  the train direct-read (9f036dbb…) now passes every attempt and the opaque train
  bundle (ae2c8b34…) fails every attempt (unsatisfiable without a listing tool), so
  CURRENT bank windows have ZERO outcome variance in both train groups and the PG
  channel will honestly report no-contrast and fall back to SFT until a task flips,
  a merge resets competence, or a bigger gym lands. The QUALIFICATION batch has real
  contrast: v1 cycles 2-6 train-split attempts of 9f036dbb… (2 fails → 3 passes, the
  loop's measured on-policy learning progression) + ae2c8b34… (5 fails, zero-variance
  group, advantage 0 — exercised but signal-free). Dev-split attempts are NEVER
  batched (sealed train/development isolation).
- Building now: scripts/rl_pg_batch.py (digest-verified trajectory batch builder),
  scripts/rl_policy_gradient_step.py (the second train step), scripts/rl_pg_screen.py
  (pre-registered parameter screen), lane integration behind the flag, unit tests.
  GPU: ONE leased GPU (the bank's own broker; the bank's 8 lanes keep their leases —
  nothing about the live bank is touched by the build or the qualification).
- BUILT: scripts/rl_pg_batch.py (CPU-verified: 8 episodes / 14 turns from v1 cycles 2-6
  train-split attempts; group 9f036dbb 1F/3P contrast; group ae2c8b34 all-fail
  zero-variance; 2 stopped episodes honestly skipped by re-encode verification —
  mid-turn truncations absent from the canonical transcript; batch sha
  38fdf117f3c938fa907572f9404ea686cea8058ffecc757af2dcdc115983ecd9) and
  scripts/rl_policy_gradient_step.py.
- MEASURED (build-phase, one leased GPU, no lane touched): the chunked-CE
  torch.utils.checkpoint REPLAY is incompatible with the grad-mode head-hook
  capture — the non-reentrant checkpoint's saved-tensor consistency check
  rejects the hook's escaped logprob tensors ("10 saved vs 6 recomputed",
  CheckpointError).  Fix (recorded as an honest deviation in the precision
  policy + every receipt): CE-chunk replay DISABLED for the capture forward —
  identical chunked FP32-logit numerics, per-row CE graphs freed at each row's
  immediate backward; the SFT step's layer-group gradient checkpointing stays ON.
- V2 QUALIFICATION ATTEMPT 1 (fixed E3-u256 substrate anchor, lr 1e-5, kl_beta
  0.01): the FULL gradient path ran end-to-end for the first time in programme
  history — GRPO group advantages computed from the sealed-validator outcomes
  (fail episode -1.732, passes +0.577, zero-variance group 0.0 — the audited
  candidate's exact arithmetic), 14 rows captured through the qualified
  chunked-head FP32 path with per-row chunked-CE consistency gates passing,
  the candidate policy_loss summed per row, gradients accumulated, ONE REAL
  ADAMW OPTIMIZER STEP TAKEN — and the fail-closed guard TRIPPED exactly as
  designed: realized per-token k3-KL vs the pre-step weights 0.718 > guard 0.25,
  NO checkpoint saved, clean nonzero exit.  Evidence:
  smoke/pg-v2/qualification/log.jsonl.  This is the "first updates WILL
  surprise" the directive predicted, measured: one lr-1e-5 AdamW update moves
  the policy ~0.72 nats/token on the supervised span.
- KEY FINDING (changes the anchor choice, recorded as a measured negative
  result): with the E3-u256 substrate as the k3-KL anchor, the bank lineages
  sit ~3555 nats/token away in k3 terms — the loss was dominated by the anchor
  term ~10^6x (kl_part 4886 vs surrogate_part 0.0023; grad_norm 2.97e6
  pre-clip, almost entirely anchor-pull) — the update direction was effectively
  "undo the bank's SFT learning toward the substrate".  The k3 estimator is a
  local (quadratic) tether; a multi-generation anchor makes it an explosion.
  DECISION (honest, textbook-correct): anchor-mode "parent" is the v2 default —
  the reference IS the trajectory-generating policy == the checkpoint being
  trained (the bank's lanes generate each cycle's episodes from the same
  lineage they then train), i.e. the standard GRPO reference; the k3 term has
  exactly zero gradient on the first (production) update and tethers steps 2+
  in screen runs.  The fixed-substrate variant remains a screenable,
  explicitly-chosen option with the negative result recorded above.
- PRE-REGISTERED SCREEN (scripts/rl_pg_screen.py — register-before-run
  discipline): lr x kl_beta grid {1e-5, 2e-6, 1e-6, 5e-7} x {0.01, 0.1 at the
  two middle lrs}, anchor-mode parent, 3 updates per combo, KL guard DISARMED
  during the screen so the measured realized KL calibrates the production
  guard; selection rule frozen in the plan (S1 finite+CE-consistent; S2 max
  realized KL/update <= 0.05; S3 surrogate descends; S4 largest qualifying lr,
  larger kl_beta breaks ties; S5 production guard = 3x selected max realized
  KL rounded up; S6 honest fallback + rule miss).  NOT screened, recorded
  why: advantage normalization (the audited candidate has no toggle —
  screening an unaudited variant is out of scope), the fixed-substrate anchor
  (measured negative result above), the KL guard itself (calibrated BY the
  screen).  Screen runs use --skip-checkpoint-save (receipts carry the
  evidence; no 24GB screen checkpoints).

## OPERATOR REGIME CHANGE — FULL-RL BANK (2026-09-27, mid-screen steering)

- OPERATOR RULING (received mid-screen): "move into an RL regime; go aggressive" —
  the 1-lane A/B is WAIVED in favor of regime speed: if the screen shows sane
  gradients, switch ALL 8 LANES to the policy-gradient train step; keep only the
  three load-bearing guards (verified-only receipts, fail-closed NaN/KL fallback
  to the SFT step per lane, disk GC); the lake's new tasks flow at grading pace;
  compress the screen to the minimum needed to confirm sane gradients.
- SCREEN RESULT (smoke/pg-v2/screen/, one leased GPU, 6 pre-registered combos x
  3 updates, parent anchor, guard disarmed for measurement):
  * lr 1e-5: DISQUALIFIED — realized k3-KL 4.2e8, worst supervised token crushed
    ~27 nats in one update; the guard tripped (fail-closed, no checkpoint) —
    the k3 estimator's exp() term explodes on probability collapse; exactly
    the "first updates WILL surprise" case, and exactly what the guard exists
    for.  NOT a sane gradient.
  * lr 2e-6: SANE — realized KL mean 0.0184/update (2.7x under the 0.05
    ceiling), worst-token k3 4.39 stabilizing to 0.19 by update 2, grad_norm
    ~0.22 steady.  kl_beta 0.1 vs 0.01 indistinguishable at this scale.
  * lr 1e-6: SANE (0.00068/update).  lr 5e-7: SANE (0.00018/update).
  * PRE-REGISTERED RULE DEFECT, recorded honestly: S3 ("surrogate descends") is
    structurally vacuous under the recomputed-on-policy ratio — the surrogate
    VALUE is constant by construction; only its gradient carries signal.  The
    mechanical S6 fallback therefore picks lr 5e-7 (smallest movement);
    selection.json records that mechanically, and operator-selection.json
    records the operator-mandated aggressive pick that supersedes it.
- PRODUCTION PARAMETERS (operator-selection.json): lr 2e-6, kl_beta 0.01,
  kl_guard 0.056 (pre-registered S5: 3x the selected combo's max realized KL,
  rounded up), anchor-mode parent (textbook GRPO reference == the
  trajectory-generating policy), 1 update per cycle, batch = the lane's last
  16 cycles of train-split on-policy attempts capped at the 24 most recent
  episodes, GRPO groups keyed by SEALED TEMPLATE FAMILY (the lakeexp tranches
  are one-shot instances: per-task groups would all be singletons and the
  audited candidate fails closed on them; family instances share the task
  structure and the sealed validator program — the measured live window of
  lane-05 gives a 21-episode family group, 9 pass / 12 fail, real contrast).
- V2 QUALIFICATION COMPLETED (smoke/pg-v2/qualification-final/, one leased
  GPU): the FIRST COMPLETED GUARDED POLICY-GRADIENT UPDATE ever taken by this
  machinery — parent bec77602… -> checkpoint_agent_pg_u000001_loss_0.0023.pt
  (sha bff4adec0c9712ddb2220766e48b5ec97d625a3bc9d178e73217303ecbe42986),
  loss 0.0022693, grad_norm 0.2178, realized k3-KL vs pre-step weights 0.01842
  (< guard 0.056), all per-row chunked-CE consistency gates passed, canonical
  checkpoint schema (loads through the standard engine path), digest-linked
  PG train-event receipt dd6495fd… (chain genesis of the qualification
  stream).  Wall 3m53s.  The gradient path is verified END-TO-END: sealed
  validator outcomes -> GRPO group advantages -> qualified logprob capture ->
  clipped-surrogate + k3-KL loss -> AdamW step -> guarded checkpoint.
- UNIT VALIDATION now 50/50 (8 new CPU tests: PG batch family grouping +
  sealed split isolation, singleton semantics under per-task grouping,
  recency cap, selection-rule S2/S3/S4/S5 mechanics incl. the S3-vacuity
  path, digest-linked PG receipt chain + fail-closed digest drift, lane
  run-parser default UNCHANGED (sft-receipts) + policy-gradient flag parse).
  run_bank.sh preflight now requires 50 and supports LANES_ON_POLICY_GRADIENT
  (comma list of lanes to launch on the PG channel; default empty = unchanged).
- LANE INTEGRATION (scripts/rl_bank_lane.py, flag-gated per-lane
  --lane-train-step policy-gradient|sft-receipts, DEFAULT sft-receipts):
  the PG channel builds its trajectory batch from the lane's own recent
  train-split attempts (passes AND failures — the reward contrast), takes the
  guarded PG step, and on no-contrast batches or ANY fail-closed guard trip
  falls back to the receipts-SFT step for that cycle (LANE_PG_FALLBACK) —
  the bank never loses a lane.  Every PG update appends a digest-linked
  train-event receipt to the lane's training/pg-stream.jsonl (parameter set +
  loss + KL + advantage stats + parent/batch/checkpoint shas).  Lane state
  records train_step_channel + pg_parameters + pg_updates_total +
  pg_fallbacks_total (surfaces in every coordinator report).
- NEXT: graceful stop + relaunch of all 8 lanes with --lane-train-step
  policy-gradient (SIGTERM -> lanes finish in-flight cycles and exit at cycle
  boundaries; coordinator and run_bank.sh monitor untouched; bank/STOP never
  touched), then first-hours full-RL bank telemetry.

## THE FULL-RL BANK — SWITCH EXECUTED + FIRST-HOURS TELEMETRY (2026-09-27)

- SWITCH: all 8 lanes gracefully stopped at cycle boundaries (SIGTERM -> in-flight
  cycles completed, 7 trained + 1 collected, every GPU lease released, no state
  lost; bank/STOP never touched; coordinator + run_bank.sh monitor untouched)
  and relaunched with --lane-train-step policy-gradient.  Lane states now
  record train_step_channel=policy-gradient for all 8 lanes (the receipt:
  every lane.json carries the channel + the screened pg_parameters).
- FIRST-HOURS TELEMETRY (live, ~1.5h post-switch on the recalibrated geometry):
  * PG updates taken and CHAIN-VERIFIED (per-lane digest-linked
    training/pg-stream.jsonl): 17.  Guard-trip fallbacks to receipts-SFT: 47.
    The KL guard is pacing exactly as the operator kept it to: every PG
    update that fired moved 0.0086-0.0464 realized k3-KL/token (all under the
    0.056 guard); every batch that would have moved further was refused and
    the SFT step carried the cycle — the bank never lost a lane or a cycle.
  * PG grad norms: mean 0.135, max 0.188 (meaningful, steady, finite).
  * PRODUCTION-SCALE RECALIBRATION (measured, recorded in
    smoke/pg-v2/screen/operator-selection.json): the first full-RL cycles ran
    21-24-episode batches at lr 2e-6 and moved 0.17-0.99 realized KL/token
    (worst supervised tokens collapsed ~7 nats) — 8 of the first 10 PG
    attempts guard-tripped.  Diagnosis: the screen validated the 8-episode
    geometry; realized KL scales superlinearly with batch token mass.
    Fix: --pg-max-episodes 24 -> 8 (production batch = the SCREENED
    geometry); the guard was NOT raised — admitting 0.5-1.0/token steps
    would be corruption, not pace.  Post-recalibration the successful-update
    KL distribution sits at the intended 3x margin under the guard.
  * PG-era verified receipts (digest-chained episode stream): 14; pool
    drained from 372 pending lakeexp tasks to 57 (grading-paced, no injection
    throttle added — the pool contract is the only coordination).
  * Train-split policy pass rate in the PG-era window: 8/35 (22.9%) — small
    noisy sample; the honest verdict on whether PG updates lift the pass
    rate needs more hours (the lakeexp families give the reward signal real
    room to bite: fresh direct-read instances fail on-policy ~27%).
  * merges: epoch 1 still current; the K=32 triggers fire as PG/SFT update
    cadence accumulates (~22+/hour bank-wide) — the soup weights remain the
    honest long-run verdict between lineages.
- RECEIPT-CHAIN BUG found and fixed during telemetry verification (honest
  record): the PG receipt body's advantages.per_episode was emitted with INT
  keys; canonical_json sorted those NUMERICALLY at write time, but the
  round-tripped STRING keys sort LEXICOGRAPHICALLY — receipts from batches
  with >=10 episodes could not re-verify (the qualification batch's 8
  episodes coincided in both orders, masking it).  All 14 written receipts'
  digests were VALID (they bind the write-time bytes); verify_pg_stream now
  reproduces both serialization forms (recorded per receipt); new receipts
  emit string keys.  Regression tests added; unit validation now 52/52
  (run_bank.sh preflight updated).
- Unit validation: 52/52 (10 v2 tests: batch builder family/task grouping,
  sealed split isolation, singleton semantics, recency cap, selection-rule
  mechanics incl. the S3-vacuity path, receipt chain + digest-drift
  fail-closed + both serialization forms, lane flag default UNCHANGED +
  policy-gradient parse).
- The full-RL bank is RUNNING (indefinite; clean stop remains
  `touch bank/STOP`).  Artifacts: smoke/pg-v2/ (qualification batch +
  qualification-final first completed guarded update + screen), per-lane
  training/pg-stream.jsonl chains, per-cycle training/cycle-*/pg-log.jsonl.
- 2026-09-28 GPU DOC-NLL PROBE (operator-ordered fast relative anchor check).
  Built scripts/score_docnll_gpu.py: teacher-forced NLL of the SAME frozen
  8-doc panel (probe-panel.token-ids.jsonl sha 5725d820..., 198,295 scored
  tokens, 32,768/doc cap, document-causal reset per row, first token
  unsupervised) computed on ONE leased GPU through the existing loader
  (ndm.e97.load_e97_checkpoint, weight_mode='saved' — the same saved-eval-x
  weights the CPU chain converts to GGUF — dtype bf16 + bf16 autocast,
  use_triton, fp32 cross-entropy, double accumulation). HONEST INSTRUMENT
  FRAMING (in the script docstring and result JSON): GPU-bf16/fp32 readings
  are a NEW instrument series, NOT numerically identical to the CPU q8_0
  drift-table chain (quantization + accumulation differ); valid only for
  relative A-vs-B reads through the identical GPU path — which the operator
  ruled is the read that matters. The CPU q8_0 chain (e97-runner --score-nll)
  remains the archival absolute series; GPU values must not be mixed into it.
  READ (both models, one leased GPU, identical path; artifacts
  docnll-gpu-probe-v1/{docnll-gpu.json,baseline-gpu-series-seed.json,run.log}):
  * e3-u256 (sha 3770621c...): 8-doc mean NLL 2.556073 -> this is the GPU
    instrument's BASELINE SERIES SEED (baseline-gpu-series-seed.json).
    Cross-instrument sanity: CPU q8_0 reading for the same checkpoint was
    2.555940 (seg2-u256 juncture) — offset +1.3e-4, consistent with the
    different-instrument framing.
  * bank merge-0005 (sha 7479164b...): 8-doc mean NLL 2.556410.
  * DELTA merge-5 minus u256 = +0.000337 (per doc: gutenberg +3e-6,
    pre_1929_books +7.0e-4, arxiv +1.0e-4, caselaw -1.6e-4, ubuntu_irc
    +2.2e-3, libretexts +5.6e-4, news -3.2e-4, stackexchange -8.0e-4).
  Verdict: after ~20h of RL (1,154 updates, 8-lane weighted soup) the anchor
  is HELD to within +3.4e-4 nats (0.013% relative) — 3 of 8 docs improved,
  largest single-doc move +0.0022 (ubuntu_irc); no document-scale
  regression. The tiny positive mean is far inside the scale the CPU chain's
  own juncture comparisons treat as noise-adjacent (parent->u256 CPU spread
  was -6.9e-4), but the sign is honestly recorded as a (negligible)
  regression, not a hold, per the probe's own threshold convention.
