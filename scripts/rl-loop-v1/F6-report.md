# F6 — Batched multi-episode E97 serving: feasibility and staged design

Read-only study, 2026-10-08, lambda01. **No serving/training code, bank state, running process, GPU workload, teacher request, or scheduler submission was changed or launched.** The only deliverable is this report, with an identical repository copy at `scripts/rl-loop-v1/F6-report.md`. This is a design proposal, not implementation or GPU qualification.

## Decision summary

- **4 -> 8–16 episodes per GPU is memory-feasible**, with one model owner and independent session states. The sequential split-edit Triton kernel already has a batch axis; the serving adapters and scheduling do not.
- Important correction: the requested BF16 matrix calculation is **8,847,360 bytes/episode**, but **the actual eval carry is FP32: 17,694,720 bytes/episode**. Sixteen FP32 states occupy only **283,115,520 bytes (270 MiB)**. Persistent state is not the limiting resource; full-prompt logits, prefill workspaces, CPU/Pi work, and teacher capacity are.
- The bank does **not** call the OpenAI server or `TorchE97AgentEngine.generate()`. Its deployed `rl_loop_driver.make_policy_generate()` uses the engine as a loaded-model holder, replays the complete native prompt every turn, and directly calls the scalar cache functions. Changing only `e97_agent_server.py` would not accelerate collection.
- Current sampled collection is **89.09% correction-episode wall time**, not demonstrably dominated by bash/file waits. Tool-wait interleaving is architecturally valuable, but do not assume a large tool-wait multiplier without timestamps. Asynchronous teacher correction is the first lever.
- On seven collectors plus one learner, plan for **700–1,000 completed task attempts/hour and 105k–180k accepted target-tokens/hour** after batching, bounded teacher concurrency, and a qualified short-task diet. These are conditional planning targets, not measurements. A sustained 100x gain over the sampled single-learner end-to-end baseline is near the upper end, not guaranteed; 100x over eight continuously collecting cards is unsupported.

## 1. Source identity and actual serving chain

Repository citations below are relative to `/home/erikg/emender`. `D` denotes the deployed scripts directory `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts`; `W` denotes its parent. The driver is **not tracked at `scripts/rl_loop_driver.py` in this checkout**; citations to `D/rl_loop_driver.py` intentionally name the code the bank imports. SHA-256 of that file: `da9c5f79d3584c69e933c04e8b4719c792c1ed2b4d6fae2e22ede4e19af00821`. Repository HEAD at initial inspection: `c83b01d1`; no source edits were made during this study.

| Seam | Observed behavior and line-number evidence |
|---|---|
| Bank cycle | `scripts/rl-loop-v1/rl_bank_lane.py:183–185` loads one policy engine per collect subprocess; `:222–245` runs claimed tasks **serially**; `:247–258` grades each attempt before proceeding. `:656–666` spawns that collect subprocess and waits; `:681–684` then runs ordered train/probe/adopt channels. |
| Real engine | `D/rl_loop_driver.py:130–146` loads saved weights, CUDA BF16, Triton, mmap, and instantiates `TorchE97AgentEngine(ingest_mode="segment")`. `ndm/e97.py:393–411` drops checkpoint/optimizer mappings before device transfer, enables fused inference, and sets eval. |
| Real bank generation | `D/rl_loop_driver.py:149–206`: full prompt encode at `:164`; **fresh full-prefix segment ingest with no prior cache** at `:170`; per-token generation at `:171–177`; greedy temperature 0; RS=218; native five-line validation at `:182–198`; deadline checked once per decoded token. The closure's `state` is aggregate counters, not an episode cache. |
| Engine API, not bank path | `ndm/e97_agent_server.py:205–215` selects tokenwise/segment ingest; `:217–245` generates one token at a time, stopping on a different structured-turn protocol. Reusing this stop parser for bank native frames without an adapter would be incorrect. |
| Cache adapters | `ndm/e97.py:591–628` makes tokens `[1,T]` and takes `logits[0,-1]`; `:632–682` makes `[1,1]` calls; `:686–719` samples one sequence and advances one token. `_sample_token`, `:449–471`, has scalar `.item()`/global RNG semantics. These are not batch APIs. |
| Prompt -> action -> result | `D/rl_loop_driver.py:313–367` creates one bridge and synchronously runs its Pi transport. `scripts/e97_pi_native_tool_bridge.py:43–51` verifies and ingests the exact matching tool result; `:60–76` builds the next prompt, synchronously generates, validates, and returns a tool action or final answer. Internal think actions loop at `:67–70`. |
| Serial tool waits | `scripts/e97_pi_native_tool_transport.py:28–59` runs one Pi subprocess and blocks in `select`, socket accept/read, and `bridge.next(req)`; Pi executes tools before the next request. `:60–66` bounds subprocess exit/termination. Independent bridges can interleave; one episode's dependencies cannot be reordered. |
| Native transcript | `scripts/e97_pi_native_codec.py:109–139` appends exact context and assistant text with native delimiters; `:128–133` shows the generation prompt prefix. This serialization is **not** OpenAI chat serialization. |
| Existing multi-session store | `ndm/e97_agent_server.py:92–152` already has an LRU, default eight sessions, prefix checks and versioned compare-and-commit. But `:112–133` holds its global lock while advancing the model. `:665–676` binds session headers, and `:718–721` commits only delivered responses. `:730–747` uses a single-request-at-a-time `HTTPServer`; SSE is constructed after completion (`:699–711`), not live decode. Sixteen session IDs alone do not create batching. |

Selected source hashes: `ndm/e97.py` = `f55787cf871237856552924229d5edbd91e0543c2a3659a006949dffecfe4c96`; `ndm/e97_agent_server.py` = `1992ddb1772d8368f137dd9b37960e5f931bcb5b3240a225d218f6ef6248e007`; `ndm/triton/e88_triton_forward.py` = `8aa17795f582d8065dbf921057da25fe5f43f02bdc709aedc4afe6ffe1b861f5`; `scripts/rl-loop-v1/rl_bank_lane.py` = `9490b40f6fd5e4b6acf469d381eb689d91f902c6d10442873de722e85a5f395e`.

## 2. Exact observable timing, and the instrumentation gap

Evidence cohort: **completed single-learner-B lane-00 cycles 0047–0056**, all four attempts per cycle, with the start of 0057 used only as the end-to-end boundary. These are archived reads, not timing experiments. Let `E = W/bank-singlelearner-B/lanes/lane-00/episodes`.

For each immutable `episode-private.json`, derive the interval as **file `st_mtime` minus recorded `started_unix`**; derive collect span as summary `st_mtime` minus the earliest attempt start. This includes transport startup, generation, tools, shutdown and episode serialization. It is exact arithmetic on recorded/filesystem timestamps, **not an independently instrumented GPU duration**; file write time precedes completion of publication, clock adjustments/copying can invalidate this method, and sub-millisecond precision is not physical accuracy. Files were read in place, not copied/restored by this task. Source of start/write: `D/rl_loop_driver.py:335–339,370–385`.

| Cycle | Attempts | Receipts | Attempt intervals, s | Correction intervals, s | API latency sum, s | Collect span, s | Accepted targets |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0047 | 4 | 4 | 32.765764 | 355.464961 | 353.025 | 389.103930 | 1,160 |
| 0048 | 4 | 3 | 48.771839 | 668.858969 | 647.060 | 718.503675 | 684 |
| 0049 | 4 | 4 | 53.349632 | 397.395722 | 375.066 | 451.647983 | 958 |
| 0050 | 4 | 4 | 32.709719 | 0 | 0 | 33.230680 | 679 |
| 0051 | 4 | 3 | 69.692527 | 594.122257 | 571.649 | 664.704763 | 624 |
| 0052 | 4 | 3 | 69.256486 | 581.417835 | 558.263 | 651.685328 | 692 |
| 0053 | 4 | 4 | 27.656824 | 77.926030 | 75.555 | 106.440879 | 1,111 |
| 0054 | 4 | 3 | 45.646208 | 74.734622 | 72.210 | 121.281955 | 713 |
| 0055 | 4 | 3 | 36.348446 | 273.451580 | 271.791 | 310.596606 | 758 |
| 0056 | 4 | 1 | 51.589401 | 866.336377 | 823.682 | 918.805773 | 211 |
| **Total** | **40** | **32** | **467.786846** | **3,889.708354** | **3,748.301** | **4,366.001572** | **7,590** |

Per task attempt, including expected correction: **109.150039 s = 11.694671 policy + 97.242709 correction + 0.212659 residual**. Policy intervals are 10.7143% of collect span; corrections 89.0909%; residual 0.1948%. The 26 corrections average 149.604167 s (median 55.097594; range 4.781507–585.164960). Policy median 7.808810 s (range 3.353958–31.394014). API records contain 86 calls/attempts, six errors; mean latency 43.584895 s, median 12.5475, maximum 241.482. API duration is recorded to 0.001 s; it is a **subset** of correction time, not an additional serial phase. API retries/sleeps and Pi startup/tools explain some residual correction time, but it is not fully disaggregated.

Cycle 0056 per-task details, in actual serial order:

| Task | Policy, s | Correction, s |
|---|---:|---:|
| tas-58c52f806f3488a0 | 10.349836 | 442.531597 |
| tas-6bfe67a7d09d9724 | 6.421541 | 393.086763 |
| tas-6c03b1aa14eef34b | 27.208358 | 30.718017 |
| tas-6c076d18d47d34d7 | 7.609666 | none |

`E/cycle-0056/collect-summary.json` SHA-256: `1292136e6038c3be9e6ea613bb828c3c5ca9bc531da5a61beaf7a0b2c12161c7`. The corresponding cycle log `W/bank-singlelearner-B/lanes/lane-00/logs/cycle-0056.log:2–4,23–34` confirms load/task order, failures, correction order and counts. `:36` starts the separate PG channel. **No per-phase collect subprocess timing lines exist in these sampled logs**; `_spawn` (`rl_bank_lane.py:518–528`) logs child stdout but no successful elapsed time. A broader read of archived `.log` files did not find the promised fine-grained timing lines. Do not fabricate them.

The cohort contains **13,999 generated policy tokens / 95 policy turns** and **7,652 canonical correction tokens / 71 correction turns**. Teacher API usage is 481,280 prompt tokens and 47,110 completion tokens (including reasoning/invalid replies), **not** 47,110 supervised targets. Accepted targets average **189.75/task attempt**, receipt yield 80%. Collect-only rates: **32.982 attempts/h, 26.386 receipts/h, 6,258 accepted targets/h**, and 11,543 raw policy tokens/h. During policy intervals the effective rate is 29.926 generated tokens/s, including prefill, tools and startup; it is not pure decode throughput.

End-to-end boundary: first attempt of 0047 at Unix `1791468134.1972053` to first attempt of 0057 at `1791484564.4340456`, **16,430.236840 s**. Thus **8.764 attempts/h and 1,663 accepted targets/h** for the single-learner cycle loop, active collect fraction **26.573%**. The other **12,064.235268 s** includes training, probes, checkpoint/load/lease/poll/queue gaps; the evidence cannot assign exact time to each. Model load before the first attempt is excluded from collect-only spans but included in this aggregate gap.

**Unavailable exact phases:** prompt encode/build, prefill CUDA time, per-token decode CUDA time, validator time, individual bash/read/write waits, grading, load, and lease wait. Pi event timestamps do not fix this: `message_start` and `message_end` reuse message timestamps, and assistant timestamps can precede generation. Example old-bank `W/bank/lanes/lane-00/episodes/cycle-1150/bank-r3658-9f036dbba4b8/attempt/pi/pi-events-private.jsonl:6,10,13,17,21` has that pattern. Treat intervals between those timestamps as composites, never pure tool time. Stage 1 must add paired monotonic phase events and CUDA events with causal session/turn IDs before claiming any tool-wait or prefill multiplier.

### Reproduce the archived timing arithmetic (read-only, CPU stdlib)

Run on lambda01 with `/home/erikg/emender/.venv/bin/python`; no torch or GPU import:

```python
import json, pathlib
E = pathlib.Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/'
                 'e97-rl-loop-v1/bank-singlelearner-B/lanes/lane-00/episodes')
starts = []; totals = dict(attempts=0, receipts=0, targets=0, A=0., C=0., span=0., api=0.)
for c in range(47, 58):
    p = E / f'cycle-{c:04d}'
    summary = json.loads((p/'collect-summary.json').read_text())
    rows = []
    for o in summary['outcomes']:
        for stage in ('attempt', 'correction'):
            f = p/o['task_id']/stage/'episode-private.json'
            if f.exists():
                d = json.loads(f.read_text())
                rows.append((stage, d['started_unix'], f.stat().st_mtime-d['started_unix']))
    start = min(x[1] for x in rows if x[0] == 'attempt'); starts.append(start)
    if c == 57: break
    a = sum(x[2] for x in rows if x[0] == 'attempt')
    corr = sum(x[2] for x in rows if x[0] == 'correction')
    span = (p/'collect-summary.json').stat().st_mtime-start
    m = p/'teacher-metrics.json'
    calls = json.loads(m.read_text())['calls'] if m.exists() else []
    api = sum(x['latency_s'] for x in calls)
    targets = sum(o.get('targets', 0) for o in summary['outcomes'])
    print(c, summary['attempts'], summary['receipts'], a, corr, api, span, targets)
    for key, val in dict(attempts=summary['attempts'], receipts=summary['receipts'],
                         targets=targets, A=a, C=corr, span=span, api=api).items():
        totals[key] += val
print(totals, 'end_to_end_seconds', starts[-1]-starts[0])
```

## 3. Exact state sizing and the real memory constraint

Config `A = /mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json`, bound by `scripts/rl-loop-v1/run_bank.sh:29,131`. `A:4–16` records p50k_base, dim 3840, depth 18, expansion 1.0, n_state 64, n_heads 60. `ndm/e97.py:134–168` forwards these values. `ndm/models/e88_fla_hybrid.py:1008–1011` computes value dimension, so **head_v_dim = 64**, not `state_expansion * 64`; that latter field is not a multiplier on this E97 matrix. `use_conv=0`, dropout=0 and nonlinear/non-chunked state are also recorded in A.

Matrix elements per episode: **60 × 64 × 64 × 18 = 4,423,680**. A records 4,045,972,080 total parameters: their ideal BF16 weight payload is **8,091,944,160 bytes**, distinct from the supplied ~18.5 GB process footprint (which includes other allocations/reservations). No model or checkpoint tensor payload was loaded by this study.

| Sessions/GPU | Requested BF16 matrix bytes | Actual FP32 matrix bytes | Actual FP32 matrix MiB |
|---:|---:|---:|---:|
| 1 | 8,847,360 | 17,694,720 | 16.875 |
| 4 | 35,389,440 | 70,778,880 | 67.5 |
| 8 | 70,778,880 | 141,557,760 | 135 |
| 16 | 141,557,760 | 283,115,520 | 270 |

**Why FP32:** eval mode is set by `ndm/e97.py:411`; `ndm/models/e88_fla_hybrid.py:1839–1857` explicitly keeps FP32 state when not training. Per-head hidden views (`:2165`) partition the returned state, not duplicate all heads. `ndm/triton/e88_triton_forward.py:630–641` allocates `S_final` with the input state's dtype, although legacy temporary replay checkpoints can be BF16. BF16 model projections do not imply BF16 carry. Do not halve the carry precision as an unqualified optimization.

With compact BF16 boundary logits for p50k_base vocab 50,281, add **100,562 bytes/session**. Logical cache tensors become **17,795,282 bytes/session**; totals 4/8/16 = **71,181,128 / 142,362,256 / 284,724,512 bytes**. `ndm/e97.py:175–193` derives vocab from checkpoint embedding or tokenizer, and `:530–542` defines logical `state_bytes`. This excludes Python token tuples, transcripts, Pi processes, tensor metadata/allocator rounding, weights and transient workspaces. Confirm embedding vocabulary and logits dtype in the actual loaded checkpoint at qualification; the matrix count above is independent of vocabulary.

Assuming the task's **~18.5 decimal GB single-serve footprint** and including its one existing logical cache, extrapolation for eight/sixteen compact caches is **~18.625 / ~18.767 GB**, leaving ~29.375/~29.233 GB against a decimal 48 GB budget **before extra batched workspaces**. If 18.5 is GiB rather than GB, recompute in the same units. This baseline is supplied context, **not a new measured CUDA peak**. Sixteen committed + sixteen shadow caches take ~0.569 GB; budget gather/scatter input/output state and one extra transient cohort as well. Persistent slots and cache versions must have finite admission limits.

### Do not confuse compact cache bytes with prefill peak

The current segment path materializes **all `[B,T,Vocab]` logits** (`ndm/models/ladder_lm.py:1641–1643`). `ndm/e97.py:621` takes a detached **view**, not a clone: a final-row view can retain the entire prefill logits backing storage until replacement. For 16 sequences with BF16 logits:

- T=512: **823,803,904 bytes (0.824 GB)**;
- T=6,144: **9,885,646,848 bytes (9.886 GB)**;
- T=65,536: **105,446,899,712 bytes (105.447 GB)** — impossible on 48 GB even before weights.

The recurrence additionally allocates sparse checkpoint workspace (`e88_triton_forward.py:636–641`): for T=512, B=16, 60 heads and BF16 checkpoint dtype, **33 × 16 × 60 × 64² × 2 = 259,522,560 bytes per active layer call**, separate from the persistent carry. No-grad does not mean this temporary allocation disappears. Model projection/MLP activations add more. The projection chunk setting alone does not eliminate final full-sequence logits.

Design: **decode B=4/8/16, prefill initially one full prompt at a time** to preserve the existing full-segment reference and cap peak memory. Clone retained boundary logits into independent storage. Subsequently qualify bounded prefill token budgets (e.g. B×T ≤ 8,192 with ≤512-token segments), or a model inference option that computes only each row's last valid logits. Chunking changes BF16 segment semantics and must not be silently substituted for current full-prefix replay. Keep peak allocation/reservation and backing-storage bytes in the stage gates; admit no new batch above a tested ceiling (suggest ≤40 GB decimal, leaving an 8 GB safety margin).

## 4. Kernel feasibility and what cannot batch as-is

**Inference does have B>1 support.** `ndm/triton/e97_sequential.py:15–38,54–78` is a split-edit facade over the shared E88 optimized engine. `ndm/triton/e88_triton_optimized.py:55–60` takes `[B,T,H,*]`; `:65–95` explicitly pads unaligned **eval** lengths; `:108–115` transposes to `[T,B,H,*]`. `ndm/triton/e88_triton_forward.py:125–128,708–724` launches `(B, ceil(H/BLOCK_H))`; the program separately loads each batch row. This is not a training-only batch axis.

The loaded eval path permits optimized fused inference (`ndm/models/e88_fla_hybrid.py:1661–1668,1914–1928`). Long prefills use the existing projection-chunk loop (`:1961–1993`), whose `_process_chunk` has explicit `[B,C,*]` shapes and forwards both masks through the split-edit facade (`:1521–1545,1569–1582`); it has no B=1 restriction. This projection chunking is distinct from the linear-state parallel recurrence backend. Both model (`ladder_lm.py:1419–1436,1514–1516`) and kernel (`e88_triton_forward.py:198–208,262,280`) accept token-aligned `valid_mask` and `reset_before`. The recurrence invalid positions are strict no-ops; final capture (`:289–300`) uses global valid length, which is sufficient for right-padded rows **only because per-row invalid tokens preserve state**. `valid_length` alone is not a per-sequence length vector. Must select `logits[b,length[b]-1]`, never `logits[b,-1]` for padded prompts. This is source feasibility, not measured batched decode parity.

| Cannot batch as-is | Specific workaround |
|---|---|
| Scalar adapters and scalar sample `.item()` | Add explicit batched cache advance/decode APIs around `loaded.model([B,1], prev_hiddens=...)`; sample per row, one vector transfer for selected tokens, row-local stop/deadline/budget/parser. Keep old scalar API as reference. |
| Ragged prompts/full logits | Initially serialize independent full-prompt prefill, then batch ready decode rows. Qualify mask-aware length buckets or bounded segmented prefill plus last-valid-logit selection. Never treat padding token IDs as inert without `valid_mask`. |
| Different episode positions | No transformer KV/absolute-position offset is required for this audited dense E97 recurrence: position is carried by its state. Gather state by slot/row mapping; compact active rows or mask frozen rows; keep row-specific token lineage and history. A finished/tool-waiting row must not advance while another row continues. |
| Cache history and checkpoint ownership | Current cache is one tuple/list tree, one vector and checkpoint path, not N independent histories. Slot pool keeps per-session cache records; pack each layer/head along B; scatter independent outputs back. Check checkpoint **digest/generation**, not path alone (current path check `e97.py:577–579,700–701`). Never share mutable cache views across sessions. |
| Global lock during model advance | Split session metadata/version preparation from GPU execution; enqueue work outside the store lock and compare-and-commit on completion. One outstanding turn per session; stale retries rejected or replayed from unchanged committed state. |
| Single HTTP handler/complete-then-SSE | Add concurrent request intake feeding one bounded GPU scheduler; HTTP threads must not call CUDA concurrently. Existing OpenAI compatibility is optional; bank requires a **native-frame** endpoint/adapter with its existing validator. Do not replace bank canonicalization with OpenAI chat serialization. |
| Serial bank/Pi closures | Independent Pi transports retain blocking local calls, but those calls submit to a shared scheduler and await row-specific futures. Phase 3 gives each claimed episode its own bridge, workspace, deadline, socket and state. Only one GPU executor owns the model. |
| Teacher closure mutable state | Build one teacher generator per correction episode; shared metrics is lock-protected but closure `state['raw']` etc. is not per-episode isolated. Bound independent I/O workers, global rate/concurrency budget, deadlines and cancellation. |
| Chunked/fast-step substitutes | This config has `linear_state=0,use_chunked_e97=0`. The chunked backend requires linear split-edit (`e88_fla_hybrid.py:967–970`); it is not a drop-in associative scan for tanh. Experimental step-kernel precision warnings at `:1859–1865` are not proof of E97 end-to-end parity. Retain the sequential kernel. |
| Convolution state | `ndm/e97.py:580–586` rejects cache inference with convolution. Config has use_conv=0; do not generalize this design to conv configs without carrying their buffers. |

### Segment caches: feasible to isolate, not numerically equivalent by assumption

Sixteen episodes can share one model/engine and own sixteen segment caches; no kernel global state inherently prevents this. However, **the bank currently discards the cache after every turn**. Reuse must first verify that the reconstructed native token prefix is append-only (`ndm/e97.py:549–565`), including BPE boundary behavior, raw assistant text, native delimiters, internal-think results and exact tool results. A mismatch means replay, not a guessed suffix.

`advance_e97_cache_segment` documents BF16 segment-boundary rounding dependence (`e97.py:596–600`). Even an exact append-only prefix does not prove that suffix-ingest state matches current full-prefix replay: GEMM shapes have changed. Tokenwise canonical ingest (`:650–655`) is the library's correctness authority, but switching the bank to tokenwise prefill can be slow and changes its current reference semantics. **Stage 1 preserves full-prefix segment replay; stage 2 treats suffix caching as a separately qualified option, not a prerequisite for B=16 decode.** Reject speculative claims of a replay-elimination multiplier until state/logit/action parity is tested.

## 5. One server, N sessions, interleaved tool scheduling

Proposed protocol for the native bank adapter (not implemented): `open(checkpoint_sha, task_id, episode_id, tool_schema_sha, budgets)` -> session ID; `next(session_id, request_id, expected_version, canonical_prompt, deadline)` -> exact native frame/token IDs/stop reason/checkpoint identity; explicit accepted-delivery commit/version advance; `close/cancel`. Retries are idempotent by request ID; at most one turn in flight per session. Pin each episode to immutable saved checkpoint weights. Do not update the model in place during a decode or reinterpret collector seed provenance as the learner's current policy.

Lifecycle: `PREFILL_READY -> DECODING -> RESPONSE_READY -> WAIT_TOOL -> PREFILL_READY`, or `FINISHED/FAILED/CANCELLED`. A GPU scheduler owns a bounded slot pool, builds one microbatch from ready rows, samples and advances one token per active row, validates each row's growing native frame, and releases that row as soon as its frame completes. Bound queue length and batch formation delay (initial design: ≤2 ms when other rows are ready; never wait indefinitely to fill a batch), and apply age/fairness limits to prevent long prefills starving decode. Tools and receipt publication run outside the GPU executor.

Example: A emits a bash call and enters `WAIT_TOOL`; B–P continue decode. On A's tool completion, its verified result is appended and A is queued for prefill/replay, without blocking B–P. B's shorter frame may stop while C's longer frame continues. Compact the active state rows, or use invalid-row masks with retained boundary logits; never zero/reset a waiting row accidentally. There is **no benefit from batching a single dependency chain's future turns**. Tool latency can be hidden, not removed.

Transaction safety: committed cache remains immutable while shadow state is generated; only a validated, accepted/delivered frame commits. Cancellation/disconnect discards the shadow. Do not evict in-flight slots under the existing eight-session LRU; reserve capacity for 4 then 16 admitted episodes and handle exhaustion explicitly. Distinct episode sockets must retain their Pi peer UID/PID verification and exact result matching. Out-of-order episode completion feeds a single lane receipt writer that calculates the current stream tail at publication time, not multiple workers carrying the same old `prev_digest`. Claim heartbeats and crash outcomes remain per task. Preserve sealed train/development eligibility and all grading/native-close gates.

## 6. Actual GLM correction blockers and bounded async design

Bank correction, not standing supply: `rl_bank_lane.py:310–316` sleeps for a lane-local minimum interval, then `:331–353` runs a fresh or spliced teacher episode and `:355–364` grades it **before the next policy attempt**. `D/rl_loop_driver.py:209–215` delegates live generation to the pilot. `scripts/collect_e97_teacher_generation_pilot.py:110–128` performs blocking urllib HTTP calls, a 900 s per-call timeout, four transport attempts and 20/40/60/60 s retry sleeps; `:166–185` can retry invalid frames three times. The passed episode deadline is not a hard bound inside `lunaroute_call`. An API timeout can exceed the nominal teacher episode budget (480 s in `D/rl_loop_driver.py:83–89`). This tail behavior must be fixed in the proposed I/O adapter, not hidden by the average.

Async design: after attempt and sealed grade publication, enqueue an immutable correction specification (task/checkpoint/grade/transcript digests, fresh/spliced decision, workspace identity) and immediately allow another policy episode. Each correction owns a separate Pi process and generator; its tool steps remain serial. Use a bounded thread/async I/O pool, shared semaphore and service-wide rate policy. Keep the current min-start-interval semantics at the lane admission boundary unless an operator explicitly approves changing quotas; concurrency is not permission to remove throttles. Supply/judge calls consume the same global API budget. Enforce remaining-deadline timeouts, bounded retries, cleanup and explicit failed outcomes. One publisher validates and appends completed receipts; late/cancelled failures publish no training receipt. Do not retire a failed policy task as successfully corrected before correction grading. The engine need not participate in teacher generation.

## 7. Staged implementation and acceptance, one worker per stage

Calendar estimates are **sequential single-worker engineering estimates** (roughly 6 productive hours/day), not launch reservations. GPU-hour estimates count proposed future qualification/debug time, not work run by this study. All jobs/relaunches require separate operator approval. CPU tests can use small/fake engines first; real 4B parity is a required GPU gate.

| Stage | Scope and proposed gates | One-worker estimate | Proposed GPU-hours |
|---|---|---:|---:|
| **1: four episodes/GPU** | Native-compatible scheduler and batched `[B,1]` decode, isolated FP32 states, scalar reference retained, independent serial full-segment prefills, compact boundary logits. Four independent bridged episodes and per-row stop/budget/version control. Test equal/ragged lengths, slot permutation, finish/tool-wait masks, RS/invalid frame, retries/disconnect/cancel, session leak and immutable committed state. Instrument causal phase timings, peak/backing storage and B=1/B=4 performance. | **4–6 workdays** | **8–16** (one card, short controlled sessions) |
| **2: eight–sixteen + teacher pipeline** | Grow bounded capacity; qualify B=8 and B=16, prefill limits and compact/gather overhead. Add bounded asynchronous correction/judge handling, global rate limits and deadline-aware retry. Separately qualify optional suffix/segmented ingest; leave replay fallback. Soak 8/16 with tool stalls, teacher failures and capacity exhaustion; report max/p99 turn waits, targets/h and memory. | **5–8 workdays** | **16–32** (mostly one card, limited multi-card integration) |
| **3: lanes become multiplexers** | Replace serial `for task` with N live episode contexts, one model owner/GPU, session routing and fair ready queue. Single-writer receipt/retirement path; independent heartbeat/timeout/recovery; immutable collector checkpoint rollover only at session boundaries. Connect F4 cross-stream learner selection and F5 checkpoint provenance without changing learning arithmetic. Full factory read: seven collectors + learner, plus eight-collector serving-only comparison. | **5–8 workdays** | **24–48** (includes short 8-card soak, not eight cards for days) |
| **Total** | No extra worker hidden in calendar estimate | **14–22 workdays (~3–5 weeks)** | **48–96** |

### Bit-exactness expectations

State isolation, input tokens, masks, checkpoint identity, versioning, stop handling and protocol outcomes must be exact. **BF16 batched inference is not promised bit-exact to B=1**: changing GEMM batch/time shapes and kernel selection can change rounding; tanh recurrence can amplify small differences; near-tied greedy logits can change actions. Config dropout=0 and temperature=0 remove sampling/dropout randomness, not floating-point shape dependence. Pin kernels/runtime/numerical policy and test row permutations/repeated runs. If sampling is enabled later, use per-session RNG streams, not interleaving-dependent global RNG.

For fixed teacher-forced token streams compare every layer's state and logits, maximum absolute/relative errors, top-1 mismatch rate and margins; for free generation compare exact actions, stop reasons, sealed grades and task yield on a frozen corpus including long trajectories. Publish errors and mismatch examples, not just average CE. No numeric tolerance is pre-approved by this study; select it from observed B=1 repeatability and reference distributions for reviewer/operator approval. A bit-exact claim requires actually proving it. Preserve a scalar fallback and do not declare quality equivalence when action divergences occur. If the operator requires strict scalar token parity, keep shape-invariant/scalar projection or re-evaluate ambiguous rows with the scalar reference, measure the cost, and qualify the recurrence too; neither workaround is automatically a correctness proof.

## 8. Pipelining quick wins without engine work

These require orchestration/I/O changes and approval, **not edits in this task**. Multipliers are conditional estimates with explicit denominators; they cannot simply be multiplied together.

1. **Asynchronous correction across independent episodes, scalar GPU owner unchanged.** Ideal local collect bound: `4366.001572 / (467.786846 + 8.506372) = 9.167x` if all teacher work is off the policy critical path and capacity is unlimited. Eight API workers at observed workload cap final output at `8*3600/(3748.301/40) = 307.34 tasks/h`, roughly the same as the ~302 tasks/h scalar policy-side ceiling; practical local gain **3–6x**, not guaranteed 9x. Begin with a small bounded pool, don't invoke the shared engine from teacher threads. Teacher parallelism must apply across episodes, not dependent turns.
2. **Cycle overlap / independent collector lanes feeding one learner (F4).** Sampled loop spends only 26.573% in active collect. An independent collector not waiting for learner train/probe can theoretically recover **3.763x** versus that loop, before load/lease contention. Budget **1.5–3x** practical on the same serving work. Launch seven collector lanes plus one learner on eight cards only after resource approval; multiplying by seven is a card-allocation change, not a batching speedup. Do not overlap mutation and generation of the same live model; immutable seed collectors/current-checkpoint boundaries are mandatory. Increasing max-tasks can amortize model load but its actual multiplier is unknown without load timestamps.
3. **Diet toward real short first-action tasks, preserving complete verification.** Bias selection toward existing tasks solvable by one grounded tool action and finish, not synthetic answers or arbitrary transcript truncation. Stage an explicit mixture that retains hard/recovery and development measurements. The sample policy has 95/40=2.375 turns per task and teacher has 71/26=2.731 turns per correction; reducing to about two turns gives only **1.19x policy / 1.37x teacher turn-count leverage** on this already short cohort. Reducing invalid/overlong/retry incidence may raise episodes/h to **1.2–2x**, but is unmeasured. Accepted targets/episode may fall, so target-tokens/h gain could be **0.8–1.5x**, not 2x automatically. Track first-action sealed correctness and retained hard-task outcomes; do not optimize only receipt count.

This week's feasible package is bounded teacher I/O plus independently authorized F4 collectors and a small, explicit short-task mix, using existing scalar generation. Together plan **3–8x local collected output** before engine work, strongly dependent on API capacity; seven collectors can additionally use previously idle cards. Teacher async and cycle overlap recover overlapping idle time, so `9.17*3.76*diet` is not an honest estimate.

## 9. Honest eight-card 100x arithmetic

Definitions: **episode/hour below means completed claimed-task attempt with correction outcome finalized if needed**, not counting an attempt and its correction as two successes. **Target-tokens/hour means sealed eligible accepted supervised targets** (`outcomes[].targets`), not prompt tokens, teacher reasoning tokens, generated tokens or training exposure. Raw unsuccessful policy tokens are useful PG observations but not accepted SFT targets. Assume seven serving GPUs and one learner; an eight-serving-GPU-only test excludes learner resource cost.

### Capacity model, not a benchmark

Scalar policy-side ceiling from this cohort is about **302 tasks/h/GPU** if correction waits are removed. Illustrative incremental decode throughput design goals versus that active scalar path: B=4 **2–4x**, B=8 **4–7x**, B=16 **6–10x**. These are hypotheses requiring measurement, not kernel-derived promises; prefill, CPU validation and active occupancy can dominate. B=16 is not inherently 16x, and removing tool idle is already partly included in these integrated goals. Seven scalar collectors could already offer ~2,116 policy attempts/h before teacher bottlenecks; batching creates headroom for longer tasks but cannot multiply finished corrections beyond API capacity.

At the observed **2.15 API calls/task, 43.584895 s/call**, average API occupancy is **93.707525 worker-seconds/task**. Optimistic capacities (no contention/tails, enough ready work):

| Global simultaneous API calls | Final tasks/h ceiling | Accepted targets/h at observed 189.75/task |
|---:|---:|---:|
| 8 | 307 | 58,318 |
| 16 | 615 | 116,635 |
| 32 | 1,229 | 233,270 |

These are stronger bottlenecks than small-cache memory. Per-lane ten-second teacher-start admission intervals alone also limit scaling if unchanged; the global quota must explicitly allow the proposal. Long retries, failures, supply/judge contention and end-of-window drains reduce these ceilings.

**Conditional integrated target after batching + pipelining + diet:** if a qualified short-task mix lowers API occupancy to about **37.5 s/task** (illustratively 1.5 calls/task × 25 s mean), 16 global calls can support 1,536 tasks/h ideally. At 45–65% effective utilization after tails, tool/CPU/queue/yield overhead, aim **700–1,000 tasks/h**, with **150–180 accepted targets/task**, giving **105k–180k targets/h**. Raw policy generation, assuming 150–300 emitted tokens/task for that diet, is **105k–300k policy tokens/h**. Assumptions about occupancy, shortening and target yield must be measured; without them, use the observed-workload 307/615/1,229 ceilings instead. Initial conservative operation at eight teacher calls is closer to **200–300 tasks/h, 38k–57k targets/h** on the original mix, regardless of having B=16.

### Which baseline can be 100x?

- Sampled **single learner end-to-end**: 1,663 targets/h, 8.764 tasks/h. 100x means **166,303 targets/h and 876 tasks/h** (different metrics; shorter tasks may not preserve both). The proposed 105k–180k range is **63–108x targets** and 700–1,000 tasks/h is **80–114x attempts**. Thus 100x is a plausible upper planning goal **only** when recovering idle/teacher waits, reallocating seven idle cards, meeting API capacity and retaining target yield. It is not measured or assured.
- One **continuously active serial collector**: 6,258 targets/h, 32.982 tasks/h. The same target range is only **17–29x targets**, not 100x.
- **Eight continuously active serial collectors** at that workload: about 50,067 targets/h, 264 tasks/h. The proposed range is about **2.1–3.6x targets**. Genuine 100x over this eight-card baseline would require **~5.01 million accepted targets/h**, unsupported by this design.

### What remains, and cost

Reaching 166k targets/h at 150–180 targets/task needs **924–1,109 tasks/h**. At observed API occupancy that requires at least **24–29 simultaneous calls** before inefficiency; at the assumed 37.5 s/task short mix it needs **10–12**, realistically ~16 with tail headroom. Observed billing workload is **12,032 prompt + 1,177.75 completion API tokens/task** (completion includes reasoning). At 1,000 tasks/h on the original mix that is **12.032M input and 1.178M output tokens/h**, ~2,150 requests/h. Cost/h is `12.032 * input_price_per_M + 1.178 * output_price_per_M`, before retries/judges/supply and cache-pricing discounts; no price schedule was provided. Reserve quota, not just GPUs. Eight GPUs running 24 h consume **192 GPU-hours/day**, even if one is the learner.

True 100x over eight active collectors, even optimistically at 189.75 targets/task, needs ~26,386 tasks/h, **~687 concurrent calls** at observed latency and **~56,730 API requests/h**. With the 37.5 s/task assumption it still needs ~275 simultaneous calls (and 39,579 requests/h at 1.5/task), plus CPU/Pi/workspace/validator capacity, much larger verified task supply, and learner ingestion bandwidth. At unchanged ratios that factory would bill ~317M input + 31M output API tokens/h. Local B=16 state memory does not solve this. Larger hardware/model-optimized decode, reduced verified tool/teacher work per accepted target, or a better policy producing more correct on-policy outputs would be needed; none establishes preserved learning quality automatically.

Remaining work to hit the upper one-learner-baseline goal: measure B=1/4/8/16 phase timings and peak memory; secure teacher quota; prove cache/greedy correctness and cancellation; keep F4 stream publication single-writer; provision CPU/Pi and immutable task supply; demonstrate the learner can consume the accumulated targets with its existing guarded PG/SFT rules. More collected targets are not a 100x training-quality or generalization improvement. Proposed engineering cost is the **48–96 qualification GPU-hours and 3–5 weeks single-worker calendar** above, plus ongoing serving/API cost; additional specialized kernels/quantization/speculation or policy-quality improvements are outside this approved design pass and cannot be assigned guaranteed multipliers.

## 10. Validation, scope and next decision

Completed checks: actual bank/driver/bridge/transport/teacher/kernel source inspection with line numbers; stdlib reads of archived episodes/summary/API metrics; exact integer state/logit arithmetic; full-cycle timestamp derivation; source hashes; report-copy equality and whitespace validation. No tests were added/run against serving code because no implementation changed, and no CUDA benchmark/teacher call was authorized. One exploratory timing read initially failed because cycle 0050 has no `teacher-metrics.json`; the corrected read treats that completed no-correction cycle as zero calls and reproduced all ten rows. No success timings were substituted for missing phases.

Architecture authority read: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable evidence/safety intent: **R14/NDP13** bounded termination/deadlines (proposed teacher timeout correction), **R15** numerical/reference qualification, **R16** honest exact-source evidence boundaries. This serving proposal is not resilient training, an ADR-003 production scale submission, ISP/V21S overlap qualification, or permission to change learner checkpoint/publication rules. No R07/R12 checkpoint mutation occurs. Frontier activation/Partition/QOS evidence is not applicable: host is lambda01 and no Frontier execution/submission occurred. Any future Frontier job must use canonical activation and separately verify requested `Partition=batch,QOS=debug`.

Recommended next decision: independent review of this report, then approve **stage 1 only** with immutable checkpoint, frozen native-frame test corpus, phase-timing acceptance schema and bounded qualification budget. Teacher async/cycle-overlap rollout can be approved separately this week. Do not relaunch the bank or implement suffix-cache/precision changes based solely on this feasibility report.
