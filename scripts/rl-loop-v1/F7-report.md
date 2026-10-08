# F7 — teacher throughput, cognitive A/B and bounded corrections

2026-10-08, lambda01. **No B relaunch, model flip, GPU acquisition, training change or bank receipt publication.** B kept its eight GPU leases; A's nominally freed capacity had already been absorbed by B. All qualification was CPU/API/Pi/grading, using archived TRAIN failures and private fresh workspaces.

## Verdict in brief

- DeepSeek's requested-model ID was exactly **`deepseek-4.1-flash-background`**. The full frozen A/B produced **21/30 eligible receipts**, versus **18/30 corrected-eligible live GLM flash** and **15/30 GLM nonflash**. One initial flash candidate had a probe-side checkpoint bookkeeping defect: **strict uncorrected count 17/30, corrected count 18/30**. The pre-registered receipt noninferiority point criterion passes under **both** counts (+13.333pp strict-raw; +10pp corrected).
- Decision-relevant *sealed-complete paired* grades are **16/20 DeepSeek versus 17/20 live flash: −5pp, exactly the margin**. Missing operational outcomes are not model-inferiority evidence. This small, selective, single-template panel does not establish general cognitive equivalence or mixed-workload quality.
- At the capped 12-worker probe, DeepSeek measured **891.444 native-closed corrections/h, 658.893 eligible receipts/h and 216,815 eligible target-tokens/h**. These are **burst+drain capacity, finite sample, NOT sustained production limit**, and not live B output.
- The operator-stated endpoint limit is **12 global background lanes**. A 240s HTTP 429 is a wait for a free lane, **not endpoint degradation**. No 16/32/64 concurrency rung ran.
- The pooled operator-formula DeepSeek proxy is **993.919 complete correction episodes/h**, corresponding to **1,529.106 attempts/h** at F6's 0.65 corrections/attempt. It is a client-wall-latency proxy, not isolated server occupancy or a sustained benchmark.
- **Do not enable the pool from these headline rates alone.** The actual default-width-1 four-attempt timing replay admitted one correction, explicitly exhausted three, produced zero receipts and waited **539.939s at the terminal fence**. Async stays OFF; the durable model default stays live GLM flash.

**Teacher-ID discrepancy:** operator-facing summaries said `glm-5.3-background`; live deployed configuration was `glm-5.3-flash-background`. The three-way A/B covers both; the flash model is the decision-relevant baseline.

## 1. Workload, scope and measurement contract

Private evidence root: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/F7-probe` (owner-only). `plan.json` froze 30 distinct TRAIN task identities, their archived failed policy records, task/source hashes and the original pilot bytes before the first request. Every task ran under all three models. No dev request, episode, grade or receipt was produced by F7.

Selection used the unchanged lane fresh-solve gate: degenerate policy prefix or irreversibly sealed-wrong first action. Only `first-party-read-observe` qualified in the available archive, so cross-template stratification was impossible without new policy attempts. These are real read/recovery tasks, not newly authored synthetic benchmark replacements. Tools executed through real isolated Pi processes; both focused and regression sealed validators, degeneracy screening, native close and the canonical receipt encoder were exercised. Validator program/spec hashes are checked by the existing grader; all models used the same project interpreter. Its existing grade note explicitly records that the spec's hermetic interpreter is absent on this host; no new hermetic-interpreter claim is made.

The archived B correction mix at freeze was **146 fresh / 387 = 37.726%**, **241 spliced / 387 = 62.274%**. This is historical archived traffic, not a current live-window estimate. Spliced attempts' archived workspaces have already been modified in-place by their original corrections; copying them would fabricate pre-correction state. Supervisor approved fresh-only qualification. **The majority-spliced mixed workload is unmeasured.**

The original pilot remained byte-identical in the isolated copy: SHA-256 `06f870fa4d6a0cc2d05202902ebce19859bc2e0917513e758427de052d21e2df`. Its HTTP calls retained 900s socket timeouts, four transport attempts and existing retry sleeps; each probe child had an external 1200s group-teardown bound. Teacher episode budget remained 480s. The separate deadline adapter was not used by these probes or the pool timing replay.

The initial proposed ladder was superseded before its first rung by operator steering: **4/8/12**, two waves each (**8/16/24 correction outcomes/model**), GLM nonflash and DeepSeek. Flash's full A/B at concurrency 4 supplies its reference latency; flash was not additionally probed at 8/12. The immutable early plan's old ladder intent and the ladder-design approval's legacy word “degradation” are not executed-rung evidence; `ladder-design.json` records the authoritative 12-lane cap and queue interpretation.

All API timings are client-observed monotonic intervals (pilot rounded to 0.001s), including response/queue waits. Successful calls' queue component cannot be separated from service time. The 429 durations are observed queue-timeout waits, not uncensored successful queue-wait measurements. Cohort wall includes process imports/startup, tools, retries/sleeps, grading, failures and the slowest drain. Episode latency below starts after worker imports. Live B and the standing-supply daemon shared the endpoint throughout; no exclusive 12-slot reservation or server-side occupancy telemetry was available. Models/rungs were sequential, and cache/background-load changes confound causal speedup attribution.

Eligible receipts here are private **candidate receipts**, not B stream appends or learner consumption. “Completed corrections” means native-finished **and close-verified**, including sealed grade failures; “receipts” additionally requires sealed passing grade and positive canonical supervised targets. Finalized failed outcomes are separately counted, never relabeled completed corrections.

## 2. Full-size convergence A/B

Pre-registered operational criterion: `DeepSeek receipts/30 >= live-flash receipts/30 - 0.05`. This is a point-estimate noninferiority rule, not a statistical-confidence proof. All raw counts are retained; queue/time-window/protocol missing outcomes are inconclusive for cognitive capacity, not sealed cognitive failures.

| Model | Corrected-eligible receipts / 30; grade passes / 30 | Native closed | Sealed failures among closed | Incomplete/missing | Canonical targets/receipt | API completion tokens/receipt | API latency p50 / p95, s | Episode latency p50 / p95, s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| glm-5.3-flash-background | 18; 18 (60%) | 21 | 3 | 9 | 313.222 | 2,419.556 | 11.241 / 240.226 | 65.504 / 872.780 |
| glm-5.3-background | 15; 15 (50%) | 18 | 3 | 12 | 439.933 | 8,157.333 | 50.877 / 240.209 | 312.534 / 788.005 |
| deepseek-4.1-flash-background | 21; 21 (70%) | 27 | 6 | 3 | 322.143 | 2,033.000 | 1.420 / 116.597 | 6.607 / 486.581 |

API tokens/receipt includes known usage from unsuccessful/invalid-frame attempts, not just the receipt-producing calls; hidden reasoning and API output are not supervised targets. Prompt tokens/receipt and full min/mean/p90/p95/max distributions are in `F7-measurements.json`. Closed-episode API call counts were flash **2–5, median 3**; nonflash **2–6, median 4**; DeepSeek **2–10, median 4**. DeepSeek's faster individual requests do not imply one call per correction.

| Sealed-complete paired comparison | Matched n | Baseline passes | DeepSeek passes | DeepSeek delta |
|---|---:|---:|---:|---:|
| Live flash versus DeepSeek | 20 | 17/20 (85%) | 16/20 (80%) | −5pp |
| Nonflash versus DeepSeek | 15 | 12/15 (80%) | 12/15 (80%) | 0pp |
| Complete under all three | 9 | flash 8/9; nonflash 7/9 | 7/9 | −11.111pp versus flash on this smaller subset |

The raw receipt gate passes. The decision-relevant paired cognitive point is exactly at its margin, with only 20 observable pairs and one net additional DeepSeek failure (index 29). The all-three subset is smaller and less favorable. **No broad cognitive-convergence or training-convergence claim, and no automatic model flip.**

## 3. Capped endpoint probe — measured numbers

Headline column definition: **burst+drain capacity, finite sample, NOT sustained production limit**. Concurrency is additional probe workers, not a measured total count of active global requests; B could simultaneously issue its own calls. A queued successful correction is a latency event, not a failed correction. The library's HTTP `error` records are transport events, not automatic model failures.

| Model | Probe concurrency; n | Closed / receipts | Wall, s | API mean / p50 / p95, s | Closed corrections/h | Receipts/h | Eligible targets/h |
|---|---:|---:|---:|---:|---:|---:|---:|
| glm-5.3-background | 4; 8 | 3 / 2 | 1,032.638 | 91.220 / 52.350 / 240.270 | 10.459 | 6.972 | 3,260 |
| deepseek-4.1-flash-background | 4; 8 | 8 / 6 | 56.336 | 2.934 / 1.721 / 9.485 | 511.216 | 383.412 | 153,812 |
| glm-5.3-background | 8; 16 | 9 / 7 | 876.305 | 66.331 / 33.510 / 238.283 | 36.973 | 28.757 | 11,815 |
| deepseek-4.1-flash-background | 8; 16 | 15 / 11 | 239.898 | 19.040 / 2.724 / 140.117 | 225.095 | 165.070 | 59,200 |
| glm-5.3-background | 12; 24 | 9 / 7 | 1,153.558 | 121.244 / 97.781 / 240.283 | 28.087 | 21.845 | 7,221 |
| deepseek-4.1-flash-background | 12; 24 | 23 / 17 | 92.883 | 6.311 / 3.707 / 19.680 | 891.444 | 658.893 | 216,815 |

| Model/rung | 429 queue-wait events / API attempts | Observed ~240s queue-timeout events | Transport retry attempts | Receipts after API error / no receipt after error | Finalized worker crashes |
|---|---:|---:|---:|---:|---:|
| Nonflash 4 | 6/33 | 6 | 6 | 2 / 3 | 0 |
| DeepSeek 4 | 0/26 | 0 | 0 | 0 / 0 | 0 |
| Nonflash 8 | 4/64 | 4 | 4 | 0 / 3 | 0 |
| DeepSeek 8 | 0/89 | 0 | 0 | 0 / 0 | 0 |
| Nonflash 12 | 17/98 | 17 | 17 | 0 / 11 | 0 |
| DeepSeek 12 | 0/97 | 0 | 0 | 0 / 0 | 0 |

The zero observed DeepSeek 429 counts at these rungs do not mean its successful requests had zero queue wait. Non-monotonic rates are finite burst/tail/composition/background effects, not proof of a concurrency knee. Queue-wait distributions, successful/error latency distributions, frame retries and exact raw references are machine-readable. Across A/B plus ladder there were 57 HTTP 429 records and one RemoteDisconnected record; no logged-success/empty-content double-accounting occurred in this dataset.

## 4. Endpoint limit and campaign sizing

| Limit/verdict row | Value | Source/qualification |
|---|---|---|
| **Endpoint limit: global background concurrency** | **12 lanes** | **Operator statement, 2026-10-08**, not inferred from 429s |
| 240s 429 behavior | Queue wait for a free lane | Operator clarification; observed durations retained |
| Sustained exclusive endpoint rate | **Unmeasured** | B/standing-supply contention; no server occupancy/queue decomposition |

Unit-correct operator formula: `call-equivalent/h = 12*3600 / mean observed API-call latency`. Then **divide by measured calls per complete correction** to get full-episode/h. Counting 12*3600/latency as multi-turn corrections/h would overstate capacity. All returned API-attempt latencies enter the mean, including queue waits/retries. Complete-episode call counts avoid treating truncated episodes as cheaper full solves; unfinished in-flight calls remain right-censored. This client-wall proxy is **not** a physical queue-free server-service ceiling.

| Model/sample | Mean call latency, s | Mean calls / complete correction | Call-equivalent ceiling/h | Full-episode proxy/h | Attempt proxy/h at q=0.65 |
|---|---:|---:|---:|---:|---:|
| Live flash, A/B c4 only | 62.444 | 3.381 | 691.816 | 204.622 | 314.802 |
| Nonflash, pooled capped rungs | 98.140 | 4.000 | 440.187 | 110.047 | 169.303 |
| DeepSeek, pooled capped rungs | 10.253 | 4.239 | 4,213.352 | 993.919 | 1,529.106 |

F6 observes a single continuous collector at **32.982 attempts/h, 6,258 targets/h**, and the sampled single-learner cycle loop at **8.764 attempts/h, 1,663 targets/h**. For campaign sizing, the *F6-normalized reconstructed* current B arm is `7*continuous collector + 1*learner loop = 239.639 attempts/h, 45,472 targets/h`; this is **not a measured current B aggregate**. A 2x campaign is 479.278 attempts/h. F6's policy-side ceiling is approximately 300 attempts/h/GPU.

**Minimum GPU footprint, ideal continuous-generation planning:** current campaign needs **1 policy GPU + 1 learner = 2 GPUs**; 2x needs **2 policy GPUs + 1 learner = 3 GPUs**. B currently allocates eight, not this minimum. Corrections themselves need **zero GPUs** and use the independent 12 background lanes. No allocation was changed by F7. Load, training sharing, terminal drains and actual policy improvement can require more GPUs than these active-policy lower bounds.

| Model | Current B demand backlog growth, corrections/h | 2x backlog growth, corrections/h | At full 12-lane episode proxy: attempts/h | Minimum policy GPUs at that proxy (+ one learner) |
|---|---:|---:|---:|---:|
| Live flash | 0 | 106.909 | 314.802 | 2 (+1) |
| Nonflash | 45.719 | 201.484 | 169.303 | 1 (+1) |
| DeepSeek | 0 | 0 | 1,529.106 | 6 (+1) |

Backlog growth is `max(0, 0.65*campaign attempts/h - episode proxy/h)`. It is an **ideal continuous-generation arrival/service projection**, not a deployment guarantee: the current bounded pool rejects exhausted admissions explicitly rather than storing an unbounded queue, and its cycle-end fence can stall the GPU lease. “No backlog” in this table is not proof of zero queue waits, full correction coverage, or fully-busy GPUs. The DeepSeek projection supports the operator's point that current/2x campaigns need few policy GPUs; their scarce shared resource can be correction capacity and queue scheduling, not eight GPU cards.

## 5. B-grid verdict and 10x versus 20x

Projection label: **fresh-correction capacity measured; mixed-capacity projected, unmeasured**.

The following uses each rung's *finalized correction outcomes/hour* (failed outcomes included, unlike the closed-correction column above), divided by F6 q=0.65, capped by the scalar B policy grid. It assumes the fresh workload rate transfers to the majority-spliced workload, unchanged 10s admission, seven continuous collectors plus one learner retaining F6 noncollect gaps, and **F6 yield held at 189.75 accepted targets/attempt**. It is a conditional sizing table, not observed B performance. Measured fresh target yields and a separate fresh-only yield alternative are in the JSON; neither qualifies the mixed diet.

| Model | Concurrency | Conditional B attempts/h | Conditional B accepted targets/h (F6 yield held) | Ratio versus F6-normalized same grid |
|---|---:|---:|---:|---:|
| glm-5.3-background | 4 | 42.907 | 8,142 | 0.179x |
| deepseek-4.1-flash-background | 4 | 786.487 | 149,236 | 3.282x |
| glm-5.3-background | 8 | 101.124 | 19,188 | 0.422x |
| deepseek-4.1-flash-background | 8 | 369.387 | 70,091 | 1.541x |
| glm-5.3-background | 12 | 115.229 | 21,865 | 0.481x |
| deepseek-4.1-flash-background | 12 | 1,431.080 | 271,547 | 5.972x |

Using the pooled DeepSeek operator formula instead gives **1,529 attempts/h / 290,148 targets/h**, **6.381x** this reconstructed same-grid baseline. These assumptions do not include the measured pool drain loss. Against the F6 *observed single learner* this conditional rate is about 174.5x, but most of that denominator change comes from already-added F4 collectors and recovered waits; it is not a measured teacher-only gain.

**Where 10x and 20x land:** on the same-grid/F6-yield denominator, 10x requires **2,396.392 attempts/h / 454,715 targets/h**; 20x requires **4,792.784 / 909,431**. Keeping F6's scalar policy cost and learner noncollect gap gives an ideal grid ceiling **2,127.826 attempts/h / 403,755 targets/h = 8.879x**, even with unlimited teacher capacity and no terminal drain. A single continuously collecting card has the analogous ideal teacher-removal bound **9.167x**. Therefore **neither 10x nor 20x same-resource task throughput is established by this teacher lever**. Tenfold *target-token* output could additionally depend on longer accepted targets/different yield, which is not automatically more completed tasks or better learning. More policy serving efficiency or a changed campaign/diet/resource denominator is needed; none was implemented here.

## 6. Durable knob and bounded pool — implemented, OFF

Staged matching scripts: `e97-rl-loop-v1/F7-staged/scripts`, **not** live `e97-rl-loop-v1/scripts`. `F7-deployment.md` explains the manual, operator-approved relaunch procedure. No live `correction-config.json` was created or changed.

- Durable `bank/correction-config.json` controls `teacher_model`, `async_enabled`, `pool_width`; the daemon rereads it each cycle. Missing file = **glm-5.3-flash-background / OFF / width 1**. CLI model overrides remain explicit. Writing-judge identity stays pinned to the live flash baseline, not the correction-model knob.
- Width is per lane: eight B lanes at default 1 leave four of the operator's 12 global slots for other callers. This is conservative budgeting, not a global semaphore or proof those four slots are free. Larger widths/lane-count changes require reviewing the total caller budget.
- After policy grading, immutable specs bind task, served checkpoint, attempt/grade/transcript identity, owned workspace and fresh/spliced decision. Each correction owns a separate Python/Pi process and generator. Policy attempts continue within the cycle; accepted admissions preserve the lane's 10s minimum interval. Full capacity fails immediately and explicitly—no silent drop, no receipt, existing failed-task round retry semantics retained.
- Only the collect owner forms/appends receipts, using the current stream tail at publication, so out-of-order results cannot reuse an old `prev_digest`. All existing eligibility, sealed grading, encoding and native-close gates remain; PG/SFT arithmetic and checkpoint adoption/exposure are unchanged.
- End-of-cycle drain/fence precedes summary and training. No cross-cycle correction publication is implemented. Worker hard wall is 750s, with Linux parent-death/process-group containment and atomic result handoff; ordinary Pi descendants are killed on owner loss. Daemonizing tools/writing/mixed families are not qualified by this read-only panel. Completed API attempt logs survive worker crashes; incomplete log tails are explicit.
- Cycle summaries separately record terminal-drain time and exhaustion counts. **Do not claim fully-busy GPUs**: admission throttling and terminal drain remain.

### Actual terminal-fence measurement

One four-attempt timing replay used the **actual width-1 pool**, frozen fresh TRAIN specs, the isolated original pilot, real API/Pi/sealed graders, no GPU and no B publication. Policy spacing was **simulated from F6's 11.694671s**, not newly benchmarked policy generation.

Result: **1 admission / 3 explicit exhaustions / 0 eligible receipts**; cycle wall **586.718s**, terminal fence **539.939s (92.027% of that wall)**. The admitted job logged five API attempts: **56.370, 28.062, 14.035, 240.170 (429 queue wait), 208.149s**. Its native result stopped at `transport_deadline`: the 480s transport window expired. This is a queue/time-window miss, not a sealed cognitive-inferiority result. The data demonstrates why fast burst measurements cannot promise deployment throughput and why a width-1 no-queue pool can exhaust under a long tail.

The 12-process **CPU sleep-fixture** test also measured a roughly 0.2s fence; its exact fresh-run value is in `F7-unit/F7-pool-drain-test.json` and is **not API latency**. Real I/O-probe drains after the last admission were nonflash **554.371 / 648.119 / 276.704s** and DeepSeek **18.902 / 61.974 / 45.494s** at 4/8/12; these are probe cohort drains, not four-task GPU-bank cycle drains.

**Named follow-up, not implemented:** a persistent cross-cycle correction writer/service, bounded durable admission/backlog and summary/claim lifetime/fencing changes. It must preserve original served-checkpoint provenance, late receipt eligibility and single-writer ordering. The measured 539.939s gap is material enough to justify considering that scope, but not permission to implement it here. Global caller budgeting and controlled mixed-workload measurement are prerequisites to an OFF-to-ON rollout.

## 7. Validation, evidence and commit receipts

Public reproducible accounting: `F7-measurements.json`, `F7-evidence-manifest.json` (1,062 raw/derived-file hashes; private TRAIN contents stay private), source-named report tests. Raw API records are authoritative even for incomplete workers. The early A/B controller's misleading `completed_per_hour` field counted all finalized outcomes; **it was not used** in reported closed-correction rates. Nine initial flash rows lack row-level probe-source annotations (four produced raw candidates); no single probe-source version is asserted across all rows. The frozen pilot remained byte-identical; harness instrumentation/bookkeeping changed during the run.

### Checkpoint-binding erratum and dual-count validation

The final audit checked **all 104 original candidate receipts** against their original archived collect-summary served-checkpoint SHA, task identity, failed-attempt SHA, receipt digest, grade gate and a full canonical token/mask re-encode. **103/104 original bindings were clean.** Flash `ab-4/case-0005` predates the probe's checkpoint-metadata fix: its constructor used the **attempt-episode SHA instead of the served-checkpoint SHA**. This was probe-side bookkeeping, not a defect in the teacher's unchanged real episode, sealed grade, native close or 364 supervised targets.

Supervisor approved a separately named `corrected-candidate-receipt.json`, derived through the real receipt constructor from the unchanged episode/grade and original archived collect summary. No API rerun, bank append or original-artifact rewrite occurred. All fields except checkpoint metadata, construction time and resulting receipt digest match the original. The original SHA remains **`58c6e6fa39dbf9b6d030f3fc0909c20d1fe301626ef0752dea3c144affc77c94`** and both artifacts are hashed in the manifest. The corrected candidate passes full receipt/re-encode and served-checkpoint binding validation; **104/104 corrected-eligible candidates** now validate.

The raw constructor count is still flash **18/30**, but strict uncorrected eligibility is **17/30**; corrected eligibility is **18/30**. DeepSeek is **21/30** under both bases. Noninferiority passes with **+13.333pp strict-raw** and **+10pp corrected**—the verdict cannot hinge on this repair. Teacher grade rates and the 20-pair cognitive comparison are unchanged. The table uses corrected-eligible counts; JSON exposes both. `validate_candidate_binding` now runs at future construction time and in the report's all-candidate audit; a source-named regression test rejects attempt-SHA-as-checkpoint before receipt acceptance.

Commands (lambda01 project `.venv/bin/python`, no Frontier/Slurm submission):

```bash
export PYTHONPATH=/home/erikg/emender:/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts
.venv/bin/python scripts/rl-loop-v1/rl_teacher_probe.py freeze --root "$ROOT" --bank "$B"
.venv/bin/python scripts/rl-loop-v1/rl_teacher_probe.py run --root "$ROOT" --phase ab
.venv/bin/python scripts/rl-loop-v1/rl_teacher_probe.py run --root "$ROOT" --phase ladder
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/rl-loop-v1/rl_teacher_probe.py pool-timing --root "$ROOT" --model deepseek-4.1-flash-background
.venv/bin/python scripts/rl-loop-v1/rl_teacher_probe_report.py --root "$ROOT" --output scripts/rl-loop-v1/F7-measurements.json --manifest scripts/rl-loop-v1/F7-evidence-manifest.json
.venv/bin/python scripts/rl-loop-v1/rl_bank_unit_test.py --scratch /home/erikg/emender-scratch/fix-fanout/F7-unit
.venv/bin/python "$W/F7-staged/scripts/rl_bank_unit_test.py" --scratch /home/erikg/emender-scratch/fix-fanout/F7-unit-staged
.venv/bin/python -m pytest -q tests/test_rl_teacher_deadlines.py tests/test_rl_teacher_probe_report.py
```

Here `ROOT=$W/F7-probe`, `W=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1`, `B=$W/bank-singlelearner-B`. Frozen/running directories are create-once; reruns require a new private root, not overwriting evidence.

**185/185 CPU bank checks** passed in final repo and staged-copy runs, including exhaustion, delayed/out-of-order publication, late closed-cycle fencing, crash/partial result rejection and metric retention, deadlines and real owner SIGKILL containment. **6/6 pytest tests** passed (3 deadline adapter, 3 throughput/binding accounting). Compile/whitespace/shell syntax and copy-equality checks passed. The original fixture mock needed a true native-close field after the explicit gate check; that initial test failed and was corrected. An initial staged run lacked the unchanged companion `rl_pg_batch.py`; the full unchanged companion bundle was supplied and the suite passed. One combined two-suite wrapper timed out at 120s after repo success; the staged suite passed separately with a 240s wrapper.

Commit receipts, scoped and pushed:

- `cfd63317`: durable model knob, default actual live flash; staged only.
- `46dbc06d`, `1ce899fc`, `1f21e956`: frozen three-way/capped probe, owner-private artifacts and unit-correct accounting.
- `313130e1`: bounded default-OFF pool, immutable specs, single writer and measured timing replay.
- **`fb80cfff`: separate deadline-aware HTTP retry adapter, DEPLOY AT NEXT OPERATOR RELAUNCH.** Three network-free deadline tests; excluded from probe and isolated timing paths. Socket timeouts/retry sleeps use remaining episode budget; the worker wall bound contains full-response/tool tails. This is not a measured speedup or a live B change.

Live deployed lane SHA remains `9490b40f6fd5e4b6acf469d381eb689d91f902c6d10442873de722e85a5f395e`; driver remains `da9c5f79d3584c69e933c04e8b4719c792c1ed2b4d6fae2e22ede4e19af00821`; shared pilot remains the frozen SHA above. The independently running standing-supply daemon modified `configs/pi/e97-firstparty-collection-authorizations-v1.json`; F7 neither edited/reverted nor committed that unrelated working-tree change.

Architecture authority read: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable safety/evidence intent: **R14/NDP13** bounded local termination; **R15** unchanged reference/grading gates and honest qualification; **R16** exact-source evidence and scope/denominator discipline. This is not resilient async-v2.1/ISP overlap or a scale gate; no R07/R12 checkpoint mutation occurs. Elastic/native/v2.1 retired research requirements are explicitly unclaimed. Frontier activation and separate Partition/QOS evidence are not applicable on lambda01 without submission.

**Next step:** independent review, then operator decision on a controlled fresh-only model trial/global quota plan; leave live B, model default and async OFF unchanged meanwhile. Mixed/spliced quality/capacity, sustained rates, provider cancellation, and the terminal-drain/backlog problem remain open—not hidden behind the headline burst numbers.
