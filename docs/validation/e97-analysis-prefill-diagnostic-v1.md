# E97 analysis-prefill diagnostic v1

Date: 2026-09-09. Local/uncommitted. Status: all three GPU comparisons complete.

## Motivation and authorization

The operator authorized continuing the controlled prefill comparison after the
read-only likelihood diagnostic found improved teacher-forced continuations but
failed first-token decisions. This is **not new SFT**, parser relaxation, a
replacement for the original 0/24 autonomous development results, or promotion.

## Response-boundary audit

`scripts/audit_e97_response_boundaries.py` reconstructed all 512 actually consumed
u64 packs with the executed training sampler, verifying source hashes for the
dataset, trainer and `LadderLM`. All **6,949 Analysis target-run starts** across
169 agent records had:

- A preceding `Assistant:\n` header.
- The `Analysis` token as the next-token target, with its loss mask enabled.
- Valid predecessor/target tokens and no reset before the target.

One exact example: record 978, pack 849, update 1/rank 0. Input position 1455
is newline; target position 1456 is token 32750 (`Analysis`); prediction position
1455 has loss enabled. Colon and space-prefixed quote follow at positions
1457–1458. Thus the target is predicted **before** being consumed.

Audit artifact:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/learning-diagnostic-v1/response-boundary-audit.json`,
SHA-256 `4ac6e57ae63c547462a5df590292097c8950dba5d0184d8f1abb7775dec3d2af`.
Two CPU tests exercised actual `LadderLM` dense/chunked CE: gradients select the
correct target at the correct prediction position, and masked logits get zero
gradient. Managed process `proc_3f25` passed in 18 seconds.

This rules out a straightforward header/label/mask shift in the checked path,
not production fused GPU backward defects or competing/suppressed updates.

## Frozen experiment

Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/analysis-prefill-diagnostic-v1`.

- Same already-consumed 24-task development panel, SHA-256
  `a29094fa9934d83c7f89ce6f4f4cf93fbaa9982aa295bec78684616896d0cb90`.
- Supply exactly **`Analysis:`**, tokens `[32750,25]`, on every assistant turn.
  Do not supply the quote, reasoning, action, arguments or answer.
- Compare parent train/y, u64 train/y, u64 saved/x, in that order.
- Same system/user messages, bounded read tool, authentic observations, exact
  task scoring, six-turn limit and 2,048-token analysis cap.
- Total assistant completion budget remains 4,096 tokens per turn: two supplied
  tokens plus at most 4,094 model-generated tokens, accounted separately.
- Fail fast only on prefixes that no appended suffix can make canonically
  valid. This saves evaluation cost without admitting otherwise-invalid turns.
- Validate BPE concatenation at the actual assistant boundary before supplying
  the prefix. Preserve canonical analysis and observation history across turns.
- Separate prefilled schemas/metric names; do not reuse the autonomous
  first-turn-protocol field or change the stopped training gate.

New scripts: `scripts/eval_e97_analysis_prefill.py` and
`scripts/aggregate_e97_analysis_prefill.py`.
CPU regressions: **28 passed**, including valid-prefix preservation, early
invalid-prefix stopping, actual fixture replay with exact history, token
accounting, shard identity/membership checks and eight-rank CUDA routing.

Read-only source inventory SHA-256:
`9b4a83c4170245f2426c16bc957589e2539500bb05ec5be1930b6b034bac892f`.
`recipe.json` records the supplied-prefix intervention and all three checkpoint
identities before GPU execution. Managed process **`proc_a384`** executes
`run-prefill.sh` under an exclusive eight-GPU lease, with NUMA placement,
explicit CUDA device routing and isolated Triton caches. Any execution failure
stops the chain; no automatic retry.

## Interpretation

Success would demonstrate conditional task execution **with supplied opening**,
not acquisition of that opening or independent held-out generalization. Failure
would show that providing this one header does not suffice. Neither outcome
alone proves the root cause of the training failure.

## Results so far

Parent train/y: **0/24 prefilled successes**, **0/24 valid prefilled first turns**,
no admitted read calls. Each task failed after one model-generated token: it
continued the supplied `Analysis:` with a newline or unquoted text rather than
the required JSON-string opening. Total: **24 model tokens + 48 supplied tokens**.
Fail-fast rejected these provably invalid prefixes rather than generating
4,096 tokens per failed task. This is the parent control, not the trained result.

Parent aggregate SHA-256:
`4bd4fa9ed4f0d6a8bc13b442e4a26d1912f7b7c29e6e97dd4a37271b7321efc3`.
u64 **train/y** completed: **0/24 prefilled successes**, **1/24 valid prefilled
first turns**, no admitted read calls. Unlike the parent, it generated the JSON
string opening after the supplied header. However, 23 tasks continued repetitive,
unfinished analysis until the 4,096-token total budget. The remaining task
(document_sum-02, explicitly requesting no tools) produced a complete analysis
and a read call to unrelated `src/read_tool.py` with limit 40; the fixture tool
rejected its arguments (supported limit is at most 10).

Total: **94,205 model-generated tokens + 48 supplied tokens**. The repeated
outputs included `The read tool is available` and echoes of user instructions,
not completed answers. Providing the opening alone therefore did not uncover
successful task execution in train/y. The earlier improvement in teacher-forced
continuation likelihood must not be described as demonstrated free-generation
competence on fresh prompts.

u64 train/y aggregate SHA-256:
`098405b006e94d6557a082e748bbed52cbb64af254a8cffc4c2ccfdce419bc6c`.
u64 **saved/x** completed: **0/24 prefilled successes**, **12/24 valid
prefilled first turns**, no admitted read calls. All 12 completed turns were
read calls with **limit=200**, contrary to the requested limit=10 on tool tasks
(and the fixture tool's maximum of 10). Two of those calls selected the correct
first path (`two_reads-01` and `two_reads-03`), but still violated the bounds.
Three were inappropriate reads on no-tool instruction tasks; remaining calls
altered or shortened requested paths. Some analysis text repeated the requested
limit=10 while the actual arguments still used 200.

The other 12 outputs comprised 11 token-budget stops and one provably invalid
prefix stop. Total: **46,255 model-generated tokens + 48 supplied tokens**.
Saved/x therefore shows more conditional framing than train/y, but not successful
execution under the actual instructions/tool contract. Because no call was
admitted, neither variant was tested on a real second-turn observation here.

Saved/x aggregate SHA-256:
`bb35bd3f59b734ae155183847b383fca648d9db2260871e32b12934ac31e866a`.

`proc_a384` completed successfully in **839 seconds**. All three runs are
reported; execution success is not model success. Final summary:

| Representation | Prefilled task success | Valid prefilled first turn | Admitted reads |
|---|---:|---:|---:|
| Parent train/y | 0/24 | 0/24 | 0 |
| u64 train/y | 0/24 | 1/24 | 0 |
| u64 saved/x | 0/24 | 12/24 | 0 |

The supplied opening alone is insufficient. The result supports a broader
instruction-to-action/argument and free-generation problem, not a claim that
nothing was learned or that the first-token decision was the sole bottleneck.
The next diagnostic priority is production fused-gradient/effective-update
behavior, with no further prefix adaptation or automatic training restart.
