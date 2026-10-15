# Frontier training: evidence and manuscript integration plan

Date: 2026-10-15. Repository evidence baseline: [`baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e`](https://github.com/spinozans/emender/commit/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e). This is a proposed integration document; it does not change the canonical manuscript.

Companion: [Reader-first transfer review](READER_FIRST_TRANSFER_REVIEW.md).

## Recommendation

Give the large Frontier continuation a substantive subsection in the language-modeling results, with its own training protocol and, when the raw logs are recovered, its own accepted-token loss curve. The evidence supports a **1.2866-billion-parameter nonlinear E97-MLP checkpoint at 513.014 billion accepted training tokens**, continued from the released 150.794-billion-token workstation seed. This is important evidence of extended training and operation at 256 nodes / 2,048 accelerator ranks.

The scientific distinction belongs in the presentation itself:

- The controlled E97/GDN2 comparison remains the approximately 150.79-billion-token, eight-GPU workstation study on **The Pile**.
- The Frontier continuation is **single-arm E97 training whose final segment uses CommaPile mainmix**, with changed world size and DiLoCo cadence. Its lower recorded loss must not be described as a controlled same-distribution improvement over the workstation result.
- The 513B export's **2.2798 nats/token** is a rank-0 training-log average, not held-out loss, a cross-rank loss mean, or a new evaluation of the final checkpoint weights.
- The independent 8 → 32 → 256-node E97/GDN2 qualification ladder is useful systems evidence, but its short runs are not a matched 513B training comparison or a scaling-law experiment.

A continuous Frontier loss curve is **not yet recoverable from the inspected public export and committed late-run evidence**. Endpoint metadata, lineage anchors, a complete sequence of 73 merge timings for the first 256-node continuation, and short-ladder summary statistics are available now. The missing curve should be an explicit data-recovery item, not filled by interpolation or by relabeling the existing workstation figure.

## 1. Evidence checked and current confidence

### Fresh public-object verification

On 2026-10-15, anonymous HTTPS reads freshly retrieved the following public objects. Their downloaded SHA-256 values exactly match both the export's `SHA256SUMS` and the corrected repository audit:

| Object | SHA-256 |
|---|---|
| [`args.json`](https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/args.json) | `174aaa91f02c3e6cd537eca9737506b0bb088a7b08c30c12134e79f8a94c69b6` |
| [`model_config.json`](https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/model_config.json) | `03cf3faab1bb9f783fec6dea553225a6f42327849b26a229923011abe71f601d` |
| [`manifest.json`](https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/manifest.json) | `a5217aa479d99b1752a9a1927a87be6db29f3848cd8246c3f52a6604d1752253` |
| [`metadata.env`](https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/metadata.env) | `97ba7c8413e0abd8570d8e01540b6b10cd6439b23c7dc9093ae0ae32d4d53c96` |

The [513B prefix listing](https://spinozans.s3.amazonaws.com/?list-type=2&prefix=emender%2Fe97-diloco%2Femender_E97_1.3B_20260804_513B%2F) returned seven objects and `IsTruncated=false`: the five sidecar/checksum files, the checkpoint, and `latest.json`. It contains **no raw training log or loss CSV**. A delimiter listing of `emender/e97-diloco/` exposed the 150B and 513B run prefixes and two latest pointers, without another public training-log prefix under that directory.

The 7.72 GB checkpoint was **not downloaded, loaded, or rehashed in this review**. Payload identity and embedded values are supported by the prior direct-payload audit, the current matching sidecars, and the independent Frontier inspection receipt. These are different levels of verification and should remain distinguishable.

### Repository source inventory

All paths below exist at the pinned repository baseline unless an execution commit is specified.

| Source | What it establishes | Boundary |
|---|---|---|
| [Corrected 513B audit](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/reports/audits/verify-e97-513b-correction-20260805.md) | Exact public sidecars; nonzero MLP; payload metadata; final step/tokens/loss; source job | Supersedes the earlier missing-arguments audit; this review did not repeat the payload inspection |
| [First final-seed 256-node continuation](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/docs/validation/e97-final-seed-production-256n-5134243.md) | Exact 150B seed hash, all-rank resume, K40 progress, final checkpoint, clean shutdown | One 1h46m49s allocation, not the full 513B trajectory |
| [Machine verdict](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/reports/frontier/e97-final-seed-production-256n-5134243-verdict.json) and [Slurm accounting](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/reports/frontier/e97-final-seed-production-256n-5134243-sacct.txt) | 73 merge steps/durations, 72 inter-merge intervals, counts and terminal accounting | Does not include the 291 individual loss/throughput records that were parsed to produce it |
| [Legacy token migration receipt](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/docs/validation/e97-total-token-migration-step2303840.json) and [clock contract](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/docs/validation/embed-monotonic-total-20260802.md) | Exact 199.615B bootstrap and embedded-clock semantics thereafter | The legacy checkpoint itself had no embedded token counter; the receipt is an explicit one-time migration authority |
| [400.942B milestone / four-hour preflight](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/docs/validation/e97-4h-normal-preflight-step2315840.md) | Step/token/loss/hash anchor and intended next-run configuration | A preflight is not proof that its proposed scheduler transaction ran exactly as written |
| [513B checkpoint graph inspection](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/docs/validation/e97-35b-moe-graph-inspection-step2322520-513b.json) | Separate inspection of the dense base checkpoint subsequently used in MoE work | Does not make the dense 513B training run a 35B/MoE run |
| [E97/GDN2 Frontier qualification](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/docs/validation/e97-gdn2-paper-frontier-qualification.md) | Serialized 8/32/256-node systems gates and stabilized linear-control follow-up | Short qualification only; no 6,000-step production comparison was submitted by that qualification task |
| [Execution-version training code](https://github.com/spinozans/emender/blob/33bd7a2dc8f7705daf1685cdb825da0cbeed64f0/train.py#L3381-L3434) and [final metric/checkpoint code](https://github.com/spinozans/emender/blob/33bd7a2dc8f7705daf1685cdb825da0cbeed64f0/train.py#L3554-L3647) | What tokens, log loss, throughput, and final loss actually mean | Use the recorded execution SHA, not an assumption about current code |
| [Existing curve generator](https://github.com/spinozans/emender/blob/baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e/scripts/plot_e97_diloco_loss.py) | Existing figure is explicitly an eight-GPU workstation plot | Its default source root is `/mnt/nvme1n1/erikg/diloco_8gpu/emender`; it is not a Frontier curve |

### Branch inventory and recovery scope

The live GitHub branch list contained **295 branches**. This review inspected current main and the editorial checkout, and compared 11 targeted historical branch tips against main. It was a targeted evidence recovery, not an exhaustive inspection of every file on all 295 branches.

| Branch / pinned tip | Recovery finding |
|---|---|
| `paper-1p3b-spine` / `0325d925b50958b1098c3bdc036131d4e53b80de` | Ancestor of current main; no ahead-only files |
| `wg/agent-1735/publish-final-seed-2` / `c7f4600a4698e426846fa2af18c3809217d60374` | Ancestor of main; production source for job 5134243 |
| `wg/agent-1739/publish-token-clock` / `d81c92ef303880770f14276d983764979479e2bd` | Ancestor of main; clock publication already retained |
| `wg/agent-1736/monitor-existing-256-node` / `b3fbc4a695134463c53132f10295102084f999d2` | Report, verdict, accounting, and request marker; report blob is byte-identical to main (`befa8046c9645f28be61051262274bd28ffd94d3`) |
| `wg/agent-1737/embed-monotonic-total` / `3e7625a37b771b8dcf5af040c7a872e902a0195c` | Migration receipt, implementation, and tests; no new long-run loss file in its changed-file list |
| `wg/agent-1731/wire-final-150b` / `b576866b96f09e64db4830a9f93db5f3dd0627d1` | Seed retry/launcher provenance, not a full production loss series |
| `audit-e97-513b-args-20260805` / `17fb91f0f0cd9d122fabc5b4a3a81c6d0b2d414a`; `verify-public-e97` / `b69861a9cd90277207cd4826e1d31ab0eedf1ece` | Older audit and historical logs; use the corrected sidecars and audit on main for current model identity |
| `wg/agent-53/frontier-results-synthesis` / `43e814f8a68a5361fe1674e81e04af09a4c10c5b` | Early June debug/kernel/full-model evidence, not the August final-seed continuation |
| `wg/agent-579/run-olcf-e97-256n-debug` / `6280e10e8fdf43a9542ec225db3642dacd12a39e`; `wg/agent-585/monitor-olcf-e97-256n12h-production` / `f89f12055e11b7c0988d605b6ab34a765423bcc4` | July gate/monitor reports; related production submission report says no-submit, so branch titles are not evidence of completed large runs |

A content search of committed `logs/`, `frontier_runs/`, and raw log/CSV/JSONL files in `reports/` found no late-production loss stream matching jobs 5134243/5157856 or the 2,300,930–2,322,520 step range. This negative finding applies to the inspected main tree and targeted branch comparisons, not to uninspected private storage or every historical branch.

## 2. Run identity, lineage, and accepted-token clock

### Architecture and optimizer

The final 513B sidecars identify **E97 with nonlinear state and an FFN/MLP**, not mixer-only E97:

- 1,286,589,072 total/trainable parameters; width 1,792; depth 11
- 216 heads; state width 32; 32 groups; 64 slots
- `mlp_ratio=2.2623`, `mlp_multiple=64`, `linear_state=0`
- BF16; sequential fused Triton route (`use_triton=1`, `use_chunked_e97=0`)
- Schedule-free optimizer; base LR `0.001007`; weight decay `0.01`; gradient clip `1.0`
- Per-rank batch 4; context length 2,048; gradient accumulation 1; tokenizer `p50k_base`
- Pure DiLoCo: `diloco_island_size=1`, K40, outer optimizer `avg`, outer LR 1.0, outer beta 0.0, export basis `x`
- Hierarchical merge, group size 4, bucket size 67,108,864 elements
- Final source commit `33bd7a2dc8f7705daf1685cdb825da0cbeed64f0`; source job 5157856; execution epoch 18; world size 2,048

The legacy command argument `params=100m` is overridden by the explicit geometry. It must not be used as the model size; the actual parameter count is recorded independently in the sidecars and checkpoint inspection.

The freshly read [150B seed arguments](https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/args.json) retain the same E97-MLP geometry, schedule-free optimizer and LR, B4/context 2,048, and tokenizer, but use **8 ranks, K250, log every 25 steps, and `pile.txt`**. The 513B arguments use **2,048 ranks, K40, log every 10 steps, and `commapile_mainmix_v0.1_1tb.txt`**. These are material protocol changes. Corpus identity at each intervening execution epoch still needs its launch receipt; the final arguments alone should not be projected backward onto every epoch.

### Verified lineage anchors

| Stage | Worker/optimizer step | Accepted tokens | Recorded loss | Authority |
|---|---:|---:|---:|---|
| Released workstation seed | 2,300,930 | 150,793,748,480 | 2.4365 (rounded seed checkpoint loss) | Public seed manifest/hash and all-rank resume audit |
| First 256-node continuation, job 5134243 | 2,303,840 | 199,615,447,040 | 2.317775578260422 (final checkpoint field); last logged window separately 2.2665 | Independently loaded checkpoint plus explicit legacy token migration receipt |
| Intermediate milestone | 2,315,840 | 400,942,039,040 | 2.2928028006553647 | Milestone preflight; exact matching resume anchor in final-job arguments |
| Retained 513B endpoint, job 5157856 | 2,322,520 | 513,013,841,920 | 2.279764811873436 (`loss_last_100`) | Current public manifest/arguments and corrected payload audit |

The first continuation's seed SHA-256, `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`, exactly matches the [public 150B seed manifest](https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json). This binds the workstation seed to the Frontier continuation, rather than merely matching a model name.

Additional checkpoint identities:

- 199.615B: `39c13e6a54dd6609d033cc118cde3a0bd42a2b1e128eea64925d387b961f9913`
- 400.942B: `c1f1fcb98099e7d141b998dda810baaa76208d36e3f1cc749c1b6ddddf370895`
- 513.014B: `e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857`

These anchors support the retained continuation history. They do **not** yet provide the complete job-by-job or epoch-by-epoch graph between 199.615B and 400.942B, nor a continuous record of accepted losses between anchors.

### Clock to use on the plot

Use the accepted-token authority saved with each retained checkpoint. Do not multiply the absolute worker step by the final world size.

For a fixed-world epoch, the implementation advances the counter after a successful optimizer update by:

`world_size × batch_size × chunk_size × grad_accum`

At 2,048 ranks, B4, context 2,048, accumulation 1, this is **16,777,216 accepted tokens per update**. The anchors satisfy:

- `150,793,748,480 + (2,303,840 − 2,300,930) × 16,777,216 = 199,615,447,040`
- `199,615,447,040 + (2,315,840 − 2,303,840) × 16,777,216 = 400,942,039,040`
- `400,942,039,040 + (2,322,520 − 2,315,840) × 16,777,216 = 513,013,841,920`

The first equality has an explicit migration receipt. The later equalities are arithmetic consistency checks; they do not independently prove every intermediate epoch's world size or resume history. Total retained Frontier increment is **362,220,093,440 tokens** beyond the seed.

The historical step-2,303,840 checkpoint lacks an embedded token counter; its one-time receipt supplies the trusted 199.615B bootstrap. New checkpoints store `total_tokens` with model and optimizer state. A failed epoch rolls back to the retained checkpoint's counter. Work performed and subsequently discarded must not be added to the accepted-token x-axis. Tokens here are the implementation's aggregate training-token clock, not distinct corpus tokens, total consumed compute including failed work, or an independently counted held-out scoring denominator.

## 3. What the loss and performance numbers mean

### Final loss

At the recorded execution commit, each rank accumulates its local losses. Only rank 0 prints the training metric and writes the final checkpoint. Each printed loss covers `log_every=10` local optimizer updates. The final checkpoint's `loss` is the arithmetic mean of the last 100 logged windows, ordinarily **1,000 local updates** for this long final epoch. With the nominal final-world token rate, that window spans **16.777216 billion aggregate accepted tokens**.

The source computes this average from training-time losses before the final schedule-free `optimizer.eval()` checkpoint export. Therefore:

- label it **“final recorded training-loss average, rank 0”**;
- do not call it a fresh evaluation of the final consensus/averaged checkpoint;
- do not call it held-out bpb, validation loss, or a cross-rank mean;
- do not compare its smoothing window directly to the workstation run's 25-step logging interval without respecifying a common token-width window;
- preserve the corpus transition when discussing changes in its numerical value.

The final arguments disable held-out curves and final held-out/train evaluation (`heldout_curve_every=0`, `heldout_tensor=null`, `final_heldout_eval=false`, `final_train_eval=false`). No held-out curve should be inferred from this export.

### Job 5134243: measured 256-node production segment

The committed validator records 256 nodes, 2,048 initialized/resumed/finalizing ranks, 291 finite loss/throughput records, and 73 real K40 hierarchical RCCL merges. The allocation completed `0:0` in **01:46:49**, with `Partition=batch`, `QOS=debug` explicitly preserved.

| Metric | Result | Interpretation |
|---|---:|---|
| Median inter-merge interval | 72 s (71–87 s) | 72 intervals, from the 73 merges |
| Median merge duration | 6.320 s (5.068–15.227 s) | 73 measured merge durations |
| Last-10 median reported global throughput | 9,410,375.5 tokens/s | Ten logged windows at the end of this particular segment |
| Last reported global throughput | 7,683,935 tokens/s | One logged window, not the segment average |
| Retained token increment | 48,821,698,560 | Seed-to-final difference from the migration receipt |

The source calculates `global_tok/s` as rank-0 measured token rate × world size, with merge time inside the logging interval. It is **reported aggregate throughput**, not a separately instrumented sum of all ranks. The 9.410M value is not a full-run average, not the final 513B job's measured throughput, and not proof of near-linear scaling.

### Final job 5157856: checkpoint success and shutdown failure

The current public manifest records all 2,048 ranks ready for finalization, final consensus merge 167, and atomic publication complete. It also records **`FAILED_AFTER_FINAL_CHECKPOINT`, exit `143:0`, and a post-checkpoint shutdown fault**. The corrected audit independently read the embedded payload metadata. Report the retained valid endpoint and the teardown failure together; do not describe every production allocation as a clean completion.

### Independent systems qualification ladder

The later qualification used source `0a49bb6fabb4084a2a465950ad2801fb60b69c88`, B2/context 2,048, pure DiLoCo K40, hierarchical groups of eight, 160 steps / four merges per rung:

| Arm | Nodes / ranks | Job | Accepted tokens | Mean merge duration |
|---|---:|---:|---:|---:|
| E97-MLP | 8 / 64 | 5215231 | 41,943,040 | 8.7815 s |
| E97-MLP | 32 / 256 | 5215343 | 167,772,160 | 9.9767 s |
| E97-MLP | 256 / 2,048 | 5215375 | 1,342,177,280 | 10.0635 s |
| GDN2-MLP | 8 / 64 | 5215056 | 41,943,040 | 8.6275 s |
| GDN2-MLP | 32 / 256 | 5215081 | 167,772,160 | 9.1972 s |
| GDN2-MLP | 256 / 2,048 | 5215097 | 1,342,177,280 | 10.8178 s |

All six completed `0:0`. These are short, changing-aggregate-workload qualification rungs, not a strong-scaling benchmark. The document's opening linear-arm verdict is historical: its follow-up records the repaired non-affine per-head RMSNorm arm passing a 256-node gate in job 5217377. Do not quote the opening “FAIL CLOSED” as the final state without the follow-up, and do not merge the repaired control into the original unmodified linear arm.

## 4. Proposed figures and present data availability

### Figure A: extended E97 training on Frontier — main results figure

**Target:** training loss versus cumulative accepted tokens, with a compact configuration strip identifying hardware, corpus, and K changes.

**Required construction:**

1. Parse each epoch's actual command/arguments, resume checkpoint hash/step/token count, rank-0 loss windows, timestamps, merge markers, and final checkpoint receipt.
2. Construct a directed retained-checkpoint lineage. Keep superseded or rolled-back work separate from the accepted trajectory. Never deduplicate solely by worker step across unrelated runs.
3. Within a verified fixed-world epoch, map each logged step from the epoch's accepted-token anchor. Check every segment end against its retained checkpoint counter.
4. Show the observed ten-update windows lightly; optionally add a trailing mean with a disclosed **token-width** window. Do not smooth across corpus changes, resets, missing intervals, or unreconciled lineage boundaries.
5. Keep the existing matched workstation comparison as its own panel or preceding figure. Mark the Pile → CommaPile change explicitly and avoid a visual implication of a controlled same-corpus improvement.
6. Use breaks for missing intervals. Show checkpoint anchors separately if only milestone data are available; do not connect four checkpoint losses into an invented measured curve.

**Available now:** four lineage anchors and the final export's exact metric; 291 loss records are attested for job 5134243 but their individual values are absent from its committed verdict. The public 513B export contains no raw loss history.

**Status:** full curve blocked on raw log recovery. The PNG `docs/experiments/figures/e97_diloco_loss_curve_20260623.png` is an eight-GPU workstation curve and is not a substitute.

Suggested caption once the data are recovered:

> Extended training of the 1.2866B-parameter E97-MLP model on Frontier. The horizontal axis follows retained-checkpoint accepted-token accounting from the 150.794B-token workstation seed to 513.014B tokens. Losses are rank-0 training windows; smoothing and any missing intervals are disclosed. The configuration strip marks the transition from the Pile workstation run to CommaPile continuation and the change from eight-rank K250 to 2,048-rank K40 DiLoCo. This single-arm continuation is separate from the matched E97/GDN2 workstation comparison.

### Figure B: measured communication over a production segment — ready from committed data

**Source:** `collective_duration_ms`, `merge_steps`, and `k40_cadence_seconds` in the job-5134243 verdict.

Plot the 73 merge durations versus merge index or accepted-token location, with a separate cadence panel for the 72 between-merge intervals. Use the migration receipt to map token positions. This can be reproduced without Frontier access, model tensors, or new training.

Suggested caption:

> Hierarchical RCCL communication during the first final-seed 256-node E97 continuation (job 5134243). Seventy-three K40 merges had median duration 6.320 s; the 72 between-merge intervals had median duration 72 s. The allocation resumed the released 150.794B-token seed and retained a checkpoint at 199.615B tokens. These are measurements from one production segment, not a scaling-efficiency estimate.

### Figure C: qualification across machine sizes — optional systems appendix

Plot the E97 and GDN2 mean merge durations in the six-row ladder above. Label each rung “160-step qualification, four merges,” and include B2, group size eight, and the source SHA. Do not add confidence intervals from unavailable individual observations; do not combine these means with the production segment's B4/group-size-four median as though they were one controlled scaling series.

## 5. Concrete manuscript insertion plan and suggested text

### A. Language-modeling results: add a visible subsection

**Canonical insertion anchor:** in `paper/main.typ`, after the paragraph beginning “Each arm has one training seed” and its matched-token interpretation, immediately before the heading **“The convergent-loss null already appears at search scale.”** At the pinned baseline this is after the matched-token table and before search-scale/provenance material. Use the text anchor rather than unstable line numbers.

**Reader-first counterpart:** add after that version's primary matched-token result and before model-selection provenance. Do not bury the entire Frontier result in limitations or an exclusions list.

Proposed heading: **Extended training on Frontier**

Suggested main-text paragraph:

> The 1.2866-billion-parameter E97-MLP model was also continued on Frontier from the released 150.794-billion-token workstation checkpoint to a retained checkpoint at 513.014 billion accepted training tokens. The verified 256-node production segment used 2,048 accelerator ranks, batch size four and context length 2,048 per rank, schedule-free optimization, and hierarchical DiLoCo averaging every 40 local updates. The final export also records the 2,048-rank K40 configuration. The final export records a rank-0 training-loss average of 2.2798 nats/token over its last 100 logging windows. This extends the evidence for sustained training of the nonlinear recurrent architecture at large aggregate token count and machine scale. The final Frontier segment trained on CommaPile mainmix, whereas the paired workstation study used The Pile; the E97/GDN2 comparison therefore remains the matched 150.79-billion-token workstation experiment.

Suggested accompanying protocol paragraph:

> We track progress with the accepted-token counter stored alongside the retained model and optimizer state. The workstation seed contains 150,793,748,480 tokens. A one-time migration receipt binds the first Frontier continuation's legacy checkpoint to 199,615,447,040 tokens, after which the embedded counter governs resumption. The final run resumed at 400,942,039,040 tokens and published its checkpoint at 513,013,841,920 tokens. The final loss is a training-log statistic rather than a held-out evaluation. Atomic checkpoint publication completed before a distributed teardown fault, so the retained endpoint and the scheduler's failed terminal state are reported separately.

If the raw curve is still unavailable at integration time, present the anchor table and say its continuous trajectory is awaiting log recovery. Do not cite Figure A as if it already exists.

### B. Systems results: add one compact measurement paragraph

**Insertion anchor:** the Frontier/backend evidence paragraph near the end of `paper/main.typ`'s systems section, currently ending “paired backend evidence remains pending.” Replace that blanket phrasing with the distinctions actually supported: there is now short paired systems qualification, but not a long matched Frontier learning comparison.

Suggested paragraph:

> A 256-node production continuation initialized all 2,048 ranks from the same released seed and completed 73 hierarchical K40 merges, with median merge duration 6.320 s and median inter-merge interval 72 s. Its last ten logged windows reported a median aggregate throughput of 9.410 million tokens/s. Separately, E97-MLP and GDN2-MLP each passed serialized 8-, 32-, and 256-node qualification runs with four K40 merges and validated checkpoint token counts. These observations establish execution and checkpoint/communication behavior at the tested machine sizes; the short qualification runs do not measure matched long-run learning or scaling efficiency.

Place Figure B beside this paragraph or in the appendix; place Figure C only if systems detail warrants it.

### C. Introduction / abstract: optional single-sentence visibility

After the primary matched-token result, a restrained sentence is sufficient:

> A separate Frontier continuation extends E97-MLP training to 513.0 billion accepted tokens on 256 nodes.

The body must immediately supply the CommaPile and metric distinctions. Do not use “outperforms,” “scales linearly,” “513B matched comparison,” “held-out 2.2798,” or “513B tokens trained from scratch on Frontier.”

### D. Limitations and reproducibility: reconcile, do not erase

Retain the manuscript's accurate statement that the 513B continuation has no matched GDN2 control. Cross-reference the new positive result rather than leaving the continuation visible only as a limitation. Update the backend evidence wording to distinguish paired short systems gates from unperformed paired long-run learning. Keep the absence of a full recovered loss series, a full epoch provenance chain, and a matched held-out endpoint explicit in the reproducibility note.

## 6. Smallest remaining data request

Recover **text/JSON metadata and logs only**, from the established stable run directory:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/runs/e97-final-seed-production-256n/`

The launcher writes the concrete relative paths below; the final arguments identify execution epoch 18. Verify that these paths still exist before treating them as recovered evidence.

1. `epochs/epoch-*/launch.env`, `train-command.txt`, `train.out`, and `train.err`, covering the retained seed-to-513B lineage. If full stdout is unwieldy, retain every rank-0 step/loss/timestamp/throughput line, merge line, resume line, and checkpoint/finalization line with original line numbers and hashes.
2. `supervisor/execution-epoch.txt` and epoch outcome/identity records sufficient to identify each resumed checkpoint and any discarded work.
3. Per-epoch source/payload hashes, exact arguments, Slurm job IDs, and scheduler accounting; ensure partition and QoS are separate fields. This is needed especially for the 199.615B → 400.942B middle interval and the final job's teardown.
4. Small milestone manifests / metadata receipts at 199.615B, 400.942B, and 513.014B, with checkpoint hashes and embedded-token inspection results. The multi-gigabyte checkpoint payloads are not needed to draw the training curve.
5. Corpus identity/cache manifest and launch evidence for the actual transition from The Pile to CommaPile; verify exact source digest and sampler/resume semantics before calling the transition's location established.
6. For any **new held-out** or matched-control claim, obtain a separately specified evaluation or experiment. That is additional scientific work, not something the existing training logs can supply.

## 7. Integration acceptance checklist

- [x] Exact 513B E97-MLP architecture and public sidecar hashes verified freshly
- [x] Workstation seed hash bound to the first Frontier continuation
- [x] Explicit legacy-clock migration and final-job resume/end token anchors identified
- [x] Corpus, world-size, logging-window, and DiLoCo-cadence differences disclosed
- [x] Final recorded loss distinguished from validation and checkpoint evaluation
- [x] Post-publication teardown failure retained in the narrative
- [x] Existing workstation PNG correctly identified
- [x] Reproducible production merge-time series and short-ladder summaries located
- [ ] Complete retained epoch lineage and raw Frontier loss windows recovered
- [ ] Figure A regenerated from verified accepted-token records, with gaps/configuration changes visible
- [ ] Final source/data manifest committed with any generated figure
- [ ] New subsection and cross-references reviewed before canonical manuscript integration

No new training, private-compute access, checkpoint payload download, canonical manuscript edit, or merge was performed to prepare this integration plan.
