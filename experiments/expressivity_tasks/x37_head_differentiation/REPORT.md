# X37 — head differentiation mechanism test

Primary final identical-peak counts: b 16/16, c 16/16, d 0/16 (d init 2/16). 3/3 new arms are >5% worse in near-matched-token perplexity than X34. W_in separation and occupancy differentiation are separate outcomes.

X34 control was reused without training. Three new arms each received 100,007,936 TinyStories tokens on the optimized X36 part-4 integrated replay lane (gated b30edb8b snapshot).

| Arm | Init identical stories | Final identical stories | Mean final peak/head | PPL | vs control | CUDA event step ms |
|---|---:|---:|---|---:|---:|---:|
| a: X34 reused | — | 16/16 | [119.625, 119.625, 119.625, 119.625] | 6.04476 | 1.00000x | historical |
| b | 11/16 | 16/16 | [119.625, 119.625, 119.625, 119.625] | 6.74948 | 1.11658x | 280.668 |
| c | 11/16 | 16/16 | [119.625, 119.625, 119.625, 119.625] | 6.69298 | 1.10724x | 280.956 |
| d | 2/16 | 0/16 | [58.625, 101.5625, 119.625, 119.625] | 6.68268 | 1.10553x | 2212.829 |

Primary endpoint is the same first 16 held-out stories as X34; all raw per-story occupancy/routing/pressure data are in eval_* receipts. Identical peaks are a coarse statistic and do not establish identical computation.

**Source audit changes the mechanism interpretation.** X34 head rows are already unequal at initialization (control.json receipts every projection) and at the trained endpoint (control_checkpoint_audit.json). Copying the full W_k matrix into W_q aligns queries with keys within a head; it does not replicate the independently initialized head slices. The control uses learned transformer hidden states directly, not a token-only frozen feature table. F12g write corrections select age-1 slots, not a write-content argmax. Thus the purported identical-parameter permutation fixed point is not established by the 16/16 occupancy receipt. X37 tests the registered new interventions; it cannot prove escape from an exact weight-symmetry fixed point that the control does not occupy. From an empty store with continuous writing, nonbinding capacity, no strength deaths and nonzero correction residuals, F12g alternates APPEND and age-1 CORRECTION, yielding ceil(T/2) peak-live independently of the head geometry. Pops/deaths, write-disable events and capacity pressure can break that occupancy pattern; read differences alone do not establish an escape from it.

| Arm | Frozen+learned total | Trainable | Matched base total | Discrepancy |
|---|---:|---:|---:|---:|
| b | 30204176 | 21781260 | 30204199 | 0.0000761% |
| c | 30269712 | 21846796 | 30269863 | 0.0004988% |
| d | 30269712 | 21846796 | 30269863 | 0.0004988% |

Each matched base count is from a separately widened replacement MLP in the same six-block skeleton; these bases were not trained. Frozen embeddings are counted. The original X34 base is not claimed to match the enlarged frozen tables.

| Arm | Final R-CONFLICT per head | R-NOMATCH per head | R-CAP per head | Mean pressure per head |
|---|---|---|---|---|
| b | [1905, 1905, 1905, 1905] | [1898, 1898, 1898, 1898] | [0, 0, 0, 0] | [0.0, 0.0, 0.0, 0.0] |
| c | [1905, 1905, 1905, 1905] | [1898, 1898, 1898, 1898] | [0, 0, 0, 0] | [0.0, 0.0, 0.0, 0.0] |
| d | [1515, 1904, 1903, 1905] | [1503, 1896, 1898, 1898] | [785, 3, 0, 0] | [0.0965430780488532, 1.0975814802804962e-05, 0.0, 0.0] |

| Arm | Final W_in distance from init per head | Pairwise distance range | Monotonic at registered evaluations |
|---|---|---|---|
| b | [] | N/A | [] |
| c | [4.391904354095459, 4.547687530517578, 4.636148929595947, 4.498025894165039] | 5.456006–5.746367 | [True, True, True, True] |
| d | [3.94462513923645, 4.745864391326904, 4.699118137359619, 4.44852352142334] | 5.269196–5.938251 | [True, True, True, True] |

All W_in layers start with the same identity slice; geometry is independently seeded. Pairwise W_in distances therefore measure acquired separation, not a nonzero random initial offset. Nonmonotonic movements remain reported. Gradient norms are taken after clipping at evaluation boundaries.

Winner by final 8192-token validation NLL: **d**. The complete X34 depth/context/story suite is in winner_probe.json.

The combined X31 oracle/X36 regression suite passed 141 tests before training; four X37 tests verify frozen ownership, nonzero distinct W_in gradients, ReplayBackward for b/c/d, and exact independent-head composition. The initial test collection error and its successful correction are retained. After concurrent upstream edits triggered the boundary gate, all 109 X31 oracle tests and four X37 tests passed again under the pinned source loader (113 total). Its initial Triton source-lookup isolation failure occurred before any c optimizer update; both that failure and the corrected source/linecache validation are retained. No numerical gate was relaxed.

Actual CUDA-event step samples are in runs/*/curve.jsonl. They exclude input transfer before the start event and evaluations/checkpoint saves. Host dispatch timings are asynchronous and are not wall-step latencies. train_* receipts retain loop wall time (including evaluation/save boundaries, excluding initial setup and hashing), hardware and broker-granted card. Synchronization occurs at reporting boundaries, not for each step; X37 does not claim a new zero-sync profiler certification.

Residual risks: X34 heads already have unequal initialized parameters. Age-1 write addressing can synchronize occupancy despite different reads. B64 vs historical B8 changes optimizer update count. Endpoint differs by 4096 tokens; quality comparison is near-matched. Widened base_lm controls are parameter receipts, not newly trained baselines. Policy hierarchy changes budget, pop and decay together; no causal isolation. Batch size changes the number of optimizer updates by 8x; quality comparisons cannot isolate head geometry from that optimization confound. Story telemetry is 16 deterministic examples, not a population confidence interval. Policy ordering is partly imposed by budgets; decay horizons exceed many stories. Geometry is frozen at width 64; this is one registered design, not a hyperparameter sweep.

No imported X31/X34/X36 sources were edited by X37. Concurrent X36 work changed the shared working tree and correctly triggered the arm-boundary gate after b. Continued c/d execution used the exact byte-verified source archive through receipts/pinned/run_leased.sh; the completed b checkpoint received no further optimizer updates. pinned_execution.json records source drift; training_console.txt retains the boundary refusal. gate_sources.json pins the runtime bytes for all arms. No model checkpoints are committed. No WG machinery, delegated agents, Frontier/Slurm or resilient behavior was used. Resilient requirement IDs: N/A. AGENTS.md and CLAUDE.md were checked synchronized. No callable contact_supervisor tool was exposed; coordination attempts and source-audit discoveries are recorded in receipts/coordination.txt.

Prediction adjudication (see receipts/predictions.json):

- primary_control: met; expected 16/16 identical.
- primary_b: FAILED; expected modest fall below 16/16.
- primary_c: FAILED; expected <8/16.
- primary_d_immediate: met; expected immediate structured occupancy differentiation.
- b quality within 2%: FAILED; >5% worse: True.
- c quality within 2%: FAILED; >5% worse: True.
- d quality within 2%: FAILED; >5% worse: True.

Winner probe outcomes (same zero-shot suite as X34):

| Depth bin | Pairs | Discrimination accuracy |
|---|---:|---:|
| ID | 8 | 0.5000 |
| 2x | 2 | 0.5000 |
| 4x | 2 | 0.5000 |
| 9x | 2 | 0.5000 |

| Context | Windows | Perplexity |
|---|---:|---:|
| 512 | 2 | 7.12042 |
| 1024 | 2 | 8.41630 |
| 2048 | 2 | 11.73576 |
| 4096 | 2 | 19.67201 |

One training seed was run per arm. The registered outcomes and source audit support these specific mechanism tests; they do not establish broad specialization or quality equivalence.

The heterogeneous-policy d arm measured **7.876x** c's median CUDA-event step duration. It uses four independent scalar-policy scans and different budgets/pop/death behavior; these contributions were not independently profiled or causally separated. The operator estimate of quick minutes per arm did not hold for d. The full registered token budget was retained without upstream patches or reduced batching.

Additional prediction adjudication:

- b routing-mix divergence: FAILED.
- c routing-mix divergence: FAILED.
- d routing-mix divergence: met.
- c W_in movement, pairwise separation and sampled monotonic growth: met.
- d W_in movement, pairwise separation and sampled monotonic growth: met.

---

## Parent adjudication — quality bands re-adjudicated against the matched-exposure control

The registered quality comparison ("within ~2% of control") used the X34 tape_lm
control, which trained at BATCH 8 (24,415 optimizer steps at 100M tokens) while
all X37 arms trained at BATCH 64 (3,052 steps) — a batch/step confound the X39
part-6 session receipted when it trained the protocol-matched B64 control
(x39_memory_usage/receipts/part6_family_comparability.json; all protocol fields
verified: batch, seed, context, tokenizer, train/valid data, tokens).

Matched-exposure perplexities at 100,007,936 tokens (B64 control = 6.8564):

| arm | perplexity | vs matched control | registered band (within ~2%) |
|---|---:|---:|---|
| b (seeds) | 6.7495 | **-1.6%** | PASSES (better than control) |
| c (seeds + W_in) | 6.6930 | **-2.4%** | PASSES (better than control) |
| d (+ policy hierarchy) | 6.6827 | **-2.5%** | PASSES (better than control) |
| x38 λ=0.01 (diversity penalty, full) | 6.7340 | **-1.8%** | PASSES (see X38 report) |

Adjudication: the "3/3 arms >5% worse" verdict is a BATCH CONFOUND, not a
quality failure. Under matched exposure every intervention arm lands AT OR
BETTER than the control — and the arm with the strongest, structured
differentiation (d: 0/16, hierarchy-ordered occupancy) is also the best
perplexity in the family. Differentiation does not cost quality here; the
hierarchy arm is the family's best model on both axes.

Honest scope: the B8 control remains the best ABSOLUTE model (6.045 at 100M
tokens — 8x more optimizer steps); the matched-exposure statements are about
arms within the B64 protocol family, not about the batch-size choice itself.
The X37 agent's registered-comparison reporting was correct as written; this
adjudication supplies the comparison the registration intended (and that
X39 part 6 trained the control for, per its own amendment note).
