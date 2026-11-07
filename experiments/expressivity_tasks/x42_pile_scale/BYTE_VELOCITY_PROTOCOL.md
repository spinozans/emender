# X42 byte-native velocity registration — 2026-11-07

Operator decision: do byte level. This independent flagship preparation does not
alter the token velocity board or its nominations. Three independent CMA-ES
families: byte transformer on card 2, byte GDN2 on card 3, bounded pure byte
Emender on card 4. Card 1 token pure search is untouched; 0/5/6 stay reserved.
No WorksGood service, worker, dispatcher or completion machinery is used.

Each family has population 12, sigma 0.2, RNG seed 4301 and exactly 12 generations.
One candidate per family runs at a time, with 2700 synchronized active seconds
including byte assembly, transfer, all forward/backward accumulation, clipping
and AdamW. Setup and evaluation are separately outside that training clock;
complete-step overshoot is retained. Effective B64, T2048 bytes, seed 3401.
Natural raw bytes, vocabulary 256; no tokenizer or donor transfer. Candidate
weights initialize afresh. Total capacity including frozen geometry matches
150,003,984 within 0.1%; both total/trainable counts are receipted.

The JSON registration binds the complete source-ordered C14 pile_offsets.jsonl,
the documents.jsonl hash, the full binding receipt, source stat identity and
runner dependencies before candidate 1 trains. Corpus authority is the existing
C14 take-4 mapping: 7,831,213 matched receipts, two token-truncated exclusions,
7,831,211 admitted documents. pile.txt is opened read-only and read by offset;
all consumed document slices are SHA256-checked against the registered mapping.
No text is materialized. Omitted documents/gaps never enter inputs. Exhaustion
fails, never wraps. Each shifted input/target row stays within one document;
short tails are retained and grouped by exact length, without padding. Every
row starts fresh model state. Training uses source-order train documents.

Search validation uses the first 24 source-order valid-document windows at
T2048 (tails retained), weighted by actual target bytes. This is proxy search
validation, not the flagship final test. Pure occupancy slopes use the first
full valid-document window at each of 512/1024/2048/4096 bytes. No cross-document
window is allowed. Final 1.3B calibration/final-test must be document-disjoint,
frozen separately; proxy scores cannot be reported as final-test outcomes.

Baselines score held-out raw byte NLL only. Pure scores raw byte NLL +
0.01 * mean live occupancy fraction + standing degeneracy penalties: dead store
(any head touched fraction <0.01): 0.05; pressure runaway (R-CAP per target >0.05):
0.05; maximum per-head log-log occupancy slope >=0.9: 0.02. Raw NLL and components
are retained separately. Pure's detached gate-cost term is retained during
training and is not claimed to add management gradients.

Exact dimension names, centers, ranges and transforms are in the hash-bound JSON.
Architecture-level token-search spaces carry over: pure head counts, horizons,
slot budgets, input widths and per-head policies; transformer anchor width/depth
with capacity-solving MLP; GDN2 capacity solver and expansion 1/2. Baseline search
also carries initialization, beta2, weight decay and B64 microbatch divisor.
LR is honestly widened from [1e-4,1e-3] to [1e-5,3e-3], warmup to [10,8000]
updates for the byte landscape. Log/linear transforms and reflected unit-cube
sampling remain the token harness conventions. Pure uses fixed microbatch 8
with exact byte-weighted B64 accumulation; baselines search divisors 1–64.
Dense compute is BF16/FP32 masters, pure store remains certified FP32. FLA version
and dependencies are bound. The token nominations are not tuned byte baselines.

Unbounded pure is this search's winning policy with capacity released and proven
nonbinding over the registered scale lengths; it is not independently searched.
All three nominations must be committed and qualify at 1.3B before the six-arm
BYTE flagship launch gate opens. Exact scale configs, precision/operation checks,
Wikipedia XSA binding, extraction/storage and matched 24-hour allocation remain
in PREREGISTRATION_1P3B.md's final scale-launch gate.

Atomic CMA state retains population, RNG, covariance, completed scores and phase.
Child failure stops that family without a score; restart keeps banked candidates
and retries the failed candidate from scratch in a new attempt directory. Failed
attempts and the entire unsaved candidate span are retained as failures. No
partial candidate checkpoint is claimed. No .pt is committed. Detached launcher
uses Popen(start_new_session=True), DEVNULL stdin and retained logs, with broker
lease owner/starttime, heartbeat renewal and advancing steps checked before exit.
Artifact monitor checks runs/ log bytes, mtimes, step/byte counters and banked
scores; system/service flags are not progress evidence.

Forecast: 12 generations × 12 candidates × 2700s = 108 active hours per family,
4.5 days before setup/eval/contended-host overhead; at ~6 hours/generation the
optimistic pace is 72h/3 days but requires fewer charged seconds than this serial
population budget. Honest planning horizon is about 4.5–5+ days, not a guaranteed
3-day completion. Three cards run concurrently. No DiLoCo behavior changes;
resilient-compute-pool requirement IDs are inapplicable.
