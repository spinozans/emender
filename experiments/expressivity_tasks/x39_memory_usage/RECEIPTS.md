# X39 — memory-usage regularization: pre-registration

Registered before any X39 training. Status: **blocked at mechanism preflight**.
This registration preserves the operator's four levers; it does not silently
replace their semantics. No X39 scientific outcome is claimed.

## Control and invariants

Reuse X34 `runs/pilot_tape_lm/latest.pt` and
`receipts/probe_pilot_tape_lm.json` without retraining. Six blocks, width 256,
four attention heads, FFN 1024, RoPE, tape at block 3, four heads with identical
policies, context 512, seed 3401, X34 tokenizer and packed data order.
AdamW lr 3e-4, betas (.9,.95), weight decay .1, clip 1, fp32.
No X38 diversity machinery. X31/X33/X34/X36/X37/X38 remain read-only.

The requested effective batch is 64; **X34 actually used 8** (4096 target
tokens/update). Grouping eight consecutive X34 batches preserves exposure order
but changes optimizer updates. Consequently an X34 checkpoint is not a
batch-matched control for the requested training. This discrepancy must be
resolved before claiming an otherwise matched experiment or selecting lambda.
100M at batch 64 rounds up to 3052 updates / 100,007,936 target tokens;
20M rounds up to 611 updates / 20,021,248 tokens. Record exact exposure.

## Requested arms

0. Frozen, already-trained transcript control; no retraining.
1. Only decay changes: `decay_rate = exp(log(1e-3)/128)` with the existing
   free threshold 1e-3. Receipt both requested double and float32 constants and
   the T2 horizon `log(free_threshold)/log(decay_rate)`. This is the lifetime
   of untouched strength-1 slots, not a guaranteed cap under refresh/POP.
2. Append cost: lambda in {1e-3, 1e-2, 1e-1}, 20M tokens per trial. Select
   the largest lambda with held-out perplexity <=1.02 times control at the
   exact same exposure; no qualifying trial means **no operating point**.
   A selected lambda gets an independent seed-3401 100M run. The detached
   residual magnitude must remain detached. **Blocked:** the existing gate
   scales appended values; it does not gate allocation. Discrete allocation
   rate has no differentiable logit path. A gate-strength penalty must not be
   labeled an append-rate penalty. Defining a new allocation decision and its
   gradient requires an operator decision and new oracle semantics.
3. Train with S=128/head, evaluate the same trained weights at S=128 and
   S=2944. All other policies stay symmetric. Receipt R-CAP directly; scarcity
   is not a guarantee of learned selectivity.
4. A single shared learned W_in before frozen per-head-seeded key geometry,
   adapting X37 arm-c machinery, without X38 objectives. **Blocked:** X31/X36
   write matching addresses the age-1 slot, not a key-similarity match; changing
   keys affects reading, not concept novelty routing. R-ZERO-RESIDUAL allocates
   a new slot too. Making repeated concepts stop appending requires changed
   routing semantics and oracle/replay certification, beyond a W_in change.

## Primary compression receipt

Evaluate T={512,1024,2048,4096}, two held-out packed windows per T, tape reset
per window, plus the first 16 validation stories (cap 4096 with truncation
flags). Preserve per-head mean live count over tokens, peak live, final live,
and occupancy profile; do not substitute final count for mean count.
Fit ordinary least squares of log(peak live) on log(T), separately per head,
using maximum peak across the two windows. No zero-count log pseudocount:
zero occupancy is reported as degenerate and its slope is undefined.
Slope near 1 is transcript behavior; <=0.5 is strong compression only when
quality also passes. Budget-capped slope alone does not establish compression.

X34's existing control receipt already covers all four lengths; analyze it
directly rather than spending another GPU run. It cannot supply mean occupancy
or profiles that were not recorded. Retain source hash and mark missing fields.

## Secondary receipts and quality bands

Distinguish (a) write-enabled fraction, (b) successful allocations/token/head,
(c) append requests/token/head including R-CAP, (d) sigmoid gate strength.
The operator's predicted control "append rate 1" describes (a), whereas
physical successful allocation is about .5. Pre-register selective successful
allocation <.5, but report all four and avoid a denominator switch.

Receipt per-head pressure nonzero fraction, mean/max, raw R-CAP fraction and
W64 saturation (>=90% R-CAP in a trailing 64-token window). X33 establishes
an exhaustion detector, sometimes 96 depth before collapse, not a general
health detector. At control T<=4096, ~T/2=2048 remains below S=2944:
zero pressure in a rich deployment at these lengths cannot distinguish it
from control. A longer saturation stress is a separate, clearly labeled probe.

At matched tokens and fixed X34 validation windows, register short-context
perplexity ratios <=1.05 for decay, <=1.02 for append cost, poverty (each
deployment budget separately), and semantic keys. These are acceptance bands,
not predictions of measured results. For decay additionally report the entire
per-position NLL curve, 128-token bins, and each context's perplexity ratio;
long-context degradation is expected and must not be hidden by averaging.
Use story occupancy profiles to distinguish recent retention from flat growth.

Nulls: no lambda operating point; unchanged semantic append behavior; scarcity
does not transfer to rich deployment; low slope solely from destructive
saturation; poor read quality despite reduced occupancy. All are valid outcomes.

## Execution gates

Use only `/home/erikg/emender/.venv/bin/python`. GPU work must be leased via a
local run_leased.sh wrapper using the broker, cards 0–6 only, excluding 7 and
the occupied/certified cards. Sequential arms. No GPU run is authorized by
this audit while the mechanism blockers remain unresolved. Before every
training invocation, require fresh oracle parity for its exact configuration,
assert ReplayBackward in its loss graph, and record source/config/data hashes,
parameter totals/trainable counts versus base_lm, physical GPU lease, and
preregistration hash. No .pt files in git. Failed gates remain failures.

The Intercom contact_supervisor tool was not exposed in this session; tool
discovery found no supervisor or intercom tool. Pending decisions are recorded
here rather than guessed: allocation gate semantics, semantic match/no-op
semantics, and the batch/control mismatch. Training is not implemented or run.
