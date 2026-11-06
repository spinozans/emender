# Unbounded Emender velocity family — DRAFT, no search launch

This draft registers the family concept and proposed numerical choices. It is
not executable launch authorization. Phase 3 measures the C14 training path;
it does not silently promote that 2.13M-parameter fixture to the 150M proxy.
The existing X42 search on card 1 remains unchanged.

The proposed proxy has 12 memory/FFN blocks, width 768, approximately 150M
trainable parameters, raw byte vocabulary 256 and no attention. Retain the
C14 consultation adapter and document-local cache identity checks. Inputs,
shifted targets, consultation contexts and held-out sequences remain inside
one document. Reset all state for each window; mask truncated/padded targets.
Use document-disjoint calibration and final test. Bind the exact corpus
`documents.jsonl`, source bytes and cache journal before executable registration.
The existing GPT-2 tokenized X42 corpus is not an acceptable substitute.

All READ/APPEND/POP/DECAY/CORRECT/MERGE operations remain enabled. The capacity
is a safety allocation bound rather than a tunable policy budget. Each token
can allocate at most one stream and one array row, so choose cap = 2*T+1 per
head/window: 1025 at training T512, 4097 at evaluation T2048, and 8193 at the
T4096 length probe. Require no refused insertion or capacity merge; fail a
candidate that violates this. Duplicate/natural merges remain semantic ops.
Cap257 is nonbinding only for the receipted T128 training fixture; it must not
be copied unchanged to T512 or T2048. The allocation proof needs an executable
long-context oracle before launch.

Copy the current velocity registration's h_min, ratio, N, write_gate_bias,
match_coef, pop_threshold, w_in_width, gate_cost_lambda, lr, warmup_steps,
match_wire0..7, occ_onset0..7 and occ_slope0..7 dimensions and numerical bounds
verbatim. Remove budget_min and budget_ratio from the vector. The companion
JSON includes those retained ranges. Inactive head dimensions remain in the
vector. Freeze management, embeddings and key geometry as in the reference;
train the existing query/value/content/read-out/FFN paths. The C14 consultation
projection remains260-to-256 because it enters the cell after the backbone
input projection, whose output is256 at every width. Validate this wiring at
width768/2048 before launch. Donor initialization is a separate unresolved
executable registration gate, never an implicit transplant.

Proposed training: seed3401, contiguous verified training prefix, physical and
effective batch1, context512, 2700 active seconds plus one whole-step overshoot.
Include load/decode/verification/transfer, forward/backward, telemetry, clipping
and optimizer completion in the clock. Exclude setup, checkpoint/receipt writes
and evaluation, receipt every exclusion. Use FP32 store state, BF16 dense
operations, AdamW betas .9/.95, weight decay .1, clip1, searched LR and linear
warmup. The sustained C14 run instead uses its matched AdamW recipe (wd .01,
default betas); that difference is explicit. A completed proxy arm always
spends2700 active seconds; its wall duration and exposure are measurements,
not promises inferred from the C14 compute microbenchmark.

Keep the exact score utility and rule: minimize held-out NLL + .01*occ_frac,
plus .05 if any ever-touched fraction <.01, .05 if R-CAP events/token >.05,
and .02 if maximum per-head occupancy slope >=.9. Use 24 consecutive reset
held-out document-local windows T2048, equal-window NLL mean; probe lengths
512/1024/2048/4096. For this family normalize occupancy and touched rows by
its declared 2*T+1 cap at each evaluation length. This keeps the arithmetic
rule but changes the denominator versus the bounded family; report raw live
counts and cap alongside normalized metrics. Occupancy-pressure gate onset
and slope also use that declared cap. No cross-family quality equivalence is
claimed from sharing the arithmetic score. The detached gate-cost coordinate
still has no trainable-gradient effect under frozen management.

One full2700-second smoke precedes population12, one generation, full covariance
CMA-ES seed4201 sigma.2, with durable population/RNG state and failure receipts.
Select minimum exact score with stable candidate-index ties; compare strictly
against the full smoke. Failed, interrupted, nonfinite or shortened candidates
receive no substitute score. Draft cards3/5/6 only, broker leased, detached
launcher with internal owner/heartbeat and advancing-step verification. No
search, donor training, scale-transfer run or terminal run is launched here.

The terminal cost scenario uses the existing X42 final_pure budget class:
width2048, 22 layers, context1024, physical batch1, accumulation256,
2,000,000,000 targets and86400 wall seconds. That historical config is not an
unbounded-family terminal registration; the velocity nominations explicitly
leave terminal duration/corpus registration pending. Bind both before launch.

Prerequisites for promotion: exact long-context/width transfer oracles and
masked gradients; measured full proxy smoke with full telemetry and natural
occupancy; BS1024/2048/4096/8192 compile, memory and throughput receipts;
width768/2048 consultation wiring; fresh 1.3B memory/throughput qualification;
corpus/source/score/initialization hashes; supervisor/operator launch moment.
No resilient DiLoCo behavior changes; its gap-matrix IDs are inapplicable.
