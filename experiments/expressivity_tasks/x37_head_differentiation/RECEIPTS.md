# X37 preregistration — before training

Date: 2026-10-27. Operator-designed head differentiation mechanism test.
Control (a) is read-only X34 runs/pilot_tape_lm and
receipts/probe_pilot_tape_lm.json; never retrain it. Its primary outcome is
16/16 stories with identical four-head peak-live. This equality establishes
an occupancy fixed point; equality of model parameters is a separate claim
and will be checked rather than inferred from occupancy.

Predictions, registered before any X37 training:
1. Primary: count/fraction among the same first 16 held-out stories whose
four head peak-live values are identical. (a) remains 16/16; seed-only (b)
falls modestly; learned input (c) falls substantially, expected <8/16;
policy hierarchy (d) differs immediately with ordered occupancy means.
2. Per-head R-CONFLICT/R-NOMATCH/R-CAP counts diverge; pressure differs in d.
3. Matched-token perplexity b/c/d expected within 2% of a; >5% worse is an
explicit adverse result. Compare at the common endpoint and retain actual
token counts; do not call nearby endpoints exact token matches.
4. In c/d, per-head W_in distances from init expected to grow monotonically;
record every evaluation, all pairwise distances, per-head gradient norms,
and failures of monotonicity. Nonmonotonic optimization is not censored.

Use read-only X36 optimized integrated replay, X34 six blocks / block 3 lane,
TinyStories tokenizer and affine data order seed 3401, context 512, physical
batch 64, AdamW lr 3e-4 betas .9/.95 decay .1, clip 1. Train 3052 steps
(100,007,936 tokens), the smallest whole B64 batch >=100M. X34 used B8,
24415 steps / 100,003,840 tokens; batching changes optimizer updates and is
an explicit quality-comparison confound mandated by the requested protocol.
All arms use the same stream prefix; no new tokenizer/data download.

Frozen geometry implementation: independent head/token F12f embedding tables
with feature width 64, independent frozen 64x64 W_k/W_q seeds. Both matrices
start equal within each head (preserving the initial key/query alignment),
but differ across heads. b hashes token features only; c/d add learned
W_in (256->64, identity slice init) applied to the block hidden state before
the frozen geometry. Keys are detached as in X31; query/read gradients reach
W_in. Original learned content/value/pop/append paths remain. No routing
manager or routing gradient is introduced. Tables count as frozen parameters.
Parameter matching: create a separate, untrained X34 base_lm with its
replacement MLP widened for each arm; total count discrepancy <0.1%.
This receipts capacity, not a new quality baseline.

d policies head 0..3: budgets 64/256/736/1472, decay rates
.999/.9999/.99999/.999999, pop thresholds .01/.1/.5/.9. Run four X36 scans
because policy constants are scalar in its API; receipt this timing cost.
b/c retain four heads in a single X36 scan, budget2944 per head.
The hierarchy predicts occupancy by increasing budget/retention. Decay alone
at these horizons may not cause a death within a story; budget pressure and
pop thresholds are the immediate interventions. Never attribute budget-driven
differences to learning or decay alone.

Before training: full X31 oracle suite against X36 via its process-local
adapter; X37 gradient/read replay and policy composition checks. All arms
assert ReplayBackward on every training graph. Log actual step times, finite
loss/gradients, seed/config/source hashes, parameter counts, training and
validation curves. Evaluate all arms at init, steps 64/512/1536/3052 using
same 16 stories plus 8192 validation tokens and per-head telemetry.
Winner: lowest final validation NLL, tie within 1e-6 broken b then c then d;
full original X34 depth [16,32,64,128,256,512,1152], context
[512,1024,2048,4096] two windows, 16 stories on winner. Also retain all-arm
final stories/telemetry. No winner until all three complete.

Null differentiation is a valid negative. Actual F12g age-1 routing is
architectural, not write-content argmax. Seed changes may therefore leave
occupancy unchanged even if parameters and reads differ. Report this
possibility rather than treating peak-live equality as proof of equal weights.

Operational: exact /home/erikg/emender/.venv/bin/python; broker wrapper only;
physical whitelist 0,1,5,6 excludes certified 2/3/4 and forbidden 7. No WG,
Slurm, Frontier or resilient-training changes; resilient requirement IDs N/A.
x31/x34/x36 imports remain read-only. No .pt files committed.
