# X34 LM pilot: pre-registration

Registered 2026-10-27 before training. Part 1 is engineering and smoke evidence;
no smoke result is a test of the scientific hypothesis. No full run is authorized
by an apparent smoke advantage. X28E boundaries do not imply learned LM ability.

## Arms and training

Six pre-norm transformer blocks, d_model=256, four MHA heads, FFN width 1024,
RoPE positions (no learned position ceiling), training context 512. Untied input
and output embeddings, byte-level BPE vocabulary up to 32768, no dropout.
Tape in block 3 (zero-index 2), four heads, 2944 slots EACH, other X31 defaults.
Import X31 unchanged, including frozen key/write projections. Residual output
projection maps the 512-dimensional tape read to 256. X31 custom autograd uses
recorded routing decisions and reverse replay; never eager full-graph BPTT.
Baseline replaces this lane with a two-layer GELU MLP whose width minimizes
TOTAL parameter discrepancy; report total and trainable counts separately.
Target 15–40M total parameters and <10% total discrepancy. Matching parameters
is not evidence of matching compute: record seconds/token and estimated dense
MACs; the fused recurrent lane's actual overhead is a potential confound. A
compute-matched scientific verdict is withheld until measured cost is matched.

TinyStories official train/validation text, downloaded from Hugging Face;
validation excluded from tokenizer fitting. Fit BPE on the full training text,
then use a deterministic prefix of 200,000,000 packed training tokens for the
pilot; the full validation text is packed. This exceeds the 100M budget and
avoids repeat exposure within the registered full run. Byte BPE trained on train only;
fixed corpus/tokenizer SHA256 receipts. If download fails, flag deterministic
synthetic fallback prominently; never mix fallback evidence with real-text claims.
Seed 3401 for data ordering and initialization; deterministic contiguous packed
training blocks in identical order, repeat a fixed affine block permutation each epoch. AdamW
lr=3e-4, betas=(0.9,0.95), weight_decay=0.1, clip=1, fp32. Effective batch 8,
4096 target tokens/update. Both arms same updates, order and optimizer. Smoke
512 updates = 2,097,152 tokens/arm. Full 24,415 updates = 100,003,840 tokens/arm.
Validation every 64 updates (same fixed 8192 validation tokens); checkpoint
every 64 updates and on wall censoring. Resume includes optimizer, counters,
RNG and elapsed time. Hard cumulative 43,200 GPU wall seconds/arm, including
validation/checkpoint costs. Stop before a new update after deadline, report
overshoot from current update; no restarting the clock. Compare only common
completed token counts. Failures retain last checkpoint/log and are not successes.

## Evaluations and numerical predictions

1. Report held-out NLL and exp(NLL) at every common 64-update milestone. Full-run
prediction: tape perplexity <=0.98 times baseline at 100M; smoke has no target.
2. Zero-shot literal text a^k b^k (no probe fine tuning or label head). Score mean
next-token log probability INCLUDING end-of-document. Compare each valid string
with symmetric corruptions j=k±max(1,round(k/8)), normalizing by target token
count, reporting both raw total and normalized scores. Ties within 1e-6 nat count 0.5. ID k:
16,32,64,128; OOD 2x:256; 4x:512; 9x:1152. Report per-depth and pooled bins,
valid-minus-corrupt margin and paired accuracy. Tokenization is ordinary BPE;
record token lengths so byte depth is not confused with token depth. Strings
are artificial out-of-distribution text; length and tokenization biases remain.
Prediction: tape >=75% OOD pair accuracy, base <=55%, tape ID >=75%. Honest
null: if both OOD <=55%, say 'the tape lane is decorative at this scale'; name
observed occupancy/pressure and whether the lane ignited. A zero-shot failure
is not a falsification of X28E's trained state-tracking boundary.
3. Context decay: held-out nonoverlapping windows of 512,1024,2048,4096 tokens,
reset tape at window start; causal attention across entire window. Report NLL,
perplexity, per-position NLL and counts. Same text prefix and windows per arm;
no sliding-window double counting. Prediction tape NLL degradation at 4096
relative to 512 <= base degradation minus 0.02 nat/token.
4. Record pressure mean/max/nonzero frequency, exact peak live count from
allocation/free traces and final live slots per head, routing counts every step
and held-out window. Prediction story peak <2944/head and pressure nonzero
fraction <1%; this may mean the budget is underused, not that state is unnecessary.
Training resets state every 512 tokens: filling 2944 slots is impossible there;
only long evaluation can test budget saturation. Packed training/evaluation windows
may cross EOD boundaries within a window; state resets at window start. Also
evaluate the first 16 held-out stories individually, resetting between stories,
with a 4096-token cap and explicit truncation receipts. Short stories cannot
reach the budget; absence of pressure there is not a bound on all language.

## Part 1 evidence

Generated machine receipts live under receipts/; untracked data/checkpoints/logs
under runs/. Tests and smoke completion must be read literally. No full-run
verdict is part of this commit. See STATUS.md for achieved versus pending scope.

## Compute matching interpretation (registered before smoke)

At context 512 and vocabulary 32768 the analytical dense/scoring MAC estimate
is recorded in receipts/compute_estimate.json. It conservatively includes tape
scoring against all context-reachable slots; both arms are within 10% by this
estimate. It excludes scalar routing, softmax, journal traffic and launch costs.
We therefore distinguish approximate arithmetic matching from measured GPU
cost. Report actual seconds/token with compilation warmup visible, and withhold
a claim of measured compute equivalence if the replay lane costs substantially
more. No dummy FLOPs, sleeps or repeated baseline operations are used to hide
this overhead. Optimizer steps and data exposure remain exactly matched.
Common transformer/embedding/output tensors have identical initialization in
both arms; a fixed per-block RNG isolates replacement-lane initialization.
