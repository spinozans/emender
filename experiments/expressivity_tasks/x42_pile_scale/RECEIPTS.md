# X42 registered Pile scale infrastructure — 2026-10-28

This registration precedes all pilot/search runs. X40 gates runs, not construction.
No pilot/search is authorized without an attested pure/hybrid verdict. No claim
of trained performance, measured 48GB peak memory, or completed corpus is made.

## Sources and reuse

Read X34 model/recipe, X36 optimized replay receipts, X37-d hierarchy, X39
semantic wire/operating points. Read stage3_8m receipt inventory and stage5c
receipts/full_launch launcher and broker snapshot. Reuse scripts/gpu_lease.sh
(acquire/adopt, heartbeat keeper, release traps) through run_leased.sh, following
X39. Cards 0/1/5/6 only: certified 2/3/4 and forbidden 7 excluded; broker
excludes X40's held card. No WorksGood service/dispatcher is started. The broker
uses its existing .wg lock directory only. No DiLoCo behavior is changed.

The scale backbone generalizes X34 pre-norm LayerNorm, RoPE, causal SDPA,
GELU FFN, untied embeddings/output, .9/.95 AdamW, .1 decay, clip 1, 3e-4 LR.
The verdict-dependent lane is a factory that returns a width-preserving
residual and handles its own auxiliary gate cost. Current X39 cell hardcodes
256/4 heads: it cannot honestly be presented as a 2048-wide certified lane.
The bundled emender.py adapter projects scale hidden states into the existing
256-wide X39 semantic cell, reusing its frozen geometry and X36 per-head scans.
It supports pure FFN/memory blocks and hybrid attention/memory blocks. Its
W_in width is an intermediate 256->width->64 projection; this is a new adapter,
not an assertion of upstream scaled parity. Meta construction passes; GPU
parity/numerical validation remains pending X40. evaluator.py
provides the fixed-exposure evaluator and rejects omitted policy consumption
or missing rich-capacity occupancy/pressure profiles. A whole-model factory
supports pure families without silently retaining transformer attention.

## Data and tokenizer

GPT-2/GPT-Neo-compatible BPE: upstream gpt2 tokenizer.json from Hugging Face,
50257 vocabulary, EOD 50256, uint16 little endian. Saved tokenizer bytes and
SHA256 identify exact identity, not a fitted TinyStories tokenizer. Dataset
primary accessible candidate monology/pile-uncopyrighted, an explicitly
substituted uncopyrighted Pile subset; not full original Pile. Raw source and tokenizer revisions are pinned in data.py; upstream_revisions.json
records resolution. The access sample used the same current revision, with
produced-byte hashes retained.
Default tokenized budget 24 GiB (12.88B tokens, more than 1B/2B needs), 99/1
train/valid by document SHA256, no document crossing splits, exact dedup after
1M-character document cap. No fuzzy near-duplicate filtering. Packed windows
may cross document EOD; recurrent state resets every window. No overlap of
training windows; one extra next-token boundary token is shared. Streaming
source and mmap affine window sampler avoid corpus RAM residency. SQLite
exact dedup guard 2 GiB plus output cap and 10 GiB free-space reserve.
Short builds remain incomplete and are rejected for scientific runs.

## Single-card memory model

FP32 master parameters trained with bf16 autocast and fused AdamW. Conservative
18 bytes/parameter = bf16 forward copy 2 + master 4 + gradient 4 + moments 8;
~1.3B gives 2.6/5.2/5.2/10.4 GB respectively. Checkpointed activations,
logits/loss workspace, and 6 GiB CUDA reserve added separately. Memory receipt
uses meta-device exact model counts for each shape and asserts <44 GiB on
one 48 GiB card. No sharding required. Lane-specific state must be included
and revalidated after integration. Analytical arithmetic is not measured peak
memory; hardware measurement remains pending. Batch 256*1024 target tokens
per update, physical microbatch 1 and accumulation 256. Identical seed, corpus,
context, affine permutation, effective batch and rounded steps for paired arms.
Pilot ~148M (768x10), final ~1.3B (2048x22), GPT-2 vocab. No .pt outputs.
Failure JSON and registration/curve receipts survive failed runs. This initial
harness deliberately has no checkpoint/resume; an interrupted run fails and
must not be reported as a completed endpoint.

## CMA-ES pre-registration

Full covariance rank-one/rank-mu CMA-ES equations implemented locally in
cmaes.py; validate against Sphere and rotated anisotropic quadratic before use.
Proxy backbone pilot shape (~150M); fixed 150,208,512 target tokens/seed
(573 updates at batch 262144), seeds 3401/3402, same data prefix and validation
windows per candidate. Primary objective mean held-out NLL (equivalent ordering
to perplexity); report PPL too. Lexicographic secondary occupancy-vs-length
slope clipped [0,1], only within NLL quantization bucket 1e-4 nat. Evaluate
lengths 512/1024/2048/4096 with reset state, rich capacity and pressure reported;
capacity clipping is not compression. Baseline has no slope comparison.
Population 12, generations 8, optimizer seed 4201, initial sigma .2 normalized
coordinates, hard total 96 single-card GPU-hours, <=1 hour per candidate for
both seeds combined. Fail/stop if budget cannot complete fixed exposure; do
not reward censored candidates. Sequential execution, no parallel workers.

Vector: four decay horizons, write-gate bias, match_coef, pop threshold,
four head budgets, gate-cost lambda, W_in width, lane count and placement
fraction, LR and warmup. Exact numerical bounds/center in search.py. Log-space
positive scale coordinates, reflected bounds, rounded integer coordinates.
X37-d center horizons approximately 1000/10000/100000/1000000 and budgets
64/256/736/1472; X39 horizon128 motivates lower horizon100 bound, match -1,
gate cost .01 and explored .001/.01/.1; X34 LR center 3e-4. Expanded bounds
(including W_in32..256, lane count1..4, LR1e-4..1e-3, warmup100..2000) are
new X42 engineering choices, NOT claimed as previously validated ranges.
Per-lane policy broadcasting and placement mapping must be explicitly
receipted by the selected architecture evaluator; unsupported fields fail.

Scale-transfer assumption: constants selected at proxy scale apply unchanged
at 1.3B except placement fraction mapped to layer index. This is an unproven
assumption and known limitation. No post-hoc retuning concealed as transfer.

## Build evidence

See receipts/validation.json, receipts/data_attempt.json and memory.json.
Runs and adapter GPU validation are pending X40. Rich occupancy evaluation
uses budget>=8192/head separately from training budgets; candidate policy
state is reset and original budgets restored after profiling. Meta memory
receipts include the bundled lane journal/state workspace estimate. Per-lane
horizons map to exp(-1/horizon); these are e-folding constants, distinct from
X40's threshold-crossing horizon convention. W_in adapter initialization is
random rather than the X37 identity slice and is an explicit recipe change. CPU tests are engineering
checks only and cannot establish a model verdict.

First data access failed because zstandard was absent; pip itself was absent.
Installed zstandard with uv against the required interpreter and retried.
Exact attempt outcomes are in data_attempt.json; no full corpus completion
is inferred from the access sample.

Pilot control uses 10 blocks (148.1M); projected memory adapters use 8 blocks
(~147.5M), final uses 22 blocks (~1.314B control/~1.328B memory). This matches
parameter scale without claiming matched compute. The per-head policy vector
is shared across inserted memory blocks; it does not independently tune every
layer/head combination. Full 24 GiB token construction remains NOT RUN; the
256-document access sample is incomplete and rejected by the training guard.
