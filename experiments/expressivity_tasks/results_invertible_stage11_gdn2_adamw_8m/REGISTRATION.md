# X11 — missing GDN2 + ordinary AdamW cell (pre-launch)

Registered before any confirmatory evaluation or launch. One authorized run:
`gdn2_adamw_seed0`, from scratch, deep-first T16, 500,000-update safety ceiling.
This is the scale historically labeled “8M”; actual stock GDN2 size is 9,887,056
parameters, matching X7b exactly. No architecture or kernel adapter is introduced.

## Fixed recipe and data

- Real `fla` GDN2, stock bf16 chunk kernel, fp32 parameters/bf16 autocast,
  dim256/layers4, p61/vocab3721, batch32, uniform target CE, clip norm1.
- Ordinary `torch.optim.AdamW`, exact X10-adamw factory and update function:
  betas(.9,.999), eps1e-8, all-parameter weight decay .01. In particular,
  A_log/dt_bias `_no_weight_decay` tags remain intentionally ignored, just as X10.
- Exact X9/X10 LR per update: linear warmup updates1..5000 to1e-3,
  cosine thereafter to1e-5 at500000; actual LR logged every update.
- Same seed0/init discipline as X7b: `torch.manual_seed(0)`, no checkpoint
  loaded; procedural training `numpy.default_rng(0)`. Same frozen 512-example
  T16 validation arrays, verified against BOTH X7b and X10 registration hashes.
  Validation is never trained on. Per-position P1–P16 joint correctness/CE at
  baseline and every500 updates; full frozen archive and array hashes retained.
- No ladder, carry supervision, stagnation kill, threshold stop or automatic
  extension/retry. Fixed-final500K is primary; first50/first512, contiguous
  perfect prefix and maximum-anywhere reported separately, transients descriptive.
- Entry + aggregate-best + latest/final only (at most3 retained checkpoints,
  plus one atomic staging file). Deletion hashes retained. Probe-derived memory
  and wall bounds are infrastructure guards; failures are censored, not losses.
  Disposable seed99 probe/GPU tests do not use confirmatory pools or select recipes.

## Predictions registered before launch

(a) If GDN2+AdamW reaches **5+ perfect positions**, architecture gains depth
from the optimizer too, and the P5+ question opens. **Any P5≥50/512 is material;
512/512 is perfect.** Report material P5 acquisition even without five perfect
positions; material and perfect are not interchangeable.

(b) If GDN2+AdamW matches **X7b's four perfect positions with no P5 gain**, GDN2
depth at this scale is bounded near4 under both regimes; E97/GDN2 reads
**parity-at4 with different optimizer sensitivity**. Report all other outcomes
explicitly rather than forcing them into (a)/(b).

(c) Either way the matched-recipe table **E97-SF / E97-AdamW / E97-Muon /
GDN2-SF / GDN2-AdamW** completes for the paper rewrite. One seed remains a
single-seed observation, not robust cross-seed evidence.

**P5 question:** X7b's P5=3/512 at ceiling means the P5 question may need longer
ceilings later. This run reports **what500K buys**, not an asymptotic depth bound.

## Launch and native watcher

`registration.json` binds sources, dependencies, this document, tests/probe
receipts, pools and physical card before launch. Registration and bound sources
must be committed; launch and trainer independently verify worktree and HEAD
blob hashes. The detached worker leases only the registered free card, heartbeats
it, records process starttimes/host, writes DONE/FAIL and releases the lease.
No WorksGood service, dispatcher or completion machinery is used.

Supervisor can arm from `/home/erikg/emender`:

```bash
.venv/bin/python -m scripts.watch_invertible_stage11_gdn2_adamw --interval 900
```

Native watcher exits0 only for matching terminal/DONE receipt and hashed complete
500K result; exits1 on failure/dead identity; `--once` exits2 while pending.
The watcher script is supplied for supervisor arming, not auto-armed here.
Scope: standalone single-card research; no resilient DiLoCo/Frontier claim or
applicable distributed requirement IDs. Existing unrelated worktree edits remain
untouched. Launch evidence will be written separately to `LAUNCH_REPORT.md`.
