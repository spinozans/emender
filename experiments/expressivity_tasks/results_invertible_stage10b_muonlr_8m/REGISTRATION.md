# X10b: higher-LR hybrid-Muon screen — pre-launch registration

Operator directive: taskX10b-muonlr.md, 2026-10-20. Exactly two scratch T16
deep-first E97 arms, seed0 only, 500,000-update safety ceiling, uniform CE.
Canonical root: `experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m`.

| Arm | Peak / floor LR | Physical card |
|---|---|---:|
| muonlr3x_seed0 | 3e-3 / 3e-5 | 3 |
| muonlr10x_seed0 | 1e-2 / 1e-4 | 4 |

Cards0/1 are forbidden (X12); W4's occupied card2 is excluded. Admission uses
the standalone exclusive lease broker and instantaneous physical-idleness check,
not WorksGood service/dispatcher/completion machinery. A busy or held card fails
closed rather than silently rebinding. Detached local experiment processes only.

## Controlled change

The new runner delegates to the **unchanged X10 training loop and hybrid_muon.py**.
Only LR scales (3x or10x); update1..5000 linear warmup and cosine to the floor
at500K are unchanged in shape. Both Muon and auxiliary AdamW groups scale once
per update; actual LR is logged every update. All-parameter decay0.01 (including
ignored no-weight-decay tags), routing, FP32 momentum/Newton-Schulz, coefficients,
Nesterov0.95 and approximate RMS matching `0.2*sqrt(max(rows,cols))` are unchanged.
No optimizer/schedule/loss repair, continuation, ladder, extra seed or tuning.

E97 has9,900,800 parameters (historical8M-class label), dim256/four layers,
p61/vocab3721, batch32, clip1, fp32 parameters/bf16 autocast, legacy state and
sequential Triton evaluation. Torch initialization0 and procedural numpy RNG0;
512-example frozen X10 pool with exact array SHA256 checked before training.
Full P1-P16 validation every500, including step0 and500K; ordinary live weights.
Entry + aggregate-best + latest/final retention, deletion receipts, per-update
finite/gradient checks, resource ceilings and no automatic retry match X10.
Infrastructure censoring is not an optimizer failure. Per-arm bounds come from
source-bound no-score probes and are at least the historical Muon bounds.

## Registered predictions (before either arm launches)

(a) **LR-UNDERSCCALED**: if either higher-LR arm consolidates P4 materially beyond
X10 Muon's448/512 final trajectory (e.g. P4>=508/512 at ceiling), or accelerates
same-threshold P4 acquisition, the lag supports an LR-scale artifact for this
learned solution. Record which multiplier wins, exact counts and update/wall
clocks; transient acquisition is not final retention.

(b) **SCALE-RIGHT**: if3x/10x show P1-P3 degradation or numerical instability
(any P1-P3 below512 at the fixed-final endpoint), this supports1e-3-class for
this hybrid at8M and a method-intrinsic P4 lag at this tested scale. This is a
one-seed conditional diagnostic, not a universal causal verdict. A mixed or
neither outcome is inconclusive; infrastructure failure is censored, not a loss.

(c) **Both baselines, always**: for each higher-LR arm report a table with exactly
three outcome columns (plus row labels):

| Position / readout | Higher-LR Muon arm | X10-muon@1e-3 | X10-adamw@1e-3 |
|---|---|---|---|
| P1 first>=50 / first512 (update / seconds) | pending | 19,000 /100,000 | 24,000 /50,000 |
| P2 first>=50 / first512 (update / seconds) | pending | 128,000 /186,000 | 187,500 /223,000 |
| P3 first>=50 / first512 (update / seconds) | pending | 189,500 /281,500 | 231,500 /306,500 |
| P4 first>=50 / first512 (update / seconds) | pending | 348,500 /censored500K | 309,000 /391,500 |
| Fixed-final P1-P4 /512 | pending | 512,512,512,448 | 512,512,512,512 |

This registration preview shows update counts; the executable readout prints
both updates AND observed elapsed seconds, all P1-P16, maxima versus final,
contiguous perfect prefix, terminal and suffix7-restricted counts. Baseline Muon
P3 is25K updates earlier than AdamW, P4-material39.5K later. First crossings are
on a500-update grid, not exact learning times. Baseline snapshots and hashes
are source-bound by registration; no historical checkpoint is used to train.

**Floors printed in the readout:** uniform joint marginal1/3721 =0.0268744961%,
CE ln3721 =8.221747728 nats. Suffix7-restricted P9-P16 count denominator4096;
the suffix-only floor concerns observers missing at least two iid drivers, not
prefix-aware recurrent models. Chance-sized suffix counts are not deep tracking.
Exact joint-state correct counts, not coordinate accuracy. Learned-solution,
local single-card scope only; no capacity bound, broad optimizer ranking,
cross-seed robustness, resilient/distributed or production claim.

## Launch and wake gate

Before launch: CPU regressions; disposable seed99 full-model peak-LR GPU
updates/per-position validation; per-arm25-update no-score probe with bit-exact
optimizer/model next-update reload; executable source/dependency/pool bindings;
commit and push registration and launchable code. Never overwrite registration.

Native arming command (parent-owned):
`.venv/bin/python scripts/watch_invertible_stage10b_muonlr.py --interval 300`
from `/home/erikg/emender`. Zero only for both successful DONE/result hash
bindings; one for failure/dead or reused process; `--once` returns2 while pending.
Host reboot needs human re-arming. No claim that providing the script arms a wake.

On wake: audit complete500K finite consecutive LR streams, sources/dependencies,
terminal/result/final-checkpoint identities; run
`.venv/bin/python -m scripts.invertible_stage10b_control analyze --output X10b-growth-report.md`
for the registered three-outcome-column comparison per arm; independently review
and commit/push the growth report. No post-score extension or automatic retry.
