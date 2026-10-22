# X21E — measured extrapolation, not unboundedness

All six champions were usable: every original and scratch snapshot matches its
prior checkpoint SHA256. Predictions were committed/pushed in **eb7f90e7**, then
sources/pools/checkpoint selection frozen in **42ceb5ed**, before champion eval.
Six native frozen controls reproduce every historical joint/X/Y/OOR integer
count. Completed **216 model/precision/horizon rungs, 3,456 evaluation minibatches,
96 control minibatches**, zero infrastructure censoring, **zero training updates**.

## First ladder, side by side

Joint accuracy (%), fresh512 at each rung; full per-position counts and losses
are in `evaluations/<champion>/native/<rung>/result.json`. T counts drivers,
not header/padding/neutral-slot tokens. Native precision is fp32 state under
bf16 autocast, not a bf16 recurrent accumulator.

|Champion (training T)|2x|4x|8x|16x|32x|>=480 columns across those rungs|
|---|---:|---:|---:|---:|---:|---|
|T16 E97 AdamW|12.55493|6.28662|3.16620|1.59683|0.80872|4 / 4 / 4 / 4 / 4|
|T16 GDN2 AdamW|14.53857|7.26013|3.62091|1.83487|0.93155|4 / 4 / 4 / 4 / 4|
|T64 GDN2 cold|4.65088|2.33994|1.18599|0.60596|0.31786|5 / 6 / 5 / 5 / 6|
|T64 E97 cold seed2|2.10419|1.05133|0.53253|0.27866|0.15316|2 / 2 / 2 / 2 / 2|
|T64 E97 curriculum extension|3.15552|1.59531|0.81139|0.42000|0.22154|4 / 4 / 4 / 4 / 4|
|T64 E97 wean (neutral)|2.04468|1.02692|0.52719|0.27981|0.15535|2 / 2 / 2 / 2 / 2|

The curves decay roughly as 1/T: a short learned prefix remains flat while its
share of all targets shrinks. There is no newly extending high-accuracy tail.
A >32x failure-horizon censor means *the old short prefix survived*, NOT that
the model solved the intervening positions or generalizes unboundedly.

## Alignment and predictions

Every native survivor at 2x–32x is **start-anchored**, zero shift from its 1x
column. Full start/suffix candidate offsets are in `summary.json`. At 1x both
coordinate systems coincide, hence explicitly ambiguous. The generic prediction
that suffix/structural columns dominate surviving transfer is not supported by
these data. Structure-relative/event-distance is not identifiable in this iid
stream with no designated events; none were invented.

Cold prefix/cliff predictions are broadly borne out, except T64 GDN2 acquires a
borderline sixth column in some pools. Native1x P6=482/512, 2x=469, 4x=480,
8x=474, 16x=478, 32x=487. The registered first-crossing failure is T128 (2x),
but recovery and fresh-instance sampling mean this is **not a monotone length
cliff**. Causal horizons are also confounded by different fresh pools at each T.

Staging >= cold E97 in surviving-column count holds on every native rung:
curriculum4 vs cold2; wean2 vs cold2. The stronger narrative 'both staged
champions retain more' is false for wean (a tie). Neither staged model shows
new horizon-robust composition beyond its acquired prefix. Cold T16 pairs share
the AdamW recipe; T64 contrast is seed2 vs seed0, not a controlled seed-matched
architecture conclusion. Staged recipes are overlays, never causal contrasts.

## State precision

Order is **native fp32 accumulator -> wrapper-fp32 control -> bf16 -> fp8 E4M3FN
-> int8 -> int4**. This is fake quantization after every token before readout,
not final-state-only casting or a packed-memory/speed benchmark. Only matrix
state is quantized, not weights/projections/GDN convolution buffers.

- All E97 strong native prefixes survive bf16, fp8 and int8 through32x.
- T16 GDN2 keeps4 columns in bf16/fp8 but only3 in int8 (P1 fails already1x).
- T64 GDN2 keeps5–6 in bf16,5 in fp8,3 in int8; int4 keeps0.
- Int4 at every rung leaves T16 E971, T16 GDN2 0, T64 GDN2 0, cold T64 E97 0,
  curriculum0, wean1 strong columns; all six lose at least one native1x strong
  column already at1x. Report those failures, not an invented extrapolated range.
- All non-T64-GDN native runs are right-censored beyond32x for the registered
  prefix-retention metric. T64 GDN native fails2x; its wrapper/bf16/fp8/int8/int4
  all fail1x against the six-column native1x reference.

**Wrapper control is not bitwise native parity.** At fresh1x, maximum absolute
per-column joint-count deltas are E97-T16 1, GDN-T16 2, GDN-T64 7, E97-T64 1,
curriculum1, wean4. Full deltas/CE are disclosed in `summary.json` and `REPORT.md`.
In particular GDN-T64 wrapper P6=475 vs native482: its already1x threshold loss
precedes any quantization, so the corresponding bf16 failure cannot be attributed
solely to state precision. GDN native uses its stock FLA evaluation; the adapter
uses explicit fp32 delta-rule equations. E97 changes scan tile/reduction geometry;
finite-precision sigmoid/FMA reassociation can alter marginal outputs. Synthetic
reference tests check equations/codecs with explicit bin-boundary tolerances,
not a false bitwise-parity claim.

These are precision thresholds on **short prefixes**, not an identified B-to-
horizon scaling law or proof of compressed computation. All-rung profiles are
flat in retained-prefix length after initial quantization loss; there is no
long-range execution from which to fit a counting/fold scaling constant.

Each integer state has one dynamic fp32 scale per example/head, not per tile.
All-four-layer state budgets per example, including this metadata:

|Architecture|native32|bf16|fp8|int8|int4|
|---|---:|---:|---:|---:|---:|
|E97 (32 heads/layer,32x32 state)|4,194,304|2,097,152|1,048,576|1,052,672|528,384|
|GDN2 (2 heads/layer,128x1104 state)|36,175,872|18,087,936|9,043,968|9,044,224|4,522,240|

Equal total model parameter scale is not equal recurrent-state size (GDN2 has
8.625x the state elements). Int8 findings are specific to the registered global
per-head dynamic scaling policy, not every possible quantizer.

## Reproduction and evidence

Run from repository root on lambda01 with `.venv/bin/python`; no Frontier/Slurm
submission took place. `validation/*.log` retains test/audit/eval command output.
`runtime_*.json` records Python/Torch/CUDA/GPU identities; every completed rung
binds its16 minibatch receipt hashes, unchanged model tensors/checkpoint SHA,
registration and generated-pool hashes. Original run directories were never
written. Existing unrelated dirty worktree files were preserved and not staged.

Read-only independent acceptance audit:

```
.venv/bin/python scripts/audit_horizon_ladder.py
```

CPU+CUDA reference guards and existing X18 regressions:

```
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest -q tests/test_eval_horizon_ladder.py tests/test_eval_invertible_stage18_freshdata.py
```

Audit PASS:6 champions,216 rungs,96 control batches,3456 eval batches,0 censoring,
0 updates. Tests:33 passed. Registered code/predictions have not changed since
source freeze. Resilient DiLoCo R01–R16/NDP/V21S/ISP are not applicable to this
independent-instance eval; no resilient conformance or training result claimed.
