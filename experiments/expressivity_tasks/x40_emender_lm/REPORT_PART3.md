# X40 part 3 — 8.03s to 1.085s, scale closure blocked

B64/T512 ignition-v1 performance is repaired: unchanged v1 **8.030s/update**,
exact-decay optimization **1.065s**, per-chunk graphs
**1.085s** (7.40x faster than v1).
The registered <=2.5s gate passes. **The complete performance program remains
incomplete:** X42-policy B1/T512 pressure parity fails, so pure scale timing,
pure/control ratios and T2048 pure amortization are not measured. No waiver.

## Decomposition before changes

[DECOMPOSITION.md](DECOMPOSITION.md) and raw baseline receipts precede every fix.
Of the profiled device time, scan uses 5240.911ms and reverse 2651.728ms;
fill/copy/journal bookkeeping is small. The composition already contained the
X36 persistent core, tiled causal reads and journal replay. Repeated exact
rounded decay and repeated unit-strength death-horizon evaluation were the
engineering debt, particularly at the 2048/8192 horizons. The X36 homogeneous
43ms lane price was not the same policy workload.

Exact unit-reference tables contain repeated FP32 multiplications, never pow.
Unit death scheduling is computed once; forward reads advance by one rounded
multiply with mutation timestamps; reverse uses the table for reference 1 and
retains the original exact fallback for all other references. Scan device busy
time falls 5240.911→214.287ms; reverse 2651.728→658.357ms in separate profiles.
Those joint changes explain the measured gain; no per-change causal split is
claimed. X36 replay was ported locally to keep every upstream source read-only.

## Graph results and parity

Per-policy chunk forward/backward sequences use make_graphed_callables. Address
maps are device-authored, eliminating transient pinned-host mapping transfers.
Input shape/dtype/device/gradient flags are checked; dense LM, clipping, AdamW
and health observations remain eager. Runner opt-in: `--cuda-graphs`.
Whole-LM capture attempts failed (OOM, then implicit-stream
capture errors); their failure receipts remain. Per-chunk capture passes loss
and every parameter gradient bitwise for two different B64/T512 batches.

| Dispatch | Host launches, including graphs | Graph launches | Device activities, including fills/copies |
|---|---:|---:|---:|
| baseline | 1974 | 0 | 2348 |
| exact_decay | 1974 | 0 | 2348 |
| graphs | 1110 | 48 | 2612 |

Graphs reduce host launch calls but add device copies/gradient bookkeeping.
Five synchronized wall samples per mode are retained; graph versus optimized
eager is 1.018x, with no established graph speed gain.
Peak graph allocation/reservation: 30.213/39.143GiB.
Graphs stay opt-in; FP32 exact-decay optimization is the default. Baseline and
initial eager measurements use physical card 1; final graph, control and slice
measurements use broker-approved card 5, the same RTX6000 Ada SKU. Clock/thermal
conditions are not calibrated, so small timing differences are not established gains.

## Scale probes: failed pure gate, independent transformer controls

X42 pilot_pure/final_pure configurations contain **one** four-head memory lane
and MLPs elsewhere. They are not six-memory-block X40 scaled uniformly.
The additional production-shape X42 gate fails before any pure timing:
142 pressure elements differ (maximum 0.0166666731); full diagnostic in
`receipts/part3_scale_final/parity_T512.json`.
Values, native state, routing decisions and input gradients pass before the
pressure assertion. This is a public-telemetry parity failure, not a numerical
training-gradient failure; it is still a failed gate. The diagnostic compares
both pressure arrays against an independent 128-token case-window calculation.
No protected X31/X42 source was changed to bypass it.

Independent validated X42 transformer probes completed 200 measured optimizer
updates each, three warmups, microbatch 1, one microbatch per update, autocast
bf16 dense layers, checkpointing as configured. These are synthetic repeated
nonpadding-token engineering probes, **not** the configured 256-microbatch
accumulation or trained quality endpoints. Both allocated and reserved peaks
are measured. Pure/control ratios and pure amortization remain unmeasured.

| Transformer shape | T | Parameters | Tokens/s | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|
| 150M | 512 | 148,075,008 | 9700.5 | 2.285 / 2.482 |
| 150M | 2048 | 148,075,008 | 38891.1 | 3.168 / 3.721 |
| 1.3B | 512 | 1,313,738,752 | 3009.3 | 19.647 / 21.301 |
| 1.3B | 2048 | 1,313,738,752 | 6686.0 | 19.802 / 22.814 |

Transformer T2048/T512 throughput ratios are 4.009x at 150M
and 2.222x at 1.3B. These do not establish pure-stack amortization.

## Precision and matched training

bf16 key/value store fails the initial production B64/T512 forward bitwise
gate. Its raw per-policy mismatch counts are in part3_bf16/gate.json. No bf16
performance or gradient claim is made after forward failure. Store remains
FP32; `--store-dtype bf16` fails closed before training.

The final graph-enabled slice completes 153 updates / 5,013,504 tokens in
165.188s: **30350.2 tokens/s**, maximum
loss difference versus ignition-v1 **0**
(tolerance 1e-5). Initialization, batches, AdamW, clipping and gates match v1;
no graph warmup counts as training exposure. Per-step differences and source
hashes are receipted. No .pt files or checkpoint-dependent claims.

The original v1 curve has 3052 updates / 100,007,936 tokens and final held-out
PPL 10.37446235 versus the cited reference 6.8564176. These are quality context,
not a new optimized full-training endpoint or new transformer quality run.
The prior part-2 report remains in REPORT.md.

## Validation, reproduction and residual work

The combined run passes 183 tests (179 upstream fixtures plus 4 local); a
subsequent local run passes 5 including the new precision guard. See
part3_validation/tests.log/XML for the X31/X36 fixtures routed through the
local reverse port, x40_tests.log for local architecture/gate tests, and all
production policy/graph parity JSONs. All GPU commands use run_leased.sh and
only the approved interpreter. Commands and their actual exit statuses are
retained in part3_validation/commands.json; expected gate failures remain
labelled failed. Raw profiles are gzip archives with full launch ledgers.

Reproduce optimized profiling with `performance --certify --output
FRESH_DIR`; graph certification/timing with `graph_probe --output FRESH_DIR`;
scale gates with `scale_probe --output FRESH_DIR`; independent controls with
`scale_probe --arm transformer --output FRESH_DIR`; precision with `bf16_gate
--output FRESH_DIR`; matched slice with `train_slice --graph --output FRESH_DIR`.
Use each as `run_leased.sh -m experiments.expressivity_tasks.x40_emender_lm.MODULE`.
`summarize_performance` regenerates this report. Baseline receipts bind the
pre-change source hashes; rerunning that baseline requires the pre-change
implementation with the new instrumentation, not the current optimized model.

Remaining blockers: resolve the X42 pressure/oracle discrepancy without
weakening certification; then run 200 pure updates at both shapes/contexts and
compute ratios/amortization. bf16 parity is failed. Graphs have no established
speed advantage and consume more memory. Long-horizon capacity regimes are
not certified by the short TinyStories curve. The local replay copy requires
maintenance against upstream changes. No pure-scale studyability claim is made.

No WG, workers, subagents, Slurm/Frontier, protected-source edits or .pt files.
Resilient DiLoCo requirement IDs do not apply to these single-GPU probes.
AGENTS.md/CLAUDE.md are unchanged. contact_supervisor/intercom is not exposed in
available tool metadata; no substitute service was started. Work is reported
as partial, with failures retained, rather than attested complete.
