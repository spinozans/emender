# X42 lineage audit and suspect ablations — 2026-10-31

The full block audit confirms that both historical registered X40 v1 and
registered X42 velocity pure candidates execute memory + MLP. X34/X37/X38/X39
receipts use hybrid attention + MLP with one tape lane. See
[the lineage table and consequence ledger](receipts/lineage_suspects/LINEAGE.md).

Two full 2700-second candidates completed under the existing compute recipe:

| Variant | Score | Held-out NLL | Score improvement | NLL improvement |
|---|---:|---:|---:|---:|
| Tuned baseline | 4.802025 | 4.742690 | — | — |
| Fourfold per-head budgets | 4.768561 | 4.765585 | 0.033464 | -0.022896 |
| Maximum w_in_width=256 | 4.890189 | 4.830853 | -0.088164 | -0.088164 |

Neither variant meets the prospectively recorded 0.1 improvement thresholds.
The budget variant's small score gain accompanies worse NLL and lower
occupancy/capacity penalties. Per-head occupancy, R-CAP and gate telemetry,
source hashes and timing are in [the result receipt](receipts/lineage_suspects/summary.json).
These are single-seed, wall-clock comparisons, not token-matched causal tests.
Event-granularity information flow received static analysis only.

`emender-x42-pure-resumed.service` restored the exact paused gen-4 state and
unchanged registration aefc7180058a3e18cc0cae9158674e7cbf5f087eef9fc2fa21926b1646c6d94b.
Its parent is active; its children request only broker card 1. At receipt time
they remain queued because the existing transformer search also requests and
reacquired card 1. No new pure training update is claimed. Existing baseline
jobs, anchors, card 3 and card 7 were not disturbed. The old loop unit remains
historical/failed; the attended resumed unit is the current parent.

The part25 probe was audited statically and left untouched. Its existing verdict
is incomplete (failed scales 1/4/16); no live part25 process was observed.
The requested ~1.6x pure speedup is not substantiated by part8's recorded ratios;
its overall status remains failed. See the ledger for the actual ratios.

## Historical initial velocity launch

# X42 velocity search launched

Preregistration commit `e0ec61ac`, SHA256
`f394fc0680f86ac806976a3162d321561087e92c0837da0677303c6a8c51a131`,
preceded the first scored candidate. The smoke completed 270 updates and
138,240 tokens in 2701.123262612615 active training seconds. Its 66.462894
seconds of evaluation are excluded. Exact utility recomputation agrees:
score **8.140112850969016**, held-out NLL 8.1380096077919, occupancy
0.2103243177116383, no degeneracy flags. Throughput was 51.178708 tok/s;
peak allocated GPU memory was 3,481,367,552 bytes. The receipt includes all
48 layer/head operating points and occupancy-length slopes.

Direct detached parent PID **974815** owns `runs/velocity_v1`. Generation 1
population **12** is launched; the first three candidates hold cards **1/6/5**.
The saved initial population has 36 coordinates, full covariance, means,
evolution paths, RNG and completion map. The parent polls child failures,
rejects failed/nonfinite/censored endpoints, and persists completed scores.
Generation 1 is running, not completed; no winning policy is claimed.

Resume the same parent module, registration and output directory. Verified
surviving child groups are stopped; incomplete candidates rerun from scratch.
No candidate model checkpoint or .pt file is used. Generation-state SIGKILL
and deterministic population/covariance/RNG continuation tests passed.

Thirty local velocity tests passed. The isolated policy scan passed exact
state/trace/output/gradient parity against unchanged X40, with independent
occupancy reconstructed from integral traces. Full 150M-trainable proxy
train/eval integration passed for N1/3/4/5, vocabulary 50257, T512 training
and T2048 telemetry. Gate coefficients, pressure onsets and slopes are
independent per-head search constants; N-dependent geometry/readout and the
FP32 bounded vocabulary projection are wired to the attested corpus.

The broader historical suite had **52 passes and two failures**: old part8
checkpoint preflight receipts have stale backward.py hashes at T512/T2048.
No certified source or parity gate was edited. The current part12 support
gate matches core source hashes; its historical parity harness hash itself
is stale, while the forward/backward core hashes match.

Residual limits: generation 1 remains active; batch1 is not throughput tuned;
the gate-cost coordinate offsets the frozen-management objective but has no
trainable-gradient effect; proxy-to-scale transfer is unproven. On exact
score ties, CMA ranking uses stable candidate index, but the parent's final
winner label uses completion order; apply the registered stable-index rule
when reviewing a tied generation before any promotion.

Card 0 remains excluded. PID 634691 was checked alive before and after smoke
and left untouched; cards 2/3/4 and 7 are excluded. No WorksGood, Slurm or
resilient DiLoCo behavior was used; DiLoCo requirement IDs are not applicable.
No supervisor/intercom tool is exposed; no unresolved semantic decision
blocked this implementation. Raw engineering receipts and live runtime
artifacts remain local; the committed compact receipts bind their hashes.
