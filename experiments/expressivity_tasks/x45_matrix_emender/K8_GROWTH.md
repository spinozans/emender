# K8 — GROWTH WIRED: the registered rule on the untangled machine

Stage 8 of the X45 kernel lane. Growth was the last unwired capability: the rule
has been registered from the beginning
(`x43_fork_resolved/PREREGISTRATION_MATRIX_EMENDER.md`: growth "at chunk
boundaries only, `r' = max(r + stride_floor, ceil(g*r))` snapped to the
precompiled set {64…2048}", trigger "a deterministic chunk statistic, fixed
before the run (write-residual mass and/or append-collision rate), reported per
chunk"), its legality is proven in Lean (zero-injection: `LiftedMemory` /
`injection_propagation` — new rows enter as exact zeros and the monoid and scan
theorems hold; `formal/lean/OPERATOR_ALGEBRA_STATUS.md`), the registered
implementation existed (`scan.repeat_write_rate` + `keys.growth_target` inside
`scan.routed_chunk_scan`), and K6/K7 were built growth-ready — but **no run in
the program's history had ever had `grow=True`**. The operator's directive this
session: the cache is too small — a transformer at T=2048 has ~2K addresses per
layer; wire growth with transformer-parity addressable state as the target.

**The registration-semantics decisions were escalated and confirmed by the
operator before the build**: 64-token statistic boundaries MID-WINDOW (the
registered chunk length, the K6 segment length — growth decided by the model
WITHIN a sequence, eager re-dispatch at the boundary), the registered
append-collision statistic VERBATIM (imported `scan.repeat_write_rate`, with the
registered smoothing/window/reset/cap bookkeeping; the occupancy variant
explicitly refused), and the slot-blocked trainable high-rank backward as the
intended scope (≤128 spans reuse K6's kernels).

**No whole-arm claim is banked in this document yet.** The three-arm probe is
LAUNCHED (detached, leased, progressing — §6) and a harvest follow-up collects
it; everything below is the wired capability, its gates, and what the short-run
evidence already establishes.

## 1. What was built

| file | role |
| --- | --- |
| `k8_growth.py` | the growth machinery: the registered trigger bookkeeping, the two-pass `growing_scan` (probe + real), the zero-injection pads, the K8 span op pairs (`x45::k8_span_scan_fwd/bwd` at rank ≤ 128 re-hosting K6's launch + adjoint kernels with the SEEDED carry; `x45::k8_hi_scan_fwd/bwd` above 128 on K7's walk kernel launched directly), the slot-blocked backward (`_k8_traj_kernel` in three modes, the four prep phases, `_k8_bwd_walk_kernel`), the mixer and `convert_model_to_growing`, `oracle_growing_scan` (the eager gate-B fixed point) |
| `test_k8_growth.py` | 16 tests: never-fires bit-equality, the trigger bookkeeping, the zero-injection receipts, prefix/probe/real bit-equality, determinism, hi-span parity vs `k7_sequential_walk` and K7's oracle, traj-replay bit-equality (incl. vs K6's own checkpoints), gradients within floors at rank 256 and 2048, gradients ACROSS growth boundaries, the bf16 decay-gate regression, refusals, the conversion |
| `k8_gate_receipt.py` | the registered-shape evidence → `receipts/k8_gates.json` |
| `k8_probe_growth.py` | the three-arm probe driver + the smoke subcommand → `receipts/k8_smoke.json`, `receipts/k8_probe_{grow,fixed2048}.json` |
| `run_k8_growth_smoke_leased.sh`, `run_k8_probe_leased.sh` | the detached leased drivers (the survival playbook: the lease owner and heartbeat keeper live in the detached process) |

Zero edits to `k6_*`/`k7_*` sources, both preregistrations, and `lattice.py` —
imported, never modified.

### 1.1 The registered form, taken verbatim

* **Granularity**: the trigger is evaluated at fixed 64-token statistic
  boundaries MID-WINDOW. A fire re-dispatches the scan to the grown rank at that
  boundary (the eager boundary re-dispatch K6/K7 were built ready for).
* **Signal**: `scan.repeat_write_rate` — imported and called VERBATIM — the
  share of writes reusing a slot written within the last 512 tokens, smoothed
  over the last 4 boundaries, firing at mean ≥ 0.5, window and `last_write`
  reset on fire, `r' = min(cap, growth_target(r, 2.0, 64))` (geometric 2.0,
  snapped to {64…2048}), capped at 2048.
* **Zero-injection**: at a fire the state pads with exact zeros, liveness off
  until first written (`F.pad`; its own backward is the exact adjoint — the
  linear embedding's).
* **No shrink**: POP/MERGE shrink content; the physical rank never shrinks.

### 1.2 The two-pass scan

Per (layer, window) scan: a **probe pass** (no grad) span-walks at the current
rank — K6's `_seq_forward_launch` at ≤128, K7's `_k7_walk_kernel` above —
evaluating the trigger at every 64-token boundary from the walk's own recorded
schedule; the boundary state comes from the walk's segment checkpoints (≤128)
or the schedule replay (above). Its output is the SPAN TABLE plus the growth
events and per-boundary rates (the "reported per chunk" receipt). The **real
pass** replays the span table through autograd, the spans chained through the
zero-injection pads. Both passes run the same kernels on the same inputs, so
the real span's schedule is bit-equal to the probe's prefix — asserted, and the
reason the re-dispatch needs no mid-walk host sync. The probe is skipped
entirely when the rank is at the cap or `grow=False` (the fixed arms).

### 1.3 The trainable high-rank backward (the K7-style re-grid)

K7 is forward-only; K6's backward kernels (and K4's trajectory replay) are
register-capped at rank 128/512. K8 re-grids K6's adjoint over SLOT BLOCKS,
every cross-block scalar combined in ONE fixed-order torch tree (the Lean
full-population law: a tile's internal defined tree plus one tree over the tile
results IS the whole-axis tree, zeros padding exactly):

* `_k8_traj_kernel` — the slot-blocked schedule replay (K1's walk at any rank),
  three modes: exit-only (probe boundary extraction), per-token trajectory (the
  backward's 64-token segment recompute), per-64-token checkpoints (the span's
  segment entries, **recomputed in the backward instead of forward-saved**: at
  rank 2048 forward-saved checkpoints would be ~554 MB/scan held by autograd per
  mixer — ~13 GB per microbatch; one extra K1-class replay per span is the
  honest price, receipted in gate D).
* `_k8_prep_scores/mass/weights/adj_kernel` — the per-token read weights and
  adjoint coefficients recomputed from the trajectory, in four phases around
  the torch combines; the per-row expressions are K6's own, so the recomputed
  weights are BIT-EQUAL the forward's collected channel (asserted).
* `_k8_bwd_walk_kernel` — the reverse adjoint walk: one program per
  (row, value-block) sweeps the slot tiles per token in two sub-sweeps (K6's
  step order preserved), the adjoint carried in global memory, register
  accumulators for `dpay0`/blend weights/`duq`/`dlam`, CTA barriers between
  the sweeps (the K7 cross-warp lesson), determinism asserted.

**The span chaining forced one K6 deviation, stated**: K6's own op pair drops
the exit-state gradient — its windows never chain, so nothing downstream ever
needed it. K8's spans DO chain, so the ≤128 spans re-host K6's launch and
adjoint kernels VERBATIM in a new op pair (`x45::k8_span_scan_fwd/bwd`) whose
only difference is the carry seeded from the span's exit-state gradient (with
`dstate_out` zero this is K6's body exactly — same kernels, same launch
parameters, same fixed-order combines; the grow-never-fires forward is
bit-equal to the banked K6 path by construction). The hi backward seeds the
same way.

## 2. Gates A/B/D (`receipts/k8_gates.json`)

**Gate A — legality and parity.**

* The never-fires row (`grow=True`, fewer than `grow_window` boundaries):
  bit-equal to `sequential_scan` on reads, state and liveness — the same launch
  by construction.
* The grown scan at the probe shape (batch 8, 2048 tokens, start 128, cap
  2048): spans `[(0,256)@128, (256,512)@256, (512,960)@512, (960,1792)@1024,
  (1792,2048)@2048]`, the events and per-boundary rates receipted; the
  **zero-injection receipts**: at every event the new rows are dead at entry
  and stay dead until the first schedule write that targets them (the first
  such write lands at the boundary itself — APPEND takes the first free slot);
  the **prefix property**: the grown run's span-1 schedule is bit-equal to the
  fixed-rank twin's own prefix and the boundary state bit-equal to the twin's
  own segment checkpoint — the growth event changes nothing about the old
  rows; **determinism**: two full growing scans bit-equal on every channel.
* **0 decision flips in 2,162,688 lanes** vs the eager oracles (the mixed K6/K7
  fixed points), per span, at every snap 128→2048; the read weights bit-equal;
  the reads at **0.5–1.0× the measured fp32-vs-fp64 floor**; the hi span
  bit-equal to `k7_sequential_walk` on every channel (the same kernel,
  launched directly, replayed from the span's own entry state).

**Gate B — gradients.** The hi backward at rank 256 and 2048 vs the eager
oracle that replays the same span table, floor first (fp32 oracle vs its own
fp64 twin): dquery 1.0×, dpayload 1.0×, ddecay 1.67× the floor — inside the
8× discipline. **Across growth boundaries** (a 128→256→512 chain, the entry
state attached so `dstate` crosses both pads): 0.89–1.67× the floors. The
prep's recomputed read weights are bit-equal the forward's own channel. The
rank-axis adjoint reductions (`duq`, `dlam`) are tile sums sequenced across
slot blocks — an order no torch expression reproduces exactly — so these are
floor gates, never bit-equality (the K6/K23 discipline, stated).

**Gate D — the honest walls (probe shape, batch 8 × 2048 tokens, bf16).**

| rank | forward | backward | peak | tok/s (scan only) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 27 ms | 15 ms | 0.4 GB | 612 k |
| 256 | 67 ms | 35 ms | 0.6 GB | 246 k |
| 512 | 113 ms | 67 ms | 0.9 GB | 145 k |
| 1024 | 212 ms | 136 ms | 1.7 GB | 77 k |
| 2048 | 414 ms | 270 ms | 3.2 GB | 40 k |

The grown scan (probe + real, climbing to 2048 within the window): 447 ms
forward, 110 ms backward. **2048 is 16× the registered state and the walls say
so** — the walk is fold-bound (K7's own receipt) and the backward pays the
checkpoint replay instead of 13 GB of forward-saved checkpoints.

## 3. The bf16 decay-gate failure (found at probe scale, fixed, pinned)

The first real-model smoke produced a **non-finite clip norm** — a full step's
parameter gradients reached 1e21. The cause: the x44 §6.2 failure class, again
— **a bf16 `lam` fed to the f32 `mul.rn` intrinsic is not a value-correct
promotion on this stack**. The training regime's autocast hands the DECAY gate
to the scan as bf16; K6's body upcasts it first and K7's kernel upcasts
in-kernel, but the new K8 kernels loaded it raw: the slot-blocked trajectory
replay replayed garbage (exit-state maxdiff 2.6) and the adjoint's `ds ← λ·ds`
blew `ddecay` up by ~1e6 (the small-fixture gate-B tests ran fp32 and never
saw it). Fixed K6's own way — upcast the gate once (`_launch_traj`,
`_hi_backward_body`) plus the in-kernel `.to(tl.float32)` — and pinned by a
bf16-regression test. The whole-model step gradient returned to the fixed-128
arm's scale (max 0.31 vs 0.33).

## 4. The smoke (the operator's short-run confirmation, committed first)

`receipts/k8_smoke.json`: 6 REAL steps at probe shape, seed 3401, the grow arm.
All four machinery assertions land:

* the trigger fires (first fire at boundary 256 of every window);
* re-dispatch executes — **all 24 mixers reach rank 2048 on every step**;
* the zero rows are exact (dead at entry, dead until the first write into them);
* the backward survives the rank changes (finite gradients, 8.6 → 3.0 across the
  six steps; loss 5.74 → 3.77).

Steady step wall **144 s** (the fixed-2048 arm: 136 s) — the honest projection
~7 h per arm at the 180-step matched exposure.

## 5. The finding the smoke surfaced — growth fires on an IDLE store

Instrumenting the first mixer on real stream data at init: the per-token policy
runs **97.66% CORRECT / 2.34% APPEND / 0 POP / 0 NONE**, holding ~48 live slots
of 128+ — natural text's payloads land within `tau_match` of a small hot set and
the store NEVER FILLS. Consequences, stated honestly:

* **grow ≡ fixed-2048 ≡ fixed-128, bit-for-bit**, for as long as the store
  stays under capacity: the three arms' losses are identical to 14 digits
  through the early steps. The grown rows exist, are legal, and are never
  read. Divergence can only emerge if trained occupancy approaches
  saturation (the banked K6 arm's trained occupancy was 73.7/128; the trained
  chunked arm reached 92% at chunk entries — near, not at, the ceiling).
* **The registered trigger fires anyway**: the append-collision statistic
  counts writes that reuse a recently-written slot — and the identity-rewrite
  regime (CORRECT to yesterday's row) makes the collision rate ~1.0 inside the
  statistic's 512-token window, so the rank climbs 128→2048 in every window
  regardless of occupancy. This is the registered implementation's own
  behaviour (its zero-initialised `last_write` additionally makes every write
  inside a window's first 512 tokens a collision), inherited VERBATIM per the
  operator's explicit instruction, receipted per boundary in the gates — growth
  here is armed by write-reuse pressure, not by a full store. The
  registration's alternative signal ("write-residual mass and/or
  append-collision rate") was NOT substituted: the implemented, registered one
  is the authority.

## 6. The three-arm probe (launched, progressing; harvest collects)

The protocol is the k6 probe's verbatim: the same `matrix_ops` model
(149,981,808 parameters), built by the same order (seed 3401, the init shared
bit for bit with every banked arm), the same registered flagship stream, the
same document-disjoint calibration eval, the same matched-exposure stopping
rule (20,000,000 targets), the same honest clock. The LR is the **transferred
LR** (the gdn2 nomination's — the CMA-ES diagnosis confirmed the transferred
config was NOT sick; it leads gdn2 then plateaus architecturally), and the
policy constants are the registered ones (`tau_match` 0.45, `read_scale` 2.0 —
NOT the CMA-ES-tuned chunk-arm config, which that diagnosis itself scoped to
the chunked arm) — comparability with the banked **fixed-128 = 2.0278
nats/byte** is the priority.

| arm | state | card |
| --- | --- | --- |
| fixed-128 (banked K6 per-token probe) | **2.0278 nats/byte**, banked (`receipts/k6_probe.json`) | — |
| grow (start 128, registered rule, cap 2048) | detached leased driver, progressing (144 s/step) | 4 |
| fixed-2048 (rank 2048 from token 0, never grows) | detached leased driver, progressing (136 s/step) | 5 |

Both drivers verified per the survival playbook: wrapper PPID 1, the live lease
owners with fresh heartbeats, advancing training steps in the retained
`training.jsonl` logs, per-step growth snapshots (final-rank histograms, fire
boundaries) in the receipts-on-landing. Each arm commits its own receipt
(`receipts/k8_probe_{grow,fixed2048}.json`) and pushes when it lands.

## 7. The verdict, as far as the banked evidence reaches

**The capability question is answered: the machine grows.** Given the chance —
the registered rule, wired with its legality proven and its parity held to the
lane's bit-exact/floor gates — the model climbs from 128 to transformer-parity
state (2048 addresses per layer) inside every window, the new rows enter exact
zeros, the old rows' evolution is untouched, the backward survives every rank
change, and the whole thing trains.

**The utility question — "does it pay?" — the probe-scale evidence already
points the honest way, pending the harvest**: at probe scale the model's own
write dynamics keep the store at ~48–92 of 128 rows; the capacity the operator
targeted is never binding, so the grown rows are never read and growth buys
nothing while costing the probe pass plus the 16×-state walls (144 s vs the
fixed-128 arm's 9.5 s per step at this shape). Whether training pushes
occupancy into saturation — the one regime where the arms can separate — is
exactly what the two 20M-target arms will receipt. If they land tied, the
verdict is the informative null: **the model, given the chance, grows
mechanically but has no use for transformer-parity state at this scale — the
binding constraint the CMA-ES diagnosis named (read-back sharpness) is not
capacity** — and the operator's 2048 target would need a regime that saturates
the store to be testable, not a bigger cache bolted onto this one.

Probe-scale caveats, inherited verbatim from the sweep's protocol: single seed,
one transferred LR, 14,336-target eval boundary, 20M-target exposure —
engineering evidence for the growth capability, pre-any-2700s-claim; the
sequential arm's registration sign-off remains the operator's gate.

## 8. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=4,5 scripts/gpu_lease.sh acquire 1 --wait)"
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k8_growth.py -q   # 16 passed
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/ -q                     # 200 passed
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k8_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k8_gates.json
# the smoke (committed before the long arms), then the arms, detached:
setsid bash -c 'experiments/expressivity_tasks/x45_matrix_emender/run_k8_growth_smoke_leased.sh \
    > experiments/expressivity_tasks/x45_matrix_emender/k8_runs/smoke.console.txt 2>&1' &
setsid bash -c 'GPU_LEASE_VISIBLE=4 experiments/expressivity_tasks/x45_matrix_emender/run_k8_probe_leased.sh grow \
    > experiments/expressivity_tasks/x45_matrix_emender/k8_runs/grow.console.txt 2>&1' &
setsid bash -c 'GPU_LEASE_VISIBLE=5 experiments/expressivity_tasks/x45_matrix_emender/run_k8_probe_leased.sh fixed2048 \
    > experiments/expressivity_tasks/x45_matrix_emender/k8_runs/fixed2048.console.txt 2>&1' &
```

## 9. Recommended next step

Harvest the two arms when they land (~7 h from launch: the receipts commit
themselves); fill the three-arm table in §6 and finalise §7's verdict with the
eval rows and the trained-state occupancy. If the arms tie, the named follow-up
is the saturation regime: a probe whose stream or horizon forces the store past
128 live rows (long-context / beyond-horizon data, the C3 claim's own probe)
— the one place the operator's 2048 target is testable in value rather than
only in capability.
