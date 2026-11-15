# K6 — the sequential matrix arm: the five-op algebra at per-token granularity

Stage 6 of the X45 kernel lane, and the answer to the sequentiality finding. The
chunk-size sweep (`RESULT_CHUNK_SEQUENTIALITY.md`) measured that the registered
chunking costs quality: the dial climbs monotonically as chunks shrink
(2.4173 → 2.3287 → 2.2462 → 2.2186 → **2.1791** nats/byte at chunk 128 → 8 → 4,
one LR, one seed, matched 20M exposure), MERGE starves 27× from chunk 128 to 8,
and the dial cannot reach its limit — chunk 2 OOMs because whole-scan checkpoint
memory scales with chunks/window (1024 chunk-entry saves × 1 MB per scan). This
stage builds the limit itself: **the registered five-op algebra evaluated at its
natural PER-TOKEN granularity** — same operators, same thresholds, same values,
decisions and reads against the CURRENT state at every token — as one persistent
whole-window kernel with a coarse-checkpoint backward, so decision granularity
and checkpoint granularity become independent knobs.

The registration is the DRAFT `PREREGISTRATION_SEQUENTIAL_MATRIX.md` (a new
registration of GRANULARITY, not a change to the algebra; the operator signs it
off before any 2700 s claim). Everything here is probe-scale evidence under that
draft's gates.

**Headline, as narrowly as the evidence supports.**

* **Per-token BEATS the dial's best banked row: 2.0278 vs 2.1791 nats/byte**
  (chunk 4; −0.1513) at the sweep's own matched-exposure protocol — 20,022,866
  targets, seed 3401, one transferred LR, the registered stream, the identical
  eval boundary, 180 steps like every sweep arm. The dial's gradient
  ACCELERATES into its limit (the biggest single step since 64), and the full
  chunk-128-to-per-token gradient is worth **0.390 nats/byte at probe scale**.
  The falsified sub-prediction is reported in §4: MERGE does not recover, it
  completes its collapse to zero — it was the dial's staleness artifact.
* **Gate A lands at twice the K5A bar**: the persistent kernel's decisions are
  **bit-exact against the eager per-token oracle — 0 flips in 17,039,360 lanes**
  at the registered shape (batch 8, 2048 tokens, 2 heads, rank 128, d_v 128,
  bf16 inputs), every schedule field, the MERGE-blend payload lane, the read
  WEIGHTS and the pre-decision liveness included. The read output — the one
  channel whose rank-axis reduction is a native sum (the K1/K23 precedent) —
  sits at **0.25× the measured fp32 floor**.
* **The K5A parity obstacle is resolved by definition, not by workaround**: the
  registration DEFINES the arithmetic (balanced binary tree over each reduced
  axis, fp32, correctly-rounded `sqrt.rn`/`div.rn`, and the read softmax's one
  transcendental as torch's own libdevice `exp` — measured bit-identical,
  0/2,000,000, where Triton's default `tl.exp` mismatches 88 %), and the eager
  oracle implements the same definition. The kernel matched the oracle's
  decisions **on the first fixture it ever ran** — the spec, not a gamble.
* **Gate B**: all four gradients (dquery, dpayload, ddecay, dstate) land within
  **1.25–2.0× the measured fp32-vs-fp64 floor** against the oracle's own
  autograd, with the fp64 oracle first validated by a full `gradcheck` over
  every adjoint branch (NONE, APPEND, CORRECT, POP, MERGE). None is bit-exact
  by design: the read-weight adjoint's cross-axis reductions are
  autograd-ordered on the oracle side, so the adjoint is not exact-replayable —
  stated, not hidden. The segment-recompute trajectory is **bit-exact against
  the forward's own checkpoints**.
* **The memory wall does not exist here**: 32 MB of segment checkpoints per scan
  against the chunk-2 dial's 1024 MB — decision granularity and checkpoint
  granularity are now independent knobs. Measured fwd+bwd peak at microbatch 8:
  **371 MB**; the card holds 47 GB.
* **The honest timing**: forward 26.5 ms, backward 15.3 ms per (layer, window)
  scan at the registered shape — **the backward is FASTER than the forward**,
  because the segment recompute is value-blocked and token-parallel while the
  forward's decision walk is a serial chain. The 150 M arm's step smoke runs at
  **0.97 s/step ≈ 14.0 k tok/s** projected (18 tests, 28 neighbouring K4/K5B
  tests green).

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_SEQUENTIAL_MATRIX.md` | the DRAFT registration (operator sign-off pending): semantics, the arithmetic spec, the claim set S1–S3, the gates |
| `k6_sequential_matrix.py` | the spec helpers (the tree fold), the eager per-token oracle, the persistent forward kernel, the backward pair (prep + walk), the `x45::seq_scan_fwd/bwd` op pair, the refusals, the mixer conversion |
| `test_k6_sequential_matrix.py` | 18 tests: gates A and B, the fp64 gradcheck, the POP/MERGE adjoint boundaries, refusals, determinism |
| `k6_gate_receipt.py` | the registered-shape receipts into `receipts/k6_gates.json` |
| `k6_probe_sequentiality.py` + `run_k6_probe_leased.sh` | the matched-exposure quality probe (gate C) and its detached leased driver |
| `K6_SEQUENTIAL_MATRIX.md` | this document |

### 1.1 The semantics: the chunk-1 limit of the registered dial

At token `t` the policy sees the state as left by token `t−1` — the state the
registered router sees at a chunk-1 boundary:

1. **decisions** — `schedule.route_chunk`'s selection chain at chunk
   granularity 1: unit payload vs unit live rows, dead rows masked, the
   tie-break penalty, the first-index rule, the five thresholds. APPEND takes
   the FIRST free slot (the chunk-1 limit of the ordinal rule); MERGE blends
   the two rows with the registered detached strength weights; POP takes the
   worst row under the registered capacity rule;
2. **the read** — the certified formula (`read.py`, unchanged:
   `Σ_j softmax_j(scale·cos(q_t, S_t[j]) over LIVE slots) · ‖S_t[j]‖ · S_t[j]`,
   no renormaliser, all-dead exactly zero), against that same current state;
3. **the update** — the scan's registered order (`algebra
   .sequential_matrix_scan`): read, DECAY, then every lane in order.

No new operators, no threshold changes. `ops=False` (the append-only control)
and `grow` are refused outright — both are chunk-dial concepts this
registration does not carry; a rank prefix is refused by name; a store too big
for one program (`rank × d_v` beyond 16384 register floats) takes the eager
oracle verbatim — the same semantics, the parity fixed point, never a silent
granularity change.

### 1.2 The arithmetic spec — the K5A obstacle, resolved

K5A could not reproduce the registered arm's decision inputs from a kernel
(~48 % of bf16 similarity entries bit-equal, 0/1024 payload norms), because the
registered chain's reduction orders ARE its semantics and they are torch's
own. K6's oracle is new, so the registration defines the arithmetic and both
implementations implement the same definition:

* **every reduction is a balanced binary tree over the reduced axis** (padded
  to its next power of two with exact zeros), a chain of `view(..., n/2, 2)`
  pair-sums — bit-exact between torch and Triton, receipted
  (`reproducibility.tree_fold_bit_equal`);
* **all decision/read arithmetic is fp32** (inputs upcast once at entry) —
  the registered arm's bf16 similarity was a bmm precision artefact, not a
  policy constant, and this registration states its own;
* **correctly-rounded IEEE ops only**: `sqrt.rn`/`div.rn` (bit-identical to
  torch, receipted), and the one transcendental — the read softmax's `exp` —
  is torch's own libdevice function: **`libdevice.exp` vs `torch.exp`:
  0/2,000,000 mismatches** (Triton's default `tl.exp`: 1,768,840);
* every product uses the `mul.rn` intrinsic — Triton contracts a plain `a*b`
  into a following add as an FMA (one rounding instead of two), the x44
  §6.2 failure class, found here again on the first run (a 1-ulp score
  difference in 26 % of read weights vanished with the intrinsic);
* the score's division sits after the value fold:
  `score_j = scale·fold(uq ⊙ S_j)/max(nr_j, EPS)` — the certified
  `scale·cos` with its clamp, and one fewer full register tile than a
  per-element normalised store (the whole store is register-resident; the
  budget is the design constraint, and the registration owns the order).

### 1.3 The kernels

**Forward** — `_k6_seq_fwd_kernel`: ONE program per (batch·head) row holds the
ENTIRE store (`rank × d_v` fp32 = 128×128 at the registered shape) in registers
and walks the window serially: per token the defined-order row norms, the read
scores, the softmax, the read, the decision cosines, the five-op selection, the
MERGE blend, then DECAY and the lanes with `mul.rn`/`add.rn`. It records the
decision channel, the per-token pre-decision liveness, and the segment-entry
checkpoints every 64 tokens. One launch per scan; no atomics; 255 registers,
**0 spills** at the registered shape (8 warps). The rank-axis reductions that
are order-independent because they hold a single nonzero (the write's old row,
the MERGE blend's two rows — the two nonzero products meet in exactly one add
whatever the tree) use native sums; the one order-matters rank reduction (the
read output) is a native sum held to the floor — the K1/K23 precedent, and the
read feeds no decision, so no flip can come of it.

**Backward** — decisions are recorded constants (the standing doctrine). The
continuous channels (read weights, payloads, DECAY gate) carry gradient to
`query`, `payload`, `decay` and the state. Per 64-token segment, in reverse:

1. `persistent_scan_trajectory` (K4's kernel, reused verbatim) replays the
   recorded schedule from the segment's entry checkpoint — the walk is the
   forward's walk, so the recomputed states are **bit-identical** to what a
   storing forward would have saved (asserted in the tests against the
   forward's own checkpoints);
2. `_k6_bwd_prep_kernel` — token-block PARALLEL: recomputes the per-token read
   weights (bit-identically to the forward — the same defined-order
   expressions) and their adjoint coefficients, using the two algebraic
   identities that keep the unit-adjoint cheap:
   `(dur·ur)_j = dscores_j·scores_j` and `(duq·uq)_t = fold(dscores·scores)`;
3. `_k6_bwd_walk_kernel` — value-blocked reverse walk, the adjoint tile
   (`RP × 16`) in registers: lanes last-first, the DECAY adjoint, the
   MERGE-blend scatter against the pre-decay rows, then the read adjoint's
   three state terms and `dquery`. `dlam` crosses value blocks and is summed
   in a fixed order in torch (the K4 partials pattern).

The memory arithmetic, stated and measured (`k6_gates.json`,
`gate_b_memory`): **32 MB of segment checkpoints per scan** (32 × 1 MB at
mb8) + one 67 MB trajectory scratch at a time + ~2.6 MB of per-segment adjoint
channels, against the chunk-2 dial's **1024 MB of chunk-entry saves per scan**
— the wall that OOMed the dial. Measured peak forward-plus-backward at
microbatch 8: **371 MB**.

### 1.4 The wiring

ONE opaque `torch.library` op pair per (layer, window) scan
(`x45::seq_scan_fwd`/`x45::seq_scan_bwd`, the K4/K5A/K5B hosting pattern), the
backward's saves riding autograd itself (the op's outputs ARE the recorded
channel, so no pool bookkeeping). The mixer conversion
(`convert_model_to_sequential`) re-wraps every `MatrixMixer` in place with the
same parameter modules, so a model built by the matched-arms build order and
converted shares the sweep arms' **init bit for bit** and `model.parameters()`
is unchanged — `model.py` and every registered file untouched (new files only).
216 launches per scan fwd+bwd at the registered shape; at 24 mixers per step
the step smoke's host share is not the wall (the launch count is per-SCAN, not
per-chunk — the K5A lesson does not recur here), so the K5B CUDA-graph layer
was deliberately NOT needed for the probe to fit; that is the honest answer to
gate D's graph question, and the graph remains an available lever if a future
2700 s arm wants it.

## 2. Gate A — decision parity (the K5A bar, doubled)

`receipts/k6_gates.json`, `gate_a`. The K5A bar's own fixture, at the
registered shape, bf16 inputs:

| channel | lanes | flips |
| --- | ---: | ---: |
| slot, erase, op, live_after | 262,144 | **0** |
| payload lane (the MERGE blend included) | 8,388,608 | **0** |
| read weights | 4,194,304 | **0** |
| pre-decision liveness | 4,194,304 | **0** |
| **total** | **17,039,360** | **0** |

Plus: exit state and liveness `torch.equal`; the read at 9.5e-7 = **0.25× the
measured fp32-vs-fp64 floor** (floor 3.8e-6 on a read scale of 43.8); the
kernel at 255 registers, 0 spills. The suite's fixtures (18 tests) cover all
five ops via crafted stores (a 0.9-cosine second row forces MERGE, a full
store with an orthogonal payload forces POP — the fresh-state regime rarely
reaches either, which is itself the sweep's finding), ragged token and segment
lengths, the bf16 regime, one-token/one-rank edges, score ties (the lowest
index wins in both implementations), and the all-dead zero read.

## 3. Gate B — gradients (floor first, then the deviation)

`receipts/k6_gates.json`, `gate_b_grads`, at a registered-class shape (batch 2,
256 tokens, rank 128, segment 64). The floor is the fp32 oracle against its own
fp64 twin, per output, measured on the same fixture; the oracle itself is
validated first by an fp64 `gradcheck` over every adjoint branch:

| output | floor | deviation | ratio | within 8× |
| --- | ---: | ---: | ---: | --- |
| dquery | 9.5e-7 | 1.19e-6 | 1.25 | yes |
| dpayload | 3.8e-6 | 7.6e-6 | 2.00 | yes |
| ddecay | 2.3e-5 | 4.6e-5 | 2.00 | yes |
| dstate | 3.8e-6 | 5.7e-6 | 1.50 | yes |

**None of the four is bit-exact, by design at this stage**: the read-weight
adjoint's cross-axis reductions (`dw`, the softmax denominator's adjoint,
`duq`) are autograd-ordered on the oracle side, so the adjoint is not
exact-replayable — the floor discipline is the honest gate, and all four sit
at 1.25–2.0× it. The structural boundaries are pinned by tests in the K4
tradition: a POPped row's gradient terminates at its death token (strictly
after it — the boundary case that bit K4), and the MERGE blend's gradient
reaches BOTH source rows.

## 4. Gate C — the quality probe: per-token BEATS the dial's best

`receipts/k6_probe.json`. The protocol is the sweep's verbatim: the same
`matrix_ops` model (149,981,808 parameters) built by the same order (seed 3401,
the init shared bit for bit with the sweep arms via the in-place mixer
conversion), the same registered flagship stream, the same transferred LR (the
gdn2 nomination's), the same matched-exposure stopping rule (20,022,866
targets — reached in exactly the sweep arms' 180 steps), the same honest
clock, the same document-disjoint calibration eval (7 full windows, 14,336
targets). The comparison reads the banked sweep receipts, including the
extension lane's committed chunk-4 row (**2.1791 nats/byte — the dial's best**,
the row to beat).

| arm | decisions every | eval NLL (nats/byte) | Δ vs chunk 128 | train wall (s) | tok/s | MERGE/tok | live slots | read entropy (nats) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk 128 | 128 tokens | 2.4173 | — | 589 | 34.0k | 0.2484 | 80.2 (at chunk entries) | 2.185 |
| chunk 64 | 64 tokens | 2.3287 | −0.0886 | 605 | 33.1k | 0.0803 | 53.7 | 2.034 |
| chunk 32 | 32 tokens | 2.2462 | −0.1711 | 633 | 31.6k | 0.0268 | 45.2 | 1.949 |
| chunk 8 | 8 tokens | 2.2186 | −0.1987 | 1,050 | 19.1k | 0.0092 | 51.2 | 1.546 |
| chunk 4 (the extension lane's committed row) | 4 tokens | 2.1791 | −0.2382 | 1,674 | 12.0k | 0.0040 | 55.2 | 1.404 |
| **per-token (this arm)** | **every token** | **2.0278** | **−0.3895** | 1,720 | 11.6k | 0.0002 | 73.7 (at every token) | 1.085 |

**S1 lands, and the gradient accelerates into the limit.** The dial was
decelerating (128→64: −0.089; 64→32: −0.083; 32→8: −0.028; 8→4: −0.040); the
per-token limit buys **−0.1513 beyond chunk 4** — more than any single dial
step after 64, and the full dial-to-limit gradient is worth **0.390 nats/byte
at probe scale**, twice the chunk dial's own 0.199-to-chunk-8 (and still
0.15 beyond the dial's last reachable row). The training-loss corroboration
agrees: final train loss 2.454 against chunk 8's 2.606 and chunk 128's 2.807,
at identical exposure and data order. The wall is honest: 1,720 s (11.6 k
tok/s) — the price of chunk-4-level throughput for the quality of the limit.

**S2, the mechanism row, lands except where it is falsified — stated, not
smoothed:** the draft's MERGE prediction ("recovers toward the chunk-8 rate")
is **falsified in the informative direction**: MERGE does not recover, it
completes its monotone collapse to its zero limit (0.2484 → 0.0803 → 0.0268 →
0.0092 → 0.0040 → **0.0002**). The sweep's own mechanism reading is thereby
confirmed at the limit: fresh state never accumulates the duplicate rows that
trigger the merge threshold, because the second similar write CORRECTs the same
row at the token it arrives — redundancy is resolved at write time, and MERGE
is the dial's staleness artifact, not a needed operator at per-token
granularity. CORRECT holds near saturation (0.884; chunk 8: 0.925) with POP at
0.031 and APPEND at 0.084 (the fresh-state store reuses its rows more), the
store stays fuller than every small-chunk row (73.7 live slots of 128, at
every token — the per-token policy never sees yesterday's occupancy), and the
read-weight entropy completes its monotone sharpening to **1.085 nats** — the
sharpest of every arm, the K23 §7 chunk-constant distortion at its zero.

## 5. Gate D — the sequential-decision cost, stated honestly

| measurement (registered shape, mb8) | value |
| --- | ---: |
| forward, one (layer, window) scan | 26.5 ms |
| backward, one scan | **15.3 ms** |
| launches per scan (fwd+bwd) | 216 |
| forward throughput (scan alone) | 618k tok/s |
| 150 M arm step smoke (steady, mb8, 64 windows/step) | 0.97 s ≈ 14.0 k tok/s |
| the probe arm's honest wall (real stream, same clock as the sweep) | 1,720 s = **11.6 k tok/s** |
| peak allocated (fwd+bwd, mb8; the probe's whole run) | 18.05 / 19.97 GB |

A per-token decision walk cannot be as fast as a chunk-128 whole scan, and it
is not: the forward's 26.5 ms is a latency-bound serial walk, real and
visible. But the honest step comparison is against the DIAL's own walls, and
it inverts the intuition: the sweep arms' step cost was never the scan's
arithmetic — it was thousands of launches of per-chunk machinery (the whole
K5 program, graph-captured) — while this arm's per-token decisions ride ONE
kernel launch per scan. Chunk 8 trained at 19.1 k tok/s and chunk 4 at 12.0 k
tok/s; the per-token arm lands at **11.6 k tok/s — chunk-4-class throughput**
at the quality the dial could not reach, and its **backward is cheaper than
its forward** (15.3 vs 26.5 ms), because the segment recompute is
value-blocked and token-parallel while the forward's decision walk is the
serial chain. The part the dial's memory wall made impossible is the cheap
half.

## 6. What did not work, and what is honestly not done

1. **The whole register tile in one program is the binding constraint, and the
   first design did not fit.** Holding the normalised store rows (`ur`) as
   well as the state pushed the 8-warp kernel to 222 spills and 83 ms; the
   fix was spec-level (move the score's division after the fold) plus
   row-level lane writes — 0 spills, 26.5 ms. The size gate (16384 register
   floats) is a refusal, not a limitation to engineer around silently.
2. **Triton's plain `*` contracted into the folds as FMAs** — the x44 §6.2
   class, alive and well on this stack: a 1-ulp score error in 26 % of read
   weights on the first run. Every spec-path product carries `mul.rn` now;
   the receipt's reproducibility rows are the proof, not a style claim.
3. **The rank-axis tree fold is unusable at the whole-store width** — ~11
   µs/token in layout conversions, ~40 µs/token with three of them. The
   order-independent reductions (single-nonzero gathers, the two-nonzero
   MERGE blend) are exact under native sums, and the read (the one
   order-matters rank reduction) is held to the floor — the same corner K1
   receipted for its reads.
4. **The oracle's first `gradcheck` failed for a fixture reason worth
   recording**: the oracle upcast everything to fp32, so the fp64 instrument
   was computing fp32 (numerical derivatives 10 ulps of garbage). The oracle
   now keeps fp64 inputs fp64 — the fp64 twin is the floor reference.
5. **The oracle's autograd had a 0·inf NaN** at the all-dead softmax (the
   `where` selects the prob away but the chain still flows through the
   division) and a 0/0 at `sqrt`'s backward on zero rows — both fixed with
   value-identical guards; the fixtures that found them are in the suite.
6. **No CUDA-graph capture on the ops** — measured not needed at this scale
   (216 launches per scan is per-SCAN, not the K5A per-chunk program); the
   lever exists if a 2700 s arm wants it. Not built, stated as a decision.
7. **No 2700 s claim** — the operator's sign-off on the draft registration
   gates it (the preregistration's own rule). The tuned-LR confirmation
   remains the sweep's named follow-up, inherited by this arm's S1 claim.
8. **The eval boundary is 14,336 targets** and single-seed — the sweep's own
   honest caveat, inherited verbatim; the training-loss corroboration is in
   the probe receipt's trajectory.

## 7. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k6_sequential_matrix.py -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k6_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k6_gates.json
setsid bash -c 'experiments/expressivity_tasks/x45_matrix_emender/run_k6_probe_leased.sh \
    > experiments/expressivity_tasks/x45_matrix_emender/k6_runs/driver.console.txt 2>&1' &
```

## 8. Recommended next step

The operator reads `PREREGISTRATION_SEQUENTIAL_MATRIX.md` and signs it off (or
amends it); with sign-off, the natural follow-up is the dial's own named
sequel: a 2700 s matched-exposure arm of the per-token store against the
banked chunk-128 arm's 1.7576 (the board's rule, at the ~0.4-nat probe-scale
gap this arm measured), with the tuned-LR confirmation as its twin.
