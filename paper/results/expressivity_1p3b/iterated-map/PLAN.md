# Task A3 — Iterated-Map Depth-Wall Test (8M reference, 1.3B decisive) — Plan

Task: `taskA3-iterated-map` (paper-fanout). Supervisor rulings from the
pre-flight decision round (recorded verbatim in intent):

- **D1**: Phase 0 (S5 record salvage, in `../capacity/`) = re-train GDN2 to
  the same pre-registered stop with weights saved, eval in-process with
  reduced eval batches. Approved.
- **D2**: The 1.3B depth-wall test runs on **`modular_quadratic`**
  x_{t+1} = (x_t^2 + c_t) mod p — the family that actually carries the 8M
  section-6 depth wall. The task brief's named `iterated_nonlinear_map`
  (logistic) family is the **contractive control**: its measured
  mixed-driver Lyapunov exponent is **lambda = -0.2012 < 0**
  ([`lyapunov_logistic_map.json`](lyapunov_logistic_map.json)), so T* =
  ln(1/eps)/lambda is not a finite kill length there (state error decays,
  not compounds). This corrects the brief, which cited the wrong task
  family as the wall vehicle; the correction goes on the record. No 1.3B
  logistic control arm (the 8M section-6 contractive control already
  exists). The T* framing is dropped for the decisive task (Z_p is
  non-contracting, not chaotic-with-lambda); the reference readout is the
  8M wall curves.
- **D3**: Phase 1 reuses the existing 8M e97/e97-linear/e97-raw results and
  RETRAINS a parameter-matched ~8M gdn (3 seeds); the old 1.6M gdn numbers
  are stated alongside as the historical arm with the mismatch noted.
- **Write budget**: take the brief's pre-authorized "document why the wall
  holds regardless via the nonlinearity requirement" route; do NOT invent a
  D>360 state-dim task variant (no 8M reference; breaks comparability).

## Phase 1 — 8M from-scratch reference (architecture-isolating)

**Wall-map 8M reference (modular_quadratic, no new training):** the 4-seed
symmetric-width sweep and 8-seed confirmation runs
(`experiments/grok_symmetric_width/RESULTS.md`, `CONFIRM.md`,
`confirm_signature.json`): e97 flat (0.98 -> 0.98 to T=4096 at the decisive
cells) while e97-lin falls to ~0.47-0.65 and gdn2 to ~0.53-0.56. These are
the curves the 1.3B readout is scored against.

**Contractive-control 8M table (logistic `iterated_nonlinear_map`, existing
runs reused verbatim):** `experiments/expressivity_tasks/results/
e97raw_iterated_nonlinear_map__{arm}__seed{42,123,456}.json` — e97, e97-linear,
e97-raw, and the historical 1.6M-param gdn arm, all trained from random init
by the existing runner (`run_e97_raw_expressivity.py` -> `train_hybrid.py`:
5000 steps, schedulefree AdamW, lr 3e-4, depth 4, dim 256, batch 32, T=128,
eval at T in {128,256,512,1024} = 8x, 8 batches/length).

**NEW: parameter-matched gdn-8m arm (3 seeds, the D3 ruling).** The historical
gdn arm is 1.6M params at the suite's matched shape vs e97's 7.4M — a review
hole. New arm `gdn-8m`: identical to the `gdn` arm (level `gdn`, head_dim 64,
dim 256, depth 4) with `--expansion 2.5` -> d_inner=640, 10 heads/layer,
**7,937,536 params vs e97's 7,376,896 (+7.6%, the closest count ABOVE e97 the
FLA layer's divisibility constraints allow; the excess favors gdn, so the
comparison is conservative)**. Same task, same 5000 steps, same seeds
{42, 123, 456}, same eval grid and format; results land in
`experiments/expressivity_tasks/results/` as
`e97raw_iterated_nonlinear_map__gdn-8m__seed{seed}.json` (runner arm table
updated; launched via `train_hybrid.py` directly under `gpu_lease.sh` leases
because the battery's own scheduler predates the lease broker).

Expected honest outcome (pre-registered): with lambda = -0.201 there is no
compounding wall on this family, so gdn-8m is expected NOT to collapse (like
the 1.6M historical arm: 0.862 -> 0.819 over 8x). Either outcome is reported
verbatim; this table isolates *parameters* as not the lever on the
contractive family, complementing the wall-map separation whose lever is the
per-step state nonlinearity.

## Phase 2 — 1.3B decisive fine-tune (modular_quadratic)

- **Task**: `modular_quadratic`, p=256, G=5 (the section-6 decisive cell),
  running supervision; inputs raw c-token ids 0..4 into the p50k LM; target =
  x_t in 0..255; metric = full-vocab argmax == class id (identical binding to
  Task A). Random baseline 1/256 = 0.0039.
- **Models**: the bound loss-parity checkpoints — E97 `step_2300930`
  (150.8B tokens) and GDN2-MLP `step_2323616` (152.28B tokens) — with
  Schedule-Free y-mode (train weight) recovery, seed 42.
- **Recipe (IDENTICAL both models)**: AdamW (0.9, 0.95), lr 5e-5 const,
  warmup 300, wd 0.01, grad clip 0.5, batch 32, bf16, full fine-tune, single
  trained length T=128 (multiple of 16, E97 Triton constraint).
- **Step cap**: derived ONLY from measured throughput, identical for both
  models, fixed BEFORE any accuracy result is seen (same discipline as
  taskA2): `cap = int(0.95 * (budget_gpu_seconds_total - reserve) / (s_E97 +
  s_GDN2))` with budget 18000 GPU-s (5.0 GPU-h, inside the brief's 4-6 GPU-h
  envelope) and reserve 1500 s (2x checkpoint load, 2x final sweep). Measured
  smokes recorded in `budget.json` before launch.
- **Plateau stop (pre-registered, identical both models)**: fixed-val-set loss
  probes every 100 steps at T=128 (4 x 32); plateau = relative improvement
  < 1% over the trailing 300 steps AND < 1% over the trailing 200 steps at
  steps >= min(1000, cap), or val loss < 1e-4. NO curriculum escape (single
  trained length by design). If the cap hits before plateau, report the
  trajectory shapes at cutoff plainly.
- **Readout (at the stop)**: length extrapolation T in {128 (trained), 256,
  512, 1024, 2048, 4096, 8192}, 8 batches/length, grad-free, B*T <= 32768
  (batch 8 at T=4096, batch 4 at T=8192), OOM-halving failover. Trajectory
  probes (T in {128, 512, 2048}) every 250 steps. Scored against the 8M wall
  curves (e97 flat vs linear arms collapsing) — the question: does 152B-token
  pretraining + a deep per-token MLP buy GDN2 out of the wall (theory says
  no: the nonlinearity cannot be delegated to the affine state fold), or does
  it crater at far T while E97 holds?
- **Write-budget documentation (the brief's pre-authorized route)**:
  GDN2-MLP's per-token state-write budget from its config
  ([`../gdn2_args.json`](../gdn2_args.json): depth 12, n_heads 30) = 12 x 30
  = **360 rank-one delta-rule writes per token** (each head applies one
  rank-one update to its (K x V) state per token). The map's state is a
  single field element (D=1 << 360), so the write budget is NOT the binding
  constraint; the binding constraint is the **nonlinearity requirement**: the
  GDN2 state channel is affine in its state (the paper's
  `gdn_state_channel_depth_bounded_in_time`), the map's per-step transition
  x -> x^2 + c mod p is not representable by an affine state update, and the
  per-token MLP is a fixed O(1)-depth function of (token, state readout) that
  cannot inject nonlinearity INTO the temporal state channel. On Z_p the
  error, once made, never decays (non-contracting), so a linear-state
  approximation has no error recovery at any length.
- **OOM hardening** (inherited from phase 0, protocol-neutral):
  expandable_segments allocator; exact-gradient sub-batch fallback with
  per-sub-batch empty_cache; weights saved at the stop (outside the worktree);
  eval-batch reduction on OOM only, never protocol changes.

## Outputs

- `phase1_8m_control_table.json` + phase-1 run jsons (in
  `experiments/expressivity_tasks/results/`, same format) — the 8M
  contractive-control table incl. the new gdn-8m arm.
- `budget.json` — measured throughput + cap derivation (before any result).
- `ft_e97_modular_quadratic.json`(+`_sweep`) / `ft_gdn2_modular_quadratic.json`
  (+`_sweep`) — the 1.3B decisive runs.
- `SUMMARY.md` — the two headline tables (8M: error-vs-T both families x 3
  seeds; 1.3B: error-vs-T both models vs the 8M wall curves), negatives
  verbatim.
