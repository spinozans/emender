# RESULT — the 150M matched-quality arms: do the five ops earn their cost at matched wall-clock?

Result document for [`PREREGISTRATION_MATRIX_EMENDER.md`](PREREGISTRATION_MATRIX_EMENDER.md)
(sha256 `20a163c9…74dec2`, bound in every arm receipt). **Registered before measurement**;
this board is its Wave A at fixed rank, run under the registered honest stopping rule
(**2700 active seconds**) on the registered flagship stream, evaluated on the registered
calibration split with the 150 M board's own eval boundary. Every arm receipt binds the
stream identity, the lattice registration (`99bed437…7c`, sha256 of
`PREREGISTRATION_150M_LATTICE.md`), the matrix registration, the K5B receipts and the
recorded cd gate receipts it re-measures. Single seed (**3401**, the standing seed) —
stated everywhere; no error bars exist in this board.

**Headline, stated as narrowly as the evidence supports.**

* **The speed program did its job.** With the K5B fast path enabled, the matrix_ops
  arm trained at **33,991 bytes/s** — the same class as gdn2's same-session
  **33,958** — so the matched-time board is no longer confounded by the 2.1–2.7×
  throughput handicap of the recorded gate-D receipts (13.2 k / 15.9 k profiled vs
  28.1 k). Exposure at matched time came out **matched to 0.13 %** (91.87 M vs
  91.75 M bytes): the arms saw essentially the same data.
* **At matched parameters, matched time and matched exposure, the five-op matrix
  arm LOSES to plain GDN2 by 0.5600 nats/byte** (calibration 1.7576 vs 1.1976
  same-session; 0.6169 against the RECORDED board's 1.1407). The registered C1
  ordering *five-ops > append-only > no-management* is **not met at 150 M in this
  configuration**, and per the falsifier's own scoping clause the 2.13 M
  0.66-nat ops result is scoped as **tape-specific**.
* **The append-only control could not be trained to the boundary at all**: it
  deterministically diverges (non-finite gradient norm, loss finite) at **step 56
  of 2700 s**, bit-identically across two independent runs, under the same
  optimizer with which the ops arm completed 832 steps. Receipted as a **DNF** —
  the registered fail-closed rule (`error_if_nonfinite=True`) was not bent
  (supervisor ruling). The stability contrast is itself the cleanest C1 evidence
  of the night: **management stabilises training in the matrix form**; but the
  append control is *whole-vs-crippled* (ops-off removes death itself — the
  operator's ruling, §Confounds), so it is not a clean ablation either.
* **C2 is untouched by this board**: the claim is monotonicity in **rank** at
  fixed parameters; this board runs one rank (128) only. The rank sweep is the
  measurement this board's own evidence demands, and it is now affordable.

---

## 1. The board

Protocol for every row: flagship stream (beyond byte 100,000,000,000,
manifest sha `48b9095e…c414`, fail-closed identity checks), 64 windows × 2048 bytes
per step, microbatch 8, seed 3401, the honest clock (starts at step 1; includes the
stream read, any compile/capture, forward/backward, clip, AdamW; **no untimed warmup
steps for any arm**), AdamW fused with the registered gdn2 nomination's settings
(lr 5.42e-4, warmup 69 steps, betas (0.9, 0.9392004369596886), weight decay
0.20059436743465126), clip 1.0 fail-closed. Eval: the calibration split, 24 full
windows max, batch 1, recorded-read path — the `runs_150m` boundary (every row here
scored the same **7 windows / 14,336 targets**).

| arm | params | calibration nats/byte | steps | bytes seen | tok/s (bytes/s) | active s | scan path that actually ran |
|---|---:|---:|---:|---:|---:|---:|---|
| `gdn2` (same-session) | 149,855,670 | **1.1976** | 831 | 91,751,599 | 33,958 | 2701.9 | its own implementation (external Triton chunk kernel, eager) |
| `matrix_ops` | 149,981,808 | **1.7576** | 832 | 91,867,164 | 33,991 | 2702.7 | **K5B whole-scan fast path** (one op per (layer, window) scan, CUDA-graph replay, compiled wrapper) |
| `matrix_append` | 149,981,808 | **DNF** — diverged at step 56 | 55 | 6,163,209 | ~37,257 (steps 6–55) | 170.5 at crash | registered cd-protocol path verbatim (compiled shared scan; K5B **refuses** the append-only control by its gate-E semantics) |
| `gdn2` (RECORDED, `runs_150m/lattice_gdn2`, an earlier session) | 149,855,670 | 1.1407 | 877 | 96,903,492 | 35,875 | 2701.1 | its own implementation |

Board context (the RECORDED 150 M board, same protocol family): transformer 1.2807,
gdn2 1.1407, tape emender bounded 2.5993 (at 36.5 M bytes — the tape arm's own
throughput handicap). The matrix_ops arm's 1.7576 sits between gdn2 and the tape
emender, and behind the transformer.

The gdn2 same-session row is labelled **same-session** against the RECORDED row, as
the matched-arms protocol requires: this session's card drew ~5.3 % slower than the
recorded one (33,958 vs 35,875 tok/s at identical kernel work), so the same-session
run saw 5.2 % fewer bytes and scores 0.057 nats worse — the session draw, not a code
difference. Both numbers are reported; the C1 comparisons below use the
**same-session** pair, whose exposure is matched to 0.13 %.

Per-arm throughput of what actually ran (so no reader mistakes an implementation
speed for a model property): gdn2 — its own eager Triton path; matrix_ops — the K5B
whole-scan op pair (dispatch ledger for the whole run: 159,744 forward + 159,744
backward op calls, of which 24 eager first-calls + 24 captures per direction and
159,720 replays per direction, **0 fallbacks**); matrix_append — the registered
compiled per-chunk scan, which unprofiled on this card runs at **~37.3 k bytes/s**
(2.95 s/step), not the recorded cd receipt's 15,856 tok/s — that figure was
measured **with `torch.profiler` attached** on a slow host draw. Even a completed
append row would therefore have been nearly exposure-matched; the DNF is not hiding
behind a confound.

## 2. C1 — management, against the pre-registered falsifier verbatim

> **C1 management.** At matched parameters, FLOPs and exposure, five-ops >
> append-only > no-management (plain GDN2) on held-out nats/byte. *Falsified if*
> the ops add nothing; then the 2.13 M 0.66-nat result is scoped as tape-specific
> and reported as such.

**Verdict: the ordering is not met at 150 M in this configuration — C1 fails on
its second inequality, and the falsifier's scoping clause applies.**

* *five-ops vs no-management*: at matched parameters (both within 0.1 % of the
  150,003,984 board), matched time (2701.9 s vs 2702.7 s) and matched exposure
  (91.75 M vs 91.87 M bytes, 0.13 % apart), the five-op matrix arm scores
  **1.7576 against gdn2's 1.1976** same-session: **worse by 0.5600 nats/byte**
  (0.6169 against the RECORDED 1.1407). FLOPs are not matched (the registration's
  ideal); time and exposure are, and both favour no model here.
* *five-ops vs append-only*: the control **could not be trained to the boundary**
  (deterministic divergence at step 56, §3), so no endpoint comparison exists.
  What the board does establish is the **stability contrast**: at the same seed,
  optimizer, stream and stopping rule, the ops arm's gradients stayed finite for
  832 steps while the append-only arm's died at 56 — bit-identically across two
  runs. The ops *do* add something (trainability); but the operator's ruling
  (§Confounds) is that the append control is whole-vs-crippled — ops-off removes
  death itself — so this is evidence about *management's* value, not a clean
  ablation of management.
* *The scoping clause*: the registered falsifier says that if the ops add nothing,
  the 2.13 M 0.66-nat result is tape-specific. The 150 M evidence is harsher
  than "add nothing": at matched time the ops arm adds −0.56 nats against plain
  GDN2. The tape-era ops result does **not** transfer to the matrix form at 150 M
  under this protocol, and is scoped as tape-specific accordingly.

Caveats that bound this verdict (all stated, none softening the numbers): single
seed; the matrix arms inherit the gdn2 nomination's optimizer **unsearched** (the
gdn2 arm's lr/init were CMA-ES-searched over 144 candidates for *that*
architecture — the matrix form has had no such search, and its DNF control shows
the borrowed optimizer is not automatically stable in this family); this is the
first 2700 s run of the matrix architecture ever, against a mature baseline lane.

## 3. The DNF receipt — the append-only control

Full receipt: [`runs_150m_matched/matrix_append/dnf.json`](runs_150m_matched/matrix_append/dnf.json).
Run dirs and consoles retained: `matrix_append/` (2700 s request, crashed),
`matrix_append_repro/` (240 s cap, same crash), `matrix_append.console.txt`.

* **Configuration**: the registered cd-protocol path verbatim
  (`compile_scan=True` on the ONE shared scan, `fused_whole_scan=False`) — the
  exact mode of the `cd_matrix_append_c128` receipt — chunk 128, microbatch 8,
  seed 3401, the registered gdn2-nomination optimizer.
* **Failure**: non-finite total gradient norm at **step 56**, with the loss
  finite (2.76 at the last completed step 55) and the lr still in warmup
  (56/69 × 5.42e-4 ≈ 4.4e-4). The harness raised through the registered
  fail-closed rule (`clip_grad_norm_(…, error_if_nonfinite=True)`); no update was
  applied at step 56 and no number was massaged.
* **Reproducibility**: two independent runs produce **bit-identical loss/bytes/step
  channels through step 55** (step 50 loss 2.7157, step 55 loss
  2.7641528076442088 in both) and the same crash at step 56; the logs differ only
  in wall-clock fields.
* **Cross-arm contrast (the C1 evidence)**: `matrix_ops` — the same model with
  the five-op policy, on the K5B path — sailed through the same step-56 prefix
  and completed 832 steps / 91.87 M bytes with finite gradients throughout.
  Single seed, stated.
* **Cross-path fact (a knife-edge, recorded as such)**: the *first* append launch
  (§Disclosures), which ran the same model on the K5B fallback path (per-chunk
  K5A+K4 stack: decisions bit-identical to the registered router, read weights
  within the measured bf16 noise floor), was healthy at **step 72** when stopped.
  A noise-floor reduction-order difference decides whether this configuration
  diverges: append-only is numerically fragile and knife-edged at this training
  point. That is implementation-precision sensitivity of a fragile configuration
  — **not** evidence the registered path is broken, and not a licence to
  substitute paths.
* **Ruling**: supervisor decision — the fail-closed rule is not bent
  mid-experiment; the divergence is receipted as the arm's result and as C1
  stability evidence.

## 4. C2 — capacity at fixed parameters, against the falsifier verbatim

> **C2 capacity.** At **fixed parameters**, held-out nats/byte improves
> monotonically with rank. *Falsified if* the curve is flat.

**Verdict: untested by this board — the rank axis is untouched.** The board runs
rank 128 only (the cd-protocol rank; rank is capacity, not a parameter, so the
matrix arms are parameter-matched at every rank). The C2 curve requires the rank
sweep over the precompiled set {64, 128, 256, …} under this same protocol, which
the K5B path now makes affordable; a within-run trajectory is not the C2 curve.
Nothing here either supports or falsifies C2; the loss trajectory (§6) is included
so the capacity question stays visible, not as C2 evidence.

## 5. Confounds and caveats (every one stated)

1. **Single seed (3401), single session where feasible.** gdn2 and matrix_ops ran
   sequentially on the same card in one session; the append control crashed in the
   same session. No error bars; a second seed is a named follow-up (§8), not run.
2. **Matched-TIME, not matched-exposure** — stated as the 150 M board did — except
   that here exposure came out matched to 0.13 % between the two finished arms by
   the speed program's doing, which *strengthens* the gdn2-vs-ops comparison. The
   append row, had it completed, would also have been nearly exposure-matched
   (~37.3 k bytes/s unprofiled, §1).
3. **The architectural asymmetry of the control (operator ruling, recorded
   verbatim in spirit)**: in the matrix form, ops-off removes not just management
   but **death itself** — death is a schedule/POP decision; with ops off no slot
   ever dies and reads forever spread over dead rows. The append control is
   **whole-vs-crippled, not a clean ablation**; the matched-exposure twin
   originally approved was **skipped** on this basis (its marginal value does not
   justify its cost). The causal question the twin would have probed — how much of
   the ops arm's advantage is *management* rather than *death/liveness* — belongs
   to the sequentiality sweep (§8).
4. **Unsearched optimizer for the matrix arms**: both matrix arms inherit the
   registered gdn2 nomination's hyperparameters (the x45 lane's standing
   convention, identical to the cd gate receipts), which were searched for a
   different architecture. The gdn2 baseline's hyperparameters were searched for
   itself. The DNF shows the borrowed settings are not automatically stable in
   this family; a per-arm search (the CMA-ES machinery) is a named follow-up for
   any future append endpoint, not run here.
5. **The matrix lane's youth**: first 2700 s run of this architecture, against a
   mature baseline (the gdn2 lane's nomination search, its recorded board, its
   re-measures). A 0.56-nat gap at matched exposure is a real result; how much of
   it is architecture and how much is untuned training is exactly what the
   follow-ups separate.
6. **K5B fast-path enablement (the config decision, stated)**: matrix_ops ran
   `fused_whole_scan=True` behind the compiled wrapper — a host-cost knob whose
   schedule, reads, state, liveness and gradients are bit-equal to the registered
   per-chunk path by construction (`x45 receipts/k5b_whole_scan.json`,
   sha `5f2c54ab…1dee`; the chunk-128 parity row this board added is in
   `runs_150m_matched/k5b_chunk128_parity.json`: bit-equal on every channel and
   gradient at the run's chunk length, all five ops exercised —
   append 6,129 / correct 1,646 / pop 14,515 / merge 210 / none 43,036 lanes).
   Config hashes of what actually ran: gdn2
   `6f27da3c47319ec9e48c2d75298007ee5bcd7d74b70c1550b789d16db734d9cc`, matrix_ops
   `afdd629e197b2fa48500f579e9d6e76e5c6b3a495e06ebae0aee53696e24323c`.
7. **Dynamo hosting observation, receipted not fixed**: under `torch.compile` the
   whole-scan wrapper hit dynamo's recompile limit at the 8th mixer slot
   (`slot_id == 8`), so mixer slots 0–7 run compiled-hosted and 8–23 eager-hosted.
   The op pair stays opaque in both forms and the K5B parity receipts cover both
   (compiled-vs-eager bit-equality is gate A/C's own row); the measured throughput
   matches the banked K5B class (33,991 here against 31,046–33,300 banked). No
   registered path was modified.
8. **The gdn2 baseline is labelled same-session vs RECORDED** (§1): this board's
   C1 numbers use the same-session pair; the RECORDED row is carried for
   continuity with the 150 M board.

## 6. The trajectory (loss curves, so the C2/capacity question stays visible)

Matched-time cross-sections (the step nearest each time point; bytes are within
0.13 % between arms at every row, so the columns are also matched-exposure):

| active s | gdn2 step / loss | matrix_ops step / loss |
|---:|---|---|
| 300 | 93 / 2.144 | 89 / 2.735 |
| 600 | 185 / 1.996 | 180 / 2.807 |
| 900 | 276 / 1.977 | 273 / 2.505 |
| 1200 | 367 / 1.561 | 365 / 3.219 |
| 1800 | 552 / 1.423 | 551 / 2.278 |
| 2400 | 737 / 1.508 | 737 / 2.166 |
| 2700 | 830 / 1.381 | 831 / 2.180 |

The matrix_ops curve is still descending at the boundary (step-832 train loss
1.9313, its best steps ~1.90 at ~2,360 s). In the matched-step comparison gdn2
leads at **every one of the 812 matched steps beyond step 19** (the first 19
steps alternate, the random-init regime); at matched time the ops arm's
instantaneous single-batch loss is lower in only two of the 53 sampled rows
(t≈100 s and t≈650 s — batch noise, not a crossing of any smoothed curve). The
append-only curve, while it lived, tracked matrix_ops closely (step-50 loss
2.7157 vs matrix_ops' 2.7275 at the same step) — then died. Full per-step curves:
`runs_150m_matched/*/training.jsonl`.

## 7. Disclosures (honesty ledger)

* **The killed first append launch.** The first `matrix_append` launch requested
  the K5B fast path; the whole-scan op **refuses the append-only control by its
  registered gate-E semantics** (`ops=False` takes the per-chunk stack verbatim),
  so it ran on the fallback at ~11.9 s/step. It was stopped at ~step 72 as a
  dominated configuration; its partial run directory was deliberately discarded
  and **its console log was not retained** (deleted in the relaunch — an
  inadvertent loss, disclosed here). The observed facts from it live in §3's
  cross-path fact, recorded from the operator transcript: healthy at step 72,
  loss 2.80, no divergence through 72 steps, on the K5B fallback path. The
  current `matrix_append.console.txt` is the registered-path crash console.
* **The skipped matched-exposure twin.** Originally approved by the supervisor
  (Option B), then **skipped by the operator's scope-change ruling**: in the
  matrix form ops-off removes death itself, so the append control is
  whole-vs-crippled and the twin's marginal value does not justify its cost.
  Recorded here as the reason no twin exists; no partial twin was run.
* **The DNF ruling.** The supervisor ruled against training through the
  divergence (skip-nonfinite-steps) and against substituting paths: the
  registered fail-closed rule stands, the divergence is receipted as the result
  (§3), and the K5B gate-E refusal semantics are correct as receipted — the
  fallback is the design working, not a bug to fix.

## 8. The next measurements this board's own evidence demands (named, not run)

1. **The sequentiality sweep** — the causal probe for the management-vs-death
   question the skipped twin leaves open: hold the ops policy fixed and vary the
   *granularity* of the decision schedule (chunk length, and death/POP cadence)
   against the same protocol. This is where "what do the ops actually buy" and
   the whole-vs-crippled asymmetry (§5.3) become separable.
2. **The rank sweep for C2** (and C3 behind it): {64, 128, 256} at fixed
   parameters under this same matched-time board, now affordable on the K5B
   path. C2 remains untested (§4); this is its falsifier's own demand.
3. **A second seed** for the gdn2-vs-matrix_ops gap (single seed here; the 0.56-nat
   margin is large but unbarred).
4. **A searched append configuration** if an append endpoint is ever wanted: the
   CMA-ES machinery can find a stable append lr (the borrowed gdn2 nomination's
   settings deterministically diverge, §3). A future lane, not this one.

## 9. Commands (the board's provenance)

```bash
# chunk-128 K5B parity (gate A re-asserted at the board's chunk length)
.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.matched_arms parity \
    --chunk-tokens 128 --out runs_150m_matched/k5b_chunk128_parity.json
# each arm, detached under an exclusive lease (cards 0/1)
setsid bash experiments/expressivity_tasks/x45_matrix_emender/run_matched_arms_leased.sh \
    gdn2          runs_150m_matched/gdn2 2700 3401 128
setsid bash experiments/expressivity_tasks/x45_matrix_emender/run_matched_arms_leased.sh \
    matrix_ops    runs_150m_matched/matrix_ops 2700 3401 128 k5b
setsid bash experiments/expressivity_tasks/x45_matrix_emender/run_matched_arms_leased.sh \
    matrix_append runs_150m_matched/matrix_append 2700 3401 128 registered   # DNF, §3
```

The harness (`x45_matrix_emender/matched_arms.py`) mirrors the registered runner's
convention (`x42_pile_scale/lattice.py`, read-only, untouched) and fails closed on
wrong streams, lease mismatches and nonfinite losses exactly as it does.
