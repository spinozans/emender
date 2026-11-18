# RESULT_GDN2_GROWTH_C3.md — the C3 saturation-regime arms: GDN2-fixed vs GDN2+growth vs the growable machine (emender PENDING)

**Date:** 2026-11-19 · **Status: the GDN2 side COMPLETE — three arms landed and committed; P-G3 ADJUDICATION PENDING the emender arm's landing (the ~36h matched-exposure long pole, detached, ETA ≈ 2026-11-20 07:40, launch receipt `receipts/gdn2_growth_c3_emender_launch.json`, a later session harvests it per the harvest-and-commit doctrine).**

**Authorisation chain:** `PREREGISTRATION_GDN2_GROWTH.md` §7 (operator-signed this session) — the four arms and the falsifiable P-G1/P-G2/P-G3 structure; the supervisor's C3 regime ruling (the long-horizon real-text variant, horizon chosen empirically by the τ_sat calibration scan, the standing 20M-target probe budget with the corrected cooldown-primary scoring, the emender arm fresh in-regime); the supervisor's calibrated-τ ruling on the calibration's escalation receipt (τ_cal below the unreachable 96, horizon 8192, the emender arm RUN at its honest ~36h cost, P-G3 pending its landing). No `k10_*`/`k11_*`/`k12_*` source, no PREREGISTRATION, no `lattice.py`, and no GDN2 model code was edited — the growth mechanism is the wrapper/registrar pattern from the committed `gdn2_growth_smoke.py`, and every PR spectrum is computed by the cross-lane `x45_spectra_sidecar` VERBATIM (exact LAPACK, 32 hard-capped single-thread CPU workers).

## 0. The headline

**The registered τ_sat = 96 is UNREACHABLE on real text at any feasible horizon — the calibration receipts deliver the registered threshold's informative-null verdict permanently — and at the supervisor's calibrated threshold (τ_cal = 19.205, the trained fixed machine's own per-layer smoothed-PR ceiling at the locked horizon), GDN2+growth DOES grow (40 events, 20 of 27 mixers doubling twice to H=8 inside the first window, the one-step wake-up receipted at model level on every event), and beats GDN2-fixed by 0.0486 nats/byte post-cooldown (EMA cross-check 0.0661) — INSIDE the pre-stated 0.075 endpoint-variance band. Growth is directionally real but is NOT established as THE differentiator at this scale: the born-at-8 control lands 0.010 from the grown arm (both ~0.05-0.06 ahead of fixed, both inside the band), so grown-large ≈ born-large at matched grown state, and the ~0.05 capacity gain itself never clears the noise band. Whether OUR addressable-slot machine beats GDN2+growth at matched grown state (P-G3) is exactly the pending emender row.**

## 1. The τ_sat calibration (§5's disclosed protocol, the gate for everything downstream)

The fixed GDN2 arm (the registered 150M machine verbatim, H=2, seed 3401) trained at the full corrected protocol on BOTH candidate horizons of the supervisor-ruled long-horizon stream (the registered flagship documents, full-context windows from eligible documents only — byte_length ≥ context+1, document-bounded, split-disjoint; eligibility receipted: 31,735 / 16,323 train documents yield ≥1 full window at 8192 / 16384; 326 / 167 calibration). The per-boundary participation ratio — exact LAPACK spectra of every (layer, head, 64-token-boundary) state via the sidecar, mean over (batch, head), smoothed over the last 4 boundaries, exactly as registered — on the trained (constant-phase-final, step-175) machine:

| horizon | PR mean | PR max | per-layer smoothed max (min/median/max) | fires at 96 | post-cooldown primary | train wall |
|---:|---:|---:|---|---:|---:|---:|
| 8192 | 7.51 | 23.89 | 5.2 / 10.8 / **19.21** | **0** | 1.8044 | 636 s |
| 16384 | 8.07 | 25.54 | 5.2 / 10.4 / 20.1 | **0** | 1.8599 | 650 s |

**The verdict:** doubling the horizon moves PR by ~0.5 of a possible 128 — GDN2's trained retention regime does not saturate its per-head key space at any feasible horizon. This is §5's own toy anchor receipted on the real module: the §5 ladder maps trained retention ≈0.7 to PR ≈ 5-8 (measured 7.5-8.1), and reaches 96 only at retention ≈ 0.995 — a regime real text does not produce. **The registered threshold's informative-null verdict (growth degenerates to fixed in-regime at τ=96) is delivered by these receipts themselves and stands permanently.** The non-gating co-diagnostic ι is receipted per boundary alongside (mean 0.273 at 8192 / 0.267 at 16384 — the delta-rule write collision against the entry state; ~3× the init anchor 0.092, the interference statistics diverging with training exactly as §5's design intended).

Per the ruling this was ESCALATED with the receipts; the supervisor ruled the disclosed calibrated branch:

* **Horizon 8192 LOCKED** (the shortest scanned horizon; same reachability verdict; the emender fold arithmetic is decisive — the K8 walk is fold-bound, so T=8192 is 4× the banked fold depth, not 8×).
* **τ_cal = 19.205211639404297 LOCKED** — the trained fixed machine's own per-layer smoothed-PR ceiling at the locked horizon (layer 5's post-cooldown smoothed max), committed BEFORE any growth arm ran (`receipts/gdn2_growth_c3_regime.json`, the §5 calibrated branch's own rule). Semantics: growth fires exactly when a boundary is MORE saturated than anything the fixed machine exhibits. The K8 idle-store hazard stated verbatim: PR ~20 of 128 is the fixed machine's exhibited-regime ceiling, NOT absolute capacity; the arm tests growth-past-the-exhibited-regime. Every firing event receipts its PR value, the per-head PR at the fire boundary, the window position, the post-event first-boundary per-head PR, and the mean PR drop at the event.

## 2. The protocol (the standing probe rule transplanted, receipted in the lock)

All arms: seed 3401, matched 20,070,400-target exposure (175 constant-LR steps × 14 full-8192 windows/step), the 24-step linear cooldown (13.7% of the constant phase, inside the Hägele band) as the **PRIMARY** score, EMA-0.99 at the constant-phase end as the ranking-only **secondary**, the constant-phase eval retained for comparability; the eval = the first 24 FULL 8192-byte windows of the document-disjoint calibration split (196,608 targets); one optimizer family across all arms (plain AdamW at the registered gdn2 nomination's transferred settings — the K6/K8/side-by-side convention, identical numbers on both machines); honest clock; the protocol's own grad clip. Kernel-shape receipts at T∈{8192,16384}, H∈{2,4,8}: **no kernel/shape impossibility** (the d_k wall is §2's finding, untouched).

## 3. The arms (post-cooldown PRIMARY / EMA secondary / constant-final; 24-window document-disjoint calibration eval)

| arm | **PRIMARY (post-cooldown)** | EMA (ranking-only) | constant-final | parameters (Δ vs registered) | growth events | trained PR mean |
|---|---:|---:|---:|---|---:|---:|
| `gdn2_fixed` (H=2, the registered arm) | **1.8044** | 2.1056 | 1.8963 | 149,855,670 (+0) | — | 7.51 |
| `gdn2_growth` (τ_cal trigger, snap {2,4,8}) | **1.7558** | 2.0395 | 1.8489 | 250,417,710 (+100,562,040) | **40** | 8.19 |
| `gdn2_fixed_8` (born at H=8, the size control) | **1.7458** | 2.0126 | 1.8259 | 285,614,424 (+135,758,754) | — | 6.95 |
| `emender_growable` (the K8 machinery) | **PENDING** — detached, ETA ≈ 2026-11-20 07:40 | | | 149,981,808 (+0) | (its registered trigger fires mechanically) | |

The margins (positive = the row ahead of the comparison):

| comparison | post-cooldown PRIMARY | EMA cross-check | the pre-stated band |
|---|---:|---:|---:|
| fixed − growth (P-G2) | **0.0486** | 0.0661 | 0.075 |
| fixed − fixed_8 (born-large control) | 0.0586 | 0.0930 | 0.075 |
| growth − fixed_8 (grown vs born at matched grown state) | 0.0100 | 0.0269 | 0.075 |
| growth − emender (P-G3) | **PENDING** | | 0.075 |

**The growth mechanism, receipted at model level (the first model-level execution of the registration's §3):** at step 1 the calibrated trigger fired in 20 of 27 mixers — each doubling twice within a single window (boundaries 64→128 in most layers; two layers fired late in the window at 3200/3264) — while 7 mixers never crossed τ_cal (the per-layer PR heterogeneity the calibration receipts predicted: only layers whose own regime exceeds the fixed machine's ceiling grow). The **one-step wake-up receipted on every one of the 40 events**: the new head's `k_proj` rows receive EXACTLY 0.0 gradient at the event step (the zeroed `o_proj` columns block them) while `o_proj`'s new columns receive nonzero gradient (8.9e-8 at 2→4, 4.9e-5 at 4→8), and the new-head internals train from the second step — the block-level S-3 receipts, now at full model scale under the real stream. Event-time per-head PR receipts: e.g. layer 0's 2→4 fire at smoothed PR 19.81 with the batch-mean PR dropping at the event (the new heads enter exact zeros), and the grown state's trained PR mean 8.19 vs the fixed machine's 7.51 — the grown model holds slightly more effective rank per layer at the same exposure, exactly the divergence the saturation statistics were designed to receipt.

## 4. The verdicts, as far as the banked rows reach

* **P-G1 (the binding receipt) — the fixed state does NOT saturate in-regime.** The calibration receipts are the verdict: at the registered τ=96 nothing fires at any feasible horizon (PR max 25.5 of 128); even the fixed machine's own exhibited ceiling (τ_cal ≈ 19.2 of 128) is crossed only transiently, early in training. The fixed-state bound's empirical shadow is NOT binding in the long-horizon byte-LM regime: capacity is not what pins GDN2-fixed at 8192. Consistent with this, the born-at-8 control buys only 0.0586 (inside the band) for +135.8M parameters.
* **P-G2 (growth is THE differentiator) — the pre-stated within-band branch, honestly: NOT SUPPORTED at this scale.** The trigger fired in-regime (40 events, receipted), and growth beats fixed by 0.0486 post-cooldown (EMA 0.0661) — a real directional gain, but INSIDE the pre-stated 0.075 endpoint-variance band, and it costs +100.6M parameters (§4's honest column: GDN2's keys are learned, so growth adds parameters — the K8 zero-parameter property does not transfer). Per the pre-stated refutation clause this is the informative-null branch's weaker cousin: growth helps, but not by more than the band — growth is not established as THE differentiator at this scale/horizon/exposure.
* **The grown-large vs born-large control (the registration's own control question):** `gdn2_growth` and `gdn2_fixed_8` land 0.0100 apart (EMA 0.0269) — growth TIMING buys nothing detectable over being born large; both sit ~0.05-0.06 ahead of fixed, inside the band. The grown state's advantage over fixed is not separable from noise, and matched grown state ≈ matched born state.
* **P-G3 (addressable slots earn their keep) — ADJUDICATION PENDING the emender arm's landing** (the ~36h matched-exposure long pole; the supervisor's ruling: the honest price of the registered K8 machinery at the locked horizon — fold-bound walk, 4× the banked fold depth, measured 645 s/step at 114.7K targets/step; launch receipt `receipts/gdn2_growth_c3_emender_launch.json` with the full survival-playbook verification — PPID=1 chain, live lease owner/starttime, renewing heartbeats, advancing steps in the retained `training.jsonl`). A later session harvests it and completes the adjudication against the committed `gdn2_growth` row.

## 5. Supersessions and disclosures (the honest engineering ledger)

* **The first `gdn2_growth` run is SUPERSEDED:** it landed (1.8061, committed) with 0 events because the wrappers probed at the DEFAULT registered τ=96 (the lock's calibrated τ was never installed on the wrappers — a driver bug found by reconciling the probe telemetry's smoothed-PR max 32.65 against the 0-events receipt). The calibrated-τ re-run above supersedes it; the stale receipt lives in git history.
* **The first `gdn2_fixed_8` run is SUPERSEDED:** `gdn_config`'s capacity probe ran under the heads-forcing constructor patch and silently re-derived the backbone (width 640→1088) — a different-width model, not the registered arm at H=8. The builder now pre-warms the geometry cache under the banked probe environment (VOCAB patch included) and asserts the banked 640/27 geometry; the corrected control (banked geometry at H=8, +135.8M params disclosed) supersedes the 1.7680 row.
* **The growth arm runs its features pass WITHOUT activation checkpointing** (receipted in `install_growth`): a growth event mutates the live module mid-forward, and reentrant-checkpoint recomputation would re-run the block against a different module — an inconsistent backward. Checkpointing is a bit-identical memory knob on the fixed path (the fixed arms keep the banked path verbatim); the growth arm pays the honest activation memory (microbatch 2, the exact-accumulation memory knob, disclosed per-arm).
* **One broker collision, receipted not patched around:** a lease-release lag (a finished runner lingering in `git push`) plus the broker's documented racy idle-check let the emender runner onto the growth arm's card mid-compile; the growth arm OOM'd, was relaunched pinned to the free card, and the collision is in the launch commit.
* The growth arm's early-checkpoint evals are skipped with the disclosed reason where the parameter set grew past the snapshot's shapes; the train-loss trajectory stands.
* Standing residuals: single seed per arm (3401, the K8 probe's convention); one optimizer family (the transferred AdamW — no Muon arm, per the K6/K8/side-by-side convention); probe scale (20M targets); the τ_cal hazard (§1); the spectra sidecar's non-fatal `SharedMemory` close warning at pool teardown (the cross-lane-owned module, unedited; the statistic is unaffected — the self-test's bit-equality receipt stands).

## 6. Commands

```bash
# the lane's CPU tests (the trigger algebra, the slice/surgery math, the
# protocol constants, the regime filter)
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_gdn2_growth_c3.py -q

# the kernel-shape receipts; the calibration scan (both horizons); the lock
experiments/expressivity_tasks/x45_matrix_emender/run_gdn2_growth_c3_leased.sh kshapes
experiments/expressivity_tasks/x45_matrix_emender/run_gdn2_growth_c3_leased.sh calibrate
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_c3 lock

# one arm (detached, leased, commits on landing); the assemble
experiments/expressivity_tasks/x45_matrix_emender/run_gdn2_growth_c3_leased.sh \
    arm --arm gdn2_growth --microbatch 2
experiments/expressivity_tasks/x45_matrix_emender/run_gdn2_growth_c3_leased.sh assemble

# the pending emender arm (a later session harvests):
#   receipts/gdn2_growth_c3_emender_launch.json (the launch receipt)
#   c3_runs/emender_growable_seed3401/training.jsonl (the live log)
#   -> on landing the runner commits receipts/gdn2_growth_c3_emender_growable.json
#      and `assemble` completes P-G3's adjudication.
```

## Provenance

New files only: `gdn2_growth_c3.py` (the C3 driver), `run_gdn2_growth_c3_leased.sh` (the detached leased runner), `test_gdn2_growth_c3.py` (11 CPU tests), the receipts (`gdn2_growth_c3_kshapes.json`, `gdn2_growth_c3_calibration_{8192,16384}.json`, `gdn2_growth_c3_regime.json`, `gdn2_growth_c3_{gdn2_fixed,gdn2_growth,gdn2_fixed_8}.json`, `gdn2_growth_c3_emender_launch.json`, `gdn2_growth_c3_summary.json`), and the `c3_runs/` artifacts. Committed per artifact throughout (kernel shapes → calibrations → lock → per arm → assemble). The GDN2 model code, the `k10_*`/`k11_*`/`k12_*` sources, the PREREGISTRATIONs, and `lattice.py` are untouched — imported only. Session lambda01, cards 4/5 (this lane's pool; the growth+POP lane's card 6 untouched), torch 2.9.1+cu128, `NDM_PIN_TRITON_AUTOTUNE=0` (the receipted pinned-autotune convention).
