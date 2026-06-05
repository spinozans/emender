# E99 1.3B LM — matched head-to-head controls

**Task:** `run-matched-1-3b`. Bounded, matched controls for interpreting the E99
LM-CMA96 result — **not** a search. Same production LM path
(`train.py` + `ndm/models/ladder_lm.py` / FLA-GDN — the path the `wire-e99-e98`
sanity wired and verified), same Pile/tokenizer/ctx, same budget caps, same
held-out method, same throughput accounting, and the same checkpoint round-trip
gate as the E99 top-up. Measured **2026-06-05 (UTC)** on **idle** RTX 6000 Ada
GPUs **0–6** (verified 2 MiB / 0 % before launch; GPU 7 left to a small sibling
job; no other job preempted).

All training is **REAL**: real Pile tokens (`/home/erikg/elman/data/pile.txt`,
`p50k_base`, ctx 2048), real forward/backward, schedule-free AdamW. No mock data.
`paper/main.typ` untouched. **No checkpoint published/staged/uploaded** (round-trip
ckpts deleted after use). No push. No multi-day run.

---

## 0. TL;DR — does E99 improve over the controls?

At the **matched 1.3B-class short-run budget** (15 min wallclock/seed, 3 seeds/arm,
handoff convention), comparing the E99 primary candidate `typed-gdn2-lm` against
the two live controls:

> **Over dense GDN-2 (`fla-gdn`): NO.** Dense native GDN-2 / GatedDeltaNet is
> better than the E99 typed candidate both **wallclock-matched** (≈1.7 nats lower
> AvgLoss, and 3.2× more tokens/min) **and token-matched** (read off each loss
> curve at the common ≈2.1 M-token budget: dense GDN-2 **5.98** vs E99 typed
> **6.98** nats/token).
>
> **Over E98-CMA (`e98-cma-lm`): only on STABILITY.** E98-CMA **diverged to NaN on
> 3/3 seeds** before the window ended (the hard-stop fired each time). But it
> **reached a *lower* min train-loss (5.56–5.91) than the E99 typed candidate ever
> did (≈6.8–7.0)** before blowing up — so this is a *stability* win for E99, not a
> loss/capability win. E98-CMA's divergence is attributable to the untuned 5.38×
> knob-LR carried from the 8M `cma-capability` winner — exactly the thing the
> sanity report flagged for the downstream CMA to re-tune.
>
> **Net: at this matched short budget and at these single (un-re-tuned) operating
> points, the E99 typed candidate does NOT demonstrate a Pile-LM-loss improvement
> over either control.** Dense GDN-2 is the strongest *stable* control. This is the
> honest input the `synthesize-e99-1-3b` task needs; the important caveats (single
> operating point, fp32 throughput handicap, search not yet run) are in §6.

**Budget:** 9 runs, **2.23 GPU-hours** spent of the **5.25 GPU-h** matched-controls
ceiling (`budget_caps.json`). Hard-stop fired 3× (e98 NaN). No overrun.

| arm | role | stable? | AvgLoss (nats/tok) | Final | held-out BPB | tok/s | round-trip |
|---|---|:---:|---:|---:|---:|---:|:---:|
| **fla-gdn** | dense GDN-2 control | ✅ 3/3 | **6.227** (6.17–6.26) | 5.80 (best 5.45) | **2.047** | 7 638 | ✅ |
| **typed-gdn2-lm** | **E99 primary** | ✅ 3/3 | 7.942 (7.92–7.96) | 7.04 (best 6.78) | 2.602 | 2 370 | ✅ |
| **e98-cma-lm** | E98-CMA control | ❌ NaN 3/3 | — (diverged) | — | — | 2 268 | ❌ (NaN) |

Mean over 3 seeds; range in parentheses. Lower is better. Held-out BPB is a
labelled extension (see §4 conversion note). Raw per-(arm,seed) JSON +
`controls_summary.{json,csv}` committed under
`experiments/e99_1p3b_controls/results/`.

---

## 1. Method — matched to E99 / the handoff

Driver: `experiments/e99_1p3b_controls/e99_lm_controls.py` (extends the sanity
driver `experiments/e99_1p3b_sanity/e99_lm_sanity.py`). Orchestrator:
`experiments/e99_1p3b_controls/run_controls.py`.

- **Configs are byte-identical** to the sanity / `candidate_configs.json` shapes
  (copied verbatim): `typed-gdn2-lm` (E99, dim 3072 / depth 22 / 96 heads, fp32,
  head-type logits `[3.9995,−1.9008,−0.9211,−2.8866,2.4146]`, knob-LR 1.0),
  `fla-gdn` (dense GDN-2, dim 2688 / depth 21 / 44 heads, bf16, native FLA
  GatedDeltaNet), `e98-cma-lm` (E98-CMA, dim 3072 / depth 17 / 192 heads, fp32,
  corner-mixture `[0.4015,0.2821,0.0089,0.3075]`, knob-LR **5.38**).
- **Budget unit = wallclock-matched `--train_minutes 15`** per seed — the handoff
  primary convention (`--train_minutes 15 --params 1270M --chunk_size 2048`).
  3 seeds/arm (the `budget_caps.json` recommendation). Each run logs its full
  **loss-vs-tokens curve** so a **token-matched** cross-walk is read off a single
  wallclock-matched run (§3).
- **Fitness = handoff AvgLoss / Final loss** over the train window (nats/token),
  reported verbatim so columns line up; held-out BPB reported **alongside**, never
  replacing.
- **Round-trip gate** (the hard requirement, reused from sanity): train → record
  loss on a fixed held batch → save `model_state_dict` → **reload in a fresh
  process** → recompute → compare (tol 1e-2 nats/token, 0 missing/0 unexpected).
  A strict-clean load is *not* treated as sufficient (the `PILE_BPB_MEASURED` E88
  failure mode: 0/0 keys yet ~17.6 nats/token forward-mismatch).
- **Hard-stop**: NaN → stop+log; CUDA OOM → stop+log; per-job outer walltime
  ceiling (`--walltime_safety 1.15`, orchestrator outer-timeout 22 min) → kill+log.
  The orchestrator also refuses to launch if the projected aggregate would exceed
  the 5.25 GPU-h ceiling, and **never co-locates two jobs on one card** (a 1.3B
  fp32 job peaks ~44 GB on a 48 GB card).

---

## 2. Results — per arm (3 seeds, wallclock-matched 15 min)

### 2.1 `fla-gdn` — dense native GDN-2 / GatedDeltaNet (CONTROL)

Stable on all 3 seeds (`budget_reached`, no NaN). 1.3524 B params, bf16, ~7 638
tok/s ⇒ ~6.5 M tokens / 15-min window, peak 20.9 GB.

| metric | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| AvgLoss (nats/tok) | 6.262 | 6.170 | 6.250 | **6.227** |
| Final loss | 5.453 | 5.548 | 6.402 | 5.801 |
| late-train (last 10 %) | 5.535 | 5.452 | 5.548 | 5.512 |
| held-out nats/tok | 5.767 | 5.723 | 5.749 | 5.746 |
| held-out BPB | 2.054 | 2.038 | 2.047 | **2.047** |
| round-trip Δ | 8.2e-5 | 3.0e-5 | 0.0 | ✅ all |

The single-seed `final` is noisy (seed 2's last step spiked to 6.40); the
late-train mean (5.51) is the stable read.

### 2.2 `typed-gdn2-lm` — E99 typed Emender (PRIMARY candidate, matched anchor)

Stable on all 3 seeds. 1.2775 B params, fp32, ~2 370 tok/s ⇒ ~2.1 M tokens /
15-min window (3.2× fewer than the bf16 control), peak 44.2 GB.

| metric | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| AvgLoss (nats/tok) | 7.938 | 7.923 | 7.965 | **7.942** |
| Final loss | 7.353 | 6.970 | 6.785 | 7.036 |
| late-train (last 10 %) | 7.168 | 7.118 | 7.168 | 7.151 |
| held-out nats/tok | 7.350 | 7.271 | 7.292 | 7.304 |
| held-out BPB | 2.618 | 2.590 | 2.597 | **2.602** |
| round-trip Δ | 5.7e-6 | 7.2e-6 | 9.1e-6 | ✅ all |

Clean, deterministic (fp32 round-trip Δ ≤ 9e-6), NaN-free — but the **highest
loss of the three arms**, and the slowest in throughput.

### 2.3 `e98-cma-lm` — E98-CMA unified Emender (CONTROL) — **NaN-diverged 3/3**

`HARD_STOP_NaN` on all 3 seeds before the 15-min window ended. The hard-stop fired
as designed; the run was terminated and logged rather than overrunning. 1.2776 B
params, fp32, ~2 268 tok/s, peak 34.8 GB.

| seed | NaN at step | tokens before NaN | last finite loss | **min loss reached** |
|---|---:|---:|---:|---:|
| 0 | 524 | 1.07 M | 7.351 @ 473 s | **5.822** |
| 1 | 563 | 1.15 M | 7.124 @ 524 s | **5.912** |
| 2 | 833 | 1.71 M | 8.172 @ 766 s | **5.560** |

**Reading:** E98-CMA was *learning well* — its min train-loss (5.56–5.91) is
**competitive with the dense GDN-2 control's late-train (5.51)** and **far below
anything the E99 typed candidate reached** — before a loss spike to NaN. This is a
**stability failure, not a capability ceiling.** The cause is the untuned
**5.38× knob-LR** (= 5.27e-3) carried from the 8 M `cma-capability` winner: the
sanity report already saw the early-transient symptom over 80 steps and flagged it
("the discovered 8M LR is untuned at 1.3B and is exactly what the downstream CMA
should re-tune"); run longer here, the same instability diverges. Re-tuning the
knob-LR is *exactly* what the E99 LM-CMA search is for — we did **not** invent a
new config to "fix" it (that would be scope creep), we recorded the failure under
the handoff fragility taxonomy.

---

## 3. Token-matched cross-walk (throughput-neutral architecture comparison)

Wallclock-matched (handoff convention) favors the bf16 dense control, which sees
3.2× more tokens in the same 15 min (`budget_caps.json` COMPARABILITY_WARNING).
For a throughput-neutral read, each arm's loss-vs-tokens curve is sampled at the
**common smallest token budget** — the E99 typed budget, **≈2.106 M tokens**:

| arm | train-loss @ ≈2.106 M tokens (per seed) | mean |
|---|---|---:|
| `fla-gdn` (dense GDN-2) | 6.048 / 6.129 / 5.768 | **5.982** |
| `typed-gdn2-lm` (E99) | 6.835 / 7.020 / 7.083 | **6.979** |
| `e98-cma-lm` (E98-CMA) | already NaN-diverged by ~1.1–1.7 M tok (min 5.56–5.91 *before* divergence) | — |

**Even token-matched, dense GDN-2 beats the E99 typed candidate by ≈1.0 nat/token.**
The E99 candidate's deficit is therefore *not* merely the fp32 throughput penalty —
it learns less per token at this operating point too. E98-CMA's pre-divergence min
(5.56–5.91) also sits below the typed candidate, reinforcing that **typed-gdn2-lm
is the weakest learner of the three at these settings.**

---

## 4. Comparability / conversion notes (REQUIRED — vs the E97/GDN-2 CMA-ES batch)

Read against `docs/HANDOFF_E97_GDN2_CMAES_20260528.md`. The `wire-e99-e98` sanity
established these mappings; this task **uses** them. Mismatches that cannot be
reused verbatim at 1.3B LM-CMA are labelled below.

1. **Candidate budget / counting — PRESERVED.** These are bounded **controls**, not
   a search: each arm = 3 seeds × one 15-min eval, NOT a CMA generation. The 96 =
   popsize-8 × 12-gen counting applies to the `run-e99-1-3b` *search*, not to these
   controls. (`budget_caps.json` counting_convention.)
2. **Run IDs / output dir — MAPPED, with a documented mismatch.** Prior scheme
   `benchmark_results/cmaes_1270M_ctx2k_<tag>_<YYYYMMDD>/<model>` is a *CMA-search*
   layout driven by `scripts/cmaes_search_v2.py`. These controls use the
   **production** `train.py`/`ladder_lm.py` path, so results live under
   `experiments/e99_1p3b_controls/results/<arm>_s<seed>_result.json` (the layout
   the sanity established). **Mismatch flag:** driver = production `train.py` (not
   `cmaes_search_v2.py`); same as the sanity's flag, carried here.
3. **Model/config field names — PRESERVED.** `dim, n_heads, n_state, depth, lr,
   batch_size` (+ `expansion`) used verbatim in every record; production-path
   fields (`level, dtype, knob_lr_mult`) are additive.
4. **Training budget — MAPPED with conversion.** Primary unit = `train_minutes 15`
   (handoff). Token count emitted per run (`total_tokens`) so the wallclock↔token
   cross-walk in §3 is computable. **Conversion note (kept):** wallclock-matched ≠
   token-matched because the fp32 arms run ~3.2× slower than the bf16 control.
5. **Dataset / split + tokenizer — PRESERVED.** `/home/erikg/elman/data/pile.txt`,
   `p50k_base`, ctx 2048 — identical to the handoff. No change.
6. **Fitness — PRESERVED.** **AvgLoss / Final** (nats/token) reported as primary
   (§2), lining up with the handoff `Best avg loss | Best final loss` columns.
7. **Held-out BPB / loss — CONVERSION NOTE.** The handoff reported **train**
   AvgLoss/Final (nats/token), NOT held-out BPB. Here held-out loss is on a slice
   with **disjoint seed** (7777, vs train 42 / round-trip 1234). BPB =
   (nats/token)/ln2/(bytes/token), with **bytes/token MEASURED on that exact
   held-out slice** by decoding the target tokens back to UTF-8 (p50k_base ⇒
   **4.05 B/tok** here, consistent with the ~3.8–3.9 B/tok p50k_base figures in
   `E88_HELDOUT_HARNESS.md` / `BPB_CONTEXT.md`). **Do NOT** compare these held-out
   BPB numbers to the handoff *train-window* nats/token without (i) the nats→bits
   factor and (ii) the train-vs-held-out caveat. They are a labelled extension.
8. **Throughput — RECORDED.** tok/s + walltime + GPU-days/10B-tok per run (§2 /
   raw JSON), so both conventions cross-walk.
9. **Failure accounting — REUSED taxonomy.** Handoff: fragility/instability under
   schedule-free AdamW. Applied verbatim: **`fla-gdn` and `typed-gdn2-lm` =
   0 NaN / stable 3/3; `e98-cma-lm` = NaN-diverged 3/3 (knob-LR fragility),
   recorded as a hard-stop, not hidden.**
10. **Raw schema + summary columns — EXTENDED, not replaced.** `controls_summary.csv`
    emits the handoff `Target | Best avg loss | Best final loss | Best config`
    columns plus additive `params_B, dtype, tok/s, heldout_bpb, stable, provenance`.

---

## 5. Reused prior baselines (handoff provenance — NOT re-run)

Carried verbatim from `docs/HANDOFF_E97_GDN2_CMAES_20260528.md` "CMA-ES Status"
summary table (2026-05-28; best of popsize-8 CMA, 15-min window, 1.27B, ctx2k,
p50k_base). **No new training** — these are comparison rows only, labelled REUSED.

| Target (reused) | Best avg loss | Best final loss | Best config |
|---|---:|---:|---|
| Transformer | 5.9046 | 5.4683 | `dim=1664, n_heads=10, expansion=6, depth=19, lr=5.164e-4, batch_size=4` |
| E88 delta | 5.9974 | 5.5529 | `dim=2048, n_heads=348, n_state=32, depth=10, lr=9.973e-4, batch_size=2` |
| E88 raw-write | 6.0395 | 5.5909 | `dim=1792, n_heads=362, n_state=32, depth=11, lr=9.413e-4, batch_size=2` |
| Mamba2 | 6.0560 | 5.6441 | `dim=1920, d_state=64, expand=4, depth=27, lr=1.417e-3, batch_size=2` |
| M2RNN XMA | 6.0626 | 6.0626 | `dim=2304, n_heads=612, n_state=16, depth=10, lr=5.607e-4, batch_size=5` |
| FLA-GDN | 6.1104 | 5.6165 | `dim=3456, expansion=2, depth=12, n_heads=38, lr=8.627e-4, batch_size=2` |

**Cross-check (path validity):** our newly-run `fla-gdn` (dense GDN-2) AvgLoss
**6.227** / best-final **5.453** sits right next to the handoff FLA-GDN row
(AvgLoss 6.1104 / Final 5.6165). The small gap is expected — the handoff number is
the **best of a popsize-8 CMA search**, ours is a **single config at one operating
point** (and a different 1.3B shape) — so the wallclock-matched production path
**reproduces the prior batch within the single-config-vs-CMA-best gap.** This
validates that the controls are measured on the same yardstick as the handoff,
and that the reused rows are directly comparable.

---

## 6. Caveats (must travel with the TL;DR to `synthesize-e99-1-3b`)

1. **Single operating points, not CMA-tuned.** Each arm is **one** config (the
   discovered/seed shape), not the winner of a search. `run-e99-1-3b`'s 96-candidate
   LM-CMA may find a `typed-gdn2-lm` operating point that beats dense GDN-2; this
   control only shows the **un-re-tuned** point does not. Symmetrically, a re-tuned
   (lower) knob-LR would very likely make `e98-cma-lm` stable (its pre-divergence
   min-loss is already competitive).
2. **fp32 throughput handicap is real but possibly removable.** The typed/E98 arms
   run fp32 (~2.3–2.4k tok/s) vs bf16 dense (7.6k). bf16 is **not yet validated**
   for the typed/unified-cell kernels (sanity §3). If bf16 is made to work, the
   wallclock gap shrinks ~3×; the **token-matched** gap (§3, ≈1.0 nat) would remain
   and is the architecture-level signal.
3. **Different shapes.** "Matched as closely as possible" means matched **params**
   (~1.28–1.35 B) and identical data/budget, not identical dim/depth/heads — the
   dense GDN-2 control is the proven 1.352 B `THROUGHPUT.md` shape; the typed arm is
   the discovered 5:1 GDN2:nonlinear mixture shape.
4. **Short-run only.** 15-min/seed short runs probe early-training loss and
   stability, not converged quality. No multi-day run was launched (out of scope,
   human-gated).
5. **E98-CMA loss columns are absent by divergence**, not by omission; its
   pre-divergence min-loss (§2.3) is the fairest available capability proxy and is
   reported.

---

## 7. Validation checklist

- [x] Dense GDN-2 (`fla-gdn`) and E98-CMA (`e98-cma-lm`) controls run at matched
      1.3B-class LM budget (same 15-min/seed caps as E99, same data/tokenizer/ctx).
      E98-CMA's blocker (NaN divergence) is documented exactly, not papered over.
- [x] Aggregate budget ceiling recorded (5.25 GPU-h; **2.23 GPU-h** used);
      hard-stop fired with logged reason on the 3 e98 NaN divergences;
      pre-launch ceiling check + no-co-location guard active.
- [x] Prior baselines reused honestly with provenance (handoff table, §5); no
      invented/retrained baseline scope creep.
- [x] Loss (AvgLoss/Final/late-train), held-out BPB, throughput, and checkpoint
      round-trip consistency recorded with the same conventions as E99 (§1–2, §4).
- [x] Report states clearly whether E99 improves over dense GDN-2, E98-CMA, both,
      or neither: **over neither on loss; over E98-CMA only on stability** (§0).
- [x] `paper/review/E99_1P3B_LM_CONTROLS.md` written; raw JSON/CSV committed; no
      `paper/main.typ` edit; no push by the agent; no HF publish; no checkpoint
      published/staged (round-trip ckpts deleted); idle-GPU-only (GPUs 0–6,
      GPU 7 left to a sibling job).
- [x] `docs/HANDOFF_E97_GDN2_CMAES_20260528.md` read; reused rows cited with exact
      configs+losses; newly-run controls use the same field names / budget units /
      fitness / throughput conventions, with each driver/dtype mismatch documented
      (§4).

---

## 8. Artifacts & reproduction

- Driver: `experiments/e99_1p3b_controls/e99_lm_controls.py` (real data, real
  fwd/bwd, time-budgeted, held-out BPB, fresh-process round-trip, NaN/OOM/walltime
  hard-stops).
- Orchestrator: `experiments/e99_1p3b_controls/run_controls.py` (GPU pool, ceiling
  guard, no co-location, per-job outer timeout).
- Raw per-(arm,seed) results: `experiments/e99_1p3b_controls/results/<arm>_s<seed>_result.json`
  (includes the full loss-vs-tokens curve used for §3).
- Summary: `experiments/e99_1p3b_controls/results/controls_summary.json` +
  `controls_summary.csv` (extends the handoff columns).
- Orchestrator log + summary: `…/results/orchestrator.log`,
  `…/results/orchestrator_summary.json`.
- Reproduce one arm/seed:
  `CUDA_VISIBLE_DEVICES=<idle> python3 experiments/e99_1p3b_controls/e99_lm_controls.py --config fla-gdn --seed 0 --train_minutes 15`

Round-trip `.pt` checkpoints are intentionally **not committed/published** (task
constraint); they are regenerated by the driver and deleted after the round-trip.
