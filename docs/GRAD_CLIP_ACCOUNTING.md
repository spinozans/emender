# Forensic Accounting: Gradient Clipping — History, Necessity, and CMAES Confound

**Task:** `forensic-accounting-gradient` · **Date:** 2026-06-21 · **Method:** git history + real
log parsing (CPU only; no GPU leased; running racers untouched).

**Scope note (provenance honesty):** This repo's history begins at the boundary commit `cbcc726`
("Create focused NDM repository", 2026-05-23). Per project memory the durable upstream is `~/emender`,
so grad-clip provenance *before* `cbcc726` is not in this tree and is not claimed here. Every number
below is parsed from a committed artifact or a live race log on disk; sources and sample sizes are
cited inline, and data gaps are flagged rather than guessed.

Two distinct mechanisms share the word "clip" and must not be conflated:

| Mechanism | Flag | Where | What it does |
|---|---|---|---|
| **Global parameter-gradient clip** (this report's subject) | `--grad_clip` (default **1.0**) | `train.py:248`, applied `train.py:1537-1538` | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — renormalizes the whole-model grad vector to L2 ≤ 1.0 before `optimizer.step()` |
| **State-gradient clip** (kernel-internal, *separate*) | `--m2rnn_state_grad_clip` (default **None**) | `train.py:145`, passed to layer `train.py:928` | clips the recurrent-state gradient inside the cell; **off** in every racer/CMAES run except `m2rnn-paper` |

The `grad N.NN` column in every training log is the **pre-clip** total norm returned by
`clip_grad_norm_` (`train.py:1538`, logged at `train.py:1608`). So every log line directly records
clip *engagement*: `grad > grad_clip` ⇒ clipping fired that step.

---

## 1. When was grad clipping added — BEFORE or AFTER the 1.3B CMAES searches?

**Verdict: BEFORE. By ~18 days. Clipping has been present and unchanged since the repo's first commit.**

`git blame` puts both the flag and the call in the repo's **root commit**:

```
^cbcc726 (Erik Garrison 2026-05-23 13:39:10 +0000 248)  parser.add_argument('--grad_clip', type=float, default=1.0,
^cbcc726 (Erik Garrison 2026-05-23 13:39:10 +0000 1537)         if args.grad_clip > 0:
^cbcc726 (Erik Garrison 2026-05-23 13:39:10 +0000 1538)             grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

`git log -S"'--grad_clip'" -- train.py` and `git log -S"clip_grad_norm_(model.parameters(), args.grad_clip)" -- train.py`
**both return only `cbcc726`** — no later commit ever added, removed, or altered the clip flag or call.
The default has been `1.0` the entire time.

**Timeline (author dates, `git show -s --date=iso`):**

| Date (UTC) | Commit | Event |
|---|---|---|
| **2026-05-23 13:39** | `cbcc726` | **train.py created WITH `--grad_clip` default 1.0 + clip call** |
| 2026-06-10 20:58 | `1f7a65f` | opt-1p3b (1.3B) |
| 2026-06-11 00:13 | `afef090` | emender-real-cap |
| 2026-06-11 03:53 | `72607a9` | emender-real-1p3b (matched verdict) |
| 2026-06-11 09:20 | `d990b94` | emender-cap-sweep |
| 2026-06-11 19:58 | `2d4205d` | **emender-1p3b-cma** (search dir `emender_20260611_152412`) |
| 2026-06-12 13:00 | `24f4de7` | lb-emender-mix |
| 2026-06-13 10:14 | `37e9678` | **lb-compare** (matched verdict) |
| 2026-06-17 04:15 | `99b73f5` | cmaes-m2-1p3b |
| **2026-06-21 06:20** | `5555b9d` | skip-on-nonfinite (AFTER every search — see §4b) |

Clipping predates the earliest 1.3B CMAES search (`1f7a65f`, 2026-06-10) by ~18 days and **every**
architecture verdict by 18–25 days.

---

## 2. Was clipping ACTIVE during each CMAES search?

**Verdict: YES, active on every candidate, via train.py's default. No harness bypassed it.**

- **Default is 1.0** (`train.py:248`).
- **`cmaes_search_v2.py` shells out to `train.py`** (`scripts/cmaes_search_v2.py:870-885`) and builds the
  command line **without `--grad_clip`** — so every candidate inherits the default 1.0. The only
  clip-related arg the driver ever passes is `--m2rnn_state_grad_clip 1.0` at
  `scripts/cmaes_search_v2.py:1001`, and that is (a) the *state* clip, not the global one, and (b)
  emitted **only** for `model_type == 'm2rnn-paper'` — not for e97/emender/gdn2/m2rnn.
- **Confirmed from emitted configs, not assumption:** all **312** committed `args.json` files across
  every experiment record `"grad_clip": 1.0` — **zero** exceptions (`find … -name args.json` → 312×`1.0`).
  The matched-verdict runs confirm it directly, e.g.
  `experiments/lb_compare_20260613/runs/pure-E97/levelE97_100m_20260613_054305/args.json` → `"grad_clip": 1.0`.
- **The grok/expressivity harness clips unconditionally** and cannot disable it:
  `experiments/grok_expressivity/train_grok.py:304` → `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
  hardcoded inside the step loop.
- **The live racers also use the default:** the running emender process command line carries no
  `--grad_clip`, and its `args.json` records `"grad_clip": 1.0`, `"m2rnn_state_grad_clip": null`
  (`/mnt/nvme1n1/erikg/race/emender/runs/levelE97_100m_20260621_072520/args.json`); gdn2 likewise
  `"grad_clip": 1.0`.

---

## 3. What were the actual gradient norms DURING CMAES?

**Verdict: gradients massively and routinely exceeded the clip threshold on nearly every step of every
candidate — yet not one candidate ever hit a non-finite grad/loss. Clipping held the entire search finite.**

### 3a. Real per-candidate grad traces — emender-1p3b-cma (`2d4205d`)

This is the **one** 1.3B search that committed per-candidate `train.py` stdout
(`experiments/emender_1p3b_cma/search/emender_20260611_152412/eval_*/stdout.txt`, 64 candidates).
Parsing all **6,580** logged `grad` samples (1 per 50 steps):

| Statistic | Value |
|---|---|
| Pooled mean grad norm | **7.36** |
| Median | 1.84 |
| Stdev | 126.85 |
| Max | **8064.00** |
| Steps with grad > 1.0 (clip engaged) | **6195 / 6580 = 94.1%** |
| Steps with grad > 2.0 | 2919 / 6580 = 44.4% |
| Non-finite (inf/nan) grad/loss log lines | **0** |
| "Stopping before optimizer step" lines | **0** |

Per-candidate: **64/64** candidates had a *mean* grad > 1.0; 61/64 had mean > 2.0; per-candidate max
grad ranged from 10.31 to 8064 (median 67). The extreme is a real logged line:

```
experiments/emender_1p3b_cma/search/emender_20260611_152412/eval_5/stdout.txt:15:
step     30 | loss 8.8067 | lr 2.70e-03 | grad 8064.00 | tok/s 4446 | …
```

These norms dwarf the live race (§4a) because CMAES candidates are evaluated in the **cold-start /
schedule-free** regime (short minutes-long runs dominated by the early high-gradient phase), whereas
the racers are 300k+ steps warmed up.

### 3b. Did any candidate diverge and get de-selected? (Erik's hypothesis)

**Across all committed searches (758 candidates total), exactly one candidate hit the divergence/penalty
sentinel.** Scanning every `generations.jsonl` fitness entry:

| Search | N candidates | inf/penalty (≥1e5) | finite fitness range |
|---|---|---|---|
| emender-1p3b-cma `2d4205d` | 64 | **0** | 6.152 – 8.184 |
| cmaes-m2-1p3b | 96 | 0 | 6.184 – 8.879 |
| lb-emender-mix (`lb_emender_mlp`) | 128 | 0 | 5.861 – 7.006 |
| e99-1p3b-cma | 96 | 0 | 7.508 – 8.450 |
| e99-mixture-aware | 24 | **1** (`1000000.0`, gen0) | 6.322 – 7.096 |
| emender-cap-sweep | 120 | 0 | 2.371 – 2.468 |
| emender-real-cap | 20 | 0 | 2.373 – 2.487 |
| s5_symmetric ×4 (gdn/m2rnn/e88-lin/e88-tanh) | 210 | 0 | 0.91 – 0.98 |

The fitness function maps any crash/OOM/no-valid-steps to `float('inf')`
(`scripts/cmaes_search_v2.py:1281, 1559, 1573, 1582, 1765`) or a `1e6` penalty (`fits` schema), and the
loss regex `r'loss\s+([0-9.]+)'` (`:1273`) only reads finite step lines — so a diverged candidate is
*structurally* pushed to the bottom of the ranking.

**The mechanism that did the de-selecting was loss, not divergence.** The worst emender-1p3b-cma
candidate (eval_5: mean grad 173, max 8064) did **not** diverge — clipping kept it finite — so CMA-ES
saw it as `loss 8.81`, a bad-but-finite fitness, and selected against it on *loss*. **Erik's hypothesis
— "if out-of-bounds grads had broken things, CMAES would have de-selected them" — is therefore refuted
in mechanism: out-of-bounds grads were rampant (median 1.84, max 8064, clip engaged 94% of steps) but
clipping prevented them from breaking anything, so there was no divergence signal to select against.**
The search optimized loss entirely *inside* the clip=1.0 regime.

### 3c. Data gap (flagged, not guessed)

Only `emender-1p3b-cma` committed per-candidate stdout. cmaes-m2, lb-emender-mix, lb-compare, and
e99-1p3b-cma committed only `generations.jsonl` (fitness/params, no grad column) — their per-candidate
grad-norm traces were transient subprocess stdout and are **not recoverable** from committed artifacts.
Engagement %s for those searches cannot be measured; only the all-finite *fitness* record survives.

---

## 4. Is clipping NECESSARY, or is it masking instability?

**Verdict: BOTH. Clipping is load-bearing for finite explosions (real work, ~94% of CMAES steps and a
majority of race steps) AND it masks instability (enormous raw gradients never surface as divergence).
It is emphatically NOT inert. But it is NOT what saves the run from a true inf — that is skip-on-nonfinite.**

### 4a. Engagement (pre-clip norm > threshold), per arm — live race

Parsed from `/mnt/nvme1n1/erikg/race/{emender,gdn2}/run.log` (the live 1.3B race; `log_every=50`, so each
sample is 1 step in 50). The **emender racer is live-appending**, so its N is a parse-time snapshot
(grows ~1 sample/50 steps); the gdn2 log is static (that racer's segment has ended — no `train.py`
gdn2 process is alive, last line 2026-06-21T06:21):

| Arm | N samples | step range | mean | median | max | >1.0 | >2.0 |
|---|---|---|---|---|---|---|---|
| **emender (E97)** | 117 (snapshot) | 301050–306850 | **1.400** | 1.13 | **5.12** | **57.3%** | **11.1%** |
| **gdn2-mlp** | 536 | 325050–351800 | 1.177 | 1.15 | 2.45 | **92.0%** | 1.1% |

(These match the live-observation figures in the task framing almost exactly — emender ~1.43 mean /
~57% >1.0 / ~13% >2.0.) Clipping is engaged on a majority of steps for **both** arms — it is not an
emender-only event.

### 4b. The inf-grad event — what clipping CANNOT catch

The authoritative record is the commit that fixed it, **`5555b9d` (2026-06-21 06:20:30)**, verbatim:

> "A one-off bf16 inf overflow in the E97 tanh pre-activation was killing multi-day single-GPU runs via
> the stop-on-nonfinite guard. **Grad clipping cannot scale an inf** (max_norm/inf -> 0, inf*0 -> NaN),
> so the inf slips past clipping to the guard."

The diff replaces stop-and-exit with single-GPU **skip-and-continue** (`train.py:1542-1558`):
`-  print("Non-finite grad norm … Stopping before optimizer step.") / stopped_nonfinite=True; break`
→ `+ … SKIPPING this step (transient overflow), continuing / accumulated_steps=0; continue` (single-GPU),
keeping the stop-guard only for multi-rank (to avoid desyncing the DiLoCo/DDP collective merge).

**So what actually protects the run from the inf is `skip-on-nonfinite` (`5555b9d`, single-GPU) /
`stop-on-nonfinite` (multi-rank) — NOT clipping.** Clipping's domain is *finite* explosions: it scaled
the `grad 8064` events of §3a down to ≤1.0. It is mathematically incapable of taming an inf
(`1.0/inf = 0`, `inf*0 = NaN`). The two protections are disjoint:

| Failure mode | Caught by clip? | Caught by | Evidence |
|---|---|---|---|
| Finite explosion (grad 5–8064) | **Yes** (renormalized to ≤1.0) | clip_grad_norm_ | §3a, §4a |
| bf16 inf / NaN grad | **No** (cannot scale inf) | skip (1-GPU) / stop (multi-rank) | `5555b9d` msg + `train.py:1542-1558` |

*Honesty note on the literal "step 322087" line:* the racer relaunch script truncates `run.log` on each
restart (the current emender `run.log` begins at step 301050 after the 2026-06-21 07:18 watchdog
relaunch from the `checkpoint_step_301000` checkpoint). A repo/nvme-wide grep for `322087` and for
`Non-finite grad norm` finds the line **no longer on disk** — it was overwritten. The event itself is
durably documented by the commit (`5555b9d`) that diagnosed and fixed it; the specific step index is
the live observation that prompted the fix and is not independently re-derivable from a surviving log.

### 4c. Any committed run WITHOUT clipping to compare against?

**No.** All **312** committed `args.json` carry `grad_clip: 1.0`; a scan for `grad_clip ∈ {0, 0.0}`
returns nothing. There is **no committed clip-off A/B** anywhere in the tree — divergence-without-clipping
has never been measured here. This is the single biggest evidentiary gap and motivates the §6 control.

---

## 5. Does clipping CONFOUND the architecture verdicts?

**Verdict: There is a REAL, directionally-identified confound — clipping renormalizes the two arms'
gradients differently — but its magnitude is MODERATE where measurable, and its direction is
*conservative* for the published NO-GO verdicts (clipping flatters the spikier emender arm). Low risk
of having flipped any verdict; one cross-arm magnitude at search-time is a data gap.**

All matched comparisons ran at clip=1.0: lb-compare (`37e9678`; `args.json` grad_clip=1.0 confirmed
for every arm), e97-lm-1p3b token/wall verdict, emender-real-1p3b (`72607a9`). So the *setting* is
matched. The question is whether equal `grad_clip` produces equal *effective* step size. It does not,
because clip renormalizes by `min(1, 1.0/‖g‖)` and the arms have different `‖g‖` distributions.

**Measured cross-arm asymmetry (live race, §4a data):**

| Arm | mean clip-factor (all steps) | on clipped steps: mean kept-fraction | worst single-step kept-fraction |
|---|---|---|---|
| emender | **0.797** | **0.645** | **0.195** (grad 5.12 → keep 19.5%) |
| gdn2 | 0.862 | 0.850 | 0.408 (grad 2.45 → keep 41%) |

emender's update is compressed harder: ~20% of magnitude discarded on average vs ~14% for gdn2; on
clipped steps, ~36% discarded vs 15%; in the tail, emender spike-steps keep as little as 19.5% vs gdn2's
worst 41%. Because emender's spikes are renormalized away, its **effective LR on spike steps is silently
lower** than its nominal LR — so a "matched-LR" comparison is not a matched-effective-step comparison.

**Direction of the confound on the verdicts (the part that matters):**
- The verdicts are *emender NO-GO* (emender ties/loses gdn2). Clipping a spiky gradient is
  **stabilizing** — it suppresses the loss spikes large steps cause. So clip=1.0 **helps** the spikier
  emender arm reach a lower/more-stable loss than its raw gradients would. Turning clip **off** would, if
  anything, make emender *worse* (more spikes; recall the inf event is emender's own E97 tanh path, §4b),
  not better. The NO-GO direction is therefore robust to this confound — clip=1.0 is *generous* to emender.
- Each arm's LR was tuned **under** clipping (CMAES/sweeps found emender lr≈1.007e-3 vs gdn2 lr≈4.74e-4),
  so each tuned operating point already internalizes its own clip behavior — further shrinking the risk
  that the comparison is unfair *at the tuned points*.

**Magnitude, stated honestly:** moderate at the warmed-up race stage (mean clip-factor 0.797 vs 0.862,
~6 pts; tail gap larger). At 1.3B CMAES *cold-start*, emender grads were enormous (mean 7.36, §3a); whether gdn2's
cold-start grads were correspondingly smaller — which would make the *search-time* asymmetry far larger —
**cannot be measured**, because no gdn2 per-candidate grad traces were committed (§3c). That specific
cross-arm, search-time magnitude is **UNQUANTIFIED**. So: confound is real and named; verdict-flip risk
is low and, in residual direction, favors emender; one magnitude remains a data gap.

---

## 6. Recommendation

**Keep `grad_clip = 1.0` as the production default**, and **do not revise any published verdict on the
basis of this audit** — clipping is standard, load-bearing (§3a, §4a), and its confound direction is
conservative for the existing NO-GO conclusions (§5). Removing it outright is unsafe: the emender E97
path is the one with the bf16 inf (§4b) and the 8064-norm spikes (§3a).

**But the §4c gap (no clip-off data exists) and the §5 unquantified search-time asymmetry justify ONE
minimal control**, proposed as a follow-up (NOT run here, per task constraints):

> **Follow-up control — clip-sensitivity A/B (4 runs, matched tokens):**
> - **Grid:** {emender E97, gdn2-mlp} × {`--grad_clip 1.0`, `--grad_clip 0`} = 4 runs.
> - **Protocol:** fresh ~100M-token runs (the lb-compare budget) for a clean matched-token comparison,
>   OR resume each arm from its existing race checkpoint to skip cold-start. **Hold each arm's tuned LR
>   fixed** (emender 1.007e-3, gdn2 4.74e-4) so the *only* variable is clip. bf16 + fused (`--use_triton 1`,
>   fused-guard asserted) per NON-NEGOTIABLE #1. Single-GPU, `--no-wait` lease (do not disturb racers).
> - **Instrument:** the racer already logs pre-clip grad norm; additionally log non-finite skip events
>   (`train.py:1555`) and final held-out BPB on the shared disjoint slice (`--heldout_tensor`, the
>   lb-compare protocol).
> - **Metrics:** (i) non-finite/skip rate per arm under clip-off; (ii) ΔBPB(clip-off − clip-on) per arm;
>   (iii) whether the emender−gdn2 BPB gap changes magnitude or sign between clip-on and clip-off.
> - **Decision rule:** if `|Δgap| < ` the lb-compare BPB noise floor (~0.01–0.09 band,
>   `lb-compare-verdict`) → verdict robust, close the question. If a **stable** (non-diverged) clip-off
>   run narrows the gap toward emender by **more** than that floor → re-open the emender verdict.
> - **Also:** make the racers/CMAES retain per-candidate grad-norm percentiles (not just the 1-in-50
>   sample), so future engagement stats aren't sample-limited and every search keeps the traces that only
>   `emender-1p3b-cma` happened to commit (§3c).

A WG follow-up task has been filed for this control (see task `clip-sensitivity-control`).

---

## 7. clip-off control — MEASURED RESULTS (follow-up `clip-sensitivity-control`, 2026-06-21)

The §6 control was run: **12 REAL fused-kernel runs** = {emender-mlp E97, gdn2-mlp}
× {`--grad_clip 1.0`, `--grad_clip 0`} × seeds {42,43,44}, matched-token
(fixed `--steps 850`, both arms bs4×2048 ⇒ equal steps == equal tokens), each
arm at its tuned LR, byte-identical to lb-compare, same disjoint held-out slice.
**All 12/12 carry the `[fused-guard] … NO eager fallback` line; 0 eager paths.**

**Verdict: the emender NO-GO is ROBUST to grad clipping — clip on/off does not
move the emender−gdn2 BPB gap beyond the lb-compare noise floor.**

- **(i) Stability under clip-off:** **0 non-finite skips, 0 non-finite-loss
  stops, 12/12 ran the full 850 steps.** Clipping engages on 85–100 % of steps
  (very active) but removing it **diverges neither arm** at this budget;
  un-clipped pre-clip grad-norm max over all runs = **36.78** (seed-42 emender
  clip-off), all other clip-off maxima ≤ 9.73 — no inf, no `8064`-scale spike at
  the *tuned* config (those were CMAES *exploration* candidates, §3a).
- **(ii) ΔBPB(clip-off − clip-on), per arm (paired, non-avg):** emender
  **−0.023 ± 0.046** (straddles 0; the only large value, −0.075, is the seed-42
  outlier and does **not** replicate) · gdn2 **+0.013 ± 0.010** (clip-off
  consistently a hair *worse* for gdn2).
- **(iii) Δgap = gap_clipoff − gap_clipon (n=3):** non-avg **−0.036 ± 0.036**
  (per-seed −0.076, −0.009, −0.022); avg **−0.053 ± 0.051**. Both means are
  **inside** the 0.01–0.09 noise band ⇒ "narrows beyond floor ⇒ re-open" branch
  **not** triggered.
- **Calibration note:** the *single* seed-42 run gave Δgap −0.076 (non-avg) /
  **−0.111 (avg, which exceeds 0.09 and alone would have nominally tripped a
  re-open)** — an un-replicated transient. Multi-seed collapses it to within
  noise; a single-seed verdict would have been a **false re-open**.

**Effect on §5:** §5's *conclusion* (verdict robust, low flip-risk) is
**CONFIRMED**. §5's conjectured *direction* (clip "flatters the spikier emender
arm"; clip-off would make emender *worse*) is **empirically corrected**: at the
measured budget clip-off is **neutral for emender** and mildly *helps* gdn2, so
the residual shift is *toward* emender, not away — but within noise. §5's
worst-case (clip-off makes the emender E97 path diverge) **did not occur** in
6/6 emender clip-off runs at this budget. Closes the §4c "no clip-off run
exists" gap. Scope: short budget (≈7 M tokens, cold-start), tuned operating
points — does **not** reach the multi-day-racer inf regime (§4b), which remains
the domain of skip-on-nonfinite, not clipping.

Full write-up + raw artifacts: `experiments/clip_sensitivity_20260621/RESULTS.md`
(driver `run_clip_ab.py`, `multiseed_aggregate.py`, `clip_ab_*_analysis.json`,
`runs/*/train.log`).

---

## CLAIM → EVIDENCE table

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | `--grad_clip` default=1.0 + `clip_grad_norm_` call present since repo root commit | `git blame` `^cbcc726` `train.py:248,1537-1538`; `git log -S` returns only `cbcc726` | **VERIFIED** |
| 2 | Clipping was added BEFORE all 1.3B CMAES (by ~18 days) | cbcc726 2026-05-23 vs earliest search 1f7a65f 2026-06-10 (`git show -s --date=iso`) | **VERIFIED** |
| 3 | Default `--grad_clip` = 1.0 | `train.py:248` | **VERIFIED** |
| 4 | `cmaes_search_v2.py` shells `train.py` without `--grad_clip` ⇒ inherits default 1.0 | `scripts/cmaes_search_v2.py:870-885` | **VERIFIED** |
| 5 | The only clip arg the driver passes is `--m2rnn_state_grad_clip 1.0`, m2rnn-paper only, = state clip | `scripts/cmaes_search_v2.py:1001`; `train.py:145,928` | **VERIFIED** |
| 6 | All 312 committed `args.json` have grad_clip=1.0; no clip-off run exists | `find -name args.json` → 312×`1.0`; scan for 0/0.0 → empty | **VERIFIED** |
| 7 | grok harness clips unconditionally at 1.0 | `experiments/grok_expressivity/train_grok.py:304` | **VERIFIED** |
| 8 | Live racers use grad_clip=1.0, state-clip null | racer `args.json` + process cmdline (`/mnt/nvme1n1/erikg/race/emender/...`) | **VERIFIED** |
| 9 | emender-1p3b-cma: 64 candidates / 6580 grad samples, mean 7.36, median 1.84, max 8064, 94.1% >1.0, 44.4% >2.0 | parse `experiments/emender_1p3b_cma/search/emender_20260611_152412/eval_*/stdout.txt` | **VERIFIED** |
| 10 | max grad 8064 is a real logged line | `…/eval_5/stdout.txt:15` `grad 8064.00` | **VERIFIED** |
| 11 | Zero non-finite/stop events across all 64 emender-cma candidates | parse (0 `Non-finite`, 0 `Stopping`) | **VERIFIED** |
| 12 | 758 committed candidates total; exactly 1 penalty (e99-mixture-aware 1e6); de-selection was loss-based | unified scan of all `generations.jsonl` | **VERIFIED** |
| 13 | Per-candidate grad traces NOT committed for cmaes-m2 / lb-emender-mix / lb-compare / e99-cma | `grep -rl "| grad "` → 0 files in those dirs | **VERIFIED (gap)** |
| 14 | Live race engagement: emender mean 1.400 / 57.3% >1.0 / 11.1% >2.0 (N=117 snapshot, live log); gdn2 mean 1.177 / 92.0% >1.0 / 1.1% >2.0 (N=536, static) | parse `/mnt/nvme1n1/erikg/race/{emender,gdn2}/run.log` | **VERIFIED** |
| 15 | Clipping cannot scale an inf; skip-on-nonfinite (5555b9d) is what protects the run | `5555b9d` commit msg + `train.py:1542-1558` | **VERIFIED** |
| 16 | skip-on-nonfinite landed AFTER all CMAES/verdicts | `5555b9d` 2026-06-21 vs searches 2026-06-10..17 | **VERIFIED** |
| 17 | Cross-arm clip compression asymmetry (race): emender kept 0.645 on clipped steps (worst 0.195) vs gdn2 0.850 (worst 0.408) | computed from §4a grad samples | **VERIFIED** |
| 18 | All matched verdicts ran at clip=1.0 | lb_compare `args.json` grad_clip=1.0 (all arms); racer/CMAES default | **VERIFIED** |
| 19 | Confound direction is conservative ⇒ NO-GO robust | §5 reasoning; **§7 control: NO-GO robust CONFIRMED (Δgap within noise, 3 seeds), but "clip flatters emender" direction empirically corrected — clip-off is neutral-to-emender / mildly-helps-gdn2** | **VERIFIED (conclusion measured; mechanism corrected in §7)** |
| 20 | Clip on/off does not flip the emender verdict | **§7 control: 12 runs, clip-off A/B at tuned points, Δgap −0.036 (non-avg)/−0.053 (avg), within 0.01–0.09 floor; 0/12 divergence** | **MEASURED (§7) — verdict robust** |
| 20b | Search-time CMAES per-candidate gdn2 grad traces (cold-start magnitude) | transient subprocess stdout, not committed (§3c, #13); §7 measures the *tuned-point* clip effect, not raw CMAES cold-start | **UNSUPPORTED (data gap — traces unrecoverable; verdict question resolved by §7)** |
| 21 | Literal "step 322087" inf log line | overwritten on racer relaunch; not on disk; event documented only via `5555b9d` | **UNSUPPORTED as a standalone line (event VERIFIED via commit)** |
