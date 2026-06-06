# E99 1.3B LM — synthesis & full-run candidate decision (synthesize-e99-1-3b)

**Date:** 2026-06-06 · **Task:** `synthesize-e99-1-3b`. Synthesize the E99 1.3B
LM-CMA96 top-up (`run-e99-1-3b`) and the matched controls (`run-matched-1-3b`)
against the prior E97/GDN-2 CMA-ES batch, and recommend **one** full-run candidate
(or the smallest next pilot) for a later **human-approved** multi-day 1.3B run.

> **This document recommends; it does NOT launch.** No long/full run was started.
> No `paper/main.typ` edit, no push, no HF publish, no checkpoint published/staged.
> The multi-day 1.3B run stays gated behind explicit human go.

---

## 0. Decision (one line)

**Scale candidate = dense native GDN-2 / GatedDeltaNet (`fla-gdn`, bf16).** On every
**real LM signal** that decides a scale run — held-out BPB, late-train loss,
token-matched loss, throughput, stability, checkpoint round-trip, and projected
cost — dense GDN-2 wins or ties and **dominates on cost (~3.2× cheaper)**. The E99
typed Emender (`typed-gdn2-lm`) is **not** chosen for the scale run: the LM-CMA
search closed most of its per-token quality gap (held-out BPB 2.602 → **2.090**),
bringing it to **near-parity** with dense GDN-2 (2.047) — but it still trains
**~3.2× slower** in its only validated dtype (fp32; bf16 unvalidated for its
kernels), so its multi-day cost is ~3× higher for equal-or-slightly-worse LM
quality. **E98-CMA is not eligible** as a full-run candidate (NaN-diverged 3/3 at
its un-re-tuned operating point), though it remains a live control.

There **is** a clear winner, so §6 gives the exact full-run SPEC for dense GDN-2.
The single highest-value, **non-gating** follow-up — a bf16 validation pilot of the
searched typed candidate (§7) — is the only result that could flip this before
multi-day compute is committed, and it is cheap; it is offered alongside the SPEC,
not in place of it.

---

## 1. Sources synthesized (all upstream reports + raw summaries)

| source | task | what it provides |
|---|---|---|
| `paper/review/E99_1P3B_LM_SANITY.md` | `wire-e99-e98` | wiring + round-trip gate; dtype paths; throughput/GPU-day projections; budget caps |
| `paper/review/E99_1P3B_LM_CMA96.md` + `experiments/e99_1p3b_cma/artifacts/{results.json,candidates.csv}` + `pilot_results/p{0,1,2}/pilot_results.json` | `run-e99-1-3b` | 96-eval LM-CMA on `typed-gdn2-lm`; top band; **token-matched pilot BPB/Final + round-trip** |
| `paper/review/E99_1P3B_LM_CONTROLS.md` + `experiments/e99_1p3b_controls/results/controls_summary.csv` | `run-matched-1-3b` | matched single-point controls: dense GDN-2, typed, E98-CMA; token-matched cross-walk |
| `docs/HANDOFF_E97_GDN2_CMAES_20260528.md` | prior batch | the E97/GDN-2/E88/Mamba2/Transformer/M2RNN reference table (§4) |
| `paper/review/TYPED_GDN2_MIXTURE_CMA_RESULTS.md`, `CMA_CAPABILITY_RESULTS.md` | `typed-gdn-2-head`, `cma-capability` | **synthetic** capability evidence (recall/S5/count/latch/nonlin) — risk context, §5; **not** the scale decision |

All numbers below are quoted from these artifacts with provenance. Synthetic
capability evidence is **separated** from LM evidence (§5) per the task: LM
BPB/throughput decides the scale candidate; synthetic probes only explain risk.

---

## 2. Real-LM head-to-head (the decision evidence)

All arms: production `train.py` + `ndm/models/ladder_lm.py`/FLA path, real Pile
(`pile.txt`, `p50k_base`, ctx 2048), schedule-free AdamW, 1.3B-class, RTX 6000 Ada,
checkpoint round-trip gate. "Held-out BPB" = canonical Pile held-out byte slice.

| arm (role) | dtype | params | stable | held-out **BPB** ↓ | late-train / **Final** loss ↓ | token-matched loss @≈2.1 M tok ↓ | tok/s ↑ | round-trip |
|---|---|---:|:--:|---:|---:|---:|---:|:--:|
| **`fla-gdn` — dense GDN-2 (CONTROL)** | bf16 | 1.352 B | ✅ 3/3 | **2.047** | 5.51 late / 5.45 best-Final | **5.982** | **7 638** | ✅ |
| `typed-gdn2-lm` — E99, **un-tuned anchor** (control) | fp32 | 1.278 B | ✅ 3/3 | 2.602 | 7.15 late / 7.04 Final | 6.979 | 2 370 | ✅ |
| `typed-gdn2-lm` — E99, **CMA-searched eval 87** (pilot, token-matched) | fp32 | 1.166 B | ✅ | **2.090** | 5.792 Final (1796 steps / 7.36 M tok) | ≈6.0 (read at ~2.1 M) | 2 686 | ✅ Δ=0.0 |
| `e98-cma-lm` — E98-CMA (control) | fp32 | 1.278 B | ❌ **NaN 3/3** | — (diverged) | min 5.56–5.91 **before** NaN | — | 2 268 | ❌ (NaN) |

Provenance: dense GDN-2 / un-tuned typed / E98-CMA rows = `E99_1P3B_LM_CONTROLS.md`
§2 + `controls_summary.csv`; searched-typed row = `E99_1P3B_LM_CMA96.md` §6 +
`pilot_results/p1/pilot_results.json` (eval 87: avg 6.51576, Final 5.79239,
heldout_bpb **2.08987**, round-trip Δ 0.0). Token-matched @≈2.106 M tok read off
loss curves = `E99_1P3B_LM_CONTROLS.md` §3.

### What the LM evidence says, axis by axis

1. **Held-out BPB (primary held-out quality signal).** Dense GDN-2 **2.047** is
   best measured. The CMA search moved the typed candidate from **2.602** (un-tuned
   anchor) to **2.090** (eval 87) — a large, real improvement that lands it
   **within ~0.04 BPB of dense GDN-2**, at a comparable token budget (dense's 2.047
   at ~6.5 M tok in 15 min; typed eval 87's 2.090 at 7.36 M tok). Per token, the two
   are now **near-parity, dense still marginally ahead**.
2. **Late-train / Final loss.** Dense GDN-2 late-train 5.51 (best-Final 5.45). The
   searched typed band's token-matched Final is **5.65–5.79** (eval 95 5.652, eval 65
   5.767, eval 87 5.792). Dense is lower; typed sits just above it, in the prior-batch
   pack (§4).
3. **Token-matched loss (throughput-neutral).** At the common ≈2.106 M-token budget,
   dense GDN-2 **5.982** vs un-tuned typed **6.979** (`CONTROLS.md` §3). The CMA search
   narrows this at the *searched* operating point, but dense remains ahead per token.
4. **Throughput / cost (decisive).** Dense GDN-2 **7 638 tok/s bf16** (sanity 8 364)
   vs typed **2 370–2 690 tok/s fp32**. bf16 is the **proven** dense path
   (`THROUGHPUT.md`; sanity cross-check 8 364 ≈ 8 248). bf16 is **NOT yet validated**
   for the typed/unified-cell kernels (sanity §3). So at equal LM quality the typed
   candidate costs ~3.2–3.5× more wallclock/GPU-days (§3 below).
5. **Stability.** Dense GDN-2 and typed are NaN-free 3/3. E98-CMA NaN-diverges 3/3
   (un-tuned 5.38× knob-LR; the sanity report predicted this transient). The E99
   LM-CMA searched the **typed** candidate, not E98-CMA, so E98-CMA has **no**
   re-tuned, stable LM operating point on record — it cannot be the scale candidate now.
6. **Checkpoint round-trip (hard gate).** Dense GDN-2 ✅ (Δ ≤ 8.2e-5), typed ✅ (Δ ≤
   9e-6, pilot Δ ≤ 4.3e-6, 0/0 keys), E98-CMA ✅ only pre-divergence. The
   `PILE_BPB_MEASURED` forward-mismatch failure mode is absent on all wired arms.

**Net:** dense GDN-2 is the strongest **stable** arm on every LM axis and is ~3×
cheaper. The E99 search materially improved the typed candidate (un-tuned → near
per-token parity) but did not overturn dense GDN-2's throughput/cost dominance.

---

## 3. Projected full-run cost (real LM throughput, not synthetic)

From sanity §5 (GPU-days/10B tok, 1× RTX 6000 Ada) and controls tok/s:

| candidate | tok/s | GPU-days / 10B tok (1 GPU) | wall-days / 10B tok on 8 GPUs (DDP, ideal) |
|---|---:|---:|---:|
| **dense GDN-2 (bf16)** | 7 638–8 364 | **13.8** | **~1.7** |
| typed-gdn2-lm (fp32, only validated dtype) | 2 370–2 690 | 47.8 | ~6.0 |
| typed-gdn2-lm (bf16, **hypothetical, unvalidated**) | ~7–8 k (if it matched dense) | ~14 (if) | ~1.7 (if) |

The CMA search also moved the typed optimum to a **smaller, faster** shape (1.166 B,
~2 690 tok/s) than the anchor (1.278 B, 2 370 tok/s), but fp32 still leaves it ~3×
behind dense GDN-2. **The entire cost case for choosing typed over dense hinges on
bf16 validation** (§7), which is not yet done.

---

## 4. Comparability / conversion notes vs `HANDOFF_E97_GDN2_CMAES_20260528.md` (REQUIRED)

The prior 2K-context 1.27B CMA-ES batch (15 train-min, bf16, p50k_base,
schedule-free AdamW). E99 results are lined up against its summary columns
(`Target | Best avg loss | Best final loss | Best config`), **extended** with E99's
held-out BPB and throughput.

| Target | Best avg loss | Best final loss | held-out BPB | tok/s | Best config | provenance |
|---|---:|---:|---:|---:|---|---|
| Transformer | 5.9046 | **5.4683** | — | — | dim1664, nh10, exp6, depth19, lr5.164e-4, bs4 | handoff (REUSED) |
| E88 delta | 5.9974 | 5.5529 | — | — | dim2048, nh348, ns32, depth10, lr9.973e-4, bs2 | handoff (REUSED) |
| E88 raw-write | 6.0395 | 5.5909 | — | — | dim1792, nh362, ns32, depth11, lr9.413e-4, bs2 | handoff (REUSED) |
| Mamba2 | 6.0560 | 5.6441 | — | — | dim1920, d_state64, expand4, depth27, lr1.417e-3, bs2 | handoff (REUSED) |
| FLA-GDN | 6.1104 | 5.6165 | — | — | dim3456, exp2, depth12, nh38, lr8.627e-4, bs2 | handoff (REUSED) |
| M2RNN XMA | 6.0626 | 6.0626 | — | — | dim2304, nh612, ns16, depth10, lr5.607e-4, bs5 | handoff (REUSED) |
| **dense GDN-2 (`fla-gdn`, NEW)** | **6.227** | **5.45 best / 5.51 late** | **2.047** | 7 638 | dim2688, nh44, ns64, depth21, lr8.63e-4, bs2 (1.352 B, bf16) | `run-matched-1-3b` (NEWLY-RUN) |
| **E99 typed `typed-gdn2-lm` (NEW, CMA-best eval 87, token-matched pilot)** | 6.516 (pilot) / 7.51 (15-min fp32) | **5.79 (token-matched)** / 7.51 (15-min fp32) | **2.090** | 2 686 | dim3328, nh102, ns32, depth17, lr1.471e-3, bs2 (1.166 B, fp32) | `run-e99-1-3b` (NEWLY-RUN) |
| E98-CMA (`e98-cma-lm`, NEW) | — (NaN 3/3) | min 5.56–5.91 pre-NaN | — | 2 268 | dim3072, nh192, ns16, depth17, lr9.79e-4, knob-LR5.38, bs2 (1.278 B, fp32) | `run-matched-1-3b` (NEWLY-RUN) |

### How E99 lines up against the prior E97/GDN-2/E88/Mamba2 results — explicitly

- **The newly-run dense GDN-2 reproduces the prior batch.** AvgLoss 6.227 ≈ handoff
  FLA-GDN 6.1104; best-Final 5.45 ≈ handoff FLA-GDN 5.6165 (gap = single-config vs
  popsize-8-CMA-best, expected). This **validates the E99 yardstick is the prior
  batch's yardstick** — the comparison is real, not apples-to-oranges.
- **E99 typed, token-matched, reaches the prior pack but does not beat its best
  arms.** The pilot token-matched Final **5.65–5.79** sits **above** Transformer
  (5.4683), E88-delta (5.5529), E88-raw-write (5.5909), FLA-GDN (5.6165), and
  Mamba2 (5.6441) — i.e. in the **same band as the GDN-2/Mamba2 tier, slightly
  worse than the strongest prior arms**, not beating them. So E99 typed **does not
  beat the prior E97/GDN-2/E88/Mamba2 numbers**; it matches the linear-attention
  tier token-matched.
- **The raw 15-min fp32 7.51 must NOT be read against the prior table.** It is the
  fp32 token deficit (~580 steps / ~2.4 M tok vs the bf16 arms' ~1800 steps / ~6.5 M
  tok in the same 15 min), **not** an architectural deficit. The token-matched pilot
  (§6.4 of CMA96) is the comparable number.
- **Bottom line vs prior batch:** E99 produced **no arm that beats the best prior
  result**. Its strongest stable arm (dense GDN-2) **equals the prior FLA-GDN**; its
  novel arm (typed) **matches the GDN-2/Mamba2 tier token-matched** and is the only
  arm carrying extra synthetic capability (§5) at a throughput cost.

### Comparability / conversion notes (mismatches that cannot be reused verbatim)

1. **dtype / token deficit (dominant caveat).** Prior arms = bf16; typed & E98-CMA =
   fp32 (their only sanity-validated dtype). fp32 1.3B ≈ 2.3–2.7 k tok/s vs bf16 ≈
   7.6–8.4 k, so a wallclock-matched 15-min window gives the fp32 arms ~3× fewer
   tokens. **Conversion:** use the token-matched pilot (≈1800 steps ≈ bf16 15-min
   token budget) for any cross-arm loss comparison; both `train_minutes` and
   `tok/s`+`tokens` are recorded per candidate. The raw 7.51 vs ~6.0 gap is confounded
   and is excluded from architectural claims.
2. **train-window loss ↔ held-out BPB.** Prior reported **train-window AvgLoss/Final
   in nats/token**, NOT held-out BPB. E99 reports the same train AvgLoss/Final **and**
   adds held-out BPB for promoted/control configs. **Conversion:** `BPB =
   nats_per_token · (tokens/byte) / ln 2`; held-out `tokens/byte` measured on the exact
   slice (0.25942 for the CMA96 pilot slice; 4.05 B/tok ⇒ 0.247 tok/byte for the
   controls held-out slice — the two held-out slices differ, see note 3). Held-out BPB
   and train-window nats/token are **different distributions** and must not be equated.
3. **Two different held-out slices.** The CMA96 pilot BPB (eval 87 = 2.090) and the
   controls dense-GDN-2 BPB (2.047) are computed on **different held-out slices** with
   different measured `tokens/byte` (0.25942 vs 0.247). They are close enough to compare
   directionally (dense marginally better), but a single canonical held-out slice should
   be fixed for the full-run validation (§6) so the scale run's BPB is unambiguous.
4. **batch_size fixed = 2** (prior searched it). fp32 1.3B at ctx 2048 is memory-bound
   to bs≈2 on a 48 GB card. Constant, not a degenerate search axis. Dense GDN-2 (bf16)
   could afford larger bs; kept at 2 for matched comparison.
5. **n_state snapped to {16,32}** for typed (its head_dim support set); the task's
   "N=64 only if headroom" was not in-set and every top typed config chose 32. Dense
   GDN-2 uses n_state=64 (its native shape) — a genuine architectural difference, not a
   search choice.
6. **Driver.** E99 uses production `train.py`/`ladder_lm.py` (the CMA search wraps it)
   vs the prior `scripts/cmaes_search_v2.py`; same data/tokenizer/ctx/fitness/budget
   unit. Documented in all three upstream reports.
7. **Counting.** Prior `popsize 8 × min_gen 8 = 64`; E99 top-up `popsize 8 × 12 gen =
   96` candidate-**evaluations** (short runs), a documented fresh popsize-8 restart
   (typed-gdn2-lm did not exist in the prior batch, so no pickle to resume).

---

## 5. Synthetic capability evidence — SEPARATED (risk context, NOT the scale decision)

The task is explicit: synthetic probes may **explain risk**, but **LM BPB/throughput
decides the scale candidate.** The synthetic suite (`typed-gdn-2-head`, length-extrap
mean over T∈{128…1024}, `TYPED_GDN2_MIXTURE_CMA_RESULTS.md` §"head-to-head"):

| probe | typed-gdn2 | E98-CMA | DeltaNet ref |
|---|---:|---:|---:|
| recall (MQAR) | **0.807** | 0.423 | 0.742 |
| s5 / track | 0.969 | **0.999** | 0.050 |
| count | 0.891 | **0.947** | 0.934 |
| latch | 0.999 | 0.967 | **1.000** |
| nonlin | **0.944** | 0.931 | 0.853 |
| **mean** | **0.919** | 0.866 | 0.694 |
| **min (worst-case)** | **0.807** | 0.423 | 0.050 |

**What this contributes (risk only):**
- The typed mixture's value proposition is **real on synthetic**: it recovers native
  GDN recall (0.807, all-GDN arm 0.9875 MQAR) **and** keeps nonlinear/iterated-map
  capability (0.944) that linear-state GDN-2 genuinely **cannot** do — and it beats
  E98-CMA on worst-case (min 0.807 vs 0.423). This is why typed was a serious candidate.
- **But synthetic capability did not convert into an LM win.** At 1.3B Pile-LM,
  dense GDN-2's per-token quality matches/edges the searched typed candidate while
  running ~3× faster. The capabilities the typed mixture adds (long-range MQAR recall,
  iterated-nonlinear-map) are **not exercised by next-token Pile loss** at this budget,
  so they do not show up in BPB. They are a **risk/upside argument for a future
  capability-sensitive eval**, not a reason to pay 3× compute on the scale run now.
- E98-CMA's synthetic competence (mean 0.866) is real but its LM run **NaN-diverged**;
  synthetic strength does not rescue an unstable LM operating point.

**Conclusion:** synthetic evidence keeps the typed candidate alive as a *capability*
bet and justifies the bf16 validation pilot (§7), but on the LM evidence that governs
the scale decision, dense GDN-2 wins.

---

## 6. Full-run config SPEC — dense GDN-2 (`fla-gdn`, bf16) — FOR HUMAN REVIEW ONLY

> **NOT an instruction to launch.** This is the spec a human approves before any
> multi-day run. Token budget, exact node count, and go/no-go remain human decisions.

**Architecture.** Native GDN-2 / GatedDeltaNet via the production `--level fla-gdn`
(`ndm/models/ladder_lm.py` FLA `GatedDeltaNet`, `allow_neg_eigval=True`), the proven
`THROUGHPUT.md` 1.352 B backbone — the same native delta-memory kernel the typed
GDN-2 heads use, with no fp32/unvalidated-bf16 risk.

| field | value | source |
|---|---|---|
| level / dtype | `fla-gdn` / **bf16** (autocast, proven path) | sanity §3, controls §2.1 |
| dim | 2688 | controls config (RT-verified, freshest BPB) |
| depth | 21 | controls config |
| n_heads | 44 | controls config |
| n_state | 64 | controls config |
| expansion | 2.0 | controls config |
| params | ~1.352 B | measured |
| batch_size (per GPU) | 2 (bf16 headroom allows ↑; confirm in warm-up) | controls |
| optimizer | schedule-free AdamW (handoff/production convention) | handoff |
| lr | 8.63e-4 (the measured operating point) | controls / handoff FLA-GDN 8.627e-4 |
| data / tokenizer / ctx | `/home/erikg/elman/data/pile.txt` / `p50k_base` / 2048 | handoff (preserved) |

**Alternative shape to confirm (optional, cheap):** the handoff FLA-GDN CMA optimum
`dim3456, exp2, depth12, n_heads38, lr8.627e-4` (1.3B-class) reached Final 5.6165 in
the prior batch. The controls shape (dim2688/depth21/44h/ns64) is recommended as
primary because it has the **freshest end-to-end E99 evidence** (BPB 2.047, RT pass,
batch reproduction). If the team wants the lowest-loss shape, run a **short
LR×shape warm pilot** (2–4 configs × ~1 h) to pick between the two before committing
multi-day compute — see §7.

**Token budget.** Human-chosen. At 10B tokens the run is ~13.8 GPU-days (≈1.7 wall-days
on 8 RTX 6000 Ada, ideal DDP). Recommend fixing the budget and a fixed **canonical
held-out slice** (resolve the two-slice mismatch, §4 note 3) up front.

**DDP command SHAPE (template — do NOT copy-paste-launch; for human review):**

```bash
# Idle GPUs only; human sets --steps/token budget, run dir, and approves launch.
# 8-GPU DDP over the production train.py path.
torchrun --standalone --nproc_per_node=8 train.py \
  --level fla-gdn --bf16 \
  --dim 2688 --depth 21 --n_heads 44 --n_state 64 --expansion 2.0 \
  --batch_size 2 --lr 8.63e-4 --optimizer schedulefree \
  --data /home/erikg/elman/data/pile.txt --tokenizer p50k_base --chunk_size 2048 \
  --steps <HUMAN_SET_TOKEN_BUDGET/global_batch> \
  --out benchmark_results/fla_gdn_1352M_ctx2k_fullrun_<DATE>/ \
  --ckpt_every <N> --val_every <N>
```

**Validation plan (run-time).**
- Held-out BPB on the fixed canonical Pile held-out slice every `--val_every`; track
  vs the 2.047 short-run reference (must keep decreasing past it).
- Checkpoint round-trip (the hard gate) at the first and a mid checkpoint:
  save → reload in a fresh process → reproduce loss on a fixed held batch within
  1e-2 nats/tok, 0 missing/0 unexpected keys (reuse `e99_lm_sanity.py` harness).
- Throughput watch: sustained tok/s within ~10% of the 7 638–8 364 reference.

**Stop conditions.**
- NaN/Inf in loss → hard stop + log (handoff fragility taxonomy).
- Round-trip fails (Δ > 1e-2 or key mismatch) → hard stop (the `PILE_BPB_MEASURED`
  failure mode).
- Held-out BPB fails to improve over a human-set patience window → stop + review.
- Throughput collapses >2× below reference → stop + investigate.
- Aggregate GPU-day ceiling (human-set) reached → stop.

---

## 7. Smallest next discriminating pilot (non-gating; the one thing that could flip §6)

The decision case rests entirely on dense GDN-2's **throughput** advantage, because
per-token quality is now near-parity after the search. The single highest-value,
cheap experiment that could change the architecture choice **before** committing
multi-day compute:

**bf16 validation pilot of the searched typed candidate (eval 87/95 shape).**
- Validate the typed/unified-cell + FLA-GDN-2 kernels under bf16 (sanity §3 flagged
  bf16 as unvalidated for these). If bf16 works **and** preserves the fp32 per-token
  quality (BPB ~2.09), the typed candidate's ~3× cost handicap **disappears**, and its
  extra synthetic capability (§5) becomes free upside — which would flip the choice.
- Budget: a handful of short runs (≤1 h each), idle-GPU-only; round-trip-gated;
  report bf16 tok/s, bf16 held-out BPB, and NaN-freeness vs the fp32 baseline.
- This is a **follow-up, not a blocker**: dense GDN-2 is ready to scale today; the
  pilot only decides whether to substitute typed before the human approves the run.

**Optional, even cheaper:** the §6 LR×shape warm pilot to pick the dense GDN-2 shape
(controls dim2688 vs handoff dim3456). 2–4 configs × ~1 h.

Both are proposed as `wg` follow-ups below — neither gates the human go/no-go on the
dense GDN-2 SPEC.

---

## 8. Validation checklist (this task)

- [x] All upstream reports + raw summaries considered; numbers/provenance cited
      (§1, §2, §4 — sanity, CMA96 + raw pilot JSON, controls + CSV, handoff, synthetic).
- [x] Decision based on **real LM** BPB/loss/throughput/stability (§2–§3), not synthetic
      probes — synthetic explicitly separated as risk context only (§5).
- [x] Clear winner exists → exact full-run SPEC + DDP command shape + validation +
      stop conditions specified (§6); smallest next pilot also given as non-gating
      follow-up (§7).
- [x] Final multi-day 1.3B run **NOT launched**; explicit human go remains required
      (banner §0, §6).
- [x] `docs/HANDOFF_E97_GDN2_CMAES_20260528.md` read; E99 tabulated against its
      summary columns extended with BPB/throughput; explicit statement of how E99 lines
      up vs prior E97/GDN-2/E88/Mamba2 (§4); labeled comparability/conversion-notes
      section incl. train-loss↔held-out-BPB and wallclock↔token budget (§4).
- [x] `paper/review/E99_1P3B_LM_DECISION.md` committed; no `paper/main.typ` edit; no
      push by the agent; no HF publish; no checkpoint published/staged.

---

## 9. Recommendation summary

1. **Approve dense GDN-2 (`fla-gdn`, bf16, §6 SPEC) as the 1.3B scale candidate** —
   best stable LM quality (BPB 2.047), ~3× cheaper, reproduces the prior batch,
   round-trip-clean.
2. **Do not scale the E99 typed candidate yet**, despite its real synthetic-capability
   edge and near per-token parity after search — its only validated dtype (fp32) costs
   ~3× more for equal-or-slightly-worse LM quality.
3. **Run the bf16-typed validation pilot (§7) as the one cheap discriminator** that
   could flip (1)↔(2) before the human commits multi-day compute. Non-gating.
4. **E98-CMA is not a scale candidate** at its un-re-tuned operating point (NaN 3/3);
   keep it as a control. A future knob-LR re-tune pilot could revive it (its
   pre-divergence min-loss is competitive), but that is out of scope here.
5. The multi-day 1.3B run **stays human-gated**. This document is the decision input,
   not a launch.
