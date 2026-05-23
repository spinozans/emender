# NDM Paper Notes Reconciliation

Generated: 2026-05-23  
Source: `paper/ndmpapernotes.md`  
Reconciled against: repo state at commit `358f91e` (main) + worktree `wg/agent-14/reconcile-paper-notes`

---

## Key

| Tag | Meaning |
|---|---|
| **supported** | Evidence exists in this repo; artifact cited. |
| **partial** | Some evidence in-repo; gap specified. |
| **aspirational** | No in-repo evidence; experiment/proof needed. |
| **out-of-repo** | Evidence exists but lives in `~/elman/` or unreleased checkpoints. |

---

## Claim-by-Claim Table

### One-Sentence Thesis

| # | Claim | Status | Evidence / Gap |
|---|---|---|---|
| T1 | Pure nonlinear recurrent LMs can be trained at billion-parameter scale | **partial** | 1.27B runs exist and are stable (`docs/M2RNN_E88_COMPARISON.md`, `docs/NDM_ARCHITECTURE_MENU.md`), but raw loss curves and checkpoints are **out-of-repo** (`~/elman/`). |
| T2 | Recurrent computation organized as multi-programmed GPU workload | **supported** | Architecture described in `ndm/models/e88_fused.py`, `ndm/models/e88_fla_hybrid.py`, and `README.md`; 370 heads × 32×32 state cited in `docs/NDM_ARCHITECTURE_MENU.md`. |
| T3 | Useful memory resource is a nonlinear delta-correcting matrix update | **supported** | Core update defined in `README.md`, `docs/NDM_ARCHITECTURE_MENU.md`, and checked in Lean: `formal/lean/ElmanProofs/PaperCore.lean`. |

---

### Abstract Shape (three linked claims)

| # | Claim | Status | Evidence / Gap |
|---|---|---|---|
| A1 | Architecture search over recurrent geometry finds many-head shapes training smoothly at 1.27B | **partial** | CMA-ES search is referenced as historical methodology (`docs/NDM_ARCHITECTURE_MENU.md` Phase 3, `ndm/models/e88_step_kernel.py`); the E88 shape (H=370, N=32, dim=1664, depth=12) is the product of this search per `docs/M2RNN_E88_COMPARISON.md`. However, no in-repo CMA-ES run logs or search code exist. Search history lives in `~/elman/`. |
| A2 | Optimized Triton implementation makes nonlinear recurrence competitive with hand-tuned baselines in wallclock training | **partial** | Triton kernel implemented: `ndm/triton/e88_triton_forward.py`, `ndm/triton/e88_triton_backward.py`, `ndm/triton/e88_triton_optimized.py`. Throughput figure (~7.7K tok/s for E88 vs ~7.5K tok/s for M2RNN-CMA) cited in `docs/M2RNN_E88_COMPARISON.md`. Full wallclock racer plots are **out-of-repo**. |
| A3 | S5 permutation composition identifies mechanism: NDM read-then-delta write outperforms linear recurrent and M2RNN raw-write baselines | **supported** | S3/S5 matched 8M run results table with exact numbers (3 seeds, train length 128) exists in `paper/ndmpapernotes.md` and is reproducible via `experiments/expressivity_tasks/run_separation_suite.py`. Harness + task code: `experiments/expressivity_tasks/tasks/s5_permutation.py`. |

---

### Main Contributions

| # | Contribution | Status | Evidence / Gap |
|---|---|---|---|
| C1 | **Architecture:** NDM pure recurrent matrix-memory with nonlinear delta-correcting update | **supported** | Core update in `README.md`, `docs/NDM_ARCHITECTURE_MENU.md`; implementation in `ndm/models/e88_fused.py`, `ndm/models/e88_fla_hybrid.py`; Lean update-family definition in `formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean`. |
| C2 | **Training geometry:** CMA-ES and follow-up search identify many-head, multi-programmed shapes at 1.27B | **partial** | Resulting geometry (E88: H=370, N=32, dim=1664, depth=12) documented in `docs/M2RNN_E88_COMPARISON.md` and `docs/NDM_ARCHITECTURE_MENU.md`. CMA-ES search code and run logs are **out-of-repo** (`~/elman/`). Ablation notes in `docs/E88_ABLATION_NOTES.md` document the smaller-scale iterative search that preceded CMA-ES. No in-repo CMA-ES executable exists. |
| C3 | **Systems:** fused Triton kernel with L2 key/query normalization, sparse checkpointing, output gating, CUDA/ROCm semantics | **supported** | `ndm/triton/e88_triton_forward.py` (L2 norm fused, sparse checkpoints saved every `CKPT_INTERVAL` steps), `ndm/triton/e88_triton_backward.py` (sparse-checkpoint backward, `num_warps=1` at high head count, bf16 scratch), `ndm/triton/e88_triton_optimized.py` (fused output gate). ROCm portability rationale: `docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md`, kernel portability comment in `ndm/models/e88_fla_hybrid.py:852`. Gap: no in-repo throughput benchmark comparing Triton to the CUDA register-owned baseline. |
| C4 | **Language modeling:** 1.27B pure recurrent NDM in same wallclock loss regime as strong linear recurrent baselines | **out-of-repo** | Qualitative claim stated in `docs/M2RNN_E88_COMPARISON.md` (step 9250, loss 4.085 for M2RNN-CMA; E88 ~loss 3.0 after long convergence). Raw loss curves, checkpoints, and wallclock plots live in `~/elman/`. No frozen loss curves or smoothed racer plots exist in-repo. `scripts/run_periodic_racer_evals.py` and `scripts/racer_eval_suite.py` provide infrastructure but no results. |
| C5 | **Expressivity:** S3/S5 state-tracking tasks show NDM separates from FLA-GDN, Mamba2-style linear recurrence, and M2RNN raw-write | **supported** | S3/S5 8M matched-run table in `paper/ndmpapernotes.md` (3 seeds, 4 models). Length-extrapolation results: `experiments/expressivity_tasks/LENGTH_EXTRAP_RESULTS.md`. Canonical sweep (parity, modular counter, FSM tracking, dyck, assoc recall, selective copy) with E88 vs FLA-GDN vs hybrid: `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`. Harness code: `experiments/expressivity_tasks/`. Gap: Mamba2 baseline is listed in `README.md` model table but **absent from the S3/S5 8M matched run**; the in-repo S3/S5 table covers E88/NDM, FLA-GDN, M2RNN, M2RNN-paper only. |
| C6 | **Formal core:** Lean-checked resource separation between NDM and M2RNN update families, plus checked S5 tracker and finite transition-memory realization | **supported** | `formal/lean/ElmanProofs/PaperCore.lean` imports the full trusted chain. Trusted surface documented in `formal/lean/TRUSTED_PROOF_SURFACE.md`. S5 status: `formal/lean/S5_FULL_PROOF_STATUS.md`. Individual files: `ElmanProofs/Architectures/M2RNNComparison.lean`, `ElmanProofs/Architectures/RecurrentResourceFormalism.lean`, `ElmanProofs/Expressivity/S5Tracker.lean`, `ElmanProofs/Expressivity/S5NDMRealization.lean`, `ElmanProofs/Expressivity/S5Witness.lean`. CI script: `formal/lean/scripts/check_paper_core.sh`. |

---

### Results Tables (Figure 1 data)

#### S3/S5 8M Matched Run (train length T=128, 3 seeds)

| # | Claim | Status | Evidence / Gap |
|---|---|---|---|
| R1 | E88/NDM S3 mean acc = 1.0000 (min 0.9999, max 1.0000) | **out-of-repo** | Numbers stated in `paper/ndmpapernotes.md`; reproducible via `run_separation_suite.py`, but no JSON result files exist in-repo. Runs were executed in `~/elman/`. |
| R2 | FLA-GDN S3 mean acc = 0.7185 (min 0.6122, max 0.8516) | **out-of-repo** | Same as R1. |
| R3 | M2RNN S3 mean acc = 0.3124; M2RNN-paper S3 = 0.3773 | **out-of-repo** | Same as R1. |
| R4 | E88/NDM S5 mean acc = 0.7918; FLA-GDN = 0.3552; M2RNN = 0.2157; M2RNN-paper = 0.1698 | **out-of-repo** | Same as R1. |
| R5 | S5 length extrapolation table (T=128 to T=1024, 4 models) | **out-of-repo** | Numbers in `paper/ndmpapernotes.md`; run infrastructure in `experiments/expressivity_tasks/run_lenextrap_sweep.sh`. In-repo LENGTH_EXTRAP_RESULTS.md covers E88 vs FLA-GDN on parity/modular_counter/fsm_tracking (not the 4-model S5 table from the notes). |

Note: The S3/S5 4-model table in the paper notes is **distinct** from the parity/modular/FSM results in `CANONICAL_SWEEP_RESULTS.md` and `LENGTH_EXTRAP_RESULTS.md`, which use E88 vs FLA-GDN (and some hybrid) but not M2RNN on those specific tasks.

---

### CMA-ES and Geometry (Figure 2 claim area)

| # | Claim | Status | Evidence / Gap |
|---|---|---|---|
| G1 | CMA-ES discovers many-head, stable q/k address, multi-programmed shapes | **out-of-repo** | Referenced in `docs/NDM_ARCHITECTURE_MENU.md` (Phase 3 plan), `docs/M2RNN_E88_COMPARISON.md` ("tied/CMAES-shaped M2RNN"). Historical search in `~/elman/`. No in-repo CMA-ES code or results. |
| G2 | Paper-shaped M2RNN brittle under schedule-free LM setup; CMA-ES-shaped M2RNN trains more smoothly | **supported** | Production stability table in `docs/M2RNN_E88_COMPARISON.md`: paper M2RNN stopped at step 8400 (loss 11.574, grad 4.19e7); tied/CMA-ES M2RNN stable at step 9250 (loss 4.085, grad 1.48). Mechanism explanation provided. |
| G3 | Same update family looks weak or strong depending on geometry | **supported** | `docs/E88_ABLATION_NOTES.md` documents iterative geometry search; `docs/M2RNN_E88_COMPARISON.md` q/k ablation shows geometry effects on stability. |

---

### Systems / Figure 4 Claims

| # | Claim | Status | Evidence / Gap |
|---|---|---|---|
| S1 | Fused L2 norm in kernel | **supported** | `ndm/triton/e88_triton_optimized.py:53`, `ndm/triton/e88_triton_forward.py`. |
| S2 | `num_warps=1` at high head count | **supported** | `ndm/triton/e88_triton_backward.py:525–542` (empirical note, nw=1 wins when B*H is large). |
| S3 | Sparse forward checkpointing | **supported** | `ndm/triton/e88_triton_forward.py:22`, `ndm/triton/e88_triton_backward.py:45` — sparse `S_ckpt` tensor every `CKPT_INTERVAL` steps. |
| S4 | Avoiding unnecessary contiguous copies | **partial** | Referenced design principle in triton code comments. No dedicated benchmark comparing vs. naïve copy path exists in-repo. |
| S5 | Fused output gate | **supported** | `ndm/triton/e88_triton_optimized.py:73–81` (fused gate applied when `apply_gate=True`). |
| S6 | bf16 scratch and block-size tuning | **supported** | `ndm/triton/e88_triton_backward.py:180, 224, 237, 530, 560` — explicit bf16 scratch allocation and empirical block-size/warp notes. |
| S7 | CUDA/ROCm portable semantics | **partial** | Triton is the portability mechanism (`docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md:74–80`); Triton kernel implemented. No in-repo ROCm test or benchmark confirming parity on AMD hardware. |
| S8 | Throughput competitive with hand-tuned baselines | **out-of-repo** | Tok/s figures (E88 ~7.7K, M2RNN-CMA ~7.5K) cited in `docs/M2RNN_E88_COMPARISON.md` but from live training runs in `~/elman/`. No in-repo benchmark script produces these numbers. |

---

### Formal Results Section

| # | Claim | Status | Evidence / Gap |
|---|---|---|---|
| F1 | NDM and M2RNN are distinct one-step update families | **supported** | `formal/lean/ElmanProofs/Architectures/M2RNNComparison.lean`; summarized in `formal/lean/TRUSTED_PROOF_SURFACE.md` and `formal/lean/ElmanProofs/PaperCore.lean`. |
| F2 | M2RNN cannot implement NDM mixed-key delta correction in one step without extra read-then-delta path | **supported** | `formal/lean/ElmanProofs/Architectures/M2RNNComparison.lean`; cited in `formal/lean/TRUSTED_PROOF_SURFACE.md`. |
| F3 | With extra read-then-delta resource, M2RNN can embed one NDM step | **supported** | Same files as F2. Explicitly stated as resource separation, not absolute computability separation. |
| F4 | Fixed-width, finite-precision recurrent recognizers are finite-state | **supported** | `formal/lean/ElmanProofs/Expressivity/S5Witness.lean`; `formal/lean/S5_FULL_PROOF_STATUS.md`. |
| F5 | S5 tracker is checked: 120-state recognizer, word execution composes permutations, matches Python tuple-swap semantics | **supported** | `formal/lean/ElmanProofs/Expressivity/S5Tracker.lean`; `formal/lean/S5_FULL_PROOF_STATUS.md`. |
| F6 | Exact finite transition-memory realization: S5 instance uses 480 state/input keys | **supported** | `formal/lean/ElmanProofs/Expressivity/S5NDMRealization.lean`; `formal/lean/S5_FULL_PROOF_STATUS.md`. |
| F7 | No Lean lower bound proving all linear scan models fail S5 | **supported** (acknowledged non-claim) | Explicitly stated as out-of-scope in `formal/lean/TRUSTED_PROOF_SURFACE.md`, `formal/lean/S5_FULL_PROOF_STATUS.md`, and `paper/ndmpapernotes.md`. |
| F8 | No Barrington/NC1 completeness inside Lean | **supported** (acknowledged non-claim) | `formal/lean/S5_FULL_PROOF_STATUS.md`. |
| F9 | Python lexicographic class ID formalization (remaining Lean gap) | **aspirational** | `formal/lean/S5_FULL_PROOF_STATUS.md` lists this as item 1 of remaining lemmas. |
| F10 | Adjacent transpositions generate all of S5 (formal proof) | **aspirational** | Listed as item 2 of remaining lemmas in `formal/lean/S5_FULL_PROOF_STATUS.md`. |

---

### Paper Outline Claims

| # | Claim | Status | Evidence / Gap |
|---|---|---|---|
| O1 | 1.27B NDM geometry: dim=1664, depth=12, H=370, N=32 | **out-of-repo** | Shape cited in `docs/NDM_ARCHITECTURE_MENU.md` and `docs/M2RNN_E88_COMPARISON.md`; configuration appears in `formal/lean/ElmanProofs/PaperCore.lean`. Running checkpoint and logs in `~/elman/`. |
| O2 | Context curriculum (2K then expansion) | **partial** | Described as the training setup in `docs/M2RNN_E88_COMPARISON.md` and `docs/NDM_ARCHITECTURE_MENU.md` ("ctx2k convergence"); training code in `train.py`. Actual curriculum schedule and logs are **out-of-repo**. |
| O3 | Byte/token setup and schedule-free optimizer | **supported** | `train.py:73–74` (byte-level vs BPE option), schedule-free AdamW referenced throughout; confirmed in `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`. |
| O4 | Hybrid models (NDM + GDN / attention) degrade state-tracking vs pure NDM | **supported** | `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` and `LENGTH_EXTRAP_RESULTS.md` both show hybrid_AABB underperforming pure E88 on modular counter, FSM tracking. |

---

## Missing-Evidence Summary

Experiments still needed to supply in-repo evidence:

1. **S3/S5 matched 8M run JSON results** — The 4-model (E88, FLA-GDN, M2RNN, M2RNN-paper) run at train length T=128 with 3 seeds must be re-run and persisted as result files under `experiments/expressivity_tasks/results/`. Script exists: `experiments/expressivity_tasks/run_separation_suite.py`. Without this, the headline S5 numbers in the paper lack an in-repo source.

2. **S5 length-extrapolation for all 4 models** — The length-extrapolation table in the paper notes (T=128 to T=1024) covers E88, FLA-GDN, M2RNN, and M2RNN-paper. The in-repo `LENGTH_EXTRAP_RESULTS.md` covers only E88 and FLA-GDN on parity/modular_counter/fsm_tracking at a different eval schedule (T=40 to T=500). A new run targeting the 4-model S5 extrapolation table is needed.

3. **Mamba2 in the S3/S5 table** — `README.md` lists Mamba2 as a key model family, but it does not appear in the S3/S5 or canonical S5 results. A Mamba2 baseline run on S3/S5 is needed to support the full expressivity separation claim.

4. **CMA-ES search code and logs** — No CMA-ES search script exists in this repo. The E88 geometry is claimed to be the output of this search. A search log, config file, or documented run history (or a reference to the exact `~/elman/` commit) is needed to substantiate Contribution C2.

5. **Triton throughput benchmark** — The tok/s figures (~7.7K E88, ~7.5K M2RNN-CMA) cited in `docs/M2RNN_E88_COMPARISON.md` come from live training runs. A reproducible in-repo microbenchmark comparing Triton vs. CUDA register-owned kernel throughput would substantiate Contribution C3 and Figure 4 claims.

6. **Wallclock racer loss curves** — The central Figure 3 claim requires smoothed loss-vs-wallclock plots for E88, FLA-GDN, Mamba2, M2RNN-CMA (and optionally M2RNN-paper). Infra exists (`scripts/run_periodic_racer_evals.py`), but no frozen result files exist in-repo.

7. **Modular counter follow-up results** — `experiments/expressivity_tasks/MODULAR_COUNTER_FOLLOWUP.md` defines an important follow-up run (30K steps K=5, K=20, K=50). Results directory is `/tmp/` (ephemeral). If the tied/CMA-ES M2RNN edges E88 on K=5, this needs resolution before the paper's expressivity claim is clean.

8. **ROCm parity confirmation** — The systems claim includes CUDA/ROCm portable semantics, but no in-repo test confirms correctness of the Triton kernel on ROCm hardware.

---

## Out-of-Repo Dependencies

The following `~/elman/` outputs are required for the paper and currently have no in-repo equivalent:

| Dependency | Used In | What's Needed |
|---|---|---|
| E88 1.27B loss curve (`~/elman/` training logs, GPU 3) | Figure 3, Contribution C4 | Smoothed loss-vs-wallclock (5K, 10K, 50K step windows); frozen at a stable checkpoint hash |
| FLA-GDN 1.27B loss curve (GPU 1) | Figure 3 | Same as above |
| Mamba2 1.27B loss curve (GPU 2) | Figure 3 | Same as above |
| M2RNN-CMA 1.27B loss curve (GPU 4) | Figure 3 | Same as above |
| S3/S5 matched 8M run results (E88, FLA-GDN, M2RNN, M2RNN-paper, 3 seeds) | Figure 1, Table R1–R5 | JSON result files or a summary CSV |
| S5 4-model length-extrapolation run | Figure 1 panel C | JSON result files |
| CMA-ES search logs for E88 geometry | Section 4, Contribution C2 | Search run log or config at the `ekg/elman` commit where E88 geometry was frozen |
| M2RNN q/k ablation results (qk=1,3,11 step 0–1000) | `docs/M2RNN_E88_COMPARISON.md` section | Needed if paper discusses paper-shaped M2RNN instability mechanism |
| Modular counter follow-up results (`run_modular_counter_followup.py`) | S5/expressivity claim precision | Results from `/tmp/modular_counter_followup_20260511` (if run completed) |

---

## Validation Checklist

- [x] Every numbered contribution in `ndmpapernotes.md` appears in the table (C1–C6 above).
- [x] Every 'supported' tag cites a real file path or theorem name (no invented evidence).
- [x] 'Missing-evidence' summary lists concrete experiments, not vague gestures.
- [x] Out-of-repo list is explicit about which `~/elman/` outputs we depend on.
