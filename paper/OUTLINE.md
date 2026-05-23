# NDM Paper Outline

**Status:** Draft outline (not the paper itself).
**Generated:** 2026-05-23 by task `paper-outline`.
**Inputs read (cited inline below):**

- `docs/related_work_nonlinear_rnns.md` — literature survey (lit-survey-nonlinear-rnns)
- `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` — expressivity consolidation (consolidate-expressivity-results)
- `docs/DESIGN_DOSSIER.md` — architecture/systems synthesis (synthesize-design-dossier)
- `formal/lean/PROOF_INVENTORY.md` — trusted Lean surface (lean-proof-inventory)
- `paper/notes_reconciliation.md` — claim-by-claim evidence reconciliation (reconcile-paper-notes)
- `paper/ndmpapernotes.md` — original paper notes

---

## 0. Title and One-Sentence Thesis

**Working title (carried from `paper/ndmpapernotes.md` line 5):**

> Nonlinear Delta Memory: Scaling Pure Recurrent Language Models by Multi-Programming

**One-sentence thesis (`ndmpapernotes.md` lines 10–13):**

Pure nonlinear recurrent language models can be trained at billion-parameter
scale when the recurrent computation is organized as a multi-programmed GPU
workload, *and* the useful memory resource is a nonlinear delta-correcting
matrix update rather than matrix state or temporal nonlinearity alone.

The two organizing pillars (per task brief):

1. **Multi-programming-parallel scaling thesis** — pure recurrent NDM/E88 at
   1.27B is enabled by exposing many small bounded memory programs (per-head,
   per-state-tile, per-batch) to the GPU, not by parallelizing along time.
2. **Nonlinear-recurrence-at-scale with the delta-correcting mechanism** — the
   tanh-on-state nonlinearity gives the computational class; the delta write
   `v - Sᵀk` is the *load-bearing* mechanism that separates NDM from M2RNN-style
   raw-write matrix RNNs at fixed update budget.

---

## 1. Abstract Sketch (4–6 sentences)

> Linear-state recurrent language models (Mamba, RWKV, GDN, mLSTM) and
> recurrent–attention hybrids dominate the large-scale RNN literature; pure
> nonlinear recurrence is widely assumed to be impractical at scale. We
> introduce **Nonlinear Delta Memory (NDM)**, a pure recurrent architecture
> whose per-head update `S = tanh(d·S + k(v − Sᵀk)ᵀ)` combines a bounded
> tanh-on-state nonlinearity with a delta-correcting matrix write, and we
> train a 1.27B-parameter pure-recurrent NDM stack to convergence on a
> Pile-class corpus. The key systems insight is multi-programming:
> recurrence stays serial along time but exposes massive parallelism across
> many small bounded memory programs, made practical by a fused Triton
> recurrence kernel with sparse checkpointing that runs identically on
> CUDA and ROCm. On controlled S3/S5 permutation-composition probes, NDM
> separates from FLA-GDN (linear-state), Mamba2 (linear-state), and from
> both published and CMA-ES-reshaped M2RNN variants (raw-write
> nonlinear-state) — establishing that *neither* matrix state *nor* temporal
> nonlinearity alone is sufficient. A trusted Lean 4 core, with no
> `sorry`/`axiom`/`opaque`/`native_decide`, proves an update-family
> resource separation between NDM and M2RNN, the S5 tracker, and the
> finite-state ceiling for fixed-precision recurrent recognizers.

---

## 2. Section List

Per task spec: at least 3 bullets of intended content per section. Each
bullet cites the upstream synthesis doc that supplies the content.

### 2.1 Introduction

- **Field assumption to reject** — pure serial RNNs are "expressive but
  impractical at scale"; the field's response has been linear recurrence
  (Mamba/RWKV/GDN/mLSTM) or hybridization. Cite
  `docs/related_work_nonlinear_rnns.md` entries 1–15 as evidence that
  every billion-scale recurrent LM published to date is linear-state or
  hybrid.
- **Counter-hypothesis** — the missing ingredient is *multi-programming*
  (many bounded per-head memory programs), not attention or linear scan.
  Frame the paper as the answer to: *did pure nonlinear recurrence fail
  because it was the wrong computation, or because it was parallelized
  along the wrong axis?* (`ndmpapernotes.md` lines 49–51.)
- **Numbered contributions (six)** — restate from `ndmpapernotes.md` lines
  54–68, but reordered around the two pillars so the multi-programming
  systems contribution and the delta-correction mechanism contribution
  bracket the empirical results.
- **Concurrent work positioning** — M2RNN (`arXiv:2603.14360`) at 410M
  pure-recurrent and xLSTM-1.3B (mixed sLSTM/mLSTM) are the two
  closest-prior comparables; cite explicitly per
  `docs/related_work_nonlinear_rnns.md` §"Closest Prior Art".

### 2.2 Background

- **Linear-state recurrence taxonomy** — Mamba/Mamba2, RetNet, GLA,
  DeltaNet (delta rule but linear-in-S), Gated DeltaNet, RWKV-4/5/6/7,
  HGRN/HGRN2, S5, MinGRU/MinLSTM, mLSTM, RG-LRU/Griffin. Use the linearity
  criterion from `docs/related_work_nonlinear_rnns.md` lines 13–15:
  `h_t = A_t h_{t-1} + b_t` with `A_t, b_t` depending only on `x_t`.
- **Nonlinear-state recurrence taxonomy** — classical LSTM/GRU (never
  scaled to ≥500M on Pile-class data — `docs/related_work_nonlinear_rnns.md`
  entry 20), sLSTM (memory mixing through `h_{t-1}`), M2RNN (matrix state
  inside tanh with raw-write injection), Titans (MLP memory with online
  gradient updates), Liquid LFM.
- **State-tracking theory and Barrington witness** — solvable vs
  non-solvable groups; S3 (solvable, 6 states) as control and S5
  (non-solvable, 120 states) as the Barrington NC¹-complete probe.
  Reference `formal/lean/PROOF_INVENTORY.md` `S5Witness.s5_not_solvable`
  and `S5Witness.s5_state_count`.
- **Why "matrix state" is background, not novelty** — MSE doc lineage in
  `docs/DESIGN_DOSSIER.md` §7 (MSE inline note): `d²` dynamic-state
  capacity at `O(d²)` cost via outer products is the precondition NDM
  inherits.

### 2.3 Architecture — Nonlinear Delta Memory

- **Per-head update equation** — `docs/DESIGN_DOSSIER.md` §1 reference
  block: `r = Sᵀk; δ = v − r; S = tanh(d·S + outer(k, δ)); y = silu(g)·Sᵀq`.
  Show the line-for-line PyTorch fallback at `ndm/models/e88_fused.py:298-316`.
- **Why each ingredient** — three pillars from `docs/DESIGN_DOSSIER.md`
  §1 *"Why this is the chosen update"*: (a) tanh-on-state is the
  UTM-class lever (E63); (b) delta write makes memory error-correcting
  (MENU); (c) many small heads expose multi-programming, not one large
  matrix (M2C).
- **Multi-programmed shape** — production geometry from
  `docs/DESIGN_DOSSIER.md` §1: 1.27B = `dim=1664, depth=12, n_heads=370,
  n_state=32` (370 independent 32×32 bounded memory programs per layer,
  per batch element). Lean-witnessed by
  `RecurrentResourceFormalism.ndm_1p27B_*` (PROOF_INVENTORY entries
  175–181).
- **Contrast with raw-write M2RNN** — `docs/DESIGN_DOSSIER.md` §5; show
  M2RNN's `H = f·H + (1−f)·tanh(H·W + k vᵀ)` side-by-side with NDM's
  delta write, and cite the empirical instability of paper-shape M2RNN
  at 1.27B (grad norms 1e6–1e7) from M2C.

### 2.4 Systems — Triton Kernel and Multi-Programmed Scaling

- **Kernel layout** — `docs/DESIGN_DOSSIER.md` §4.1: `[T,B,H,*]` internal
  layout, one program per `(batch, head_block)`, `BLOCK_H` autotuned over
  `{1,2,4,8,16}`, state tile in registers/SRAM. Source files
  `ndm/triton/e88_triton_forward.py`, `ndm/triton/e88_triton_backward.py`,
  `ndm/triton/e88_triton_optimized.py`.
- **Fusions and numerical stability** — SiLU-q/k/v, L2-q/k, output gate,
  and the `2·sigmoid(2·pre) − 1` numerically-stable tanh — all fused into
  the recurrence kernel (`docs/DESIGN_DOSSIER.md` §4.1, stability lesson
  #7 in §3). Removes 1–2 PyTorch launches per layer call;
  ~50–60 ms/step saving at depth=14 production.
- **Sparse checkpoint backward** — every `CKPT_INTERVAL = 16` steps; ~16×
  activation memory shrink at K=16 (`docs/DESIGN_DOSSIER.md` §4.1).
- **Why Triton over HIP** — `docs/DESIGN_DOSSIER.md` §4.3 (FRO): one source
  for CUDA + ROCm at ~70% of hand-tuned HIP throughput, in 1 week vs 3–6
  weeks of porting.
- **Distributed plan** — `docs/DESIGN_DOSSIER.md` §4.4: ScheduleFree-AdamW
  per-island + hierarchical local-SGD model averaging (H=250); ParaRNN as
  the high-risk speculative path. Pure-recurrent NDM does not need
  sequence parallelism to be competitive.

### 2.5 Language Modeling Results — 1.27B Comparison

- **Setup** — byte/token streams, schedule-free AdamW, context-2K
  curriculum, 4-GPU smoke established H=250 averaging interval. Cite
  `paper/notes_reconciliation.md` O2/O3 and `docs/DESIGN_DOSSIER.md` §4.4.
- **Models compared** — E88/NDM, FLA-GDN, Mamba2, M2RNN-CMA,
  M2RNN-paper (as stability control). Model-family table from
  `ndmpapernotes.md` lines 72–79.
- **Headline metric** — wallclock loss / bits-per-byte (smoothed last 5K,
  10K, 50K). Caption claim: *once recurrent geometry and kernels are
  optimized, pure nonlinear recurrence trains in the same wallclock loss
  band as strong linear-recurrent baselines* (`ndmpapernotes.md` line
  212–213).
- **Stability evidence** — paper-shape M2RNN stopped at step 8400 (loss
  11.574, grad 4.19e7); tied/CMA-ES M2RNN stable at step 9250 (loss 4.085,
  grad 1.48) — `paper/notes_reconciliation.md` G2.
- **Honest gap** — wallclock racer plots and frozen loss curves currently
  live in `~/elman/` (`paper/notes_reconciliation.md` C4, items 5–6 of
  missing-evidence summary). See "Pending Experimental Closure" below.

### 2.6 Expressivity Results — State-Tracking Mechanism Witness

- **S5 / S3 permutation composition (headline)** — train T=128, 3 seeds,
  8M parameter-matched. E88/NDM **0.7918** vs FLA-GDN 0.3552 vs M2RNN-tied
  0.2157 vs M2RNN-paper 0.1698 on S5; S3 0.10/0.72/0.31/0.38 respectively
  (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a). Source numbers from
  `paper/ndmpapernotes.md` lines 153–164.
- **S5 length extrapolation** — T=128→1024: NDM 0.79→0.42→0.22→0.11; all
  baselines collapse faster (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md`
  §5a). NDM separates *at training length*, not only under extrapolation.
- **6-task canonical sweep at 8M** — parity, modular counter, FSM
  tracking, dyck, associative recall, selective copy
  (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §§1a, 2a, 3a, 4a, 6a, 7a).
  E88 wins or ties FLA-GDN on 5/6 (FLA edges only on associative recall:
  0.997 vs 0.881 — attention-natural task).
- **Length-extrapolation curves on parity, modular counter, FSM** —
  `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §§1b, 3b, 4b. E88 retains
  meaningful accuracy at 12.5× training length on parity (0.89 at T=500
  vs FLA 0.55); FLA collapses to near-random.
- **Hybrid AABB ablation** — `[E88, E88, GDN, GDN]` *underperforms* pure
  E88 on state-tracking (modular counter hybrid 0.54 vs E88 0.90; FSM
  tracking hybrid 0.71 vs E88 1.00 — `docs/EXPRESSIVITY_RESULTS_SUMMARY.md`
  §§3a, 4a). Hybridization with linear scan layers degrades the
  state-tracking capability.
- **Interpretation** — M2RNN underperforms both at S5 *training length*
  (0.22 vs E88 0.79) and S3 (0.31 vs E88 1.00) at the same parameter
  count. This is the mechanism witness: matrix-state + tanh-on-state
  alone (M2RNN) is *not* sufficient; the delta correction is the
  load-bearing piece. Mirrors the Lean `ndm_m2rnn_one_step_resource_separation`.

### 2.7 Formal Results — Trusted Lean 4 Core

- **Trust gate and surface** — `formal/lean/PROOF_INVENTORY.md` §Trust-Gate
  Status: 8 trusted source files in the `PaperCore` import closure, scripts
  forbid `sorry`/`admit`/`axiom`/`opaque`/`native_decide`. Caveat: the
  scripts depend on `rg`; full Lean kernel build needs `lake build` in an
  environment with Mathlib resolved (PROOF_INVENTORY §Run Result).
- **Theorem set A — update-family resource separation** — feature-level
  separation (`m2rnn_features_not_e88_features`), one-step embedding
  positive direction (`m2rnn_read_then_delta_embeds_e88_delta_update`),
  and the 2D / general-K,V negative direction
  (`ndm_m2rnn_one_step_resource_separation`,
  `ndm_m2rnn_one_step_resource_separation_embeds`).
- **Theorem set B — S5 tracker** —
  `S5Witness.s5_state_count` (= 120), `s5_not_solvable`,
  `S5Tracker.recognizer_state_count`, `run_append` (composition),
  `pythonRun_eq_tracker_tuple` (Python-bridge), and
  `S5NDMRealization.s5_transition_key_count` (480 state/input keys).
- **Theorem set C — finite-state ceiling** —
  `fixed_precision_state_space_finite` plus the exact lookup-table
  realization `exactTransitionMemory_run`.
- **Theorem set D — capacity separation E88 vs E1H** —
  `e88_dof_strictly_exceeds_e1h`, `e88_addressable_e1h_not`,
  `capacity_separation`, `total_capacity_separation`.
- **Explicit non-claims (paragraph in §8 of the paper)** — no Lean
  lower-bound covering all linear scan models on S5; no Barrington/NC¹
  completeness inside Lean; no formal proof that trained real-valued NDM
  exactly recovers the lookup table. Cite
  `formal/lean/PROOF_INVENTORY.md` §"Claimed-but-Not-Formalized".

### 2.8 Related Work

- **The linear-state cohort** — Mamba/Mamba2, DeltaNet/Gated DeltaNet,
  RetNet, GLA, RWKV-4/5/6/7, HGRN/HGRN2, mLSTM, RG-LRU/Griffin/RecurrentGemma.
  All linear-in-h state updates; distinguished from NDM by the linearity
  criterion (`docs/related_work_nonlinear_rnns.md` lines 13–15).
- **The nonlinear-state cohort** — sLSTM (xLSTM-1.3B; nonlinear via
  memory mixing but mixed 7:1 with linear mLSTM blocks);
  **M2RNN** (`arXiv:2603.14360`) — closest comparable, but largest *pure*
  recurrent variant is 410M and uses raw-write `tanh(H·W + k vᵀ)` rather
  than delta correction; Titans (hybrid, MLP memory); classical
  LSTM/GRU (never published at ≥500M Pile-class);
  `arXiv:2505.17852` 1B LSTM ZOO (zero-order optimization, not standard
  gradient training).
- **Closest-prior-art treatment** — explicit M2RNN and xLSTM
  subsections per `docs/related_work_nonlinear_rnns.md` §"Closest Prior
  Art". State the recommended novelty framing
  (`docs/related_work_nonlinear_rnns.md` lines 252–260): "To the best of
  our knowledge, NDM is the first pure nonlinear recurrent language model
  (no attention, no linear-recurrent layers) trained at ≥1B parameters
  to near-convergence on a large-scale web corpus. Concurrent M2RNN
  demonstrates nonlinear matrix-state recurrence at 410M pure-recurrent;
  NDM is larger and employs a distinct delta-correcting update."

### 2.9 Limitations

- **No completed Lean lower bound** for all linear scan models on S5
  (`formal/lean/PROOF_INVENTORY.md` §"Claimed-but-Not-Formalized";
  `ndmpapernotes.md` line 317).
- **S5 length extrapolation is not solved at scale** — even NDM degrades
  to 0.11 at T=1024 (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a).
- **Geometry-sensitivity caveat** — same update family (M2RNN) looks
  weak in published shape and stronger under CMA-ES reshape; the strong
  empirical claim is conditional on the multi-programmed shape, not on
  the update equation alone (`docs/DESIGN_DOSSIER.md` §5;
  `paper/notes_reconciliation.md` G2/G3).
- **Out-of-repo training artifacts** — production loss curves and
  checkpoints currently live in `~/elman/`; release plan tracks this
  under `huggingface-release-plan` (separate task).
- **M2RNN comparison rests on local reproduction choices** — the
  published M2RNN paper-shape is a sympathetic re-implementation;
  authors may have shape/training details we did not match.
- **Loop contradictions still open** — output gate on/off, tanh vs linear
  state, simple vs Mamba2 decay (`docs/DESIGN_DOSSIER.md` §§6.1–6.4).
  Production keeps the more conservative settings; revalidation at
  1.27B is flagged as the highest-value first follow-up.

### 2.10 Conclusion

- **Pillar 1 restated** — pure recurrence is viable at billion-parameter
  scale when shaped as a many-program GPU workload; the parallelism is
  across heads/programs/batch, not along the time axis.
- **Pillar 2 restated** — delta-correcting matrix memory is the
  empirically separable mechanism; nonlinearity alone (M2RNN) is not
  sufficient.
- **What we did *not* claim** — see §2.9; in particular no
  "first nonlinear matrix-state RNN" claim, no NC¹ lower bound, no
  "linear scans cannot do S5" formal proof.
- **Open horizon** — how far nonlinear recurrent reasoning can scale once
  memory, geometry, and systems are co-designed (`ndmpapernotes.md`
  line 313).

---

## 3. Figure List

At least four figures (task spec). Each figure cites the result source
that supplies its data.

### Figure 1 — Mechanism and S5 (the "delta-correction is load-bearing" figure)

- **Panel A — Update schematic.** Side-by-side block diagrams of NDM
  (`S = tanh(d·S + k·δᵀ)`), M2RNN (`H = f·H + (1−f)·tanh(H·W + k·vᵀ)`),
  and GDN (linear delta). Source: `docs/DESIGN_DOSSIER.md` §§1, 5;
  `ndmpapernotes.md` lines 87–113.
- **Panel B — S3/S5 train-length bars.** 4 models × 2 tasks, with random
  baseline lines (1/6 and 1/120). Source numbers:
  `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a; `paper/ndmpapernotes.md`
  lines 155–164.
- **Panel C — S5 length-extrapolation curves.** 4 models × 4 lengths
  (128 / 256 / 512 / 1024). Source: `docs/EXPRESSIVITY_RESULTS_SUMMARY.md`
  §5a; `paper/ndmpapernotes.md` lines 168–173.

### Figure 2 — CMA-ES and Recurrent Geometry

- **Panel A — Searched-shape comparison table.** NDM/E88 (H=370, N=32,
  dim=1664, depth=12) vs paper-shaped M2RNN (shared q/k, hundreds of
  v/f/g heads) vs CMA-ES M2RNN (per-head q/k/v). Source:
  `docs/DESIGN_DOSSIER.md` §§1, 5; `paper/notes_reconciliation.md` G2.
- **Panel B — Stability divergence.** Loss + grad-norm vs step for
  paper-shape M2RNN (diverges by step 8400, grad ~4.19e7) vs CMA-ES
  M2RNN (stable, loss 4.085 at step 9250). Source:
  `paper/notes_reconciliation.md` G2; `docs/DESIGN_DOSSIER.md` §5.
- **Panel C — Multi-program geometry.** Schematic: 370 heads × 32×32
  state tiles, with GPU thread-block mapping. Source:
  `docs/DESIGN_DOSSIER.md` §4.1.

### Figure 3 — 1.27B Language-Model Racers

- **Main plot.** Loss / bpb vs wallclock for E88/NDM, FLA-GDN, Mamba2,
  M2RNN-CMA, M2RNN-paper. Smoothed last 5K/10K/50K windows. Source:
  `paper/ndmpapernotes.md` §"Figure 3"; data lives in `~/elman/` and
  needs to be staged into the repo (see "Pending Experimental Closure").
- **Inset.** Best-loss-by-wallclock summary table per model.
- **Caption claim.** "Once recurrent geometry and kernels are optimized,
  pure nonlinear recurrence trains in the same wallclock loss band as
  strong linear-recurrent baselines." (`ndmpapernotes.md` line 212.)

### Figure 4 — Systems

- **Panel A — Kernel-launch reduction.** Fused vs unfused step time at
  matched accuracy; 1–2 PyTorch launches saved per layer call (~50–60
  ms/step at depth=14). Source: `docs/DESIGN_DOSSIER.md` §4.1.
- **Panel B — Sparse-checkpoint memory.** Activation memory `T` vs
  `T/K + 1` at K=16, with the matching ~16× shrink. Source:
  `docs/DESIGN_DOSSIER.md` §4.1.
- **Panel C — Triton vs CUDA register-owned throughput.** Per-step
  ms at `(H, B, T)` sweep on NVIDIA + ROCm; ROCm parity row needed.
  Source: `paper/notes_reconciliation.md` S1–S8, S7/S8 currently flagged
  as out-of-repo. Required new microbenchmark per
  `paper/notes_reconciliation.md` missing-evidence #5.

### Figure 5 (optional, conditional on space) — Canonical Expressivity Sweep

- **6-task bar chart.** E88 / FLA-GDN / hybrid AABB on parity, modular
  counter, FSM tracking, dyck, associative recall, selective copy
  (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §§1a, 2a, 3a, 4a, 6a, 7a).
- **Hybrid degradation callout.** Highlight `[E88,E88,GDN,GDN]` collapsing
  to FLA-level on modular counter and FSM tracking
  (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md` claim-mapping row 3).
- **Caption claim.** Hybridization with linear-scan layers *degrades*
  state-tracking — purity, not mixing, is what gives NDM its
  state-tracking profile.

---

## 4. Claim → Evidence Map (for each numbered contribution)

| # | Contribution (from `ndmpapernotes.md` lines 54–68) | Primary evidence file(s) | Lean witness if any |
|---|---|---|---|
| C1 | **Architecture:** NDM pure recurrent matrix-memory with nonlinear delta-correcting update | `docs/DESIGN_DOSSIER.md` §1 (update equation; `ndm/models/e88_fused.py:298-316`); `paper/notes_reconciliation.md` C1; `docs/related_work_nonlinear_rnns.md` §"Summary Verdict" | `PROOF_INVENTORY.md` `M2RNNComparison.write_rule_separates_m2rnn_and_e88`, `RecurrentResourceFormalism.e88_*` aliases |
| C2 | **Training geometry:** CMA-ES and follow-up search → many-head multi-programmed shape at 1.27B | `docs/DESIGN_DOSSIER.md` §1, §6.7 (production head count); `paper/notes_reconciliation.md` C2, G1/G2/G3 (CMA-ES search code itself is out-of-repo — see Pending) | `PROOF_INVENTORY.md` `ndm_1p27B_programs_per_batch_token_bs5` (= 22200), `ndm_1p27B_state_scalars_per_layer` |
| C3 | **Systems:** fused Triton kernel with L2 norm, sparse checkpointing, output gating, CUDA/ROCm | `docs/DESIGN_DOSSIER.md` §§4.1–4.3; `paper/notes_reconciliation.md` S1–S8 (S4, S7, S8 partial/OOR) | — (systems claims are not formalized) |
| C4 | **Language modeling:** 1.27B pure NDM in same wallclock loss regime as linear-recurrent baselines | `paper/notes_reconciliation.md` C4 — currently **out-of-repo**; production loss qualitative claim in `docs/M2RNN_E88_COMPARISON.md` via `DESIGN_DOSSIER.md` §5 | — |
| C5 | **Expressivity:** S3/S5 separates NDM from FLA-GDN, Mamba2, M2RNN | `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5 (S3/S5 numbers); `paper/ndmpapernotes.md` lines 155–173. Mamba2 row currently missing — see Pending. | `PROOF_INVENTORY.md` `S5Witness.*`, `S5Tracker.*`, `S5NDMRealization.s5_transition_key_count` (= 480) |
| C6 | **Formal core:** Lean-checked update-family resource separation, S5 tracker, finite transition realization | `formal/lean/PROOF_INVENTORY.md` §§Module M2RNNComparison, RecurrentResourceFormalism, S5Witness, S5Tracker, S5NDMRealization | All Lean witnesses listed in PROOF_INVENTORY §Coverage Table |

Notes on the map: rows C1, C3, C5, C6 are **supported in-repo**; rows C2
and C4 are **partial / out-of-repo** and listed under "Pending
Experimental Closure" below.

---

## 5. Ready-to-Write vs Pending Experimental Closure

### 5.1 Ready to write (evidence is in this repo as of 2026-05-23)

- **§3 Architecture** — all equations, ablation rationale, and design
  decisions are in `docs/DESIGN_DOSSIER.md` with 43 verified line refs
  into `ndm/`.
- **§5 Systems** — kernel code paths, fusion list, sparse-checkpoint
  rationale, distributed plan. (Sub-rows S4 "no contig copies",
  S7 "ROCm parity", S8 "throughput numbers" lack in-repo benchmarks; see
  pending §5.2.)
- **§6.1–§6.4 Expressivity (canonical-sweep tasks)** — parity, modular
  counter (K=5 in-distribution + length extrapolation), FSM tracking
  (in-distribution + length extrapolation), dyck, associative recall,
  selective copy — all numbers in
  `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §§1, 2, 3, 6, 7 from
  `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` and
  `LENGTH_EXTRAP_RESULTS.md`.
- **§6.5 S5/S3 expressivity headline** — numbers in
  `paper/ndmpapernotes.md` lines 153–173, surveyed in
  `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a. NOTE: the *raw JSONs* for
  those runs are not in `experiments/expressivity_tasks/results/`; only
  the numerical summaries. See pending §5.2 item 1.
- **§7 Formal results** — every theorem listed in
  `formal/lean/PROOF_INVENTORY.md` Coverage Table is in the trusted
  PaperCore import closure. The `rg`-based trust gate currently passes
  vacuously due to missing `ripgrep`; conclusive verification needs
  `lake build` (PROOF_INVENTORY §Run Result).
- **§8 Related work** — all 21 entries with citations, recurrence-type
  classifications, and verdicts in `docs/related_work_nonlinear_rnns.md`.
- **§9 Limitations** — items map directly to identified gaps and the
  open contradictions in `docs/DESIGN_DOSSIER.md` §6.

### 5.2 Pending experimental closure (concrete file paths that must fill in)

1. **`experiments/expressivity_tasks/results/s5_witness_8m_20260521/` —
   re-run and commit JSON artifacts.** Currently empty per
   `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a (line 209: "no committed
   artifacts"); numbers were transcribed manually into
   `paper/ndmpapernotes.md`. Required re-run script:
   `experiments/expressivity_tasks/run_separation_suite.py --tasks
   s3_permutation s5_permutation --use_triton_e88`. Blocks Figure 1
   panels B, C from claiming reproducibility from this repo.
2. **`experiments/expressivity_tasks/results/full_8m_matched_20260511/`
   — empty.** Affects canonical-sweep numbers (parity / dyck / fsm /
   mod_counter / assoc_recall / selective_copy). Required script:
   `run_canonical_sweep.py`. Blocks Figure 5.
3. **`experiments/expressivity_tasks/results/separation_8m_matched_20260511/`
   — empty.** Affects keyed_fsm_memory and overwrite/reset_recall — the
   tasks targeted at the NDM vs M2RNN delta-correction
   separation. Source: `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §8 (line
   303). Blocks a sharper mechanism panel in §6 (currently optional).
4. **`experiments/expressivity_tasks/results/modular_counter_followup_20260511/`
   — empty.** Affects the M2RNN-tied 10K-step edge question and the
   K=20 / K=50 hard-counter sweep
   (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §4c; missing-evidence #7 in
   `paper/notes_reconciliation.md`).
5. **Mamba2 baseline on S3/S5.** `README.md` lists Mamba2 as a key
   model family but the in-repo S3/S5 numbers cover only E88, FLA-GDN,
   M2RNN, M2RNN-paper (`paper/notes_reconciliation.md` C5 + missing
   evidence #3). Adding Mamba2 strengthens the §6 separation claim.
6. **1.27B wallclock racer plots.** Figure 3 data lives in `~/elman/`
   per `paper/notes_reconciliation.md` C4 + out-of-repo dependency
   table (rows 1–4). Need smoothed loss-vs-wallclock for E88, FLA-GDN,
   Mamba2, M2RNN-CMA at frozen checkpoint hashes, written into
   `paper/results/figure_3/`.
7. **Triton vs CUDA throughput microbenchmark.** Source:
   `paper/notes_reconciliation.md` missing-evidence #5. Affects Figure 4
   Panel C.
8. **ROCm parity test.** `paper/notes_reconciliation.md` missing-evidence
   #8; affects systems-section ROCm claim S7.
9. **CMA-ES search artifacts.** `paper/notes_reconciliation.md`
   missing-evidence #4 + out-of-repo dependency row 7. Affects
   Contribution C2 and Figure 2 Panel A.

**Gating decision rule (proposed).** §5 Systems, §6.1–§6.4
canonical-sweep expressivity, §6.5 S5/S3 headline (with re-committed
JSONs), §7 Formal results, §8 Related work, and §9 Limitations are all
fully writable today. **§4 Language modeling and Figure 3 are the
critical-path blockers** for a submittable paper. Until the racer plots
are frozen and committed, the paper has to either (a) defer the §4
plots to an arXiv-only version, or (b) stage the `~/elman/` artifacts
into a new `paper/results/` tree under the release plan.

---

## 6. Target Venue and Timing

### Recommendation: NeurIPS 2026 main track

**Why NeurIPS specifically:**

- The two pillars (multi-programming systems result + nonlinear-recurrence
  mechanism result) span systems and modeling — NeurIPS reviewers are the
  right audience for both, more so than ICLR (which leans modeling) or
  COLM (heavily LM-curated, but smaller systems audience).
- The Lean trusted core is an unusual but well-received contribution in
  NeurIPS — recent precedent in formal verification + ML at NeurIPS
  (verified ML, neuro-symbolic tracks).
- The closest prior art (M2RNN at `arXiv:2603.14360` — March 2026
  concurrent; xLSTM-1.3B; RWKV-7) is all NeurIPS-style work; reviewers
  will recognize the comparable.

### Backup ordering

1. **NeurIPS 2026 main track** — preferred.
2. **ICLR 2027** — second choice if NeurIPS timing slips. ICLR review
   cycle suits a paper that needs another round of empirical depth.
3. **COLM 2026** — viable for the LM-specific subset of the result, but
   the systems+Lean+expressivity span fits less well.
4. **arXiv preprint** — required regardless (priority date for the
   "first pure nonlinear recurrent ≥1B parameter LM" claim against any
   future M2RNN scale-up). Should be posted as soon as Figure 3 data is
   committed, even ahead of NeurIPS deadline.

### Timing constraints

- **NeurIPS 2026 abstract deadline** (typical): mid-May 2026. **Already
  past as of 2026-05-23.** If main track for 2026 is not feasible, the
  realistic targets are NeurIPS 2026 *Datasets and Benchmarks* (if
  framed around the S5 expressivity benchmark — late June typical),
  ICLR 2027, or NeurIPS 2027.
- **Priority date risk.** M2RNN (arXiv:2603.14360) is concurrent —
  March 2026. The pure-recurrent 410M vs NDM-1.27B comparison must be
  posted to arXiv soon to lock the scale-up priority claim.
- **Pending items 5.2 #1, #5, #6** (S5 JSONs, Mamba2, racer plots) are
  the critical-path closures. Item #1 is fastest (re-run an existing
  separation suite). Items #5 and #6 require the most external compute
  / `~/elman/` artifact migration.
- **Lean kernel build verification.** Per `formal/lean/PROOF_INVENTORY.md`
  §Run Result, `lake build` was not completed in the inventory
  session; before submission, the trust-gate must be run conclusively
  in an environment with `ripgrep` installed and Mathlib cached.

### Provisional milestone calendar (working assumption)

| Date | Milestone |
|---|---|
| 2026-05-30 | Pending §5.2 #1 closed (S5 JSONs re-run and committed). |
| 2026-06-10 | Pending §5.2 #6 closed (`~/elman/` racer artifacts staged into `paper/results/`). |
| 2026-06-15 | First full draft (text + Figure 1, 2, 4, 5; Figure 3 with whatever §5.2 #6 yields). |
| 2026-06-30 | arXiv v1 posted. |
| 2026-07-15 | Mamba2 baseline (§5.2 #5) and ROCm parity (§5.2 #8) added in arXiv v2. |
| 2026-08-(deadline) | NeurIPS 2026 D&B / ICLR 2027 / next-cycle decision based on what is closed. |

---

## 7. Cross-Reference Index (every upstream synthesis doc is explicitly cited)

- `docs/related_work_nonlinear_rnns.md` — §§2.1, 2.2, 2.8, Figure 5
  context (linear vs nonlinear taxonomy), §6 (M2RNN/xLSTM closest-prior
  treatment).
- `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` — §§2.6, Figure 1, Figure 5,
  Pending §5.2 items 1–4.
- `docs/DESIGN_DOSSIER.md` — §§2.3, 2.4, Figure 1A, Figure 2, Figure 4,
  §2.9 (contradiction list).
- `formal/lean/PROOF_INVENTORY.md` — §§2.7, claim→evidence rows C1/C2/C5/C6,
  §2.9 (non-claims), Lean-trust-gate caveat in §6.
- `paper/notes_reconciliation.md` — §2.5 (gap list), §§4–5 (claim
  evidence rows and pending experiments), §6 (timing constraints).
- `paper/ndmpapernotes.md` — §§0 (title, thesis), 1 (abstract sketch),
  2 (every section header), 3 (figure list — Figures 1–4 originate
  here), 4 (six contributions).

---

## 8. Validation Self-Check (against task spec)

- [x] **All 5 upstream summary docs are explicitly read and cited.** —
  §0 + §7 cross-reference index.
- [x] **Each section has at least 3 bullet points of intended content.**
  — §2.1 (4), §2.2 (4), §2.3 (4), §2.4 (5), §2.5 (5), §2.6 (5), §2.7 (5),
  §2.8 (3), §2.9 (6), §2.10 (4).
- [x] **Every numbered contribution is mapped to a specific evidence
  file (no 'TBD' for contributions verifiable in-repo).** — §4 table;
  C2 and C4 explicitly marked partial/out-of-repo with the specific
  `paper/notes_reconciliation.md` rows.
- [x] **Pending-experiments list is concrete (file paths to result dirs
  that need to fill in).** — §5.2 lists nine items with exact result
  directories or scripts.
- [x] **At least 4 figures specified.** — Figures 1–4 mandatory; Figure 5
  optional.

---

*Outline only — paper prose is out of scope per task brief. Generated by
`paper-outline`; downstream consumer `direction-forward-memo` will
integrate this with the rest of the audit set.*
