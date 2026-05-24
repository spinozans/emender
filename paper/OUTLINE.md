# NDM Paper Outline

**Status:** Draft outline (not the paper itself).
**Generated:** 2026-05-23 by task `paper-outline`.

---

## CHANGELOG

- **2026-05-24 (`reframe-paper-narrative`):** Dropped the "first ≥1B pure
  nonlinear RNN at scale" priority framing throughout. The concurrent M2RNN
  work (arXiv:2603.14360) trained a pure-recurrent variant at 410M, so the
  priority claim is contested at best and not the strongest version of the
  story. Replaced with the **three-pillar reframe**: (1) generality of
  multi-programming for pure nonlinear recurrent LMs at scale, (2) the update
  rule matters — the delta-correcting write `v − Sᵀk` separates from raw-write
  matrix RNNs and approaches the NC1 limit via S5 tracking, (3) FLOPs-per-bit
  convergence under CMA-ES (N=4) — when sensibly-tuned recurrent baselines are
  all CMA-optimized, they learn at the same FLOPs-per-bit, so the contribution
  is the architectural *option* plus the *mechanism*, not raw training speed.
  Reframed sections: §0 (thesis), §1 (abstract), §2.1 (intro and
  contributions), §2.10 (conclusion), §6 (venue/timing). Lean formal core,
  Triton kernel, and S5 expressivity result are now positioned as supporting
  evidence under the three pillars, not as separate top-level contributions.
  Cross-references added to forthcoming `docs/CMA_FLOP_RATE_FINDING.md` (task
  `harvest-cma-flop-rate-finding`) and `docs/QA_REASONING_PROGRESSION.md`
  (task `harvest-qa-reasoning-quiz-progression`). NC1 wording is held at
  "approaches the NC1 limit via S5 tracking" pending the outcome of
  `lean-formalization-gap-audit` — see TODO markers inline.

---

**Inputs read (cited inline below):**

- `docs/related_work_nonlinear_rnns.md` — literature survey (lit-survey-nonlinear-rnns)
- `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` — expressivity consolidation (consolidate-expressivity-results)
- `docs/DESIGN_DOSSIER.md` — architecture/systems synthesis (synthesize-design-dossier)
- `formal/lean/PROOF_INVENTORY.md` — trusted Lean surface (lean-proof-inventory)
- `paper/notes_reconciliation.md` — claim-by-claim evidence reconciliation (reconcile-paper-notes)
- `paper/ndmpapernotes.md` — original paper notes
- `docs/CMA_FLOP_RATE_FINDING.md` — *forthcoming*, produced by
  `harvest-cma-flop-rate-finding`; supplies the FLOPs-per-bit convergence
  evidence under CMA-ES (Pillar 3). Placeholder reference until that task
  lands.
- `docs/QA_REASONING_PROGRESSION.md` — *forthcoming*, produced by
  `harvest-qa-reasoning-quiz-progression`; supplies the over-training
  quiz/reasoning capability progression for the 1.27B model. Placeholder
  reference until that task lands.

---

## 0. Title and One-Sentence Thesis

**Working title:**

> Nonlinear Delta Memory: A Multi-Programmed Recipe for Pure Recurrent
> Language Models at Scale

**One-sentence thesis:**

Pure nonlinear recurrent language models can be trained at billion-parameter
scale by organizing the recurrent computation as a multi-programmed GPU
workload, and the *update rule* — specifically a delta-correcting matrix
write of the form `S ← tanh(d·S + k(v − Sᵀk)ᵀ)` — is what separates the
resulting models from raw-write nonlinear matrix RNNs on state-tracking
expressivity, even though raw training speed (FLOPs-per-bit) is largely
indifferent to the choice of recurrent update.

The three organizing pillars:

1. **Generality of multi-programming for pure nonlinear recurrent LMs at
   scale.** The systems recipe — many small bounded memory programs
   (per-head, per-state-tile, per-batch) exposed to the GPU instead of
   parallelism along time — is *general*. It is not specific to NDM.
   Any nonlinear matrix-state recurrence with a similar multi-head /
   small-state-tile shape becomes trainable at billion-parameter scale
   under this recipe. The paper documents *how* to do it.
2. **The update rule matters.** The delta-correcting write
   `v − Sᵀk` separates from raw-write matrix RNNs (the M2RNN family) on
   state-tracking expressivity probes and approaches the NC1 limit via
   S5 tracking. *(TODO: tighten the "approaches the NC1 limit" wording
   after `lean-formalization-gap-audit` reports back on which side of
   the formal/empirical boundary the claim sits.)*
3. **FLOPs-per-bit convergence under CMA-ES (N = 4).** When sensibly-tuned
   recurrent baselines are all CMA-ES-optimized for shape and
   hyperparameters, they learn at almost the same FLOPs-per-bit. The
   contribution of the present work is therefore the *architectural
   option* (pure nonlinear recurrence is viable at scale at all) plus
   the *mechanism* (delta correction earns the expressivity), not raw
   training speed. Evidence: forthcoming
   `docs/CMA_FLOP_RATE_FINDING.md`.

---

## 1. Abstract Sketch (6–8 sentences)

> Linear-state recurrent language models (Mamba, RWKV, GDN, mLSTM) and
> recurrent–attention hybrids dominate the large-scale RNN literature;
> pure nonlinear recurrence has been widely assumed to be impractical at
> scale. This paper makes three claims about pure nonlinear recurrent
> language models, organized around a single architecture — Nonlinear
> Delta Memory (NDM) — whose per-head update
> `S ← tanh(d·S + k(v − Sᵀk)ᵀ)` combines a bounded tanh-on-state
> nonlinearity with a delta-correcting matrix write.
>
> First, **multi-programming generalizes.** The systems recipe used to
> train a 1.27B-parameter pure-recurrent NDM stack to convergence on a
> Pile-class corpus — recurrence kept serial along time, parallelism
> exposed across many small bounded memory programs (per-head,
> per-state-tile, per-batch), packaged into a fused Triton recurrence
> kernel with sparse checkpointing that runs identically on CUDA and
> ROCm — is not specific to NDM. The same multi-programmed shape makes
> other pure nonlinear matrix-state recurrences trainable at the same
> scale.
>
> Second, **the update rule matters.** On controlled S3/S5
> permutation-composition probes, the delta-correcting write `v − Sᵀk`
> separates NDM from FLA-GDN (linear-state), Mamba2 (linear-state), and
> from both published and CMA-ES-reshaped M2RNN variants (raw-write
> nonlinear-state matrix RNNs). A trusted Lean 4 core (no
> `sorry`/`axiom`/`opaque`/`native_decide`) proves a one-step
> update-family resource separation between the delta-correcting and
> raw-write families, an S5 tracker construction, and a finite-state
> ceiling for fixed-precision recurrent recognizers. The empirical
> result is that the delta-correcting family approaches the NC1 limit
> via S5 tracking, while raw-write and linear-scan families lag
> noticeably. *(TODO: tighten "approaches the NC1 limit via S5
> tracking" after `lean-formalization-gap-audit` resolves which
> portion of this claim is Lean-proved vs empirical.)*
>
> Third, **raw training speed is not the differentiator.** Under
> matched CMA-ES hyperparameter and shape search (N = 4 recurrent
> families), the FLOPs-per-bit learning rate converges — the choice of
> recurrent update changes *what the model can compute*, not how fast
> it trains in FLOPs. The contribution of this paper is therefore the
> architectural option, the systems recipe that makes it trainable at
> scale, and the mechanism that earns the expressivity — not a
> wallclock or FLOPs win over linear-state baselines.

---

## 2. Section List

Per task spec: at least 3 bullets of intended content per section. Each
bullet cites the upstream synthesis doc that supplies the content.

### 2.1 Introduction

- **Field assumption to reject.** Pure serial nonlinear RNNs are widely
  assumed to be "expressive but impractical at scale," and the field's
  response has been to move to linear-state recurrence (Mamba, RWKV,
  Gated DeltaNet, mLSTM) or to hybridize recurrence with attention. The
  survey in `docs/related_work_nonlinear_rnns.md` (entries 1–15) shows
  that, prior to the present work, every billion-scale recurrent
  language model in the published literature was either linear-state or
  hybrid.
- **Reframed question.** Did pure nonlinear recurrence fail because it
  was the wrong computation, or because it was parallelized along the
  wrong axis? This paper argues for the second: when recurrence is kept
  serial along time but exposes parallelism across many small bounded
  memory programs — what is called *multi-programming* here — pure
  nonlinear matrix-state recurrence becomes trainable at billion-parameter
  scale. The multi-programming recipe is general; it is not specific to
  the particular architecture documented in this paper.
- **Three contributions, organized as pillars.**
    1. *A multi-programming recipe for pure nonlinear recurrent LMs at
       scale.* A reference instantiation — the Nonlinear Delta Memory
       architecture, trained as a 1.27B-parameter pure-recurrent stack
       to convergence on a Pile-class corpus — demonstrates the recipe
       end-to-end. The recipe (per-head bounded memory programs, fused
       Triton recurrence kernel, sparse checkpointing, ScheduleFree-AdamW
       per-island with hierarchical local-SGD averaging) is described in
       enough detail to be applied to other pure nonlinear matrix-state
       recurrences. The Triton kernel runs identically on CUDA and ROCm.
    2. *Empirical and partially formal evidence that the update rule
       matters.* The delta-correcting matrix write `v − Sᵀk` separates
       NDM from raw-write matrix RNNs (the M2RNN family) and from
       linear-state baselines on S3/S5 permutation-composition probes
       and a six-task canonical state-tracking sweep. A trusted Lean 4
       core, with no `sorry`/`axiom`/`opaque`/`native_decide`, proves a
       one-step update-family resource separation between the
       delta-correcting and raw-write families and a finite-state
       ceiling for fixed-precision recurrent recognizers; an S5 tracker
       is also constructed in Lean. The empirical headline is that the
       delta-correcting family approaches the NC1 limit via S5
       tracking. *(TODO: tighten this wording after the
       `lean-formalization-gap-audit` task reports back; the present
       phrasing reflects the trusted Lean surface as of 2026-05-23
       together with empirical S5 separation, and does not claim a
       formal NC1 lower bound.)*
    3. *FLOPs-per-bit convergence under CMA-ES (N = 4).* When the four
       compared recurrent baselines (NDM, FLA-GDN, Mamba2, M2RNN-class)
       are each given matched CMA-ES hyperparameter and shape budgets,
       their FLOPs-per-bit learning rates collapse to nearly the same
       curve. This is reported in detail in
       `docs/CMA_FLOP_RATE_FINDING.md` (forthcoming, produced by
       `harvest-cma-flop-rate-finding`). It reframes the contribution:
       the architectural option and the mechanism are the load-bearing
       claims, not a raw-speed win.

  The formal core (Lean), the systems (Triton kernel), and the
  expressivity result are therefore *supporting evidence* under the
  three pillars rather than separate top-level contributions.
  Additional capability evidence from the over-training quiz/reasoning
  panel — showing the 1.27B pure-recurrent NDM stack gaining real
  capability on a QA + basic-reasoning suite over the course of
  training — is documented in `docs/QA_REASONING_PROGRESSION.md`
  (forthcoming, produced by `harvest-qa-reasoning-quiz-progression`).
- **Related work peers.** Two recent pure or near-pure nonlinear
  recurrent results are direct peers and deserve close treatment.
  M2RNN (`arXiv:2603.14360`, March 2026) trains a *pure-recurrent*
  nonlinear matrix-state RNN at 410M parameters on Nemotron-CC-v2;
  this paper is positioned as a peer demonstration of a related
  capability — pure nonlinear recurrence trained to useful loss on a
  large web corpus — at smaller scale and with a different
  (raw-write) update rule. xLSTM-1.3B
  (`arXiv:2405.04517`) is a nonlinear+linear *mixed* recurrent stack
  (7:1 mLSTM:sLSTM block ratio) trained on SlimPajama; it shares the
  scale band but is not pure nonlinear-state. Both are treated as
  related-work peers in §2.8, not as priority threats.

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
  criterion (`docs/related_work_nonlinear_rnns.md` lines 13–15). These are
  the comparison cohort for the FLOPs-per-bit convergence finding
  (Pillar 3) and for the state-tracking separation (Pillar 2).
- **The nonlinear-state cohort** — sLSTM (used in xLSTM-1.3B; nonlinear
  via memory mixing but mixed 7:1 with linear mLSTM blocks); **M2RNN**
  (`arXiv:2603.14360`) — pure-recurrent at 410M with a raw-write update
  `tanh(H·W + k vᵀ)`; Titans (hybrid, MLP memory); classical LSTM/GRU
  (no published model at ≥500M on a Pile-class corpus);
  `arXiv:2505.17852` 1B LSTM trained with zero-order optimization (not
  standard gradient training).
- **Peer treatment for pure / near-pure nonlinear recurrent results.**
  Two recent results deserve dedicated subsections in §2.8 rather than
  inline mention, and are positioned as **related-work peers**:
    - *M2RNN* (`arXiv:2603.14360`, March 2026) is a pure-recurrent
      nonlinear matrix-state RNN trained at 410M on Nemotron-CC-v2. It
      shares the high-level message that pure nonlinear recurrence can
      reach useful loss on a large web corpus, at a smaller scale, with
      a different (raw-write) update rule. The paper's relationship to
      M2RNN is comparative on the update-rule axis (Pillar 2: the
      delta-correcting write separates on state-tracking expressivity
      from the raw-write update) and on the systems axis (Pillar 1: the
      multi-programming recipe described here is general enough to apply
      to the M2RNN update family as well — the M2RNN-CMA variant in
      §2.5 is the concrete instantiation). No priority is claimed.
    - *xLSTM-1.3B* (`arXiv:2405.04517`) is a nonlinear+linear mixed
      recurrent stack with a 7:1 mLSTM:sLSTM block ratio (87.5% linear
      mLSTM blocks). It is included because it is the closest scale
      band, with an acknowledgment that it is not a pure nonlinear-state
      result.
- **Suggested framing line.** Replace any "first" / "priority" language
  with a peer framing such as: "Concurrent M2RNN
  (`arXiv:2603.14360`) demonstrates pure-recurrent nonlinear
  matrix-state language modeling at 410M with a raw-write update; the
  present work extends this line in three respects — scale (1.27B),
  update rule (delta-correcting), and a multi-programming recipe that
  is shown to be general across the nonlinear matrix-state family."

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
  published M2RNN paper-shape is a sympathetic re-implementation; the
  original authors may have shape or training details that this
  re-implementation does not match.
- **Loop contradictions still open** — output gate on/off, tanh vs linear
  state, simple vs Mamba2 decay (`docs/DESIGN_DOSSIER.md` §§6.1–6.4).
  Production keeps the more conservative settings; revalidation at
  1.27B is flagged as the highest-value early follow-up.

### 2.10 Conclusion

- **Pillar 1 restated.** Pure nonlinear recurrent language models are
  trainable at billion-parameter scale when the recurrent computation
  is shaped as a many-program GPU workload — parallelism is across
  heads, state-tiles, and batch, not along time. The recipe is general:
  any pure nonlinear matrix-state recurrence with a similar
  many-head / small-state-tile geometry is reachable under the same
  systems setup.
- **Pillar 2 restated.** The delta-correcting matrix write
  `v − Sᵀk` is the empirically separating mechanism: it pulls away
  from raw-write nonlinear matrix RNNs (M2RNN family) and from
  linear-state baselines on S3/S5 permutation composition and the
  six-task canonical state-tracking sweep, and it approaches the NC1
  limit via S5 tracking. Bounded nonlinearity *alone* is not
  sufficient; nor is matrix state alone. *(TODO: tighten the NC1
  wording after `lean-formalization-gap-audit` reports back.)*
- **Pillar 3 restated.** Under matched CMA-ES search, the FLOPs-per-bit
  learning rate of the four compared recurrent baselines converges.
  The paper does *not* claim a wallclock or FLOPs-per-bit advantage
  for any one recurrent family. The contribution is the architectural
  option and the mechanism, not training speed.
- **Explicit non-claims.** No priority claim on "first pure nonlinear
  recurrent LM at scale" (M2RNN trained a pure-recurrent variant
  concurrently at 410M and is treated here as a related-work peer);
  no formal NC1 lower bound for linear-scan models; no formal proof
  that a trained real-valued NDM exactly recovers the S5 lookup
  table. See §2.9.
- **Open horizon.** How far pure nonlinear recurrent reasoning can
  scale once memory, geometry, and systems are co-designed — and
  which other update rules earn the same expressivity separation
  under the same multi-programmed recipe.

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

## 4. Claim → Evidence Map (three pillars + supporting evidence)

### Pillar 1 — Multi-programming generalizes

| # | Sub-claim | Primary evidence file(s) | Lean witness if any |
|---|---|---|---|
| P1.a | A pure nonlinear matrix-state recurrent LM (NDM, 1.27B parameters) trains to convergence on a Pile-class corpus under the multi-programmed recipe | `paper/notes_reconciliation.md` C4 (currently **out-of-repo** — see Pending §5.2); qualitative production-loss claim via `docs/DESIGN_DOSSIER.md` §5 | — |
| P1.b | The recipe is realized by per-head bounded memory programs (370 heads × 32×32 state tiles, depth 12) packaged into a fused Triton kernel | `docs/DESIGN_DOSSIER.md` §§1, 4.1; source `ndm/triton/e88_triton_*.py`, `ndm/models/e88_fused.py:298-316` | `PROOF_INVENTORY.md` `ndm_1p27B_programs_per_batch_token_bs5` (= 22200), `ndm_1p27B_state_scalars_per_layer` |
| P1.c | The Triton kernel runs identically on CUDA and ROCm | `docs/DESIGN_DOSSIER.md` §4.3 (FRO); `paper/notes_reconciliation.md` S7 (ROCm parity test — Pending §5.2 #8) | — |
| P1.d | The recipe is described in enough detail to be applied to other pure nonlinear matrix-state recurrences (it is not specific to NDM) | §2.4 of this outline + `docs/DESIGN_DOSSIER.md` §§4.1, 4.4 | — |

### Pillar 2 — The update rule matters

| # | Sub-claim | Primary evidence file(s) | Lean witness if any |
|---|---|---|---|
| P2.a | The delta-correcting matrix write `v − Sᵀk` separates from raw-write matrix RNNs (M2RNN family) on S3/S5 permutation-composition probes at matched parameter count | `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a (S3/S5 numbers); `paper/ndmpapernotes.md` lines 153–173 | `PROOF_INVENTORY.md` `M2RNNComparison.write_rule_separates_m2rnn_and_e88`, `ndm_m2rnn_one_step_resource_separation`, `ndm_m2rnn_one_step_resource_separation_embeds` |
| P2.b | The same separation persists across the six-task canonical state-tracking sweep (parity, modular counter, FSM tracking, dyck, associative recall, selective copy) at 8M parameter-matched | `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §§1a, 2a, 3a, 4a, 6a, 7a | — |
| P2.c | The delta-correcting family **approaches the NC1 limit via S5 tracking** *(TODO: tighten this wording after `lean-formalization-gap-audit` reports back; current trusted Lean surface is the S5 tracker construction + finite-state ceiling, not a formal NC1 lower bound for linear-scan models)* | `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §5a; `formal/lean/PROOF_INVENTORY.md` §S5Witness, §S5Tracker, §S5NDMRealization | `PROOF_INVENTORY.md` `S5Witness.s5_state_count` (= 120), `s5_not_solvable`, `S5Tracker.recognizer_state_count`, `S5NDMRealization.s5_transition_key_count` (= 480) |
| P2.d | A finite-state ceiling on fixed-precision recurrent recognizers is formalized in Lean | `formal/lean/PROOF_INVENTORY.md` §Module `RecurrentResourceFormalism` | `fixed_precision_state_space_finite`, `exactTransitionMemory_run` |

### Pillar 3 — FLOPs-per-bit convergence under CMA-ES (N = 4)

| # | Sub-claim | Primary evidence file(s) | Lean witness if any |
|---|---|---|---|
| P3.a | Under matched CMA-ES hyperparameter and shape search, the four compared recurrent families (NDM, FLA-GDN, Mamba2, M2RNN-class) collapse to nearly the same FLOPs-per-bit learning rate | `docs/CMA_FLOP_RATE_FINDING.md` *(forthcoming; produced by `harvest-cma-flop-rate-finding`)* | — |
| P3.b | The paper does *not* claim a wallclock or FLOPs-per-bit advantage; the contribution is the architectural option + the mechanism | §1 Abstract; §2.10 Conclusion of this outline | — |

### Supporting capability evidence

| # | Sub-claim | Primary evidence file(s) |
|---|---|---|
| S.a | The 1.27B pure-recurrent NDM stack gains real QA / basic-reasoning capability over training (capability evidence beyond loss numbers) | `docs/QA_REASONING_PROGRESSION.md` *(forthcoming; produced by `harvest-qa-reasoning-quiz-progression`)* |
| S.b | Hybridization with linear-scan blocks (e.g. `[NDM, NDM, GDN, GDN]`) *degrades* state-tracking — purity is what gives the profile | `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` §§3a, 4a |

Notes on the map: Pillar 1's P1.a and Pillar 3 are **partial /
out-of-repo** as of 2026-05-24 — racer plots and CMA-ES artifacts live
in `~/elman/`; see "Pending Experimental Closure" §5.2 for the exact
file paths that must be staged. All Pillar 2 Lean witnesses are
**in-repo and trust-gated** per `formal/lean/PROOF_INVENTORY.md`,
subject to a conclusive `lake build`.

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

- The three pillars (multi-programming systems recipe + update-rule
  mechanism + FLOPs-per-bit-under-CMA finding) span systems and
  modeling — NeurIPS reviewers are the right audience for both, more
  so than ICLR (which leans modeling) or COLM (heavily LM-curated, but
  smaller systems audience).
- The Lean trusted core is an unusual but well-received contribution in
  NeurIPS — recent precedent in formal verification + ML at NeurIPS
  (verified ML, neuro-symbolic tracks).
- The peer pure-nonlinear and near-pure-nonlinear results (M2RNN at
  `arXiv:2603.14360` — March 2026; xLSTM-1.3B; RWKV-7) are
  NeurIPS-style work; reviewers will recognize the comparison and the
  pillar framing.

### Backup ordering

1. **NeurIPS 2026 main track** — preferred.
2. **ICLR 2027** — second choice if NeurIPS timing slips. The ICLR
   review cycle suits a paper that benefits from another round of
   empirical depth (additional CMA-ES seeds, Mamba2 baseline closure).
3. **COLM 2026** — viable for the LM-specific subset of the result, but
   the systems+Lean+expressivity span fits less well.
4. **arXiv preprint** — strongly advisable in advance of venue
   submission so the multi-programming recipe and update-rule
   separation are publicly readable while the venue review proceeds.
   The arXiv post is not a priority claim against M2RNN; it is simply
   the way work in this area becomes citeable.

### Timing constraints

- **NeurIPS 2026 abstract deadline** (typical): mid-May 2026. **Already
  past as of 2026-05-23.** If main track for 2026 is not feasible, the
  realistic targets are NeurIPS 2026 *Datasets and Benchmarks* (if
  framed around the S5 expressivity benchmark — late June typical),
  ICLR 2027, or NeurIPS 2027.
- **Pending items 5.2 #1, #5, #6** (S5 JSONs, Mamba2, racer plots) are
  the critical-path closures for Pillar 2 reproducibility and Pillar 1
  evidence. Item #1 is fastest (re-run an existing separation suite).
  Items #5 and #6 require the most external compute / `~/elman/`
  artifact migration. The CMA-ES FLOPs-per-bit panels (Pillar 3) wait
  on `harvest-cma-flop-rate-finding`.
- **Lean kernel build verification.** Per `formal/lean/PROOF_INVENTORY.md`
  §Run Result, `lake build` was not completed in the inventory
  session; before submission, the trust-gate must be run conclusively
  in an environment with `ripgrep` installed and Mathlib cached. The
  NC1 wording in §1 and the Pillar 2 P2.c row in §4 must also be
  reconciled with whatever `lean-formalization-gap-audit` concludes
  about the proved-vs-empirical boundary.

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
  context (linear vs nonlinear taxonomy), §6 (M2RNN/xLSTM peer
  treatment).
- `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` — §§2.6, Figure 1, Figure 5,
  Pending §5.2 items 1–4; Pillar 2 sub-claims P2.a, P2.b.
- `docs/DESIGN_DOSSIER.md` — §§2.3, 2.4, Figure 1A, Figure 2, Figure 4,
  §2.9 (contradiction list); Pillar 1 sub-claims P1.b–P1.d.
- `formal/lean/PROOF_INVENTORY.md` — §§2.7, Pillar 2 sub-claims P2.a,
  P2.c, P2.d, §2.9 (non-claims), Lean-trust-gate caveat in §6.
- `paper/notes_reconciliation.md` — §2.5 (gap list), §§4–5 (claim
  evidence rows and pending experiments), §6 (timing constraints).
- `paper/ndmpapernotes.md` — §§0 (title, thesis), 1 (abstract sketch),
  2 (every section header), 3 (figure list — Figures 1–4 originate
  here), 4 (original contributions list — now reorganized as three
  pillars in §4). *Preserved as historical; not reframed by this task.*
- `docs/CMA_FLOP_RATE_FINDING.md` — *forthcoming;* Pillar 3 sub-claims
  P3.a, P3.b.
- `docs/QA_REASONING_PROGRESSION.md` — *forthcoming;* supporting
  capability evidence row S.a.

---

## 8. Validation Self-Check (against task spec)

- [x] **All 5 upstream summary docs are explicitly read and cited; the
  two forthcoming reframe-companion docs are referenced as placeholders.**
  — §0 (Inputs read) + §7 cross-reference index.
- [x] **Each section has at least 3 bullet points of intended content.**
  — §2.1 (5), §2.2 (4), §2.3 (4), §2.4 (5), §2.5 (5), §2.6 (5), §2.7 (5),
  §2.8 (4), §2.9 (6), §2.10 (5).
- [x] **Every pillar sub-claim is mapped to a specific evidence file
  (no 'TBD' for items verifiable in-repo).** — §4 tables; Pillar 1
  P1.a and Pillar 3 explicitly marked partial/out-of-repo with the
  specific blocking dependencies.
- [x] **Pending-experiments list is concrete (file paths to result dirs
  that need to fill in).** — §5.2 lists nine items with exact result
  directories or scripts.
- [x] **At least 4 figures specified.** — Figures 1–4 mandatory; Figure 5
  optional.

### Reframe-task self-check (`reframe-paper-narrative`, 2026-05-24)

- [x] All "first" / "priority" / "before M2RNN" priority framing removed
  from §0, §1, §2.1, §2.8, §2.10, §6.
- [x] Three-pillar framing present in abstract (§1), contributions
  bullet (§2.1), claim→evidence map (§4), conclusion (§2.10).
- [x] M2RNN repositioned as related-work peer (§2.1 Related work peers
  bullet; §2.8 Peer treatment subsection); no "priority threat" or
  "closest comparable" framing remains.
- [x] CHANGELOG note present at top of this file with date + reason.
- [x] References to `docs/CMA_FLOP_RATE_FINDING.md` and
  `docs/QA_REASONING_PROGRESSION.md` included as placeholder
  references in §0 Inputs read, §2.1 contributions bullet (Pillar 3
  and supporting evidence row S.a), §4 evidence map (P3.a, S.a), and
  §7 cross-reference index.
- [x] NC1 wording held at "approaches the NC1 limit via S5 tracking"
  with explicit TODO markers in §0, §1, §2.1, §2.10, §4 (P2.c)
  pending `lean-formalization-gap-audit`.
- [x] Paper-facing rewritten sections (§0 title/thesis, §1 abstract,
  §2.1 introduction and contributions, §2.8 related-work peer
  framing, §2.10 conclusion) use no first-person plural and unpack
  internal codenames (NDM is introduced as the architecture's name;
  no bare E-series codenames appear in those sections).

---

*Outline only — paper prose is out of scope per task brief. Generated by
`paper-outline`; reframed 2026-05-24 by `reframe-paper-narrative`;
downstream consumer `direction-forward-memo` will integrate this with
the rest of the audit set.*
