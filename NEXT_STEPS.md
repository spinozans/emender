# NDM — Direction-Forward Memo

**Audience:** team lead.
**Generated:** 2026-05-23, task `direction-forward-memo`.
**Inputs (cited inline):**

- `docs/MODEL_ZOO.md` (audit-model-zoo) — 119-file architecture-search lineage.
- `docs/STANDALONE_USAGE.md` + `tests/test_standalone_minimal.py` (verify-standalone-import) — pip-installable verification.
- `docs/HUGGINGFACE_RELEASE.md` + `docs/MODEL_CARD_TEMPLATE.md` (huggingface-release-plan) — 8-phase release checklist.
- `paper/OUTLINE.md` (paper-outline) — section structure, claim→evidence map, 9 pending experiments.

---

## 1. State of the Project

We are **closer to publishable** than to convergence — the architectural and
formal story is well-formed and reproducible from this repo, but the
empirical centerpiece (1.27B language modeling) still lives in `~/elman/` and
has not been staged. The audit wave produced four legible artifacts that
together cover lineage (`MODEL_ZOO.md`: 119 files classified, E0→E88 critical
path documented), standalone usability (`STANDALONE_USAGE.md`: `pip install
ndm` works on CPU for both LadderLM and E88FusedLM), release mechanics
(`HUGGINGFACE_RELEASE.md`: 8-phase checklist with `trust_remote_code` path
chosen), and paper scaffolding (`paper/OUTLINE.md`: 10 sections, 4 mandatory
figures, contributions C1–C6 mapped to evidence). What is **solid**:
expressivity headline numbers (E88 0.79 vs FLA-GDN 0.36 vs M2RNN 0.22 on S5
at 8M parameter-matched, per `paper/OUTLINE.md` §2.6), Lean theorem set
covering update-family separation and S5 tracker (`paper/OUTLINE.md` §2.7),
and the E88 component ablation roll-up (`docs/MODEL_ZOO.md` §"E88 Component
Ablation Summary"). What is still **soft**: Figure 3 1.27B racer plots
(`paper/OUTLINE.md` §5.2 #6, out-of-repo), Mamba2 row on S3/S5 (§5.2 #5,
missing), CMA-ES search artifacts (§5.2 #9, out-of-repo), and the Lean
trust-gate which currently passes vacuously because `rg` is missing in the
inventory environment (`paper/OUTLINE.md` §5.1 / §6). The **highest-leverage
move right now** is staging `~/elman/` into `paper/results/` so the paper's
critical-path blocker (Figure 3) is unblocked in parallel with the HF release.

---

## 2. Top 5 Next Workstreams (prioritized)

Effort key: **S** ≤ 1 day, **M** 2–5 days, **L** > 1 week.

### W1 — Stage `~/elman/` racer artifacts into `paper/results/figure_3/`
- **Why it matters:** `paper/OUTLINE.md` §5.2 #6 names this as a
  critical-path blocker for paper submission. Contribution C4 ("1.27B pure
  NDM in same wallclock loss regime as linear-recurrent baselines") has
  **no in-repo evidence today** (`paper/OUTLINE.md` §4 Claim→Evidence row
  C4 marked "currently out-of-repo"). Without this, the paper either ships
  with an arXiv-only Figure 3 or is non-submittable.
- **Unblocks:** Figure 3, contribution C4, HF Phase 1.1 (checkpoint
  identification, `docs/HUGGINGFACE_RELEASE.md` Phase 1), and the §5.2 #9
  CMA-ES artifact migration (they share the same `~/elman/` source tree).
- **Effort:** **M.** Mostly artifact migration + smoothing scripts (5K/10K/50K
  windows); no new training runs required.

### W2 — Re-run S5 separation suite and commit JSON artifacts
- **Why it matters:** `paper/OUTLINE.md` §5.2 #1 is the *fastest* pending
  closure — the suite (`experiments/expressivity_tasks/run_separation_suite.py
  --tasks s3_permutation s5_permutation --use_triton_e88`) already exists;
  only the JSONs are missing. Figure 1 Panels B/C currently rest on numbers
  hand-transcribed into `paper/ndmpapernotes.md` lines 153–173 — reviewers
  will (correctly) treat this as not reproducible from-repo.
- **Unblocks:** Figure 1 reproducibility, contribution C5 evidence
  hardening, and lowers the bar for §6 (Expressivity Results) being writable
  today.
- **Effort:** **S.** Single re-run of an existing script, ~hours, then
  commit `experiments/expressivity_tasks/results/s5_witness_8m_20260521/`.

### W3 — Select 1.27B checkpoint and execute HF release Phases 1–3
- **Why it matters:** `docs/HUGGINGFACE_RELEASE.md` Phases 1–3 (checkpoint
  selection → package freeze → weights conversion to safetensors) gate the
  public release. The HF release is the "scale-up priority" lock against
  concurrent M2RNN (`docs/HUGGINGFACE_RELEASE.md` Phase 1.4 records the
  provenance, `paper/OUTLINE.md` §6 "Priority date risk" calls this out
  explicitly: M2RNN at arXiv:2603.14360 is March 2026, pure-recurrent at
  410M — NDM must post first at 1.27B).
- **Unblocks:** HF Phases 4–8 (wrapper, model card, upload, smoke, link),
  the arXiv v1 priority post, and the paper's "first ≥1B pure nonlinear
  recurrent LM" framing (`docs/related_work_nonlinear_rnns.md` per
  `paper/OUTLINE.md` §2.8). Also forces a fix for the
  `pyproject.toml`/`__init__.py` version mismatch (`docs/STANDALONE_USAGE.md`
  §Known issues: 0.1.0 vs 0.2.0).
- **Effort:** **L.** Includes the version-bump fix, weights conversion,
  round-trip safetensors verification, and ndm-commit-hash provenance
  recording. The wrapper writing (Phase 4) is separable and can be
  parallelized.

### W4 — Add Mamba2 S3/S5 expressivity row
- **Why it matters:** `paper/OUTLINE.md` §5.2 #5 — Mamba2 is named in the
  README as a key baseline (`docs/MODEL_ZOO.md` lists `mamba2_baseline.py`
  with "primary selective-SSM baseline"), but in-repo S3/S5 numbers cover
  only E88, FLA-GDN, M2RNN, M2RNN-paper. A reviewer asking "where is Mamba2
  on the headline expressivity table?" is the most predictable rejection
  vector for §6. Adding the row strengthens contribution C5 ("Expressivity:
  S3/S5 separates NDM from FLA-GDN, Mamba2, M2RNN") which currently has the
  Mamba2 column literally missing.
- **Unblocks:** Figure 1 Panel B completeness; contribution C5 honesty.
- **Effort:** **M.** Requires standing up an 8M parameter-matched Mamba2
  config + 3 seeds × {S3, S5} × {128, 256, 512, 1024} = 24 runs; small but
  not trivial.

### W5 — Resolve Lean trust-gate environment and run conclusive `lake build`
- **Why it matters:** `paper/OUTLINE.md` §5.1 / §6 — the trust gate
  (`scripts/lean_trust_gate.sh`) currently passes vacuously because the
  inventory environment lacks `rg`, and `lake build` was not completed in
  the inventory session. The Lean contribution (C6) is one of the paper's
  more unusual claims — submitting without a conclusive build trace is a
  reputational risk if a reviewer attempts to reproduce.
- **Unblocks:** Contribution C6 reproducibility; the "no sorry/axiom/opaque/
  native_decide" claim in §7.
- **Effort:** **S.** Install `ripgrep`, cache Mathlib, `lake build`,
  archive log. ~half-day.

---

## 3. Critical Path to Paper Submission

Ordered milestones; each `→` is a hard dependency.

1. **W2** Re-run S5 separation suite, commit JSONs → `experiments/expressivity_tasks/results/s5_witness_8m_20260521/` (`paper/OUTLINE.md` §5.2 #1).
2. **W4** Mamba2 S3/S5 row at 8M, 3 seeds, T∈{128,256,512,1024} (`paper/OUTLINE.md` §5.2 #5).
3. **W1** Stage `~/elman/` racer artifacts → `paper/results/figure_3/`; produce smoothed loss-vs-wallclock for E88, FLA-GDN, Mamba2, M2RNN-CMA at frozen checkpoint hashes (`paper/OUTLINE.md` §5.2 #6).
4. **W5** Lean trust-gate `lake build` conclusive verification (`paper/OUTLINE.md` §5.1 caveat).
5. **W3a** Pick 1.27B checkpoint from `~/elman/`; record commit hash + step in `provenance/checkpoint_anchors.txt` (`docs/HUGGINGFACE_RELEASE.md` Phases 1.1–1.4).
6. **W3b** Fix `pyproject.toml` version → `0.2.0`; cut `v0.2.0` tag; build sdist/wheel; smoke install (`docs/STANDALONE_USAGE.md` known-issue + `docs/HUGGINGFACE_RELEASE.md` Phase 2).
7. **W3c** Convert checkpoint weights → safetensors; round-trip forward-pass verification (`docs/HUGGINGFACE_RELEASE.md` Phase 3).
8. **W3d** Write `modeling_ndm.py`, `configuration_ndm.py`, `config.json`; fill `MODEL_CARD_TEMPLATE.md` (`docs/HUGGINGFACE_RELEASE.md` Phases 4–5).
9. **W3e** Create private `poietic-pbc` HF repos; upload; clean-venv private
   smoke test; tag the smoke-tested commits as `v0.1`
   (`docs/HUGGINGFACE_RELEASE.md` Phases 6–7).
10. **Paper draft v1** integrating §§3, 5, 6, 7, 8, 9 — fully writable from in-repo evidence (`paper/OUTLINE.md` §5.1); §4 + Figure 3 fill in from milestone 3.
11. **arXiv v1 posted** — locks the "first ≥1B pure nonlinear recurrent LM" priority date against M2RNN concurrent work (`paper/OUTLINE.md` §6 "Priority date risk").
12. **HF link-back commit** (`docs/HUGGINGFACE_RELEASE.md` Phase 8); paper revision references the public checkpoint.

Steps 1–2 are parallel (S5 re-run is hours; Mamba2 is 2–5 days). Steps 3
and 4 are parallel with 1–2. Steps 5–9 serialize. Steps 10–12 follow.

---

## 4. Risks

- **Priority-date risk from concurrent M2RNN (`arXiv:2603.14360`).**
  `paper/OUTLINE.md` §6 calls this out: M2RNN published March 2026 at 410M
  pure-recurrent. If NDM does not post arXiv v1 before any M2RNN scale-up
  follow-up, the "first ≥1B pure nonlinear recurrent LM" framing
  (`docs/related_work_nonlinear_rnns.md` lines 252–260 per `paper/OUTLINE.md`
  §2.8) becomes contested. **Mitigation:** post arXiv v1 as soon as
  Figure 3 is staged, ahead of any venue deadline.

- **Lean trust-gate could fail under conclusive build.** `paper/OUTLINE.md`
  §5.1 notes the gate currently passes vacuously (`rg` missing). If `lake
  build` reveals a `sorry`/`axiom`/`opaque`/`native_decide` that the script
  missed, contribution C6's "trusted Lean 4 core" claim weakens or breaks.
  **Mitigation:** run W5 early; if it surfaces problems, scope C6 narrower
  before the paper draft is locked.

- **`~/elman/` checkpoint may not load cleanly into the standalone package.**
  `docs/STANDALONE_USAGE.md` documents a version mismatch (pyproject 0.1.0 vs
  `__init__` 0.2.0), an `mamba_ssm` interaction that breaks CPU `LadderLM`,
  and an `E88FusedLM` CUDA path that needs `hasty_pytorch_lib`. The HF
  release Phase 3 round-trip verification could fail if checkpoint
  hyperparameters drifted from any of these constraints. **Mitigation:**
  Phase 3.3 round-trip is the right gate; do it before Phase 4 wrapper
  work.

- **Six ambiguous-provenance files in the model zoo** (`docs/MODEL_ZOO.md`
  §"Files Needing Human Review"): `e32_no_presilu.py`, `e33_self_gate.py`,
  `e71_delta.py` vs `e71_matrix_gated.py` (E71 naming conflict),
  `e74_ablations.py` (multi-dim ablation harness; sub-configs not
  recorded), `circulant_elman.py` vs `diagonal_elman.py` (both labeled E6),
  `gated_delta_net.py` vs `fla_gated_delta.py` (canonical baseline
  ambiguity). The paper's §2.3 (Architecture) and §2.6 (Expressivity
  results) reference specific E-series files; if any is mis-labeled the
  ablation lineage in `paper/OUTLINE.md` §2.3 has a hole.
  **Mitigation:** resolve before Figure 1A schematic is rendered.

- **Hybrid AABB ablation degrades on state-tracking.**
  `paper/OUTLINE.md` §2.6 reports `[E88,E88,GDN,GDN]` *underperforming*
  pure E88 (modular counter hybrid 0.54 vs E88 0.90; FSM tracking hybrid
  0.71 vs E88 1.00). This is a clean mechanism story for §6 / Figure 5 but
  also a reviewer-attractive ablation — a hostile reviewer can ask
  "is the result hybrid-fragile? what if you mixed differently (ABAB,
  BABA, …)?" and the paper does not yet have those rows. **Mitigation:**
  flag in §2.9 limitations or pre-empt with one additional hybrid
  configuration before submission.

- **E89 residual-state failure documented in zoo** (`docs/MODEL_ZOO.md`
  Phase 6 row: E89 "state grows ~1 per step unboundedly; E88 achieves 1.81
  vs E89 plateaus at 2.65"). This is a post-E88 *negative* result. If a
  reviewer asks for further post-E88 ablation discussion, E89's failure
  mode (residual-state unboundedness) is informative — but if it is not
  framed deliberately, it can read as fragility.

- **NeurIPS 2026 main-track deadline is already past** (mid-May 2026 per
  `paper/OUTLINE.md` §6). Realistic 2026 venue = NeurIPS Datasets &
  Benchmarks (S5 expressivity framing, late June typical) or ICLR 2027.
  This is not a derailment risk, but the schedule in
  `paper/OUTLINE.md` §6 provisional calendar must be re-anchored to
  whichever venue is chosen.

---

## 5. Recommended Task Graph (for human approval)

These are **proposed**, not created. Listed in dependency order; titles
are workgraph-ready phrasing.

1. **`stage-elman-racer-artifacts`** — Migrate `~/elman/` loss curves +
   checkpoint metadata into `paper/results/figure_3/`. Produce smoothed
   (5K/10K/50K window) loss-vs-wallclock series for E88/NDM, FLA-GDN,
   Mamba2, M2RNN-CMA, M2RNN-paper. Closes `paper/OUTLINE.md` §5.2 #6.
   *(Effort: M. Blocks: Figure 3, contribution C4, arXiv v1.)*

2. **`rerun-s5-separation-suite`** — Run
   `experiments/expressivity_tasks/run_separation_suite.py --tasks
   s3_permutation s5_permutation --use_triton_e88` (3 seeds, T∈{128, 256,
   512, 1024}); commit JSONs to
   `experiments/expressivity_tasks/results/s5_witness_8m_20260521/`.
   Closes `paper/OUTLINE.md` §5.2 #1. *(Effort: S. Blocks: Figure 1
   reproducibility, C5.)*

3. **`add-mamba2-s3s5-baseline`** — Stand up 8M parameter-matched Mamba2
   config, run 3 seeds × {S3, S5} × {128, 256, 512, 1024}, commit JSONs
   alongside #2 outputs. Closes `paper/OUTLINE.md` §5.2 #5. *(Effort: M.
   Blocks: Figure 1 Panel B completeness; contribution C5 honesty.)*

4. **`fix-package-version-mismatch`** — Update `pyproject.toml` version
   to `0.2.0` to match `ndm/__init__.py`; tag `v0.2.0`; rebuild sdist +
   wheel; smoke-install in a clean venv. Closes
   `docs/STANDALONE_USAGE.md` known-issue #1. *(Effort: S. Blocks: HF
   Phase 2; required before HF upload.)*

5. **`resolve-model-zoo-review-items`** — Disambiguate the 6 files in
   `docs/MODEL_ZOO.md` §"Files Needing Human Review" (E1/E32/E33
   docstring copy-paste, E71 naming, E74 ablation sub-configs, E6
   labeling conflict, GDN baseline canonical entry-point). Append
   resolutions to `MODEL_ZOO.md`. *(Effort: S. Blocks: §2.3 Architecture
   prose; Figure 1A schematic accuracy.)*

6. **`lean-trust-gate-conclusive-build`** — Set up an environment with
   `ripgrep` and a cached Mathlib; run `scripts/lean_trust_gate.sh` then
   `lake build` end-to-end on the `PaperCore` closure; archive the build
   log under `formal/lean/build_logs/`. Closes `paper/OUTLINE.md` §5.1 /
   §6 caveat. *(Effort: S. Blocks: contribution C6 reproducibility.)*

7. **`select-and-anchor-1p27b-checkpoint`** — Identify the canonical
   1.27B run in `~/elman/`, record `(run_dir, config.json, step, ndm
   commit hash, tokenizer)` in `provenance/checkpoint_anchors.txt`.
   Closes `docs/HUGGINGFACE_RELEASE.md` Phases 1.1–1.4. *(Effort: S.
   Blocks: weights conversion; HF Phase 3.)*

8. **`convert-checkpoint-to-safetensors`** — Depends on #4 + #7. Load
   into `E88FusedLM`, export via `safetensors.torch.save_file`, run
   round-trip forward-pass verification (`docs/HUGGINGFACE_RELEASE.md`
   Phase 3). *(Effort: M. Blocks: HF wrapper; HF Phase 4.)*

9. **`write-hf-wrapper-and-config`** — Depends on #8. Implement
   `configuration_ndm.py` (`NdmConfig(PretrainedConfig)`),
   `modeling_ndm.py` (`NdmForCausalLM(PreTrainedModel)`), `config.json`,
   tokenizer files; clean-venv `trust_remote_code` load test
   (`docs/HUGGINGFACE_RELEASE.md` Phases 4 + 7). *(Effort: M. Blocks: HF
   upload.)*

10. **`hf-upload-and-link-back`** — Depends on #9. Create private
    `poietic-pbc` model repositories; upload; tag smoke-tested commits as
    `v0.1`; smoke-test from clean env on CPU and CUDA; commit HF badge +
    checkpoint URL to `README.md` and `provenance/checkpoint_anchors.txt`.
    Set topics and public visibility only after explicit approval is logged
    (`docs/HUGGINGFACE_RELEASE.md` Phases 6, 7, 8). *(Effort: M. Blocks:
    arXiv v1 reference to public checkpoint.)*

11. **`arxiv-priority-post`** — Depends on #1, #2, #3, #6, and #10
    (HF link is ideal but not strictly required if §4 / Figure 3 is in
    place). Compile paper v1 from `paper/OUTLINE.md` skeleton; submit
    arXiv to lock priority date against M2RNN
    (`paper/OUTLINE.md` §6). *(Effort: L. Blocks: NeurIPS D&B / ICLR 2027
    submission.)*

12. **(Optional) `hybrid-permutation-followup`** — Add ABAB / BABA
    hybrid layouts beyond the AABB variant in
    `paper/OUTLINE.md` §2.6 to pre-empt the "is the result
    hybrid-fragile?" reviewer ask. *(Effort: M. Risk-mitigation only;
    not a critical-path item.)*

---

## 6. One-Paragraph Bottom Line

The audit wave delivered exactly what was needed to make the paper
writable: lineage is legible (`docs/MODEL_ZOO.md`), the package is
installable (`docs/STANDALONE_USAGE.md`), the release path is concrete
(`docs/HUGGINGFACE_RELEASE.md`), and the paper's claims are mapped to
specific evidence files (`paper/OUTLINE.md` §4). The remaining work is
**migration and verification**, not new research: stage `~/elman/`
artifacts, re-run one expressivity script, add one missing baseline row,
run one conclusive Lean build, and execute the 8-phase HF checklist. The
single highest-leverage move is **W1 / Task #1 (stage `~/elman/` racer
artifacts)** because it unblocks Figure 3, contribution C4, and HF
checkpoint selection in one step. The single largest *external* risk is
M2RNN priority — arXiv v1 should post as soon as Figure 3 is staged,
without waiting for the NeurIPS cycle.

---

*Memo only. Per task brief, no workgraph tasks were created — Section 5
recommendations are for the team lead to review and approve.*
