# BL-1 Provenance Re-check — tracing the FINAL config, not the early ablation

**Task:** re-adjudicate-bl-1 (re-check of `paper/review/BL1_adjudication.md`)
**Trigger:** author's objection that the prior SUBSTANTIATED verdict rests on `docs/E88_ABLATION_NOTES.md`, an *earlier* experimentation phase, not the pass that produced the paper's reported 8 M (`tab_s5`) and 1.3 B numbers — and is therefore unsound.
**Mode:** READ-ONLY. `main.typ` and `EVALUATION.md` not edited.
**Confidence:** High (0.85). The decisive structural-config fact (tanh kept against an LM-loss tie on state-tracking grounds) is corroborated in **three independent records** — the ablation log, `docs/DESIGN_DOSSIER.md` §6.2, and the paper's own §9 — and the reported runs' config traces to baked-in code defaults, not to any notes file.

---

## VERDICT: STANDS

**One-line rationale:** The author is correct that `E88_ABLATION_NOTES.md` is an earlier, smaller-scale (~48–100 M) pass and that its *head-count* winner (12 heads) was superseded — but the load-bearing component BL-1 turns on (`tanh` state non-linearity) was carried *unchanged* into the reported 8 M and 1.3 B configs as a code default, and its retention against an LM-loss tie is documented on state-tracking grounds in the **production-level** record (§9:1644-1649 + DESIGN_DOSSIER §6.2), independent of the ablation notes. The factual claim ("Emender defaults selected partly on state-tracking, baselines on LM loss") holds for the *reported* config; it does not rest on a superseded record.

**Net effect on the prior adjudication:** conclusion unchanged; one provenance *citation* sharpened (see "Correction to the prior adjudication's wording" below). Not OVERTURNED, not DOWNGRADED.

---

## 1. The config of the REPORTED runs, traced to the generating run (not a notes file)

### 1a. Reported 8 M expressivity probes (`tab_s5` — the S₃/S₅ numbers)

The S₃/S₅ headline numbers (`paper/ndmpapernotes.md:158-167`; mirrored `EXPRESSIVITY_RESULTS_SUMMARY.md:218-225`) were produced by:

- **Run dir:** `experiments/expressivity_tasks/results/s5_witness_8m_20260521` (`EXPRESSIVITY_RESULTS_SUMMARY.md:209, 325`).
- **Runner:** `run_separation_suite.py --tasks s3_permutation s5_permutation` (`EXPRESSIVITY_RESULTS_SUMMARY.md:210, 325`).
- **Model preset:** `E88_8M` in `run_separation_suite.py:78-84` →
  `layer_pattern=["E88"]`, `dim=384`, `n_heads=32`, `n_state=32`, **`kwargs={}`** (no structural overrides).
- **Build path:** `train_hybrid.py` (`:122-127`) instantiates `HybridLadderLM` from `layer_pattern` and the CLI dim/heads/state; it passes **no** `linear_state` / `use_gate` / `use_conv` / `use_output_norm` (confirmed: those flags do not appear in `train_hybrid.py`). The structural architecture is therefore exactly the `E88FLAHybrid` constructor defaults.
- **Structural defaults baked in** — `ndm/models/e88_fla_hybrid.py:838-849`:
  `use_conv=False` ("E88 optimal"), `linear_state=False` (**tanh on**), `use_gate=False` ("E88 optimal: gating hurts E88"), `use_output_norm=False` ("E88 optimal"), `decay_mode='mamba'`.

**The config of the reported 8 M run is fixed by code defaults, not by a notes file.** The one BL-1-relevant component — `linear_state=False` (tanh) — is a constructor default, justified in-source as the nonlinear-state mechanism kept "for expressivity / UTM capability" (docstring `:823`; `DESIGN_DOSSIER.md:57-63`).

### 1b. Reported 1.3 B production runs (Figure 2 / racer panel)

- **Sources:** `paper/results/figure_2/SOURCES.md:15-27`, `paper/results/figure_2/raw/README.md:11-13`.
- **E88/NDM:** 1,273,191,856 params (~1.27 B), Pile convergence run `/tmp/pile_convergence_3arch/ctx2k/e88*.log`, launched by `~/elman/run_pile_convergence_3arch.sh` (`SOURCES.md:76`).
- **Geometry:** "370 heads of 32×32" at 1.27 B (`DESIGN_DOSSIER.md:70`) — i.e. `n_state=32` held, `n_heads` scaled with `dim` for the param budget per the **balanced-config rule** (`docs/E88_BALANCED_CONFIG_GUIDE.md:24, 57-64`), which is an **LM-loss / throughput** geometry-selection guide.
- **Structural flags:** same `E88FLAHybrid` "E88 optimal" defaults (tanh on, no conv/gate/norm, Mamba2 decay). §6 makes the inheritance explicit: the 8 M Emender "ran on the default configuration carried down from its 1.3 B production stack" (`main.typ:944-945`).

**Conclusion (1):** The reported 8 M and 1.3 B configs share one structural spine (the `E88FLAHybrid` "E88 optimal" defaults), differing only in *geometry* (heads/state scaled to the param budget). Neither traces to `E88_ABLATION_NOTES.md` as its authoritative selection record; both trace to the code defaults plus a geometry rule.

---

## 2. Is `E88_ABLATION_NOTES.md` the final lineage, or a superseded earlier pass?

**Finding: it is an earlier, smaller-scale pass — superseded for *geometry*, but NOT superseded as the origin of the *structural* defaults that the reported runs use.** The author is half right.

**Ordering evidence:**

1. **Scale.** The ablation rounds are at **48–100 M** params (Round 4: `E88d_h12 = 48M`, `E88c_nogate = 74M`, `E88d_h20 = 100M` — `E88_ABLATION_NOTES.md:90-99`). The reported probes are **8 M**; production is **1.27 B**. The ablation is neither the reported scale nor the production scale. This matches the author's "earlier phase to set up a later run."
2. **Geometry winner was discarded.** The ablation's headline geometry result is "**12 heads beats 16 heads**" (`E88_ABLATION_NOTES.md:94, 101-102`), best config `E88d_h12` (12 heads). The reported runs use **`n_heads=32`** (`run_separation_suite.py:80-83`; `run_canonical_sweep.py:50`; `EXPRESSIVITY_RESULTS_SUMMARY.md:37-38`) and production uses 370 heads. The ablation's 12-head conclusion was **not** carried forward → that part of the notes is genuinely superseded.
3. **The 8 M probe geometry was re-chosen on a *state-tracking* basis, not LM loss.** `run_separation_suite.py:7-11`: H/N "chosen to preserve the many-head E88 setting **where the earlier parity / FSM / mod-counter results grokked**." So even the geometry of the reported probes was fixed by state-tracking grokking, not by the ablation's `avg100` LM-loss ranking. (This *independently* reinforces BL-1's asymmetry without using the ablation notes at all.)
4. **Structural flags WERE carried forward.** The ablation's structural conclusions — drop conv, drop gate, drop output-RMSNorm, keep Mamba2 decay, keep tanh (`E88_ABLATION_NOTES.md:61-87`) — are exactly the `E88FLAHybrid` defaults used by the reported runs (`e88_fla_hybrid.py:838-849`). No *later* structural ablation re-derived or overturned them; `DESIGN_DOSSIER.md:310-318` notes conv/decay choices "may not have been re-run at production scale," i.e. the small-scale structural picks stand by inheritance.

**Note on dating:** git history is flattened to a single repo-creation commit (`cbcc726`, 2026-05-23), so commit timestamps cannot order the passes. Ordering above is established from in-content scale markers and run-dir date stamps (`*_20260511`, `s5_witness_8m_20260521`), which corroborate that the ablation (architecture selection) precedes the 8 M/1.3 B reporting runs.

---

## 3. If superseded: what was the FINAL config actually selected on?

Splitting by component, for the config that produced the reported numbers:

- **Geometry (heads/state):** LM loss + param-matching at 1.3 B (`E88_BALANCED_CONFIG_GUIDE.md`); for the 8 M probe, the "many-head setting where parity/FSM/mod-counter grokked" — a **state-tracking** basis (`run_separation_suite.py:7-11`).
- **Conv / gate / output-norm:** LM loss (`avg100`) at small scale (`E88_ABLATION_NOTES.md` Rounds 1-3), carried forward.
- **State non-linearity (`tanh`) — the BL-1 load-bearing flag:** **NOT selected on LM loss in any pass.** Every LM-loss ablation found tanh ≈ linear: Round 1 `E88a_linear` Δ=−0.004, Round 4 `E88d_linear` Δ=0.000 "Linear = Tanh!" (`E88_ABLATION_NOTES.md:15, 97`; `DESIGN_DOSSIER.md:294-296`). The decision to **keep** tanh rather than take the loss-equivalent (cheaper, simpler) linear state was made on **state-tracking / expressivity** grounds. This is stated three times, independently:
  - `E88_ABLATION_NOTES.md:74, 81` — "keep tanh for expressivity," "Tanh — computational expressivity (UTM capability)."
  - `DESIGN_DOSSIER.md:304-308` — "Production keeps `tanh`; the contradiction is between loss-only ablations (which say linear is fine) and expressivity arguments… The resolution proposed by MENU is to evaluate on **state-tracking / modular-counter / parity / FSM-tracking tasks, not only loss**." MENU itself: "earlier loss-only runs sometimes showed linear state close. **Expressivity tasks should decide this, not only Pile loss**" (`DESIGN_DOSSIER.md:299-302`).
  - `main.typ:1646-1649` (§9) — see §4.

So the final config's tanh was retained *because LM loss could not separate it and a state-tracking/expressivity rationale broke the tie in its favour.* That is precisely "selected partly on state-tracking." The baselines received no analogous treatment (GDN / M²RNN-paper = published LM-selected defaults; M²RNN-CMA = CMA on LM-loss/FLOP-rate — `main.typ:945-947`, `paper/results/cma_flop_rate/`). The asymmetry holds for the reported config.

---

## 4. Is §9:1644-1649 about the FINAL config, or stale text?

**It describes the FINAL / production config. It is not stale.**

`main.typ:1644-1649`: *"the output gate, the state non-linearity (tanh vs linear), and the decay parameterisation… **All three tie on loss at small scale; the production architecture keeps the conservative settings on the strength of state-tracking and stability data, not a clean ablation at 1.3 B**."*

- "tie on loss **at small scale**" = the 48–100 M ablation (`E88_ABLATION_NOTES.md` Round 1 & 4 ties). Correctly scoped as small-scale.
- "**the production architecture keeps** the conservative settings on the strength of **state-tracking and stability data**, not a clean ablation **at 1.3 B**" = an explicit statement about the *production / final* config. The phrase "at 1.3 B" pins it to the reported production stack, and §6:944-945 confirms the 8 M probe inherits that same stack.

This sentence is the paper's own, current description of how the **final** config's tanh/gate/decay were settled: LM loss was indifferent, so they were kept on state-tracking + stability. The prior adjudication's "smoking gun" label is therefore **sound** — it quotes the paper describing the *reported* config, not a stale earlier draft. The author's claim that §9:1644 is "stale earlier text" is not supported: it is correctly scoped (small-scale tie → production retention) and consistent with the code defaults and DESIGN_DOSSIER §6.2.

---

## 5. Why the author's objection does not overturn BL-1

The author's objection has a true premise and a false inference:

- **True:** `E88_ABLATION_NOTES.md` is an earlier, ~100 M-class pass, and its *geometry* winner (12 heads) was not used in the reported runs.
- **False inference:** "therefore the state-tracking-selection story belongs to a discarded config, and the reported config was selected on LM loss only."

The inference fails because the BL-1 load-bearing component (`tanh`) is **not** geometry — it is a structural flag that propagated *unchanged* into the reported 8 M and 1.3 B configs as a code default, and **no record shows it was ever re-selected on LM loss.** Every record that addresses tanh selection (the ablation, DESIGN_DOSSIER §6.2, §9) says LM loss was *indifferent* and tanh was kept on state-tracking/expressivity grounds. The author would need a record of a *later* pass selecting the final tanh on LM loss; no such record exists, and the production-level records say the opposite.

The prior verdict also never rested on the ablation notes alone — it cited three corroborating sources (ablation log, §9, model defaults). Even if `E88_ABLATION_NOTES.md` were struck entirely, §9:1644-1649 + `DESIGN_DOSSIER.md:304-308` + the `linear_state=False` code default still substantiate the claim for the reported config.

---

## 6. Correction to the prior adjudication's wording (precision, not reversal)

The prior adjudication (`BL1_adjudication.md`) cited the Round-4 `E88d_linear` tie as if `E88_ABLATION_NOTES.md` were "the selection history of the config used in the paper's reported runs." That is imprecise: the notes are a ~100 M-class pass and their *geometry* was superseded. The cleaner, scale-correct citation for the **reported config's** tanh provenance is:

> **§9:1644-1649 (production) + `DESIGN_DOSSIER.md:304-308` (production keeps tanh against loss-only ablations) + `e88_fla_hybrid.py:839` (`linear_state=False` default carried into every reported run).**

With `E88_ABLATION_NOTES.md` used as the *origin* of the structural defaults (small-scale tie that the production then resolved on state-tracking), not as the reported runs' authoritative selection log. This sharpening changes the *citation*, not the *verdict*.

---

## 7. Verdict summary

| Question | Finding |
|---|---|
| Reported 8 M config provenance | `s5_witness_8m_20260521` → `run_separation_suite.py:78-84` (`E88_8M`, H=32 N=32, kwargs={}) → `train_hybrid.py:122-127` → `e88_fla_hybrid.py:838-849` defaults (tanh on). Not a notes file. |
| Reported 1.3 B config provenance | `figure_2/SOURCES.md:15-27`; `~/elman/run_pile_convergence_3arch.sh`; 370×32×32 geometry (`DESIGN_DOSSIER:70`) + same structural defaults. |
| Is `E88_ABLATION_NOTES.md` final or superseded? | Earlier ~100 M pass. **Superseded for geometry** (12 heads → 32 / 370); **carried forward for structural flags** incl. tanh. |
| Final config's tanh selection criterion | LM loss was *indifferent* (tied); tanh kept on **state-tracking / expressivity** — corroborated in 3 independent records. Baselines: LM loss only. |
| §9:1644-1649 | Describes the **FINAL / production** config; correctly scoped; **not stale**. "Smoking gun" label sound. |
| **BL-1 underlying factual claim** | **STANDS.** Reported Emender config selected partly on state-tracking (tanh tie-break + 8 M head-geometry-to-preserve-grokking); baselines on LM loss. Disclosure/labeling blocker scope unchanged; S₃ + NC¹ mitigation unchanged. |

**No EVALUATION.md edit required** (consistent with the prior adjudication; the verdict is reaffirmed, not downgraded).

---

## Provenance index (files read for this re-check)

- `paper/review/BL1_adjudication.md` (prior verdict)
- `docs/E88_ABLATION_NOTES.md` (Rounds 1-4; scale 48–100 M; tanh tie-break :74,:81,:97; 12-head winner :94)
- `experiments/expressivity_tasks/run_separation_suite.py` (:7-11 geometry rationale; :78-84 `E88_8M` preset; :133-189 build_command)
- `experiments/expressivity_tasks/run_canonical_sweep.py` (:49-55 H=32 N=32)
- `experiments/expressivity_tasks/train_hybrid.py` (:122-127 build path; no structural overrides)
- `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`; `docs/EXPRESSIVITY_RESULTS_SUMMARY.md` (:37-39, :206-225, :318-330 run-dir table)
- `ndm/models/e88_fla_hybrid.py` (:823 docstring; :838-849 "E88 optimal" defaults, `linear_state=False`)
- `docs/DESIGN_DOSSIER.md` (:57-63 tanh as UTM lever; :70 370-head production; :292-318 §6.2/§6.3 tanh & conv "kept against loss-only ablations on expressivity grounds")
- `docs/E88_BALANCED_CONFIG_GUIDE.md` (geometry = LM-loss/throughput rule)
- `paper/main.typ` §6 (:929-958, esp. :944-945 "carried down from 1.3 B production stack"), §9 (:1600-1615, :1642-1649)
- `paper/ndmpapernotes.md` (:158-176 S₃/S₅ reported numbers)
- `paper/results/figure_2/SOURCES.md` (:15-27, :76-78), `paper/results/figure_2/raw/README.md` (:11-29)
