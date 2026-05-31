# Adjudication — BL-1: Is the 8 M selection-history confound real?

**Task:** adjudicate-bl-1-is
**Source finding:** `paper/review/EVALUATION.md` BL-1 (← `rigor.md` OC-2 / Flag 4)
**Mode:** READ-ONLY analysis against the actual experimental record. `main.typ` and `EVALUATION.md` were not edited.
**Adjudicator confidence:** High (0.85). The decisive facts are corroborated independently in three places (the ablation log, the paper's own §9, and the model source), not inferred.

---

## VERDICT: SUBSTANTIATED

**One-line rationale:** The selection-history asymmetry is real and documented — the Emender's load-bearing state nonlinearity (`tanh`) was retained on *state-tracking / expressivity* grounds when it tied the alternative on LM loss, while every baseline's config was selected only on LM loss — so §6's bare "matched no-tuning across architectures" label does conceal an asymmetry favoring the Emender. **It is correctly classed as a *disclosure / labeling* blocker, not a results-invalidity blocker:** the §3 S₃ solvable-group control plus the §7 NC¹ theorem jointly neutralize the confound for the *direction/mechanism* of Claim 2, leaving only the *magnitude* of the within-class S₅ gap partly confounded. No downgrade of BL-1 is warranted; the existing **Fix** line already prescribes the correct remedy (co-locate the §9 caveat into §6, relabel, foreground S₃).

The author's rebuttal is **partly right and partly wrong** (detail in Q1/Q4 below): right that no probe-specific HPO was run for any arm and that training/compute budget is unchanged; wrong to dismiss it as neutral "method maturation," because the ablation record shows architectural ties were explicitly broken on a state-tracking proxy — selection on (a proxy of) the eval metric, which is exactly the confound, not neutral maturation.

---

## Q1 — What signal was the 8 M Emender default actually selected on?

**Mostly LM loss; but at least one load-bearing component was tie-broken on a state-tracking proxy. The §9 phrase "selected partly on state-tracking behaviour" is factually accurate.**

Evidence trail:

1. **The ablation lineage ranked on LM loss (`avg100`, nats).** `docs/E88_ABLATION_NOTES.md` Rounds 1–4 each rank variants by `avg100` (e.g. line 8 `avg100=1.795`, line 90 `avg100=1.709`; "Total improvement: ~0.10 nats", line 78). `avg100` is an average language-modeling loss in nats, *not* a state-tracking score. So the bulk of the config (drop conv, drop output-RMSNorm, drop gate, 12 heads) was selected on **LM loss**.

2. **The `tanh` nonlinearity is the exception — kept on state-tracking grounds when LM loss was indifferent.** Round 4 found `E88d_linear  1.709  Δ 0.000  "Linear = Tanh!"` (line 97) — i.e. on LM loss, tanh vs linear is a *dead tie*. Yet the recorded best config keeps `linear_state=False` with the explicit annotation **"keep tanh for expressivity"** (line 74) and **"✅ Tanh — computational expressivity (UTM capability)"** (line 81). The tie was broken in favor of the state-tracking/expressivity story, not on loss.

3. **The paper itself admits this — twice.**
   - §9 selection-history paragraph (`main.typ:1604–1606`): *"the Emender's defaults are the endpoint of an ablation lineage selected partly on state-tracking behaviour (Appendix), whereas GDN and M²RNN's published defaults were selected by their authors on language-modelling loss."*
   - §9 "Open architectural choices" (`main.typ:1644–1649`, the smoking gun): *"the output gate, the state non-linearity (tanh vs linear), and the decay parameterisation … **All three tie on loss at small scale; the production architecture keeps the conservative settings on the strength of state-tracking and stability data, not a clean ablation at 1.3 B.**"*

4. **The 8 M probe inherits exactly this config.** §6 (`main.typ:944–945`): the Emender "ran on the default configuration carried down from its 1.3 B production stack." The model source bakes the ablation endpoint in as defaults — `ndm/models/e88_fla_hybrid.py:838–849`: `use_conv=False` ("E88 optimal"), `use_gate=False` ("E88 optimal: gating hurts E88"), `use_output_norm=False` ("E88 optimal"), and **`linear_state=False`** (tanh on). `tanh` is the literal mechanism for nonlinear state tracking and is directly load-bearing for the S₅/S₃ result — so the one component chosen on a state-tracking proxy is precisely the one that drives the headline gap.

**Conclusion (Q1):** Selection was *predominantly* on LM loss, but a state-tracking proxy was used to break the tanh tie — and tanh is the load-bearing ingredient for the probe. "Selected partly on state-tracking behaviour" is true, by the paper's own admission and the ablation log.

---

## Q2 — How were the baseline (GDN, M²RNN) 8 M configs chosen?

**Published defaults for GDN and M²RNN-paper; a *locally* CMA-tuned reshape for M²RNN-CMA — but every baseline selection used LM loss / FLOP-rate, never a state-tracking criterion.** This confirms the asymmetry's direction.

- **GDN:** "ran on their respective published defaults" (`main.typ:947`), with `dim=640` only to match parameter count (`main.typ:931`). No local tuning of the update rule. All arms then run at the *same* probe geometry `dim=384, depth=4, H=32, N=32` (`run_canonical_sweep.py:50–51`; `main.typ:930`).
- **M²RNN-paper:** the authors' published shape. (At 1.3 B it was abandoned un-converged — `paper/results/figure_2/raw/README.md`: `m2rnn_paper.log … steps 50–8,400, NOT converged`.)
- **M²RNN-CMA:** "the analogous default from its CMA-tuned reshape" (`main.typ:946`). This baseline *did* receive local search — CMA-ES over head allocation/shape — so it is **not** a pure published default. But that CMA search optimized **LM loss / FLOP-rate** (`paper/results/cma_flop_rate/`, `figure_2/M2RNN_CMA.csv`), not state-tracking. So even the one locally tuned baseline was tuned on the LM objective, never on the probe metric.

**Conclusion (Q2):** No baseline's selection used a state-tracking signal. The Emender's did (Q1). The asymmetry the reviewer alleges — "co-evolved on state-tracking" vs "selected on LM loss" — holds in direction. The author's "best published architectures, aggressively HPO'd by their authors" is true (and indeed biases the *probe-effort* axis against the Emender), but it addresses a *different* axis (probe-specific tuning effort) than the one BL-1 flags (alignment of the *selection criterion* with the eval metric).

---

## Q3 — Does the S₃ solvable-group control neutralize the confound?

**Partially — exactly as the reviewer characterizes it. It neutralizes the *direction/mechanism* of the claim, not the *magnitude*.**

Data (`tab_s5`, `main.typ:977–981`; source `paper/ndmpapernotes.md:153–173` / `figure_2/raw/README.md`):

| Model | S₃ (T=128) | S₅ (T=128) |
|---|---|---|
| Emender | 1.0000 | 0.7918 |
| GDN | 0.7185 | 0.3552 |
| M²RNN-CMA | **0.3124** | 0.2157 |
| M²RNN-paper | 0.3773 | 0.1698 |
| random | 0.1667 | 0.0083 |

- **For M²RNN (raw-write nonlinear matrix RNN):** S₃ is a *solvable* group (lives in TC⁰), and capacity is non-binding (8 M params ≫ the 2.6-bit floor by ~6 orders of magnitude, `main.typ:932–936`). M²RNN stalls at 0.31–0.38 on S₃. Since neither non-solvability nor capacity can explain that failure, it must be an **inductive-bias deficiency of the raw-write update** — something no LM-loss-based tuning advantage for the Emender could manufacture. This genuinely **neutralizes the confound for the mechanism claim** (raw-write alone is insufficient for prefix tracking). This is the part the paper correctly calls "immune to any selection-history asymmetry" (`main.typ:955–958`, `1609–1611`).
- **For GDN (linear recurrence):** GDN *passes* the solvable control (0.72 on S₃) and fails only S₅ (0.36). So S₃ does **not** do the neutralizing work for GDN — the **§7 NC¹ theorem** does: a linear recurrence provably cannot express the NC¹-complete S₅ word problem regardless of tuning (`main.typ:675`). No selection advantage is needed to explain the GDN-on-S₅ gap.
- **What S₃ does NOT neutralize:** the *size* of the Emender's S₅ lead (0.79 vs 0.22). S₃ shows the baselines fail for tuning-independent reasons; it does not prove the Emender's *own* number is tuning-free. Because tanh — kept on a state-tracking proxy (Q1) — is load-bearing, part of the 0.79 could reflect that selection. Hence "reduces but does not eliminate."

**Conclusion (Q3):** S₃ (for M²RNN) + the NC¹ theorem (for GDN) jointly de-load the confound for the **direction and mechanism** of Claim 2. The **magnitude** of the within-class Emender-vs-M²RNN S₅ gap remains partly confounded. This is *partial* mitigation — the reviewer's own wording.

---

## Q4 — Verdict and evidence summary

**BL-1 is SUBSTANTIATED**, scoped precisely as a **disclosure / labeling blocker**:

1. **The confound exists** — corroborated by the ablation log (`E88_ABLATION_NOTES.md:74,81,97`), the model defaults (`e88_fla_hybrid.py:838–849`), and the paper's own §9 admissions (`main.typ:1604–1606, 1644–1649`). Not speculation.
2. **It favors the Emender** — its tanh tie-break used a state-tracking proxy; no baseline's selection (incl. M²RNN-CMA's LM-loss CMA search) used a state-tracking criterion (Q1, Q2).
3. **§6's "matched no-tuning across architectures" (`main.typ:941, 948`), read at the point of claim, implies a symmetric selection criterion that did not hold.** The honest caveat exists but is confined to §9 (`main.typ:1600–1615`) — a *co-location* defect, which is exactly what BL-1 asserts.
4. **But it does not invalidate Claim 2.** The S₃ control + NC¹ theorem neutralize the confound for the qualitative claim (delta-correction is the differentiating ingredient); only the magnitude of the within-class S₅ gap is partly confounded.

**Why not RETRACTED:** every load-bearing assertion in BL-1 checks out against the record; the author's strongest point (no probe-specific HPO; budget unchanged) is true but orthogonal — it rebuts a probe-effort confound, not the selection-criterion confound BL-1 actually raises.

**Why not downgraded to PARTIAL:** BL-1 already states the S₃ control as *partial* mitigation in its own **Fix** line and explicitly calls it "the partial mitigation it is." The finding does not overclaim; it is accurate as written. Downgrading would mean the finding overstated something — it did not.

### Required EVALUATION.md edit

**None required.** BL-1 is neither retracted nor downgraded. Its **Issue** and **Fix** text (`EVALUATION.md:41,43`) are accurate and prescribe the correct remedy.

**Optional precision refinement (non-blocking, not applied):** the **Fix** line could note that the NC¹ theorem (§7) — not the S₃ control — is what neutralizes the *GDN* arm, since GDN passes S₃ (0.72) and fails only S₅. As written, the Fix attributes all partial mitigation to the S₃ control, which strictly applies to the M²RNN (nonlinear raw-write) arm. This is a sharpening, not a correction, and does not change BL-1's blocker status.

---

## Provenance index (files read)

- `paper/review/EVALUATION.md` (BL-1 row :38–43, :151, :175, :201)
- `paper/review/rigor.md` (Flag 4 :118–124; OC-2 :184; verdict :259)
- `docs/E88_ABLATION_NOTES.md` (selection criterion = avg100; tanh tie-break :74,:81,:97)
- `paper/main.typ` §6 (:929–958, tab_s5 :968–994, :1014–1030), §9 (:1600–1615, :1642–1649)
- `experiments/expressivity_tasks/run_canonical_sweep.py` (:15–24, :50–51 — matched geometry, all arms)
- `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` (six-task corroboration)
- `ndm/models/e88_fla_hybrid.py` (:838–849 — "E88 optimal" defaults baked in, linear_state=False)
- `paper/ndmpapernotes.md:153–173` and `paper/results/figure_2/raw/README.md` (S₃/S₅ source numbers; M²RNN-CMA = CMA-tuned reshape, M²RNN-paper not converged)
- `paper/results/figure_2/SOURCES.md`, `paper/results/cma_flop_rate/` (M²RNN-CMA search was on LM loss / FLOP-rate)
