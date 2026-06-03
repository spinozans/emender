# S5/S3 Symmetric CMA-ES Protocol — Recon + Design (go/no-go draft)

**Task:** `s5-symmetric-recon` (RECON + PROTOCOL, **NO GPU**, read-only).
**Mode:** READ-ONLY. `paper/main.typ` **not** edited. No training launched.
**Discipline:** Real records only. Where a config/log is not in this repository
it is flagged **NOT FOUND** — never inferred. Line numbers are against the
working tree at the time of writing (`paper/main.typ`, 2651 lines).

> **Why this document exists.** The 8 M state-tracking probes (§6) compare
> Emender / GDN / M²RNN-CMA / M²RNN-paper on *fixed, hand-set* hyperparameters,
> with each architecture's *internal* structural defaults selected on different
> criteria. One Emender component (the `tanh` state nonlinearity) was retained on
> a state-tracking proxy while every baseline's config was settled on
> language-modeling loss. This is the BL-1 selection asymmetry. This protocol
> designs the symmetric experiment that removes it: CMA-ES every architecture at
> 8 M from its existing seed on the *same* state-tracking objective, to a common
> capped budget, and report whichever way it falls.

---

## A. GROUND TRUTH — what objective did each architecture's CMA-ES actually optimize?

There are **two distinct CMA-ES regimes** in the program. Neither is "CMA-ES on a
state-tracking objective." Established from the real records:

### A.0 The CMA-ES fitness function is language-modeling loss (both regimes)

The production driver's fitness metric is the **average training loss (LM
cross-entropy) over training steps**, never a state-tracking score:

- `scripts/cmaes_search_v2.py:1056-1075` — `parse_average_loss()`: *"Compute
  average loss over ALL training steps from stdout. This is the CMA-ES fitness
  metric."* It regexes `loss <float>` from `train.py` step lines and returns the
  mean. There is **no** S5/S3/accuracy term anywhere in the fitness path.
- The 480 M cross-model record corroborates the same rule by an independent
  extractor: `paper/results/cma_flop_rate/SOURCES.md:46-48` — *"The fitness
  function used by the CMA-ES sweep itself is in
  `~/elman/cmaes_search_v2.py::extract_loss` — average over the last 100 logged
  steps, with NaN/divergence rejection."*
- The six searched knobs are width, depth, head count, state width, gating, lr
  (`scripts/cmaes_search_v2.py:155-162`, `_E88_SEARCH_SPACE`). No probe metric.

### A.1 Regime 1 — the 1.3 B-class LM CMA-ES (Emender, M²RNN-CMA, GDN)

**Objective: language-modeling training loss (mean nats over a fixed
late-training window).** Per-architecture, six-knob CMA-ES.

- Paper's own description: `paper/main.typ:1030-1039` (§5 "Per-architecture
  CMA-ES protocol"): *"All three 1.3 B-class architectures (Emender, M²RNN-CMA,
  Gated DeltaNet) received independent CMA-ES … over the same six knobs: width,
  depth, head count, state width, output gating, learning rate … identical
  fitness rule of mean training nats over a fixed late-training window."*
- The **final winner configs** are pinned and present (real records):
  - Emender: `/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/e88/args.json`
    → `level=E88, dim=1664, depth=12, n_heads=370, n_state=32, use_gate=1,
    gate_activation=silu, linear_state=0, lr=8.67767847776187e-4`.
  - GDN: `…/gdn/args.json` → `level=fla-gdn, dim=2688, depth=21, expansion=2,
    n_heads=44, n_state=64, gate_activation=sigmoid, lr=2.871e-3`.
  - M²RNN-CMA: `…/m2rnn/args.json` → `level=m2rnn, dim=1920, depth=21,
    n_heads=370, n_state=16, lr=6.020919750502334e-4`.
  - The non-round `lr` values are consistent with CMA-ES outputs; the configs are
    mirrored in `hf_v03_fix_staging/{emender-e88,gdn,m2rnn-cma}-1.3b/config.json`.
- **NOT FOUND in this repo:** the actual 1.3 B *CMA-ES search* artifacts
  (`results.json`, per-candidate `eval_*.log`, generation history) that produced
  these three configs. `paper/results/figure_2/SOURCES.md:1-4, 75-78` states the
  generating logs live on the training host under `/tmp/pile_convergence_3arch/`
  and `~/elman/run_pile_convergence_3arch.sh`, **not copied into the repo**. What
  *is* committed: the final `args.json` configs (above) and the training-curve
  CSVs `paper/results/figure_2/{E88_NDM,FLA_GDN,M2RNN_CMA}.csv` (loss-vs-step,
  not search records). So the *fitness = LM loss* claim is established from the
  driver code + paper text; the per-generation 1.3 B search trace itself is
  **NOT FOUND** here.
- The one committed *cross-model* CMA search is at **480 M, not 1.3 B**, and
  **excludes M²RNN**: `paper/results/cma_flop_rate/SOURCES.md:7-22` (E88, FLA-GDN,
  Mamba2, E1 only) and `:57-65` — *"no ~480M CMA-tuned M2RNN run is present …
  M²RNN is wired into the CMA-ES search spaces … but [no run]."* `docs/CMA_FLOP_RATE_FINDING.md:33-49`
  says the same. So the only committed evidence that *M²RNN* was ever CMA-tuned at
  all is the **final 1.3 B `args.json`** plus the paper's prose; the M²RNN CMA
  *search record* is **NOT FOUND** in-repo at either scale.

**Per-architecture verdict (Regime 1):**

| Architecture | 1.3 B CMA-ES objective | Search record in repo? | Final config in repo? |
|---|---|---|---|
| Emender (E88) | **LM training loss** (mean late-window nats) | NOT FOUND (logs in `/tmp`, `~/elman`) | YES (`…/e88/args.json`) |
| M²RNN-CMA | **LM training loss** (per §5) | NOT FOUND (also absent from 480 M sweep) | YES (`…/m2rnn/args.json`) |
| GDN (fla-gdn) | **LM training loss** | NOT FOUND at 1.3 B; 480 M trace committed (`cma_flop_rate/`) | YES (`…/gdn/args.json`) |
| M²RNN-paper | **No search** — published-default shape; diverged at step 8,400 | n/a | shape in `main.typ:1061-1066`; log `/tmp/.../m2rnn_paper.log` (NOT in repo) |

### A.2 Regime 2 — the 8 M S5/S3 state-tracking probes

**Objective: NONE — no CMA-ES, no per-architecture search, no probe-specific
HPO.** The 8 M probes are *fixed hand-set* hyperparameters with per-architecture
*structural defaults* baked into the model classes.

- Harness: `experiments/expressivity_tasks/run_separation_suite.py`. The S5/S3
  task configs are **hand-set**: `s5_permutation` = `steps=20000, seq_len=128,
  K=5, lr=3e-4` (`:44-51`); `s3_permutation` = `steps=10000, seq_len=128, K=3,
  lr=3e-4` (`:36-43`). `lr=3e-4` is a literal constant for every task and every
  architecture — there is no search loop.
- Geometry is fixed per architecture to **param-match ~8 M**
  (`run_separation_suite.py:77-112`): `E88_8M` dim=384/H=32/N=32; `M2RNN_8M`
  dim=384/H=32/N=32; `FLA_8M` **dim=640** (to match param count); `M2RNN_paper_8M`
  dim=608. Depth=4 for all. Confirmed in `docs/EXPRESSIVITY_RESULTS_SUMMARY.md:29,
  38-39`: *"FLA-GDN uses `dim=640` to match parameter count."* Measured counts:
  E88 = 7,917,184 params (`paper/results/figure_4_hybrid/canon_pure_E88__fsm_tracking__seed42.json`).
- The per-architecture difference is therefore **only the structural defaults of
  each model class**, applied via empty `kwargs={}` (no overrides;
  `run_separation_suite.py:83,97,104,111`). `train_hybrid.py` passes *no*
  `linear_state`/`use_gate`/`use_conv`/`use_output_norm` flags
  (`paper/review/BL1_provenance_recheck.md:1a`), so each model uses its
  constructor defaults:
  - Emender defaults (`ndm/models/e88_fla_hybrid.py:838-849` per BL1 docs):
    `use_conv=False`, `use_gate=False`, `use_output_norm=False`,
    **`linear_state=False` (tanh ON)**, `decay_mode='mamba'`.
- **The one selection asymmetry (this is BL-1):** the Emender `tanh` default was
  retained on a *state-tracking* proxy while it *tied the linear variant on LM
  loss*. Real record: `docs/E88_ABLATION_NOTES.md:97` — `E88d_linear 1.709 Δ
  0.000 "Linear = Tanh!"` (dead tie on `avg100`, an LM loss in nats); yet the
  best config keeps `linear_state=False` annotated **`# keep tanh for
  expressivity`** (`:74`) and **`✅ Tanh — computational expressivity (UTM
  capability)`** (`:81`). Every other component in that lineage was ranked on
  `avg100` LM loss (`:8,31,90`). `tanh` is the load-bearing ingredient for the
  S5/S3 result, so the Emender's config was settled *partly on a proxy of the
  very quantity the probe measures*; no baseline's was (baselines = published or
  LM-loss-CMA defaults; `paper/review/BL1_adjudication.md:Q1-Q2`).
- **NOT FOUND in this repo:** the raw per-seed 8 M S5/S3 run JSONs. The reported
  `tab_s5` numbers were produced in
  `experiments/expressivity_tasks/results/s5_witness_8m_20260521`
  (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md:209,325`), but
  `experiments/expressivity_tasks/results/` **does not exist** in the working
  tree. The committed source of the numbers is the aggregate table
  `paper/ndmpapernotes.md:153-173` (mirrored `docs/EXPRESSIVITY_RESULTS_SUMMARY.md`).
  The harness scripts are committed; the raw probe records are not.

**Per-architecture verdict (Regime 2 / 8 M S5/S3):** *None of the four arms was
CMA-ES'd or HPO'd at 8 M.* All share hand-set `lr=3e-4`, `steps` (S5=20K, S3=10K),
fixed param-matched geometry. The only per-arm difference is structural defaults;
the only criterion asymmetry is the Emender `tanh` tie-break on a state-tracking
proxy.

---

## B. PAPER-vs-REALITY check

### B.1 The §6 "Matched no probe-tuning" subsection (verbatim)

`paper/main.typ:1226-1265`:

> #heading(level: 2, numbering: none)[Matched no probe-tuning, and the one
> selection asymmetry it does not cover]
>
> The 8 M probe scale received no probe-specific hyperparameter search and no seed
> sweep for any family in the comparison. The Emender ran on the default
> configuration carried down from its 1.3 B stack; M²RNN-CMA ran on the analogous
> default from its CMA-tuned reshape; GDN and the M²RNN-paper shape ran on their
> respective published defaults. Each family is therefore evaluated on the
> reasonable-defaults configuration it would arrive at without probe-targeted
> optimization, rather than matched-after-HPO. …
>
> That matched-no-tuning condition controls for probe-specific *effort*; it does
> not control for one *selection-criterion* asymmetry, stated here in full. The
> Emender's defaults are the endpoint of an ablation lineage ranked on
> language-modeling loss — the same objective GDN's and M²RNN's authors used for
> their published defaults — with one exception: the state nonlinearity ($tanh$
> versus a linear state) tied the linear variant on language-modeling loss and was
> kept on state-tracking grounds. That one component is load-bearing for these
> probes, so the Emender's configuration was settled partly on a proxy of the very
> quantity the probes measure, whereas no baseline's was. The asymmetry favors the
> Emender, and it bears on the *magnitude* of the within-class $S_5$ gap.

The two controls follow (`:1251-1265`): the S₃ solvable-group control (raw-write
M²RNN 0.31 on S₃) and the §7 NC¹ realizability result for GDN, which together fix
the *direction/mechanism* but *not the magnitude* of the Emender's own S₅ number —
*"A matched state-tracking search on each baseline at the 8 M shape would close
that remaining magnitude gap; it is named in §12."*

### B.2 The BL-1 disclosure content

There is **no literal "BL-1" string** in `main.typ` (BL-1 is the review-tracker id
in `paper/review/EVALUATION.md`). The substance the task calls "the BL-1
disclosure (tanh settled partly on a state-tracking proxy; baselines
LM-loss-tuned)" lives in **two** committed places, both quoted verbatim:

1. §6 selection-asymmetry paragraph — `main.typ:1240-1249` (quoted in B.1).
2. §10 Limitations → "Open architectural choices" — `main.typ:2214-2221`:
   > Several internal questions remain open: the output gate, the state
   > non-linearity ($tanh$ vs linear), and the decay parameterization (simple
   > sigmoid vs Mamba2-style log-space). All three tie on loss at small scale; the
   > 1.3 B architecture keeps the conservative settings on the strength of
   > state-tracking and stability data, not a clean ablation at 1.3 B.

(Note: prior review docs cite a "§9 selection-history paragraph at main.typ:1604"
— that line numbering is from an **earlier** revision. In the current tree the
selection-history disclosure is at **§6 / line 1240-1249** and the
open-architectural disclosure at **§10 / line 2214-2221**. Future Work restates
the tie at `main.typ:2422`.)

### B.3 The §9-equivalent wording

The section that was "§9" in the records cited by `BL1_adjudication.md` has been
renumbered. Current section map (`main.typ`): §1 Intro (160), §2 Background (448),
§3 Architecture (579), §4 Multi-Programming/Systems (865), §5 Language-Modeling
Results (1007), §6 Expressivity Results (1210), §7 Formal Results (1673), §8
Related Work (2009), §9 Limitations (2109), §10 Conclusion (2233), Testable
Predictions (2309), Future Work (2383). **The "§9 selection-history" text the
older review docs reference now lives inside §6 (1240-1249) and §9 Limitations
(2214-2221).** Anyone applying a fix must target those current spans.

### B.4 Where the paper AGREES with ground truth (A)

- §5 (`:1030-1039`): all three 1.3 B configs CMA-ES'd on LM nats. **AGREES** with
  A.1 (fitness = LM loss). Verified against driver `parse_average_loss`.
- §6 (`:1228-1234`): the 8 M probes got *no probe-specific HPO / no seed sweep*;
  arms ran on defaults. **AGREES** with A.2 (hand-set `lr=3e-4`, empty `kwargs`).
- §6 (`:1240-1249`): Emender `tanh` kept on state-tracking grounds while tying on
  LM loss; baselines LM-loss-selected. **AGREES exactly** with
  `E88_ABLATION_NOTES.md:74,81,97`.
- §6 (`:1264-1265`): a matched state-tracking search at 8 M would close the
  magnitude gap and is named in future work. **AGREES** — this protocol is that
  search.

### B.5 Where the paper DIVERGES / under-describes — text-must-match-reality flags

These are **identification only** (no edits made). Each is a span whose wording
would need to change *if and when* the symmetric experiment is run, or where the
current text is looser than the record:

- **FLAG B5-a (under-description of M²RNN-CMA provenance).** §6 `main.typ:1230-1231`
  says M²RNN-CMA "ran on the analogous default from its **CMA-tuned reshape**,"
  and §5 `:1032-1034` lists M²RNN-CMA among architectures that "received
  independent CMA-ES." **Reality:** the M²RNN CMA-ES *search record is NOT FOUND
  in this repo* at either 480 M or 1.3 B (`cma_flop_rate/SOURCES.md:57-65`;
  `CMA_FLOP_RATE_FINDING.md:48-49`). Only the final 1.3 B `args.json` and the
  paper prose attest the tuning. This is not necessarily wrong — the search may
  have run on the training host — but the *evidence is off-repo*. If the claim
  must be repo-reproducible, the exact span to revisit is
  `main.typ:1030-1039` and `1230-1231`. Not a results error; a provenance gap to
  disclose or back-fill.

- **FLAG B5-b (renumbered cross-references).** Any text or review doc pointing at
  "§9" for the selection-history caveat is stale; the caveat is now §6
  (`1240-1249`) + §9 Limitations (`2214-2221`). Spans to keep consistent if §6 is
  edited: `main.typ:1240-1249`, `2214-2221`, `2422`.

- **FLAG B5-c (the load-bearing post-experiment span).** *If symmetric tuning
  closes/erases the gap*, the spans that must change are the magnitude hedges that
  currently lean on the asymmetry being un-removed:
  - `main.typ:1248-1249`: *"The asymmetry favors the Emender, and it bears on the
    *magnitude* of the within-class $S_5$ gap."*
  - `main.typ:1262-1265`: *"what they do not fully de-confound is the *size* of
    the Emender's own $S_5$ number. A matched state-tracking search on each
    baseline at the 8 M shape would close that remaining magnitude gap; it is
    named in §12."*
  - The `tab_s5` numbers (`main.typ:1284-1288`) and the §6 prose
    (`:1321-1327`, `:1335-1337`).
  These are the exact strings to update with the symmetric result, *whichever way
  it falls* (see D.7). **No baseline result in the paper is contradicted by the
  record** — the divergence is purely that the §6 "matched no-tuning" label,
  read at the point of claim, implies a symmetric *selection criterion* that did
  not hold for the `tanh` component. The paper already discloses this; the fix
  this protocol enables is to *remove* the asymmetry rather than caveat it.

- **NOT a divergence (recorded to prevent a false flag):** §6's claim that
  baselines got *no* state-tracking tuning is **correct** — even M²RNN-CMA's
  (off-repo) CMA search optimized LM loss, never a probe metric
  (`BL1_adjudication.md:Q2`). The asymmetry is one-directional and the paper
  states it.

---

## C. INVENTORY for the symmetric experiment

| Asset | Path (committed unless noted) | Notes / citation |
|---|---|---|
| **8 M training+eval harness** | `experiments/expressivity_tasks/train_hybrid.py` | builds `HybridLadderLM` from `--layer_pattern` + dim/heads/state; runs train + periodic eval; writes `<label>.json` with `final_acc`, per-step `eval_acc`/`elapsed_s`. |
| **8 M multi-GPU runner** | `experiments/expressivity_tasks/run_separation_suite.py` | S5/S3 task configs `:32-59`; per-arch param-matched presets `:77-119`; round-robin GPU pool; writes `separation_summary.json`. |
| **S5/S3 tasks** | `experiments/expressivity_tasks/tasks/` (imported as `ALL_TASKS`) | `s5_permutation` (K=5, 120-way), `s3_permutation` (K=3, 6-way); random baselines 1/120=0.0083, 1/6=0.1667. |
| **Per-arch 8 M SEED configs** | `run_separation_suite.py:77-112` `MODEL_CONFIG` | **These are the CMA-ES seeds.** `E88_8M`{dim384,H32,N32}; `M2RNN_8M`{dim384,H32,N32}; `FLA_8M`{dim640,H32,N32}; `M2RNN_paper_8M`{dim608,H32,N32}. depth=4 all. |
| **CMA-ES driver** | `scripts/cmaes_search_v2.py` | two-phase LHS→CMA-ES; 6-knob search spaces `:155-313`; fitness `parse_average_loss` `:1056-1075` (currently **LM loss** — must be swapped for the probe objective, see D). Anchor/seed loader `:653-700` (`load_anchor_configs`, seed CMA from an existing config). |
| **CMA-ES invocation defaults** | `cma_flop_rate/SOURCES.md:70-73` (off-repo `~/elman/run_cmaes_v2.py`) | population 16, sigma 0.35, min-generations 12, converge threshold 0.002 over 3 consecutive generations. |
| **1.3 B finetune analogue** (reference) | `scripts/finetune_s3_s5.py`, `finetune_s5_symmetric.py` | symmetric-budget S5 finetune of the 1.3 B checkpoints; identical recipe across arms (template for "honest, symmetric, no-S5-tuning"). Not the 8 M experiment but the design precedent. |
| **8 M parameter-match recipe** | `EXPRESSIVITY_RESULTS_SUMMARY.md:29,38-39`; `run_separation_suite.py:77-112` | E88/M2RNN dim=384; FLA-GDN dim=640; M2RNN-paper dim=608; all depth=4, H=32, N=32 → ~8 M (E88 measured 7.92 M). |
| **Final 1.3 B configs** (provenance) | `…/emender_paper_pinned_checkpoints/{e88,gdn,m2rnn}/args.json` | the converged LM-CMA winners (A.1). |
| **NOT FOUND** | `experiments/expressivity_tasks/results/` ; 1.3 B CMA search logs | raw 8 M S5/S3 per-seed JSONs and the 1.3 B per-generation search trace are not committed (A.1, A.2). |

### C.1 Per-generation runtime / GPU cost (for budgeting GPUs 2,3,4,5)

Measured from a real committed 8 M run
(`paper/results/figure_4_hybrid/canon_pure_E88__fsm_tracking__seed42.json`,
dim=384 depth=4 H=32 N=32 batch=32, **seq_len=256**):

- step 500 @ 727.96 s, step 1000 @ 1438.88 s ⇒ **≈ 1.42 s/step**.
- 10 000-step run → `elapsed_s = 14038` ⇒ **≈ 3.9 GPU-hours** per full run.

Extrapolation for the probe lengths used by S5/S3 (**seq_len=128**, ~half the
per-step cost of seq_len=256): **≈ 0.7 s/step**.

- **S3 full run** (10 000 steps, seq128): ≈ 7 000 s ≈ **2.0 GPU-h**.
- **S5 full run** (20 000 steps, seq128): ≈ 14 000 s ≈ **3.9 GPU-h**.

These are *full* runs. A CMA-ES that ran every candidate to 20 K steps would cost
~3.9 GPU-h × population 16 × ~12 generations ≈ **750 GPU-h per architecture** —
infeasible. Hence the protocol uses a **fixed short per-candidate budget** (D.3).
At the short budget below (5 000 steps, seq128 ≈ 1.0 GPU-h/candidate) and 4 GPUs
(2,3,4,5): population 16 → 4 GPU-waves/generation ≈ **4 GPU-h wall per
generation** ⇒ a 12-generation cap ≈ **~48 GPU-h wall per architecture**, ≈
**~190 GPU-h wall for all four** if run sequentially, or ~48 h wall if the four
architectures share the 4-GPU pool round-robin across the full job set.

> GPU-cost numbers are *derived from one committed timing trace* and a seq-length
> scaling assumption; they should be re-measured with a single dry-run candidate
> before committing the full budget. Flagged, not silently assumed.

---

## D. THE SYMMETRIC PROTOCOL (human go/no-go BEFORE any GPU spend)

**Design principle.** At 8 M params, **every** architecture's config is CMA-ES'd
**from its existing seed** (C, `MODEL_CONFIG`) on the **same state-tracking
objective**, to the **same capped budget**, with identical eval. This removes the
single selection-criterion asymmetry (the Emender `tanh` tie-break) by giving
*every* arm the right to settle its config on the probe metric.

### D.1 The CMA-ES objective (proposed + justified)

**Objective = S5 accuracy at the trained length, evaluated as a held-back mean
over the last training window; S3 held OUT as an untuned control.**

Concretely, the fitness returned to CMA-ES is:

> **`fitness = 1 − mean_S5_acc@T=128`** (CMA-ES minimizes), where
> `mean_S5_acc@T=128` is the mean eval accuracy on `s5_permutation` at the
> *training* length T=128, averaged over the final 1 000 training steps of the
> capped candidate run, with NaN/divergence → `fitness = 1.0` (worst).

Justification:

- **Why S5 at the trained length, not extrapolation.** Length extrapolation
  (T=256…1024) is the *headline scientific readout* and must remain an
  **untouched evaluation**, never the selection target — tuning on extrapolation
  would re-introduce exactly the metric-selection confound BL-1 names, just at a
  different T. Selecting on T=128 (the length all arms actually train at) measures
  *learnability of the state-tracking map under SGD* — the property §6 claims —
  while leaving the extrapolation curve as an honest out-of-selection test.
- **Why S3 is held OUT (untuned control), not co-optimized.** S3 is the
  *solvable-group control* whose entire epistemic job (§6 `:1251-1257`) is to be
  *independent* of the Emender's selection advantage: raw-write M²RNN's 0.31 on S3
  is load-bearing precisely because nothing was tuned toward it. If we CMA-ES on
  S3 too, we destroy that control. So **S3 is evaluated, never selected on** — it
  becomes the symmetric-experiment's analogue of a held-out test set. (Optional
  secondary report: an S3-tuned variant *as a separate ablation*, clearly labeled,
  to show the control is not cherry-picked — but the primary protocol tunes S5
  only.)
- **Why not LM loss.** LM loss is the *old* objective whose asymmetry we are
  removing; re-using it would not change anything. The whole point is to let every
  arm optimize the probe metric symmetrically.
- **Implementation.** Replace the fitness extraction in the CMA loop: instead of
  `parse_average_loss` (`cmaes_search_v2.py:1056-1075`), each candidate runs
  `train_hybrid.py --task s5_permutation` and the runner reads `final`/windowed
  `eval_acc` from the candidate's JSON (the field `train_hybrid` already writes;
  see C.1 sample). No new training code is required — only a fitness adapter and a
  per-task eval hook; S3 is run once on each *final* winner, not inside the loop.

### D.2 Per-architecture search space (seeded from the existing config)

Seed each CMA-ES at the committed `MODEL_CONFIG` value (C). **Vary** the knobs
that the §5 search varied and that plausibly affect state-tracking; **fix**
everything that defines the architecture family or the fair-comparison frame.

| Knob | Emender (E88) | GDN (fla-gdn) | M²RNN-CMA | M²RNN-paper | Vary/Fix |
|---|---|---|---|---|---|
| seed config | dim384,H32,N32 | dim640,H32,N32 | dim384,H32,N32 | dim608,H32,N32 | seed center |
| `dim` | search ±, re-match params | search ±, re-match | search ± | search ± | **VARY** (with param re-match, D.4) |
| `depth` | search 3–6 | 3–6 | 3–6 | 3–6 | **VARY** |
| `n_heads` | search | search | search | search | **VARY** |
| `n_state` | {16,32} sweep | search | {16,32} | {16} | **VARY** (discrete sweep per `cmaes_search_v2.py:316-332`) |
| `lr` | search (log) | search | search | search | **VARY** (this is where probe-learnability lives) |
| gating / `linear_state` | **search `linear_state∈{0,1}` and `use_gate∈{0,1}`** | published | search | published | **VARY for E88** — this is the BL-1 component; let CMA decide it on S5, symmetrically |
| update-rule family | E88 delta | linear gated delta | raw-write M²RNN | raw-write M²RNN paper-shape | **FIX** (the thing being compared) |
| optimizer | schedule-free AdamW | same | same | same | **FIX** |
| train length T, eval T-grid | 128 / {128,256,512,1024} | same | same | same | **FIX** (identical eval battery) |
| target params | ~8 M | ~8 M | ~8 M | ~8 M | **FIX** (D.4) |

The crucial symmetry: **for the Emender, `linear_state` (tanh vs linear) is now a
*searched* knob optimized on S5** — exactly the component that was previously
hand-kept on a state-tracking proxy. Every baseline simultaneously gets to search
*its* config on the same S5 objective. No arm has a privileged tie-break.

### D.3 Convergence criterion + budget CAP (identical across architectures)

- **Per-candidate training budget (CAP):** `steps = 5000`, seq_len=128,
  batch=32, schedule-free AdamW. (~1.0 GPU-h/candidate; chosen so the *ranking*
  among candidates is informative while the full 20 K-step run is reserved for the
  final winners.) **Identical for all four arms.**
- **Population:** 16 (matches §5 / `run_cmaes_v2.py`), sigma 0.35.
- **Convergence:** best-fitness improvement < 0.005 (in accuracy units) over 3
  consecutive generations, **OR** the generation CAP below — whichever first.
- **Generation CAP (fairness + GPU budget):** **12 generations per
  architecture** (matches the §5 min-generations). This is a *hard identical cap*
  so no arm gets more search than another. Total ≈ 16×12 = 192 candidates ×
  ~1.0 GPU-h ≈ **~190 GPU-h per architecture**, run on GPUs {2,3,4,5} (4-wide) ⇒
  ~48 h wall per architecture (or interleave all four across the pool).
- **Seeds during search:** 1 seed (42) per candidate to keep the budget bounded
  (selection noise is absorbed by re-running winners at 3 seeds, D.5). Logged as a
  known cap, not hidden.

### D.4 Fixed FLOP/param matching

- **Target = ~8 M params**, the same band as the current probe (E88 7.92 M). When
  CMA varies `dim`/`depth`/`n_heads`/`n_state`, re-match params with the existing
  estimator + tolerance: `cmaes_search_v2.py:485-645` (`estimate_params_for_config`,
  `is_valid_param_count`, `PARAM_TOLERANCE=0.10`). Candidates outside ±10 % of
  8 M are rejected before training (same mechanism §5 used).
- **FLOP match** follows from param + identical (T, batch, steps) budget; the
  per-candidate step budget is identical across arms (D.3), so train-time FLOPs are
  matched by construction. GDN's dim=640 seed already encodes its param-match;
  the estimator keeps it at ~8 M as `dim` moves.

### D.5 Seed count + eval battery (final winners)

For each architecture's **final CMA winner config**, run the *full* evaluation
exactly as the current probe does, so results drop into `tab_s5` unchanged in
shape:

- **3 seeds** {42, 123, 456} (matches current 3-seed protocol,
  `run_separation_suite.py` default seeds; `tab_s5` is a 3-seed mean).
- **S5** (selected-on): train T=128, 20 000 steps; eval T ∈ {128, 256, 512, 1024}.
- **S3 control** (held-OUT): train T=128, 10 000 steps; eval T ∈ {128,256,512,1024}.
  Report S3 as the untuned control alongside S5.
- **Length series**: the full {128,256,512,1024} extrapolation curve per arm,
  reported as the out-of-selection test (never the selection target).
- Optional six-task canonical sweep (`run_canonical_sweep.py`) on the winners for
  robustness, clearly marked secondary.

### D.6 What is varied vs fixed — one-line summary

**Varied (symmetrically, all arms, selected on S5@T128):** dim, depth, n_heads,
n_state, lr, and (for E88) the gating/tanh structural knobs.
**Fixed (all arms):** update-rule family, optimizer, train length, eval battery,
~8 M param target, per-candidate step budget, population, generation cap, search
seed. **Held OUT (evaluated, never selected):** S3, the T>128 extrapolation curve.

### D.7 Honest-reporting plan (report whichever way it falls)

This is binding and stated up front, before any GPU spend:

- **If the ordering survives** (Emender > M²RNN-CMA > GDN, and Emender ≥ baselines
  on S5@T128 after every arm is symmetrically tuned): **Contribution 2
  strengthens and BL-1 dissolves.** The §6 magnitude hedges
  (`main.typ:1248-1249, 1262-1265`) get *replaced* with the symmetric result; the
  "matched no-tuning" framing is upgraded to "matched **after symmetric
  state-tracking HPO**," removing the one asymmetry it could not cover.
- **If symmetric tuning erases or reverses the gap** (a baseline catches or beats
  the Emender on S5@T128 once it is allowed to tune on the probe): **that is the
  finding and it is reported as such.** The §6 separation claim is then re-scoped
  to what remains true (e.g., length-extrapolation behavior, the S3 raw-write
  deficit, the GDN NC¹ collapse) and the within-class *magnitude* claim is
  retracted or softened to match. `tab_s5` is republished with the symmetric
  numbers and a clearly labeled "symmetric S5-tuned" column next to the original
  "defaults" column.
- **Either way:** both the original defaults-based numbers *and* the symmetric
  numbers are reported side by side; the per-arm winning configs and the full
  generation logs are committed to the repo (closing the A.2 NOT-FOUND gap for
  this experiment); S3 and the T-extrapolation curves are reported as
  out-of-selection controls. No result is suppressed because it is inconvenient.

### D.8 Pre-flight (no GPU)

1. Re-measure one dry-run candidate's wall-time to confirm C.1's ~1.0 GPU-h/5 K
   estimate before committing the 12-generation cap.
2. Add the S5-accuracy fitness adapter to (a repo copy of) `cmaes_search_v2.py`
   and a `--objective s5_acc@T128` switch; do **not** mutate the LM-loss path used
   by §5.
3. Confirm `load_anchor_configs` (`:676-700`) seeds each CMA at the
   `MODEL_CONFIG` center.
4. Ensure every candidate JSON commits under
   `experiments/expressivity_tasks/results/s5_symmetric_<date>/` so the records
   exist this time.

---

## Validation trace (task acceptance criteria)

- [x] Per-architecture CMA-ES objective established from REAL records with
  file:line citations (or NOT FOUND), no inference — §A (1.3 B = LM loss,
  `cmaes_search_v2.py:1056-1075` + `main.typ:1030-1039`; 8 M = no search,
  `run_separation_suite.py:36-59,77-119`; tanh asymmetry `E88_ABLATION_NOTES.md:74,81,97`;
  1.3 B + M²RNN search logs NOT FOUND).
- [x] Paper-text vs reality divergences identified with exact spans
  (§B: FLAG B5-a/b/c, spans `main.typ:1226-1265, 2214-2221, 1248-1249, 1262-1265,
  1284-1288`).
- [x] Seed configs + 8 M harness + CMA driver + per-generation GPU cost located
  and cited (§C, C.1).
- [x] Symmetric protocol fully specified — objective, search space, budget /
  convergence, matching, eval, honest-reporting clause (§D).
- [x] NO GPU launched; `main.typ` NOT edited.
