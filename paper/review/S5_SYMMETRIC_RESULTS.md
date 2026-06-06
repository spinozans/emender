# S5-SYMMETRIC WINNER-EVAL — RESULTS (task s5sym-eval, agent-1012)

Date: 2026-06-03. Protocol: `paper/review/S5_SYMMETRIC_PROTOCOL.md` §D (esp. D.7
honest-reporting clause). Upstream search: `SEARCH_RESULTS.md` (task s5sym-search,
commit 2264b77).

**One-line verdict: SURVIVED.** After every arm was symmetrically CMA-tuned on
the S5 probe at full fidelity, the Emender still separates from both baselines at
training length: **Emender 0.9997 > GDN 0.5446 > M²RNN-CMA 0.1655** on S5@T128
(seed-mean over {42,123,456}). The Emender slot is won by **e88-linear** (linear
state) over e88-tanh, 0.9997 vs 0.9888 at T128 — though both E88 variants dominate
both baselines at every length. Per protocol D.7: **BL-1 dissolves, Contribution 2
firms; the §6 spans are named for UPGRADE below (NOT applied — human-gated).**

---

## 1. What was run (REAL eval, no mocks)

Driver: `scripts/eval_s5_symmetric_winners.py` → `experiments/expressivity_tasks/train_hybrid.py`.
For EACH of the four CMA winners:

- 3 seeds {42, 123, 456}.
- **S5**: `--task s5_permutation`, train T=128, **20000 steps**.
- **S3 control**: `--task s3_permutation`, train T=128, **10000 steps**.
- Eval grid (length-extrapolation, end of training, 8 batches/length):
  **T ∈ {128, 256, 512, 1024}**.
- schedule-free AdamW, batch 32, seq_len 128 — identical recipe across all arms
  (the symmetric condition).

24 runs total (4 arms × 3 seeds × {S5, S3}), round-robin across **GPUs 0-7**.
All 24 completed, 0 failed (~8.4 h wall on 8 GPUs). Raw per-seed JSONs committed
under `experiments/expressivity_tasks/results/s5_symmetric_20260603/eval/`
(`<arm>_<S5|S3>_seed<seed>.json`); per-run stdout under `eval/logs/`; seed-mean
roll-up in `eval/summary.json` (`scripts/aggregate_s5_symmetric.py`).

The reported S5@T accuracy is the `length_extrap[T]["acc"]` field train_hybrid
writes after training; the S5@T128 column is the in-selection training length, the
256/512/1024 columns are held-out length-extrapolation.

## 2. Per-winner CMA config (from the symmetric search)

Each arm received its OWN per-architecture CMA-ES search at matched ~8 M params
(`SEARCH_RESULTS.md`); `linear_state`/`use_gate` were FIXED per arm and not in the
search space. Configs are taken verbatim from `winners/<arm>.args.json`.

| Arm | layer | dim | depth | n_heads | n_state | lr | linear_state | use_gate | real params | search acc@T128 (300-step) |
|-----|-------|----:|------:|--------:|--------:|------:|:---:|:---:|----:|----:|
| e88-tanh   | E88     | 256 | 5 | 39 | 32 | 0.002950 | 0 | 1 | 8.07 M | 0.0689 |
| e88-linear | E88     | 256 | 5 | 38 | 32 | 0.002657 | 1 | 1 | 7.86 M | 0.0865 |
| m2rnn      | m2rnn   | 512 | 5 | 19 | 32 | 0.001606 | — | — | 8.00 M | 0.0292 |
| gdn        | fla-gdn | 512 | 6 | 22 | 32 | 0.001206 | — | — | 7.99 M | 0.0392 |

The "search acc@T128" is the 300-step *truncated fitness* used to rank candidates;
it is NOT the converged accuracy. The full-fidelity 20000-step numbers below are
roughly 10× higher — the truncated fitness only had to *rank* arms, not measure
their converged ceiling.

## 3. Both columns: ORIGINAL defaults vs SYMMETRIC-tuned

### S5 (non-solvable NC¹ witness), seed-mean accuracy

| Model | S5 T=128 | S5 T=256 | S5 T=512 | S5 T=1024 |
|-------|---------:|---------:|---------:|----------:|
| **ORIGINAL defaults** (current `tab_s5`, `main.typ:1354-1357`) | | | | |
| Emender (defaults)       | 0.7918 | 0.4158 | 0.2150 | — |
| GDN (defaults)           | 0.3552 | 0.1843 | 0.0974 | — |
| M²RNN-CMA (defaults)     | 0.2157 | 0.1120 | 0.0593 | — |
| M²RNN-paper (defaults)   | 0.1698 | 0.0884 | 0.0488 | — |
| **SYMMETRIC-tuned** (this run, full 8-GPU eval) | | | | |
| **Emender = e88-linear** (slot winner) | **0.9997** | **0.7515** | **0.3909** | **0.2002** |
| e88-tanh (other Emender variant)       | 0.9888 | 0.6296 | 0.3216 | 0.1678 |
| GDN (fla-gdn)                          | 0.5446 | 0.2801 | 0.1441 | 0.0759 |
| M²RNN-CMA                              | 0.1655 | 0.0858 | 0.0479 | 0.0276 |
| random (1/120)                         | 0.0083 | 0.0083 | 0.0083 | 0.0083 |

### S3 control (solvable group, random = 1/6 = 0.1667), seed-mean accuracy

| Model | S3 T=128 | S3 T=256 | S3 T=512 | S3 T=1024 |
|-------|---------:|---------:|---------:|----------:|
| **ORIGINAL defaults** (`tab_s5` S3 T=128 only) | | | | |
| Emender (defaults)     | 1.0000 | — | — | — |
| GDN (defaults)         | 0.7185 | — | — | — |
| M²RNN-CMA (defaults)   | 0.3124 | — | — | — |
| M²RNN-paper (defaults) | 0.3773 | — | — | — |
| **SYMMETRIC-tuned** (this run) | | | | |
| e88-linear (Emender)   | 1.0000 | 0.9919 | 0.8646 | 0.6480 |
| e88-tanh               | 1.0000 | 0.9976 | 0.8929 | 0.6152 |
| GDN (fla-gdn)          | 0.9243 | 0.6525 | 0.4156 | 0.2907 |
| M²RNN-CMA              | 0.1905 | 0.1776 | 0.1737 | 0.1696 |
| random (1/6)           | 0.1667 | 0.1667 | 0.1667 | 0.1667 |

Per-seed spread (mean ± SD over 3 seeds) at S5@T128:
e88-linear 0.9997 ± 0.0005 · e88-tanh 0.9888 ± 0.0111 ·
GDN 0.5446 ± 0.2492 (seeds 0.76 / 0.60 / 0.27) · M²RNN-CMA 0.1655 ± 0.0346.
Full per-seed values: `eval/summary.json`.

## 4. tanh vs linear — the BL-1 architectural decision at full fidelity

This run IS the BL-1 decision (E88 linear-state vs tanh-state) made at full
fidelity, with both variants independently CMA-searched and 3-seed evaluated:

- **e88-linear wins the Emender slot.** S5@T128 0.9997 vs 0.9888; it also leads at
  every held-out length (256: 0.7515 vs 0.6296; 512: 0.3909 vs 0.3216; 1024:
  0.2002 vs 0.1678). At T128 both are essentially at ceiling (within ~1 pt); the
  gap widens with extrapolation length, favoring linear.
- **The architectural-family conclusion is robust to the tanh/linear choice:**
  BOTH E88 variants dominate BOTH baselines at every length. The Emender slot would
  read as a clean separation regardless of which knob is chosen; linear is simply
  the better of two winners.

→ "The Emender" result = **e88-linear** (better of the two on S5@T128).

## 5. Ordering verdict

**Did (best Emender) > M²RNN-CMA > GDN survive, dissolve, or reverse on S5@T128
after every arm was symmetrically CMA-tuned?**

**SURVIVED** — with the headline separation *amplified*, not just preserved:

- Observed symmetric ordering at S5@T128: **Emender (0.9997) > GDN (0.5446) >
  M²RNN-CMA (0.1655)**, random 0.0083.
- The load-bearing claim — *the Emender separates from all baselines at training
  length* — holds decisively. The margin is WIDER than the defaults run
  (defaults: 0.79 vs 0.36/0.22; symmetric: 1.00 vs 0.54/0.17).
- Nuance, reported honestly: the protocol D.7 hypothesis wrote the survival
  ordering as "Emender > M²RNN-CMA > GDN", but in the *original* `tab_s5` GDN
  (0.3552) already outranked M²RNN-CMA (0.2157). The symmetric result keeps that
  GDN > M²RNN sub-order (0.5446 > 0.1655) — i.e., the baseline sub-order is
  unchanged from the published table; only the *Emender-on-top* claim was ever
  load-bearing, and it survives.
- Each baseline was given its own symmetric probe-tuning budget. GDN *improved*
  under tuning (0.3552 → 0.5446) yet still collapses with length (0.5446 → 0.0759
  at T1024) and stays far below the Emender — consistent with the §7 NC¹
  realizability argument that a linear recurrence cannot track non-solvable S5 at
  length. M²RNN-CMA *regressed* slightly (0.2157 → 0.1655) and, strikingly, fails
  even the **solvable S3 control** (0.1905, at chance 0.1667) — its symmetric
  winner config does not learn S3 either, sharpening the raw-write-deficit claim.

## 6. Honest-either-way consequence (protocol D.7) — §6 spans to UPGRADE (NOT applied)

Because the ordering **SURVIVED** and the separation strengthened, per D.7
**BL-1 dissolves and Contribution 2 firms.** The following §6 / §5 spans should be
**UPGRADED** to the symmetric result. **These edits are NOT applied here** — they
are a separate human-gated step. `paper/main.typ` was NOT touched by this task.

Line numbers below are anchors from the task/protocol; the file has since shifted
slightly, so each is given with its current content for re-location.

1. **`main.typ:1248-1249`** (matched-compute / "same per-architecture CMA-ES
   hyperparameter-and-shape search budget" framing; currently ~`main.typ:1249-1251`).
   UPGRADE: the "matched **no-tuning**" framing can be upgraded to "matched
   **after symmetric state-tracking HPO**." The one selection asymmetry BL-1 could
   not cover (no baseline was tuned on the probe) is now closed: every arm got an
   equal S5 CMA search and the Emender still wins.

2. **`main.typ:1262-1265`** (the "separation shows only on synthetic algebraic
   state-tracking (S5/parity/FSM) … no model in the cohort solves" magnitude
   hedge; currently ~`main.typ:1261-1264`).
   UPGRADE: replace the magnitude hedge with the symmetric finding — the
   separation is not an artifact of asymmetric tuning; it persists (and widens)
   when each baseline is allowed to tune on the probe.

3. **`tab_s5` block `main.typ:1284-1288`** (§6 probe-config paragraph "run at 8 M
   parameter-matched scale (dim=384, depth=4 …)"; currently ~`main.typ:1283-1289`),
   together with the **table body at `main.typ:1354-1357`** and caption
   (~`main.typ:1359-1374`).
   UPGRADE / REPUBLISH: add a clearly labeled **"symmetric S5-tuned"** column next
   to the existing **"defaults"** column (both numbers side by side, per D.7), and
   note the per-arm CMA configs from §2 above. Suggested symmetric column:
   Emender (e88-linear) 1.0000 / 0.9997 / 0.7515 / 0.3909 (S3@128 / S5@128/256/512);
   GDN 0.9243 / 0.5446 / 0.2801 / 0.1441; M²RNN-CMA 0.1905 / 0.1655 / 0.0858 /
   0.0479. The probe-config paragraph should note that the symmetric column uses
   the per-arm searched shapes (§2), not the fixed dim=384/640 defaults.

The re-scope branch of D.7 (used only if the gap had dissolved/reversed) does NOT
apply: no baseline caught or beat the Emender on S5@T128. Recorded here for
completeness — had it dissolved/reversed, the SAME spans (`main.typ:1248-1249`,
`1262-1265`, `tab_s5 1284-1288`) would instead be RE-SCOPED to what remained true
(length-extrapolation behavior, the S3 raw-write deficit, the GDN NC¹ collapse),
and the within-class magnitude claim softened.

## 7. Reproduce

```bash
# Full 24-run eval (idempotent; skips completed JSONs):
python scripts/eval_s5_symmetric_winners.py
# Seed-mean roll-up + verdict:
python scripts/aggregate_s5_symmetric.py
```

Artifacts:
- `scripts/eval_s5_symmetric_winners.py`, `scripts/aggregate_s5_symmetric.py`
- `experiments/expressivity_tasks/results/s5_symmetric_20260603/eval/*.json` (24 raw per-seed)
- `experiments/expressivity_tasks/results/s5_symmetric_20260603/eval/summary.json`
- `experiments/expressivity_tasks/results/s5_symmetric_20260603/eval/logs/*.log`
