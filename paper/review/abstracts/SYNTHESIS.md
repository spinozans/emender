# Abstract fan-in synthesis

Fan-in over the eight `abstract-fanout-1..8` writer nodes. Each writer read the
whole paper independently and produced a candidate lead abstract. This node
ranks the candidates, then grafts the strongest lead and clauses into a single
recommended abstract, grounded in `paper/main.typ` and held to ~130–160 words.

**Note on coverage:** 6 of the 8 candidate files exist on disk
(`abstract-1,2,3,5,7,8.md`). `abstract-4.md` and `abstract-6.md` were never
written — those two writer nodes (both "Default Documenter") passed eval but
recorded no `artifacts:` line and left no file. They are scored N/A below rather
than penalised, since there is nothing to grade. The ranking covers all extant
candidates.

## Ranking (best-first)

Dimensions weighed: punchy lead; names E88 + the Emender class; enumerates the
real contributions (multi-programming viability, the matched-CMA-ES
multi-architecture study, the mathematically-derived-then-confirmed class
ordering); reads as a controlled study rather than one model; legibility and
length (~130–160 words, body only); honesty (grounded in the paper, parity
stated as no-penalty / "not a win", no invented facts).

| Rank | ID | Score | Words | One-line rationale |
|------|----|------:|------:|--------------------|
| 1 | abstract-2 | 93 | 158 | Cleanest punchy lead (result-first, "no attention and no time-axis scan"), natural E88 reveal, explicit "controlled study", refutation grounded, parity flat — best all-round balance, dead-on length. |
| 2 | abstract-8 | 91 | 159 | Most distinctive lead ("drops below one bit per byte"); economical "ledger" voice carries all three contributions as settled facts; adds mechanism ("correct their state rather than refill it") without bloat; in length. |
| 3 | abstract-7 | 90 | 160 | Folds the delta-correction mechanism into the class definition; "on par rather than ahead" is the most carefully honest parity phrasing; complete and legible; says "family" (paper-faithful) where the brief says "class". |
| 4 | abstract-3 | 89 | 158 | Clean impossibility-first lead, defers E88, all contributions present, in length; the "no time-axis scan **trick**" is a touch of editorialising and the lead is slightly less distinctive than #1. |
| 5 | abstract-1 | 86 | 162 | Strongest explicit anti-single-model framing ("a controlled study, not a single run"), but the lead crams E88 + the full class definition into one dense clause and the body runs 2 words long. |
| 6 | abstract-5 | 84 | 163 | Distinctive fragment hook ("A single workstation-class GPU, 23 days, 0.974 bits per byte") and a sharp "the barrier was the parallelization axis, not the nonlinearity", but the densest lead clause and the longest body (3 over). |
| — | abstract-4 | N/A | — | No file produced; nothing to grade. |
| — | abstract-6 | N/A | — | No file produced; nothing to grade. |

All six extant candidates are genuinely strong (each passed its own eval), so
scores cluster high; the spread reflects mild differentiators in lead
distinctiveness, length discipline, and parity phrasing rather than any
substantive correctness gap. No candidate invents facts; every number
(0.974 bpb, ~23 days, 1.273 B, 22,200 programs/token, 0.79 vs 0.22 on S₅) is
confirmed against `paper/main.typ`.

## Top 3 — full text

### 1. abstract-2 (93)

> A 1.273-billion-parameter recurrent language model with no attention and no time-axis scan reaches 0.974 bits per byte on The Pile after about 23 wall-clock days on a single workstation-class GPU. That model, E88, is the production instance of the Emender, a class of nonlinear-in-time recurrent layers whose throughput comes not from linearizing time but from running 22,200 small recurrent programs per token, each stepping its own serial time loop — refuting the assumption that pure-nonlinear-in-time recurrence cannot reach billion-parameter scale at competitive wallclock. This is a controlled study: three 1.3B-class architectures — E88, raw-write M²RNN-CMA, and linear-recurrent Gated DeltaNet — trained under matched per-architecture CMA-ES, with E88 holding the same loss-vs-wallclock band as Gated DeltaNet (parity, not a win). Within the class we derive an ordering: a Lean 4 trusted core proves the delta-correcting update reaches a strictly larger one-step function class than raw-write at matched FLOP, confirmed on an 8M state-tracking probe (0.79 versus 0.22 on S5).

### 2. abstract-8 (91)

> On one workstation-class GPU, a 1.273-billion-parameter recurrent model drops below one bit per byte on The Pile — 0.974, after about 23 wall-clock days, with no cluster and no time-axis scan. The model is E88, the production instance of the Emender, a class of pure-nonlinear-recurrent layers that correct their state rather than refill it. The throughput route is width-axis multi-programming: ~22,200 small recurrent programs per token, each serial in time — so nonlinear-in-time recurrence reaches this scale on competitive wallclock, a regime long treated as closed. It is a controlled study: E88, M²RNN-CMA, and Gated DeltaNet, each tuned by its own CMA-ES, E88 holding the same loss-vs-wallclock band as Gated DeltaNet — parity, not a win. And the class carries its own ordering, derived then confirmed: a Lean 4 trusted core proves the delta-correcting update spans a strictly larger one-step function class than raw-write at matched FLOP, borne out on an 8M state-tracking probe (0.79 vs 0.22 on S5).

### 3. abstract-7 (90)

> A pure-nonlinear-recurrent language model reaches 0.974 bits per byte on The Pile after 23 days on a single workstation-class GPU — no cluster, no sequence parallelism. That model is E88, the 1.273-billion-parameter instance of the Emender, a family of nonlinear-recurrent layers that update memory by delta correction, not raw write. Throughput comes from width-axis multi-programming: each token drives 22,200 small recurrent programs in parallel while time stays serial inside each, refuting the assumption that pure-in-time nonlinear recurrence cannot reach billion-parameter scale on competitive wallclock. It is a controlled study of three 1.3-B-class architectures — E88, raw-write M²RNN-CMA, and linear-recurrent Gated DeltaNet — each tuned under matched per-architecture CMA-ES; E88 holds the same loss-vs-wallclock band as Gated DeltaNet, on par rather than ahead. Within the class, a Lean 4 trusted core proves delta correction reaches a strictly larger one-step function class than raw write at matched compute, confirmed on an 8M probe (0.79 vs 0.22 on the S₅ word problem).

## Recommended synthesized abstract

Grafts abstract-2's clean, result-first lead and its explicit "controlled
study" framing; abstract-8 / abstract-7's economical mechanism gloss
("correct their state by delta rather than overwrite"); and the
proved-then-confirmed ordering clause shared by abstract-2 and abstract-3.
160 words (body), grounded in `paper/main.typ`, parity stated as no-penalty.

> **A 1.273-billion-parameter recurrent language model with no attention and no time-axis scan reaches 0.974 bits per byte on The Pile after about 23 days on a single workstation-class GPU. That model is E88, the production instance of the Emender, a class of pure-nonlinear-recurrent layers that correct their state by delta rather than overwrite. Throughput is width-axis multi-programming — 22,200 small recurrent programs per token, each stepping its own serial time loop — refuting the assumption that such recurrence cannot reach billion-parameter scale at competitive wallclock. This is a controlled study: three 1.3B-class architectures — E88, raw-write M²RNN-CMA, and linear-recurrent Gated DeltaNet — trained under matched per-architecture CMA-ES, with E88 holding the same loss-vs-wallclock band as Gated DeltaNet (parity, not a win). Within the class we derive an ordering: a Lean 4 trusted core proves the delta-correcting update reaches a strictly larger one-step function class than raw-write at matched FLOP, confirmed on an 8M state-tracking probe (0.79 versus 0.22 on S5).**

### Why this wins on every dimension

- **Punchy lead** — opens on the single-GPU sub-1-bpb result with the two
  things the literature says you need and this model omits ("no attention and
  no time-axis scan"); E88 is named in the second sentence as a reveal, not
  crammed into the hook.
- **Names E88 + the class** — "E88, the production instance of the Emender, a
  class of pure-nonlinear-recurrent layers", with the delta-correcting
  mechanism stated in one clause.
- **Enumerates the real contributions** — (1) multi-programming viability,
  (2) the matched-CMA-ES three-architecture controlled study, (3) the
  Lean-4-derived one-step ordering confirmed on the 8 M S₅ probe.
- **Reads as a study, not one model** — "This is a controlled study: three
  1.3B-class architectures … trained under matched per-architecture CMA-ES."
- **Length / legibility** — 160 words, one idea per sentence.
- **Honesty** — parity stated flatly as "parity, not a win"; no "existence
  proof" phrasing; every figure is sourced from the paper; the ordering is
  presented as derived-then-confirmed, not overclaimed.

*Grounding note: `paper/main.typ` is **not** modified by this node. The human
approves exact wording before any change to the paper.*
