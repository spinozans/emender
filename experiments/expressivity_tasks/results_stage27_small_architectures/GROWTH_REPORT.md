# X27 Growth Report — Small-Architecture Size Grid Outcome

Registration commit **4b8ef784** (pushed before launch; ancestry verified).
X27/X27q machinery reviews accepted (X27R, X27QR; the X27q in-place
`source_sha256` refresh was the single supervisor-authorized append-only
exception, receipted in the ledger). This report adjudicates the terminal
small-arm outcomes against the X25 big arms and the pre-registered
predictions. size verdicts per registration. Batch 64 retained for the X25<->X27 size A/B
(concession #19: the task directive's batch-128 was corrected; X26/X26b
ran batch 128, X26c ran their raw default batch 256 — cross-recipe
comparisons are qualitative only).

## Arms

| arm | params | arch | activation | card |
|---|---|---|---|---|
| e97_graph_seed0 (small-tanh) | 570,464 | E97 depth1 dim192 | tanh | 6 |
| e97softplus_graph_seed0 (small-softplus) | 570,464-class | E97 depth1 dim192 | softplus | 0 |
| gdn2_graph_seed0 (small-GDN2) | 563,218 | GDN2 depth1 dim128 | linear | 2 |
| (X25 big comparators) | 8,950,272 (E97) / 8,936,528 (GDN2) | depth4 | tanh/softplus/linear | 1,7 |

Size ratio ~15.7x (concession #20: earlier 9.9M/17x figures were supervisor
errors). Same frozen graph-connectivity datasets (hash-identical to X25),
paper recipe lr 1e-4, 60,000-update cap.

## Terminal receipts

- `e97_graph_seed0`: COMPLETE, 60,000 updates (`60000_update_cap`).
- `gdn2_graph_seed0`: COMPLETE, 60,000 updates (`60000_update_cap`).
- `e97softplus_graph_seed0`: **CENSORED — infra death, not a training
  verdict.** FloatingPointError (nonfinite logits) at update 12,661.
  Fail-closed: the arm is NOT adjudicated as failed-to-learn. Receipted in
  its `result.json` and terminal receipts.
- Fold-in (big comparators, now terminal): **X25 big-GDN2 COMPLETE at
  60,000 updates** (checkpoint selected at 53,500; best VALIDATION 0.5325
  — validation accuracy, distinct from the selected-checkpoint test
  metrics in the table: tests 0.5240/0.5000/0.5000). It never left chance. X25
  big-tanh's verdicts (100% ID at ~step 9.5k; OOD collapse 56%->49.8%)
  unchanged from the X25 report.

## Results (graph connectivity task)

| arm | ID acc | OOD1 | OOD2 | verdict |
|---|---|---|---|---|
| big-E97-tanh (X25) | 1.0000 | 0.5602 | 0.4979 | learned ID; OOD degradation across bins |
| **small-E97-tanh (X27)** | **0.5226** | 0.5000 | 0.5000 | **never learned** |
| big-GDN2 (X25, terminal) | 0.5240 | 0.5000 | 0.5000 | never learned |
| **small-GDN2 (X27)** | **0.5287** | 0.5005 | 0.4989 | **never learned** |
| small-E97-softplus (X27) | — | — | — | CENSORED (nonfinite @ 12,661) |
| big-E97-softplus (X25) | 0.5143 @ 17k/60k | — | — | still training (card 7) |

(Graph task classes are balanced; accuracy = balanced accuracy here.)

## Adjudication vs pre-registered outcomes

- **size-binding: REJECTED, with the sign reversed.** The prediction was
  small-ours >= 99% ID where big-ours fail ("too big" hypothesis). The
  receipts show the opposite: small-tanh FAILED (52.3% at 60k) where
  big-tanh reached 100% ID at ~9.5k steps on the same recipe, same data,
  same batch. The phase transition is capacity-gated: 570k params did not
  reach it; 8.95M did. Capacity was NECESSARY, not obstructive. The
  train-small-then-transplant production hypothesis is not supported by
  this grid and is downgraded.
- **architecture-binding: partially supported, within our arms.** GDN2
  failed at BOTH sizes (big 53.3%, small 52.9%) while E97-tanh succeeded at
  big on the same task — the in-distribution architecture separation
  (nonlinear-state learned the L-complete task; DPLR never did, at any
  size or recipe tested) is the session's only theory-aligned empirical
  cell and now rests on four data points. However, the registered
  cross-check ("X26 RNN learns/generalizes") did not materialize — their
  models failed at all three fidelity levels (X26/X26b/X26c) — so this
  binding is an our-arms comparison with cross-recipe limits retained.
- **nothing-learns: not the grid outcome for tanh** (big learned); GDN2
  cells align with it locally.
- **other (retained clauses)**: exact counts recorded for all cells; ID
  failure separated from OOD degradation; the softplus-small error is
  censored, not a scientific verdict.

## The softplus-small death as a datum

The observed facts: nonfinite-logits FloatingPointError at update 12,661
(unbounded-activation state, no squashing bound, 90,904-token sequences,
small capacity), while the big-softplus arm survives the same recipe.
Possible explanations — insufficient magnitude management at low capacity,
capacity as the stabilizer — are UNTESSED HYPOTHESES; the error receipt
establishes only the nonfinite death, not a mechanism or remedy. Any
replacement small-softplus arm requires a NEW registration (not
automatic); the fail-closed censor stands until then.

## Open cells

- big-E97-softplus (card 7): the activation thesis's live test; ~17k/60k
  at this report, no transition yet (its tanh twin transitioned at 9.5k).
  Its terminal verdict feeds the unified separator report and the X28
  memory-head training design (if it never transitions, pre-registered
  outcome (c) fires: unbounded state alone is not learnable at this recipe
  — staging required).

## Scope and limits

n=1 seed per cell; single host; qualitative size verdicts per registration.
No claim that 60k is the global training ceiling (right-censoring at the
cap retained). Cross-recipe comparisons (batch 128 upstream arms) remain
qualitative-only per concession #19.
