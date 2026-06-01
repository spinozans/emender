# Abstract Candidate Report — "Emending Nonlinear Recurrence"

Reporter, not judge. This lays out what the six writers produced and how each
relates to the points the author cares about. No winner is chosen, nothing is
ranked as a verdict, no merged abstract is synthesized, and `main.typ` is
untouched. Observations only; the author decides.

Grounding facts used as the reference (from `paper/main.typ`):

- Existence / viability: E88 (1.273 B) reaches 0.973 bpb on The Pile after
  ~23 stitched GPU-days on a single workstation-class GPU; sub-1-bpb band
  (E88 0.973, GDN 0.973, M²RNN-CMA 0.979). Width-axis multi-programming:
  22,200 small programs per token in E88 (§5).
- Held-out statistical tie: on held-out bpb the three 1.3 B-class architectures
  tie; bulk language-model loss does not separate them (§5).
- Two axes (§6/§7): linear-vs-nonlinear = computability (Barrington / NC¹, the
  linear recurrence provably cannot track non-solvable S₅ at length — converged
  ceiling, not undertraining); delta-vs-raw-write = learning efficiency at
  matched FLOP, both nonlinear and both in principle able, raw-write under-reaches
  (this is NOT a can/can't claim).
- Deployed 1.3 B expressivity: on the actual released v0.3 weights, E88 (delta)
  length-generalizes on S₅ (0.965 at trained T=64 → 0.162 at T=512, ~20× chance);
  GDN (linear) converges to a wall (collapses to ~0.038, near chance); M²RNN
  (raw-write) fits short S₅ but under-reaches with length (§6, `tab_s5_1p3b`).
- Lean 4 trusted core: delta update reaches a strictly larger one-step function
  class than raw-write at matched compute (§7).

Style rules being checked: no em-dashes, no bold/emphasis markup, precise
property-based definition of nonlinear-in-time (no vague "stay"), no PNIT/PNR
acronym, length in the writers' 120–180 word band. (Semicolons are not
prohibited by the task; counts are noted for transparency only.)

---

## Candidate 1 (abstract-1.md)

- Seed: "stillness, one disturbance, then stillness; angle: the held-out
  bulk-loss tie as the quiet, with the two-axis state-tracking separation as
  the only real sound."
- Lead sentence: "A single workstation-class GPU is enough."
- Foregrounds: the viability/affordability claim first, then a quiet→tie→
  separation arc. Shortest of the six; clipped, declarative sentences.

Against the author's points:
- Held-out lead / train-loss demoted: HELD-OUT present ("The measurement is
  held out"). Does not explicitly call training loss a diagnostic, but demotes
  bulk loss ("bulk loss goes quiet"). Does not lean on the wallclock racer.
- Held-out statistical tie: PRESENT and precise. "A delta-correcting model, a
  raw-write model, and a linear-recurrent model stand in a statistical tie, and
  language-model bits do not separate them."
- Two-axis framing: PRECISE. "Computability: the linear recurrence cannot track
  non-solvable group state at length, a converged ceiling, not undertraining.
  Efficiency: ... the raw-write update, which in principle can, under-reaches."
  Explicitly says raw-write "in principle can" → no can/can't error.
- Deployed 1.3 B result: PARTIAL/IMPLICIT. The viability number is the 1.3 B
  E88, but the length-generalization claim is stated generically and not pinned
  to the deployed 1.3 B scale; S₅ not named ("non-solvable group state").
- Existence proof / viability: PRESENT (single GPU, sub-1-bpb, width-axis
  multi-programming).
- "Loss is blind, a targeted probe is not" thesis: PRESENT implicitly ("bulk
  loss goes quiet ... The separation is elsewhere, on two axes"). Not stated as
  an explicit aphorism.
- Forward-looking idea (cheap exploration of update rules): ABSENT (so no
  overreach).
- Open questions: uses "below one bit per byte" (no exact bpb number); does not
  relate to GPT-2 or other named models.
- Style: 159 words. 0 em-dashes, 0 semicolons, no bold. Nonlinear-in-time
  defined by property ("state passes through a nonlinearity at every step"); no
  vague "stay". No PNIT/PNR acronym.
- Honesty: no overclaim detected.
- Notable strength: most compressed, strong rhythm; the "in principle can"
  guard on raw-write is the most explicit can/can't safeguard of the six.
- Notable risk: omits the explicit "training loss is a diagnostic" sentence and
  does not name S₅ or the deployed 1.3 B scale, so the held-out and
  deployed-scale framings are carried by implication rather than statement.

---

## Candidate 2 (abstract-2.md)

- Seed: "a fragment that nonetheless sees you and turns its gaze into a verdict;
  angle: foreground that bulk loss is silent and that the two real distinctions
  are computability and learning efficiency."
- Lead sentence: "A language model whose recurrence is nonlinear in time,
  meaning the state passes through a nonlinearity at every step ..."
- Foregrounds: an explicit in-line definition of "nonlinear in time" up front,
  then the held-out comparison as the deciding measurement.

Against the author's points:
- Held-out lead / train-loss demoted: HELD-OUT foregrounded strongly ("On
  held-out bits per byte, the measurement that decides the comparison"). Does
  not explicitly name training loss as a diagnostic. Does not lean on the racer.
- Held-out statistical tie: PRESENT and precise ("form a statistical tie; bulk
  language-model loss does not tell them apart").
- Two-axis framing: PRECISE and labeled. "Linearity is a question of
  computability ... The write rule is a question of efficiency." Raw-write
  framed as "under-reaches" (efficiency), no can/can't error. Does not use the
  explicit "in principle able" phrase, but the efficiency framing is clear.
- Deployed 1.3 B result: PARTIAL/IMPLICIT (generic length-generalization; S₅
  and the deployed-scale tie not named).
- Existence proof / viability: PRESENT.
- Thesis ("loss is blind"): PRESENT implicitly ("bulk language-model loss does
  not tell them apart. Two distinct axes do").
- Forward-looking idea: ABSENT (no overreach).
- Open questions: "below one bit per byte" (no number); no named models.
- Style: 174 words. 0 em-dashes, 1 semicolon, no bold. Nonlinear-in-time given
  the most explicit property definition of the six ("meaning the state passes
  through a nonlinearity at every step"). No vague "stay". No acronym.
- Honesty: no overclaim detected.
- Notable strength: the clearest standalone definition of nonlinear-in-time and
  the cleanest "computability vs efficiency" labeling of the two axes.
- Notable risk: front-loaded definitional clause makes the first sentence long;
  the Lean line drops "than raw-write" ("a strictly larger one-step function
  class at matched compute"), leaving the comparison target implicit.

---

## Candidate 3 (abstract-3.md)

- Seed: "immense force held inside exact structure; angle: the two axes bulk
  loss hides, computability versus learning efficiency."
- Lead sentence: "A recurrent language model whose state passes through a
  nonlinearity at every step ... reaches below one bit per byte on The Pile on
  a single workstation-class GPU."
- Foregrounds: mechanism then result in one breath; then the most explicit
  train-loss demotion of the six.

Against the author's points:
- Held-out lead / train-loss demoted: BOTH EXPLICIT. "The measurement is
  held-out bits per byte; training loss is only a diagnostic." Strongest match
  to the author's "held-out focus, training loss demoted to diagnostic" point.
- Held-out statistical tie: PRESENT and precise ("are a statistical tie, so
  bulk language-model loss does not separate them").
- Two-axis framing: PRECISE and the most carefully hedged. "Delta-correcting
  against raw-write, both nonlinear and both in principle able, is learning
  efficiency." Explicit "in principle able" → no can/can't error; also names
  computability as "provably cannot" for the linear case only.
- Deployed 1.3 B result: PARTIAL/IMPLICIT (length-generalization stated
  generically; S₅ not named; not pinned to deployed 1.3 B).
- Existence proof / viability: PRESENT, with extra mechanism detail ("each a
  bounded matrix tile, the time loop serial inside each").
- Thesis ("loss is blind"): PRESENT, well-aligned to the seed ("the two axes
  bulk loss hides").
- Forward-looking idea: ABSENT (no overreach).
- Open questions: "below one bit per byte" (no number); no named models.
- Style: 179 words (near the top of the band). 0 em-dashes, 1 semicolon, no
  bold. Property-based definition; no vague "stay"; no acronym. Lean line is the
  most complete ("strictly larger one-step function class than raw-write at
  matched floating-point cost").
- Honesty: no overclaim detected; "provably cannot" / "in principle able" split
  matches the paper precisely.
- Notable strength: tightest fidelity to the two-axis precision and the
  diagnostic-demotion point simultaneously.
- Notable risk: densest of the six; near the upper length bound.

---

## Candidate 4 (abstract-4.md)

- Seed: "independent voices, each complete, converging into one rigorous
  architecture; angle: the held-out tie as the pivot, then two cleanly
  separated axes resolving onto the proved one-step separation."
- Lead sentence: "A recurrent language model whose state passes through a
  nonlinearity at every step, foreclosing the parallel scan that linear
  recurrences depend on, reaches below one bit per byte on The Pile ..."
- Foregrounds: short, separately-punctuated claims; held-out tie as a pivot
  into the two axes.

Against the author's points:
- Held-out lead / train-loss demoted: BOTH EXPLICIT, in two short sentences.
  "We measure on held-out data. Training loss is only a diagnostic."
- Held-out statistical tie: PRESENT and precise; enumerates the three
  architectures ("two nonlinear in time, one delta-correcting and one
  raw-write, and one linear recurrence").
- Two-axis framing: PRECISE. "Delta-correcting against raw-write, both able in
  principle, is learning efficiency." Explicit "both able in principle" → no
  can/can't error. Linear case "provably cannot ... a converged ceiling rather
  than undertraining."
- Deployed 1.3 B result: PARTIAL/IMPLICIT (generic length-generalization; S₅
  not named; deployed-scale not pinned).
- Existence proof / viability: PRESENT.
- Thesis ("loss is blind"): PRESENT ("Bulk language-model loss does not
  separate them. Two distinct axes do").
- Forward-looking idea: ABSENT (no overreach).
- Open questions: "below one bit per byte" (no number); no named models.
- Style: 171 words. 0 em-dashes, 0 semicolons, no bold. Property-based
  definition; no vague "stay"; no acronym. Lean line complete with "than
  raw-write at matched FLOP cost".
- Honesty: no overclaim detected.
- Notable strength: the most readable cadence (short declarative sentences) of
  the candidates that also hit both the diagnostic-demotion and explicit
  in-principle-able points; zero semicolons.
- Notable risk: like the others, the length-generalization claim is not pinned
  to the deployed 1.3 B weights and S₅ is unnamed.

---

## Candidate 5 (abstract-5.md)

- Seed: "many independent units cohering into one moving shape; angle: the
  width-axis multitude as the opening image, then the held-out tie that splits
  cleanly into a computability axis and a learning-efficiency axis."
- Lead sentence: "Hundreds of small recurrent programs run inside each token,
  each turning its bounded matrix state through a nonlinearity at every step."
- Foregrounds: the width-axis multi-programming image first (the only candidate
  that opens on the programs rather than on the model or the GPU).

Against the author's points:
- Held-out lead / train-loss demoted: BOTH EXPLICIT ("We measure on held-out
  data and treat training loss as diagnostic only").
- Held-out statistical tie: PRESENT ("tie statistically, and bulk
  language-model loss does not separate them").
- Two-axis framing: PRECISE. "Linearity sets computability ... The write rule
  sets learning efficiency: at matched budget the delta correction
  length-generalizes on S5 ... while raw-write under-reaches." Raw-write framed
  as efficiency/under-reaches, no can/can't error. Does not use the explicit
  "in principle able" phrase, but the linear case is "provably cannot" and the
  write-rule axis is labeled efficiency.
- Deployed 1.3 B result: PARTIAL but the STRONGEST of the six on naming. Only
  candidate to name S₅ explicitly ("non-solvable S5 state", "length-generalizes
  on S5"). Still not pinned to the deployed 1.3 B weights.
- Existence proof / viability: PRESENT, and uniquely adds "attention-free
  recurrent model with no time-axis scan".
- Thesis ("loss is blind"): PRESENT ("The result is flat ... bulk
  language-model loss does not separate them. Two axes do").
- Forward-looking idea: ABSENT (no overreach).
- Open questions: "below one bit per byte" (no number); no named models.
- Style: 180 words (top of the band). 0 em-dashes, 1 semicolon, no bold.
  Property-based definition; no vague "stay"; no acronym. Note: writes the group
  as "S5" (bare, no subscript), the only candidate to surface the symbol.
- Honesty: no overclaim detected.
- Notable strength: the only candidate that names S₅ and that leads on the
  width-axis multi-programming image; vivid opening.
- Notable risk: at the maximum word count; "S5" appears unsubscripted as plain
  text (formatting the author may want to normalize); does not state the
  explicit "in principle able" guard (relies on the efficiency framing instead).

---

## Candidate 6 (abstract-6.md)

- Seed: "Euler identity, five constants on one line, nothing wasted; angle:
  foreground the held-out tie, then split the two distinct axes computability
  and learning efficiency."
- Lead sentence: "A recurrent language model whose state passes through a
  nonlinearity at every step ... reaches below one bit per byte on The Pile on
  a single workstation-class GPU."
- Foregrounds: maximal economy; ends by locating the Lean proof "at its root"
  of the second (efficiency) gap.

Against the author's points:
- Held-out lead / train-loss demoted: HELD-OUT present ("The measurement is
  held-out bits per byte"). Does NOT explicitly call training loss a diagnostic
  (one of three that omit the diagnostic sentence, with 1 and 2). Does not lean
  on the racer.
- Held-out statistical tie: PRESENT but the loosest wording: "two such
  nonlinear models and a linear-recurrent baseline tie" uses "tie" without the
  qualifier "statistical" (the other five say "statistical tie"). Still says
  bulk loss "does not tell them apart".
- Two-axis framing: PRECISE. "Computability: the linear recurrence cannot track
  non-solvable group state ... Learning efficiency: between two nonlinear writes
  at matched budget, the delta-correcting update length-generalizes ... while
  raw-write under-reaches." Raw-write framed as efficiency, no can/can't error.
  No explicit "in principle able" phrase.
- Deployed 1.3 B result: PARTIAL/IMPLICIT (generic; S₅ not named; deployed scale
  not pinned).
- Existence proof / viability: PRESENT.
- Thesis ("loss is blind"): PRESENT ("bulk language-model loss does not tell
  them apart. Two axes do").
- Forward-looking idea: ABSENT (no overreach).
- Open questions: "below one bit per byte" (no number); no named models.
- Style: 153 words (shortest with candidate 1). 0 em-dashes, 0 semicolons, no
  bold. Property-based definition; no vague "stay"; no acronym.
- Honesty: no overclaim detected. The "proves the second gap at its root"
  framing correctly ties the Lean result to the efficiency axis specifically.
- Notable strength: most economical; the only one that explicitly connects the
  Lean one-step separation to "the second gap" (the efficiency axis) rather than
  leaving the proof free-floating.
- Notable risk: drops the word "statistical" from the tie, and omits the
  explicit training-loss-is-a-diagnostic sentence; the held-out framing is
  thinner than in 3, 4, 5.

---

## Cross-candidate observations

What they CONVERGED on (covered by all six):
- The existence/viability result: sub-1-bpb on The Pile, single workstation-class
  GPU, width-axis multi-programming ("hundreds of small recurrent programs per
  token"). Universal.
- Leading on held-out and NOT leaning on the train-loss / wallclock racer. None
  of the six leads with the loss-vs-wallclock comparison; all route the
  comparison through held-out bpb.
- The held-out tie: all six state that the three architectures tie and that bulk
  language-model loss does not separate them (candidate 6 omits the word
  "statistical" but keeps the claim).
- The two-axis framing with correct polarity: linear-vs-nonlinear = computability
  (a "provably cannot" / "converged ceiling, not undertraining" claim about the
  LINEAR recurrence), delta-vs-raw-write = learning efficiency.
- The can/can't flag (the error to catch): NONE of the six commits it. Every
  "cannot / provably cannot" is attached to the linear recurrence's computability
  limit; raw-write is uniformly framed as "under-reaches" / efficiency. Candidates
  1, 3, and 4 add an explicit "in principle can / both able in principle" guard;
  candidates 2, 5, and 6 rely on the efficiency framing without that explicit
  phrase but still do not misframe raw-write as "cannot".
- The Lean 4 one-step separation (delta > raw-write function class at matched
  compute): all six. (Candidate 2 leaves the "than raw-write" comparator
  implicit; candidate 6 uniquely ties it to "the second gap".)
- Style compliance: all six have 0 em-dashes, 0 bold/emphasis, a property-based
  definition of nonlinear-in-time (no vague "stay"), and no PNIT/PNR acronym.
  Word counts span 153–180, all inside the writers' 120–180 band. Semicolons
  (not prohibited): candidates 2, 3, 5 use 1 each; 1, 4, 6 use none.

What VARIED:
- Opening image. Candidate 1 opens on affordability ("A single workstation-class
  GPU is enough"); 2 opens on a definition of nonlinear-in-time; 3, 4, 6 open on
  mechanism-then-result; 5 is the only one to open on the width-axis program
  multitude.
- Train-loss-as-diagnostic sentence. EXPLICIT in 3, 4, 5; IMPLICIT (demoted but
  not named) in 1, 2, 6.
- Naming S₅. Only candidate 5 names the symmetric group S₅ (as bare "S5"); the
  other five say "non-solvable group state" / "non-solvable state".
- Explicit "in principle able" guard on raw-write: present in 1, 3, 4; absent
  (efficiency framing only) in 2, 5, 6.
- Length / density. 6 and 1 are the leanest (153, 159); 5 and 3 the fullest
  (180, 179).
- Lean comparator completeness. 3 and 4 spell out "than raw-write at matched
  FLOP/floating-point cost"; 2 omits the comparator; 6 reframes it as proving
  "the second gap at its root".

Author-points WELL-COVERED by some candidate:
- Held-out focus + train-loss demoted to diagnostic: candidates 3, 4, 5 state it
  explicitly.
- Two-axis precision with the explicit can/can't guard: candidates 1, 3, 4.
- Naming the S₅ probe: candidate 5.
- Tying the Lean proof to the efficiency axis specifically: candidate 6.
- Most complete Lean statement (delta > raw-write at matched cost): candidates
  3, 4.

Author-points covered by NONE (gaps for the author to weigh):
- The DEPLOYED 1.3 B expressivity result as such. All six carry the viability
  number at 1.3 B (E88 sub-1-bpb), and all six describe length-generalization,
  but none pins the length-generalization / under-reach / converged-ceiling
  result to the deployed 1.3 B released weights (the §6 `tab_s5_1p3b` result).
  The expressivity claims read as scale-agnostic; the distinct "same separation
  survives on the actual production weights" point is not surfaced by any
  candidate.
- The "loss is blind, a targeted probe is not" thesis as an explicit, named
  statement. All six carry it by implication ("bulk loss does not separate them,
  the axes do"), but none states the probe-vs-loss thesis as a standalone line.
- The forward-looking idea (a wide class of update rules is cheap to explore in
  this parallel framework). ABSENT from all six. This means none overreaches on
  it (a positive for honesty), but if the author wants the forward-looking note
  in the abstract, no candidate currently supplies it.
- An exact bpb number. All six use "below one bit per byte" / "sub-1-bpb"; none
  gives 0.973. (This matches one of the author's open questions; no candidate
  forces the decision.)
- Relating to known models (GPT-2 etc.). None does. (Again matches an open
  question left open.)

Honesty / overclaim across the set: no overclaims detected in any candidate. The
linear "provably cannot" is correctly scoped to computability, raw-write is
correctly scoped to efficiency, the tie is reported as a tie, and the absent
forward-looking idea is absent rather than overstated.
