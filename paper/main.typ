// Emender — main paper source
// Format: Typst (https://typst.app), NOT LaTeX.
// Build: bash paper/build.sh  →  paper/Garrison_2026_Emender-<commit>.pdf
//
// Template: arkheion 0.1.2 (the de-facto Typst arxiv-preprint template).
// See paper/template_choice_v7.md for rationale and the porting notes.

#import "@preview/arkheion:0.1.2": arkheion, arkheion-appendices

// ALTERNATE ABSTRACTS — kept for A/B reference. The active (rendered)
// abstract is the result-first version below (passed to arkheion).
//
/*
=== ALTERNATE 1 — Verdict-rebuttal opener (v11 version) ===

The dominant operating verdict in recurrent language modeling is
that pure-nonlinear-in-time recurrence cannot reach foundation-model
scale on competitive wallclock, because it forecloses the time-axis
parallel scan that linear-recurrent variants depend on for GPU
throughput. We test this verdict by training three pure-recurrent
language models in the 1.3 B-class under per-architecture
CMA-ES hyperparameter search (cohort band, not equal exact size: E88
1.273 B, M²RNN-CMA 1.307 B, GDN 1.352 B): two with nonlinear time recurrence,
*the Emender* (delta-correcting update $S <- tanh(d S + k("silu"(v) - S^T k)^T)$)
and *M²RNN-CMA* (raw-write update $tanh(H W + k v^T)$); and one
with linear time recurrence, *Gated DeltaNet* (GDN). All three
land in the same loss-vs-wallclock band on The Pile, so at this
scale and training extent, nonlinearity in time is not the
wallclock barrier it was assumed to be. The systems contribution that makes this
possible is *multi-programming*, a width-axis parallelization that
replicates the recurrent computation across many independent heads
while keeping the time loop serial inside each head; this replaces
the time-axis linearization that linear recurrences exploit for
throughput. Within the pure-nonlinear-recurrent class, the Emender trains
consistently ahead of M²RNN-CMA, and a one-step representability
separation between the delta-correcting and raw-write update rules,
formalized in Lean 4, is confirmed empirically on
capacity-overparameterized state-tracking probes. The three 1.3 B-class
checkpoints, the per-architecture CMA-ES configurations, and the Triton
multi-programming kernel are assigned to the v0.3 HuggingFace release
target; the trusted Lean 4 core has no
sorry/admit/axiom/opaque/native_decide in the import closure.

=== ALTERNATE 2 — Tight ~110-word cold-lead variant ===

We train three pure-recurrent 1.3 B-class language models (cohort
band, not equal exact size: E88 1.273 B, M²RNN-CMA 1.307 B, GDN
1.352 B) on The Pile and find that nonlinearity in time is not a cost: two
nonlinear-in-time recurrences (the Emender, M²RNN-CMA) land in the same
loss-vs-wallclock band as the linear-recurrent baseline Gated
DeltaNet under per-architecture CMA-ES. The field's operating
verdict — that pure-nonlinear-in-time recurrence cannot reach this
scale on competitive wallclock — is an artefact of parallelizing
the time axis; width-axis multi-programming recovers throughput
while the time loop stays serial. Within the nonlinear class, the Emender
trains consistently ahead of M²RNN-CMA, with the one-step
representability separation formalized in Lean 4. Checkpoints,
CMA-ES configs, and the Triton kernel released.
*/

#show: arkheion.with(
  title: "Emending Nonlinear Recurrence",
  authors: (
    (
      name: "Erik Garrison",
      email: "erik.garrison@gmail.com",
      affiliation: "Poietic PBC / UTHSC, Memphis, TN, USA",
      orcid: "0000-0003-3821-631X",
    ),
  ),
  abstract: [
    Recurrent neural networks read a sequence one step at a time,
    updating an internal state as they go. Training them efficiently at
    scale usually requires the state update to be linear, so the whole
    sequence can be processed in parallel. A recurrence whose update is
    nonlinear loses this parallelism and must run serially, which has
    been assumed too slow to be practical at billion-parameter scale.
    It has been unclear whether a purely nonlinear recurrence can be
    trained to competitive scale on its own, and whether the form of
    its update rule matters once it is. Here we show that a purely
    nonlinear recurrent language model, made trainable by running
    hundreds of small recurrent programs in parallel across the
    network's width rather than across time, reaches below one bit per
    byte on a standard corpus using a single workstation-class GPU,
    matching strong linear-recurrent baselines. On held-out text these
    models are statistically tied, so language-modeling loss does not
    distinguish them; their differences emerge instead in state
    tracking. A linear recurrence provably cannot follow certain
    structured state as sequences grow, whereas a nonlinear one can.
    Among the nonlinear rules, a delta-correcting update learns this
    far more efficiently than a plain overwrite. A machine-checked proof
    establishes the enabling direction: the delta-correcting update
    is strictly more expressive per step than plain overwrite and can realize this
    structured state-tracking. The matching impossibility for linear
    recurrence rests on classical complexity results, and in the trained
    billion-parameter models we observe the same ordering. These findings indicate that loss is a
    poor guide to what a recurrent architecture can compute, and that
    massively parallel nonlinear recurrence is a practical and largely
    open design space.
  ],
  keywords: (
    "recurrent neural networks",
    "language modeling",
    "nonlinear recurrence",
    "state-space models",
    "expressivity",
    "Lean 4",
  ),
)

// ── Porting shims (preserve current paper's typographic conventions) ─────────
// Italic-as-bold fix: the author wrote `*…*` throughout intending italic
// emphasis, but in Typst `*…*` is the strong (bold) form. Override at the
// template level so every `*…*` renders as italic. Genuine bold survives
// because headings and other label uses set weight explicitly via
// `text(weight: "bold", …)` or `set text(weight: "bold")`, which bypass
// the `strong` element.
#show strong: it => emph(it.body)

// Bibliography heading is unnumbered (References, not "12. References").
#show bibliography: set heading(numbering: none)

// Display-math numbering: arkheion turns on "(1)"-style equation numbers by
// default. The current paper has zero `<eq:…>` labels and zero `@eq:…`
// cross-references, so adding visible (1)…(N) markers would be a content
// change. Restore the unnumbered display-equation style.
#set math.equation(numbering: none)

// Figure captions: smaller than body text (9pt vs ~11pt body) with visible
// vertical breathing room above and below. The "Figure 1:" label is rendered
// in semibold to match a polished arxiv preprint and to separate the label
// from the caption text. Addresses the author's "massive figure captions"
// feedback.
#show figure.caption: it => block(
  inset: (top: 0.4em, bottom: 0.6em, x: 0.5em),
  text(size: 9pt, [
    #text(weight: "semibold")[#it.supplement #context it.counter.display(it.numbering):]
    #h(0.35em)
    #it.body
  ]),
)

// Figure placement policy:
// Use Typst's float primitive for image figures by default. In-flow images stay
// exactly where written, which leaves page tails blank when a chart cannot fit
// in the remaining space. `auto` lets Typst choose the nearer page edge while
// preserving source order for accessibility and references. Table figures stay
// in flow so local derivation tables remain adjacent to their setup prose.
// Use explicit per-figure placement only when a figure has a real page-edge
// need, as with the large loss-vs-wallclock comparison below.
#set figure(placement: auto)
#show figure.where(kind: table): set figure(placement: none)

// Math shortcuts used throughout the body.
#let nd = $upright("Emender")$
#let s5 = $S_5$
#let s3 = $S_3$

// ── 1. Introduction ───────────────────────────────────────────────────────────
= Introduction <sec:intro>

A recurrent sequence model reads its input one step at a time, folding
each token into an internal state that summarizes everything seen so
far. How such a model can be trained, and how fast, turns almost
entirely on a single design choice: how that state is updated from one
step to the next. Over the past several years the field has converged
on updates that are *linear* in the state. A linear update can be
re-expressed as a parallel scan over the whole sequence at once instead
of a step-by-step loop, and that re-expression is what lets recurrent
models reach the training throughput of attention on modern hardware.
State-space models and gated linear-attention architectures — Mamba and
its successors @mamba2024 @mamba2_2024 @mamba3_2026, gated linear
attention @gla2023, RWKV @rwkv7_2025, and the delta rule and its gated
form @deltanet2024 @gated_deltanet2024 — now stand alongside the
transformer as the efficient backbone of choice for long sequences.

This efficiency is bought with a structural concession. A state update
that is linear in the state can be parallelized, but it also bounds what
the state can compute as the sequence grows: a fixed-precision
linear-in-time recurrence is asymptotically a weak recognizer that
cannot track certain structured state, the prefix products of a
non-solvable group being the canonical example @merrill2024transformers
@barrington1986. The direct way to lift that ceiling is to pass the
state through a nonlinearity at every step. But a nonlinear-in-time
update breaks the parallel scan and forces the time loop to run
serially, and serial training at billion-parameter scale was presumed
too slow to be practical. The field's response has been to route around
the obstruction rather than meet it: parallelize the time axis by other
means, such as Newton iteration on a block-bidiagonal Jacobian
@pararnn2025, or interleave recurrent layers with attention so the
hardest state-tracking work falls to the attention @m2rnn2026
@olmohybrid2026 @titans2025 @griffin2024.

Two questions are left open by this detour. First, whether a *purely*
nonlinear-in-time recurrence — no parallel scan, no attention to lean
on — can be trained to competitive scale at all, on a realistic
training budget. Second, once it can, whether the particular form of the
nonlinear update rule makes any difference to what the trained model can
do. Neither had been tested directly, because the pure setting had been
set aside as the expensive case to avoid rather than the case to
measure.

Here we show that it can be trained, and that the form of the update
does matter. A purely nonlinear recurrent language model, made trainable
by running many small recurrent programs in parallel across the
network's *width* rather than across time, reaches below one bit per
byte on a standard byte-level corpus (The Pile @thepile2020) using a
single workstation-class GPU, with no cluster and no sequence
parallelism. Trained at the billion-parameter class against strong
linear-recurrent baselines under matched per-architecture tuning, it
lands in the same loss-versus-wallclock band; on held-out text the
models are statistically tied, so language-modeling loss does not
separate them. Their differences emerge instead in state tracking, and
there the nonlinear recurrence reaches something a linear one provably
cannot: it can follow the prefix products of the non-solvable group
$S_5$ as sequences grow, where a linear-in-time recurrence is held away
from the task by a classical complexity argument. The form of the
nonlinear update matters as well — a *delta-correcting* update, which
writes the correction between what the memory predicts and what the
input demands, learns this structured tracking far more efficiently than
a plain overwrite, and the advantage holds across a large swing in
training budget. The separation is established in two directions: a
machine-checked proof that the delta-correcting update is strictly more
expressive per step than plain overwrite and can realize the
state-tracking construction, and the matching impossibility for linear
recurrence resting on classical complexity results. In the trained
billion-parameter models we observe the same ordering. Throughput on the
width axis comes from many small recurrent programs running side by side,
each stepping its time loop serially, which is what makes the serial
recurrence practical at scale.

The architecture is a residual stack of recurrent layers. Each layer
pairs a matrix-state memory with a write rule that has two halves.
*Delta correction* is the first half. The layer reads what the memory
predicts at the addressed slot, computes the prediction error, and
writes the correction. Memory can change its mind, retracting a stale
binding when the evidence forces it. *Tanh-with-latching* is the second
half. A slot driven near $plus.minus 1$ becomes insensitive to further
bounded writes, sub-threshold counter-input leaves the sign unchanged,
and a sufficient counter-delta releases the slot. Memory can stand
firm, holding a commitment across many irrelevant tokens, and remains
revisable when the data demand it. State tracking needs both. A memory
that can only add accumulates without correction; a memory that can
only correct cannot hold a fact long enough to use it. An *emender* is
one such layer; the *Emender* is the architecture family obtained by
stacking emender layers; *E88* is the 1.3 B instance
evaluated throughout the paper.

A Lean 4 trusted core is the spine of the argument. The core proves
that delta-correcting and raw-write updates separate at one step and at
every $k$-step composition at matched per-token FLOP cost; it proves
the three latching properties (saturation insensitivity,
sign-preserving hold, and counter-delta release); it proves that an
orthonormal-key configuration of the Emender realizes the $S_5$ prefix
tracker, the canonical NC#super[1] witness. The empirical work is a
separate trainability question: the Lean results establish realizability
and representational scope, while the $S_5$/$S_3$ probes and the 1.3 B
wallclock comparison measure what SGD actually finds under the stated protocols. At
parameter-matched 8 M scale the Emender reaches 0.79 accuracy on the
$S_5$ word problem against 0.36 for Gated DeltaNet and 0.22 for the
raw-write baseline, with the §6 floor argument making capacity
non-binding by orders of magnitude. The lever for scaling serial
nonlinear recurrence is parallelism within each time step, not across
it.

#heading(level: 2, numbering: none)[Delta correction is one response; hybrids are another]

The recurrent state-tracking problem admits more than one architectural
response. *Hybrid* architectures interleave recurrent layers with
attention layers and recover much of what attention-free recurrence
loses on state tracking and long-range mixing. M²RNN @m2rnn2026,
OLMo-Hybrid @olmohybrid2026, Titans @titans2025, and Griffin
@griffin2024 are recent instances, and the limit-of-pure-recurrence
story is documented by Merrill, Petty and Sabharwal
@merrill2024transformers. *Delta correction* is a different response.
It does not add cross-token mixing; it changes what the recurrent state
writes at each step from a raw additive overwrite to a correction
against the existing slot. The two responses are complementary, not
rival: a hybrid stack can use any update rule in its recurrent layers,
and delta correction is well-defined wherever a matrix-state recurrent
layer is.

We evaluate delta correction in the attention-free, time-serial
recurrent arena, the *pure nonlinear recurrent* (PNR) setting in the
sense of "no cross-token attention mechanism"; gating, projections,
normalizations and other intra-token nonlinear operations are
permitted and load-bearing (§3 ingredient (d) and §5 on gradient
conditioning). The PNR arena is where the contrast is cleanest (a
delta-vs-raw-write contrast at matched per-token FLOP class, §7),
where the Lean 4 separation theorems are tractable (sets C and C′),
and where the scaling demonstration leaves no architectural escape
hatch from the recurrent state.

The two recent attempts to scale nonlinear recurrence each took a
different concession. M²RNN @m2rnn2026 trains nonlinear matrix-state
recurrence at 7 B MoE in *hybrid form* (nonlinear-recurrent layers
interleaved with attention layers); ParaRNN @pararnn2025 trains
nonlinear-recurrent LSTM and GRU at 7 B by parallelizing the time loop
via Newton iteration on a block-bidiagonal Jacobian. In both cases the
pure-recurrent setting was set aside rather than filled. This work
takes the pure-recurrent route and compares a delta-correcting
instance directly with a raw-write instance under matched conditions.

#heading(level: 2, numbering: none)[Contributions]

The contributions are three results.

#set enum(numbering: "1.")

+ *Viability.* E88 reaches 0.973 bits per byte on The Pile after about
  23 wall-clock days of training on a single workstation-class
  GPU, with no cluster and no sequence parallelism. The throughput
  route is width-axis *multi-programming*: 22,200 small bounded
  recurrent programs per token (370 heads $times$ batch size 5
  $times$ depth 12), each a $32 times 32$ matrix-state tile that
  fits in registers, with the time loop serial inside each program
  (Lean-certified; see §4 and §7).
  Pure-nonlinear-recurrent language models train into this regime at
  this scale without a time-axis parallelization trick or attention
  hybridization.

+ *Power separation within the formalized resource class: for the
  matched-signature update family studied here, the delta-correcting
  update is strictly more expressive than raw-write at matched
  per-token compute.* A Lean 4 trusted core machine-verifies that no
  fixed-weight raw-write matrix RNN with row, column, or cell forget
  gates can match the Emender's delta-correcting update in one step
  (`emender_m2rnn_one_step_resource_separation_embeds`, set C in §7),
  and that the gap persists for every finite $k$ under composition on
  a constructed witness alphabet
  (`emender_m2rnn_k_step_separation`, set C′), at matched per-token
  FLOP class (`emender_m2rnn_flop_class_equiv`, set D). This is a
  representability statement inside the formalized fixed-weight
  raw-write resource class, not a claim of general or learned-weight
  superiority. Empirically the delta-vs-raw-write contrast is a
  matched-budget learning-efficiency gap rather than a categorical
  impossibility: at matched no-tuning budget raw-write under-reaches
  the delta update's $S_5$ length generalization, and even at a tuned
  best-effort budget it does not catch up (§5). The predicted ordering
  already appears at the 8 M parameter-matched probe shape where the §6
  floor argument makes capacity non-binding by orders of magnitude: the
  Emender reaches 0.79 on the $S_5$ word problem against 0.22 for
  raw-write M²RNN-CMA (§6).

+ *Supporting comparison.* Under per-architecture CMA-ES @cmaes2003 at
  matched candidate budget, E88 lands in the same loss-vs-wallclock
  band as Gated DeltaNet @gated_deltanet2024, the linear-recurrent
  baseline included here, at the 1.3 B-class on The Pile (§5).

#set enum(numbering: "(a)")

Three pure-recurrent instances are trained at the 1.3 B-class under
the same protocol on The Pile @thepile2020: E88 (1.273 B, the
Emender, delta-correcting update
$S <- tanh(d S + k("silu"(v) - S^T k)^T)$); M²RNN-CMA (1.307 B, a
CMA-reshaped pure-recurrent variant of the M²RNN architecture
@m2rnn2026, raw-write update $tanh(H W + k v^T)$); and Gated DeltaNet
(GDN @gated_deltanet2024, 1.352 B, the linear-recurrent baseline).
Each received per-architecture CMA-ES @cmaes2003 hyperparameter and
shape search at matched candidate budget, with range repositioning
when limits were hit. The cohort runs at near-matched parameter count;
exact configurations and results are reported in §5.

The paper proceeds as follows. §2 and §3 set up the linear-state
versus nonlinear-state classification and the Emender architecture;
§4 covers the multi-programming systems contribution; §5 presents the
1.3 B loss-vs-wallclock comparison with the per-architecture CMA-ES protocol; §6
reports the 8 M expressivity probes with the capacity-non-binding
justification; §7 the Lean 4 formalization, including Theorem set F
on saturation latching alongside the separation, $S_5$-realization,
and FLOP-class theorems; §8 related work, opening with the ancestry of
the delta-correction line from Widrow–Hoff through fast-weight
programmers to DeltaNet; §9 limitations; §10 conclusion; §11 testable
predictions; §12 future work. The current release hub is
`https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V02_PUBLIC_RELEASE_HUB.md`.
The release targets are E88
(`https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.3`),
GDN (`https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.3`), and
M²RNN-CMA
(`https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.3`), with
the public paper PDF mirrored at
`http://hypervolu.me/~erik/ndm/Garrison_2026_Emender.pdf`.

#heading(level: 2, numbering: none)[What we claim, and what we do not]

To make the scope legible up front, the table below
states each load-bearing claim with its evidence, its scope, and — in the
last column — what the same result does *not* license. The body sections
substantiate each row; the pointers are forward references.

#figure(
  align(center)[#text(size: 8pt)[#table(
    columns: (1.45fr, 2.05fr, 1.35fr, 1.95fr),
    align: (left + top, left + top, left + top, left + top),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Claim*], [*Evidence*], [*Scope*], [*Non-claim*],
    ),
    [Pure nonlinear-recurrent LMs train at the 1.3 B-class],
    [E88 reaches 0.973 train bpb on The Pile, no time-axis trick, no
     attention hybrid (§1, §5)],
    [One E88 run, single seed; viability demonstration],
    [Not a scaling law and not seed-averaged; one artifact shown to exist],

    [Bulk LM loss does *not* distinguish these architectures (a defended null)],
    [Held-out bpb 0.966 / 0.966 / 0.961 is a statistical tie across 5
     slices; FLOP-locked after equal CMA-ES tuning (§5)],
    [Defends a *null*: the loss tie is real and reproduces across 5 held-out
     slices and the Common-Pile control (single-seed per architecture)],
    [Does not license the train-loss ordering as an architecture verdict],

    [Delta-correcting $>$ raw-write on state tracking, at every length],
    [8 M $S_5$ probe, 3 seeds (@fig_s5_bars); 1.3 B symmetric-budget
     length-extrapolation, delta $>$ raw-write $>$ linear to 16×
     (@tab_s5_1p3b, @fig_s5_symmetric); Lean matched-resource separation
     (§7, sets C/C′/D)],
    [Budget-robust *ordering* and length-extrapolation efficiency: delta
     strictly ahead at every length under a doubled, symmetric, no-tuning
     budget],
    [Not "delta solves / length-generalizes $S_5$": both nonlinear updates
     plateau below ceiling at length; it is an efficiency gap, not
     impossibility],

    [Raw-write's $S_5$ shortfall is an efficiency gap, not a converged wall],
    [Raw-write fits the trained length (0.658 at $T=64$) but
     under-reaches with length even at the symmetric doubled budget (§6)],
    [Defends a *null*: its solvable-task curve was still rising; both nonlinear
     updates plateau below ceiling at length],
    [No converged expressivity wall asserted for raw-write],

    [Linear recurrence cannot track non-solvable $S_5$ at length],
    [GDN converges to a ceiling: $S_5$ decays to chance at length while it
     stays competent on solvable $S_3$/parity (@fig_1p3b_lengthgen, §7)],
    [Computability statement (Barrington / NC#super[1]), converged],
    [Not a statement about finite-length memorization, which it can do],
  )]],
  caption: [
    *Claim / Evidence / Scope / Non-claim.* Each row is substantiated in
    the cited section; the final column marks the boundary the result does
    not cross. The two architectural separations are *efficiency* and
    *computability* statements, not blanket can/can't claims; the bulk-loss
    row is a defended null.
  ],
) <tab_claims>

// ── 2. Background ─────────────────────────────────────────────────────────────
= Background <sec:background>

What a recurrent model can compute about its input, and how cheaply it
can be trained to do so, both turn on whether its state update is linear
or nonlinear in the state — the hinge §1 introduced. This section makes
that hinge precise. It gives the structural test that sorts an
architecture into one regime or the other, the complexity ceiling the
linear regime cannot pass, and the word problem the paper uses to detect
when a model has passed it. These three are the toolkit the architecture,
systems, and results sections assume.

The stakes are the general ones of sequence modeling at scale, and one
workload makes them concrete. Pangenomic sequence data runs to terabases
per study @hprc2023 @guarracino2023acrocentric @pggb2024, so modeling it
at the byte level means ingesting trillions of tokens for any downstream
operation, which rules out routine use. Linear-recurrent byte-level
foundation models were the first attempt in this regime and did not scale
reliably. The Merrill–Petty–Sabharwal results @merrill2024transformers
make the limitation legible: a linear-in-time recurrence sits inside a
complexity class too weak for non-solvable state-tracking. A
nonlinear-in-time model is needed to lift that ceiling, and no
off-the-shelf design fit; the Emender, developed in §3, is the
construction introduced here.

#heading(level: 2, numbering: none)[Linear-state and nonlinear-state recurrence]

The recurrent language model lineage runs from Elman networks
@elman1990 and the LSTM @lstm1997 through the modern linear-recurrent
variants: state-space models such as Mamba and Mamba2 @mamba2024
@mamba2_2024, gated linear attention @gla2023, and the delta rule
@deltanet2024. Each step in this lineage traded some expressive power
for time-axis parallelism, because nonlinear-in-time recurrence resisted
the parallel scan that makes GPU throughput tractable at modern scale.
Gated DeltaNet @gated_deltanet2024 is the selected linear-recurrent
baseline for the matched wallclock comparison reported here. Mamba-3
@mamba3_2026 is reported by its authors to outperform GDN; its
compilation-heavy HPO does not fit the §5 matched protocol, so
it is left to future work.

A single explicit criterion classifies recurrent architectures.
A recurrent layer is *linear-state* if its update can be written
$h_t = A_t h_(t-1) + b_t$
with $A_t$ and $b_t$ depending on the current input $x_t$ only.
Otherwise, if the previous state $h_(t-1)$ appears nonlinearly in
its own update equation, the layer is *nonlinear-state*. The
distinction is structural; it does not depend on whether the
non-linearity is a $tanh$, an exponential, a $tanh$-of-product, or a
sigmoid.

Linear-state designs admit a chunkwise or scan-based parallel
implementation along the time axis because composing
$h_T = A_T h_(T-1) + b_T = ...$ unfolds into a product of inputs only.
This is the reason linear-state recurrence dominates the
billion-parameter landscape. Mamba, Mamba2 (SSD), RetNet, GLA,
DeltaNet, Gated DeltaNet, RWKV-4/5/6/7, HGRN2, mLSTM, MinGRU/MinLSTM
@mingru_2024 and Griffin's RG-LRU @griffin2024 are all linear-state by
this criterion. The catch is that, asymptotically, a linear-state
recurrence at fixed precision and width is a regular-language recognizer
that lives inside TC#super[0] and therefore cannot solve
non-solvable-group word problems @merrill2024transformers @barrington1986.
For fixed-depth transformers in the same formal-language setting, this is
also the relevant ceiling under explicit arithmetic assumptions:
log-precision transformers are captured by first-order logic with
majority quantifiers @merrill_sabharwal2023log_precision, and
average-hard attention transformers exactly, plus softmax-attention
transformers with $O("poly"(n))$ floating-point precision or
$2^(-O("poly"(n)))$ absolute-error approximation, lie in DLOGTIME-uniform
TC#super[0] @chiang2025uniform_tc0.
The classical empirical witness is the symmetric-group $S_5$, which has
120 elements and is the smallest non-solvable group.

Nonlinear-state recurrences include the classical LSTM and GRU; the
sLSTM block of xLSTM @xlstm2024 (memory-mixing through the hidden state
inside exponential gates); M²RNN @m2rnn2026 (matrix state inside $tanh$
with raw-write injection); Titans @titans2025 (MLP memory with online
gradient updates); and the architecture introduced here. Nonlinearity
inside the recurrence is what lifts the expressive ceiling: the
Siegelmann–Sontag result @siegelmann1995 requires a non-linearity
$f(h_(t-1), x_t)$ in the state update for Turing-completeness, and
Barrington's theorem identifies $S_5$ as the canonical
NC#super[1]-complete witness.

Two nonlinear matrix-state designs (the Emender and M²RNN) therefore share the
necessary preconditions (matrix state, nonlinearity on the state, no
attention, no linearization, no hybridization) that define the
pure-nonlinear-recurrent class introduced in §1 and differ in one
place: the per-step update rule on the matrix state. The Emender uses a
*delta-correcting* update; M²RNN uses a *raw-write* update; both are
pure-nonlinear-recurrent instances. The within-class observation in §1
is that, *at matched per-token FLOP class and under per-architecture
CMA-ES*, the delta-correcting write is the ingredient identified here
that makes $S_5$/$S_3$-style prefix-tracking *learnable in practice*.
The matched-cost condition is what makes the within-class ordering
meaningful; without it, "more expressive" collapses into "spends more
compute". The Lean anchor for the matched cost is
`RecurrentResourceFormalism.emender_m2rnn_flop_class_equiv`, which shows
the per-token FLOP count of an Emender head and an M²RNN head sit inside
a common $c_1 d^2 + c_2 d$ envelope (§7 Theorem set D).

#heading(level: 2, numbering: none)[Matrix state]

Replacing a vector hidden state $h in RR^d$ with a matrix state
$S in RR^(N times V)$ is not specific to the Emender. Linear-state designs
(mLSTM, RWKV-5/6, DeltaNet) already use matrix or expanded states;
RetNet's accumulation $S_t = gamma S_(t-1) + k_t v_t^T$ is matrix-valued.
The point of matrix state is that an outer-product update
$S_t = ... + k_t delta_t^T$ provides $O(N V)$ scalars of dynamic state
at $O(N V)$ computational cost per token, with content-addressable
retrieval via $S^T q$. Matrix state is common to the post-Mamba
landscape (Mamba2 with `d_state`, GLA, DeltaNet, RWKV-5/6/7, mLSTM,
M²RNN) and is treated here as a precondition shared by the Emender and its
baselines, not as a contribution.

#heading(level: 2, numbering: none)[The $S_5$ state-tracking probe]

The symmetric group $S_5$ is the group of permutations of five items; it
has $120$ elements and is the smallest non-solvable group. Its *word
problem* is the following task: the model reads a sequence of
permutations — here, adjacent transpositions, the swaps of two
neighboring items — one at a time, and after each one must report the
running composition of everything read so far, its *prefix product*. By
Barrington's theorem @barrington1986 this task is complete for the
complexity class NC#super[1]. A recognizer that solves $S_5$ at length
$T$ with bounded precision and width must therefore reach the top of
NC#super[1] in the canonical regular-language witness; one that cannot
solve $S_5$ at training length lives below it. The solvable-group control
$S_3$ (permutations of three items, 6 elements) is included to factor out
the part of difficulty that comes from prefix tracking *per se* rather
than from non-solvability.

// ── 3. Architecture — the Emender ─────────────────────────────────────────────
= Architecture <sec:arch>

#heading(level: 2, numbering: none)[Per-head update]

Section 1 named the emender layer by what it does — read what the memory
predicts at the addressed slot, compute the prediction error, and write the
correction — and named E88 as the 1.3 B instance carried through the paper.
This section makes the layer concrete: the per-step computation, the choices
that fix it, and the shape numbers that make E88 one specific member of the
family.

An emender layer runs $H$ independent heads side by side. Each head $h$ owns
its own matrix-state memory $S_h in RR^(N times V)$ — the matrix state of §2,
here addressed by an $N$-dimensional key and holding a $V$-dimensional value
at each addressed slot. (Section 7's theorems use a single symbol $d$ for the
common matrix dimension; in E88, $d = N = V = 32$, so each head's memory is a
$32 times 32$ tile.) At each token, the layer's input projections supply, per
head, a key $k_h$ and query $q_h$ in $RR^N$ that name the slots to write and
read, a value $v_h in RR^V$ to store, a scalar decay $d_h in (0,1)$ that fades
the old state, and an output gate $g_h in RR^V$. The display below is one
head's full step in execution order: normalize the two addresses, read the
memory at the key, form the retrieval error, update the bounded state, and
gate the read at the query.

$
k_h &<- "silu"(k_h) / norm("silu"(k_h))_2 quad ("L"^2 "-normalized key")\
q_h &<- "silu"(q_h) / norm("silu"(q_h))_2 quad ("L"^2 "-normalized query")\
r_h &= S_h^T k_h \
delta_h &= "silu"(v_h) - r_h \
S_h &<- tanh(d_h dot S_h + k_h delta_h^T) \
y_h &= "silu"(g_h) dot.o S_h^T q_h
$

Here $"silu"$ is the smooth pointwise activation $x dot sigma(x)$, with
$sigma$ the logistic sigmoid, and $norm(dot)_2$ is the Euclidean norm. The first two lines put the key and query on
the unit sphere so the addressing is bounded; the middle lines make the write
a correction against the content already read, not a raw addition; and the
final line gates only the read output. This is the per-head primitive; E88
repeats it across a residual stack rather than changing the per-head equation.

E88 fixes these family choices to one billion-parameter instance: model width
1664, depth 12, $H = 370$ heads per layer, $N = V = 32$, expansion 1.0, batch
size 5, a recurrence chunk length of 2048 tokens, and 1.273 B parameters in
total. A Lean 4 definition (`emender_1p27B`) pins this geometry — twelve
layers of 370 heads, each with a $32 times 32$ tile — and from it the trusted
core derives the two counts §1 quoted: 22,200 independent recurrent programs
per token at batch size 5, and $370 times (32 times 32)$ recurrent-state
scalars per layer. Around the recurrent core the language-model wrapper is
conventional — a pre-norm residual stack with a final normalization and tied
input/output embeddings. Heads never share recurrent state; each layer learns
its own key, query, value, decay, and output-gate projections; and the
recurrent state starts at zero when no prior state is carried in.

#figure(
  kind: image,
  block(width: 100%, [
    #align(center)[
      #stack(dir: ltr, spacing: 1.5em,
        // Left panel: per-head dataflow schematic
        block(width: 46%, [
          #align(center)[*A. Per-head update step*]
          #v(0.4em)
          #align(center)[
            #stack(dir: ttb, spacing: 0.6em,
              // Inputs row
              stack(dir: ltr, spacing: 0.6em,
                rect(inset: 5pt, fill: rgb("#dde5f0"), text(size: 9pt)[$k_h$]),
                rect(inset: 5pt, fill: rgb("#dde5f0"), text(size: 9pt)[$v_h$]),
                rect(inset: 5pt, fill: rgb("#dde5f0"), text(size: 9pt)[$q_h$]),
                rect(inset: 5pt, fill: rgb("#dde5f0"), text(size: 9pt)[$d_h$]),
                rect(inset: 5pt, fill: rgb("#dde5f0"), text(size: 9pt)[$g_h$]),
              ),
              text(size: 8pt)[$arrow.b$ #h(0.4em) $L^2$ norm $q,k$  #h(0.4em) $arrow.b$],
              // Read at k
              rect(inset: 5pt, fill: rgb("#ffe5cc"), width: 100%, align(center, text(size: 9pt)[$r_h = S_h^T k_h$  #sym.space  #text(size: 8pt)[(read at address $k$)]])),
              text(size: 8pt)[$arrow.b$],
              // Delta
              rect(inset: 5pt, fill: rgb("#ffe5cc"), width: 100%, align(center, text(size: 9pt)[$delta_h = "silu"(v_h) - r_h$  #sym.space  #text(size: 8pt)[(prediction error)]])),
              text(size: 8pt)[$arrow.b$],
              // State update
              rect(inset: 5pt, fill: rgb("#cfe9cf"), width: 100%, align(center, text(size: 9pt)[$S_h <- tanh(d_h dot S_h + k_h delta_h^T)$])),
              text(size: 8pt)[$arrow.b$],
              // Output
              rect(inset: 5pt, fill: rgb("#dde5f0"), width: 100%, align(center, text(size: 9pt)[$y_h = "silu"(g_h) dot.o S_h^T q_h$])),
            )
          ]
        ]),
        // Right panel: multi-programmed shape
        block(width: 46%, [
          #align(center)[*B. Multi-programmed shape (per layer, per batch element)*]
          #v(0.4em)
          #align(center)[
            #stack(dir: ttb, spacing: 0.4em,
              text(size: 9pt)[$H = 370$ heads, each $S in RR^(32 times 32)$],
              v(0.2em),
              // Grid of small squares representing heads
              {
                let cols = 20
                let rows = 6  // ~120 visible cells; ellipsis indicates rest
                grid(
                  columns: (1fr,) * cols,
                  gutter: 1.5pt,
                  ..range(cols * rows).map(i => rect(width: 100%, height: 0.8em, fill: blue.lighten(40%), stroke: 0.3pt))
                )
              },
              text(size: 8pt)[$dots.h.c$  370 small recurrent programs per layer  $dots.h.c$],
              v(0.4em),
              text(size: 9pt)[At batch size 5 and depth 12:],
              text(size: 9pt)[*22,200 independent programs per token*],
              text(size: 8pt, style: "italic")[(Lean: `emender_1p27B_programs_per_batch_token_bs5`)],
              v(0.3em),
              rect(inset: 5pt, stroke: 0.5pt, [
                #set text(size: 8pt)
                Each program: a $32 times 32$ state tile in registers/SRAM;
                serial along time; parallel across $H times "batch"$.
              ]),
            )
          ]
        ]),
      )
    ]
  ]),
  caption: [
    *Emender architecture and multi-programmed shape.*
    *(A)* Per-head recurrent dataflow. Each head owns a small bounded
    associative matrix memory $S_h$. The model reads at address $k_h$,
    computes a prediction error $delta_h$, writes the bounded delta
    correction into $S_h$, and gates the read at $q_h$ for output. The
    $tanh$ on the state, the delta-correcting write, and the
    $L^2$-normalization of $q,k$ are visible here; §3 enumerates the
    four load-bearing ingredients of the Emender family.
    *(B)* The 1.3 B stack exposes 370 small
    independent heads per layer per batch element. Each head is a
    $32 times 32$ state tile that fits in registers/SRAM and runs its
    time loop serially; parallelism is harvested across heads, depth and
    batch (§4). The Lean theorem
    `emender_1p27B_programs_per_batch_token_bs5` certifies 22,200 programs
    per token at batch size 5.
  ],
) <fig_arch>

For the Emender family, four ingredients are load-bearing; E88 uses all four
in the instance above.

#set enum(numbering: "(a)")

+ *Bounded matrix state via $tanh$ on $S$.* Linear-in-state recurrences
  cannot pass the Siegelmann–Sontag boundary even in principle; placing
  the non-linearity on the state itself, not only on a gate or the
  output, is what makes an emender layer nonlinear-state.

+ *Delta-correcting write $"silu"(v) - S^T k$.* The emender layer reads what memory
  predicts at address $k$, computes the prediction error $delta$,
  and writes the correction. With an orthonormal key family this gives
  exact overwrite at one slot while preserving the others; with
  arbitrary keys it gives bounded error-correcting binding. A raw-write
  update $S <- S + k v^T$ (the M²RNN family) accumulates without
  correction and cannot, with fixed weights, satisfy a uniform one-step
  overwrite specification (§7).

+ *Saturation latching of the $tanh$ bound.* A slot driven near
  $plus.minus 1$ is insensitive to further bounded writes, sub-threshold
  counter-input leaves the sign unchanged, and a sufficient
  counter-delta releases the slot. Together these properties let a
  binding persist across many irrelevant tokens while keeping the
  memory revisable when the data demand it. §7's Theorem set F
  (saturation insensitivity, sign-preserving hold, counter-delta
  release) formalizes all three as slot-wise statements on the full
  emender-layer update.

+ *Many small heads, not one large matrix.* E88 at 1.3 B
  uses $H = 370$ heads of $32 times 32$ each (Lean-witnessed:
  `RecurrentResourceFormalism.emender_1p27B_programs_per_batch_token`,
  yielding 22,200 independent programs per token at batch size 5).
  Per-head $L^2$-normalized $q,k$ give many independent addressing
  programs and avoid the gradient-conditioning failure mode seen when a
  single shared $q,k$ pair feeds hundreds of value heads.

#heading(level: 2, numbering: none)[Parameterization choices]

E88 parameterizes the per-head decay $d_h$ in log space, following Mamba2
@mamba2_2024: each head learns a scalar $A_(log)$ and a bias
$delta_(text("bias"))$ — the code names `A_log` and `dt_bias` — and forms the
decay as $d = exp(-exp(A_(log)) dot "softplus"(alpha(x) + delta_(text("bias"))))$,
where $alpha(x)$ is the input-dependent term and $sigma$ is again the logistic
sigmoid. Both decay parameters are exempt from weight decay, as is
standard for such timescale parameters, so the regularizer does not pull the
learned decay toward a default. The recurrence is computed in float32 and cast
back to the storage precision, and the $tanh$ on the state uses the
numerically stable form $tanh(z) = 2 sigma(2 z) - 1$ in the fused kernel to
avoid overflow at large pre-activations. Three remaining choices are
deliberate omissions rather than additions: there is no output normalization
inside the recurrent layer — an ablation removed it for a 0.10-nat gain, and
the wrapper's block-level pre-norm plus final norm suffice — no short
convolution on the input, and no gate on the write path; the output gate
$"silu"(g)$ acts on the read alone.

#heading(level: 2, numbering: none)[The 1.3 B stack]

The 1.3 B stack's shape is not arbitrary. Ingredient (d) sets its character —
many small heads rather than one large matrix — and the numbers follow from
it: each head's $32 times 32$ tile is small enough to live in registers (§4),
the head count $H = 370$ is large enough to give that many independent
addressing programs per layer, and twelve such layers reach the 1.3 B-class
budget the cohort comparison fixes. The exact values were set by the
per-architecture search of §5, not by hand. At batch size 5, depth 12 and
$H = 370$ multiply to $12 times 370 times 5 = 22,200$ independent recurrent
programs per token, each carrying its own $32 times 32$ state tile. The
reference implementation and the fused Triton recurrence kernel that steps
these tiles are listed in the Appendix.

#heading(level: 2, numbering: none)[Ablation by architecture: isolating the write rule]

Three architectural properties are candidates for the one that lets a model
track state: *matrix state*, *temporal nonlinearity on that state*, and *delta
correction in the write*. The closest update-rule comparator to the Emender in
the literature is M²RNN @m2rnn2026. Its nonlinear matrix-state update has two
parts, written here as one display: the first line builds a nonlinear write
candidate $Z_t$ from the matrix state $H_(t-1)$, a learned weight $W$, and the
current key–value outer product $k_t v_t^T$; the second blends that candidate
back into the state through a forget gate $f_t$.

$
Z_t &= tanh(H_(t-1) W + k_t v_t^T)\
H_t &= f_t H_(t-1) + (1 - f_t) Z_t.
$

That split is exactly what E88 lacks: its carry sits inside the bounded
delta-correcting update, and its gate appears only on the read output.
M²RNN is *nonlinear-state* by the criterion of §2, since $H_(t-1)$
appears inside $tanh$ via $H_(t-1) W$, but the write into $H$ is a raw
outer product $k v^T$ rather than a delta correction. A three-row ablation
across the three candidate properties isolates the write rule by
elimination:

#figure(
  align(center)[#table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header(
      [*Property*], [*GDN*], [*M²RNN*], [*Emender*], [*What it shows*],
    ),
    [Matrix state], [yes], [yes], [yes],
      [Cannot separate: GDN has it and fails $S_5$],
    [Temporal nonlinearity on state], [no], [yes], [yes],
      [Cannot separate: M²RNN-CMA stalls at 0.22 on $S_5$],
    [Delta correction in write], [no], [no], [yes],
      [Surviving candidate],
  )],
  caption: [
    *Ablation by elimination on the three candidate differentiators for
    state-tracking.* GDN has matrix state and fails $S_5$ at training
    length (0.36 vs Emender 0.79, §6), so matrix state alone cannot be the
    differentiator. M²RNN-CMA has matrix state *and* temporal nonlinearity
    on the state and still stalls at 0.22 on $S_5$, so temporal
    nonlinearity is not the differentiator either. Delta correction is
    the only property left that the Emender has and the two baselines do not.
  ],
) <tab_ablation>

Two checks keep the isolation clean. First, M²RNN-CMA scores 0.31 on the
solvable control $S_3$, where non-solvability is not the obstruction: a model
held back only by the non-solvable structure of $S_5$ would still reach
ceiling on $S_3$, and at matched no-tuning budget M²RNN-CMA does not, which
rules out a complexity-ceiling explanation for its shortfall. Second, the
shortfall is not a capacity ceiling. At the 8 M probe shape
($N times V times H times "depth" = 32 times 32 times 32 times 4 =
131{,}072$) the per-token recurrent state carries 131,072 scalars — about five
orders of magnitude above the $log_2 6 approx 2.6$-bit information-theoretic
floor for representing the $S_3$ prefix-tracking table. What remains is a
matter of trainability under SGD, not representability or capacity, in keeping
with the efficiency reading the results sections develop. The empirical curves
are in §6 (@tab_s5); the formal counterpart is a machine-checked one-step
separation between the delta-correcting and raw-write updates
(`emender_m2rnn_one_step_resource_separation_embeds`, §7).

State capacity, then, is not the property that separates these models. Mamba2
@mamba2_2024, GLA @gla2023, DeltaNet @deltanet2024, RWKV-5+ @rwkv7_2025, and
GDN all carry matrix or expanded state, and GDN has matrix state yet still
fails $S_5$ at length. By elimination across the three candidates, the one
property the Emender has and both baselines lack is the delta-correcting
write; what that write buys is measured in §5–§6 and bounded formally in §7.

// ── 4. Systems ─────────────────────────────────────────────────────────────
= Multi-Programming and Systems <sec:systems>

#heading(level: 2, numbering: none)[Multi-programming: the throughput-enabling design choice]

Throughput comes from width, not from time. A linear recurrence earns
its throughput on the *time axis*: because $h_t = A_t h_(t-1) + b_t$
unfolds into a product that depends on the inputs alone, the
step-by-step loop can be re-expressed as a prefix scan or a chunkwise
matrix multiplication over the whole sequence at once. A purely
nonlinear recurrence cannot be rewritten that way without giving up the
nonlinear update that makes it expressive, so its time loop has to run
serially. *Multi-programming* recovers the throughput on the other
axis. Rather than parallelize one recurrence along time, it replicates
the recurrent computation across many independent heads — each with its
own small bounded matrix state — and harvests parallelism across those
heads, across state tiles, and across batch elements, while the time
loop inside each head still runs one step at a time.

The design choice is a decoupling: it frees throughput from the *shape*
of the update. The recurrent body must stay a bounded, register-resident
per-step map at a fixed matrix-state signature — that is what keeps each
head cheap — but it need not be compatible with an associative scan or a
chunkwise WY product. The cost is paid in per-head serial time, most
exposed at long sequence lengths where the serial loop is longest; the
gain is that a nonlinear recurrence can keep a modern accelerator busy
with no time-axis scan at all. The route is agnostic to the update
rule: both pure-nonlinear instances trained here — the Emender and the
raw-write M²RNN-CMA — satisfy the same multi-programming predicate at
the 1.3 B-class, machine-checked in the Lean core as
`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_emender`
(§7).

#heading(level: 2, numbering: none)[The 1.3 B Emender shape under multi-programming]

A modern GPU exposes thousands of independent streaming
multiprocessors, each able to run a small program out of its own
registers and shared memory, and multi-programming is built to fill
them. E88, the 1.3 B Emender instance, runs 370 such programs per layer
per batch element — one per head — so at depth 12 and batch size 5 the
layer stack issues $12 times 370 times 5 = 22,200$ small independent
recurrent programs per token, each carrying its own $32 times 32$ state
tile small enough to live in registers. Keeping the accelerator busy
therefore asks nothing of parallelism *along time*: the parallelism
across these programs is already enough to fill it. How fully it does
is something the next section measures directly.

#heading(level: 2, numbering: none)[Measured throughput and utilization]

That the card stays busy is a claim about occupancy, and it is measured
directly. Driving the 1.273 B E88 (batch 5, context 2048) through the
1.3 B run's own training path on a free NVIDIA RTX 6000 Ada (48 GB) and
sampling `nvidia-smi` once a second over a sustained post-warmup window,
the GPU holds *median 100% utilization (mean 99.8%, minimum 96% across
133 samples)* while drawing 97% of its 300 W power cap. The accelerator
stays busy on width-axis multi-programming alone, with no time-axis
scan. Sustained throughput is *7,492 tokens/s* (mean over steady
50-step windows; median 7,468); a slower sibling GPU sustained
7,277 tokens/s, so the absolute figure is GPU-dependent at the
few-percent level.

This figure is occupancy, not peak arithmetic, and the two should not
be run together. Model-FLOPs utilization (MFU) — computed with the
standard $6N$ convention against the card's dense bf16 peak of
364 TFLOPS — is *15.7%*, and that is a conservative lower bound: $6N$
counts only the weight-matmul FLOPs and excludes the recurrence and
gate operations, so the true MFU is slightly higher. High occupancy
with modest MFU is the expected profile of a model bound by memory
bandwidth and the serial recurrence rather than by arithmetic: the GPU
is busy nearly all of the time but converts about 16% of its peak FLOPs
into useful work.

What that throughput buys on the width axis is best read against a real
time-axis kernel. A chunkwise linear-scan baseline — Gated DeltaNet as
implemented in the flash-linear-attention (FLA) library (1.352 B, the
CMA-matched shape used here), run on the same GPU, context, and time
budget — sustains 8,248 tokens/s at MFU 18.4% and the same near-100%
occupancy. E88 reaches *≈ 91% of that linear-scan kernel's tokens/s*
(7,492 / 8,248) *without performing the sequential time-axis scan at
all*. Width-axis multi-programming therefore recovers
linear-scan-class throughput on the width axis. It does not beat the
chunked-scan kernel on raw tokens/s — FLA is about 10% faster here, at a
slightly larger parameter count and a different batch — so the standing
is parity-class, not superiority.

#heading(level: 2, numbering: none)[Fused Triton recurrence kernel]

The forward kernel uses an internal `[T, B, H, *]` layout (time, batch,
head, $...$) and dispatches one program per `(batch, head_block)`. The
`BLOCK_H` parameter is autotuned over $\{1, 2, 4, 8, 16\}$ because at
$H >= 256$ per-program-per-head launch overhead dominates. The state
tile lives in registers and shared memory for the duration of the
$T$-step loop. The following pieces are fused into the same Triton
program: $tanh$/$"silu"$ activations on input projections, $L^2$
normalization of $q, k$, the recurrent delta write, and the output
gate. Each fusion removes one to two `torch.cuda` launches per layer
call; at depth 12 the aggregate saving is approximately
50–60 ms per step.

#heading(level: 2, numbering: none)[Sparse-checkpoint backward]

The forward kernel saves $S$ only every $K = 16$ steps. The backward
kernel processes one $K$-step segment at a time: it forward-replays the
$K$ steps to rebuild per-step $S_(t-1)$ from the saved checkpoint, then
walks backward to apply the chain rule. The activation memory used by
the kernel is $T / K + 1$ checkpoint slots instead of $T$. At $K = 16$
this is approximately a $16 times$ shrink, at the cost of a single
extra forward pass per segment.

#heading(level: 2, numbering: none)[Portable kernels]

The kernel is written once in Triton @triton2019 and dispatched
identically on NVIDIA CUDA and AMD ROCm. This is the practical reason
to choose Triton over hand-tuned HIP: a single source achieves
~70% of the hand-tuned throughput in ~one week of porting work, versus
three to six weeks to port to HIP from scratch.

#heading(level: 2, numbering: none)[Distributed training]

The 1.3 B-class result reported here was produced on a single
workstation GPU, with no cluster and no sequence parallelism; the
distributed recipe described next is how the same training would scale
past one machine, not how E88 was run. That recipe pairs schedule-free
AdamW @schedulefree2024 per island with hierarchical local-SGD model
averaging in the DiLoCo @diloco2023 style: each island is one node of
8 GCDs with intra-island DDP, and inter-island synchronization averages
model weights every $tau = 250$ local steps, an interval chosen
empirically. The structural point holds either way: because the
Emender's parallelism is across programs rather than along time, it
does not need sequence parallelism to stay competitive at the
1.3 B-class — a simplification relative to the chunked-scan
implementations that linear-state recurrences rely on.

A different route would parallelize the time loop itself. ParaRNN
@pararnn2025 does so with Newton's method on a block-bidiagonal
Jacobian; we tried it on the $tanh(d S + k delta^T)$ map at
$32 times 32$ block size and found it markedly slower in throughput than
the multi-programmed Triton kernels above. Newton iteration carries a
data-dependent solve count per step, and its convergence on the bounded
$tanh$ map is unestablished, whereas multi-programming is a single
serial pass per head.

// ── 5. Language Modeling Results ────────────────────────────────────────────
= Language-Modeling Results <sec:lm>

#heading(level: 2, numbering: none)[Setup]

We train three pure-recurrent language models at the 1.3 B-class
parameter band on The Pile @thepile2020 with a 2048-token context window,
byte-pair encoding (p50k_base, 50,281-vocab), schedule-free AdamW
@schedulefree2024, and bf16 mixed precision. The three models and their
CMA-tuned shapes are:

#align(center)[#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, right, left),
  stroke: 0.5pt,
  inset: 6pt,
  table.header(
    [*Model*], [*Params*], [*Batch*], [*Shape*],
  ),
  [Emender], [1.273 B], [5], [dim=1664, depth=12, H=370, N=32],
  [M²RNN-CMA], [1.307 B], [5], [dim=1920, depth=21, H=370, N=16],
  [GDN], [1.352 B], [4], [dim=2688, depth=21, exp=2, H=44],
)]

#heading(level: 2, numbering: none)[Per-architecture CMA-ES protocol]

All three 1.3 B-class architectures (Emender, M²RNN-CMA, Gated DeltaNet)
received independent CMA-ES @cmaes2003 hyperparameter and shape search
over the same six knobs: width, depth, head count, state width, output
gating, learning rate. Each search ran under matched candidate budget:
population 16, fixed wall-clock per candidate, identical fitness rule of
mean training nats over a fixed late-training window. When CMA-ES
drifted onto a configured-range edge, the range was repositioned and
search continued, applied identically across the three architectures.
The per-family winner shapes in the table above carry into the §6
expressivity probes and §7 formal analysis unchanged. The concurrent
M²RNN paper (Mishra et al. @m2rnn2026 §5.2) holds width, depth,
optimizer, learning rate, weight decay and gradient clipping uniform
across compared architectures and varies only the sequence-mixing block,
a fair-by-uniformity protocol that does not give each architecture its
own best-tuning.

The per-architecture search was adopted to avoid the undisclosed HPO
budgets common in nearby work and the contradictory within-class
results that follow from them. It also functioned as a measurement. On
the Emender, CMA-ES repeatedly pushed head count up against whatever
ceiling the bounds set, and each range repositioning revealed the same
pressure on the next iteration. The final $H = 370$ is the interior
optimum reached after the bounds were placed far enough out that
CMA-ES stopped against open ground. A search configured for
cross-architecture fairness thus independently selected the design's
central feature: throughput comes from many small heads.

#heading(level: 2, numbering: none)[Gradient conditioning is a third recipe property]

A fourth run, *M²RNN-paper* (the paper-default shape from @m2rnn2026
re-implemented at 1.3 B with dim=3072, depth=10, H=759, N=16), was
attempted under the same training setup and *diverged* at step 8,400
with gradient norm $approx 4.2 times 10^7$. The CMA-tuned reshape
*M²RNN-CMA* (dim=1920, depth=21, H=370, N=16) of the same update family
is stable under the same optimizer and the same data; the divergent
paper shape is the stability control and is not in the comparison panel. The
two configurations differ in one structural parameter: the ratio of
$q,k$ projections to value heads. Many value heads sharing few $q,k$
pairs concentrate gradient through a narrow projection and accumulate
gradient norm at high step counts; redistributing toward more
independent $q,k$ pairs per value head (the same ratio the Emender uses
at 1.3 B, $H = 370$) removes the explosion. This is a property of
head geometry, not of the algebraic form of the write: a third recipe
axis the multi-programmed family must respect alongside matrix state
and update rule.

#heading(level: 2, numbering: none)[Loss-vs-wallclock comparison]

#figure(
  placement: top,
  image("results/figure_2/figure_2.png", width: 95%),
  caption: [
    *Training-dynamics diagnostic for the 1.3 B comparison:
    training-loss-vs-wallclock trajectories on The Pile under matched
    per-architecture CMA-ES. These curves are a diagnostic of training
    dynamics, not the basis for the architecture comparison — the
    reported measurement is the held-out bpb (below). All three
    trajectories occupy the same sub-1-bpb band; at their released v0.3
    checkpoints E88 and GDN are at 0.973 BPB and M²RNN-CMA at 0.979
    across the sampled window.* Schedule-free AdamW on The
    Pile with a 2048-token context. The single panel uses linear
    wall-clock hours from the start of the run to each model's released
    v0.3 checkpoint step, but each curve is drawn only after the 100K-step
    trailing window is fully populated and the curve is inside the
    plotted BPB range. Curves are 100K-step trailing moving averages
    of training loss in bits per byte; the nats/token
    $arrow$ bits/byte conversion uses the canonical
    $"bytes/token" = 3.92$ for `p50k_base` on The Pile (@sec:appendix_bpb). E88 is at
    1.273 B parameters; M²RNN-CMA at 1.307 B; GDN at 1.352 B; the
    three plotted models have trained about 22.1-23.8
    GPU-days at their released v0.3 checkpoints. The plotted
    trajectory is one realization per architecture. These training-loss
    trajectories are a diagnostic; the architecture comparison rests on
    the held-out bpb (below) and the §9 delta-off ablation and CMA-ES
    sweeps (250+ configs/architecture), not on these curves. The
    multi-week per-architecture training extent is the standard unit
    at this scale class. Endpoint values are 100K-step trailing
    averages read at each model's released v0.3 checkpoint step
    (E88 step 1,542,000; GDN step 2,031,000; M²RNN-CMA step
    1,491,000): E88 0.973 BPB, GDN 0.973 BPB, and M²RNN-CMA
    0.979 BPB. E88 and GDN remain in the same sub-1-bpb
    wallclock band. M²RNN-CMA has
    higher loss than the other two across the sampled window. The
    paper-shape M²RNN baseline (not shown)
    diverged at step 8,400. Color convention used throughout the
    paper: Emender = blue, GDN = orange, M²RNN-CMA = red. Curves end
    at each model's released v0.3 checkpoint step — the same
    checkpoints used for the held-out bpb below; training continued
    past these points.
  ],
) <fig_lm_racers>

After about 22–24 GPU-days, E88 reaches 0.973 bpb on The
Pile, GDN 0.973, M²RNN-CMA 0.979; all three are sub-1-bpb. These
values are read at each model's released v0.3 checkpoint step (E88
1,542,000; GDN 2,031,000; M²RNN-CMA 1,491,000), which are the
canonical checkpoints reported throughout. The corresponding
100K-trailing
training losses are 2.644 nats/token (E88, step 1,542,000),
2.644 (GDN, step 2,031,000), and 2.660 (M²RNN-CMA, step 1,491,000).
Under the training tokenizer (`p50k_base` BPE) on The Pile,
mean bytes per token is 3.92 over a 2000-sample sweep at the training
context of 2048 tokens (@sec:appendix_bpb), so
$"bpb" = "nats/token" times log_2(e) / "bytes/token" approx "nats/token" times 0.368$.
These training-loss curves are a training-dynamics diagnostic, not the
basis for the architecture comparison: the reported measurement is the
held-out bpb (next subsection), where the three models are a
statistical tie, and the architectures separate on the §6
state-tracking probes rather than on bulk language-model bits-per-byte.

#heading(level: 2, numbering: none)[Held-out bits-per-byte: the three are a statistical tie]

The loss-vs-wallclock curve above plots *training-loss* bpb under the fixed
canonical $"bytes/token" = 3.92$ normalization. To test whether that
within-class ordering survives off the training stream, we separately
measure *held-out* bpb: each model's real negative log-likelihood, in
nats, over a held-out Pile slice it never trained on, divided by the
slice's real UTF-8 byte count — tokenizer-invariant, with no 3.92
constant (method in @sec:appendix_bpb). On the canonical 10 MB held-out
slice (context 2048, stride 1024), Emender reaches *0.966* held-out bpb,
GDN *0.966*, and M²RNN-CMA *0.961* — all sub-1-bpb, and all at or
slightly below their respective train-loss figures (0.973 / 0.973 /
0.979). These held-out numbers are measured at the same released v0.3
checkpoint steps as the train-loss figures above (E88 1,542,000; GDN
2,031,000; M²RNN-CMA 1,491,000), so train-loss and held-out now sit on
one canonical checkpoint per model; generalization is healthy with no
held-out blow-up.

Held-out bpb and train-loss bpb are different objects — different data
*and* different byte-normalization — and are not directly comparable.
Critically, the train-loss ordering does *not* reproduce held-out.
Across five independent held-out Pile slices drawn from random deep
offsets across the 1.31 TB corpus, the three models sit inside a
0.006-bpb band of mean bpb, narrower than the largest cross-slice
standard deviation (0.025 bpb); the between-model mean gaps are
0.05–0.31$times$ the pooled cross-slice std, i.e. within noise. On
held-out bulk bits-per-byte the three update rules are a *statistical
tie*: which model is lowest is slice-dependent (lowest-bpb counts:
Emender 0/5, GDN 1/5, M²RNN-CMA 4/5), so bulk held-out perplexity does
*not* distinguish the architectures at this budget. This is the expected
reading: the architectures separate on the §6 state-tracking
probes, not on bulk language-model bits-per-byte. @sec:appendix_bpb
places these numbers in the landscape of open Pile-trained models,
out-of-distribution anchors, and classical compressors.

#heading(level: 2, numbering: none)[The loss tie reproduces under matched compute]

The tie reflects the matched compute given to each update rule. The three
rules — delta-correcting (E88), raw-write (M²RNN-CMA), and linear (GDN) —
each received the *same* per-architecture CMA-ES hyperparameter-and-shape
search budget and were then trained to *matched* FLOPs, and they converge
to within $approx 0.005$–$0.006$ bpb of one another on *both* axes: held-out
0.966 / 0.966 / 0.961 and train-loss 0.973 / 0.973 / 0.979. Equal tuning
effort and equal compute land three structurally different recurrences in
the same narrow loss band.

So even after each family is tuned to its own best operating point, the
converged loss does not move with the update rule: bulk language-modeling
loss does not distinguish what these architectures compute. Where they come
apart is the §6 state-tracking probes, not bpb, and so far that separation
shows only on synthetic algebraic state-tracking ($S_5$/parity/FSM, §6) —
tasks that *no* model in the cohort solves. The non-synthetic signals tie:
bulk held-out loss, the QA panel, and the reasoning panel (§6) are all
within noise.

The §5 loss figures are single-seed per architecture, so the loss-tie
reading is corroborated independently of any within-run spread: the five
held-out Pile slices (@sec:appendix_bpb) put the lowest-bpb model
slice-dependent with the between-model gaps inside cross-slice noise; the
Common-Pile contamination control (@sec:appendix_bpb) reproduces the tie on
a second distribution; and the converged value is consistent with an
architecture-agnostic compute-optimal loss at this matched FLOP budget. The
state-tracking results that carry the paper do not rest on these single-seed
loss runs: they rest on the 3-seed 8 M expressivity probes (§6) and the
1.3 B length-generalization separation (below), where the architectures
genuinely come apart.

// ── 6. Expressivity Results ───────────────────────────────────────────────────
= Expressivity Results <sec:expressivity>

#heading(level: 2, numbering: none)[Capacity is non-binding at 8 M parameters for these probes]

State-tracking probes in this section run at 8 M parameter-matched scale
(dim = 384, depth = 4, $H = 32$, $N = 32$, schedule-free AdamW, 10K–20K
steps per task, three seeds; GDN uses dim = 640 to match parameter
count). The $S_5$ and $S_3$ transition tables have information-theoretic
floors of $log_2 120 approx 6.9$ bits and $log_2 6 approx 2.6$ bits; an
8 M-parameter model exceeds those floors by roughly seven orders of
magnitude in parameter bits and six in recurrent-state scalars per
token. Failure to learn at this scale is a property of the update
rule's inductive bias under SGD, not of capacity. The probes test what
configurations SGD finds under matched no-tuning conditions, where the
§7 realizability theorem fixes what configurations exist.

#heading(level: 2, numbering: none)[Matched no probe-tuning, and the one selection asymmetry it does not cover]

The 8 M probe scale received no probe-specific hyperparameter search and
no seed sweep for any family in the comparison. The Emender ran on the
default configuration carried down from its 1.3 B stack; M²RNN-CMA ran on
the analogous default from its CMA-tuned reshape; GDN and the M²RNN-paper
shape ran on their respective published defaults. Each family is therefore
evaluated on the reasonable-defaults configuration it would arrive at
without probe-targeted optimization, rather than matched-after-HPO. With
capacity non-binding (above) and no probe-specific tuning, any accuracy
gap reflects the architecture's inductive bias under SGD.

That matched-no-tuning condition controls for probe-specific *effort*; it
does not control for one *selection-criterion* asymmetry, stated here in
full. The Emender's defaults are the
endpoint of an ablation lineage ranked on language-modeling loss — the
same objective GDN's and M²RNN's authors used for their published
defaults — with one exception: the state nonlinearity ($tanh$ versus a
linear state) tied the linear variant on language-modeling loss and was
kept on state-tracking grounds. That one component is load-bearing for
these probes, so the Emender's configuration was settled partly on a proxy
of the very quantity the probes measure, whereas no baseline's was. The
asymmetry favors the Emender, and it bears on the *magnitude* of the
within-class $S_5$ gap.

Two controls bound what the asymmetry can explain. First, the $S_3$
solvable-group control: raw-write M²RNN reaches only 0.31 on the
six-element solvable group $S_3$, well below the non-binding capacity
ceiling, where neither non-solvability nor capacity can account for the
shortfall — so raw-write's prefix-tracking deficit is an inductive-bias
property of the update rule, which no language-modeling-loss tuning
advantage for the Emender could manufacture. Second, the §7
NC#super[1] realizability result, not the $S_3$ control, accounts for the
linear GDN's $S_5$ collapse: GDN passes $S_3$ (0.72) and fails only
$S_5$, which a linear recurrence provably cannot track at length
regardless of tuning. Together these fix the *direction* and *mechanism*
of the within-class separation independently of the selection asymmetry;
what they do not fully de-confound is the *size* of the Emender's own
$S_5$ number. A matched state-tracking search on each baseline at the 8 M
shape would close that remaining magnitude gap; it is named in §12.

#heading(level: 2, numbering: none)[The $S_5$ permutation-composition probe]

The symmetric-group word problem in $S_5$ (track the running product
of adjacent transpositions, output the 120-way class at each prefix)
is the canonical NC#super[1]-complete witness. Random baseline is
$1 / 120 = 0.0083$. We train at $T = 128$, evaluate at
$T in {128, 256, 512, 1024}$, three seeds.

#figure(
  align(center)[#table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    inset: 6pt,
    table.header(
      [*Model*], [*$S_3$ T=128*], [*$S_5$ T=128*], [*$S_5$ T=256*], [*$S_5$ T=512*],
    ),
    [Emender], [*1.0000*], [*0.7918*], [*0.4158*], [*0.2150*],
    [GDN], [0.7185], [0.3552], [0.1843], [0.0974],
    [M²RNN-CMA], [0.3124], [0.2157], [0.1120], [0.0593],
    [M²RNN-paper], [0.3773], [0.1698], [0.0884], [0.0488],
    [random], [0.1667], [0.0083], [0.0083], [0.0083],
  )],
  caption: [
    *State-tracking accuracy on the permutation-composition probes.*
    Mean over three seeds. $S_3$ is the solvable-group control; $S_5$
    is the non-solvable NC#super[1] witness. The Emender separates from all
    three baselines *at training length*, not only under length
    extrapolation. M²RNN, the head of the raw-write nonlinear matrix
    RNN family, underperforms both at $S_5$ training length and at
    $S_3$, supporting the mechanism claim that nonlinear matrix state
    *alone* is not sufficient. Source numbers in
    `paper/ndmpapernotes.md` lines 153–173.
  ],
) <tab_s5>

#figure(
  image("figures/s5_expressivity_seeds.png", width: 95%),
  caption: [
    *Expressivity separation on the permutation-composition probes at
    parameter-matched 8 M scale.* Three independent seeds per
    architecture shown as individual points; error bars span the SEM
    (standard error of the mean across 3 seeds). Bars are the seed mean
    (light fill). The Emender separates from linear-recurrent (GDN)
    and raw-write nonlinear matrix RNN (M²RNN) baselines on the
    non-solvable $S_5$ probe at training length. On the solvable $S_3$
    control, the Emender is perfect (1.0000); GDN reaches 0.72; both
    M²RNN variants stall in the 0.31–0.38 band, indicating that at
    matched no-tuning budget the raw-write update under-reaches on the
    prefix-tracking task even without the non-solvability obstruction. Source: `paper/ndmpapernotes.md`
    lines 153–173; figure script: `paper/figures/plot_expressivity_seeds.py`.
  ],
) <fig_s5_bars>

At $S_5$ training length the Emender reaches 0.79, GDN 0.36,
M²RNN-CMA 0.22, M²RNN-paper 0.17. On $S_3$ the Emender is at 1.00, GDN
at 0.72, and both M²RNN variants stall in the 0.31–0.38 band; at
matched no-tuning budget the raw-write update under-reaches on the
prefix-tracking task even when the group is solvable. The Emender starts at 0.79 at $T = 128$ and declines to
0.42, 0.22, and 0.11 at $T = 256$, $512$, $1024$; it climbs higher and
falls slower than the baselines but does not reach ceiling at length.

The $S_3$/$S_5$ split is also where the scope of the M²RNN paper's own
state-tracking evaluation matters. Mishra et al. @m2rnn2026 §3.2 report
length generalization on $S_3$ alone (the smallest non-trivial
*solvable* group, which lives inside TC#super[0]) and do not evaluate
$S_5$ or any other non-solvable group. The "perfect state-tracking
generalization" framing in that paper does not bear on the
NC#super[1] regime: at parameter-matched 8 M, the Emender reaches
0.79 on $S_5$ at training length against the paper-default M²RNN's
0.17.

#heading(level: 2, numbering: none)[The six-task canonical sweep]

To verify that the $S_5$ result is not a single-task artefact we run a
six-task canonical sweep covering parity (binary XOR over a stream),
modular counter (K=5), FSM tracking (K=4 states),
Dyck-1 (balanced brackets), associative recall (key→value lookup), and
selective copy (mark-and-copy). At 8 M parameter-matched scale, the Emender ties
or exceeds GDN on five of six tasks (parity 1.00 vs 0.86; modular
counter 0.90 vs 0.65; FSM tracking 1.00 vs 0.83; Dyck-1 1.00 vs 1.00;
selective copy 1.00 vs 1.00). GDN is ahead of the Emender on associative recall
(0.997 vs 0.881), the only attention-natural task in the suite.

Under length extrapolation (train $T = 40$, evaluate up to $T = 500$),
the Emender retains 0.89 accuracy on parity at $T = 500$ where GDN collapses
to 0.55 (near random 0.50). On FSM tracking at $T = 500$, the Emender is at
0.59 versus GDN 0.39. The gap widens monotonically with length.

The gap grows with sequence length on parity, FSM tracking, and
modular counter: *multi-step persistence* on natural-language-shaped
sequences, the empirical companion of the §7 k-step separation
(`emender_m2rnn_k_step_separation`, Theorem set C′) and the one-step
resource separation
(`emender_m2rnn_one_step_resource_separation_embeds`, set C). The
remaining frontier, an $S_5$-generator-specific capacity bound with
explicit $T(d)$, is named in §12.

#heading(level: 2, numbering: none)[Hybrid degradation: purity matters]

A natural question for any architecture that leads on state tracking is
whether the gain survives mixing with linear-scan blocks. We test the
pattern $[upright("Emender"), upright("Emender"), upright("GDN"), upright("GDN")]$
(four-layer "AABB" hybrid) on the same canonical sweep and find that
*hybridization degrades state tracking below either pure family*:

#figure(
  image("figures/hybrid_degradation_seeds.png", width: 95%),
  caption: [
    *Hybrid degradation: per-seed accuracy at 8 M scale, 3 seeds per
    condition.* Interleaving Emender layers with linear-scan (Gated
    DeltaNet) layers in an `[Emender, Emender, GDN, GDN]` AABB pattern
    *underperforms* pure Emender on both modular counter ($K=5$) and
    FSM tracking ($K=4$ states), and underperforms pure GDN on modular
    counter. Bars show the seed mean; error bars span the SEM (standard
    error of the mean across 3 seeds); individual seed points are
    overlaid (seeds 42, 123, 456). The dashed line is the random
    baseline ($1/K$). State-tracking capability does not survive
    dilution by linear-scan (GDN) blocks in this Emender/GDN pairing;
    §11 prediction 5 addresses Emender + attention. Source: per-seed JSON under
    `paper/results/figure_4_hybrid/`; figure script:
    `paper/figures/plot_hybrid_degradation.py`. Mean $plus.minus$ std
    mirrors `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`.
  ],
) <fig_hybrid>

The hybrid result is the same finding as §3 from the other side:
linear-scan blocks do not inherit state tracking from neighboring
Emender blocks.

#heading(level: 2, numbering: none)[The same separation at 1.3 B scale]

The 8 M probes above run where capacity is non-binding so that any gap
is inductive bias, not size. The complementary question is whether the
separation survives at the 1.3 B scale, in the trained checkpoints
themselves. We fine-tune each released v0.3 checkpoint
— E88 (delta, 1.273 B), GDN (linear-recurrent, 1.352 B), and
M²RNN-CMA (raw-write, 1.307 B) — on the $S_3$ and $S_5$ word problems
and measure *length generalization*: train on prefixes of length
$T <= 64$, evaluate out to $T = 512$ (8× the longest trained length).
The trainable harness is initialized by a strict weight-load of the public
v0.3 checkpoint; a full held-out-slice load-sanity check reproduces each
model's published readback to $<= 1.3 times 10^(-4)$ nats, so the
fine-tune is of the released artifact itself.

To keep the comparison about expressivity rather than trainability, each
model is first driven *to competence on the solvable tasks*: at the
longest trained length $T = 64$, E88 reaches parity 1.000 / $S_3$ 1.000,
GDN 0.997 / 0.928, and M²RNN 0.999 / 0.748 ($S_3$ 0.990 at $T = 32$).
Under the identical matched recipe (lr $2 times 10^(-4)$, 2500 steps)
the raw-write M²RNN under-fits even solvable parity (0.557); reaching
solvable competence took a gentler, longer recipe (lr $5 times 10^(-5)$,
12,000 steps) where E88 and GDN saturate the solvable tasks at lr
$2 times 10^(-4)$ in $<= 5000$ steps. This *effort-to-competence* gap is
the matched-FLOP efficiency statement at 1.3 B: raw-write needs more
budget to reach the same solvable-task competence the delta update
reaches cheaply.

That to-competence procedure removed the *baseline's* trainability
confound but left an asymmetry: M²RNN got a gentler, longer recipe to reach
competence while E88 never got the same extra budget on $S_5$, so E88's
$S_5$ length-extrapolation shortfall (0.162 at $T = 512$) was ambiguous
between a real capacity ceiling and mere under-training. The headline run
below *fixes the asymmetry*: all three models get the *same* a-priori recipe
and the *same*, larger, fixed budget on $S_5$: M²RNN's exact gentle recipe
(lr $5 times 10^(-5)$ constant) with the step count *doubled* to 24,000 (2×
M²RNN's to-competence budget), identical length curriculum $T <= 64$, and
*no* $S_5$-tuning (a single fixed step count, no early-stopping; constant LR
so a flat curve cannot be confused with a decayed learning rate). The recipe
was chosen and frozen before any $S_5$ result was seen. @tab_s5_1p3b and
@fig_s5_symmetric report where each model lands.

#figure(
  align(center)[#table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: 0.5pt,
    inset: 6pt,
    table.header(
      [*Model*], [*$S_5$ T=64*], [*$S_5$ T=128*], [*$S_5$ T=256*], [*$S_5$ T=512*], [*$S_5$ T=1024*],
    ),
    [E88 (delta)], [*0.921*], [*0.536*], [*0.272*], [*0.143*], [*0.076*],
    [M²RNN (raw-write)], [0.658], [0.335], [0.172], [0.090], [0.049],
    [GDN (linear)], [0.117], [0.063], [0.035], [0.022], [0.015],
    [random], [0.0083], [0.0083], [0.0083], [0.0083], [0.0083],
  )],
  caption: [
    *$S_5$ length-extrapolation at 1.3 B scale, symmetric
    24,000-step budget; all three models, one identical a-priori recipe, no
    $S_5$-tuning.* Each released `@v0.3` checkpoint is loaded strictly and
    full-fine-tuned on $S_5$ alone under a single fixed recipe (lr
    $5 times 10^(-5)$ constant, 24,000 steps $=$ 2× M²RNN's to-competence
    budget, identical length curriculum $T <= 64$, seed 42), chosen and
    frozen before any $S_5$ result was seen. $T = 64$ is the longest trained
    length; columns to its right are extrapolation out to 16× ($T = 1024$).
    Chance is $1 / 120 = 0.0083$. At the trained length E88 (delta) nearly
    solves $S_5$ (0.921) while the linear GDN never learns it (0.117, far
    below competence); raw-write M²RNN reaches 0.658. Under extrapolation all
    three degrade and E88 $>$ M²RNN $>$ GDN at *every* length, but E88
    plateaus below ceiling (0.143 at $T = 512$, flat from $approx 12,000$
    steps) rather than tracking $S_5$ to arbitrary length. Source:
    `paper/review/S5_SYMMETRIC_BUDGET.md`; data
    `paper/review/s5_symmetric_data/{e88,m2rnn,gdn}.json`.
  ],
) <tab_s5_1p3b>

#figure(
  image("figures/s5_symmetric_acc_vs_T.png", width: 85%),
  caption: [
    *$S_5$ length-extrapolation at the symmetric 24,000-step budget.*
    Running-state accuracy vs sequence length $T$ (log axis) for the three
    released 1.3 B architectures, each fine-tuned on $S_5$ under the *same*
    a-priori recipe (lr $5 times 10^(-5)$ constant, 24,000 steps, no
    $S_5$-tuning): solid curves are this symmetric run (E88 blue, M²RNN red,
    GDN green); faint dashed curves are the prior per-model to-competence run,
    shown for contrast. Trained on $T <= 64$ (vertical marker), evaluated out
    to $T = 1024$ (16× the longest trained length); chance is the dotted line
    ($1 / 120$). At the trained length the delta model nearly solves $S_5$
    while the linear GDN never learns it; under extrapolation E88 $>$ M²RNN
    $>$ GDN at *every* length, with E88 plateauing below ceiling rather than
    tracking $S_5$ to arbitrary length. Numbers match @tab_s5_1p3b. Real
    data: `paper/review/s5_symmetric_data/{e88,m2rnn,gdn}.json`; figure
    generated by `scripts/finetune_s5_symmetric.py` /
    `scripts/analyze_s5_symmetric.py`.
  ],
) <fig_s5_symmetric>

#figure(
  image("figures/s5_s3_1p3b_lengthgen.png", width: 100%),
  caption: [
    *Solvable controls and the prior to-competence $S_5$ cross-check at
    1.3 B scale.* Accuracy vs sequence length $T$ for the three released
    architectures on the non-solvable $S_5$ probe (left) and the solvable
    $S_3$ and parity controls (center, right). All three are fine-tuned on
    $T <= 64$ (dotted marker) and evaluated out to $T = 512$ — 8× the longest
    trained length; chance is the dashed line ($1 / 120$, $1 / 6$, $1 / 2$).
    Colors follow the paper convention (E88 blue, M²RNN-CMA red, GDN
    orange). The $S_5$ panel here is the earlier per-model *to-competence*
    run (the headline symmetric-budget $S_5$ result is @tab_s5_1p3b /
    @fig_s5_symmetric); it is retained as a cross-check and reproduces the
    same E88 (delta) $>$ M²RNN (raw-write) $>$ GDN (linear) ordering at
    *every* length under a different recipe and budget: E88 degrades the
    most slowly (0.162 at $T = 512$, far from solving), the linear GDN
    converges toward the chance floor, and raw-write tracks GDN's collapse
    shape. The ordering is consistent across both runs, but all three degrade
    with length and none tracks $S_5$ to arbitrary length. On the solvable
    controls all three are competent at the trained length and degrade only
    with extrapolation, confirming the $S_5$ gap is the non-solvability
    obstruction, not a generic difficulty gap. Real data:
    `paper/review/s3_s5_finetune_v03_data_tocomp/`; figure script:
    `paper/figures/plot_s5_s3_1p3b_lengthgen.py`.
  ],
) <fig_1p3b_lengthgen>

*Trained-length competence vs length-extrapolation plateau.* Two facts must
be kept apart. First, *at the trained length* ($T = 64$) the delta model
clearly *learns* the $S_5$ algorithm: E88 reaches 0.921, near-solving a
non-solvable task, whereas the linear-recurrent GDN never learns it even at
the trained length (0.117, far below competence) and raw-write M²RNN reaches
0.658. So at the trained length the delta update nearly solves $S_5$ where
the linear update cannot acquire it at all. Second, *under
length-extrapolation* (out to $T = 1024$, 16× the trained length) all three
degrade; E88 degrades the slowest and stays strictly ahead at *every* length,
but it *plateaus below ceiling* (0.143 at $T = 512$, 0.076 at $T = 1024$).
This is a length-generalization / capacity-at-this-$d$ limit, *not* an
expressive impossibility and *not* mere under-training: the $T = 512$ curve
is flat from $approx 12,000$ steps at 2× M²RNN's budget under constant LR
(see *The delta update is also length-bounded* below). The accurate statement is therefore neither
"E88 length-generalizes on $S_5$" nor "E88 fails $S_5$": the delta update
learns $S_5$ at the trained length and leads at every length, with its
length-extrapolation plateauing below ceiling at this width.

*The linear recurrence converges to a wall.* GDN reaches only 0.117 at the
trained length and decays to 0.022 at $T = 512$ and 0.015 at $T = 1024$ (near
the chance floor $1 / 120$), flat by $approx$ step 6,000: a converged
failure, not under-training, exactly the Barrington / NC#super[1] prediction
that a linear recurrence provably cannot maintain the non-solvable $S_5$
product at arbitrary length. The same GDN is competent on *both* solvable
controls (parity 0.997, $S_3$ 0.928 at $T = 64$; see *The solvable controls
hold*), so its $S_5$ collapse is the non-solvability obstruction, not a
generic difficulty gap. The clean expressivity contrast is delta E88 (learns
$S_5$ at the trained length and leads at every length) versus linear GDN
(cannot learn it at any length), both competent on the solvable controls.

*Delta vs raw-write: a budget-robust ordering, not solve-vs-fail.* The
symmetric run removes the recipe asymmetry directly. E88 received
the *same* doubled budget M²RNN got, on the *same* gentle recipe, with *no*
$S_5$-tuning, and still plateaus, so both confounds (M²RNN's trainability
deficit and E88's never-having-had-the-extra-budget) are bought out
symmetrically. Under that symmetric budget raw-write M²RNN fits the trained
length less well than the delta model (0.658 vs 0.921) and stays below it at
every extrapolation length (0.090 vs 0.143 at $T = 512$; 0.049 vs 0.076 at
$T = 1024$), tracking the linear GDN's collapse shape far more than the delta
model's. The ordering E88 $>$ M²RNN $>$ GDN holds at *every* length out to
16× and is robust to a large budget swing: the matched 2,500-step
(lr $2 times 10^(-4)$), the per-model to-competence, and this symmetric
24,000-step (lr $5 times 10^(-5)$) runs all land E88 $approx 0.14$ at
$T = 512$ strictly ahead of M²RNN $approx 0.09$. The claim is therefore one
of *length-extrapolation efficiency*, not can/can't: at matched (indeed
doubled) budget the delta update reaches strictly further with length than
raw-write, and both *nonlinear* updates plateau below ceiling as length
grows.

*The delta update is also length-bounded.* As noted below, M²RNN's $S_5$ shortfall
is partly a *trainability* matter. Symmetrically, E88's length shortfall is
partly a *capacity / length-generalization* matter, not only optimization: at
this width $d$ the delta model's $T = 512$ accuracy is flat from
$approx 12,000$ steps under constant LR, three independent runs across three
recipes all converge to $approx 0.14$ there, and extending its own
flattening late-training rate, even doubling the budget *again* would add
only $approx 0.03$, nowhere near ceiling. We therefore do *not* claim the
delta update "solves" or "length-generalizes" $S_5$ at extrapolated length
given enough steps; at fixed $d$ both the delta and raw-write updates plateau
below ceiling, with delta strictly ahead at every length. What survives is
the *budget-robust ordering* (delta $>$ raw-write $>$ linear), paired with
the explicit admission that the delta update too is capacity-bounded
at length.

*Why this is computation, not memorization.* The input is a
length-$T$ sequence over a fixed generator alphabet, so the number of
distinct inputs grows as $g^T$ — astronomically larger than any
training set by $T = 64$ — and evaluation sequences are held out and
test-disjoint from training. The primary metric is length
generalization itself (train $T <= 64$, evaluate to $T = 1024$): a
lookup table fit to $<= 64$-length prefixes cannot extend to $16×$ that
length, so above-chance accuracy out at $T = 1024$ is evidence of a learned
recurrence, not recall. The linear GDN's *failure* sharpens the point:
it is fully competent on the solvable $S_3$ control (0.928 at $T = 64$)
yet converged-fails $S_5$, which it could only do if $S_5$ demands
state-tracking computation a linear recurrence cannot perform —
memory alone would not distinguish the two. $S_3$ is the solvable
control throughout: all three models reach it, so the $S_5$ gap is the
non-solvability obstruction, not a generic difficulty gap.

*Raw-write's shortfall is partly trainability.* M²RNN's $S_5$ (and $S_3$) accuracy was still slowly
rising at 12,000 steps — its curve is not the flat, converged ceiling
GDN's is — so its $S_5$ shortfall is partly entangled with optimization
difficulty: full fine-tuning of the 1.3 B raw-write model is unstable
and needed the gentler/longer recipe merely to fit parity. We therefore
do *not* claim a pure-expressivity wall for raw-write the way we can for
the linear GDN; the statement here is the matched-FLOP efficiency one:
raw-write needs more budget to approach
what the delta update reaches cheaply, and within the budgets tested
(including the symmetric doubled budget) it does not get there. The
recipe-independent statements are (a) the clean expressivity contrast is
delta E88 (learns $S_5$ at the trained length, leads at every length) vs
linear GDN (cannot learn it at any length, converged-fails), and (b)
raw-write M²RNN, even at best-effort competence, exhibits the
length-extrapolation collapse and stays below the delta update's reach at
every length; as noted above, the delta update too
plateaus below ceiling at this width.

*The solvable controls hold (S3 and parity).* @fig_1p3b_lengthgen surfaces
the two solvable controls alongside $S_5$. All three architectures are
competent at the trained length on both: at $T = 64$, parity is 1.000 /
0.997 / 0.999 and $S_3$ is 1.000 / 0.928 / 0.748 (E88 / GDN / M²RNN), and
they degrade only under extrapolation — E88 still 0.565 on $S_3$ and 0.745
on parity at $T = 512$. Because every model reaches the solvable tasks,
the $S_5$ separation is not a generic "harder task" gap: it is the
non-solvability obstruction isolating the algebraic structure $S_5$
demands and $S_3$/parity do not. (The raw-write M²RNN's *own* $S_3$ decay
at length, falling below the linear GDN past $T = 96$, is the same
matched-FLOP efficiency story, not an expressivity wall — its solvable-task
curve was still rising; see *Raw-write's shortfall is partly trainability* above.)

#heading(level: 2, numbering: none)[Micro and mega: the two probes bracket the mechanism]

The expressivity claim rests on two complementary measurements that are
load-bearing only *together*. The *micro* probe is the 8 M from-scratch
sweep (@fig_s5_bars, three seeds): it isolates expressivity in the cleanest
setting, where the §6 floor argument makes capacity non-binding by
orders of magnitude, so any gap is inductive bias and not size; its
limitation is that an 8 M model is small. The
*mega* probe is the 1.3 B fine-tune above (@fig_1p3b_lengthgen): it shows
the *same* delta $>$ raw-write $>$ linear ordering in the trained 1.3 B
checkpoints — the released v0.3 weights, loaded strictly; its limitation
is that a fine-tune may inherit structure rather than demonstrate it under
controlled capacity. Neither probe alone closes both questions; *together*
they bracket the mechanism. The micro probe controls capacity and seeds;
the mega probe controls scale and the trained artifact. The separation that
appears in both, under opposite confounds, is the one we claim: at matched
per-token compute the delta-correcting recurrence tracks non-solvable state
with length where the raw-write and linear updates do not. The 8 M and
1.3 B results are not two anecdotes but one mechanism seen from two sides.

#heading(level: 2, numbering: none)[QA and reasoning panel at 1.3 B: parity-rate evidence]

For capability beyond loss numbers we evaluate the three 1.3 B-class
models on a 300-item multi-choice continuation harness sampled from
ARC-C/E @arc2018, HellaSwag @hellaswag2019, SciQ @sciq2017, OpenBookQA
@openbookqa2018, and BoolQ @boolq2019. At the current training budget, E88
reaches 0.367 (random ~0.29), GDN 0.380, and M²RNN-CMA 0.367; all
three sit within one standard error of one another
($"SE" approx 6$ pp at 50 items per task). On a separate reasoning
panel (BIG-Bench Hard @bbh2022, ReCLor @reclor2020, FOLIO @folio2022),
all three families collapse on multi-step object tracking
(the seven-object shuffled-objects tracking task, 0.10–0.13, near-random) and
on FOLIO/ReCLor (near-random for all three), with GDN leading
on formal fallacies and web-of-lies. E88's overall reasoning
accuracy (0.319) is within one standard error of M²RNN-CMA (0.336).
None has crossed the threshold where reasoning benchmarks differentiate,
consistent with §1's class-level claim at this training budget. The
state-tracking-specific reasoning separation, if it exists, will emerge
at longer training horizons; §11 states the prediction explicitly.

// ── 7. Formal Results ─────────────────────────────────────────────────────────
= Formal Results <sec:formal>

We have a trusted Lean 4 @lean42021 core built on Mathlib
@mathlib4. The import closure of the `ElmanProofs.PaperCore` module
(ten source files at the time of writing) contains no `sorry`, no
`admit`, no `axiom`, no `opaque`, and no `native_decide`. Each result
below is identified by its exact theorem name so that the reader can
locate it in the source. Two results carry the spine of the argument.
The *k-step separation*
(`emender_m2rnn_k_step_separation`, theorem set C′ below) shows that
for every $k >= 1$ and every fixed-right raw-write resource with
row/column/cell external forget gates, there is an explicit $k$-token
input sequence on which the $k$-step trajectories disagree: the gap
strictly persists for every finite $k$ on the constructed witness
alphabet rather than washing out under composition. The *latching set*
(theorem set F below) formalizes the saturation half of the Emender
primitive, with three slot-wise statements covering saturation
insensitivity, sign-preserving hold under sub-threshold counter-input,
and counter-delta release. The scope of what is and is not proved is
summarized under "Frontier and unproven targets" below.

#heading(level: 2, numbering: none)[Theorem set A: finite-state ceiling and $S_5$ tracker]

#set list(indent: 1em)
- *Finite-state ceiling at fixed precision.*
  `S5Witness.fixed_precision_state_space_finite` shows that every
  fixed-precision online recognizer has a finite state space. This bounds
  the Emender (at fixed width and precision) to regular-language
  recognizers, and therefore strictly inside NC#super[1].

- *$S_5$ word problem.* `S5Witness.s5_state_count` proves $|S_5| = 120$;
  `S5Witness.s5_not_solvable` proves $S_5$ is non-solvable;
  `S5Tracker.recognizer_state_count` shows the prefix-product tracker is
  a 120-state recognizer; `S5Tracker.run_append` proves that word
  execution composes permutations. The bridge
  `S5Tracker.pythonRun_eq_tracker_tuple` shows the Lean tracker agrees
  on every input with the Python evaluation harness.

- *Lookup-table realization.* `S5EmenderRealization.s5_transition_key_count`
  shows the $S_5$ tracker uses exactly $120 times 4 = 480$ state/input
  keys; `S5EmenderRealization.exactTransitionMemory_run` shows that any
  finite recognizer admits an exact lookup-table realization.

#heading(level: 2, numbering: none)[Theorem set B: Emender realizes $S_5$]

The bridge from the abstract lookup-table realization to the Emender update
equation is `EmenderRealizesS5.emender_realizes_s5_tracker`: there exist an integer $d$, an
orthonormal family of keys ${k_g}$ indexed by the adjacent-transposition
generators, a value family ${v_g}$ and a decay scalar $lambda = 1$
such that the Emender update
$
  S_t = tanh(lambda dot S_(t-1) + k_(g_t) (v_(g_t) - S_(t-1)^T k_(g_t))^T)
$
produces a state trajectory that, decoded through a fixed linear
readout, reconstructs the $S_5$ transition table on every input word.
Here $v_g$ is the bounded post-nonlinearity value supplied to the abstract
theorem update, corresponding to $"silu"(v_h)$ in the concrete per-head
architecture of §3 rather than to a raw value projection.
The proof uses
`OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` for the
orthonormal-key write step and `S5Tracker.run_append` for compositional
correctness.

#heading(level: 2, numbering: none)[Theorem set C: update-family separation]

The Emender's delta-correcting write and M²RNN's raw outer-product write are
provably distinct as update families.

- *Mechanism separation.*
  `OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` proves
  that the delta write exactly overwrites the addressed slot while
  preserving all orthogonal queries.
  `OnlineMemory.rawOuterWrite_not_uniformOneStepOverwrite` proves that
  the raw outer-product write cannot satisfy the uniform one-step
  overwrite specification.

- *One-step resource separation.* The main embedded statement
  `RecurrentResourceFormalism.emender_m2rnn_one_step_resource_separation_embeds`
  proves that for every $K >= 2$, $V >= 1$, no fixed-weight M²RNN
  parameterization with row, column or cell forget gates can match the Emender's
  mixed-key delta correction in one recurrent step. The result is sharp:
  it covers every external-forget shape that respects the M²RNN
  signature.

- *Positive embedding.* For completeness,
  `M2RNNComparison.m2rnn_read_then_delta_embeds_e88_delta_update` shows
  that *if* M²RNN is given the extra read-then-delta resource, it can
  embed one Emender step. The separation in the previous bullet says that
  without that extra resource M²RNN cannot.

#heading(level: 2, numbering: none)[Theorem set C′: multi-step (k-step) separation]

The one-step separation of set C does *not* wash out under composition.

#block(inset: (x: 1.5em), [
*For every $k >= 1$, there is an explicit $k$-token input sequence such
that the $k$-step trajectory of any fixed-weight raw-write resource (in
the row/column/cell external-forget class) diverges from the Emender's
$k$-step trajectory by a margin bounded away from zero (Lean:
`emender_m2rnn_k_step_separation`).*
])

The witness alphabet is the 2-dimensional construction from
set C composed with zero-input filler tokens
(`kStepWitnessInputs k = (mixedKey, 0) :: zeroSteps (k - 1)`); the
proof inducts on the tail, using the row-0 preservation lemma
(`FixedRightRawExternalForget2_preserves_zero_row`) to keep entry
$(0, 0)$ at zero on the raw-write side for every $k$, while the Emender
side reduces to the $k$-fold composition of $tanh$ at $-1$, which is
nonzero by injectivity of $tanh$. The existential form
`emender_m2rnn_k_step_separation_exists` packages the same statement
existentially over $k$-step input sequences. The two-step case is
exposed separately (`emender_m2rnn_two_step_separation`) as the
inductive base. All three live in
`ElmanProofs.Architectures.MultiStepSeparation`, which is part of the
trusted import closure of `PaperCore`. The §6 length-extrapolation
curves (parity, FSM tracking, modular counter; Emender-vs-baseline gap
widening monotonically with sequence length) are the empirical
companion of this formal multi-step persistence on a
natural-language-shaped alphabet.

#heading(level: 2, numbering: none)[Frontier and unproven targets]

The k-step separation above is the *clean ceiling reachable from the
one-step proof by direct composition*. The scope of what it does and
does not establish is named below, before stating Theorem set D.

#set list(indent: 1em)
- *What is proved.* k-step trajectory separation on a constructed 2D
  input alphabet, for every $k >= 1$ and every resource in the
  row/column/cell external-forget class
  (`emender_m2rnn_k_step_separation`). The gap *does* compound on this
  alphabet — it does not wash out.

- *What is not proved.* That the gap compounds *on the $S_5$ generator
  alphabet specifically* with an explicit length bound $T(d)$, i.e., a
  statement of the form "for any fixed-weight raw-write RNN at state
  dimension $d$ there is an explicit $T(d)$ past which $S_5$ coset
  tracking is unreachable, while the Emender at the same $d$ tracks it".
  That is the stronger inseparability claim a reviewer would naturally
  want; the trusted Lean core does not contain it.

- *Why the stronger claim is not in the trusted core.* The $S_5$-coset
  inseparability claim is a *capacity* statement, not a witness
  statement; it requires bounded-precision raw-write RNNs and a
  pigeonhole / state-counting lower bound in the style of Merrill,
  Petty and Sabharwal @merrill2024transformers. Mathlib does not
  currently provide NC#super[1] / TC#super[0] circuit classes,
  finite-state-tracking lower bounds for parameter-bounded RNNs, or
  pigeonhole capacity arguments specialized to recurrent maps with
  bounded weight matrices. Each piece is a research-grade
  mechanization project in its own right; the integration is what
  would close the gap.

- *Connection to the empirical claim.* The $S_5$ accuracy curves of §6
  (@fig_s5_bars, @tab_s5) and the canonical-sweep length-extrapolation
  curves are the *empirical* evidence for the stronger claim that the
  formal core does not yet reach. §7 closes the "wash-out" objection on
  a constructed witness; §6 is the empirical evidence that the same
  wash-out failure mode is also absent on the $S_5$ generator alphabet
  itself.

The full status of this push, including what bridging machinery would
be required (bounded-precision raw-write class, reachable-state
counting, Merrill-style capacity lemma), is documented in
`formal/lean/LEAN_FRONTIER.md`.

#heading(level: 2, numbering: none)[Theorem set D: per-token FLOP class]

`RecurrentResourceFormalism.emender_m2rnn_flop_class_equiv` proves that the
per-token floating-point operation count for one Emender head and one M²RNN
head is bounded by a common $c_1 d^2 + c_2 d$ form, with explicit
constants $c_1, c_2$. Equal-token-budget comparisons at matched $d, H,
"depth"$ are therefore within a constant factor; the within-class
empirical ordering of §5 cannot be charged to one PNR instance spending
asymptotically more compute per token than the other.

#heading(level: 2, numbering: none)[Parameter-efficiency corollary (informal; follows from sets C and D)]

The per-step representational separation
(`RecurrentResourceFormalism.emender_m2rnn_one_step_resource_separation_embeds`,
set C) together with the matched per-token FLOP class equivalence
(`RecurrentResourceFormalism.emender_m2rnn_flop_class_equiv`, set D) carries
an informal *parameter-efficiency corollary*: any raw-write matrix RNN
that one-step-realizes the Emender's mixed-key delta overwrite at the matched
signature must allocate more state capacity (or, equivalently at fixed
state shape, more fixed weights) than the Emender, because no fixed-weight
raw-write parameterization at the matched signature with row, column or
cell forget gates can produce the same one-step result (set C,
sharpness clause). This corollary is *not* a standalone theorem in the
trusted Lean core; it *follows from* the two named theorems above
combined with the universal-overwrite specification
`OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` and the
negative counterpart `OnlineMemory.rawOuterWrite_not_uniformOneStepOverwrite`.

#heading(level: 2, numbering: none)[Theorem set E: multi-programming as a structural predicate]

`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_emender` defines
a predicate `IsMultiProgrammed` on architecture signatures capturing the
three multi-programming features (many independent heads per layer,
per-head state tile, per-batch independence). The 1.3 B Emender signature
and a CMA-reshaped pure M²RNN signature both satisfy the predicate, and
a non-trivial hybrid signature *fails* it. This is the small formal
anchor for the *class-level* claim of §1: multi-programming is a
property of the PNR class shared across both PNR instances trained
here, not specific to the Emender. The trusted core formalizes the PNR class
through two instances (the Emender and M²RNN-CMA), with the Emender as the
delta-correct contribution of this work and M²RNN-CMA as the raw-write
comparator.

#heading(level: 2, numbering: none)[Theorem set F: latching]

These three results formalize the latching half of the Emender primitive:
saturation makes a slot insensitive to further bounded writes, sign is
preserved under sub-threshold counter-input, and a sufficient
counter-delta releases the latch. All three are slot-wise statements on
the full abstract Emender update, where $v$ denotes the bounded
post-nonlinearity value input already supplied to the recurrent body,
$S' = tanh(lambda dot S + k (v - S^T k)^T)$
at slot $(i,j)$, with $W := k_i (v_j - (S^T k)_j)$ the write contribution
into that slot. The proofs run through the standard tanh-perturbation
foundation.

#set list(indent: 1em)
- *Saturation insensitivity.* If the slot is saturated in the sense
  $|S'_(i,j)| > 1 - epsilon$, and a perturbation $delta$ to the
  pre-activation entry has $|delta| <= M$ with $epsilon + M < 1$, then
  the post-tanh slot moves by at most $2(epsilon + M) |delta|$. The
  bound shrinks linearly in both the saturation slack $epsilon$ and the
  perturbation magnitude $M$; at deep saturation a bounded write barely
  moves the slot. Lean:
  `EmenderLatching.emender_saturation_insensitivity`. A two-input
  corollary `EmenderLatching.emender_saturation_two_inputs_close` lifts
  the bound to two value vectors sharing one key.

- *Sign-preserving hold.* With $lambda >= 1$ and a bounded write
  $|W| <= T < lambda (1 - eta)$ at slot $(i,j)$, a slot already
  saturated with $|S_(i,j)| > 1 - eta$ keeps its sign after one Emender
  step and stays bounded below in magnitude by
  $tanh(lambda (1 - eta) - T)$. For $lambda > 1$ and small $eta, T$ the
  argument $lambda (1 - eta) - T$ exceeds $1$, so the lower bound
  exceeds $tanh 1 approx 0.76$ and the latch holds with margin. The
  hypothesis $T < lambda (1 - eta)$ is the "no sufficient counter-delta"
  clause; counter-delta release is the next result. Lean:
  `EmenderLatching.emender_latch_holds_by_default`.

- *Counter-delta release.* With $lambda > 0$, $S_(i,j) > 0$, and a write
  contribution $W < -lambda S_(i,j)$ in the opposite sign, the
  post-update slot is strictly negative — the latch flips out of its
  basin in one step. Symmetrically for the negative basin. The release
  threshold is $T^* := lambda |S_(i,j)|$; any counter-delta exceeding
  $T^*$ in the opposite sign suffices. Lean:
  `EmenderLatching.emender_latch_releases_on_counter_delta`. The
  quantitative companion
  `EmenderLatching.emender_latch_release_quantitative` shows that if the
  counter-delta exceeds the threshold by margin $mu > 0$, the
  post-update slot lies on the opposite side of the origin with
  magnitude at least $tanh mu$.

#set list(indent: 0em)

#heading(level: 2, numbering: none)[The NC#super[1] statement]

The precise NC#super[1] claim, stated conservatively:

#block(inset: (x: 1.5em), [
*The Emender is, at fixed width and precision, a finite-state recognizer (Lean:
`fixed_precision_state_space_finite`). Within that ceiling, an
orthonormal-key configuration of the Emender update realizes the $S_5$
prefix tracker (Lean: `EmenderRealizesS5.emender_realizes_s5_tracker`); $S_5$ is
non-solvable (`s5_not_solvable`) and by Barrington's theorem
@barrington1986 (cited; not formalized in this work) the $S_5$ word
problem is NC#super[1]-complete. The Emender therefore reaches the top of
NC#super[1] in the canonical regular-language witness.*
])

#heading(level: 2, numbering: none)[Explicit non-claims]

The trusted core does *not* prove the following, and the paper does not
claim them. (i) A Lean lower bound covering all linear-scan models on
$S_5$. (ii) Barrington's theorem itself; we cite it. (iii) Any "the Emender
goes beyond NC#super[1]" or "the Emender goes beyond TC#super[0]" claim;
these are families-wide impossibility statements outside the trusted surface. (iv)
A formal proof that a trained real-valued Emender with empirically learned
weights exactly recovers the lookup table; only the realizability is
proved.

// ── 8. Related Work ───────────────────────────────────────────────────────────
= Related Work <sec:related>

#heading(level: 2, numbering: none)[Ancestry: the fast-weight line on delta correction]

The Emender update belongs to a long line that treats memory as a
structure to correct rather than to refill. The earliest formulation is
Widrow and Hoff's least-mean-square rule @widrow_hoff_1960, which writes
the error between target and current output back into the weight vector
of an adaptive linear element. Schmidhuber recast the same idea in a
recurrent setting under the name *fast-weight programmers*
@schmidhuber_1992_fastweights: a slow controller emits keys and values
that update a fast-weight matrix online, with the update being a
correction against what the matrix currently predicts at the addressed
slot. Schlag, Irie, and Schmidhuber made the bridge to modern attention
explicit @schlag_irie_schmidhuber_2021, showing that linear-transformer
kernels with cumulative outer-product updates *are* fast-weight
programmers with a raw additive write, and proposing the delta-rule
write as a stability and capacity fix. DeltaNet @deltanet2024 carried
the delta rule into a parallelizable linear-recurrent language model,
demonstrating the rule at billion-parameter scale in the linear-state
regime. The Emender extends this line with three properties the linear
instantiations forfeit: a nonlinear matrix state, the saturation
latching of §3 ingredient (c) that turns slot-wise overwrite
into persistent binding, and width-axis parallelism via
multi-programming that recovers throughput without time-axis
linearization.

#heading(level: 2, numbering: none)[Linear-state recurrent language models]

The dominant comparison cohort is the family of linear-state
billion-parameter recurrent LMs surveyed in §2: Mamba @mamba2024 and
Mamba2 @mamba2_2024 (selective SSMs); RetNet @retnet2023 (decayed
linear attention); GLA @gla2023 (gated linear attention); DeltaNet
@deltanet2024 (delta rule, linear in $S$); Gated DeltaNet
@gated_deltanet2024 (gated delta rule); RWKV-4/5/6/7 @rwkv4_2023
@rwkv7_2025 (linear-state WKV / generalized delta rule with linear
state); HGRN2 @hgrn2_2024 (gated linear RNN with state expansion);
MinGRU/MinLSTM @mingru_2024 (input-only gates, linear scan);
mLSTM/xLSTM-7B @xlstm7b2025 (covariance update); Griffin/RecurrentGemma
@griffin2024 (RG-LRU, often hybridized with local attention). As a
structural class these models sit inside TC#super[0]
@merrill2024transformers. The empirical panel in §6 directly tests
Gated DeltaNet as the matched representative of this cohort and observes
$S_5$ collapse at length.

A newer linear-state result, Gated DeltaNet-2 @gated_deltanet2_2026,
decouples channel-wise erase and write gates and is reported by its
authors to outperform GDN. It is concurrent with this work, and a
comparison against it is left to future work. GDN-2 extends
retrieval and state-tracking behavior inside the linear-state
envelope, whereas this paper tests the expressivity axis opened by
serial nonlinear state updates. Decoupled erase/write gates are also a
concrete next-ablation prompt for the multi-programmed substrate,
since they are the kind of bounded per-step-body variant that should
not require a new time-parallel scan.

#heading(level: 2, numbering: none)[Pure-nonlinear-recurrent peers and adjacent nonlinear-state work]

*M²RNN* @m2rnn2026 (Mishra, Tan, Stoica, Gonzalez, Dao 2026) introduces
nonlinear matrix-state recurrence with a raw-write update
$Z = tanh(H W + k v^T)$ and demonstrates that it trains at 7 B MoE
scale in *hybrid form* (nonlinear-matrix-recurrent layers interleaved
with attention layers) on Nemotron-CC-v2 at 410 M dense and 7 B MoE.
M²RNN-CMA, used as a
within-class baseline throughout this paper, is the *pure-recurrent
variant* of M²RNN's architecture with no attention layers, CMA-tuned to
restore stable training at 1.3 B. The contribution of this work
relative to M²RNN is to extend the viability demonstration to the
*pure-recurrent* setting and to add the direct comparison with
the Emender under matched per-architecture CMA-ES. The empirical separation in
§6 and the formal one-step resource separation in §7 quantify the
update-rule difference between delta-correcting (the Emender) and raw-write
(M²RNN-CMA) within the pure-nonlinear-recurrent class. Mishra et al.
also report hybrid M²RNN configurations favorably against
Mamba2 and Gated DeltaNet hybrids at matched parameter and token
budgets under a uniform fixed-hyperparameter protocol (their §5.2);
those hybrid configurations fall outside the pure-nonlinear-recurrent
class, since the inserted attention layers are excluded by the
no-hybridization criterion of §1.

*xLSTM-1.3B* @xlstm2024 is a 7:1 mixture of mLSTM (linear) and sLSTM
(nonlinear) blocks; 87.5% of its blocks are linear-state, so xLSTM-1.3B
is not pure-nonlinear-recurrent by the no-linearization criterion of
§1. It is the closest scale band among prior nonlinear-recurrent
results, included as a peer with the caveat that its nonlinear-block
share is small. The xLSTM-7B follow-up @xlstm7b2025 uses *only* mLSTM
(linear) blocks and is correspondingly further from the
pure-nonlinear-recurrent class.

*Titans* @titans2025 uses an MLP memory with online gradient updates
and is qualitatively nonlinear-state, but it is hybridized with
attention (also outside the no-hybridization criterion) and has
not been evaluated as a pure recurrent model at Pile-class scale.
*Zeroth-order LSTM scaling* @lstm_zoo_2025 demonstrates 1 B-scale LSTM
training under a non-gradient regime; it is not a standard-gradient
training comparable.

*Classical LSTM/GRU* @lstm1997 is the historical background: no
published classical LSTM/GRU model has reached $>= 500$ M parameters on
a Pile-class corpus. It is the unsuccessful empirical record that
motivated the field's move to linear-state and hybrid alternatives.
This is the assumption this paper revisits.

// ── 9. Limitations ───────────────────────────────────────────────────────────
= Limitations <sec:limitations>

#heading(level: 2, numbering: none)[Formal scope]

The trusted Lean core covers: orthonormal-key realization of the $S_5$
tracker; one-step separation from raw-write matrix RNNs; the k-step
extension on a constructed 2D witness alphabet for every $k >= 1$
(`emender_m2rnn_k_step_separation`); the finite-state ceiling. The
explicit non-claims, retained as the boundary of the trusted surface:
(i) a Lean lower bound covering all linear-scan models on $S_5$; (ii)
Barrington's theorem itself; (iii) an $S_5$-generator-specific $T(d)$
capacity bound (the k-step separation runs on the constructed 2D
alphabet, not the $S_5$ generators); (iv) families-wide "goes beyond
NC#super[1]" or "goes beyond TC#super[0]" impossibility; (v) that
empirical Emender weights recover the lookup-table realization; (vi)
the slot-wise latching set lifted to an architecture-level
`latchAttractor`, nor an $S_5$-coset basin-survival statement against
an active adversary. Each is named next-theorem work — bullets (iii)
and (vi) are the load-bearing targets of §12, the former requiring a
bounded-precision raw-write class plus Merrill-style state-counting
machinery not yet in Mathlib, the latter requiring the
`MemorySemantics.latchAttractor` lift on the
`RecurrentResourceFormalism` signature. The §6 $S_5$ accuracy of 0.79
at $T = 128$ and the length-extrapolation curves are the empirical
companion to the constructed-alphabet k-step result on the $S_5$
generator alphabet itself.

#heading(level: 2, numbering: none)[Evidence structure]

What rests on what. The Lean separation is seed-independent: the
proof is the proof. The 8 M expressivity gap on $S_5$ ($0.79$ vs
$0.36$ vs $0.22$) is across three seeds per architecture. The 1.3 B
wallclock comparison is one continuously-trained seed per architecture at
the current multi-week training extent.

The within-class ordering the comparison records (Emender ahead of the
raw-write update under matched per-architecture CMA-ES) is not a
free-standing observation: the same sign is selected by four
independent CMA-ES sweeps at the same 1.3 B parameter scale,
spanning 250+ candidate configurations per architecture in aggregate
across chunk-512 and chunk-2048 training budgets and across
reseed-and-reposition rounds. Under the exact E88 delta-off
ablation the candidate-budget gap is 0.033 nats/token, and the comparison
keeps the same sign at 0.014 nats/token at the current multi-week
extent. The $H = 370$
shape used here was not hand-chosen but sits inside the
$H = 270$–$460$ interior band that the searches repeatedly selected
for E88 at $N = 32$, while the raw-write arm drifts to systematically
higher head counts. What remains single-seed at this scale is the
multi-week trajectory itself; the within-class ordering and the
head-geometry preference are CMA-replicated.

#heading(level: 2, numbering: none)[Length extrapolation is the next frontier on $S_5$]

The Emender stays ahead at every sequence length under length extrapolation (0.79 at
$T = 128$, 0.42 at $T = 256$, 0.22 at $T = 512$, 0.11 at $T = 1024$
on $S_5$), and the gap to baselines widens with length on parity and
FSM tracking (§6). Solving $S_5$ to ceiling at $8 times$ training
length is the next $S_5$ result for any recurrent family in this
sweep; it requires either an explicit $S_5$-generator capacity bound
(§12) or a curriculum that closes the within-Emender degradation
curve.

#heading(level: 2, numbering: none)[Per-architecture CMA-ES is best-effort matched; broader-search is the next experiment]

The 1.3 B comparison (§5, @fig_lm_racers) runs each baseline under its
own CMA-tuned shape and hyperparameter choice with matched candidate
budget. The protocol delivers symmetry of effort across architectures
at the configured search budget. The cleanest sharpening is to extend
the search with further CMA-ES generations per family at 1.3 B under
the same budget rule; this separates "under the search effort used
here" from "under any matched search effort", and §12 names it as the
load-bearing follow-up.

#heading(level: 2, numbering: none)[Geometry-sensitivity of the update-rule claim]

The empirical within-class ordering scopes to CMA-equilibrated
geometry. The same update family (M²RNN) diverges at 1.3 B in the
published paper shape and is stable under the CMA-tuned reshape (loss
2.67 after roughly 18 wall-clock days). The §6 expressivity comparison runs at
parameter-matched 8 M with geometry held constant across families;
the §5 language-modeling comparison runs against the CMA-reshaped
M²RNN. The Lean separation of §7 is unconditional on shape but is
per-step. The shape-independent ceiling is the family-level
representational separation already covered by Theorem set C
(`emender_m2rnn_one_step_resource_separation_embeds`); a
shape-conditional empirical ordering and a shape-independent one-step
formal separation cover the geometry axis from both sides.

#heading(level: 2, numbering: none)[Design-space asymmetry of the 8 M defaults]

The 8 M defaults are matched in the sense that no architecture
received probe-specific HPO, but they are not matched in *design
space*: the Emender's defaults are the endpoint of an ablation
lineage selected on language-modeling loss, the same objective GDN's
and M²RNN's authors used for their published defaults, but explored
within a design space chosen under an expressivity-efficiency
hypothesis (§13). The matched-no-tuning condition controls for
differential probe-specific effort, not for this design-space
asymmetry. The §6 $S_3$ control isolates the part of the
within-class claim that is immune to it (raw-write at 0.31 on a
six-element solvable group, well below the non-binding capacity
ceiling). Closing the asymmetry itself is a matched-search experiment
on the 8 M shape (running CMA-ES on each baseline at 8 M under a
state-tracking fitness); it is named alongside the 1.3 B
wider-search follow-up.

#heading(level: 2, numbering: none)[The opposite architectural bet: hybrids]

A concurrent strand of work places the opposite architectural bet.
*OLMo-Hybrid 7B* @olmohybrid2026 interleaves state-space blocks with
attention, on the premise that hybrid stacks express things beyond
what either pure transformers or pure linear RNNs can do. The Emender
takes the other bet: a pure-nonlinear Emender stack at 1.3 B matches
GDN in the wallclock loss band, which refutes the assumption that
pure nonlinear recurrence cannot scale at all. The hybrid-degradation
finding in §6
($[upright("Emender"), upright("Emender"), upright("GDN"), upright("GDN")]$
underperforms either pure family on modular counter and FSM tracking
at 8 M scale) is a capability-preservation observation: state-tracking
capability does not survive dilution by linear-scan blocks in our
sweep. The two bets answer different questions ("can pure nonlinear
recurrence scale at all?" here versus "what does a well-mixed hybrid
express?" for OLMo-Hybrid); the answers do not contradict.

#heading(level: 2, numbering: none)[Training duration and result scope]

The language-modeling results are for the released v0.3 checkpoints,
a 22.1-23.8 wall-clock-day training extent per architecture.
The comparison panel (@fig_lm_racers)
records the loss-vs-wallclock curve at this extent; further rounds
extend the curves.

#heading(level: 2, numbering: none)[Open architectural choices]

Several internal questions remain open: the output gate, the
state non-linearity ($tanh$ vs linear), and the decay parameterization
(simple sigmoid vs Mamba2-style log-space). All three tie on loss at
small scale; the 1.3 B architecture keeps the conservative
settings on the strength of state-tracking and stability data, not a
clean ablation at 1.3 B.

#heading(level: 2, numbering: none)[Transformer comparison]

A matched-scale transformer is not evaluated here, and a 1.3 B
attention baseline trained under the same protocol is deferred to
future work. At the 2 K context this paper trains on, attention is
likely competitive or better; the Emender's case is about full
nonlinear-recurrent expressivity and the long-sequence regimes where
attention's quadratic cost bites. The omission is scope, not result.

// ── 10. Conclusion ────────────────────────────────────────────────────────────
= Conclusion <sec:conclusion>

Two results.

First, a pure-nonlinear-recurrent language model reaches sub-1-bpb on
The Pile at 1.3 B-class on a single workstation-class GPU: E88 at
0.973 bpb after about 23 wall-clock days. The throughput
route is width-axis *multi-programming*, which runs 22,200 small
bounded recurrent programs per token, each a $32 times 32$
matrix-state tile in registers, with the time loop serial inside each
program. The recipe is update-rule-agnostic: both pure-recurrent
nonlinear instances trained here satisfy the same multi-programming
predicate at 1.3 B (Lean: `multiProgrammed_admits_m2rnn_and_emender`).

Second, the delta-correcting update accesses a strictly larger
one-step function class than the raw-write update at matched per-token
FLOP cost, and the gap persists under every $k$-step composition. The
Lean 4 trusted core establishes this via
`emender_m2rnn_one_step_resource_separation_embeds` (set C),
`emender_m2rnn_k_step_separation` (set C′), and
`emender_m2rnn_flop_class_equiv` (set D); the same core proves that
an orthonormal-key Emender configuration realizes the $S_5$ tracker
(`EmenderRealizesS5.emender_realizes_s5_tracker`). The predicted
ordering shows up empirically at the 8 M overparameterized probe
shape: with capacity non-binding by orders of magnitude, raw-write
M²RNN-CMA stalls at 0.31 on the six-element solvable-group $S_3$
control, while the Emender solves $S_3$ to ceiling and reaches 0.79
on the non-solvable $S_5$ probe at training length. The trusted Lean
4 core has no `sorry`/`admit`/`axiom`/`opaque`/`native_decide` in the
import closure.

Under per-architecture CMA-ES at matched candidate budget, E88 lands
in the same loss-vs-wallclock band as Gated DeltaNet at the 1.3 B-class
on The Pile; §5 has the comparison in full.

*Release.* The current release hub is
`https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V02_PUBLIC_RELEASE_HUB.md`.
The checkpoint and loading-code targets are E88
(`https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.3`),
Gated DeltaNet
(`https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.3`), and
M²RNN-CMA
(`https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.3`).
The public paper PDF mirror is
`http://hypervolu.me/~erik/ndm/Garrison_2026_Emender.pdf`.
Companion release docs cover the per-architecture CMA-ES
configurations, the training protocol, and the Triton multi-programming
kernel source. The held-out bits-per-byte appendix (@sec:appendix_bpb)
is reproducible from the same release: the re-exported y-mode checkpoints
ship in the HuggingFace v0.3 trees above, and the measurement scripts
(`scripts/measure_pile_bpb.py`, `scripts/measure_pile_bpb_elman.py`) and
sha-verified held-out slice manifests are in the public `emender` repo.

// ── 11. Testable Predictions ─────────────────────────────────────────────────
= Testable Predictions <sec:predictions>

The class-level and within-class findings of this paper extend to
predictions testable at the next scale step or under modest extra
training. We state them as predictions so that future
training rounds can falsify them.

#set list(indent: 1em)
- *Width-axis multi-programming scales beyond 1.3 B without throughput
  collapse.* The multi-programmed substrate harvests throughput from
  many small bounded heads rather than from time-axis linearization;
  the per-head register/SRAM working set is independent of total
  parameter count, so the recipe is expected to survive scale-up to the
  3–7 B band. Falsified if per-token throughput degrades faster than
  the parameter count grows, beyond the kernel-arithmetic baseline.

- *The Emender's state-tracking advantage persists at scaled training
  budgets.* The §6 $S_5$ and $S_3$ gaps are at the 8 M parameter-matched
  probe scale under matched no-tuning. At longer training-token budgets
  at 1.3 B and beyond, the within-PNR ordering and the Emender-vs-GDN
  gap on state-tracking probes should hold qualitatively. Falsified if
  the $S_5$ or $S_3$ gap closes under matched-compute training at scale.

- *Emender fine-tuned on reasoning benchmarks separates from
  GDN/M²RNN-CMA on tasks requiring genuine state-tracking.* The §6 QA
  panel at the current training budget sits below the threshold where
  reasoning benchmarks differentiate. After targeted fine-tuning on a
  reasoning suite (BIG-Bench Hard, ReCLor, FOLIO, multi-step object
  tracking), the Emender is expected to separate from the linear and
  raw-write baselines specifically on items where multi-step prefix
  tracking is load-bearing (object tracking, formal fallacies,
  multi-step entailment). Falsified if Emender, GDN, and M²RNN-CMA
  remain within standard error on state-tracking-shaped subsets after
  matched fine-tuning.

- *PNR update-rule variants are per-step-body edits under the
  multi-programmed kernel.* The fused forward/backward Triton kernel
  (§4) is parameterized by the update map and the gating. A different
  bounded PNR update rule at the same matrix-state signature should be
  expressible by editing the per-step body, without satisfying an
  associative-scan or chunkwise-WY compatibility constraint and with the
  multi-programmed time loop unchanged. Falsified if a representative
  variant (e.g., decoupled erase/write gates in the style of
  Gated DeltaNet-2 @gated_deltanet2_2026, or a higher-order delta with
  two-step memory) requires kernel rewrites beyond the per-step body.

- *Emender combined with hybrid attention should outperform either
  alone.* The §6 hybrid-degradation result (Emender/GDN AABB
  underperforms either pure family on state tracking at 8 M) is read
  here as a property of the Emender/GDN pairing specifically. The
  prediction is that an Emender-with-attention hybrid (Emender layers
  interleaved with attention layers, in the style of M²RNN's hybrid
  configuration) outperforms both pure Emender and pure attention on
  combined state-tracking and long-range mixing benchmarks at
  matched parameter and token budgets. Falsified if the hybrid
  underperforms either constituent at matched budget on the combined
  benchmark.

- *A broadened CMA-ES sweep places the head-count interior optimum
  above $H = 370$.* The §5 per-architecture search arrived at $H = 370$
  after successive bound repositionings against persistent upward
  pressure from the optimizer. The conjecture is that the same fairness
  search under broader bounds, subject only to the kernel-occupancy
  ceiling of the multi-programmed substrate, returns an interior
  optimum strictly above 370 on the Emender at 1.3 B; $H tilde 1000$
  is the next bracket worth testing. Falsified if a broadened CMA-ES
  sweep under matched per-token compute returns an interior optimum at
  or below $H = 370$.

#set list(indent: 0em)

These predictions are neutrally stated. Some will falsify, and the
falsifications are the next round of evidence.

= Future Work <sec:future_work>

The results above leave the following directions open.

#heading(level: 2, numbering: none)[$S_5$-generator-specific capacity bound]

The k-step separation in §7 (`emender_m2rnn_k_step_separation`) proves
that for every finite $k$ the gap persists on the constructed 2D
witness alphabet. The missing piece — and the obvious next-paper
target — is a formal capacity bound on the $S_5$ generator alphabet
itself: a statement of the form "for any fixed-weight raw-write RNN at
state dimension $d$ there is an explicit $T(d)$ past which $S_5$ coset
tracking is unreachable, while the Emender at the same $d$ tracks it."
That requires bounded-precision class machinery and Merrill-style
state-counting lower bounds not yet in Mathlib. The §6
length-extrapolation curves (parity, FSM tracking, modular counter;
Emender-vs-baseline gap widening monotonically with sequence length)
are the empirical signature on natural-language-shaped sequences.

#heading(level: 2, numbering: none)[A partial order on PNR update rules]

The within-PNR ordering established here (delta-correct $>$ raw-write)
is the first strict instance of an expressivity order on the
pure-nonlinear-recurrent class at matched per-token FLOP class
(`emender_m2rnn_flop_class_equiv`). Whether this relation extends to a
*partial order* with an NC#super[1] ceiling (Barrington) on the broader
PNR space (in particular, whether antisymmetry and transitivity hold
over the full space) is open. The maximal element under matched
asymptotic FLOP class (which PNR update rule has the highest one-step
expressive power for its compute class) is the open horizon, with the
`RecurrentResourceFormalism` Lean machinery as the tool for climbing
the order. Follow-up experiments are exploring more flexible bounded
update rules, including GDN-2-style split erase/write variants; those
runs are ongoing, and any empirical or formal results for these variants
will be reported in an update rather than treated as evidence here.

#heading(level: 2, numbering: none)[Additional seeds and architecture-internal revalidation]

Each 1.3 B model trains for a multi-week wall-clock span per
architecture, the standard unit of evidence at this scale class;
additional seeds at this band are a multi-week investment per seed.
Several architecture-internal choices in the Emender, including the
output gate, the non-linearity on the state ($tanh$ vs linear), and
the decay parameterization (simple sigmoid vs Mamba2-style log-space),
show loss-only ties at small scale; revalidation of each at 1.3 B is
open.

#heading(level: 2, numbering: none)[Scale beyond 1.3 B]

The wallclock-band convergence is observed at 1.3 B parameters on
The Pile. Whether the same band convergence holds at larger scale, on
larger corpora, or under longer training is open.

#heading(level: 2, numbering: none)[Cleanest within-class HPO follow-up]

The within-PNR ordering reported here is conditional on the
per-architecture CMA-ES protocol of §5. Extending that protocol with
further CMA-ES generations per family at 1.3 B, under the same budget
rule, separates the conditional reading "under the search effort used
here" from the family-level reading "under any matched search effort."
It is the sharpest single experiment for fixing where the
Emender-vs-M²RNN-CMA gap actually sits.

#heading(level: 2, numbering: none)[Latching-to-$S_5$-basin bridge and `latchAttractor` refinement]

The §7 latching set establishes the slot-level half of the Emender
primitive, with three theorems covering saturation insensitivity,
default hold under bounded delta, and release on counter-delta. Two
formal targets sit on top of that set. The architecture-level lift
promotes the slot-wise latching theorems to a
`MemorySemantics.latchAttractor` label on the Emender signature in the
`RecurrentResourceFormalism`, making slot-level behavior a statement
about the matrix-state attractor structure of the whole layer. The
basin-survival bridge takes an $S_5$-coset realized in an
orthonormal-key Emender and shows it survives an adversary whose
per-slot perturbation budget sits below the §7 release threshold,
lifting the slot-level theorems to a coset-level guarantee on the
state-tracking task. The first is a formalism refinement; the second
is a Lean-targeted theorem that connects §7 to the §6 state-tracking
empirics.

// ── Appendix A — E63→E88 lineage and ablation notes ────────────────────────────
= Appendix: Lineage of the E63 $arrow$ E88 experimental program <sec:appendix>

The Emender architecture as presented in §3 is the endpoint of a multi-year
ablation lineage. The named milestones referred to here are internal
codenames; they are documented in the appendix for reproducibility of
the design history and are not used in the body of the paper. The
reference implementation of the 1.3 B Emender stack lives at
`ndm/models/e88_fused.py`; the fused Triton recurrence kernel sources
are at `ndm/triton/e88_triton_forward.py` and `e88_triton_backward.py`.

- *E63 (nonlinear delta design).* Established that linear-in-state
  recurrences cannot pass the Siegelmann–Sontag boundary
  @siegelmann1995 and that placing the non-linearity on the matrix
  state itself (rather than only on gates or the output) is the
  expressivity-determining choice. The Emender's $tanh$ on $S$ is inherited
  from this design.

- *E70–E75 (matrix-Elman / delta predecessors).* Established the fused
  Triton kernel pattern (outer product + decay + tanh as one kernel)
  and surfaced the gradient-spike failure mode of unbounded outer
  products. The L#super[2]-normalized key write inherited by the Emender is the
  fix.

- *E88 (the 1.3 B Emender).* Stable at 1.3 B parameters under
  schedule-free AdamW. Key parameterization choices: log-space Mamba2
  decay (with weight-decay exemption); L#super[2]-normalized
  $q, k$; SiLU on input projections before normalization; numerically
  stable $tanh(z) = 2 sigma(2z) - 1$ in the fused kernel; output
  RMSNorm inside the layer removed (−0.10 nats); short convolutions
  removed (−0.027 nats); no write gate; SiLU output gate.

- *M²RNN-CMA versus M²RNN-paper.* The paper-default M²RNN shape
  (dim = 3072, depth = 10, $H = 759$, $N = 16$) diverged at step 8,400
  under the training setup of §5 with gradient norms of $approx 4.2
  times 10^7$. The CMA-tuned reshape (dim = 1920, depth = 21, $H = 370$,
  $N = 16$) is stable; it is the M²RNN-CMA baseline used in §5 and §6.
  The paper-shape divergence and the CMA-tuned stability together show
  that the gradient-conditioning failure is a *geometry* property
  (shared $q, k$ across many value heads), not a property of the
  raw-write update family per se.

= Appendix: Held-out bits-per-byte measurement <sec:appendix_bpb>

This appendix documents how the held-out bits-per-byte figures in §5 are
measured and places them in a landscape of reference models and classical
compressors. Every number is measured by the eval harness described below;
nothing here is re-fitted. The measurement code, slice manifests, and
re-exported checkpoints are public (see the availability statement in §10).

#heading(level: 2, numbering: none)[Tokenizer-invariant held-out bpb]

So that models with different tokenizers are comparable on identical
bytes, held-out bpb is computed from each model's real summed negative
log-likelihood over the slice and the slice's real UTF-8 byte count:

$ "bpb" = ("total NLL"_"nats" \/ ln 2) \/ "total UTF-8 bytes". $

Each model uses its *own* tokenizer, and the denominator is the shared
byte count, so bytes/token is a *measured* per-tokenizer quantity (4.00
for the GPT-NeoX BPE, 3.82 for our `p50k_base` runs on the canonical
slice) — *not* the fixed $"bytes/token" = 3.92$ constant used for the
train-loss curve in §5. The held-out and train-loss figures are
therefore distinct objects: they differ in normalization as well as in
the data scored. Every token is scored once under a sliding window
(context 2048, stride 1024) with up to 2047 tokens of left context, the
standard fixed-length-model perplexity recipe; out-of-distribution
anchors GPT-2 XL and OPT-1.3B run at context 1024 (GPT-2 XL position
limit). Pipeline: `scripts/measure_pile_bpb.py` (open models) and
`scripts/measure_pile_bpb_elman.py` (our models, in-harness forward).

#heading(level: 2, numbering: none)[Held-out slice provenance]

The canonical held-out slice is a whole-line 10 MB region (9,999,511
bytes, sha256 `3e4241a9…`) taken at byte offset $10^12$ (≈76.5% into the
1.31 TB Pile master file) — a deep offset the 1.3 B runs' sub-one-epoch
stream from the file start is least likely to have consumed. The
five-slice robustness check of §5 adds four further independent,
non-overlapping, sha-verified slices at offset fractions
0.137 / 0.341 / 0.523 / 0.911, each ≈10 MB and ≈2.5 M scored tokens.
Identical bytes feed every model, so the byte denominator is shared and
bpb is comparable across tokenizers. A block-loss sanity gate (mean
nats/token on the first 2048-token block must lie in $[1.5, 4.0]$, model
train loss ≈2.6) ran before any of our models' bpb was trusted; all
gated runs passed. The sha-verified slice manifests are released with the
measurement code (see the §10 availability statement).

#heading(level: 2, numbering: none)[Second distribution: Common Pile (contamination control)]

Because the open Pile-trained references (Pythia, GPT-Neo) saw the Pile
in training, their Pile numbers are effectively in-distribution. As an
independent check we re-measure through the identical pipeline on a
held-out slice of Common Pile (v0.1 main-mix; 9,999,606 bytes,
document-aligned on the `0x1E` record separator, at a random deep offset
65.5% into its 1 TB corpus), a permissively-licensed corpus none of these
models trained on. The Pile-trained models score within $<= 0.003$ bpb of
their Pile numbers there (Pythia-1.4B $+0.0028$, GPT-Neo-1.3B $+0.0002$,
Pythia-1B $+0.0003$), so the Pile result is *not* contamination-inflated.
Our three models on the same Common-Pile slice measure Emender 0.981, GDN
0.963, M²RNN-CMA 0.973 — within ≈0.02 bpb of their Pile figures, with
ordering GDN < M²RNN-CMA < Emender (again not the train-loss ordering),
consistent with a small distribution shift rather than a blow-up. Best
classical coder on Common Pile is xz -9 at 2.061 bpb.

#heading(level: 2, numbering: none)[Checkpoints and reproducibility]

The three held-out checkpoints (Emender step 1,542,000; GDN 2,031,000;
M²RNN-CMA 1,491,000) are pinned to persistent storage with full
schedule-free optimizer state and sha-verified. Recovering a correct inference
forward from these schedule-free @schedulefree2024 runs requires the
*y-mode* weight swap (load the optimizer state and call
`optimizer.train()`); the saved model weights alone are the
eval-extrapolated *x-mode* view and are catastrophic at inference
(≈18 nats/token). The public HuggingFace v0.3 release was re-exported with
the corrected y-mode weights and verified, by a clean-cache readback from
the hub through the bundled modeling code, to reproduce the held-out bpb
to within $2 times 10^(-5)$ nats (Emender 0.96613, GDN 0.96612,
M²RNN-CMA 0.96132). Only the `v0.3` revision carries this corrected
re-export; the earlier public revisions (`v0.1`, `v0.2`) were shipped from
the same schedule-free *x-mode* pipeline, predate the fix, and are not
usable for reproduction — cite `@v0.3` only.

#heading(level: 2, numbering: none)[Where the held-out bpb sits — landscape, not a ranking]

#figure(
  align(center)[#table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, right, right, left),
    stroke: 0.5pt,
    inset: 5pt,
    table.header(
      [*Tier*], [*Model / coder*], [*Params*], [*Held-out bpb*], [*Trained on Pile?*],
    ),
    [Ours (matched budget)], [Emender (E88)], [1.273 B], [0.966], [yes, ≈0.05 epoch],
    [Ours (matched budget)], [GDN], [1.352 B], [0.966], [yes, ≈0.05 epoch],
    [Ours (matched budget)], [M²RNN-CMA], [1.307 B], [0.961], [yes, ≈0.05 epoch],
    [Open Pile-trained], [Pythia-1.4B], [1.42 B], [0.7157], [yes, ≈1 epoch (in-dist.)],
    [Open Pile-trained], [GPT-Neo-1.3B], [1.32 B], [0.7403], [yes, ≈1 epoch (in-dist.)],
    [Open Pile-trained], [Pythia-1B], [1.01 B], [0.7423], [yes, ≈1 epoch (in-dist.)],
    [Familiar OOD], [OPT-1.3B], [1.32 B], [0.8615], [no (OOD)],
    [Familiar OOD], [GPT-2 XL], [1.56 B], [1.0137], [no (OOD)],
    [Classical], [xz -9 (LZMA)], [—], [2.190], [n/a (model-free)],
    [Classical], [gzip -9 (DEFLATE)], [—], [2.803], [n/a (model-free)],
  )],
  caption: [
    *Held-out bits-per-byte landscape — context, not an architecture
    ranking.* All neural numbers are tokenizer-invariant held-out bpb on
    the canonical 10 MB Pile slice; classical coders are single-stream
    over the same bytes. The vertical ordering is *dominated by training
    budget and train/test distribution, not by the update rule*, and must
    not be read as a quality verdict. Our three models trained on
    ≈15–16 B tokens (≈0.05 epoch of the Pile); the open Pile-trained
    references saw ≈10–20$times$ more (≈1 epoch, ≈300 B tokens) *and* are
    in-distribution on this slice (it is effectively training data for
    them — possible contamination), so their 0.72–0.74 is closer to a
    train-loss than a held-out number. The OOD anchors (OPT, GPT-2 XL)
    are out-of-distribution on the Pile, so their higher bpb is expected
    and is not a quality knock; GPT-2 XL measures 1.0137 here versus the
    Pile paper's published 1.0468 @thepile2020, a ≈0.03-bpb pipeline
    check confirming this single slice reads slightly low against the full
    Pile test set. The only architecture-isolating comparison is the
    matched-budget three-way at the top, which §5 shows is a statistical
    tie.
  ],
) <tab_bpb_landscape>

#bibliography("refs.bib", title: "References", style: "ieee")
