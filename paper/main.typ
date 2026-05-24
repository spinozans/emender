// Nonlinear Delta Memory — main paper source
// Format: Typst (https://typst.app), NOT LaTeX.
// Build: bash paper/build.sh  →  paper/Garrison_2026_NDM-<commit>.pdf
//
// Template: arkheion 0.1.2 (the de-facto Typst arxiv-preprint template).
// See paper/template_choice_v7.md for rationale and the porting notes.

#import "@preview/arkheion:0.1.2": arkheion, arkheion-appendices

#show: arkheion.with(
  title: "Pure Nonlinear Recurrent Language Models",
  authors: (
    (
      name: "Erik Garrison",
      email: "erik.garrison@gmail.com",
      affiliation: "Poietic PBC; Department of Genetics, Genomics and Informatics, University of Tennessee Health Science Center, Memphis, TN 38163, USA",
      orcid: "0000-0003-3821-631X",
    ),
  ),
  abstract: [
    Recurrent language models are presumed to require linearisation
    (selective state-space models, gated linear attention, delta-rule
    linear scans) or hybrid attention bolt-ons to scale; pure nonlinear
    matrix-state recurrence has been treated as wall-clock-impractical at
    the billion-parameter band. This paper introduces *Pure Nonlinear
    Recurrent Language Models* (PNR) and trains two PNR instances,
    *NDM* (delta-correcting update $S <- tanh(d S + k(v - S^T k)^T)$) and
    *M²RNN-CMA* (raw-write update $tanh(H W + k v^T)$), at 1.27 B
    parameters under per-architecture CMA-ES hyperparameter optimisation,
    alongside *Mamba2* and *Gated DeltaNet* (GDN) as frontier-class
    linear-recurrent baselines. NDM matches the linear baselines on the
    shared wall-clock scaling curve, while M²RNN-CMA trails throughout —
    isolating the delta-correcting update rule as the within-PNR
    differentiator. The
    *one-step representational* counterpart of the within-PNR gap is
    formalised in Lean 4 — `NDMRealizesS5.ndm_realizes_s5_tracker`
    realises the $S_5$ tracker through the delta-correcting update, and
    `RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`
    proves no fixed-weight raw-write parameterisation can match it; the
    predicate `RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_ndm`
    places both PNR instances in a common multi-programmed parallelism
    class, and `RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv`
    shows their per-token FLOP counts share a common $c_1 d^2 + c_2 d$
    envelope. These results establish PNR as a viable architecture class
    for scalable language modelling and exhibit one strict expressivity
    instance within it (delta-correct $>$ raw-write); whether the
    within-PNR ordering extends to a partial order with an
    NC#super[1] ceiling is the open next-paper question. The trusted Lean
    4 core has no `sorry`/`admit`/`axiom`/`opaque`/`native_decide` in the
    import closure.
  ],
  keywords: (
    "recurrent neural networks",
    "language modelling",
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

// Math shortcuts used throughout the body.
#let nd = $upright("NDM")$
#let s5 = $S_5$
#let s3 = $S_3$

// ── 1. Introduction ───────────────────────────────────────────────────────────
= Introduction <sec:intro>

The dominant recipe for scaling recurrent language models has been to
make the recurrence *linear in the hidden state*. Selective state-space
models (Mamba, Mamba2 @mamba2024 @mamba2_2024), gated linear attention
(GLA @gla2023), delta-rule transformers (DeltaNet
@deltanet2024, Gated DeltaNet (GDN) @gated_deltanet2024), RWKV
@rwkv4_2023 @rwkv7_2025, RetNet @retnet2023, HGRN2 @hgrn2_2024 and the
mLSTM block of xLSTM @xlstm7b2025 all write the recurrence in the form
$h_t = A_t h_(t-1) + b_t$ where the transition $A_t$ and the drive $b_t$
depend only on the current input $x_t$. This form has two practical
consequences. First, it can be evaluated in parallel along the time
axis through prefix-scan or chunkwise matrix-multiplication. Second, by
work of Merrill, Petty and Sabharwal @merrill2024transformers and
others, every such architecture lives strictly inside the complexity
class TC#super[0]: at fixed precision and width it cannot, asymptotically,
solve any problem outside that class — including the simplest
non-solvable group word problem, $S_5$.

The *nonlinear*-state counterpart — a recurrence in which $h_(t-1)$ (or a
nonlinear function thereof) enters a non-linearity that governs the
update itself — has, historically, been the kind of model the field
trained at sub-billion scale on small corpora and then abandoned. The
classical LSTM @lstm1997 and GRU never appeared at $>=$500 M parameters
on a Pile-class corpus; large-scale language modelling moved to
Transformers, then to linear-state recurrence, on the implicit premise
that pure nonlinear recurrence was too slow to train at scale. The
result is a literature in which every billion-parameter recurrent
language model is either linear-state or a hybrid of recurrence with
attention.

This paper introduces *Pure Nonlinear Recurrent Language Models*
(PNR) — recurrent language models with *nonlinear matrix-valued
state*, *no attention layers*, *no linearisation tricks* (no
kernel-feature linearisation, no SSM-style linear recurrence) and *no
hybrid bolt-ons* (no inserted attention layers, no transformer
interleaving). A PNR architecture is defined by its per-step update
rule on the matrix state; different update rules give different PNR
instances. The paper studies two: *NDM*, with a *delta-correcting*
update that writes the residual $v - S^T k$ against the current state's
read on the key; and *M²RNN-CMA*, a CMA-reshaped variant of the
M²RNN @m2rnn2026 update family with a *raw-write* update that writes
the new key/value without delta correction. PNR is presented as an
*architecture class*; this paper introduces two early instances and
demonstrates that PNR is competitive with frontier-class
linear-recurrent baselines at 1.27 B parameters.

The premise behind PNR is that pure nonlinear recurrence was
parallelised along the wrong axis, not that the underlying computation
is impractical. PNR achieves linear-time training cost not by
linearising the recurrence — which would forfeit the nonlinear-update
expressivity — but by scaling parallelism along the *width* axis (many
independent recurrent heads operating in parallel) rather than along
the *time* axis. Recurrence kept serial along *time* can still expose
massive parallelism *across many small bounded memory programs*. A modern accelerator with
thousands of independent streaming multiprocessors is happy to run, in
parallel, hundreds of small recurrences if each one fits in registers
and shared memory. This organisation is called *multi-programming*
throughout the paper: the recurrent computation is broken into
independent per-head, per-state-tile, per-batch programs, and
parallelism is harvested across these programs while the time loop
inside each program runs serially. The recipe is structural, not
specific to any single update rule — both PNR instances studied here
satisfy the same multi-programming predicate at 1.27 B (Lean witness
`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_ndm`).

As the reference PNR instance, this paper introduces the *Nonlinear
Delta Memory* (NDM) architecture. Each NDM head owns a small bounded
associative memory $S in RR^(N times V)$ (production: $N = V = 32$). For
input projections $k, q in RR^N$, value $v in RR^V$, scalar decay
$d in (0,1)$ and gate $g in RR^V$, one recurrent step is

$
r &= S^T k          quad &(text("read at address ") k)\
delta &= v - r quad &(text("prediction error"))\
S &= tanh(d dot S + k delta^T) quad &(text("bounded delta write"))\
y &= "silu"(g) dot S^T q  quad &(text("gated read at ") q)
$

We train a 1.27 B-parameter pure-recurrent NDM stack
(dim = 1664, depth = 12, $H = 370$ heads, $N = V = 32$ per head) on The
Pile @thepile2020
with a 2048-token context window, using schedule-free AdamW
@schedulefree2024 and a fused Triton @triton2019 recurrence kernel. The
production implementation is referred to internally by the codename
*E88*; the name *NDM* is used throughout the paper for the architecture
itself, and *E88* only when referring to the specific production model.
The same Triton source runs on NVIDIA CUDA and on AMD ROCm.

#v(0.5em)
*Contributions.* Three distinct contributions are staked here. The first
two are robust results; the third is an open research program that the
first two open up.

#set enum(numbering: "1.", indent: 1em)

+ *C1 — PNR is a viable architecture class for scalable language
  modelling (class-level existence + parity).* Two PNR instances (NDM
  and M²RNN-CMA) train stably at 1.27 B parameters under matched
  CMA-ES hyperparameter optimisation. PNR matches frontier-class
  linear-recurrent baselines (Mamba2 and Gated DeltaNet (GDN)) on the
  shared wall-clock loss-vs-compute curve over the entire snapshot
  window. Trainability at LLM scale is enabled by
  *multi-programming* — running thousands of small bounded recurrent
  programs in parallel across heads, state tiles and batch elements —
  which is an *update-rule-agnostic* recipe. The structural anchor is
  the Lean predicate
  `RecurrentResourceFormalism.IsMultiProgrammed` on architecture
  signatures, with witness
  `RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_ndm`
  showing both the 1.27 B NDM signature and the CMA-reshaped pure
  M²RNN signature satisfy it. Gradient conditioning — the
  $q,k$-to-value-head ratio — is a third recipe property distinct from
  the update rule (§5, §11).

+ *C2 — Within PNR, delta-correct $>$ raw-write at matched CMA-ES
  protocol and matched per-token FLOP class (instance-level
  mechanism).* The two PNR instances share matrix state, temporal
  nonlinearity, and the same multi-programmed parallelism class; they
  differ only in the per-step update rule. The within-PNR separation
  therefore isolates the update rule as the differentiator. The two
  rules — delta-correcting $S <- tanh(d S + k (v - S^T k)^T)$ (NDM) and
  raw-write $Z = tanh(H W + k v^T)$ (M²RNN-CMA) — first sit inside the
  same per-token FLOP class
  (`RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv`), so "more
  expressive" cannot collapse into "spends more compute". The claim
  then rests on three lines of evidence which must be read with care,
  because they speak to two different properties. (a) A *one-step
  representational* separation, formalised in Lean: NDM realises the
  $S_5$ prefix tracker
  (`NDMRealizesS5.ndm_realizes_s5_tracker`), while
  `RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`
  proves that no fixed-weight raw-write matrix RNN with row, column, or
  cell forget gates can match NDM's mixed-key delta correction in one
  recurrent step. These results bound what each update rule *can
  represent* at fixed precision and width; they are not impossibility
  results for the full M²RNN family under SGD. (b) An *empirical
  trainability* gap: at 8 M parameter-matched scale, NDM reaches mean
  accuracy $bold(0.79)$ on $S_5$ against $0.36$ for GDN, $0.22$ for
  the CMA-tuned M²RNN raw-write baseline, and $0.17$ for the M²RNN
  paper-default shape. (c) A *capacity-non-binding* control on $S_3$:
  M²RNN-CMA stalls at $0.31$ on the six-element solvable group at the
  same 8 M scale, where representing $S_3$ requires $log_2 6 approx 2.6$
  bits per state against $approx 10^6$ recurrent-state scalars per
  token at the probe shape (independently, $approx 1.3 times 10^8$
  parameter bits at fp16) — so the failure is *trainability under the
  raw-write inductive bias*, not a capacity ceiling. The Lean theorems
  anchor (a); the probes deliver (b) and (c). At LLM scale, the same
  ordering is corroborated by the matched-CMA-ES wall-clock racer of
  §5: under the 1.27 B protocol M²RNN-CMA trails both NDM and Mamba2
  across the sampled window.

+ *C3 — A comparison relation on PNR update rules under matched
  per-token FLOP class (open program).* The within-PNR ordering
  observed here (delta-correct $>$ raw-write) is the first strict
  instance of an expressivity order on the PNR class. C2 implies a
  *comparison relation* — of which one strict instance is established
  here — over update rules indexed by *one-step expressive power at
  matched asymptotic per-token FLOP class*. The matched-cost condition
  is over the asymptotic equivalence class $c_1 d^2 + c_2 d$ proved in
  `RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv`, not over
  exact FLOP counts: the empirical $approx 1.55 times$ constant by
  which NDM exceeds the leading linear pair at the tight threshold
  (@tab_thresholds) lies within this common class and does not
  contradict it. Whether this relation extends to a *partial order*
  with an NC#super[1] ceiling (Barrington) on the broader PNR space —
  in particular, whether antisymmetry and transitivity hold over the
  full space — is the obvious next-paper question. Empirically, the
  NDM-vs-baseline accuracy gap on the canonical-sweep
  length-extrapolation curves (parity, modular counter, FSM tracking;
  §7) widens monotonically with sequence length, consistent with —
  though not formal proof of — multi-step persistence of the one-step
  gap. The rigorous research target this paper stakes is *the maximal
  element under matched asymptotic FLOP class* — the PNR update rule
  with maximal one-step expressive power for its compute class. This is
  distinct from the *aspiration* "the best PNR update rule", which
  involves further properties (multi-step capability, gradient
  conditioning, downstream performance) and is not claimed here. The
  Lean resource-separation formalism in `RecurrentResourceFormalism` is
  the tool for climbing this relation; the present paper deposits one
  comparison and leaves the maximal element open.

A concurrent pure-recurrent nonlinear matrix RNN, *M²RNN*
(`arXiv:2603.14360`, March 2026 @m2rnn2026), trains a homogeneous
recurrent variant at 410 M dense and 7 B MoE on Nemotron-CC-v2 with the
*raw-write* update $Z = tanh(H_(t-1) W + k v^T)$. M²RNN is therefore
itself a PNR instance, and the M²RNN-CMA baseline carried throughout
this paper is its CMA-reshaped form trained under the same matched
protocol as NDM. Mishra et al. additionally report favourable hybrid
configurations against Mamba-2 and Gated DeltaNet at matched parameter
and token budgets under a uniform fixed-hyperparameter protocol (their
§5.2: shared optimiser, learning rate, weight decay and gradient
clipping across all compared architectures, varying only the
sequence-mixing block). The comparative ranking is established at those
matched budgets but without per-architecture hyperparameter search,
reported seed variance, or hybrid-layer position search, so the
hybrid-superiority margin is read here as preliminary. M²RNN is the
closest peer demonstration of a PNR instance at scale, and the
delta-correcting update studied here is compared against the raw-write
update of which M²RNN-CMA is the canonical instance. The contribution
staked here is *explanatory*: identifying which ingredient inside the
PNR class earns the expressivity separation. *xLSTM-1.3B* @xlstm2024
is a 7:1 mixture of mLSTM (linear) and sLSTM (nonlinear) blocks at the
same scale band, and is included as a peer with the caveat that 87.5%
of its blocks are linear and the model is therefore not PNR by the
"no linearisation, no hybrid bolt-ons" criterion.

// ── 2. Background ─────────────────────────────────────────────────────────────
= Background <sec:background>

#heading(level: 2, numbering: none)[Linear-state and nonlinear-state recurrence]

We use a single explicit criterion to classify recurrent architectures.
A recurrent layer is *linear-state* if its update can be written
$h_t = A_t h_(t-1) + b_t$
with $A_t$ and $b_t$ depending on the current input $x_t$ only.
Otherwise — if the previous state $h_(t-1)$ appears nonlinearly in
its own update equation — the layer is *nonlinear-state*. The
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
recurrence at fixed precision and width is a regular-language recogniser
that lives inside TC#super[0] and therefore cannot solve
non-solvable-group word problems @merrill2024transformers @barrington1986.
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

Two nonlinear matrix-state designs — NDM and M²RNN — therefore share the
necessary preconditions (matrix state, nonlinearity on the state, no
attention, no linearisation, no hybrid bolt-ons) that define the PNR
class introduced in §1 and differ in one place: the per-step update
rule on the matrix state. NDM uses a *delta-correcting* update;
M²RNN uses a *raw-write* update; both are PNR instances. The
contribution staked in C2 (§1) is that, *at matched per-token FLOP
class and within the PNR class*, the delta-correcting write is the
ingredient identified here that makes $S_5$/$S_3$-style prefix-tracking
*learnable in practice*. The matched-cost condition is what makes the
order over update rules meaningful — without it, "more expressive"
collapses into "spends more compute". The Lean anchor for the matched
cost is `RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv`, which
shows the per-token FLOP count of an NDM head and an M²RNN head sit
inside a common $c_1 d^2 + c_2 d$ envelope (§8 Theorem set D). Without
that anchor, the partial order on PNR update rules (C3) loses meaning.

#heading(level: 2, numbering: none)[Matrix state]

Replacing a vector hidden state $h in RR^d$ with a matrix state
$S in RR^(N times V)$ is not novel to NDM. Linear-state designs
(mLSTM, RWKV-5/6, DeltaNet) already use matrix or expanded states;
RetNet's accumulation $S_t = gamma S_(t-1) + k_t v_t^T$ is matrix-valued.
The point of matrix state is that an outer-product update
$S_t = ... + k_t delta_t^T$ provides $O(N V)$ scalars of dynamic state
at $O(N V)$ computational cost per token, with content-addressable
retrieval via $S^T q$. Matrix state is common to the post-Mamba
landscape (Mamba2 with `d_state`, GLA, DeltaNet, RWKV-5/6/7, mLSTM,
M²RNN) and is treated here as a precondition shared by NDM and its
baselines, not as a contribution.

#heading(level: 2, numbering: none)[The $S_5$ state-tracking probe]

The symmetric group $S_5$ has $120$ elements; it is the smallest
non-solvable group. The associated *word problem* — given a sequence of
adjacent transpositions in $S_5$, compute the prefix product after each
token — is, by Barrington's theorem @barrington1986, complete for the
complexity class NC#super[1]. A recogniser that solves $S_5$ at length
$T$ with bounded precision and width must therefore reach the top of
NC#super[1] in the canonical regular-language witness; one that cannot
solve $S_5$ at training length lives below it. The solvable-group control
$S_3$ (6 elements) is included to factor out the part of difficulty
that comes from prefix tracking *per se* rather than from non-solvability.

// ── 3. Architecture — Nonlinear Delta Memory ──────────────────────────────────
= Architecture <sec:arch>

#heading(level: 2, numbering: none)[Per-head update]

Each NDM layer maintains $H$ independent heads. Each head $h$ owns a
matrix state $S_h in RR^(N times V)$ — at production scale, $N = V = 32$.
Per token, the input gives projections $k_h, q_h in RR^N$,
$v_h in RR^V$, and a scalar input-dependent decay
$d_h in (0,1)$ and gate $g_h in RR^V$. The recurrent step is

$
k_h &<- "silu"(k_h) / norm("silu"(k_h))_2 quad ("L"^2 "-normalised key")\
q_h &<- "silu"(q_h) / norm("silu"(q_h))_2 quad ("L"^2 "-normalised query")\
r_h &= S_h^T k_h \
delta_h &= "silu"(v_h) - r_h \
S_h &<- tanh(d_h dot S_h + k_h delta_h^T) \
y_h &= "silu"(g_h) dot.o S_h^T q_h
$

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
              rect(inset: 5pt, fill: rgb("#ffe5cc"), width: 100%, align(center, text(size: 9pt)[$delta_h = v_h - r_h$  #sym.space  #text(size: 8pt)[(prediction error)]])),
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
              text(size: 8pt, style: "italic")[(Lean: `ndm_1p27B_programs_per_batch_token_bs5`)],
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
    *NDM architecture and multi-programmed shape.*
    *(A)* Per-head recurrent dataflow. Each head owns a small bounded
    associative matrix memory $S_h$. The model reads at address $k_h$,
    computes a prediction error $delta_h$, writes the bounded delta
    correction into $S_h$, and gates the read at $q_h$ for output. The
    $tanh$ on the state, the delta-correcting write, and the
    $L^2$-normalisation of $q,k$ are the three load-bearing design
    choices (§3).
    *(B)* The production stack at 1.27 B parameters exposes 370 small
    independent heads per layer per batch element. Each head is a
    $32 times 32$ state tile that fits in registers/SRAM and runs its
    time loop serially; parallelism is harvested across heads, depth and
    batch (§4). The Lean theorem
    `ndm_1p27B_programs_per_batch_token_bs5` certifies 22,200 programs
    per token at batch size 5.
  ],
) <fig_arch>

Three ingredients are load-bearing.

#set enum(numbering: "(a)")

+ *Bounded matrix state via $tanh$ on $S$.* Linear-in-state recurrences
  cannot pass the Siegelmann–Sontag boundary even in principle; placing
  the non-linearity on the state itself, not only on a gate or the
  output, is what makes NDM nonlinear-state.

+ *Delta-correcting write $v - S^T k$.* The model reads what memory
  predicts at address $k$, computes the prediction error $delta$,
  and writes the correction. With an orthonormal key family this gives
  exact overwrite at one slot while preserving the others; with
  arbitrary keys it gives bounded error-correcting binding. A raw-write
  update $S <- S + k v^T$ (the M²RNN family) accumulates without
  correction and cannot, with fixed weights, satisfy a uniform one-step
  overwrite specification (§Formal Results).

+ *Many small heads, not one large matrix.* Production NDM at 1.27 B
  uses $H = 370$ heads of $32 times 32$ each (Lean-witnessed:
  `RecurrentResourceFormalism.ndm_1p27B_programs_per_batch_token`,
  yielding 22,200 independent programs per token at batch size 5).
  Per-head $L^2$-normalised $q,k$ give many independent addressing
  programs and avoid the gradient-conditioning failure mode seen when a
  single shared $q,k$ pair feeds hundreds of value heads.

#heading(level: 2, numbering: none)[Parameterisation choices]

Decay is parameterised in log-space following Mamba2: per head we learn
$A_(log) in RR$ and a scalar bias $delta_(text("bias"))$, then compute
$d = exp(-exp(A_(log)) dot "softplus"(alpha(x) + delta_(text("bias"))))$,
with `A_log` and `dt_bias` excluded from weight decay. Computation is
done in float32 with cast-back to the storage dtype. The non-linearity on
$S$ uses the numerically stable form $tanh(z) = 2 sigma(2z) - 1$ in the
fused kernel to avoid $exp$ overflow at high pre-activations. There is
no output `RMSNorm` inside the layer (ablation removed it as a 0.10-nat
win); the LM wrapper uses block-level prenorm and a final norm only.
Short convolutions on the input are absent. The write path is *not*
gated; the gate `silu(g)` is applied to the read output only.

#heading(level: 2, numbering: none)[The 1.27 B production stack]

The 1.27 B parameter NDM stack used throughout the paper has dim = 1664,
depth = 12, $H = 370$, $N = V = 32$, with a tied LM-head embedding and
prenorm residual blocks. The reference implementation lives in
`ndm/models/e88_fused.py`; the fused Triton recurrence kernel that this
paper builds on is in `ndm/triton/e88_triton_forward.py` and
`e88_triton_backward.py`. These references are for the reader; the paper
does not reproduce code.

#heading(level: 2, numbering: none)[Ablation by architecture: isolating the write rule]

Three properties are candidates for the load-bearing differentiator in
state-tracking: *matrix state*, *temporal nonlinearity on that state*,
and *delta correction in the write*. The closest update-rule comparator
to NDM in the literature is M²RNN @m2rnn2026, whose nonlinear
matrix-state update is

$
Z_t &= tanh(H_(t-1) W + k_t v_t^T)\
H_t &= f_t H_(t-1) + (1 - f_t) Z_t.
$

M²RNN is *nonlinear-state* by the criterion of §2 — $H_(t-1)$ appears
inside $tanh$ via $H_(t-1) W$ — but the write into $H$ is a raw outer
product $k v^T$ rather than a delta correction. A three-row ablation
across the three candidate properties isolates the write rule by
elimination:

#figure(
  align(center)[#table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header(
      [*Property*], [*GDN*], [*M²RNN*], [*NDM*], [*Verdict*],
    ),
    [Matrix state], [yes], [yes], [yes],
      [Cannot separate — GDN has it and fails $S_5$],
    [Temporal nonlinearity on state], [no], [yes], [yes],
      [Cannot separate — M²RNN stalls at 0.22 on $S_5$],
    [Delta correction in write], [no], [no], [yes],
      [Surviving candidate],
  )],
  caption: [
    *Ablation by elimination on the three candidate differentiators for
    state-tracking.* GDN has matrix state and fails $S_5$ at training
    length (0.36 vs NDM 0.79, §7), so matrix state alone cannot be the
    differentiator. M²RNN has matrix state *and* temporal nonlinearity
    on the state and still stalls at 0.22 on $S_5$, so temporal
    nonlinearity is not the differentiator either. Delta correction is
    the only property left that NDM has and the two baselines do not.
  ],
) <tab_ablation>

M²RNN scores 0.31 on $S_3$, the solvable control where non-solvability
is *not* the obstruction. This rules out a complexity-ceiling
explanation. If raw-write could do clean prefix tracking even on
solvable groups, M²RNN should clear $S_3$ — the smallest non-trivial
permutation group, six elements, no NC#super[1] obstruction at all. It
does not. Nor is the failure a capacity ceiling. Two distinct
non-binding bounds make this point. (i) At the 8 M probe shape
($N times V times H times "depth" = 32 times 32 times 32 times 4
approx 1.3 times 10^6$), the per-token recurrent state already carries
on the order of $10^6$ scalars — six orders of magnitude above the
$log_2 6 approx 2.6$-bit information-theoretic floor for representing
the $S_3$ prefix-tracking table. (ii) Independently, the learned
function is encoded in $approx 8 times 10^6 times 16 approx 1.3 times
10^8$ parameter bits at fp16 — eight orders of magnitude above the same
floor. Either bound suffices to render capacity non-binding; the
recurrent-state bound is the relevant one for the prefix-tracking table
itself. With capacity non-binding on both accountings, the residual must
be inductive bias — the raw-write update does not, under SGD at this
scale, find a configuration that prefix-tracks $S_3$. The
deficit is therefore a *trainability* failure of the raw-write update,
not a representability impossibility (M²RNN's matrix state can in
principle store an $S_3$ table) and not a complexity-class ceiling.
The empirical data lives in §7 (@tab_s5); the one-step *formal*
counterpart — a per-step representational separation, not a global
impossibility result —
is `RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`
(§8).

State capacity is *not* the differentiator. Mamba2 @mamba2_2024 with
its `d_state` expansion, GLA @gla2023, DeltaNet @deltanet2024 and
RWKV-5+ @rwkv7_2025 all already carry matrix-valued or expanded states
of comparable order; the entire post-Mamba landscape has matrix state.
The three-row ablation makes the argument by elimination: capacity
cannot be the differentiator if GDN has it and still fails $S_5$.
The argument turns on the *write rule*, not on state size.

// ── 4. Systems ─────────────────────────────────────────────────────────────
= Multi-Programming and Systems <sec:systems>

#heading(level: 2, numbering: none)[Why time-axis serial recurrence can still saturate a modern accelerator]

A modern GPU exposes thousands of independent streaming multiprocessors,
each able to run a small program in registers and shared memory. The
NDM 1.27 B stack invites 370 such programs per layer per batch element
(370 heads $times$ batch element); with depth 12 and batch size 5 this
is 22,200 small independent recurrent programs per layer per token.
Each program is a $32 times 32$ state tile that fits in
registers. The accelerator does not need parallelism *along time* to
stay busy; parallelism across these programs already saturates it.
This is the structural argument behind *multi-programming*: the
recurrent computation is broken into many small bounded memory programs
that the GPU executes in parallel, while the time loop inside each
program runs serially.

#heading(level: 2, numbering: none)[Fused Triton recurrence kernel]

The forward kernel uses an internal `[T, B, H, *]` layout (time, batch,
head, $...$) and dispatches one program per `(batch, head_block)`. The
`BLOCK_H` parameter is autotuned over $\{1, 2, 4, 8, 16\}$ because at
$H >= 256$ per-program-per-head launch overhead dominates. The state
tile lives in registers and shared memory for the duration of the
$T$-step loop. The following pieces are fused into the same Triton
program: $tanh$/$"silu"$ activations on input projections, $L^2$
normalisation of $q, k$, the recurrent delta write, and the output
gate. Each fusion removes one to two `torch.cuda` launches per layer
call; at depth 14 production the aggregate saving is approximately
50–60 ms per step.

#heading(level: 2, numbering: none)[Sparse-checkpoint backward]

The forward kernel saves $S$ only every $K = 16$ steps. The backward
kernel processes one $K$-step segment at a time: it forward-replays the
$K$ steps to rebuild per-step $S_(t-1)$ from the saved checkpoint, then
walks backward to apply the chain rule. The activation memory used by
the kernel is $T / K + 1$ checkpoint slots instead of $T$ — at $K = 16$
this is approximately a $16 times$ shrink — at the cost of a single
extra forward pass per segment.

#heading(level: 2, numbering: none)[Portable kernels]

The kernel is written once in Triton @triton2019 and dispatched
identically on NVIDIA CUDA and AMD ROCm. This is the practical reason
to choose Triton over hand-tuned HIP: a single source achieves
approximately 70% of the hand-tuned throughput in approximately one
week of porting work, versus three to six weeks to port to HIP from
scratch.

#heading(level: 2, numbering: none)[Distributed training]

The training plan uses schedule-free AdamW @schedulefree2024 per island
with hierarchical local-SGD model averaging in the DiLoCo
@diloco2023 style: each island is one node of 8 GCDs with intra-island
DDP, and inter-island synchronisation averages model weights every
$H = 250$ local steps (an empirically-chosen interval; see §Limitations
for the open question on $H$). Because parallelism is across programs
rather than along time, NDM does not require sequence parallelism to be
competitive at 1.27 B; this is a notable simplification relative to
chunked-scan implementations of linear-state recurrences. A speculative
high-risk path — ParaRNN @pararnn2025 — would parallelise the time loop
itself via Newton's method on a block-bidiagonal Jacobian, but its
convergence on the $tanh(d S + k delta^T)$ map at $32 times 32$ block
size is untested.

// ── 5. Language Modelling Results ────────────────────────────────────────────
= Language-Modelling Results <sec:lm>

#heading(level: 2, numbering: none)[Setup]

We train four pure-recurrent language models at the 1.27 B parameter
band on The Pile @thepile2020 with a 2048-token context window, byte-pair
encoding (p50k_base, 50,257-vocab), schedule-free AdamW
@schedulefree2024, and bf16 mixed precision. The four models and their
CMA-tuned shapes are:

#align(center)[#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, right, left),
  stroke: 0.5pt,
  inset: 6pt,
  table.header(
    [*Model*], [*Params*], [*Batch*], [*Shape*],
  ),
  [NDM (E88)], [1.273 B], [5], [dim=1664, depth=12, H=370, N=32],
  [GDN], [1.352 B], [4], [dim=2688, depth=21, exp=2, H=44],
  [Mamba2], [0.934 B], [4], [dim=2048, depth=32, exp=3, d_state=160],
  [M²RNN-CMA], [1.307 B], [5], [dim=1920, depth=21, H=370, N=16],
)]

The hyperparameter and shape choice for each baseline came from a
matched CMA-ES @cmaes2003 search at ~480 M parameters (see §6); the
1.27 B-band runs reuse the shape ratios identified at the search scale.

#heading(level: 2, numbering: none)[Per-architecture CMA-ES versus uniform fixed-recipe]

All four 1.27 B baselines above carry a *per-family CMA-ES winner shape*
identified at the 480 M search scale (§6) and rescaled to the 1.27 B
band — NDM, GDN, Mamba2 and M²RNN-CMA each receive their own
canonical-best configuration at the chosen search scale rather than a
shared fixed recipe. This contrasts with the protocol of the concurrent
M²RNN paper (Mishra et al. @m2rnn2026 §5.2), in which model width, MLP
width, layer count, optimiser, learning rate, weight decay and gradient
clipping are held *uniform* across all compared architectures and only
the sequence-mixing block is varied — fair-by-uniformity but not
per-architecture best-tuning. Under the matched-CMA-ES protocol the
wallclock training curves separate cleanly: the delta-correcting PNR
instance (NDM) and the two linear-recurrent baselines (Mamba2 and GDN)
track the same loss-vs-wallclock band and trade leadership through
training (@fig_lm_racers), while the raw-write PNR instance
(M²RNN-CMA) trails throughout the sampled window. The §7 mechanism
reading is what isolates the source of this within-PNR separation:
matrix state plus temporal nonlinearity is competitive with
frontier-class linear recurrence under matched per-architecture tuning,
and the *delta-correcting* write is what places NDM at parity with
Mamba2 and GDN, its absence what places M²RNN-CMA behind.

#heading(level: 2, numbering: none)[Gradient conditioning is a third recipe property]

A fifth run — *M²RNN-paper*, the paper-default shape from @m2rnn2026
re-implemented at 1.27 B (dim=3072, depth=10, H=759, N=16) — was
attempted under the same training setup and *diverged* at step 8,400
with gradient norm $approx 4.2 times 10^7$. The CMA-tuned reshape
*M²RNN-CMA* (dim=1920, depth=21, H=370, N=16) of the *same* update
family is stable under the same optimiser and the same data; its loss
curve appears in the racer panel below. The paper shape is the
stability control; it is not in the loss-racer panel below because no
usable curve exists.

The two configurations differ in one structural parameter: the ratio
between the number of $q,k$ projections and the number of value heads
that consume them. Many value heads sharing few $q,k$ pairs concentrate
gradient through a narrow projection, and bursty inputs at high
training step counts accumulate gradient norm in that bottleneck. The
CMA-tuned reshape redistributes the ratio toward more independent
$q,k$ pairs per value head (it is also the shape ratio that NDM uses
at production: $H = 370$ per-head $q,k$ pairs) and the explosion
disappears. The Lean-witnessed structural anchor here is the same
predicate `RecurrentResourceFormalism.IsMultiProgrammed` (§8): the
CMA-reshaped M²RNN signature satisfies it; the paper-default shape's
shared-$q,k$ geometry sits closer to the bottleneck regime that the
predicate's "many independent heads per layer" clause forbids.

This is *a third factor distinct from the update rule*. It is a
geometry/recipe property — a property of how the heads are wired, not
of the algebraic form of the write. It reinforces C1 (§1): the
multi-programmed recipe is update-rule-agnostic *and* geometry-sensitive.
The paper is not selling NDM-the-architecture; it is selling the
recipe — *matrix state plus temporal nonlinearity plus a correct
geometry* — of which NDM is the cleanest instance and CMA-reshaped
M²RNN is the proof the recipe generalises across the nonlinear matrix
RNN family. The expressivity claim in §7 still cleanly separates the
two update rules at parameter-matched 8 M scale where geometry is held
constant; the geometry property is a separate axis along which the
multi-programmed recipe must be respected for either family to train
at 1.27 B.

#heading(level: 2, numbering: none)[Loss-vs-wallclock racer]

#figure(
  image("results/figure_3/figure_3_draft.png", width: 95%),
  caption: [
    *Loss versus wallclock for the four 1.27 B-parameter pure-recurrent
    racers, as of 2026-05-24.* Schedule-free AdamW on The Pile with a
    2048-token context. Curves are 10K-step centred moving averages of
    raw training loss (nats per token). Mamba2 is at 0.934 B parameters;
    NDM at 1.273 B; GDN at 1.352 B; M²RNN-CMA at 1.307 B. Training
    is in progress at the time of writing — this snapshot covers
    approximately 8–15 GPU-days per model. *Panel A:* full curve on
    log-wallclock from h = 1. *Panel B:* tail (h ≥ 40) on linear
    wallclock. NDM, GDN, and Mamba2 share a single loss band through the
    bulk of training, with leadership trading between them at the
    fractional-nat scale; GDN and NDM are nearly co-linear. M²RNN-CMA
    has higher loss than the other three across the sampled window.
    The paper-shape M²RNN baseline (not shown) diverged at step 8,400.
    Color convention used throughout the paper: NDM = blue,
    GDN = orange, Mamba2 = green, M²RNN-CMA = red.
  ],
) <fig_lm_racers>

Across the shared wall-clock window, NDM, GDN, and Mamba2 occupy the
same band: leadership trades between them through training, and at no
sampled point do the three separate by more than a small fraction of a
nat. Final raw losses at the snapshot are approximately 2.66 (NDM,
step 1,035,000), 2.68 (GDN, step 1,371,000), 2.71 (Mamba2,
step 1,862,000), and 2.77 (M²RNN-CMA, step 958,000). The within-PNR
comparison is qualitatively different: M²RNN-CMA trails NDM and Mamba2
across the entire sampled window under the matched-CMA-ES configuration
protocol (each baseline uses its §6 CMA-ES winner shape scaled to the
1.27 B band; no architecture received post-hoc per-family HPO at this
scale). The robust empirical claims supported by @fig_lm_racers are
therefore (i) *(class-level, C1)* NDM is competitive with frontier-class
linear recurrence on this scaling curve, contradicting the long-standing
assumption that pure nonlinear recurrence is impractical at scale; and
(ii) *(instance-level, C2)* within the PNR class the delta-correct
update rule is consistently ahead of the raw-write update rule under
matched CMA-ES protocol, isolating the update rule as the within-PNR
differentiator. Source: smoothed CSVs and snapshot table under
`paper/results/figure_3/` (`AS_OF.md`).

// ── 6. FLOPs-per-bit ──────────────────────────────────────────────────────────
= FLOPs-per-Bit Convergence under CMA-ES <sec:flops>

A wallclock comparison conflates architecture with kernel quality and
hardware utilisation. To strip that confound we ran a *matched
hyperparameter and shape search* under CMA-ES at ~480 M parameters and
recovered the per-family best configuration. CMA-ES @cmaes2003 is a
derivative-free black-box optimiser that maintains a multivariate
Gaussian over the search space and adapts both the mean and the full
covariance from the fitness ranking of sampled candidates. Each family
was searched over the same six-dimensional space (width, depth, head
count, state width, gating, learning rate), with the same population
size (16), the same per-candidate budget (30 minutes of single-GPU
training), and the same fitness rule (mean nats over the last 100
logging steps). The compute budget per family is therefore symmetric;
the *winner* per family is each family's local optimum under matched
search.

#heading(level: 2, numbering: none)[The four families]

The four families compared are NDM, GDN @gated_deltanet2024, Mamba2
@mamba2_2024, and a *vanilla nonlinear Elman recurrence*
$h_t = tanh(W_h h_(t-1) + W_x x_t)$ included as a low-baseline witness.
*M²RNN was not available in the upstream CMA-ES sweep at the 480 M
scale* — the raw-write M²RNN family is therefore absent from
@fig_cma and from @tab_thresholds and no FLOPs-per-bit
slope is computed for it in this panel; the 1.27 B-scale comparison
with M²RNN-CMA is delivered separately via the matched-CMAES wallclock
racer of §5 (@fig_lm_racers). We flag the missing 480 M
M²RNN-CMA run as the most informative open follow-up (§Limitations).

#heading(level: 2, numbering: none)[FLOPs and bits-per-token]

We measure cumulative training FLOPs by the Chinchilla-style approximation
$"FLOPs" = 6 N B T s$ where $N$ is the parameter count, $B$ the batch
size, $T$ the chunk length, $s$ the step count. Loss in nats per token is
converted to bits per token by dividing by $ln 2$. The ratio
$"FLOPs"$ / (bits saved versus a uniform-vocabulary baseline of
$log_2 50257 = 15.617$ bits) is "FLOPs spent per bit of compression
delivered" and is the natural slope of the loss-vs-compute curve.
Two families with the same FLOPs-per-bit slope are interchangeable from
a training-economy standpoint, even if their final asymptotic loss
differs.

#figure(
  image("results/cma_flop_rate/convergence.png", width: 95%),
  caption: [
    *FLOPs-per-bit-of-compression convergence under CMA-ES architecture
    search (N = 4 families: NDM, GDN, Mamba2, vanilla Elman).* Each
    curve is one CMA-tuned recurrent architecture replayed at the
    matched ~480 M parameter target. *Panel A:* training loss (nats per
    token, smoothed) versus cumulative training FLOPs (log-$x$). The
    four curves share a slope; the linear pair (GDN, Mamba2) sits
    approximately 0.10 nats (≈ 0.15 bits) below the nonlinear pair
    (NDM, vanilla Elman) through the bulk of training. *Panel B:*
    running FLOPs-per-bit-of-compression-delivered, log–log. Over more
    than two decades of FLOPs the four traces visually collapse onto a
    single line. At the tight threshold (1.80 bits/token) NDM uses
    approximately $1.55 times$ the FLOPs of the leading linear pair to
    reach the same loss; vanilla Elman does not cross 1.80 inside the
    wall-clock budget at all. *Color convention* (shared with
    @fig_lm_racers): NDM = blue, GDN = orange, Mamba2 =
    green, Elman = grey. *M²RNN-CMA is absent from this 480 M CMA
    sweep* and therefore does not appear in Panel A or Panel B; the
    FLOPs-per-bit collapse is asserted only for the four families that
    are present, and the 1.27 B M²RNN-CMA comparison is delivered via
    the matched-CMA-ES wallclock racer of §5 (@fig_lm_racers)
    instead (§Limitations). *N = 4 caveat:* this is four model
    families, not four seeds; differences smaller than the gap between
    "linear" and "nonlinear" pairs should not be over-interpreted.
  ],
) <fig_cma>

The numerical thresholds are reproduced in @tab_thresholds. At the
loose threshold the four FLOPs-per-bit values agree to within ~10%; at
the medium threshold (2.00) the linear pair agrees with itself to within
2% and the nonlinear pair is within 22% of itself and within 90% of the
linear floor. The convergence is in *slope*, not in asymptote — the
four families do reach different final bits-per-token in the wall-clock
budget. What the finding rules out is that the architectural family is
what determines compute economy at matched HPO; the recipe says, instead,
that compute economy is set by the HPO budget and that what the
architecture changes is the asymptote and the expressivity ceiling
(§7,§8).

#figure(
  align(center)[#table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Family*], [*bits$<=$2.50*], [*bits$<=$2.00*], [*bits$<=$1.80*]),
    [NDM (E88)], [$1.01 times 10^15$], [$2.73 times 10^15$], [$7.11 times 10^15$],
    [GDN], [$0.99 times 10^15$], [$1.45 times 10^15$], [$4.60 times 10^15$],
    [Mamba2], [$0.94 times 10^15$], [$1.42 times 10^15$], [$4.65 times 10^15$],
    [Vanilla Elman], [$1.03 times 10^15$], [$2.25 times 10^15$], [not reached],
  )],
  caption: [
    *FLOPs spent per bit of compression delivered* at three matched
    bits-per-token thresholds. CMA-tuned configurations at the matched
    ~480 M parameter target. Source: `paper/results/cma_flop_rate/thresholds.csv`.
    The empirical $approx 1.55 times$ constant separating NDM from the
    leading linear pair at the 1.80-bit threshold sits within the
    common $c_1 d^2 + c_2 d$ per-token FLOP class proved in
    `RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv` (§8,
    Theorem set D); the asymptotic-class claim and the deployment-scale
    constant are at different levels of abstraction and are not in
    tension.
  ],
) <tab_thresholds>

#heading(level: 2, numbering: none)[Caveats]

The "N = 4" here refers to four model families and not to four random
seeds, four CMA-ES generations, four parameter budgets or four data
sources. Each curve carries single-seed noise that is not separately
quantified. The four families also do not densely cover the recurrent
design space: one nonlinear matrix-state delta (NDM), one linear
matrix-state delta (GDN), one linear scalar-state selective SSM
(Mamba2), and one nonlinear vector-state baseline (vanilla Elman). The
most informative missing comparator is *a nonlinear matrix-state design
without delta correction* — M²RNN — which would isolate the delta term
itself. A CMA-tuned 480 M M²RNN run is the open follow-up that would
sharpen this comparison.

The honest reading of @fig_cma is *not* that NDM is the fastest
recurrent family at converting FLOPs to bits — at the tight threshold
the leading linear pair is approximately $1.55 times$ faster. The
reading is that pure nonlinear recurrence is *within a small constant
factor* of the leading linear-recurrent baselines at matched HPO, which
makes the architectural-option framing of the paper viable: there is no
disqualifying FLOPs penalty for choosing nonlinear matrix state.

// ── 7. Expressivity Results ───────────────────────────────────────────────────
= Expressivity Results <sec:expressivity>

The FLOPs-per-bit finding of §6 says that the architectural family does
not strongly change *training economy*. What it does change is the
*ceiling on capability*. We test this on a battery of state-tracking
probes at 8 M parameter-matched scale (dim = 384, depth = 4,
$H = 32, N = 32$, schedule-free AdamW, 10K–20K steps per task, three
seeds). GDN uses dim = 640 to match parameter count.

#heading(level: 2, numbering: none)[Matched no-tuning across architectures at 8 M]

The 8 M probe scale received no probe-specific hyperparameter search and
no seed sweep for any family in the comparison. NDM ran on its E88
lineage default configuration carried down from the production stack;
M²RNN-CMA ran on the analogous default from its own lineage; GDN and
the M²RNN-paper shape ran on their respective published defaults. The
8 M probe is therefore *matched no-tuning across architectures* — each
family is evaluated on the reasonable-defaults configuration it would
arrive at without probe-targeted optimisation — not matched-after-HPO.
This is the appropriate baseline for a mechanism claim: under matched
no-tuning conditions, any accuracy gap reflects the architecture's
inductive bias, not differential HPO investment. The capacity argument
below (a separate non-binding bound on what the recurrent state can in
principle hold) rules out the third confound, leaving the write rule as
the load-bearing differentiator. Reading the gap as evidence of
undertraining on one side would require asymmetric tuning that did not
occur on either side.

A second potential asymmetry deserves explicit treatment.
"Reasonable defaults" are matched in the sense that no architecture
received probe-specific HPO, but they are not matched in *selection
history*: NDM's defaults are the endpoint of a multi-year E63→E88
lineage selected partly on state-tracking behaviour (Appendix, E63
nonlinearity-on-state notes; E70–E75 gradient-stability fixes via
L#super[2]-normalised keys), whereas GDN and M²RNN's published
defaults were selected by their authors on language-modelling loss.
The matched-no-tuning condition therefore controls for differential
probe-specific effort, not for this selection asymmetry. The $S_3$
control isolates the part of the C2 claim that is immune to it:
raw-write's 0.31 on a six-element solvable group, where the
$log_2 6 approx 2.6$-bit table sits eight orders of magnitude below
either of the non-binding capacity ceilings, is a property of the
update rule under SGD at the 8 M probe shape, not a property of NDM's
design history. The selection-asymmetry caveat narrows the empirical
ordering of (a)/(b) but leaves (c) — and with it the mechanism claim
— intact.

#heading(level: 2, numbering: none)[Headline: $S_5$ permutation composition]

The symmetric-group word problem in $S_5$ — track the running product of
adjacent transpositions, output the 120-way class at each prefix — is
the canonical NC#super[1]-complete witness. Random baseline is
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
    [NDM], [*1.0000*], [*0.7918*], [*0.4158*], [*0.2150*],
    [GDN], [0.7185], [0.3552], [0.1843], [0.0974],
    [M²RNN-CMA], [0.3124], [0.2157], [0.1120], [0.0593],
    [M²RNN-paper], [0.3773], [0.1698], [0.0884], [0.0488],
    [random], [0.1667], [0.0083], [0.0083], [0.0083],
  )],
  caption: [
    *State-tracking accuracy on the permutation-composition probes.*
    Mean over three seeds. $S_3$ is the solvable-group control; $S_5$
    is the non-solvable NC#super[1] witness. NDM separates from all
    three baselines *at training length*, not only under length
    extrapolation. M²RNN — the head of the raw-write nonlinear matrix
    RNN family — underperforms both at $S_5$ training length and at
    $S_3$, supporting the mechanism claim that nonlinear matrix state
    *alone* is not sufficient. Source numbers in
    `paper/ndmpapernotes.md` lines 153–173.
  ],
) <tab_s5>

#figure(
  kind: image,
  block(width: 100%, [
    #align(center)[
      #stack(dir: ltr, spacing: 1em,
        // Left: S5 bar chart
        block(width: 45%, [
          #align(center)[*$S_5$ accuracy at training length T=128*]
          #let bars_s5 = (
            ("NDM", 0.7918, blue),
            ("GDN", 0.3552, gray),
            ("M²RNN-CMA", 0.2157, red),
            ("M²RNN-paper", 0.1698, orange),
          )
          #let bar_w = 100%
          #for (name, val, color) in bars_s5 [
            #stack(dir: ltr, spacing: 0.5em,
              align(right, box(width: 5em, text(size: 9pt)[#name])),
              box(width: 8em, [
                #rect(width: val * 9em, height: 0.9em, fill: color)
              ]),
              text(size: 9pt)[#calc.round(val * 100, digits: 1)\%]
            )
            #v(0.2em)
          ]
          #v(0.4em)
          #text(size: 8pt, style: "italic")[Random = 0.83%]
        ]),
        // Right: S3 bar chart
        block(width: 45%, [
          #align(center)[*$S_3$ accuracy at training length T=128*]
          #let bars_s3 = (
            ("NDM", 1.0000, blue),
            ("GDN", 0.7185, gray),
            ("M²RNN-CMA", 0.3124, red),
            ("M²RNN-paper", 0.3773, orange),
          )
          #for (name, val, color) in bars_s3 [
            #stack(dir: ltr, spacing: 0.5em,
              align(right, box(width: 5em, text(size: 9pt)[#name])),
              box(width: 8em, [
                #rect(width: val * 9em, height: 0.9em, fill: color)
              ]),
              text(size: 9pt)[#calc.round(val * 100, digits: 1)\%]
            )
            #v(0.2em)
          ]
          #v(0.4em)
          #text(size: 8pt, style: "italic")[Random = 16.67%]
        ]),
      )
    ]
  ]),
  caption: [
    *Expressivity separation on the permutation-composition probes at
    parameter-matched 8 M scale.* NDM separates from linear-recurrent
    (GDN) and raw-write nonlinear matrix RNN (M²RNN) baselines on
    the non-solvable $S_5$ probe at training length. On the solvable
    $S_3$ control, NDM is perfect (1.0000); GDN reaches 0.72; both
    M²RNN variants stall in the 0.31–0.38 band, indicating that the
    raw-write update fails on the prefix-tracking task even without the
    non-solvability obstruction.
  ],
) <fig_s5_bars>

NDM is the only family that crosses 0.5 accuracy on $S_5$ at training
length. It is also the only family that solves $S_3$ to ceiling — both
M²RNN variants stall in the 0.31–0.38 band, indicating that the
raw-write update fails on the prefix-tracking task even when the group
is solvable. The gap between NDM and the next-best baseline shrinks
under length extrapolation but does not close: at $T = 512$ NDM is at
0.215 and GDN at 0.097.

The $S_3$/$S_5$ split is also where the scope of the M²RNN paper's own
state-tracking evaluation matters. Mishra et al. @m2rnn2026 §3.2 report
length generalisation on $S_3$ alone — the smallest non-trivial
*solvable* group, which lives inside TC#super[0] — and do not evaluate
$S_5$ or any other non-solvable group. The unhedged "perfect
state-tracking generalisation" framing in that paper therefore does not
bear on the NC#super[1] regime: NDM's $0.79$ at parameter-matched 8 M
scale against the paper-default M²RNN's $0.17$ on $S_5$ at training
length is direct evidence that length generalisation on $S_3$ does not
extend across the TC#super[0]/NC#super[1] boundary under the raw-write
update.

#heading(level: 2, numbering: none)[The six-task canonical sweep]

To verify that the $S_5$ result is not a single-task artefact we run a
six-task canonical sweep covering parity (binary XOR over a stream),
modular counter (K=5), FSM tracking (K=4 states),
Dyck-1 (balanced brackets), associative recall (key→value lookup), and
selective copy (mark-and-copy). At 8 M parameter-matched scale, NDM ties
or wins GDN on five of six tasks (parity 1.00 vs 0.86; modular
counter 0.90 vs 0.65; FSM tracking 1.00 vs 0.83; Dyck-1 1.00 vs 1.00;
selective copy 1.00 vs 1.00). GDN edges NDM on associative recall
(0.997 vs 0.881) — the only attention-natural task in the suite.

Under length extrapolation (train $T = 40$, evaluate up to $T = 500$),
NDM retains 0.89 accuracy on parity at $T = 500$ where GDN collapses
to 0.55 (near random 0.50). On FSM tracking at $T = 500$, NDM is at
0.59 versus GDN 0.39. The gap widens monotonically with length.

This monotonic-widening pattern is the empirical shadow of *multi-step
persistence*. C3 stakes a one-step expressive-power ordering and
explicitly labels multi-step composition as open (Theorem D bounds one
recurrent step; whether the gap survives an unbounded trajectory is not
proved). A one-step ordering that washed out over a trajectory would be
a weaker lattice than one that compounds. The canonical-sweep
length-extrapolation curves give the empirical signature one would
expect if the one-step gap compounds rather than washes out: on
parity, FSM tracking, and modular counter, the NDM-vs-baseline accuracy
gap grows with sequence length rather than closing. We label this
*empirical evidence for multi-step persistence at training-relevant
trajectory lengths*; it is distinct from a formal multi-step
separation, which the trusted Lean core does not provide
(Limitations §10).

#heading(level: 2, numbering: none)[Hybrid degradation: purity matters]

A natural question for any architecture that wins on state tracking is
whether the gain survives mixing with linear-scan blocks. We test the
pattern $[upright("NDM"), upright("NDM"), upright("GDN"), upright("GDN")]$
(four-layer "AABB" hybrid) on the same canonical sweep and find that
*hybridisation degrades state tracking below either pure family*:

#figure(
  kind: image,
  block(width: 100%, [
    #align(center)[
      #stack(dir: ttb, spacing: 0.4em,
        // Modular counter
        [
          #set text(size: 9pt)
          #align(center)[*Modular counter (K=5)*]
          #stack(dir: ltr, spacing: 0.5em,
            align(right, box(width: 6em)[pure NDM]),
            box(width: 9em)[#rect(width: 0.903 * 9em, height: 0.8em, fill: blue)],
            [0.903]
          )
          #v(0.2em)
          #stack(dir: ltr, spacing: 0.5em,
            align(right, box(width: 6em)[pure GDN]),
            box(width: 9em)[#rect(width: 0.648 * 9em, height: 0.8em, fill: gray)],
            [0.648]
          )
          #v(0.2em)
          #stack(dir: ltr, spacing: 0.5em,
            align(right, box(width: 6em)[NDM+GDN hybrid]),
            box(width: 9em)[#rect(width: 0.536 * 9em, height: 0.8em, fill: red)],
            [0.536]
          )
        ],
        v(0.6em),
        // FSM tracking
        [
          #set text(size: 9pt)
          #align(center)[*FSM tracking (K=4 states)*]
          #stack(dir: ltr, spacing: 0.5em,
            align(right, box(width: 6em)[pure NDM]),
            box(width: 9em)[#rect(width: 1.0 * 9em, height: 0.8em, fill: blue)],
            [1.000]
          )
          #v(0.2em)
          #stack(dir: ltr, spacing: 0.5em,
            align(right, box(width: 6em)[pure GDN]),
            box(width: 9em)[#rect(width: 0.830 * 9em, height: 0.8em, fill: gray)],
            [0.830]
          )
          #v(0.2em)
          #stack(dir: ltr, spacing: 0.5em,
            align(right, box(width: 6em)[NDM+GDN hybrid]),
            box(width: 9em)[#rect(width: 0.713 * 9em, height: 0.8em, fill: red)],
            [0.713]
          )
        ],
      )
    ]
  ]),
  caption: [
    *Hybrid degradation.* Interleaving NDM layers with linear-scan
    (Gated DeltaNet) layers in an `[NDM, NDM, GDN, GDN]` pattern
    *underperforms* pure NDM on both modular counter and FSM tracking
    — and underperforms pure GDN on modular counter. State-tracking
    capability is not a property the NDM block can lend to a stack
    of mixed blocks; purity is part of the recipe.
  ],
) <fig_hybrid>

The mechanism interpretation. M²RNN underperforms both at the $S_5$
training length (0.22 vs NDM 0.79) and on $S_3$ (0.31 vs NDM 1.00) at
the same parameter count. Matrix state plus temporal nonlinearity *alone*
— what NDM and M²RNN share — is not sufficient; the delta correction
$v - S^T k$ is the load-bearing piece. The $S_3$ probe is the cleaner
control for reading this as a *trainability* claim. $S_3$ has six
elements; storing its transition table requires $log_2 6 approx 2.6$
bits of state. At the probe shape the per-token recurrent state has
$N V H "depth" approx 10^6$ scalars (and independently the learned
function is encoded in $approx 1.3 times 10^8$ parameter bits at fp16),
so capacity is non-binding by many orders of magnitude on either
accounting, and M²RNN's matrix state can in principle hold an $S_3$
prefix-tracker. M²RNN's 0.31 is therefore evidence that SGD under the
raw-write inductive bias *does not find* such a configuration at this
scale, not that one fails to exist; the failure is empirical
learnability under the raw-write rule, not representational
impossibility. The hybrid degradation result strengthens the same
conclusion from the other side: linear-scan blocks cannot inherit
state-tracking capability from neighbouring NDM blocks. The Lean
formalisation of §8 provides the *representational* counterpart — a
per-step separation at fixed precision and width — not a global
trainability claim about the M²RNN family. Reconciling §3 with §8: the
Lean result
(`RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`)
bounds a *one-step specification* — the precise mixed-key delta
overwrite that NDM performs at each step, which no fixed-weight
raw-write update can reproduce — while the §3 $S_3$ argument concerns
*eventual representability across an unbounded number of steps*, for
which raw-write has the capacity in principle but for which SGD under
the raw-write inductive bias does not, at the 8 M probe scale, locate a
configuration that prefix-tracks. These are distinct claims about
different timescales of expressivity, both indicting the write rule.

#heading(level: 2, numbering: none)[QA and reasoning panel at 1.27 B]

For capability beyond loss numbers we evaluate the four 1.27 B-band
models on a 300-item multi-choice continuation harness sampled from
ARC-C/E @arc2018, HellaSwag @hellaswag2019, SciQ @sciq2017, OpenBookQA
@openbookqa2018, and BoolQ @boolq2019. At the latest snapshot, NDM
reaches 0.367 (random ~0.29), GDN 0.380, M²RNN-CMA 0.367,
Mamba2 0.360 — all four are within one standard error of one another
($"SE" approx 6$ pp at 50 items per task). A separate reasoning panel
(BIG-Bench Hard @bbh2022, ReCLor @reclor2020, FOLIO @folio2022) shows
that all four families collapse on multi-step object tracking
(`tracking_shuffled_objects_7_objects` at 0.10–0.13, near-random) and
on FOLIO/ReCLor (near-random for all four), with GDN modestly
leading on formal fallacies and web-of-lies. NDM is not systematically
weaker on reasoning — its overall reasoning accuracy (0.319) is within
one standard error of M²RNN (0.336) and Mamba2 (0.324). At 1.27 B and
this training stage, none of the four families crosses out of the
near-random band on the hardest multi-step reasoning tasks; the
QA panel says that pure nonlinear NDM acquires standard-benchmark
capability at the same rate as the linear-recurrent and raw-write
nonlinear baselines.

// ── 8. Formal Results ─────────────────────────────────────────────────────────
= Formal Results <sec:formal>

We have a trusted Lean 4 @lean42021 core built on Mathlib
@mathlib4. The import closure of the `ElmanProofs.PaperCore` module —
nine source files — contains no `sorry`, no `admit`, no `axiom`, no
`opaque`, and no `native_decide`. Each result below is identified by its
exact theorem name so that the reader can locate it in the source.

#heading(level: 2, numbering: none)[Theorem set A — finite-state ceiling and $S_5$ tracker]

#set list(indent: 1em)
- *Finite-state ceiling at fixed precision.*
  `S5Witness.fixed_precision_state_space_finite` shows that every
  fixed-precision online recogniser has a finite state space. This bounds
  the NDM family — at fixed width and precision — to regular-language
  recognisers, and therefore strictly inside NC#super[1].

- *$S_5$ word problem.* `S5Witness.s5_state_count` proves $|S_5| = 120$;
  `S5Witness.s5_not_solvable` proves $S_5$ is non-solvable;
  `S5Tracker.recognizer_state_count` shows the prefix-product tracker is
  a 120-state recogniser; `S5Tracker.run_append` proves that word
  execution composes permutations. The bridge
  `S5Tracker.pythonRun_eq_tracker_tuple` shows the Lean tracker agrees
  on every input with the Python evaluation harness.

- *Lookup-table realisation.* `S5NDMRealization.s5_transition_key_count`
  shows the $S_5$ tracker uses exactly $120 times 4 = 480$ state/input
  keys; `S5NDMRealization.exactTransitionMemory_run` shows that any
  finite recogniser admits an exact lookup-table realisation.

#heading(level: 2, numbering: none)[Theorem set B — NDM realises $S_5$]

The bridge from the abstract lookup-table realisation to the NDM update
equation is closed by the new result
`NDMRealizesS5.ndm_realizes_s5_tracker`: there exist an integer $d$, an
orthonormal family of keys ${k_g}$ indexed by the adjacent-transposition
generators, a value family ${v_g}$ and a decay scalar $lambda = 1$
such that the NDM update
$
  S_t = tanh(lambda dot S_(t-1) + k_(g_t) (v_(g_t) - S_(t-1)^T k_(g_t))^T)
$
produces a state trajectory that, decoded through a fixed linear
readout, reconstructs the $S_5$ transition table on every input word.
The proof uses
`OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` for the
orthonormal-key write step and `S5Tracker.run_append` for compositional
correctness.

#heading(level: 2, numbering: none)[Theorem set C — update-family separation]

NDM's delta-correcting write and M²RNN's raw outer-product write are
provably distinct as update families.

- *Mechanism separation.*
  `OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` proves
  that the delta write exactly overwrites the addressed slot while
  preserving all orthogonal queries.
  `OnlineMemory.rawOuterWrite_not_uniformOneStepOverwrite` proves that
  the raw outer-product write cannot satisfy the uniform one-step
  overwrite specification.

- *One-step resource separation.* The main embedded statement
  `RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`
  proves that for every $K >= 2$, $V >= 1$, no fixed-weight M²RNN
  parameterisation with row, column or cell forget gates can match NDM's
  mixed-key delta correction in one recurrent step. The result is sharp:
  it covers every external-forget shape that respects the M²RNN
  signature.

- *Positive embedding.* For completeness,
  `M2RNNComparison.m2rnn_read_then_delta_embeds_e88_delta_update` shows
  that *if* M²RNN is given the extra read-then-delta resource, it can
  embed one NDM step. The separation in the previous bullet says that
  without that extra resource M²RNN cannot.

#heading(level: 2, numbering: none)[Theorem set D — per-token FLOP class]

`RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv` proves that the
per-token floating-point operation count for one NDM head and one M²RNN
head is bounded by a common $c_1 d^2 + c_2 d$ form, with explicit
constants $c_1, c_2$. Equal-token-budget comparisons at matched $d, H,
"depth"$ are therefore within a constant factor. This is the only formal
anchor that the Lean core gives to the FLOPs-per-bit convergence finding
of §6; the empirical convergence is *not* a budget artefact.

#heading(level: 2, numbering: none)[Parameter-efficiency corollary (informal; follows from sets C and D)]

The per-step representational separation
(`RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`,
set C) together with the matched per-token FLOP class equivalence
(`RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv`, set D) carries
an informal *parameter-efficiency corollary*: any raw-write matrix RNN
that one-step-realises NDM's mixed-key delta overwrite at the matched
signature must allocate more state capacity — or, equivalently at fixed
state shape, more fixed weights — than NDM, because no fixed-weight
raw-write parameterisation at the matched signature with row, column or
cell forget gates can produce the same one-step result (set C,
sharpness clause). This corollary is *not* a standalone theorem in the
trusted Lean core; it *follows from* the two named theorems above
combined with the universal-overwrite specification
`OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` and the
negative counterpart `OnlineMemory.rawOuterWrite_not_uniformOneStepOverwrite`.

#heading(level: 2, numbering: none)[Theorem set E — multi-programming as a structural predicate]

`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_ndm` defines
a predicate `IsMultiProgrammed` on architecture signatures capturing the
three multi-programming features (many independent heads per layer,
per-head state tile, per-batch independence). The 1.27 B NDM signature
and a CMA-reshaped pure M²RNN signature both satisfy the predicate, and
a non-trivial hybrid signature *fails* it. This is the small formal
anchor for the claim that *multi-programming is not specific to NDM*:
the same structural property is also satisfied by the M²RNN update
family when CMA-reshaped.

#heading(level: 2, numbering: none)[NC#super[1] paragraph (verbatim)]

We adopt the audit-recommended wording from the formalisation gap
analysis:

#block(inset: (x: 1.5em), [
*NDM is, at fixed width and precision, a finite-state recogniser (Lean:
`fixed_precision_state_space_finite`). Within that ceiling, an
orthonormal-key configuration of the NDM update realises the $S_5$
prefix tracker (Lean: `NDMRealizesS5.ndm_realizes_s5_tracker`); $S_5$ is
non-solvable (`s5_not_solvable`) and by Barrington's theorem
@barrington1986 (cited; not formalized in this work) the $S_5$ word
problem is NC#super[1]-complete. NDM therefore reaches the top of
NC#super[1] in the canonical regular-language witness.*
])

#heading(level: 2, numbering: none)[Explicit non-claims]

The trusted core does *not* prove the following, and the paper does not
claim them. (i) A Lean lower bound covering all linear-scan models on
$S_5$. (ii) Barrington's theorem itself; we cite it. (iii) Any "NDM
exceeds NC#super[1]" or "NDM exceeds TC#super[0]" claim — these are
families-wide impossibility statements outside the trusted surface. (iv)
A formal proof that a trained real-valued NDM with empirically learned
weights exactly recovers the lookup table; only the realisability is
proved.

// ── 9. Related Work ───────────────────────────────────────────────────────────
= Related Work <sec:related>

#heading(level: 2, numbering: none)[Linear-state recurrent language models]

The dominant comparison cohort is the family of linear-state
billion-parameter recurrent LMs surveyed in §2: Mamba @mamba2024 and
Mamba2 @mamba2_2024 (selective SSMs); RetNet @retnet2023 (decayed
linear attention); GLA @gla2023 (gated linear attention); DeltaNet
@deltanet2024 (delta rule, linear in $S$); Gated DeltaNet
@gated_deltanet2024 (gated delta rule); RWKV-4/5/6/7 @rwkv4_2023
@rwkv7_2025 (linear-state WKV / generalised delta rule with linear
state); HGRN2 @hgrn2_2024 (gated linear RNN with state expansion);
MinGRU/MinLSTM @mingru_2024 (input-only gates, linear scan);
mLSTM/xLSTM-7B @xlstm7b2025 (covariance update); Griffin/RecurrentGemma
@griffin2024 (RG-LRU, often hybridised with local attention). All of
these models live below NDM on the $S_5$ probe of §7 and inside the
TC#super[0] complexity class @merrill2024transformers — confirmed by
their empirical $S_5$ collapse at length.

#heading(level: 2, numbering: none)[PNR-class peers and adjacent nonlinear-state work]

*M²RNN* @m2rnn2026 trains a pure-recurrent nonlinear matrix-state RNN
at 410 M parameters on Nemotron-CC-v2 with a raw-write update $tanh(H W
+ k v^T)$ — itself a PNR instance by the class definition of §1.
M²RNN is the closest PNR-class peer at the time of this paper: it
demonstrates that a PNR instance reaches useful loss on a Pile-class
corpus at smaller scale, with a different update rule; and the M²RNN
raw-write update is one of the two PNR instances against which the
within-PNR ordering is established here. The empirical $S_5$
separation in §7 and the formal one-step resource separation in §8
quantify the update-rule difference. The CMA-reshaped pure-M²RNN
variant that appears in the 1.27 B language-modelling racer (§5) is a
concrete instantiation of the claim that the multi-programming recipe
is class-general — i.e. update-rule-agnostic within PNR. Mishra et al.
additionally report hybrid M²RNN configurations favourably against
Mamba-2 and Gated DeltaNet hybrids at matched parameter and token
budgets under a uniform fixed-hyperparameter protocol (their §5.2);
those hybrid configurations fall outside PNR (the inserted attention
layers violate the no-hybrid-bolt-ons criterion of §1), and those
numbers are read here as preliminary in light of the single-seed,
no-per-architecture-HPO, no-position-search protocol disclosed.
Downstream inferences in this paper are not chained through them.

*xLSTM-1.3B* @xlstm2024 is a 7:1 mixture of mLSTM (linear) and sLSTM
(nonlinear) blocks; 87.5% of its blocks are linear-state, so xLSTM-1.3B
is not PNR by the no-linearisation criterion of §1. It is the closest
scale band among prior nonlinear-recurrent results, included as a peer
with the caveat that its nonlinear-block share is small. The xLSTM-7B
follow-up @xlstm7b2025 uses *only* mLSTM (linear) blocks and is
correspondingly further from the PNR class.

*Titans* @titans2025 uses an MLP memory with online gradient updates
and is qualitatively nonlinear-state, but it is hybridised with
attention (also failing the PNR no-hybrid-bolt-ons criterion) and has
not been evaluated as a pure recurrent model at Pile-class scale.
*Zeroth-order LSTM scaling* @lstm_zoo_2025 demonstrates 1 B-scale LSTM
training under a non-gradient regime; it is not a standard-gradient
training comparable.

*Classical LSTM/GRU* @lstm1997 is the historical background: no
published classical LSTM/GRU model has reached $>= 500$ M parameters on
a Pile-class corpus. It is the unsuccessful empirical record that
motivated the field's move to linear-state and hybrid alternatives —
the assumption this paper revisits.

// ── 10. Limitations ───────────────────────────────────────────────────────────
= Limitations <sec:limitations>

#heading(level: 2, numbering: none)[Formal scope]

The trusted Lean core proves the realisability of the $S_5$ tracker by
an orthonormal-key NDM configuration, the one-step resource separation
from raw-write matrix RNNs, and the finite-state ceiling. It does *not*
prove: (i) a Lean lower bound covering all linear-scan models on $S_5$;
(ii) Barrington's theorem itself; (iii) any multi-step trajectory
separation between NDM and raw-write matrix RNNs; (iv) any "exceeds
NC#super[1]" or "exceeds TC#super[0]" families-wide impossibility; (v) a
formal guarantee that empirical NDM weights recover the lookup-table
realisation. The empirical $S_5$ accuracy of 0.79 at $T = 128$ is
*evidence* that real training approaches the realisable solution, not a
proof. The canonical-sweep length-extrapolation curves of §7 (parity,
FSM tracking, modular counter; NDM-vs-baseline gap widening
monotonically with sequence length) supply *empirical* evidence
consistent with multi-step persistence of the one-step gap, but do not
constitute a formal multi-step Lean separation. A multi-step formal
counterpart to `ndm_m2rnn_one_step_resource_separation_embeds` is the
obvious next-paper target.

#heading(level: 2, numbering: none)[Length extrapolation is not solved at scale]

NDM separates from baselines at training length and remains ahead under
length extrapolation, but accuracy degrades monotonically with length:
on $S_5$, NDM is at 0.79 at $T = 128$, 0.42 at $T = 256$, 0.22 at
$T = 512$, 0.11 at $T = 1024$. No recurrent family in our sweep solves
$S_5$ at $8 times$ training length.

#heading(level: 2, numbering: none)[Matched-CMA-ES protocol for the wallclock racer]

The 1.27 B wallclock racer (§5, @fig_lm_racers) is run under
*matched CMA-ES*: each baseline uses the CMA-tuned shape and
hyperparameter choice that won the §6 480 M CMA-ES search for its
family, scaled to the 1.27 B band. No architecture received post-hoc
per-family HPO at 1.27 B beyond its CMA-ES winner. The within-PNR
ordering recorded in §5 — M²RNN-CMA trailing NDM and Mamba2 across
the sampled window — is therefore an order *under the matched-CMA-ES
protocol*, not under post-hoc per-architecture tuning. The protocol
does not rule out the counterfactual that further per-family tuning at
1.27 B would close the gap at some budgets; the point of the matched
protocol is *symmetry of effort* — NDM did not receive 1.27 B-band
tuning either, and the order is read as evidence about what the
matched-search recipe selects, not as a families-wide claim that no
M²RNN configuration could ever beat NDM at matched wallclock.

#heading(level: 2, numbering: none)[Geometry-sensitivity of the update-rule claim]

The same update family (M²RNN) looks weak in the published paper shape
(diverges at 1.27 B) and stronger under CMA-tuned reshape (stable; loss
2.77 at the snapshot). The expressivity comparison of §7 holds at
parameter-matched 8 M scale; the language-modelling comparison of §5
relies on the CMA-reshaped M²RNN. The strongest reading of the empirical
claim is therefore *conditional* on the multi-programmed shape, not on
the update equation alone. The Lean separation of §8 is unconditional on
shape but is per-step.

#heading(level: 2, numbering: none)[N = 4 caveat on FLOPs-per-bit]

The CMA-ES FLOPs-per-bit convergence finding (§6) is over four model
families at one parameter budget with one seed per family. M²RNN is
absent from the comparison at 480 M, leaving the cleanest "nonlinear
matrix state without delta" test as the most informative follow-up. The
finding is consistent with the FLOPs-per-bit slope being a function of
HPO budget, not of architectural family, but cannot rule it out with the
present N.

#heading(level: 2, numbering: none)[The opposite architectural bet: hybrids]

A concurrent strand of work places the opposite architectural bet.
*OLMo-Hybrid 7B* @olmohybrid2026 interleaves state-space blocks with
attention, on the premise that hybrid stacks express things that lie
beyond what either pure transformers or pure linear RNNs can do. NDM
is positioned honestly against that bet. A pure-nonlinear NDM stack at
1.27 B matching GDN in the wallclock loss band does not refute
hybrids; what it refutes is the *assumption* that pure nonlinear
recurrence cannot scale at all. The hybrid-degradation finding in §7
($[upright("NDM"), upright("NDM"), upright("GDN"), upright("GDN")]$
underperforms either pure family on modular counter and FSM tracking at
8 M scale) is a *capability-preservation* observation: state-tracking
capability does not survive dilution by linear-scan blocks in our
sweep. It is not an anti-hybrid claim. The two architectural bets
address different questions — "can pure nonlinear recurrence scale at
all?" (this paper) versus "what does a well-mixed hybrid express?"
(OLMo-Hybrid) — and the answers do not contradict.

#heading(level: 2, numbering: none)[Snapshot status of the 1.27 B racer]

The four 1.27 B-band language-model training runs were in progress at the
time of submission. The loss-vs-wallclock racer (@fig_lm_racers)
is a snapshot as of 2026-05-24; the curves will continue and the
final-loss numbers may shift. The qualitative claim — four families in
the same wallclock loss band — is robust to the remaining training; the
quantitative ordering should be re-read at the published-checkpoint
stage.

#heading(level: 2, numbering: none)[Open architectural contradictions]

Several ablation findings remain unresolved. The output gate, the
non-linearity on the state (tanh vs linear), and the decay
parameterisation (simple sigmoid vs Mamba2-style log-space) all show
loss-only ties at small scale; the strongest evidence for each
production choice is the empirical state-tracking and stability data
rather than a clean ablation at 1.27 B. The production architecture
keeps the conservative settings; revalidation of each at 1.27 B is the
highest-value follow-up.

// ── 11. Conclusion ────────────────────────────────────────────────────────────
= Conclusion <sec:conclusion>

This paper introduces *Pure Nonlinear Recurrent Language Models* (PNR)
as an architecture class, and demonstrates that the long-standing
assumption "pure nonlinear recurrence does not scale" was a
parallelisation choice rather than a computational verdict. PNR scales
along the *width* axis (many independent small recurrent heads in
parallel) rather than along the *time* axis: recurrence kept serial
along time still exposes thousands of small bounded memory programs
per layer per batch element, and a modern GPU is more than capable of
running them in parallel.

The class-level evidence (*C1*) is that two PNR instances — NDM
(delta-correcting update) and M²RNN-CMA (raw-write update) — train
stably at 1.27 B parameters under matched CMA-ES, and NDM shares the
wall-clock loss band of frontier-class linear-recurrent baselines
(Mamba2 and Gated DeltaNet). Under matched CMA-ES architecture search
at 480 M, four recurrent families share a single FLOPs-per-bit slope
to within a small constant factor — training compute economy is set by
the HPO budget, not by the architectural family. The instance-level
mechanism evidence (*C2*) is that, within PNR at matched per-token
FLOP class and matched CMA-ES, the delta-correcting update is
consistently ahead of the raw-write update: on the
permutation-composition probes ($S_5$ headline, $S_3$ solvable-group
control) NDM separates from both linear-recurrent and raw-write PNR
baselines, and at the 1.27 B wallclock racer M²RNN-CMA trails NDM
across the sampled window. On $S_3$ in particular, where capacity is
non-binding ($log_2 6 approx 2.6$ bits required against $approx 10^6$
recurrent-state scalars per token at the probe shape, and
independently $approx 1.3 times 10^8$ parameter bits at fp16), the
raw-write 0.31 reads as a *trainability* failure of the update, not a
representability ceiling — the delta-correcting update is the
load-bearing mechanism for making prefix-tracking learnable in
practice. A trusted Lean 4 core (no
`sorry`/`admit`/`axiom`/`opaque`/`native_decide`) supplies the matching
*representational* anchor at fixed precision and width: an
orthonormal-key NDM configuration realises the $S_5$ tracker
(`NDMRealizesS5.ndm_realizes_s5_tracker`), no fixed-weight raw-write
matrix RNN can match NDM's mixed-key delta correction in one recurrent
step (`ndm_m2rnn_one_step_resource_separation_embeds`), the per-token
FLOP class is the same for the two PNR instances
(`ndm_m2rnn_flop_class_equiv`), and both PNR instances satisfy the
multi-programming predicate
(`multiProgrammed_admits_m2rnn_and_ndm`).

The within-PNR ordering established here (delta-correct $>$ raw-write)
is the first strict instance of an expressivity order on the PNR
class. *C3* is the open program opened by it: a *comparison relation*
over PNR update rules at matched *asymptotic* per-token FLOP class
(`ndm_m2rnn_flop_class_equiv`), of which raw-write $<$ delta-correct
is the one strict instance established here. Whether this relation
extends to a partial order with an NC#super[1] ceiling on the broader
PNR space is the obvious next-paper question. The maximal element
under matched asymptotic FLOP class — *which* PNR update rule has the
highest one-step expressive power for its compute class — is the open
horizon, with the `RecurrentResourceFormalism` Lean machinery as the
tool for climbing the order.

// ── Appendix A — Model zoo / ablation lineage ─────────────────────────────────
= Appendix: Model-Zoo Lineage and Ablation Notes <sec:appendix>

The Nonlinear Delta Memory architecture as presented in §3 is the
product of a multi-year ablation lineage. The named milestones referred
to here are internal codenames; we document them only for reproducibility
of the design history.

- *E63 (nonlinear delta design).* Established that linear-in-state
  recurrences cannot pass the Siegelmann–Sontag boundary
  @siegelmann1995 and that placing the non-linearity on the matrix
  state itself (rather than only on gates or the output) is the
  expressivity-determining choice. NDM's $tanh$ on $S$ is inherited
  from this design.

- *E70–E75 (matrix-Elman / delta predecessors).* Established the fused
  Triton kernel pattern (outer product + decay + tanh as one kernel)
  and surfaced the gradient-spike failure mode of unbounded outer
  products. The L#super[2]-normalised key write inherited by NDM is the
  fix.

- *E88 (production NDM).* Stable at 1.27 B parameters under
  schedule-free AdamW. Key parameterisation choices: log-space Mamba2
  decay (with weight-decay exemption); L#super[2]-normalised
  $q, k$; SiLU on input projections before normalisation; numerically
  stable $tanh(z) = 2 sigma(2z) - 1$ in the fused kernel; output
  RMSNorm inside the layer removed (−0.10 nats); short convolutions
  removed (−0.027 nats); no write gate; SiLU output gate.

- *M²RNN-CMA versus M²RNN-paper.* The paper-default M²RNN shape
  (dim = 3072, depth = 10, $H = 759$, $N = 16$) diverged at step 8,400
  under our training setup with gradient norms of $approx 4.2 times
  10^7$. The CMA-tuned reshape (dim = 1920, depth = 21, $H = 370$,
  $N = 16$) is stable; it is the M²RNN-CMA baseline used in §5–§7. The
  paper-shape divergence and the CMA-tuned stability together show
  that the gradient-conditioning failure is a *geometry* property
  (shared $q, k$ across many value heads), not a property of the
  raw-write update family per se.

#bibliography("refs.bib", title: "References", style: "ieee")
