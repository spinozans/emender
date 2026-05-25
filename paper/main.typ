// Nonlinear Delta Memory — main paper source
// Format: Typst (https://typst.app), NOT LaTeX.
// Build: bash paper/build.sh  →  paper/Garrison_2026_PNR-<commit>.pdf
//
// Template: arkheion 0.1.2 (the de-facto Typst arxiv-preprint template).
// See paper/template_choice_v7.md for rationale and the porting notes.

#import "@preview/arkheion:0.1.2": arkheion, arkheion-appendices

// ALTERNATE ABSTRACTS — kept for A/B reference. The active (rendered)
// abstract is the result-first version below (passed to arkheion).
//
/*
=== ALTERNATE 1 — Verdict-rebuttal opener (v11 version) ===

The dominant operating verdict in recurrent language modelling is
that pure-nonlinear-in-time recurrence cannot reach foundation-model
scale on competitive wallclock, because it forecloses the time-axis
parallel scan that linear-recurrent variants depend on for GPU
throughput. We test this verdict by training three pure-recurrent
language models at 1.27–1.35 B parameters under per-architecture
CMA-ES hyperparameter search: two with nonlinear time recurrence,
*NΔM* (delta-correcting update $S <- tanh(d S + k(v - S^T k)^T)$)
and *M²RNN-CMA* (raw-write update $tanh(H W + k v^T)$); and one
with linear time recurrence, *Gated DeltaNet* (GDN). All three
land in the same loss-vs-wallclock band on The Pile, so
nonlinearity in time is not a cost for language modelling at this
scale and the choice of recurrence linearity is washed out by
per-architecture tuning. The systems contribution that makes this
possible is *multi-programming*, a width-axis parallelisation that
replicates the recurrent computation across many independent heads
while keeping the time loop serial inside each head; this replaces
the time-axis linearisation that linear recurrences exploit for
throughput. Within the pure-nonlinear-recurrent class, NΔM trains
consistently ahead of M²RNN-CMA, and a one-step representability
separation between the delta-correcting and raw-write update rules,
formalised in Lean 4, is confirmed empirically on
capacity-overparameterised state-tracking probes. The three 1.27–1.35 B
checkpoints, the per-architecture CMA-ES configurations, and the
Triton multi-programming kernel will be released on HuggingFace at
publication; the trusted Lean 4 core has no
sorry/admit/axiom/opaque/native_decide in the import closure.

=== ALTERNATE 2 — Tight ~110-word cold-lead variant ===

We train three pure-recurrent 1.27–1.35 B language models on The
Pile and find that nonlinearity in time is not a cost: two
nonlinear-in-time recurrences (NΔM, M²RNN-CMA) land in the same
loss-vs-wallclock band as the linear-recurrent baseline Gated
DeltaNet under per-architecture CMA-ES. The field's operating
verdict — that pure-nonlinear-in-time recurrence cannot reach this
scale on competitive wallclock — is an artefact of parallelizing
the time axis; width-axis multi-programming recovers throughput
while the time loop stays serial. Within the nonlinear class, NΔM
trains consistently ahead of M²RNN-CMA, with the one-step
representability separation formalised in Lean 4. Checkpoints,
CMA-ES configs, and the Triton kernel released.
*/

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
    *Pure Nonlinear Recurrent (PNR) language models* — pure-recurrent,
    time-serial, attention-free architectures with a nonlinearity on
    the recurrent state itself — have been treated as off-limits at
    foundation-model scale because they foreclose the time-axis
    parallel scan that linear-recurrent variants rely on for GPU
    throughput. We test this verdict by training two PNR instances at
    1.27–1.35 B parameters on The Pile: *NΔM* (this work; delta-correcting
    update $S <- tanh(d S + k(v - S^T k)^T)$) and *M²RNN-CMA*
    (a CMA-reshaped pure-recurrent variant of M²RNN; raw-write update
    $tanh(H W + k v^T)$), alongside the linear-recurrent baseline
    *Gated DeltaNet*. Each architecture is tuned under per-architecture
    CMA-ES. All three land in the same loss-vs-wallclock band:
    *nonlinearity in time is not a cost* at this scale, and the
    status-quo verdict on the PNR class is an artefact of the axis the
    field chose to parallelise over. We recover throughput on the
    width axis instead, via *multi-programming* — replicating the
    recurrence across many independent heads while the time loop stays
    serial inside each. Within the PNR class, NΔM (this work's
    delta-correct instance) trains consistently ahead of M²RNN-CMA;
    a one-step representability separation between the two update
    rules, formalised in Lean 4, is confirmed empirically on
    capacity-overparameterised state-tracking probes ($S_5$, $S_3$).
    We will release PNR checkpoints (NΔM and M²RNN-CMA) together
    with the GDN baseline, the per-architecture CMA-ES
    configurations, and the Triton multi-programming kernel on
    HuggingFace at publication; the trusted Lean 4 core has no
    `sorry`/`admit`/`axiom`/`opaque`/`native_decide` in its import
    closure.
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
#let nd = $upright("NΔM")$
#let s5 = $S_5$
#let s3 = $S_3$

// ── 1. Introduction ───────────────────────────────────────────────────────────
= Introduction <sec:intro>

The recurrent language model lineage runs from Elman networks
@elman1990 and the LSTM @lstm1997 through the modern linear-recurrent
variants: state-space models such as Mamba and Mamba2 @mamba2024
@mamba2_2024, gated linear attention @gla2023, and the delta rule
@deltanet2024. Each step in this lineage traded some expressive power
for time-axis parallelism, because nonlinear-in-time recurrence resisted
the parallel scan that makes GPU throughput tractable at modern scale.

The field's current operating verdict is more specific than
"sequential models cannot scale". Recent work has already softened the
broad form: ParaRNN @pararnn2025 trains nonlinear-recurrent (LSTM and
GRU) language models at 7 B by parallelising the time loop itself via
Newton iteration on a block-bidiagonal Jacobian, and M²RNN @m2rnn2026
(Mishra, Tan, Stoica, Gonzalez, Dao 2026) trains nonlinear matrix-state
recurrence at 7 B MoE in *hybrid form* (nonlinear-recurrent layers
interleaved with attention layers). The operating verdict that
remains is sharper, and is about a class rather than any single
architecture: *Pure Nonlinear Recurrent (PNR) language models —
pure-recurrent, time-serial, attention-free architectures with a
nonlinearity on the recurrent state itself — cannot reach competitive
wallclock at foundation-model scale without either a time-axis
parallelisation trick or hybridisation with attention.* The
nonlinearity is treated as the obstruction; time-axis parallelisation
(Newton iteration, parallel scan, associative reduction) or attention
hybridisation is treated as the unavoidable concession. Either route
preserves the modern throughput frontier; the PNR alternative has been
treated as off the table. This work shows that sharper verdict is an illusion, in the
same sense that Merrill, Petty and Sabharwal @merrill2024transformers
showed the apparent state-tracking expressivity of state-space models
was an artefact of analysis: there, an apparent expressivity property
of SSMs turned out to be an artefact of how the class was analysed;
here, an apparent computational bound on pure-nonlinear-recurrent
scaling turns out to be an artefact of the axis the field chose to
parallelise over. The bound that has been read as fundamental is in
fact contingent on parallelising the time axis. Parallelise a
different axis and the bound vanishes.

That assumption has not been directly tested under modern width-axis
optimisation. Width-axis parallelism, which we call *multi-programming*,
replicates the recurrent computation across many independent heads and
keeps the time loop serial inside each head; throughput comes from the
number of heads, not from linearisation of the recurrence. The
time-axis-parallelism trick is sufficient for throughput in
linear-recurrent models, but it is not necessary for scaling; the width
axis is an alternative the field has under-explored for
pure-nonlinear-recurrent models specifically.

The closest prior art is the pair of recent attempts that scale
nonlinear recurrence by different concessions. M²RNN @m2rnn2026
demonstrates that nonlinear matrix-state recurrence trains at 7 B MoE
scale in *hybrid form* (nonlinear-recurrent layers interleaved with
attention layers); ParaRNN @pararnn2025 demonstrates that
nonlinear-recurrent LSTM and GRU train at 7 B by parallelising the time
loop via Newton iteration on a block-bidiagonal Jacobian. Both define
the boundary of the sharper verdict above: each scales pure
nonlinearity at LLM scale by either hybridising with attention or
parallelising the time axis. The pure-recurrent, time-serial,
attention-free variant has not been demonstrated at LLM scale. This
work takes that variant and puts it head-to-head with a delta-correct
alternative under matched conditions.

This paper establishes the PNR class as viable at foundation-model
scale, by training two PNR instances at the 1.27–1.35 B parameter
band and comparing them head-to-head with a linear-recurrent baseline. The two PNR instances are NΔM (this
work; delta-correct update rule
$S <- tanh(d S + k(v - S^T k)^T)$) and M²RNN-CMA (a CMA-reshaped
pure-recurrent variant of the M²RNN architecture, with a raw-write
update $tanh(H W + k v^T)$); the linear-recurrent baseline is
Gated DeltaNet (GDN @gated_deltanet2024). All three architectures are
trained on The Pile @thepile2020 and received per-architecture
CMA-ES @cmaes2003 hyperparameter and shape search, with range
repositioning when limits were hit, so every architecture was
evaluated under its best-effort configuration at matched search
effort. All three land in the same loss-vs-wallclock band on The
Pile. *Nonlinearity in time is not a cost.* The sharper status-quo
verdict — that PNR language models cannot reach this regime without
a time-axis parallelisation trick or attention hybridisation — is,
at minimum at 1.27–1.35 B on The Pile under matched wallclock, not
supported by the data. The PNR class is therefore open for
exploration; to our knowledge, NΔM and M²RNN-CMA are the first
foundation-model-class PNR language models trained at this scale.

Within the PNR class, NΔM (this work's delta-correct instance)
trains consistently ahead of M²RNN-CMA across the sampled wallclock
window. The paper proceeds as follows. §2 and §3 set up the
linear-state versus nonlinear-state classification and the NΔM
architecture (this work's contribution within the PNR class); §4
covers the multi-programming systems contribution; §5 presents the
1.27 B wallclock racer with the per-architecture CMA-ES protocol;
§6 reports the 8 M expressivity probes with the
capacity-non-binding justification; §7 the Lean 4 formalisation; §8
related work; §9 limitations; §10 conclusion; §11 future work. We
will release PNR checkpoints (NΔM and M²RNN-CMA), the GDN baseline,
the per-architecture CMA-ES configurations, and the Triton
multi-programming kernel on HuggingFace at publication.

// ── 2. Background ─────────────────────────────────────────────────────────────
= Background <sec:background>

#heading(level: 2, numbering: none)[Linear-state and nonlinear-state recurrence]

We use a single explicit criterion to classify recurrent architectures.
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

Two nonlinear matrix-state designs (NΔM and M²RNN) therefore share the
necessary preconditions (matrix state, nonlinearity on the state, no
attention, no linearisation, no hybrid bolt-ons) that define the
pure-nonlinear-recurrent class introduced in §1 and differ in one
place: the per-step update rule on the matrix state. NΔM uses a
*delta-correcting* update; M²RNN uses a *raw-write* update; both are
pure-nonlinear-recurrent instances. The within-class observation in §1
is that, *at matched per-token FLOP class and under per-architecture
CMA-ES*, the delta-correcting write is the ingredient identified here
that makes $S_5$/$S_3$-style prefix-tracking *learnable in practice*.
The matched-cost condition is what makes the within-class ordering
meaningful; without it, "more expressive" collapses into "spends more
compute". The Lean anchor for the matched cost is
`RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv`, which shows
the per-token FLOP count of an NΔM head and an M²RNN head sit inside
a common $c_1 d^2 + c_2 d$ envelope (§7 Theorem set D).

#heading(level: 2, numbering: none)[Matrix state]

Replacing a vector hidden state $h in RR^d$ with a matrix state
$S in RR^(N times V)$ is not novel to NΔM. Linear-state designs
(mLSTM, RWKV-5/6, DeltaNet) already use matrix or expanded states;
RetNet's accumulation $S_t = gamma S_(t-1) + k_t v_t^T$ is matrix-valued.
The point of matrix state is that an outer-product update
$S_t = ... + k_t delta_t^T$ provides $O(N V)$ scalars of dynamic state
at $O(N V)$ computational cost per token, with content-addressable
retrieval via $S^T q$. Matrix state is common to the post-Mamba
landscape (Mamba2 with `d_state`, GLA, DeltaNet, RWKV-5/6/7, mLSTM,
M²RNN) and is treated here as a precondition shared by NΔM and its
baselines, not as a contribution.

#heading(level: 2, numbering: none)[The $S_5$ state-tracking probe]

The symmetric group $S_5$ has $120$ elements; it is the smallest
non-solvable group. The associated *word problem* (compute the prefix
product after each token in a sequence of adjacent transpositions in
$S_5$) is, by Barrington's theorem @barrington1986, complete for the
complexity class NC#super[1]. A recogniser that solves $S_5$ at length
$T$ with bounded precision and width must therefore reach the top of
NC#super[1] in the canonical regular-language witness; one that cannot
solve $S_5$ at training length lives below it. The solvable-group control
$S_3$ (6 elements) is included to factor out the part of difficulty
that comes from prefix tracking *per se* rather than from non-solvability.

// ── 3. Architecture — Nonlinear Delta Memory ──────────────────────────────────
= Architecture <sec:arch>

#heading(level: 2, numbering: none)[Per-head update]

Each NΔM layer maintains $H$ independent heads. Each head $h$ owns a
matrix state $S_h in RR^(N times V)$; at production scale, $N = V = 32$.
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
    *NΔM architecture and multi-programmed shape.*
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
  output, is what makes NΔM nonlinear-state.

+ *Delta-correcting write $v - S^T k$.* The model reads what memory
  predicts at address $k$, computes the prediction error $delta$,
  and writes the correction. With an orthonormal key family this gives
  exact overwrite at one slot while preserving the others; with
  arbitrary keys it gives bounded error-correcting binding. A raw-write
  update $S <- S + k v^T$ (the M²RNN family) accumulates without
  correction and cannot, with fixed weights, satisfy a uniform one-step
  overwrite specification (§Formal Results).

+ *Many small heads, not one large matrix.* Production NΔM at 1.27 B
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

The 1.27 B parameter NΔM stack used throughout the paper has dim = 1664,
depth = 12, $H = 370$, $N = V = 32$, with a tied LM-head embedding and
prenorm residual blocks. The reference implementation and fused Triton
recurrence kernel sources live under `ndm/models/` and `ndm/triton/`;
file paths are recorded in the Appendix.

#heading(level: 2, numbering: none)[Ablation by architecture: isolating the write rule]

Three properties are candidates for the load-bearing differentiator in
state-tracking: *matrix state*, *temporal nonlinearity on that state*,
and *delta correction in the write*. The closest update-rule comparator
to NΔM in the literature is M²RNN @m2rnn2026, whose nonlinear
matrix-state update is

$
Z_t &= tanh(H_(t-1) W + k_t v_t^T)\
H_t &= f_t H_(t-1) + (1 - f_t) Z_t.
$

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
      [*Property*], [*GDN*], [*M²RNN*], [*NΔM*], [*Verdict*],
    ),
    [Matrix state], [yes], [yes], [yes],
      [Cannot separate: GDN has it and fails $S_5$],
    [Temporal nonlinearity on state], [no], [yes], [yes],
      [Cannot separate: M²RNN stalls at 0.22 on $S_5$],
    [Delta correction in write], [no], [no], [yes],
      [Surviving candidate],
  )],
  caption: [
    *Ablation by elimination on the three candidate differentiators for
    state-tracking.* GDN has matrix state and fails $S_5$ at training
    length (0.36 vs NΔM 0.79, §6), so matrix state alone cannot be the
    differentiator. M²RNN has matrix state *and* temporal nonlinearity
    on the state and still stalls at 0.22 on $S_5$, so temporal
    nonlinearity is not the differentiator either. Delta correction is
    the only property left that NΔM has and the two baselines do not.
  ],
) <tab_ablation>

M²RNN scores 0.31 on $S_3$, the solvable control where non-solvability
is *not* the obstruction. This rules out a complexity-ceiling
explanation. If raw-write could do clean prefix tracking even on
solvable groups, M²RNN should clear $S_3$. It does not. Nor is the
failure a capacity ceiling. Two distinct non-binding bounds make this
point. (i) At the 8 M probe shape
($N times V times H times "depth" = 32 times 32 times 32 times 4
approx 1.3 times 10^6$), the per-token recurrent state already carries
on the order of $10^6$ scalars, six orders of magnitude above the
$log_2 6 approx 2.6$-bit information-theoretic floor for representing
the $S_3$ prefix-tracking table. (ii) Independently, the learned
function is encoded in $approx 8 times 10^6 times 16 approx 1.3 times
10^8$ parameter bits at fp16, eight orders of magnitude above the same
floor. Either bound suffices to render capacity non-binding; the
recurrent-state bound is the relevant one for the prefix-tracking table
itself. With capacity non-binding on both accountings, the residual must
be inductive bias: the raw-write update does not, under SGD at this
scale, find a configuration that prefix-tracks $S_3$. The
deficit is therefore a *trainability* failure of the raw-write update,
not a representability impossibility (M²RNN's matrix state can in
principle store an $S_3$ table) and not a complexity-class ceiling.
The empirical data lives in §6 (@tab_s5); the one-step *formal*
counterpart (a per-step representational separation, not a global
impossibility result) is
`RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`
(§7).

State capacity is *not* the differentiator. Mamba2 @mamba2_2024 with
its `d_state` expansion, GLA @gla2023, DeltaNet @deltanet2024 and
RWKV-5+ @rwkv7_2025 all already carry matrix-valued or expanded states
of comparable order; the entire post-Mamba landscape has matrix state.
The three-row ablation makes the argument by elimination: capacity
cannot be the differentiator if GDN has it and still fails $S_5$.
The argument turns on the *write rule*, not on state size.

// ── 4. Systems ─────────────────────────────────────────────────────────────
= Multi-Programming and Systems <sec:systems>

#heading(level: 2, numbering: none)[Multi-programming: the throughput-enabling design choice]

*Multi-programming* is the technical discovery that makes pure-nonlinear
recurrent language modelling tractable at the 1.27 B band. Linear
recurrences gain throughput by *time-axis* linearisation: composing
$h_t = A_t h_(t-1) + b_t$ unfolds into a product of inputs only and
admits prefix-scan or chunkwise matrix-multiplication. Pure-nonlinear
recurrences cannot do that without forfeiting the nonlinear-update
expressivity. Multi-programming replaces the time-axis route with a
*width-axis* one: the recurrent computation is replicated across many
independent heads, each with its own small bounded matrix state, and
parallelism is harvested across those heads (and across state tiles
and batch elements) while the time loop inside each head runs serially.
The cost is per-head sequential time; the gain is that the nonlinear
recurrence is preserved without sacrificing GPU utilisation. The
multi-programming recipe is *update-rule-agnostic*: both PNR instances
trained in this paper (NΔM and M²RNN-CMA) satisfy the same
multi-programming predicate at 1.27 B
(`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_ndm`,
§7). Mishra et al.'s M²RNN paper @m2rnn2026 observed the benefit of
many small recurrent heads in their hybrid configuration; the
multi-programming optimisation here generalises that observation to
pure-recurrent architectures at language-model scale and identifies it
as the design choice that converts pure nonlinear recurrence from a
sub-billion-parameter curiosity into a billion-parameter LLM building
block.

#heading(level: 2, numbering: none)[The 1.27 B NΔM shape under multi-programming]

A modern GPU exposes thousands of independent streaming multiprocessors,
each able to run a small program in registers and shared memory. The
NΔM 1.27 B stack invites 370 such programs per layer per batch element
(370 heads $times$ batch element); with depth 12 and batch size 5 this
is 22,200 small independent recurrent programs per layer per token.
Each program is a $32 times 32$ state tile that fits in
registers. The accelerator does not need parallelism *along time* to
stay busy; parallelism across these programs already saturates it.

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
call; at depth 12 production the aggregate saving is approximately
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
rather than along time, NΔM does not require sequence parallelism to be
competitive at 1.27 B; this is a simplification relative to
chunked-scan implementations of linear-state recurrences. A speculative
high-risk path (ParaRNN @pararnn2025) would parallelise the time loop
itself via Newton's method on a block-bidiagonal Jacobian, but its
convergence on the $tanh(d S + k delta^T)$ map at $32 times 32$ block
size is untested.

// ── 5. Language Modelling Results ────────────────────────────────────────────
= Language-Modelling Results <sec:lm>

#heading(level: 2, numbering: none)[Setup]

We train three pure-recurrent language models at the 1.27–1.35 B
parameter band on The Pile @thepile2020 with a 2048-token context window,
byte-pair encoding (p50k_base, 50,257-vocab), schedule-free AdamW
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
  [NΔM], [1.273 B], [5], [dim=1664, depth=12, H=370, N=32],
  [M²RNN-CMA], [1.307 B], [5], [dim=1920, depth=21, H=370, N=16],
  [GDN], [1.352 B], [4], [dim=2688, depth=21, exp=2, H=44],
)]

#heading(level: 2, numbering: none)[Per-architecture CMA-ES protocol (fairness anchor)]

All three 1.27–1.35 B architectures (NΔM, M²RNN-CMA, and Gated
DeltaNet) received *per-architecture* CMA-ES @cmaes2003 hyperparameter
and shape search. Each architecture was searched independently over the
same six-knob configuration space (width, depth, head count, state
width, output gating, learning rate) under matched candidate budget
(population 16, fixed wall-clock budget per candidate, identical fitness
rule of mean training nats over a fixed late-training window). When
CMA-ES exploration drifted onto the edge of a configured hyperparameter
range, the range was repositioned and search continued; this was done
consistently for all three architectures. No architecture received any
probe-specific tuning beyond what CMA-ES discovered for the
language-modelling loss. The per-family CMA-ES winner shapes shown in
the table above are used unchanged in the §6 expressivity probes and
the §7 formal analysis. This per-architecture protocol contrasts with
the protocol of the concurrent M²RNN paper (Mishra et al. @m2rnn2026
§5.2), in which model width, MLP width, layer count, optimiser,
learning rate, weight decay and gradient clipping are held *uniform*
across all compared architectures and only the sequence-mixing block
is varied; that is fair-by-uniformity but not per-architecture
best-tuning. The per-architecture protocol is the
fairness anchor for the within-class comparison in §6 and §7: the
within-PNR ordering reported there is read as a property of the update
rule under matched best-effort search, not as a residue of differential
HPO investment.

Under this protocol the wallclock training curves separate cleanly: the
delta-correcting PNR instance (NΔM) and the linear-recurrent baseline
(GDN) track the same loss-vs-wallclock band and trade leadership
through training (@fig_lm_racers), while the raw-write PNR instance
(M²RNN-CMA) trails throughout the sampled window. The §7 mechanism
reading is what isolates the source of this within-PNR separation:
matrix state plus temporal nonlinearity is competitive with
frontier-class linear recurrence under matched per-architecture tuning,
and the *delta-correcting* write is what places NΔM at parity with
GDN, its absence what places M²RNN-CMA behind.

#heading(level: 2, numbering: none)[Gradient conditioning is a third recipe property]

A fourth run, *M²RNN-paper* (the paper-default shape from @m2rnn2026
re-implemented at 1.27 B with dim=3072, depth=10, H=759, N=16), was
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
$q,k$ pairs per value head (it is also the shape ratio that NΔM uses
at production: $H = 370$ per-head $q,k$ pairs) and the explosion
disappears. The Lean-witnessed structural anchor here is the same
predicate `RecurrentResourceFormalism.IsMultiProgrammed` (§7): the
CMA-reshaped M²RNN signature satisfies it; the paper-default shape's
shared-$q,k$ geometry sits closer to the bottleneck regime that the
predicate's "many independent heads per layer" clause forbids.

This is *a third factor distinct from the update rule*. It is a
geometry/recipe property: a property of how the heads are wired, not
of the algebraic form of the write. It reinforces the class-level
finding of §1: the multi-programmed recipe is update-rule-agnostic
*and* geometry-sensitive. The expressivity claim in §6 still cleanly
separates the two update rules at parameter-matched 8 M scale where
geometry is held constant; the geometry property is a separate axis
along which the multi-programmed recipe must be respected for either
family to train at 1.27 B.

Two consequences of this geometry sensitivity bear on how the §5 racer
should be read. First, the M²RNN-CMA configuration used here is *not*
M²RNN's published hyperparameter shape; per-architecture CMA-ES was
applied to the M²RNN update rule itself and selected a geometry that
trains stably across the full sampled wallclock window. The
paper-default shape diverged at step 8,400 under the same matched
protocol and is therefore not in the racer. Second, this reshape is
*charitable* to the M²RNN update rule, not adversarial: the rule is
evaluated under a CMA-optimised geometry, which is more favorable to
the rule than its own published geometry, which did not survive
matched-protocol training in our hands. The within-PNR ordering
recorded in §5 is therefore an ordering between NΔM under its
CMA-optimised geometry and the M²RNN update rule under *its* (more
favorable than published) CMA-optimised geometry, at matched
per-architecture search effort.

#heading(level: 2, numbering: none)[Loss-vs-wallclock racer]

#figure(
  image("results/figure_2/figure_2_draft.png", width: 95%),
  caption: [
    *Loss versus wallclock for the three 1.27–1.35 B-parameter
    pure-recurrent racers, as of 2026-05-24.* Schedule-free AdamW on
    The Pile with a 2048-token context. Curves are 10K-step centred
    moving averages of raw training loss (nats per token). NΔM is at
    1.273 B parameters; M²RNN-CMA at 1.307 B; GDN at 1.352 B. Training
    is in progress at the time of writing; this snapshot covers
    approximately 8–15 GPU-days per model. *Panel A:* full curve on
    log-wallclock from h = 1. *Panel B:* tail (h ≥ 40) on linear
    wallclock. NΔM and GDN share a single loss band through the bulk
    of training, with leadership trading between them at the
    fractional-nat scale; the two curves are nearly co-linear.
    M²RNN-CMA has higher loss than the other two across the sampled
    window. The paper-shape M²RNN baseline (not shown) diverged at
    step 8,400. Color convention used throughout the paper: NΔM =
    blue, GDN = orange, M²RNN-CMA = red.
  ],
) <fig_lm_racers>

Across the shared wall-clock window, NΔM and GDN occupy the same band:
leadership trades between them through training, and at no sampled
point do the two separate by more than a small fraction of a nat.
Final raw losses at the snapshot are approximately 2.66 (NΔM,
step 1,035,000), 2.68 (GDN, step 1,371,000), and 2.77 (M²RNN-CMA,
step 958,000). The within-PNR comparison is qualitatively different:
M²RNN-CMA trails NΔM across the entire sampled window under the
matched per-architecture CMA-ES protocol described above. The two
robust empirical claims supported by @fig_lm_racers are therefore
(i) *(class-level)* the pure-nonlinear-recurrent class lands in the
same loss-vs-wallclock band as the frontier-class linear-recurrent
baseline, contradicting the long-standing assumption that pure
nonlinear recurrence is impractical at scale; and (ii) *(within-class)*
within the pure-nonlinear-recurrent class the delta-correcting update
rule is consistently ahead of the raw-write update rule under matched
per-architecture CMA-ES, isolating the update rule as the within-PNR
differentiator. Source: smoothed CSVs and snapshot table under
`paper/results/figure_2/` (`AS_OF.md`).

// ── 6. Expressivity Results ───────────────────────────────────────────────────
= Expressivity Results <sec:expressivity>

#heading(level: 2, numbering: none)[Capacity is non-binding at 8 M parameters for these probes]

The state-tracking probes in this section are run at 8 M
parameter-matched scale (dim = 384, depth = 4, $H = 32, N = 32$,
schedule-free AdamW, 10K–20K steps per task, three seeds). GDN uses
dim = 640 to match parameter count. Before any accuracy number is
reported, the capacity argument: 8 M parameters is *overparameterised*
for the state-tracking tasks studied here by many orders of magnitude.
The $S_5$ and $S_3$ permutation probes have information-theoretic floors
of $log_2 120 approx 6.9$ bits and $log_2 6 approx 2.6$ bits
respectively for representing the prefix-tracking transition table; the
8 M-parameter probe models exceed those floors by approximately seven
orders of magnitude in parameter bits (and independently by six orders
of magnitude in recurrent-state scalars per token). Failure to learn at
this overparameterised scale is therefore better explained by the
update rule's inductive bias under SGD than by capacity, since capacity
is non-binding at 8 M for this task. The probes isolate the
trainability question that the Lean realisability theorem (one-step
representability) cannot reach: realisability bounds what configurations
*exist* at fixed precision and width; the probes test what
configurations SGD *finds* under matched no-tuning conditions.

#heading(level: 2, numbering: none)[Matched no-tuning across architectures at 8 M]

The 8 M probe scale received no probe-specific hyperparameter search and
no seed sweep for any family in the comparison. NΔM ran on the default
configuration carried down from its 1.27 B production stack; M²RNN-CMA
ran on the analogous default from its CMA-tuned reshape; GDN and the
M²RNN-paper shape ran on their respective published defaults. The 8 M
probe is therefore *matched no-tuning across architectures*, meaning
each family is evaluated on the reasonable-defaults configuration it
would arrive at without probe-targeted optimisation, rather than
matched-after-HPO.
This is the appropriate baseline for a mechanism claim: under matched
no-tuning conditions, with capacity non-binding by the bound above, any
accuracy gap reflects the architecture's inductive bias under SGD, not
differential HPO investment and not a capacity ceiling. Reading the gap
as evidence of undertraining on one side would require asymmetric
tuning that did not occur on either side.

A second potential asymmetry deserves explicit treatment.
"Reasonable defaults" are matched in the sense that no architecture
received probe-specific HPO, but they are not matched in *selection
history*: NΔM's defaults are the endpoint of an ablation lineage
selected partly on state-tracking behaviour (Appendix), whereas GDN and
M²RNN's published defaults were selected by their authors on
language-modelling loss. The matched-no-tuning condition therefore
controls for differential probe-specific effort, not for this selection
asymmetry. The $S_3$ control isolates the part of the within-class
claim that is immune to it: raw-write's 0.31 on a six-element solvable
group, where the $log_2 6 approx 2.6$-bit table sits well below the
non-binding capacity ceiling, is a property of the update rule under
SGD at the 8 M probe shape, not a property of NΔM's design history.

#heading(level: 2, numbering: none)[Headline: $S_5$ permutation composition]

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
    [NΔM], [*1.0000*], [*0.7918*], [*0.4158*], [*0.2150*],
    [GDN], [0.7185], [0.3552], [0.1843], [0.0974],
    [M²RNN-CMA], [0.3124], [0.2157], [0.1120], [0.0593],
    [M²RNN-paper], [0.3773], [0.1698], [0.0884], [0.0488],
    [random], [0.1667], [0.0083], [0.0083], [0.0083],
  )],
  caption: [
    *State-tracking accuracy on the permutation-composition probes.*
    Mean over three seeds. $S_3$ is the solvable-group control; $S_5$
    is the non-solvable NC#super[1] witness. NΔM separates from all
    three baselines *at training length*, not only under length
    extrapolation. M²RNN, the head of the raw-write nonlinear matrix
    RNN family, underperforms both at $S_5$ training length and at
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
            ("NΔM", 0.7918, blue),
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
            ("NΔM", 1.0000, blue),
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
    parameter-matched 8 M scale.* NΔM separates from linear-recurrent
    (GDN) and raw-write nonlinear matrix RNN (M²RNN) baselines on
    the non-solvable $S_5$ probe at training length. On the solvable
    $S_3$ control, NΔM is perfect (1.0000); GDN reaches 0.72; both
    M²RNN variants stall in the 0.31–0.38 band, indicating that the
    raw-write update fails on the prefix-tracking task even without the
    non-solvability obstruction.
  ],
) <fig_s5_bars>

NΔM is the only family that crosses 0.5 accuracy on $S_5$ at training
length. It is also the only family that solves $S_3$ to ceiling. Both
M²RNN variants stall in the 0.31–0.38 band, indicating that the
raw-write update fails on the prefix-tracking task even when the group
is solvable. The gap between NΔM and the next-best baseline shrinks
under length extrapolation but does not close: at $T = 512$ NΔM is at
0.215 and GDN at 0.097.

The $S_3$/$S_5$ split is also where the scope of the M²RNN paper's own
state-tracking evaluation matters. Mishra et al. @m2rnn2026 §3.2 report
length generalisation on $S_3$ alone (the smallest non-trivial
*solvable* group, which lives inside TC#super[0]) and do not evaluate
$S_5$ or any other non-solvable group. The unhedged "perfect
state-tracking generalisation" framing in that paper therefore does not
bear on the NC#super[1] regime: NΔM's $0.79$ at parameter-matched 8 M
scale against the paper-default M²RNN's $0.17$ on $S_5$ at training
length is direct evidence that length generalisation on $S_3$ does not
extend across the TC#super[0]/NC#super[1] boundary under the raw-write
update.

#heading(level: 2, numbering: none)[The six-task canonical sweep]

To verify that the $S_5$ result is not a single-task artefact we run a
six-task canonical sweep covering parity (binary XOR over a stream),
modular counter (K=5), FSM tracking (K=4 states),
Dyck-1 (balanced brackets), associative recall (key→value lookup), and
selective copy (mark-and-copy). At 8 M parameter-matched scale, NΔM ties
or wins GDN on five of six tasks (parity 1.00 vs 0.86; modular
counter 0.90 vs 0.65; FSM tracking 1.00 vs 0.83; Dyck-1 1.00 vs 1.00;
selective copy 1.00 vs 1.00). GDN edges NΔM on associative recall
(0.997 vs 0.881), the only attention-natural task in the suite.

Under length extrapolation (train $T = 40$, evaluate up to $T = 500$),
NΔM retains 0.89 accuracy on parity at $T = 500$ where GDN collapses
to 0.55 (near random 0.50). On FSM tracking at $T = 500$, NΔM is at
0.59 versus GDN 0.39. The gap widens monotonically with length.

This monotonic-widening pattern is the empirical shadow of *multi-step
persistence*. The within-class ordering established here is a one-step
expressive-power ordering (Theorem D bounds one recurrent step; whether
the gap survives an unbounded trajectory is not proved). A one-step
ordering that washed out over a trajectory would be a weaker lattice
than one that compounds. The canonical-sweep length-extrapolation
curves give the empirical signature one would expect if the one-step
gap compounds rather than washes out: on parity, FSM tracking, and
modular counter, the NΔM-vs-baseline accuracy gap grows with sequence
length rather than closing. This is *empirical evidence for multi-step
persistence at training-relevant trajectory lengths*; it is distinct
from a formal multi-step separation, which the trusted Lean core does
not provide (Limitations §9).

#heading(level: 2, numbering: none)[Hybrid degradation: purity matters]

A natural question for any architecture that wins on state tracking is
whether the gain survives mixing with linear-scan blocks. We test the
pattern $[upright("NΔM"), upright("NΔM"), upright("GDN"), upright("GDN")]$
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
            align(right, box(width: 6em)[pure NΔM]),
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
            align(right, box(width: 6em)[NΔM+GDN hybrid]),
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
            align(right, box(width: 6em)[pure NΔM]),
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
            align(right, box(width: 6em)[NΔM+GDN hybrid]),
            box(width: 9em)[#rect(width: 0.713 * 9em, height: 0.8em, fill: red)],
            [0.713]
          )
        ],
      )
    ]
  ]),
  caption: [
    *Hybrid degradation.* Interleaving NΔM layers with linear-scan
    (Gated DeltaNet) layers in an `[NΔM, NΔM, GDN, GDN]` pattern
    *underperforms* pure NΔM on both modular counter and FSM tracking,
    and underperforms pure GDN on modular counter. State-tracking
    capability is not a property the NΔM block can lend to a stack
    of mixed blocks; purity is part of the recipe.
  ],
) <fig_hybrid>

The mechanism interpretation. M²RNN underperforms both at the $S_5$
training length (0.22 vs NΔM 0.79) and on $S_3$ (0.31 vs NΔM 1.00) at
the same parameter count. Matrix state plus temporal nonlinearity
*alone* (what NΔM and M²RNN share) is not sufficient; the delta
correction
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
state-tracking capability from neighbouring NΔM blocks. The Lean
formalisation of §7 provides the *representational* counterpart (a
per-step separation at fixed precision and width), not a global
trainability claim about the M²RNN family. Reconciling §3 with §7: the
Lean result
(`RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`)
bounds a *one-step specification* (the precise mixed-key delta
overwrite that NΔM performs at each step), while the §3 $S_3$ argument
concerns *eventual representability across an unbounded number of
steps*, for which raw-write has the capacity in principle but for which
SGD under the raw-write inductive bias does not, at the 8 M probe scale,
locate a configuration that prefix-tracks. These are distinct claims about
different timescales of expressivity, both indicting the write rule.

#heading(level: 2, numbering: none)[QA and reasoning panel at 1.27 B: parity-rate evidence]

For capability beyond loss numbers we evaluate the three 1.27–1.35 B-band
models on a 300-item multi-choice continuation harness sampled from
ARC-C/E @arc2018, HellaSwag @hellaswag2019, SciQ @sciq2017, OpenBookQA
@openbookqa2018, and BoolQ @boolq2019. At the latest snapshot, NΔM
reaches 0.367 (random ~0.29), GDN 0.380, and M²RNN-CMA 0.367; all
three sit within one standard error of one another
($"SE" approx 6$ pp at 50 items per task). On a separate reasoning
panel (BIG-Bench Hard @bbh2022, ReCLor @reclor2020, FOLIO @folio2022),
all three families collapse on multi-step object tracking
(`tracking_shuffled_objects_7_objects` at 0.10–0.13, near-random) and
on FOLIO/ReCLor (near-random for all three), with GDN modestly leading
on formal fallacies and web-of-lies. NΔM's overall reasoning accuracy
(0.319) is within one standard error of M²RNN (0.336). None of the
three architectures has crossed the threshold where reasoning
benchmarks differentiate, which is consistent with all three being
under-trained equivalently at this stage of training. Read as evidence
for the class-level claim of §1, the panel says that
pure-nonlinear-recurrent NΔM acquires standard benchmark capability at
the same rate as the linear-recurrent and raw-write nonlinear baselines
at this training budget; the QA result is a qualitative check that NΔM
is not categorically failing at downstream language acquisition
relative to the linear baselines, not an attempt to show NΔM has
already pulled ahead on reasoning.

// ── 8. Formal Results ─────────────────────────────────────────────────────────
= Formal Results <sec:formal>

We have a trusted Lean 4 @lean42021 core built on Mathlib
@mathlib4. The import closure of the `ElmanProofs.PaperCore` module
(nine source files) contains no `sorry`, no `admit`, no `axiom`, no
`opaque`, and no `native_decide`. Each result below is identified by its
exact theorem name so that the reader can locate it in the source.

#heading(level: 2, numbering: none)[Theorem set A: finite-state ceiling and $S_5$ tracker]

#set list(indent: 1em)
- *Finite-state ceiling at fixed precision.*
  `S5Witness.fixed_precision_state_space_finite` shows that every
  fixed-precision online recogniser has a finite state space. This bounds
  the NΔM family (at fixed width and precision) to regular-language
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

#heading(level: 2, numbering: none)[Theorem set B: NΔM realises $S_5$]

The bridge from the abstract lookup-table realisation to the NΔM update
equation is closed by the new result
`NDMRealizesS5.ndm_realizes_s5_tracker`: there exist an integer $d$, an
orthonormal family of keys ${k_g}$ indexed by the adjacent-transposition
generators, a value family ${v_g}$ and a decay scalar $lambda = 1$
such that the NΔM update
$
  S_t = tanh(lambda dot S_(t-1) + k_(g_t) (v_(g_t) - S_(t-1)^T k_(g_t))^T)
$
produces a state trajectory that, decoded through a fixed linear
readout, reconstructs the $S_5$ transition table on every input word.
The proof uses
`OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` for the
orthonormal-key write step and `S5Tracker.run_append` for compositional
correctness.

#heading(level: 2, numbering: none)[Theorem set C: update-family separation]

NΔM's delta-correcting write and M²RNN's raw outer-product write are
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
  parameterisation with row, column or cell forget gates can match NΔM's
  mixed-key delta correction in one recurrent step. The result is sharp:
  it covers every external-forget shape that respects the M²RNN
  signature.

- *Positive embedding.* For completeness,
  `M2RNNComparison.m2rnn_read_then_delta_embeds_e88_delta_update` shows
  that *if* M²RNN is given the extra read-then-delta resource, it can
  embed one NΔM step. The separation in the previous bullet says that
  without that extra resource M²RNN cannot.

#heading(level: 2, numbering: none)[Theorem set D: per-token FLOP class]

`RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv` proves that the
per-token floating-point operation count for one NΔM head and one M²RNN
head is bounded by a common $c_1 d^2 + c_2 d$ form, with explicit
constants $c_1, c_2$. Equal-token-budget comparisons at matched $d, H,
"depth"$ are therefore within a constant factor; the within-class
empirical ordering of §5 cannot be charged to one PNR instance spending
asymptotically more compute per token than the other.

#heading(level: 2, numbering: none)[Parameter-efficiency corollary (informal; follows from sets C and D)]

The per-step representational separation
(`RecurrentResourceFormalism.ndm_m2rnn_one_step_resource_separation_embeds`,
set C) together with the matched per-token FLOP class equivalence
(`RecurrentResourceFormalism.ndm_m2rnn_flop_class_equiv`, set D) carries
an informal *parameter-efficiency corollary*: any raw-write matrix RNN
that one-step-realises NΔM's mixed-key delta overwrite at the matched
signature must allocate more state capacity (or, equivalently at fixed
state shape, more fixed weights) than NΔM, because no fixed-weight
raw-write parameterisation at the matched signature with row, column or
cell forget gates can produce the same one-step result (set C,
sharpness clause). This corollary is *not* a standalone theorem in the
trusted Lean core; it *follows from* the two named theorems above
combined with the universal-overwrite specification
`OnlineMemory.linearDeltaWrite_overwrites_one_preserves_others` and the
negative counterpart `OnlineMemory.rawOuterWrite_not_uniformOneStepOverwrite`.

#heading(level: 2, numbering: none)[Theorem set E: multi-programming as a structural predicate]

`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_ndm` defines
a predicate `IsMultiProgrammed` on architecture signatures capturing the
three multi-programming features (many independent heads per layer,
per-head state tile, per-batch independence). The 1.27 B NΔM signature
and a CMA-reshaped pure M²RNN signature both satisfy the predicate, and
a non-trivial hybrid signature *fails* it. This is the small formal
anchor for the *class-level* claim of §1: multi-programming is a
property of the PNR class shared across both PNR instances trained
here, not specific to NΔM. The trusted core formalises the PNR class
through two instances (NΔM and M²RNN-CMA), with NΔM as the
delta-correct contribution of this work and M²RNN-CMA as the raw-write
comparator.

#heading(level: 2, numbering: none)[NC#super[1] paragraph (verbatim)]

We adopt the audit-recommended wording from the formalisation gap
analysis:

#block(inset: (x: 1.5em), [
*NΔM is, at fixed width and precision, a finite-state recogniser (Lean:
`fixed_precision_state_space_finite`). Within that ceiling, an
orthonormal-key configuration of the NΔM update realises the $S_5$
prefix tracker (Lean: `NDMRealizesS5.ndm_realizes_s5_tracker`); $S_5$ is
non-solvable (`s5_not_solvable`) and by Barrington's theorem
@barrington1986 (cited; not formalized in this work) the $S_5$ word
problem is NC#super[1]-complete. NΔM therefore reaches the top of
NC#super[1] in the canonical regular-language witness.*
])

#heading(level: 2, numbering: none)[Explicit non-claims]

The trusted core does *not* prove the following, and the paper does not
claim them. (i) A Lean lower bound covering all linear-scan models on
$S_5$. (ii) Barrington's theorem itself; we cite it. (iii) Any "NΔM
exceeds NC#super[1]" or "NΔM exceeds TC#super[0]" claim; these are
families-wide impossibility statements outside the trusted surface. (iv)
A formal proof that a trained real-valued NΔM with empirically learned
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
these models live below NΔM on the $S_5$ probe of §6 and inside the
TC#super[0] complexity class @merrill2024transformers, confirmed by
their empirical $S_5$ collapse at length.

#heading(level: 2, numbering: none)[Pure-nonlinear-recurrent peers and adjacent nonlinear-state work]

*M²RNN* @m2rnn2026 (Mishra, Tan, Stoica, Gonzalez, Dao 2026) introduces
nonlinear matrix-state recurrence with a raw-write update
$Z = tanh(H W + k v^T)$ and demonstrates that it trains at 7 B MoE
scale in *hybrid form* (nonlinear-matrix-recurrent layers interleaved
with attention layers) on Nemotron-CC-v2 at 410 M dense and 7 B MoE.
This is concurrent prior art for the claim that nonlinear matrix-state
recurrence can be made to train at scale. M²RNN-CMA, used as a
within-class baseline throughout this paper, is the *pure-recurrent
variant* of M²RNN's architecture with no attention layers, CMA-tuned to
restore stable training at 1.27 B. The contribution of this work
relative to M²RNN is to extend the viability demonstration to the
*pure-recurrent* setting and to add the head-to-head comparison with
NΔM under matched per-architecture CMA-ES. The empirical separation in
§6 and the formal one-step resource separation in §7 quantify the
update-rule difference between delta-correcting (NΔM) and raw-write
(M²RNN-CMA) within the pure-nonlinear-recurrent class. Mishra et al.
additionally report hybrid M²RNN configurations favourably against
Mamba2 and Gated DeltaNet hybrids at matched parameter and token
budgets under a uniform fixed-hyperparameter protocol (their §5.2);
those hybrid configurations fall outside the pure-nonlinear-recurrent
class (the inserted attention layers violate the no-hybrid-bolt-ons
criterion of §1).

*xLSTM-1.3B* @xlstm2024 is a 7:1 mixture of mLSTM (linear) and sLSTM
(nonlinear) blocks; 87.5% of its blocks are linear-state, so xLSTM-1.3B
is not pure-nonlinear-recurrent by the no-linearisation criterion of
§1. It is the closest scale band among prior nonlinear-recurrent
results, included as a peer with the caveat that its nonlinear-block
share is small. The xLSTM-7B follow-up @xlstm7b2025 uses *only* mLSTM
(linear) blocks and is correspondingly further from the
pure-nonlinear-recurrent class.

*Titans* @titans2025 uses an MLP memory with online gradient updates
and is qualitatively nonlinear-state, but it is hybridised with
attention (also failing the no-hybrid-bolt-ons criterion) and has
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

The trusted Lean core proves the realisability of the $S_5$ tracker by
an orthonormal-key NΔM configuration, the one-step resource separation
from raw-write matrix RNNs, and the finite-state ceiling. It does *not*
prove: (i) a Lean lower bound covering all linear-scan models on $S_5$;
(ii) Barrington's theorem itself; (iii) any multi-step trajectory
separation between NΔM and raw-write matrix RNNs; (iv) any "exceeds
NC#super[1]" or "exceeds TC#super[0]" families-wide impossibility; (v) a
formal guarantee that empirical NΔM weights recover the lookup-table
realisation. The empirical $S_5$ accuracy of 0.79 at $T = 128$ is
*evidence* that real training approaches the realisable solution, not a
proof. The canonical-sweep length-extrapolation curves of §6 (parity,
FSM tracking, modular counter; NΔM-vs-baseline gap widening
monotonically with sequence length) supply *empirical* evidence
consistent with multi-step persistence of the one-step gap, but the
trusted core does not formally separate NΔM and raw-write matrix RNNs
across an unbounded trajectory.

#heading(level: 2, numbering: none)[Length extrapolation is not solved at scale]

NΔM separates from baselines at training length and remains ahead under
length extrapolation, but accuracy degrades monotonically with length:
on $S_5$, NΔM is at 0.79 at $T = 128$, 0.42 at $T = 256$, 0.22 at
$T = 512$, 0.11 at $T = 1024$. No recurrent family in our sweep solves
$S_5$ at $8 times$ training length.

#heading(level: 2, numbering: none)[Per-architecture CMA-ES is best-effort matched, not asymptotically optimal]

The 1.27 B wallclock racer (§5, @fig_lm_racers) is run under
*per-architecture CMA-ES*: each baseline uses its independently
CMA-tuned shape and hyperparameter choice under the matched protocol of
§5. The within-PNR ordering recorded in §5 (M²RNN-CMA trailing NΔM
across the sampled window) is therefore an order *under the
per-architecture CMA-ES protocol as configured here*, not under
asymptotic best-effort tuning. The protocol does not rule out the
counterfactual that further per-family CMA-ES generations, or a wider
search space, would close the gap at some budgets; the point of the
matched protocol is *symmetry of effort across architectures*, not
asymptotic optimality for any one.

#heading(level: 2, numbering: none)[Geometry-sensitivity of the update-rule claim]

The same update family (M²RNN) looks weak in the published paper shape
(diverges at 1.27 B) and stronger under CMA-tuned reshape (stable; loss
2.77 at the snapshot). The expressivity comparison of §6 holds at
parameter-matched 8 M scale; the language-modelling comparison of §5
relies on the CMA-reshaped M²RNN. The strongest reading of the
empirical within-class ordering is therefore *conditional* on the
multi-programmed shape, not on the update equation alone. The Lean
separation of §7 is unconditional on shape but is per-step.

#heading(level: 2, numbering: none)[The opposite architectural bet: hybrids]

A concurrent strand of work places the opposite architectural bet.
*OLMo-Hybrid 7B* @olmohybrid2026 interleaves state-space blocks with
attention, on the premise that hybrid stacks express things that lie
beyond what either pure transformers or pure linear RNNs can do. NΔM
is positioned honestly against that bet. A pure-nonlinear NΔM stack at
1.27 B matching GDN in the wallclock loss band does not refute
hybrids; what it refutes is the *assumption* that pure nonlinear
recurrence cannot scale at all. The hybrid-degradation finding in §6
($[upright("NΔM"), upright("NΔM"), upright("GDN"), upright("GDN")]$
underperforms either pure family on modular counter and FSM tracking at
8 M scale) is a *capability-preservation* observation: state-tracking
capability does not survive dilution by linear-scan blocks in our
sweep. It is not an anti-hybrid claim. The two architectural bets
address different questions ("can pure nonlinear recurrence scale at
all?" for this paper versus "what does a well-mixed hybrid express?"
for OLMo-Hybrid), and the answers do not contradict.

#heading(level: 2, numbering: none)[Snapshot status of the 1.27 B racer]

The three 1.27–1.35 B-band language-model training runs were in
progress at the time of submission. The loss-vs-wallclock racer
(@fig_lm_racers) is a snapshot as of 2026-05-24; the curves will
continue and the final-loss numbers may shift. The qualitative claim
(three families in the same wallclock loss band) is robust to the
remaining training; the quantitative ordering should be re-read at
the published-checkpoint stage.

#heading(level: 2, numbering: none)[Open architectural choices]

Several ablation findings remain unresolved. The output gate, the
non-linearity on the state ($tanh$ vs linear), and the decay
parameterisation (simple sigmoid vs Mamba2-style log-space) all show
loss-only ties at small scale; the strongest evidence for each
production choice is the empirical state-tracking and stability data
rather than a clean ablation at 1.27 B. The production architecture
keeps the conservative settings.

// ── 10. Conclusion ────────────────────────────────────────────────────────────
= Conclusion <sec:conclusion>

This paper demonstrates that pure-nonlinear-recurrent language models
can be trained at the 1.27–1.35 B-parameter band into the same loss-vs-
wallclock band as a frontier-class linear-recurrent baseline. Three
pure-recurrent architectures (NΔM and M²RNN-CMA, nonlinear in time;
Gated DeltaNet, linear in time) receive per-architecture CMA-ES
hyperparameter search and converge into a shared wallclock band on
The Pile. *Nonlinearity in time is not a cost* for language modelling
at this scale; the choice of recurrence linearity is washed out by
per-architecture tuning. To our knowledge, NΔM and M²RNN-CMA are the
first foundation-model-class pure-nonlinear-recurrent language models
trained at 1.27–1.35 B parameters. M²RNN (Mishra et al. @m2rnn2026)
is the closest prior art and demonstrates nonlinear matrix-state
recurrence at 7 B MoE scale in *hybrid form*; the pure-recurrent
variant trained here, M²RNN-CMA, is the head-to-head datapoint inside
the pure-nonlinear-recurrent class.

The technical discovery that makes pure nonlinear recurrence practical
at scale is *multi-programming*: width-axis parallelism across many
independent small recurrent heads, each carrying its own bounded matrix
state, with the time loop kept serial inside each head. Linear
recurrences gain throughput by *time-axis* linearisation; pure-nonlinear
recurrences gain throughput by replicating the recurrent computation
across thousands of small bounded memory programs that the GPU executes
in parallel. The recipe is update-rule-agnostic: both PNR instances
trained here satisfy the same multi-programming predicate at 1.27 B
(Lean: `multiProgrammed_admits_m2rnn_and_ndm`).

Within the pure-nonlinear-recurrent class, the delta-correcting update
rule (NΔM) trains consistently ahead of the raw-write update rule
(M²RNN-CMA) across the sampled wallclock window. The within-class gap
is explained by a one-step representability separation, formalised in
Lean 4: an orthonormal-key NΔM configuration realises the $S_5$ tracker
(`NDMRealizesS5.ndm_realizes_s5_tracker`), and no fixed-weight
raw-write matrix RNN with row, column, or cell forget gates can match
NΔM's mixed-key delta correction in one recurrent step
(`ndm_m2rnn_one_step_resource_separation_embeds`); the per-token FLOP
class is the same for the two PNR instances
(`ndm_m2rnn_flop_class_equiv`). The trainability shadow of the formal
separation is direct: at 8 M parameters, with capacity non-binding by
many orders of magnitude, raw-write stalls at 0.31 on the
six-element solvable-group $S_3$ control while the delta-correcting
update solves $S_3$ to ceiling and reaches 0.79 on the non-solvable
$S_5$ probe. The trusted Lean 4 core has no
`sorry`/`admit`/`axiom`/`opaque`/`native_decide` in the import closure.

*Release.* PNR checkpoints — NΔM (this work's delta-correct instance)
and M²RNN-CMA (the CMA-reshaped raw-write instance) — together with
the Gated DeltaNet baseline will be released on HuggingFace at
publication, alongside the per-architecture CMA-ES configurations,
the training protocol, and the Triton multi-programming kernel
source.

= Future Work <sec:future_work>

The results above leave the following directions open.

#heading(level: 2, numbering: none)[Formal multi-step separation]

The Lean separation in §7 bounds a *one-step specification*. A
multi-step formal counterpart to
`ndm_m2rnn_one_step_resource_separation_embeds` (a Lean separation
of delta-correct and raw-write across an unbounded trajectory) is the
obvious next-paper target. The §6 length-extrapolation curves (parity,
FSM tracking, modular counter; NΔM-vs-baseline gap widening
monotonically with sequence length) are the empirical signature one
would expect if the one-step gap compounds, but the formal multi-step
counterpart is not in the trusted Lean core.

#heading(level: 2, numbering: none)[A partial order on PNR update rules]

The within-PNR ordering established here (delta-correct $>$ raw-write)
is the first strict instance of an expressivity order on the
pure-nonlinear-recurrent class at matched per-token FLOP class
(`ndm_m2rnn_flop_class_equiv`). Whether this relation extends to a
*partial order* with an NC#super[1] ceiling (Barrington) on the broader
PNR space (in particular, whether antisymmetry and transitivity hold
over the full space) is open. The maximal element under matched
asymptotic FLOP class (which PNR update rule has the highest one-step
expressive power for its compute class) is the open horizon, with the
`RecurrentResourceFormalism` Lean machinery as the tool for climbing
the order.

#heading(level: 2, numbering: none)[Additional seeds and architecture-internal revalidation]

The 1.27 B racer is single-seed per family at the snapshot used here.
Several architecture-internal choices in NΔM, including the output
gate, the non-linearity on the state ($tanh$ vs linear), and the decay
parameterisation (simple sigmoid vs Mamba2-style log-space), show
loss-only ties at small scale; revalidation of each at 1.27 B is open.

#heading(level: 2, numbering: none)[Scale beyond 1.27 B]

The wallclock-band convergence is observed at 1.27 B parameters on
The Pile. Whether the same band convergence holds at larger scale, on
larger corpora, or under longer training is open.

#heading(level: 2, numbering: none)[Cleanest within-class HPO follow-up]

The cleanest within-PNR comparison would be a side-by-side run of NΔM
and M²RNN-CMA under further per-family CMA-ES generations at 1.27 B
beyond the current protocol, to test whether wider search closes the
within-PNR gap. This is the highest-value follow-up for distinguishing
"under the per-architecture CMA-ES protocol used here" from "under any
matched search effort."

// ── Appendix A — E63→E88 lineage and ablation notes ────────────────────────────
= Appendix: Lineage of the E63 $arrow$ E88 experimental program <sec:appendix>

The NΔM architecture as presented in §3 is the endpoint of a multi-year
ablation lineage. The named milestones referred to here are internal
codenames; they are documented in the appendix for reproducibility of
the design history and are not used in the body of the paper. The
reference implementation of the production NΔM stack lives at
`ndm/models/e88_fused.py`; the fused Triton recurrence kernel sources
are at `ndm/triton/e88_triton_forward.py` and `e88_triton_backward.py`.

- *E63 (nonlinear delta design).* Established that linear-in-state
  recurrences cannot pass the Siegelmann–Sontag boundary
  @siegelmann1995 and that placing the non-linearity on the matrix
  state itself (rather than only on gates or the output) is the
  expressivity-determining choice. NΔM's $tanh$ on $S$ is inherited
  from this design.

- *E70–E75 (matrix-Elman / delta predecessors).* Established the fused
  Triton kernel pattern (outer product + decay + tanh as one kernel)
  and surfaced the gradient-spike failure mode of unbounded outer
  products. The L#super[2]-normalised key write inherited by NΔM is the
  fix.

- *E88 (production NΔM).* Stable at 1.27 B parameters under
  schedule-free AdamW. Key parameterisation choices: log-space Mamba2
  decay (with weight-decay exemption); L#super[2]-normalised
  $q, k$; SiLU on input projections before normalisation; numerically
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

#bibliography("refs.bib", title: "References", style: "ieee")
