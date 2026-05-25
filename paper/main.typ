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

The dominant operating verdict in recurrent language modelling is
that pure-nonlinear-in-time recurrence cannot reach foundation-model
scale on competitive wallclock, because it forecloses the time-axis
parallel scan that linear-recurrent variants depend on for GPU
throughput. We test this verdict by training three pure-recurrent
language models at 1.27–1.35 B parameters under per-architecture
CMA-ES hyperparameter search: two with nonlinear time recurrence,
*the Emender* (delta-correcting update $S <- tanh(d S + k(v - S^T k)^T)$)
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
throughput. Within the pure-nonlinear-recurrent class, the Emender trains
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
nonlinear-in-time recurrences (the Emender, M²RNN-CMA) land in the same
loss-vs-wallclock band as the linear-recurrent baseline Gated
DeltaNet under per-architecture CMA-ES. The field's operating
verdict — that pure-nonlinear-in-time recurrence cannot reach this
scale on competitive wallclock — is an artefact of parallelizing
the time axis; width-axis multi-programming recovers throughput
while the time loop stays serial. Within the nonlinear class, the Emender
trains consistently ahead of M²RNN-CMA, with the one-step
representability separation formalised in Lean 4. Checkpoints,
CMA-ES configs, and the Triton kernel released.
*/

#show: arkheion.with(
  title: "Emending Nonlinear Recurrence",
  authors: (
    (
      name: "Erik Garrison",
      email: "erik.garrison@gmail.com",
      affiliation: "Poetic PBC / UTHSC, Memphis, TN, USA",
      orcid: "0000-0003-3821-631X",
    ),
  ),
  abstract: [
    *The update rule of a nonlinear recurrent network is one design
    choice in a space of maps, not a single locked decision.* This
    paper introduces that space, a machine-checked method for ordering
    it (Lean 4, §7), and the systems substrate — *multi-programming*
    — that makes any point in the space trainable at foundation-model
    scale. The Lean core is the methodological anchor: the rules are
    *provably distinct as computational objects* and the order is
    derivable rather than asserted; the trusted import closure of
    `ElmanProofs.PaperCore` has no
    `sorry`/`admit`/`axiom`/`opaque`/`native_decide`. We place the
    first rule in the order: *delta correction* writes against what
    the existing slot already predicts —
    $S <- tanh(d S + k(v - S^T k)^T)$ — and is proved one-step and
    $k$-step distinct from raw write for every $k >= 1$ (Theorem sets
    C and C′, §7). The empirical shadow: we train one delta-correct
    instance (the *Emender*, this work) and one raw-write instance
    (*M²RNN-CMA*) at 1.27–1.35 B on The Pile alongside the
    linear-recurrent baseline *Gated DeltaNet* under per-architecture
    CMA-ES; all three land in the same loss-vs-wallclock band —
    *nonlinearity in time is not a cost* — recovered on the width
    axis through 22,200 small independent recurrent programs per
    token, with the time loop kept serial inside each. The Emender
    trains consistently ahead of M²RNN-CMA across the sampled window
    and clears state-tracking probes ($S_5$, $S_3$) where both
    baselines stall under matched no-tuning at parameter-matched 8 M
    scale. The cross-class M²RNN-CMA gap is the within-class evidence
    available today; the cleanest within-class probe — ablating the
    delta-correction rule out of the Emender's exact geometry, one
    knob, matched FLOPs by Theorem D — is named as the next
    experiment. The Emender is one element of the rule space; the
    contribution is *the space and the method for ordering it*. We
    will release Emender checkpoints together with M²RNN-CMA and the
    GDN baseline, the per-architecture CMA-ES configurations, and the
    Triton multi-programming kernel on HuggingFace at publication.
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
#let nd = $upright("Emender")$
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
Gated DeltaNet @gated_deltanet2024 is the linear-recurrent model
Mamba-3 @mamba3_2026 measures itself against and beats by
$tilde 0.6$ points downstream; we treat it as the wall-clock bar to
clear.

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

#heading(level: 2, numbering: none)[The space of update rules and the method for ordering it]

A recurrent network's update rule is one design choice in a *space*
of nonlinear maps, not a fixed canonical decision. *The contribution
of this paper is the space and the method for ordering it.* Two
moves make that space tractable. (i) *Lean 4 as the methodological
core* (§7): the rules are computational objects whose per-step
behaviour can be machine-checked, so distinctions between rules
become *derivable* — at fixed precision and width, in the
multi-programmed signature — rather than asserted. The trusted
import closure of `ElmanProofs.PaperCore` contains no
`sorry`/`admit`/`axiom`/`opaque`/`native_decide`. (ii)
*Multi-programming as the systems substrate* (§4): width-axis
parallelism across many small bounded heads makes any point in the
space trainable at foundation-model scale without resorting to a
time-axis linearisation trick or attention hybridisation; the recipe
is *update-rule-agnostic*
(`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_emender`,
§7).

Within that frame, *delta correction* — write the correction against
what the slot already predicts — is the first rule we place in the
order. Its companion is *raw write* (the M²RNN family's update).
The one-step and $k$-step Lean separations (§7, Theorem sets C and
C′) prove the two rules are *distinct as computational objects* for
every finite $k$ on the constructed witness alphabet; the
length-extrapolation curves of §6 are the empirical shadow on
natural-language-shaped sequences. Whether the relation extends to a
*partial order* on the broader pure-nonlinear-recurrent space —
antisymmetric, transitive, with an NC#super[1] ceiling
@barrington1986 — is the open horizon named in §12; the
`RecurrentResourceFormalism` Lean machinery is the tool for climbing
that order. One trajectory at 1.27 B is *not* "my architecture wins";
it is the first datum after that order is opened.

#heading(level: 2, numbering: none)[Emendation is one response; hybrids are another]

The recurrent state-tracking problem — how does a fixed-width recurrent
layer maintain enough state to track structured prefixes like the $S_5$
word problem? — admits more than one architectural response. *Hybrid*
architectures interleave recurrent layers with attention layers and
recover much of what attention-free recurrence loses on state tracking
and long-range mixing; M²RNN @m2rnn2026, OLMo-Hybrid @olmohybrid2026,
Titans @titans2025, and Griffin @griffin2024 are recent instances, and
the formal limit-of-pure-recurrence story is documented by Merrill,
Petty and Sabharwal @merrill2024transformers. *Emendation* — the
update-rule innovation introduced in this paper — is a different
response: it does not add cross-token mixing, but it changes what the
recurrent state writes at each step from a raw additive overwrite to a
delta correction against the existing slot. The two responses are
complementary, not rival: a hybrid stack can in principle use any update
rule in its recurrent layers, and emendation is well-defined wherever a
matrix-state recurrent layer is.

For this paper, we evaluate emendation in the attention-free,
time-serial recurrent arena — the *pure nonlinear recurrent (PNR)*
setting in the sense of "no cross-token attention mechanism"; gating,
projections, normalisations and other intra-token nonlinear operations
are permitted and indeed load-bearing (see §3, ingredient (c) and §5 on
gradient conditioning). The PNR arena is where the comparison is
cleanest (an emendation-vs-raw-write update-rule contrast, with FLOP
class held equal — §7), where the Lean 4 separation theorems are
tractable (sets C and C′), and where the scaling demonstration is most
load-bearing (no architectural escape hatch from the recurrent state);
emendation generalises to hybrid recurrent layers wherever a matrix
state and a delta-correcting write make sense.

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

#heading(level: 2, numbering: none)[Contribution]

The paper's load-bearing claims, in order of evidence cleanliness:

#set list(indent: 1em)
- *Methodological core (Lean 4, §7).* Machine-checked one-step and
  $k$-step separations between the delta-correcting and raw-write
  update families, plus the multi-programming predicate that both
  pure-nonlinear-recurrent instances satisfy. This evidence does not
  depend on a trajectory, a seed, or an optimiser.

- *Within-class evidence (matched geometry, §6).* At
  parameter-matched 8 M scale under matched no-tuning, the
  delta-correcting Emender clears $S_5$/$S_3$ where the raw-write
  M²RNN-CMA stalls; the cleanest variant — ablating the
  delta-correction rule out of the Emender's *exact* geometry, one
  knob, matched FLOPs by Theorem D (§7) — is named below as the next
  experiment.

- *Cross-architecture corroboration (§5).* At 1.27–1.35 B under
  per-architecture CMA-ES, three architectures (the Emender,
  M²RNN-CMA, Gated DeltaNet) land in the same loss-vs-wallclock band
  on The Pile; the Emender trains consistently ahead of M²RNN-CMA
  across the sampled window. This is a single trajectory per
  architecture and is read as *corroboration* under the §5
  geometry-sensitivity caveat, not as a head-to-head verdict.

#set list(indent: 0em)

The five-component *synthesis* described below is the working stack
at which all three lines of evidence are anchored.

#set enum(numbering: "1.")

+ *Many-headed cache-fit multi-programming.* The 1.27 B production
  stack exposes 22,200 small independent recurrent programs per token
  (370 heads $times$ batch size 5 $times$ depth 12), each a
  $32 times 32$ matrix-state tile that fits in registers / L1 / SRAM.
  This width-axis parallelism is what makes attention-free nonlinear
  recurrence competitive on wallclock with the linear-scan baselines
  that monolithic large-state RNNs forfeit. Lean-witnessed by
  `emender_1p27B_programs_per_batch_token_bs5`.

+ *Emendation / delta correction.* The general update-rule innovation:
  $S <- tanh(d S + k (v - S^T k)^T)$ writes the *correction* against
  the existing slot rather than a raw additive overwrite. With
  orthonormal keys this gives exact overwrite at one slot while
  preserving the others; with arbitrary keys it gives bounded
  error-correcting binding (§3, Theorem sets C and C′ in §7).
  Emendation is well-defined wherever a matrix-state recurrent layer
  is; the PNR demonstration here is the cleanest comparison setting,
  not the technique's scope.

+ *Gating, with the ablation that says it matters.* Earlier
  non-gated 'E-version' prototypes (the E63–E75 lineage; Appendix) were
  *not* amenders in the operational sense — they accumulated
  uncorrected drift. The SiLU output gate is load-bearing under the
  ablation lineage that produced the production stack; removing it
  costs accuracy on state tracking.

+ *Triton kernel — portable, performant foundation.* The fused
  forward and sparse-checkpoint backward kernels are written once in
  Triton @triton2019 and dispatched identically on NVIDIA CUDA and
  AMD ROCm. Moving off CUDA-specific paths to Triton gives portability
  at $approx 70%$ of hand-tuned HIP throughput in $approx 1$ week of
  porting work (versus 3–6 weeks for a HIP rewrite). The kernel is
  released as the foundation for further update-rule research, not as
  a one-off for this paper.

+ *Synthesis: Emender 88 (E88).* "Emender 88" (E88) is the specific
  version handle within the Emender family for the integration of
  (1)+(2)+(3)+(4) at 1.27 B parameters. The contribution is the
  integration: each component is small or known; together they
  constitute the first foundation-model-scale demonstration of
  emendation in the PNR arena.

#set enum(numbering: "(a)")

This paper establishes the PNR class as viable at foundation-model
scale, by training two PNR instances at the 1.27–1.35 B parameter
band and comparing them head-to-head with a linear-recurrent baseline. The two PNR instances are the Emender (this
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
exploration; to our knowledge, the Emender and M²RNN-CMA are the first
foundation-model-class PNR language models trained at this scale.

Within the PNR class, the Emender (this work's delta-correct instance)
trains consistently ahead of M²RNN-CMA across the sampled wallclock
window. The paper proceeds as follows. §2 and §3 set up the
linear-state versus nonlinear-state classification and the Emender
architecture (this work's contribution within the PNR class); §4
covers the multi-programming systems contribution; §5 presents the
1.27 B wallclock racer with the per-architecture CMA-ES protocol;
§6 reports the 8 M expressivity probes with the
capacity-non-binding justification; §7 the Lean 4 formalisation; §8
related work; §9 limitations; §10 conclusion; §11 testable predictions;
§12 future work. We
will release Emender checkpoints together with M²RNN-CMA, the GDN baseline,
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

Two nonlinear matrix-state designs (the Emender and M²RNN) therefore share the
necessary preconditions (matrix state, nonlinearity on the state, no
attention, no linearisation, no hybrid bolt-ons) that define the
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
$S in RR^(N times V)$ is not novel to the Emender. Linear-state designs
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

#heading(level: 2, numbering: none)[Pangenomics as one downstream sequence-modelling regime]

Pangenomics has a sequence-modelling problem: terabase-scale datasets,
billions of characters per genome, with the cost of developing capable
ML systems for such data still exorbitant @hprc2023
@guarracino2023acrocentric @pggb2024. This work does not target that
regime directly, but the pure-recurrent, time-serial, width-parallel
substrate developed here is the kind of building block such regimes
need.

// ── 3. Architecture — the Emender ─────────────────────────────────────────────
= Architecture <sec:arch>

#heading(level: 2, numbering: none)[Per-head update]

We introduce the Emender, a pure nonlinear recurrent language model whose
update rule reads the current state, computes the prediction error, and
writes the delta correction. Each Emender layer maintains $H$ independent heads. Each head $h$ owns a
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
    $L^2$-normalisation of $q,k$ are the three load-bearing design
    choices (§3).
    *(B)* The production stack at 1.27 B parameters exposes 370 small
    independent heads per layer per batch element. Each head is a
    $32 times 32$ state tile that fits in registers/SRAM and runs its
    time loop serially; parallelism is harvested across heads, depth and
    batch (§4). The Lean theorem
    `emender_1p27B_programs_per_batch_token_bs5` certifies 22,200 programs
    per token at batch size 5.
  ],
) <fig_arch>

Three ingredients are load-bearing.

#set enum(numbering: "(a)")

+ *Bounded matrix state via $tanh$ on $S$.* Linear-in-state recurrences
  cannot pass the Siegelmann–Sontag boundary even in principle; placing
  the non-linearity on the state itself, not only on a gate or the
  output, is what makes the Emender nonlinear-state.

+ *Delta-correcting write $v - S^T k$.* The model reads what memory
  predicts at address $k$, computes the prediction error $delta$,
  and writes the correction. With an orthonormal key family this gives
  exact overwrite at one slot while preserving the others; with
  arbitrary keys it gives bounded error-correcting binding. A raw-write
  update $S <- S + k v^T$ (the M²RNN family) accumulates without
  correction and cannot, with fixed weights, satisfy a uniform one-step
  overwrite specification (§Formal Results).

+ *Many small heads, not one large matrix.* Production Emender at 1.27 B
  uses $H = 370$ heads of $32 times 32$ each (Lean-witnessed:
  `RecurrentResourceFormalism.emender_1p27B_programs_per_batch_token`,
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

The 1.27 B parameter Emender stack used throughout the paper has dim = 1664,
depth = 12, $H = 370$, $N = V = 32$, with a tied LM-head embedding and
prenorm residual blocks. The reference implementation and fused Triton
recurrence kernel sources live under `ndm/models/` and `ndm/triton/`;
file paths are recorded in the Appendix.

#heading(level: 2, numbering: none)[Ablation by architecture: isolating the write rule]

Three properties are candidates for the load-bearing differentiator in
state-tracking: *matrix state*, *temporal nonlinearity on that state*,
and *delta correction in the write*. The closest update-rule comparator
to the Emender in the literature is M²RNN @m2rnn2026, whose nonlinear
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
      [*Property*], [*GDN*], [*M²RNN*], [*Emender*], [*Verdict*],
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
    length (0.36 vs Emender 0.79, §6), so matrix state alone cannot be the
    differentiator. M²RNN has matrix state *and* temporal nonlinearity
    on the state and still stalls at 0.22 on $S_5$, so temporal
    nonlinearity is not the differentiator either. Delta correction is
    the only property left that the Emender has and the two baselines do not.
  ],
) <tab_ablation>

M²RNN scores 0.31 on $S_3$, the solvable control where non-solvability
is *not* the obstruction. This rules out a complexity-ceiling
explanation. If raw-write could do clean prefix tracking even on
solvable groups, M²RNN should clear $S_3$. It does not. Nor is the
failure a capacity ceiling. Two distinct non-binding bounds make this
point. (i) At the 8 M probe shape
($N times V times H times "depth" = 32 times 32 times 32 times 4 =
131{,}072$), the per-token recurrent state carries 131,072 scalars,
about five orders of magnitude above the $log_2 6 approx 2.6$-bit
information-theoretic floor for representing the $S_3$ prefix-tracking
table. (ii) Independently, the learned
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
`RecurrentResourceFormalism.emender_m2rnn_one_step_resource_separation_embeds`
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
recurrent language modelling tractable at the 1.27 B band. The solution
turned out to be width, not time. Linear recurrences gain throughput by
*time-axis* linearisation: composing
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
trained in this paper (the Emender and M²RNN-CMA) satisfy the same
multi-programming predicate at 1.27 B
(`RecurrentResourceFormalism.multiProgrammed_admits_m2rnn_and_emender`,
§7). Mishra et al.'s M²RNN paper @m2rnn2026 observed the benefit of
many small recurrent heads in their hybrid configuration; the
multi-programming optimisation here generalises that observation to
pure-recurrent architectures at language-model scale and identifies it
as the design choice that converts pure nonlinear recurrence from a
sub-billion-parameter curiosity into a billion-parameter LLM building
block.

#heading(level: 2, numbering: none)[The 1.27 B Emender shape under multi-programming]

A modern GPU exposes thousands of independent streaming multiprocessors,
each able to run a small program in registers and shared memory. The
Emender 1.27 B stack invites 370 such programs per layer per batch element
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
~70% of the hand-tuned throughput in ~one week of porting work, versus
three to six weeks to port to HIP from scratch.

#heading(level: 2, numbering: none)[Distributed training]

The training plan uses schedule-free AdamW @schedulefree2024 per island
with hierarchical local-SGD model averaging in the DiLoCo
@diloco2023 style: each island is one node of 8 GCDs with intra-island
DDP, and inter-island synchronisation averages model weights every
$H = 250$ local steps (an empirically-chosen interval; see §Limitations
for the open question on $H$). Because parallelism is across programs
rather than along time, the Emender does not require sequence parallelism to be
competitive at 1.27 B; this is a simplification relative to
chunked-scan implementations of linear-state recurrences. Separately,
ParaRNN @pararnn2025 parallelises the time loop itself via Newton's
method on a block-bidiagonal Jacobian; whether that method converges on
the $tanh(d S + k delta^T)$ map at $32 times 32$ block size is open.

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
  [Emender], [1.273 B], [5], [dim=1664, depth=12, H=370, N=32],
  [M²RNN-CMA], [1.307 B], [5], [dim=1920, depth=21, H=370, N=16],
  [GDN], [1.352 B], [4], [dim=2688, depth=21, exp=2, H=44],
)]

#heading(level: 2, numbering: none)[Per-architecture CMA-ES protocol (fairness anchor)]

All three 1.27–1.35 B architectures (Emender, M²RNN-CMA, and Gated
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
delta-correcting PNR instance (the Emender) and the linear-recurrent baseline
(GDN) track the same loss-vs-wallclock band and trade leadership
through training (@fig_lm_racers), while the raw-write PNR instance
(M²RNN-CMA) trails throughout the sampled window. The §7 mechanism
reading is what isolates the source of this within-PNR separation:
matrix state plus temporal nonlinearity is competitive with
frontier-class linear recurrence under matched per-architecture tuning,
and the *delta-correcting* write is what places the Emender at parity with
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
$q,k$ pairs per value head (it is also the shape ratio that the Emender uses
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
recorded in §5 is therefore an ordering between the Emender under its
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
    moving averages of raw training loss (nats per token). Emender is at
    1.273 B parameters; M²RNN-CMA at 1.307 B; GDN at 1.352 B. Each
    model has trained 8–15 GPU-days at this recording; the
    $tilde 14$-day per-architecture training extent is the standard
    unit at this scale class. Training continues. *Panel A:* full
    curve on
    log-wallclock from h = 1. *Panel B:* tail (h ≥ 40) on linear
    wallclock. Emender and GDN share a single loss band through the bulk
    of training, with leadership trading between them at the
    fractional-nat scale; the two curves are nearly co-linear.
    M²RNN-CMA has higher loss than the other two across the sampled
    window. The paper-shape M²RNN baseline (not shown) diverged at
    step 8,400. Color convention used throughout the paper: Emender =
    blue, GDN = orange, M²RNN-CMA = red.
  ],
) <fig_lm_racers>

Across the shared wall-clock window, the Emender and GDN occupy the same band:
leadership trades between them through training, and at no sampled
point do the two separate by more than a small fraction of a nat.
After $tilde 14$ wall-clock days of training, recorded losses are
2.66 (Emender, step 1,035,000), 2.68 (GDN, step 1,371,000), and 2.77
(M²RNN-CMA, step 958,000). These numbers sit in the loss band reported
for 1–2 B parameter models on The Pile under matched tokenization. The within-PNR comparison is qualitatively different:
M²RNN-CMA trails the Emender across the entire sampled window under the
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

#heading(level: 2, numbering: none)[Ordering the evidence by cleanliness]

A within-class claim about the *update rule* is cleanest when the
geometry around it is held fixed. The cleanest possible variant is a
single-knob ablation in the Emender's *exact* geometry:

#block(inset: (x: 1.5em), [
*Take the Emender stack at 1.27 B (dim=1664, depth=12, H=370,
N=V=32), CMA-tuned. Substitute the delta-correcting write
$delta = v - S^T k$ with the raw-write $delta = v$. Keep everything
else fixed — gating, $L^2$ normalisation, decay parameterisation,
optimiser, tokenizer, data, search budget. Re-tune under matched
CMA-ES. By Theorem set D (§7) the two heads sit inside the same
$c_1 d^2 + c_2 d$ FLOP envelope, so the comparison is FLOP-matched
by construction.*
])

That experiment is *designed and infrastructure-ready*: a
documented protocol exists, the Triton kernel exposes a
`RAW_WRITE` flag, and the per-architecture CMA-ES driver carries an
`e88-raw` model entry. Compute scheduling places the result outside
this paper's training-data window. *This is the within-class probe
named as the next experiment in §11.*

What §6 reports is the within-class and cross-architecture evidence
that *is* available today, ordered by how cleanly each isolates the
update rule:

#set list(indent: 1em)
+ *Lean (no trajectory, no seed; §7).* The one-step resource
  separation and the $k$-step persistence on the constructed witness
  alphabet (`emender_m2rnn_one_step_resource_separation_embeds`,
  `emender_m2rnn_k_step_separation`). This is the methodological
  spine, independent of any training run; the empirical sections
  below are its *shadow*.

+ *Matched-no-tuning, parameter-matched 8 M state-tracking probes
  (this section).* Each family is evaluated on the
  reasonable-defaults configuration it would arrive at without
  probe-targeted optimisation. The Emender clears $S_5$ and $S_3$
  where M²RNN-CMA and M²RNN-paper stall; geometry differs across
  families, so this is *cross-architecture under matched no-tuning*,
  not single-knob.

+ *Cross-architecture corroboration: 1.27 B loss racer (§5) and
  length extrapolation (this section).* The 1.27 B M²RNN-CMA-vs-Emender
  gap holds across an independent codebase and at chinchilla-scale
  training, with the §5 geometry-sensitivity caveat made explicit:
  M²RNN-CMA is the CMA-reshaped M²RNN, not the published one, so the
  comparison reads the *update rule under its own CMA-optimised
  geometry* against the Emender's. This is corroboration, not
  ablation.

#set list(indent: 0em)

Across the two scales: the update-rule gap is large at the 8 M
state-tracking probes (Emender 0.79 vs M²RNN-CMA 0.22 on $S_5$ at
training length) and small at the 1.27 B language-modelling loss
(Emender 2.66 vs M²RNN-CMA 2.77 after $tilde 14$ days under per-arch
CMA-ES). The two are different metrics on different tasks; the
honest reading is that *the write rule is load-bearing for prefix
tracking and contributes a fractional-nat margin on Pile loss*. The
cleanest within-class number is the one the named experiment above
will deliver.

#heading(level: 2, numbering: none)[Capacity is non-binding at 8 M parameters for these probes]

The state-tracking probes in this section are run at 8 M
parameter-matched scale (dim = 384, depth = 4, $H = 32, N = 32$,
schedule-free AdamW, 10K–20K steps per task, three seeds). GDN uses
dim = 640 to match parameter count. Before any accuracy number is
reported, the capacity argument: 8 M parameters is *overparameterised*
for the state-tracking tasks studied here by many orders of magnitude.
The 8 M probe is the Emender's default configuration carried down from its 1.27 B production stack.
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
no seed sweep for any family in the comparison. The Emender ran on the default
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
history*: the Emender's defaults are the endpoint of an ablation lineage
selected partly on state-tracking behaviour (Appendix), whereas GDN and
M²RNN's published defaults were selected by their authors on
language-modelling loss. The matched-no-tuning condition therefore
controls for differential probe-specific effort, not for this selection
asymmetry. The $S_3$ control isolates the part of the within-class
claim that is immune to it: raw-write's 0.31 on a six-element solvable
group, where the $log_2 6 approx 2.6$-bit table sits well below the
non-binding capacity ceiling, is a property of the update rule under
SGD at the 8 M probe shape, not a property of the Emender's design history.

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
    M²RNN variants stall in the 0.31–0.38 band, indicating that the
    raw-write update fails on the prefix-tracking task even without the
    non-solvability obstruction. Source: `paper/ndmpapernotes.md`
    lines 153–173; figure script: `paper/figures/plot_expressivity_seeds.py`.
  ],
) <fig_s5_bars>

The Emender is the only family that crosses 0.5 accuracy on $S_5$ at training
length. It is also the only family that solves $S_3$ to ceiling. Both
M²RNN variants stall in the 0.31–0.38 band, indicating that the
raw-write update fails on the prefix-tracking task even when the group
is solvable. The gap between the Emender and the next-best baseline shrinks
under length extrapolation but does not close: at $T = 512$ the Emender is at
0.215 and GDN at 0.097.

The $S_3$/$S_5$ split is also where the scope of the M²RNN paper's own
state-tracking evaluation matters. Mishra et al. @m2rnn2026 §3.2 report
length generalisation on $S_3$ alone (the smallest non-trivial
*solvable* group, which lives inside TC#super[0]) and do not evaluate
$S_5$ or any other non-solvable group. The unhedged "perfect
state-tracking generalisation" framing in that paper therefore does not
bear on the NC#super[1] regime: the Emender's $0.79$ at parameter-matched 8 M
scale against the paper-default M²RNN's $0.17$ on $S_5$ at training
length is direct evidence that length generalisation on $S_3$ does not
extend across the TC#super[0]/NC#super[1] boundary under the raw-write
update.

#heading(level: 2, numbering: none)[The six-task canonical sweep]

To verify that the $S_5$ result is not a single-task artefact we run a
six-task canonical sweep covering parity (binary XOR over a stream),
modular counter (K=5), FSM tracking (K=4 states),
Dyck-1 (balanced brackets), associative recall (key→value lookup), and
selective copy (mark-and-copy). At 8 M parameter-matched scale, the Emender ties
or wins GDN on five of six tasks (parity 1.00 vs 0.86; modular
counter 0.90 vs 0.65; FSM tracking 1.00 vs 0.83; Dyck-1 1.00 vs 1.00;
selective copy 1.00 vs 1.00). GDN edges the Emender on associative recall
(0.997 vs 0.881), the only attention-natural task in the suite.

Under length extrapolation (train $T = 40$, evaluate up to $T = 500$),
the Emender retains 0.89 accuracy on parity at $T = 500$ where GDN collapses
to 0.55 (near random 0.50). On FSM tracking at $T = 500$, the Emender is at
0.59 versus GDN 0.39. The gap widens monotonically with length.

This monotonic-widening pattern is the empirical shadow of *multi-step
persistence*. On parity, FSM tracking, and modular counter, the
Emender-vs-baseline accuracy gap grows with sequence length rather than
closing — the signature one would expect if the one-step ordering
compounds rather than washes out. The formal companion is the k-step
separation proved in §7
(`emender_m2rnn_k_step_separation`, Theorem set C′): for every finite
$k$ the gap persists on the constructed 2D witness alphabet,
complementing the one-step resource separation (Theorem set C,
`emender_m2rnn_one_step_resource_separation_embeds`). The
length-extrapolation curves above are the empirical signature that the
same failure-to-wash-out holds on natural-language-shaped sequences;
the remaining frontier — an $S_5$-generator-specific capacity bound
with explicit $T(d)$ — is named in §12.

#heading(level: 2, numbering: none)[Hybrid degradation: purity matters]

A natural question for any architecture that wins on state tracking is
whether the gain survives mixing with linear-scan blocks. We test the
pattern $[upright("Emender"), upright("Emender"), upright("GDN"), upright("GDN")]$
(four-layer "AABB" hybrid) on the same canonical sweep and find that
*hybridisation degrades state tracking below either pure family*:

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
    baseline ($1/K$). State-tracking capability is not a property the
    Emender block can lend to a stack of mixed blocks; purity is part of
    the recipe. Source: per-seed JSON under
    `paper/results/figure_4_hybrid/`; figure script:
    `paper/figures/plot_hybrid_degradation.py`. Mean $plus.minus$ std
    mirrors `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`.
  ],
) <fig_hybrid>

The mechanism interpretation. M²RNN underperforms both at the $S_5$
training length (0.22 vs Emender 0.79) and on $S_3$ (0.31 vs Emender 1.00) at
the same parameter count. Matrix state plus temporal nonlinearity
*alone* (what the Emender and M²RNN share) is not sufficient; the delta
correction
$v - S^T k$ is the load-bearing piece. The $S_3$ probe is the cleaner
control for reading this as a *trainability* claim. $S_3$ has six
elements; storing its transition table requires $log_2 6 approx 2.6$
bits of state. At the probe shape the per-token recurrent state has
$N V H "depth" = 131{,}072$ scalars (and independently the learned
function is encoded in $approx 1.3 times 10^8$ parameter bits at fp16),
so capacity is non-binding by many orders of magnitude on either
accounting, and M²RNN's matrix state can in principle hold an $S_3$
prefix-tracker. M²RNN's 0.31 is therefore evidence that SGD under the
raw-write inductive bias *does not find* such a configuration at this
scale, not that one fails to exist; the failure is empirical
learnability under the raw-write rule, not representational
impossibility. The hybrid degradation result strengthens the same
conclusion from the other side: linear-scan blocks cannot inherit
state-tracking capability from neighbouring Emender blocks. The Lean
formalisation of §7 provides the *representational* counterpart (a
per-step separation at fixed precision and width), not a global
trainability claim about the M²RNN family. Reconciling §3 with §7: the
Lean result
(`RecurrentResourceFormalism.emender_m2rnn_one_step_resource_separation_embeds`)
bounds a *one-step specification* (the precise mixed-key delta
overwrite that the Emender performs at each step), while the §3 $S_3$ argument
concerns *eventual representability across an unbounded number of
steps*, for which raw-write has the capacity in principle but for which
SGD under the raw-write inductive bias does not, at the 8 M probe scale,
locate a configuration that prefix-tracks. These are distinct claims about
different timescales of expressivity, both indicting the write rule.

#heading(level: 2, numbering: none)[QA and reasoning panel at 1.27 B: parity-rate evidence]

For capability beyond loss numbers we evaluate the three 1.27–1.35 B-band
models on a 300-item multi-choice continuation harness sampled from
ARC-C/E @arc2018, HellaSwag @hellaswag2019, SciQ @sciq2017, OpenBookQA
@openbookqa2018, and BoolQ @boolq2019. At the current training budget, the Emender
reaches 0.367 (random ~0.29), GDN 0.380, and M²RNN-CMA 0.367; all
three sit within one standard error of one another
($"SE" approx 6$ pp at 50 items per task). On a separate reasoning
panel (BIG-Bench Hard @bbh2022, ReCLor @reclor2020, FOLIO @folio2022),
all three families collapse on multi-step object tracking
(`tracking_shuffled_objects_7_objects` at 0.10–0.13, near-random) and
on FOLIO/ReCLor (near-random for all three), with GDN leading
on formal fallacies and web-of-lies. The Emender's overall reasoning
accuracy (0.319) is within one standard error of M²RNN (0.336). None
of the three architectures has crossed the threshold where reasoning
benchmarks differentiate, consistent with all three acquiring
capability at the same rate at this training budget. The panel supports the class-level claim
of §1: pure-nonlinear-recurrent Emender acquires standard benchmark
capability at the same rate as the linear-recurrent and raw-write
nonlinear baselines at this training budget. The state-tracking-specific reasoning
separation, if it exists, will emerge at longer training horizons; §11
states the prediction explicitly.

// ── 8. Formal Results ─────────────────────────────────────────────────────────
= Formal Results <sec:formal>

#heading(level: 2, numbering: none)[Lean 4 as the methodological core]

The Lean 4 development of this section is the paper's *methodological
spine*, not a corroborating appendix. The results below do not depend
on a trajectory, a seed, an optimiser, or a tokenizer; they make the
rule-space claim of §1 a *derivable* property of the update map
rather than an artefact of any one training run. Two consequences
follow.

#set list(indent: 1em)
- *The update families are computational objects.* Delta correction
  and raw write are, at fixed precision and width, distinct under
  every fixed-weight parameterisation with row, column or cell
  external forget gates (set C). The distinction *does not wash out*
  across $k$ steps on the constructed witness alphabet for every
  $k >= 1$ (set C′). These are sharp computational facts about the
  rules, not properties of a trained instance.

- *The order is derivable in the multi-programmed signature.*
  Theorem set E names the structural predicate `IsMultiProgrammed`,
  shows that the 1.27 B Emender and the CMA-reshaped M²RNN both
  satisfy it, and that a non-trivial hybrid does not. The
  rule-ordering machinery (`RecurrentResourceFormalism`) is the tool
  for climbing the partial order on the broader PNR space §12 names
  as the open horizon.

#set list(indent: 0em)

We have a trusted Lean 4 @lean42021 core built on Mathlib
@mathlib4. The import closure of the `ElmanProofs.PaperCore` module
(ten source files at the time of writing) contains no `sorry`, no
`admit`, no `axiom`, no `opaque`, and no `native_decide`. Each result
below is identified by its exact theorem name so that the reader can
locate it in the source. The headline §7 result is the *k-step
separation*
(`emender_m2rnn_k_step_separation`, theorem set C′ below): for every
$k >= 1$ and every fixed-right raw-write resource with row/column/cell
external forget gates, there is an explicit $k$-token input sequence on
which the $k$-step trajectories disagree. This is the machine-checked
answer to the reviewer-of-record concern that *"a one-step advantage
could in principle wash out over a trajectory or compound"*: the gap
strictly persists for every finite $k$ on the constructed witness
alphabet. The scope of what is and is not proved is summarised under
"Frontier and unproven targets" below.

#heading(level: 2, numbering: none)[Theorem set A: finite-state ceiling and $S_5$ tracker]

#set list(indent: 1em)
- *Finite-state ceiling at fixed precision.*
  `S5Witness.fixed_precision_state_space_finite` shows that every
  fixed-precision online recogniser has a finite state space. This bounds
  the Emender (at fixed width and precision) to regular-language
  recognisers, and therefore strictly inside NC#super[1].

- *$S_5$ word problem.* `S5Witness.s5_state_count` proves $|S_5| = 120$;
  `S5Witness.s5_not_solvable` proves $S_5$ is non-solvable;
  `S5Tracker.recognizer_state_count` shows the prefix-product tracker is
  a 120-state recogniser; `S5Tracker.run_append` proves that word
  execution composes permutations. The bridge
  `S5Tracker.pythonRun_eq_tracker_tuple` shows the Lean tracker agrees
  on every input with the Python evaluation harness.

- *Lookup-table realisation.* `S5EmenderRealization.s5_transition_key_count`
  shows the $S_5$ tracker uses exactly $120 times 4 = 480$ state/input
  keys; `S5EmenderRealization.exactTransitionMemory_run` shows that any
  finite recogniser admits an exact lookup-table realisation.

#heading(level: 2, numbering: none)[Theorem set B: Emender realises $S_5$]

The bridge from the abstract lookup-table realisation to the Emender update
equation is closed by the new result
`EmenderRealizesS5.emender_realizes_s5_tracker`: there exist an integer $d$, an
orthonormal family of keys ${k_g}$ indexed by the adjacent-transposition
generators, a value family ${v_g}$ and a decay scalar $lambda = 1$
such that the Emender update
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
  parameterisation with row, column or cell forget gates can match the Emender's
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

This is the direct, machine-checked answer to the reviewer-of-record
concern that the one-step advantage might compound away over a
trajectory. The witness alphabet is the 2-dimensional construction from
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
inductive base. All three live in the new module
`ElmanProofs.Architectures.MultiStepSeparation`, which is part of the
trusted import closure of `PaperCore`. The result rules out the
"wash-out over composition" failure mode the reviewer worried about,
for the resource class covered by set C, on the constructed witness.
The §6 length-extrapolation curves (parity, FSM tracking, modular
counter; Emender-vs-baseline gap widening monotonically with sequence
length) are the empirical companion of this formal multi-step
persistence on a different — natural-language-shaped — alphabet.

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
  pigeonhole capacity arguments specialised to recurrent maps with
  bounded weight matrices. Each piece is a research-grade
  mechanisation project in its own right; the integration is what
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
that one-step-realises the Emender's mixed-key delta overwrite at the matched
signature must allocate more state capacity (or, equivalently at fixed
state shape, more fixed weights) than the Emender, because no fixed-weight
raw-write parameterisation at the matched signature with row, column or
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
per-head state tile, per-batch independence). The 1.27 B Emender signature
and a CMA-reshaped pure M²RNN signature both satisfy the predicate, and
a non-trivial hybrid signature *fails* it. This is the small formal
anchor for the *class-level* claim of §1: multi-programming is a
property of the PNR class shared across both PNR instances trained
here, not specific to the Emender. The trusted core formalises the PNR class
through two instances (the Emender and M²RNN-CMA), with the Emender as the
delta-correct contribution of this work and M²RNN-CMA as the raw-write
comparator.

#heading(level: 2, numbering: none)[NC#super[1] paragraph (verbatim)]

We adopt the audit-recommended wording from the formalisation gap
analysis:

#block(inset: (x: 1.5em), [
*The Emender is, at fixed width and precision, a finite-state recogniser (Lean:
`fixed_precision_state_space_finite`). Within that ceiling, an
orthonormal-key configuration of the Emender update realises the $S_5$
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
exceeds NC#super[1]" or "the Emender exceeds TC#super[0]" claim; these are
families-wide impossibility statements outside the trusted surface. (iv)
A formal proof that a trained real-valued Emender with empirically learned
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
these models live below the Emender on the $S_5$ probe of §6 and inside the
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
the Emender under matched per-architecture CMA-ES. The empirical separation in
§6 and the formal one-step resource separation in §7 quantify the
update-rule difference between delta-correcting (the Emender) and raw-write
(M²RNN-CMA) within the pure-nonlinear-recurrent class. Mishra et al.
additionally report hybrid M²RNN configurations favourably against
Mamba2 and Gated DeltaNet hybrids at matched parameter and token
budgets under a uniform fixed-hyperparameter protocol (their §5.2);
those hybrid configurations fall outside the pure-nonlinear-recurrent
class (the inserted attention layers violate the no-hybrid-bolt-ons
criterion of §1).

*Mamba-3* @mamba3_2026 is concurrent and orthogonal: it pushes on
state-tracking within the linear-state paradigm via complex-valued
updates (a data-dependent rotary update that lifts the linear-recurrent
ceiling without leaving the linear class); the Emender pushes by
abandoning that paradigm.

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
an orthonormal-key Emender configuration, the one-step resource
separation from raw-write matrix RNNs, the k-step extension on the
constructed 2D witness alphabet
(`emender_m2rnn_k_step_separation`, for every $k >= 1$), and the
finite-state ceiling. It does *not* prove: (i) a Lean lower bound
covering all linear-scan models on $S_5$; (ii) Barrington's theorem
itself; (iii) an $S_5$-generator-specific multi-step capacity bound —
a statement of the form "for any fixed-weight raw-write RNN at state
dimension $d$ there is an explicit $T(d)$ past which $S_5$ coset
tracking is unreachable, while the Emender at the same $d$ tracks it"
(the k-step separation covers every finite $k$ on the constructed 2D
witness, not the $S_5$ generator alphabet with a stated $T(d)$); (iv)
any "exceeds NC#super[1]" or "exceeds TC#super[0]" families-wide
impossibility; (v) a formal guarantee that empirical Emender weights
recover the lookup-table realisation. The empirical $S_5$ accuracy of
0.79 at $T = 128$ is *evidence* that real training approaches the
realisable solution, not a proof. The canonical-sweep
length-extrapolation curves of §6 (parity, FSM tracking, modular
counter; Emender-vs-baseline gap widening monotonically with sequence
length) supply *empirical* evidence that the same failure-to-wash-out
persists on natural-language-shaped sequences, beyond the constructed
alphabet on which the formal k-step separation lives.

#heading(level: 2, numbering: none)[Length extrapolation is not solved at scale]

The Emender separates from baselines at training length and remains ahead under
length extrapolation, but accuracy degrades monotonically with length:
on $S_5$, the Emender is at 0.79 at $T = 128$, 0.42 at $T = 256$, 0.22 at
$T = 512$, 0.11 at $T = 1024$. No recurrent family in our sweep solves
$S_5$ at $8 times$ training length.

#heading(level: 2, numbering: none)[Per-architecture CMA-ES is best-effort matched, not asymptotically optimal]

The 1.27 B wallclock racer (§5, @fig_lm_racers) is run under
*per-architecture CMA-ES*: each baseline uses its independently
CMA-tuned shape and hyperparameter choice under the matched protocol of
§5. The within-PNR ordering recorded in §5 (M²RNN-CMA trailing the Emender
across the sampled window) is therefore an order *under the
per-architecture CMA-ES protocol as configured here*, not under
asymptotic best-effort tuning. The matched protocol provides *symmetry
of effort across architectures*; it is not a claim of asymptotic
optimality for any one. §12 lists a wider-search follow-up as the
cleanest next step.

#heading(level: 2, numbering: none)[Geometry-sensitivity of the update-rule claim]

The same update family (M²RNN) looks weak in the published paper shape
(diverges at 1.27 B) and stronger under CMA-tuned reshape (stable; loss
2.77 after $tilde 14$ days of training). The expressivity comparison of §6 holds at
parameter-matched 8 M scale; the language-modelling comparison of §5
relies on the CMA-reshaped M²RNN. The empirical within-class ordering is
therefore *conditional* on the multi-programmed shape, not on the
update equation alone. The Lean
separation of §7 is unconditional on shape but is per-step.

#heading(level: 2, numbering: none)[The opposite architectural bet: hybrids]

A concurrent strand of work places the opposite architectural bet.
*OLMo-Hybrid 7B* @olmohybrid2026 interleaves state-space blocks with
attention, on the premise that hybrid stacks express things that lie
beyond what either pure transformers or pure linear RNNs can do. The
Emender is positioned against that bet without contradicting it: a
pure-nonlinear Emender stack at 1.27 B matching GDN in the wallclock
loss band does not refute hybrids; what it refutes is the *assumption*
that pure nonlinear recurrence cannot scale at all. The hybrid-degradation finding in §6
($[upright("Emender"), upright("Emender"), upright("GDN"), upright("GDN")]$
underperforms either pure family on modular counter and FSM tracking at
8 M scale) is a *capability-preservation* observation: state-tracking
capability does not survive dilution by linear-scan blocks in our
sweep. It is not an anti-hybrid claim. The two architectural bets
address different questions ("can pure nonlinear recurrence scale at
all?" for this paper versus "what does a well-mixed hybrid express?"
for OLMo-Hybrid), and the answers do not contradict.

#heading(level: 2, numbering: none)[Training duration and result scope]

Each 1.27–1.35 B model trains for $tilde 14$ wall-clock days per
architecture; these are the language-modelling results for that
training budget. The racer panel (@fig_lm_racers) records the
loss-vs-wallclock curve as of 2026-05-24. Training continues;
additional rounds will extend the curves.

#heading(level: 2, numbering: none)[Open architectural choices]

Several architecture-internal questions remain open. The output gate, the
non-linearity on the state ($tanh$ vs linear), and the decay
parameterisation (simple sigmoid vs Mamba2-style log-space) all show
loss-only ties at small scale; the strongest evidence for each
production choice is the empirical state-tracking and stability data
rather than a clean ablation at 1.27 B. The production architecture
keeps the conservative settings.

// ── 10. Conclusion ────────────────────────────────────────────────────────────
= Conclusion <sec:conclusion>

The update rule of a nonlinear recurrent network is one design choice
in a space of maps. This paper opens that space, places the first
rule in the order, and supplies the substrate that makes any point in
the space trainable. The Lean 4 development of §7 is the
methodological core: the rules are computational objects, the order
is derivable rather than asserted, and the trusted import closure of
`ElmanProofs.PaperCore` has no
`sorry`/`admit`/`axiom`/`opaque`/`native_decide`.

The empirical demonstration is the systems claim. Three pure-recurrent
architectures (the Emender and M²RNN-CMA, nonlinear in time; Gated
DeltaNet, linear in time) receive per-architecture CMA-ES hyperparameter
search and converge into a shared wallclock band on The Pile.
*Nonlinearity in time is not a cost* for language modelling at this
scale; the choice of recurrence linearity is washed out by
per-architecture tuning. To our knowledge, the Emender and M²RNN-CMA are
the first foundation-model-class pure-nonlinear-recurrent language
models trained at 1.27–1.35 B parameters. M²RNN (Mishra et al.
@m2rnn2026) is the closest prior art and demonstrates nonlinear
matrix-state recurrence at 7 B MoE scale in *hybrid form*; the
pure-recurrent variant trained here, M²RNN-CMA, is the head-to-head
datapoint inside the pure-nonlinear-recurrent class.

The technical discovery that makes pure nonlinear recurrence practical
at scale is *multi-programming*: width-axis parallelism across many
independent small recurrent heads, each carrying its own bounded matrix
state, with the time loop kept serial inside each head. Linear
recurrences gain throughput by *time-axis* linearisation; pure-nonlinear
recurrences gain throughput by replicating the recurrent computation
across thousands of small bounded memory programs that the GPU executes
in parallel. The recipe is update-rule-agnostic: both PNR instances
trained here satisfy the same multi-programming predicate at 1.27 B
(Lean: `multiProgrammed_admits_m2rnn_and_emender`).

Within the pure-nonlinear-recurrent class, the delta-correcting update
rule (the Emender) trains consistently ahead of the raw-write update
rule (M²RNN-CMA) across the sampled wallclock window. The cleanest
within-class probe is the *single-knob ablation in the Emender's own
geometry* (substitute $delta = v - S^T k$ with $delta = v$, re-tune
under matched CMA-ES; protocol documented, infrastructure ready); that
experiment is named as the next probe in §11 / §12. The cross-architecture
gap reported here — the 1.27 B M²RNN-CMA-vs-Emender trail across the
sampled window and the 8 M parameter-matched $S_5$/$S_3$ split — is
corroboration of the formal separation, with the §5 geometry caveat
explicit, not a head-to-head verdict on a single trajectory. The
formal separation itself is unconditional on geometry and on training:
an orthonormal-key Emender configuration realises the $S_5$ tracker
(`EmenderRealizesS5.emender_realizes_s5_tracker`); no fixed-weight
raw-write matrix RNN with row, column, or cell forget gates can match
the Emender's mixed-key delta correction in one recurrent step
(`emender_m2rnn_one_step_resource_separation_embeds`); the gap does
not wash out across $k$ steps for any $k >= 1$ on the constructed
witness alphabet (`emender_m2rnn_k_step_separation`); the per-token
FLOP class is the same for the two PNR instances
(`emender_m2rnn_flop_class_equiv`). The trainability shadow at 8 M:
with capacity non-binding by many orders of magnitude, raw-write stalls
at 0.31 on the six-element solvable-group $S_3$ control while the
delta-correcting update solves $S_3$ to ceiling and reaches 0.79 on
the non-solvable $S_5$ probe.

*Release.* Emender checkpoints — the Emender (this work's delta-correct instance)
and M²RNN-CMA (the CMA-reshaped raw-write instance) — together with
the Gated DeltaNet baseline will be released on HuggingFace at
publication, alongside the per-architecture CMA-ES configurations,
the training protocol, and the Triton multi-programming kernel
source.

// ── 11. Testable Predictions ─────────────────────────────────────────────────
= Testable Predictions <sec:predictions>

The class-level and within-class findings of this paper extend to
predictions testable at the next scale step or under modest extra
training. We state them as predictions so that future
training rounds can falsify them.

#set list(indent: 1em)
- *Width-axis multi-programming scales beyond 1.27 B without throughput
  collapse.* The multi-programmed substrate harvests throughput from
  many small bounded heads rather than from time-axis linearisation;
  the per-head register/SRAM working set is independent of total
  parameter count, so the recipe is expected to survive scale-up to the
  3–7 B band. Falsified if per-token throughput degrades faster than
  the parameter count grows, beyond the kernel-arithmetic baseline.

- *The Emender's state-tracking advantage persists at chinchilla-optimal
  training.* The §6 $S_5$ and $S_3$ gaps are at the 8 M parameter-matched
  probe scale under matched no-tuning. At chinchilla-optimal training
  budget at 1.27 B and beyond, the within-PNR ordering and the
  Emender-vs-GDN gap on state-tracking probes should hold qualitatively.
  Falsified if the $S_5$ or $S_3$ gap closes under chinchilla-optimal
  budgets at matched compute.

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

- *The Triton kernel substrate makes update-rule variants quick to
  prototype.* The fused forward/backward Triton kernel
  (§4) is parameterised by the update map and the gating; a new
  PNR update rule at the same matrix-state signature should be
  expressible by editing the per-step body and re-running the
  multi-programmed time loop. Falsified if a representative new update
  rule (e.g., a higher-order delta with two-step memory) requires
  fundamental kernel rewrites rather than per-step edits.

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

- *Ablating the delta correction out of the Emender's exact geometry
  degrades the run.* The cleanest within-class probe (named in §12):
  take the production Emender at 1.27 B, substitute
  $delta = v - S^T k$ with the raw-write $delta = v$, keep everything
  else fixed, re-tune under matched per-architecture CMA-ES. By
  Theorem set D the comparison is FLOP-matched by construction. The
  prediction is that the resulting `e88-raw` configuration trains
  measurably behind the delta-correcting Emender — both on Pile loss
  at $tilde 14$ days and on the §6 state-tracking probes carried
  down from the 1.27 B shape. Falsified if `e88-raw` matches the
  Emender within fractional-nat loss *and* matches it on $S_5$/$S_3$
  under matched-no-tuning. The protocol is documented; the Triton
  kernel exposes a `RAW_WRITE` flag; the per-architecture CMA-ES
  driver carries an `e88-raw` entry.

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
the order.

#heading(level: 2, numbering: none)[Additional seeds and architecture-internal revalidation]

Each 1.27 B model trains for $tilde 14$ wall-clock days per
architecture, the standard unit of evidence at this scale class;
additional seeds at this band are a multi-week investment per seed.
Several architecture-internal choices in the Emender, including the
output gate, the non-linearity on the state ($tanh$ vs linear), and
the decay parameterisation (simple sigmoid vs Mamba2-style log-space),
show loss-only ties at small scale; revalidation of each at 1.27 B is
open.

#heading(level: 2, numbering: none)[Scale beyond 1.27 B]

The wallclock-band convergence is observed at 1.27 B parameters on
The Pile. Whether the same band convergence holds at larger scale, on
larger corpora, or under longer training is open.

#heading(level: 2, numbering: none)[Cleanest within-class ablation: delta-rule out of the Emender's exact geometry]

The within-class evidence reported in this paper compares *different
architectures* (the Emender vs the CMA-reshaped M²RNN); the geometry
differs even when the parameter count and search effort do not. The
cleanest within-class probe is a *single-knob ablation in the
Emender's own geometry*: take the production Emender stack at 1.27 B
(dim=1664, depth=12, $H=370$, $N=V=32$), substitute the
delta-correcting write $delta = v - S^T k$ with the raw-write
$delta = v$, keep gating, $L^2$ normalisation, decay parameterisation,
optimiser, tokenizer, data, and search budget fixed, and re-tune
under matched per-architecture CMA-ES. By Theorem set D (§7) the two
heads share a common $c_1 d^2 + c_2 d$ per-token FLOP envelope, so
the comparison is FLOP-matched by construction. The protocol is
documented; the Triton kernel exposes a `RAW_WRITE` flag; the
per-architecture CMA-ES driver carries an `e88-raw` model entry. This
is the named next experiment.

A second within-class follow-up — further per-family CMA-ES
generations at 1.27 B beyond the current protocol — would
additionally distinguish "under the per-architecture CMA-ES protocol
used here" from "under any matched search effort".

// ── Appendix A — E63→E88 lineage and ablation notes ────────────────────────────
= Appendix: Lineage of the E63 $arrow$ E88 experimental program <sec:appendix>

The Emender architecture as presented in §3 is the endpoint of a multi-year
ablation lineage. The named milestones referred to here are internal
codenames; they are documented in the appendix for reproducibility of
the design history and are not used in the body of the paper. The
reference implementation of the production Emender stack lives at
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
  products. The L#super[2]-normalised key write inherited by the Emender is the
  fix.

- *E88 (production Emender).* Stable at 1.27 B parameters under
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
