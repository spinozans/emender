# DEEP NOVELTY SYNTHESIS — is the WHOLE situation laid out anywhere, cross-domain?

**Task:** `deep-novelty-synthesis` · model claude:opus · cross-domain literature research, web, no GPU.
**`paper/main.typ` was NOT edited.** Committed, not pushed.

**Citation discipline (ABSOLUTE).** Every work cited below was confirmed to exist via a web
search **this session**, with a stable identifier (arXiv id / DOI / venue URL) fetched and
checked. **Zero fabrication.** Citations were gathered two ways and cross-checked: (i) I
web-verified each load-bearing work myself (the verifier of record), and (ii) one sub-agent
per domain (dynamics-neuro / reservoir / liquid-CT-RNN / ML-arch-synthesis) returned
web-verified citations with an explicit `unverifiable_dropped` list — each agent reported
`web_available = true` and dropped works it could not confirm rather than asserting them. The
one citation that looked riskiest — Siems, Grazzi et al. **arXiv:2602.14814** (Feb 2026) — I
fetched directly and confirmed it is real. The full verified identifier list is in §7.

**Relationship to the first pass.** [`NOVELTY_POSITIONING.md`](NOVELTY_POSITIONING.md) did the
*component-by-component* positioning inside the modern transformer/SSM literature (Hymba,
xLSTM, Grazzi, DeltaProduct, GDN, MoM, Weiss, Were-RNNs, Merrill…) and concluded **NOVEL-SYNTHESIS**.
This document answers the **sharper, different** question the first pass did not:
**is the whole *framing*/*synthesis* already laid out in any key paper(s) — especially OUTSIDE
the modern architecture literature?** The short answer (§5): **the regime-taxonomy is classic
dynamics; the tension is a known reservoir-computing law; the heterogeneous-mixture resolution
has a direct reservoir precedent; only the trainable-placement result and the specific
four-corner matrix-state instantiation are genuinely new.** That is a real reframe and it is
argued in full below.

---

## 0. The synthesis to be located (the four facets)

The artifact's whole situation, decomposed:

- **(a) Gain/eigenvalue structure of a recurrence determines its computational regime.**
  Leaky/fading memory (|eig|<1) · perfect integrator = **counting** (eig=1, marginal) ·
  bistable **latch** (self-loop gain ≥ 1 + saturation) · **reflection** = group-tracking
  (negative/complex eig) — as **REGIONS of ONE parameterized cell**.
- **(b) These regimes are MUTUALLY EXCLUSIVE within a single update rule** — a fundamental
  tension; one gain/eigenvalue setting cannot occupy two regimes at once.
- **(c) The resolution is a HETEROGENEOUS MIXTURE of rules** — a population / compartments /
  multiple co-existing cell-types with different dynamics.
- **(d) The specialization must be PLACED** (by init/architecture), **not learned-from-generic**
  — a generic homogeneous trainable system does not self-organize the needed heterogeneity.

For each facet: the closest prior framing, in which domain, with verified citations, and the
**precise residual gap**. Then the decisive verdict (§5).

---

## 1. Facet (a) — eigenvalue/gain → computational regime, as regions of one cell

### This is CLASSIC. Its home is dynamical-systems / computational neuroscience.

The attractor-landscape view of neural computation has, for three decades, mapped **distinct
computational primitives onto distinct dynamical/eigenvalue regimes of a recurrence**:

| regime (ours) | dynamical-systems object | eigenvalue signature | canonical source |
|---|---|---|---|
| leaky memory | stable point attractor (contracting) | \|eig\|<1 | Hopfield 1982 |
| **counting / integration** | **line attractor** (continuous line of fixed points) | one marginal mode, **eig=1** | **Seung 1996** |
| **bistable latch** | multistable point attractors (energy basins) | self-loop gain ≥ 1 + saturation | **Hopfield 1982** |
| **group / angular tracking** | **ring / continuous-periodic attractor**, moving bump | rotational / complex / odd-symmetry shift mode | **Zhang 1996** |

- **Hopfield 1982** (PNAS 79:2554, `10.1073/pnas.79.8.2554`): symmetric recurrence → **point
  attractors** = content-addressable memory; an energy landscape of multiple basins is exactly
  the **multistable-latch** regime.
- **Seung 1996** (PNAS 93:13339, `10.1073/pnas.93.23.13339`): the oculomotor neural integrator
  is a **line attractor** — a continuous line of fixed points produced by *precisely tuned
  positive feedback* (the marginal **eig=1** mode) performing **perfect temporal integration**
  (i.e. counting/accumulation). This is, verbatim, our "eig=1 line-attractor = counting" corner.
- **Zhang 1996** (J. Neurosci. 16:2112, `10.1523/JNEUROSCI.16-06-02112.1996`): the
  head-direction **ring attractor**; an **odd-symmetry** (asymmetric) weight component induces a
  controlled **rotation** of the bump — the rotational/shift mode underlying angular tracking,
  the closest dynamical analog to our negative/complex-eigenvalue **reflection = group-tracking**
  corner.
- **Sussillo & Barak 2013** (Neural Comp. 25:626, `10.1162/NECO_a_00409`): the *methodology* that
  makes facet (a) literal — find fixed/slow points, read the **Jacobian eigenvalues**: near-zero
  = slow integrating (line/plane attractor), negative-real = contracting/decision, complex =
  rotational. This is the explicit "eigenvalue structure determines the computational regime"
  claim, as a reverse-engineering tool.
- **Mante, Sussillo, Shenoy & Newsome 2013** (Nature 503:78, `10.1038/nature12742`): a trained
  RNN / PFC implements an approximate **line attractor** (slow near-unity mode) for evidence
  integration *plus* a context-dependent **selection vector** — different eigenmodes carry
  different computational roles in one network.
- **Khona & Fiete 2022** (Nat. Rev. Neurosci. 23:744, `10.1038/s41583-022-00642-0`,
  arXiv:2112.03978): **the single closest statement of facet (a)** — an explicit review
  cataloguing **point / line / ring / plane / toroidal** attractors and integrators as a
  **family of dynamical regimes selected by recurrent gain and connectivity**.

The same principle is independently anchored in deep learning by **Pascanu, Mikolov & Bengio
2013** (ICML, arXiv:1211.5063): the **eigenvalues / spectral radius** of the recurrent matrix
set the vanishing-vs-exploding boundary — gain controls the dynamical regime — and in reservoir
computing the spectral radius is *the* knob (§2). The same eigenvalue-region intuition appears,
narrower (2 of 4 regimes), in the 2024–26 linear-RNN cluster: **Grazzi et al.** (arXiv:2411.12537)
make the eigenvalue **range** the lever for fade-vs-track, **DeltaProduct/Siems** (2502.10297) and
**Cirone et al.** (2402.19047) / **Movahedi et al.** (2503.10799) make transition **structure**
(diagonal→dense) the lever, and **Ebrahimi & Memisevic 2025** (arXiv:2505.21749, NeurIPS 2025)
give an explicit **hierarchy** of bilinear state-transition complexity with Mamba/diagonal LRNNs
"residing at the lowest-complexity center."

### Precise residual gap for (a)

The dynamics/neuro literature enumerates the regimes but builds **each by a bespoke construction**
(tuned positive feedback for the line; symmetric Hebbian weights for points; odd weights for the
ring) — it presents them as a **menu of network types**, not as **operating points of ONE
continuously-parameterized scalar gain/eigenvalue knob** swept |eig|<1 → =1 → >1+saturation →
negative/complex. The 2024–26 ML cluster has the "one cell, one knob" shape but covers only **two
of the four** corners (leak and reflection/track, plus the diagonal-dense axis) and **never names
the eig=1 counting / line-attractor regime or the saturating bistable-latch regime** as co-equal
operating points — that vocabulary lives in neuroscience, which in turn lacks the single-knob
framing. **Nobody puts all four corners on one parametric continuum.** Our (a) is therefore a
**re-instantiation of a classic principle in a new substrate** (matrix-state delta-rule / linear-
attention cells), not a new principle.

---

## 2. Facet (b) — mutual exclusivity / the tension

### Strongest match: reservoir computing states it as a conservation LAW.

- **Dambre, Verstraeten, Schrauwen & Massar 2012** (Sci. Rep. 2:514, `10.1038/srep00514`):
  **Information Processing Capacity of Dynamical Systems.** Total computational capacity of a
  fading-memory system is **bounded by the number of linearly independent state variables and is
  conserved**; capacity spent on **nonlinear** functions is *subtracted from* **linear-memory**
  capacity — **you cannot maximize both at once**. This is the closest thing to a clean
  impossibility statement of facet (b) anywhere in any domain.
- **Verstraeten, Dambre, Dutoit & Schrauwen 2010** (IJCNN, `10.1109/IJCNN.2010.5596492`):
  "Memory versus non-linearity in reservoirs" — an **antagonistic trade-off**; a single
  (spectral-radius, input-scaling) operating point cannot maximize memory and nonlinearity
  simultaneously.
- **Inubushi & Yoshimura 2017** (Sci. Rep. 7:10199, `10.1038/s41598-017-10257-6`): via the
  variational equation, the *mechanism* — **nonlinear dynamics intrinsically degrades stored
  memory** — formalizing the mutual exclusivity within one system.
- Edge-of-chaos framing: **Bertschinger & Natschläger 2004** (Neural Comp. 16:1413,
  `10.1162/089976604323057443`) and **Legenstein & Maass 2007** (Neural Networks 20:323,
  `10.1016/j.neunet.2007.04.017`): gain moves a network along an **ordered → chaotic** axis and
  computational performance peaks at the boundary — one gain cannot be simultaneously deep in the
  ordered (memory) and chaotic (nonlinear-mixing) regimes.

### In neuroscience the tension is the "fine-tuning / knife-edge" problem.

- **Seung 1996**: a true integrator requires **eig exactly 1**; detune to <1 → leak/drift, to
  >1 → runaway/bistability. The counting regime is a **knife-edge mutually exclusive** with the
  leaky and latch regimes at one gain.
- **Khona & Fiete 2022**: continuous (line/ring) attractors are **non-generic / fine-tuned** — the
  marginal integrating manifold is destroyed by perturbations that would yield discrete point
  attractors; the integrator and the latch cannot robustly coexist on the same axis.
- **Mastrogiuseppe & Ostojic 2018** (Neuron 99:609, `10.1016/j.neuron.2018.07.003`,
  arXiv:1711.09672): a given low-rank connectivity (overlap geometry + overall gain)
  **deterministically sets ONE dynamical regime** — single fixed point *vs* bistable *vs*
  oscillatory — one minimal structure cannot host two at once.

### In the ML cluster it appears as expressivity-vs-cost, or binary fade-vs-track.

**Grazzi et al.** (2411.12537): the same positive-eigenvalue choice that gives stable fading
memory is precisely what **forbids parity** — the closest ML statement of within-gain mutual
exclusivity, but **binary** (fade vs track), not four-way. **Movahedi** (2503.10799),
**Merrill–Petty–Sabharwal** (2404.08819, the TC⁰ ceiling), and the **Tiezzi et al. survey**
(2406.09062) frame it as **expressivity vs compute cost** (diagonal cheap/weak vs dense
costly/strong), a *fixable design choice*, not an irreducible per-unit dynamical tension.

### Precise residual gap for (b)

The tension **exists and is well-established** — most cleanly as the reservoir **memory-nonlinearity
conservation law** (Dambre 2012). But **no work states it as a four-way mutual exclusivity among
the named regimes** {leak, count, latch, track} *within one update rule*. Reservoir computing lumps
all temporal/nonlinear structure into a single "nonlinear capacity" bucket and trades it against a
single "linear-memory" bucket (2 regimes); neuro frames it as a robustness nuisance (attractors are
fragile); ML frames it as a compute-cost design choice. Our (b) is a **sharpening and re-naming** of
the reservoir law into a four-corner exclusion — not a newly discovered tension.

---

## 3. Facet (c) — heterogeneous mixture as the resolution

### The single closest cross-domain precedent: the mixture reservoir.

- **Inubushi & Yoshimura 2017** (Sci. Rep. 7:10199, `10.1038/s41598-017-10257-6`): having
  *proved the mechanism* of the memory-nonlinearity trade-off, they **propose a mixture reservoir
  endowed with BOTH linear and nonlinear dynamics**, showing the heterogeneous mixture **beats
  either pure regime**. This is facet (b)→(c) **in a single paper**: tension identified → resolved
  by a heterogeneous mixture of dynamical rules. It is the tightest "whole-situation-in-miniature"
  found in any domain.
- **Tanaka, Matsumori, Yoshida & Aihara 2022** (Phys. Rev. Research 4:L032014,
  `10.1103/PhysRevResearch.4.L032014`, arXiv:2108.09446): a reservoir of **heterogeneous
  leaky-integrator neurons spanning diverse timescales** (a population of distinct effective decay
  rates) outperforms a homogeneous-timescale reservoir on multiscale dynamics.
- **Dubreuil, Valente, Beiran, Mastrogiuseppe & Ostojic 2022** (Nat. Neurosci. 25:783,
  `10.1038/s41593-022-01088-4`): flexible input-output computation requires **multiple non-random
  subpopulations** (cell-type-like connectivity clusters) co-existing in one network; a single
  homogeneous population cannot produce the flexible gain-modulated regime switching.
- **Khona & Fiete 2022**: explicitly advocates **recombining multiple attractor modules** (point +
  line + ring …) to increase a circuit's flexibility — heterogeneous modules co-hosted.
- **Lechner et al. 2020 — Neural Circuit Policies** (Nat. Mach. Intell. 2:642,
  `10.1038/s42256-020-00237-3`): a structured population of **distinct typed continuous-time
  cells** (sensory / inter / command / motor) co-existing in one controller — the right *shape* of
  resolution (a typed mixture of compartments).
- ML analogue: **MoM** (arXiv:2502.13685) and **Linear-MoE** (arXiv:2503.05447) put multiple
  recurrent **memories/experts** in one layer.

### Precise residual gap for (c)

Every precedent mixes along a **narrower axis than the four-corner repertoire**: Inubushi mixes
**linear vs nonlinear** (2 regimes); Tanaka and the multi-timescale literature mix **decay rates /
timescales** (all within the leaky |eig|<1 regime); NCP mixes **anatomical role / wiring**, not
eigenvalue regime; Dubreuil mixes **connectivity-statistics subpopulations**, not per-unit
gain-regimes; MoM/Linear-MoE mix **homogeneous** memories/experts (same update rule). **No prior
work assembles a population whose cell-types are pinned to the distinct eigenvalue regimes**
{leaky · eig=1 counter · bistable latch · reflection tracker} — "one cell-type per computational
corner." The mixture *principle* is old (reservoir computing owns it); the **four-corner
eigenvalue-regime mixture drawn from one parameterized cell** is not stated.

---

## 4. Facet (d) — placement, not emergence

### This is the most distinctive facet — and the cross-domain evidence is genuinely split.

**Where placement is supported (but trivially or narrowly):**
- **Reservoir computing makes placement structurally inevitable**: reservoirs are **untrained**, so
  *any* heterogeneity (Inubushi's linear units, Tanaka's diverse timescales) is **installed at init
  by definition**. RC **sidesteps learning** rather than demonstrating that a *trainable*
  homogeneous system fails. **Carroll 2020** (Chaos 30:121109, `10.1063/5.0038163`,
  arXiv:2012.01409) even shows there is **no universal self-organized optimum** — the best regime
  is task- and construction-dependent — undercutting "a generic reservoir self-tunes to the need."
- **CT-RNN**: **Beer 2006** (Neural Comp. 18:3009, `10.1162/neco.2006.18.12.3009`) shows most CTRNN
  parameter space is dynamically trivial/saturated and the rich regimes occupy bounded regions
  (motivating "place the active regime by init," the center-crossing idea); **NCP/Lechner** fixes
  neuron types and wiring **by construction** before training.
- **ML cluster**: the whole DeltaProduct/Grazzi/Movahedi/Ebrahimi-Memisevic program presumes the
  capability **must be engineered into the spectrum/structure** — generic diagonal Mamba "resides
  at the lowest-complexity center" and does not reach state-tracking; **Siems, Grazzi et al. 2026**
  (arXiv:**2602.14814**, *Learning State-Tracking from Code Using Linear RNNs*) shows empirically
  that **only DeltaNet with extended (negative) eigenvalues** learns state-tracking while standard
  models fail to generalize — capability follows from the chosen spectrum, not generic training.

**Where the literature CUTS AGAINST our placement claim (important, and honest):**
- **Dubreuil et al. 2022** finds the necessary subpopulation structure can be **DISCOVERED by
  training** when the task forces it — i.e. closer to "learnable-when-demanded" than "must be
  placed."
- **Maheswaranathan, Williams, Golub, Ganguli & Sussillo 2019** (NeurIPS, arXiv:1907.08549):
  fixed-point **topology is universal** across architectures for a given task — the **task, not the
  architecture, determines the realized regime**, suggesting the needed regime is **reliably
  reached by training**.
- **Yang, Joglekar, Song, Newsome & Wang 2019** (Nat. Neurosci. 22:297, `10.1038/s41593-018-0310-2`):
  training one RNN on 20 cognitive tasks makes recurrent units **self-organize into functionally
  specialized clusters** — heterogeneity that **EMERGES** from generic training, the *opposite* of
  placement-required.

### Precise residual gap for (d)

**No domain makes our specific claim**: that a generic **homogeneous, trainable** matrix-recurrence
**fails to self-organize** the multi-corner regime mixture, so each gain-regime must be **placed**
(init-spread + knob-specific LR), with a controlled generic-fails / spread-holds ablation. RC's
placement is a definitional artifact of not training; ML's "design it in" is about getting **one**
cell to **one** regime, never about manufacturing a **mixture**; and the neuro evidence (Yang 2019;
Maheswaranathan 2019; Dubreuil's "discoverable") actually **leans toward emergence**. **Facet (d) is
where the artifact is most clearly un-precedented — and notably it is a result that runs *against*
the prevailing computational-neuroscience expectation that functional heterogeneity emerges from
generic training.** That tension with the neuro literature is itself worth stating in the paper.

---

## 5. Decisive verdict — is the WHOLE situation laid out anywhere?

**No single paper, and no single tight cluster, lays out all four facets together.** But the
honest, sharper truth is that **most of the synthesis is a re-instantiation of older theory**, and
only a minority is new:

**Two clusters each own roughly half of it:**

1. **The dynamical-systems / attractor-landscape cluster** — *Khona & Fiete 2022* (the taxonomy)
   + *Hopfield 1982 / Seung 1996 / Zhang 1996* (the four corners) + *Sussillo & Barak 2013 /
   Mante 2013* (the eigenvalue-reading method) + *Mastrogiuseppe-Ostojic 2018 / Dubreuil 2022*
   (structure→regime; subpopulation mixture). This cluster owns **(a)** fully, **(c)** well, **(b)**
   weakly (as fine-tuning), and **leans against (d)** (heterogeneity emerges). It treats regimes as
   a **menu of network types**, not operating points of one parameterized cell, and lacks the
   learnability-failure result.

2. **The reservoir-computing cluster** — *Dambre 2012* (the tension as a conservation **law**) +
   *Inubushi & Yoshimura 2017* (the **linear+nonlinear mixture** resolution) + *Tanaka 2022*
   (diverse-timescale mixture). This cluster owns **(b)** as a clean law, **(c)** as an explicit
   mixture-of-rules, and **(d)** implicitly (untrained ⇒ placed). It is the **tightest single
   (b)+(c)+implicit-(d) pairing in any literature** — but only along the **linear-vs-nonlinear /
   timescale axis (2 regimes, not 4)**, with no count/latch/track naming and no trainable-placement
   claim.

**The single closest "whole situation in miniature" is the reservoir tradeoff→mixture pair:
Dambre 2012 (tension as law) + Inubushi & Yoshimura 2017 (heterogeneous mixture of dynamical rules
as the resolution).** If a reviewer asks "isn't this just an old idea?", *that* is the pair they
will cite. It is **(b)+(c)+(d-implicit) in one breath** — just on one axis, in fixed reservoirs,
not in trained matrix-state LMs, and without the four-corner eigenvalue map.

### Plainly: the regime-taxonomy IS classic dynamics.

**The "gain/eigenvalue structure → computational regime" principle (facet a) is a classic,
canonical result of dynamical-systems / computational neuroscience (Hopfield, Seung, Zhang,
Sussillo–Barak; reviewed in Khona & Fiete 2022) and of reservoir computing (spectral radius / edge
of chaos).** This **reframes the contribution** exactly as the task anticipated: the artifact did
**not "discover the tension"** — it **instantiated a known dynamical principle** in input-dependent
**matrix-state** recurrences (delta-rule / linear-attention) **at LM scale**, with the
**learned heterogeneous mixture** and the **decay-clamp-lock + init-spread/knob-LR placement** as
the new parts.

---

## 6. Genuinely un-precedented vs re-instantiation of older theory

**Re-instantiation of older theory (must be attributed; do NOT claim as discovered):**
- **(a) the regime taxonomy itself** — point=latch / line=counting / ring=tracking is **classic
  attractor neuroscience** (Hopfield 1982; Seung 1996; Zhang 1996; Khona & Fiete 2022) and the
  spectral-radius regime knob is **classic reservoir computing** (Jaeger; Bertschinger–Natschläger
  2004; Legenstein–Maass 2007) and classic RNN training theory (Pascanu 2013).
- **(b) the tension** — the **memory-nonlinearity conservation law** (Dambre 2012; Verstraeten 2010;
  Inubushi 2017) and the continuous-attractor **fine-tuning knife-edge** (Seung 1996; Khona & Fiete
  2022; Mastrogiuseppe–Ostojic 2018). We **sharpen and re-name** it; we did not find it.
- **(c) resolution-by-heterogeneous-mixture** — the **mixture reservoir** (Inubushi & Yoshimura
  2017), diverse-timescale reservoirs (Tanaka 2022), and multi-subpopulation circuits (Dubreuil
  2022; Khona & Fiete's "recombine modules"). The *principle* of mixing dynamical rules to beat the
  single-rule tradeoff is **already owned by reservoir computing**.

**Genuinely new after this deeper search (defensible as contribution):**
1. **The specific four-corner repertoire {leaky · counting(line-attractor, eig=1) · latch(point
   attractor) · tracking(reflection, negative eig)} as operating points of ONE parameterized
   matrix-state delta-rule cell**, with the **SSM-inherited (0,1) decay clamp named as the lock**
   that forecloses the count and latch corners. No prior work places all four on one knob, and none
   does it in a matrix-state input-dependent recurrence.
2. **A LEARNED heterogeneous mixture of these four UPDATE RULES within one layer** — distinct from
   the reservoir's *untrained* mixture, from MoM/Linear-MoE's *homogeneous* memories/experts, from
   NCP's *anatomical-role* typing, and from Hymba's *attention+SSM* pairing.
3. **The placement-not-emergence learnability-failure result for a TRAINABLE recurrence** (generic
   init collapses to one regime; init-spread + knob-LR holds all four). This is the **most
   distinctive** piece: reservoir computing sidesteps learning, and the neuro literature (Yang 2019;
   Maheswaranathan 2019; Dubreuil 2022) **leans the other way** (heterogeneity emerges). Stating —
   and demonstrating — that it does **not** emerge for a generic trainable matrix-recurrence is a
   genuine, somewhat counter-prevailing contribution.
4. **The eigenvalue-causal demonstration on real LM-scale delta-rule models** (GDN+neg creates
   length-robust S₅; clamp destroys it) — a causal confirmation of the negative-eigenvalue
   mechanism on trained models at scale, on the paper's own architectures
   (see [`EIGENVALUE_CAUSAL_TEST.md`](EIGENVALUE_CAUSAL_TEST.md),
   [`S5_MECHANISM_SYNTHESIS.md`](S5_MECHANISM_SYNTHESIS.md)).

**Strongest honest framing for the paper:** *"We instantiate the classic attractor-regime taxonomy
(point=latch / line-attractor=counting / reflection=group-tracking) — long known in dynamical
neuroscience and mirrored by the reservoir memory-nonlinearity law — inside a single parameterized
matrix-state recurrence, and show that (i) the four classical capabilities are operating points of
one (gain, correction, nonlinearity) cell with the (0,1) decay clamp as the lock, and (ii) unlike
the untrained reservoir mixture and contrary to the rate-RNN expectation that functional
heterogeneity emerges, a trainable cell requires explicit placement (init-spread + knob-LR) to hold
a stable heterogeneous mixture-of-specialists across all four corners."* Cite Khona & Fiete 2022,
Seung 1996, Dambre 2012, and Inubushi & Yoshimura 2017 as the prior framings; claim only (1)–(4).

---

## 7. Prioritized READING LIST for Erik — the papers that, together, ARE the situation

The 5 in **bold** are the must-reads: together they *are* the prior synthesis. The remainder
complete each facet.

1. **Khona & Fiete 2022 — "Attractor and integrator networks in the brain."** *Nat. Rev. Neurosci.*
   23:744-766. `10.1038/s41583-022-00642-0` (arXiv:2112.03978). — **Facet (a), the master
   taxonomy:** point/line/ring/plane/toroidal attractors as a family of regimes selected by gain and
   connectivity, with the fine-tuning tension (b) and "recombine modules" mixture (c). *The single
   closest paper to our whole framing.*
2. **Dambre, Verstraeten, Schrauwen & Massar 2012 — "Information Processing Capacity of Dynamical
   Systems."** *Sci. Rep.* 2:514. `10.1038/srep00514`. — **Facet (b) as a conservation LAW:**
   linear-memory and nonlinear-processing capacity trade off within a conserved budget. The cleanest
   impossibility statement of the tension anywhere.
3. **Inubushi & Yoshimura 2017 — "Reservoir Computing Beyond Memory-Nonlinearity Trade-off."**
   *Sci. Rep.* 7:10199. `10.1038/s41598-017-10257-6`. — **Facets (b)→(c) in one paper:** proves the
   trade-off mechanism, then resolves it with a **heterogeneous (linear+nonlinear) mixture
   reservoir**. The closest cross-domain precedent for "mixture of rules beats one rule."
4. **Seung 1996 — "How the brain keeps the eyes still."** *PNAS* 93:13339-13344.
   `10.1073/pnas.93.23.13339`. — **Facet (a) counting corner + (b) knife-edge:** the line attractor
   (eig=1) as a perfect integrator, mutually exclusive with leak (<1) and runaway (>1).
5. **Sussillo & Barak 2013 — "Opening the black box."** *Neural Comp.* 25:626-649.
   `10.1162/NECO_a_00409`. — **Facet (a), the method:** read Jacobian eigenvalues at fixed/slow
   points to identify which regime carries the computation (near-0 = integrate, contracting =
   decide, complex = rotate).

Supporting / facet-completing:

6. Hopfield 1982 — *PNAS* 79:2554. `10.1073/pnas.79.8.2554`. — (a) point-attractor latch / energy
   landscape; the multistable-memory corner.
7. Zhang 1996 — *J. Neurosci.* 16:2112. `10.1523/JNEUROSCI.16-06-02112.1996`. — (a) ring attractor;
   odd-weight rotation = the group-tracking/reflection corner.
8. Dubreuil, Valente, Beiran, Mastrogiuseppe & Ostojic 2022 — *Nat. Neurosci.* 25:783.
   `10.1038/s41593-022-01088-4`. — (c)/(d) heterogeneous subpopulations are needed — but can be
   *discovered* by training (the counterpoint to our placement claim).
9. Yang, Joglekar, Song, Newsome & Wang 2019 — *Nat. Neurosci.* 22:297. `10.1038/s41593-018-0310-2`.
   — (d) **counterpoint:** functional clusters *emerge* from generic multitask training; our
   placement-required result runs against this.
10. Inubushi/Tanaka diverse-timescale + edge-of-chaos pair — Tanaka et al. 2022 *PRR* 4:L032014
    (`10.1103/PhysRevResearch.4.L032014`); Bertschinger & Natschläger 2004 *Neural Comp.* 16:1413
    (`10.1162/089976604323057443`); Carroll 2020 *Chaos* 30:121109 (`10.1063/5.0038163`). — (a)/(c)
    spectral-radius/edge-of-chaos as the gain knob and timescale-mixture; Carroll = (d) counterpoint
    (no universal self-organized optimum).
11. For the modern-architecture bridge (already in [`NOVELTY_POSITIONING.md`](NOVELTY_POSITIONING.md)):
    Grazzi et al. 2024 (arXiv:2411.12537), Ebrahimi & Memisevic 2025 (arXiv:2505.21749), Merrill–Petty–
    Sabharwal 2024 (arXiv:2404.08819), Siems–Grazzi et al. 2026 (arXiv:2602.14814). — the 2-of-4-regime
    eigenvalue/structure lever and the "engineer the spectrum" (single-cell placement) precedents.

### Full verified-identifier appendix (every id confirmed via web this session)

*Dynamics / neuro:* Hopfield 1982 `10.1073/pnas.79.8.2554` · Seung 1996 `10.1073/pnas.93.23.13339`
· Zhang 1996 `10.1523/JNEUROSCI.16-06-02112.1996` · Sussillo & Barak 2013 `10.1162/NECO_a_00409` ·
Mante et al. 2013 `10.1038/nature12742` · Maheswaranathan et al. 2019 arXiv:1907.08549 ·
Vyas, Golub, Sussillo & Shenoy 2020 `10.1146/annurev-neuro-092619-094115` · Khona & Fiete 2022
`10.1038/s41583-022-00642-0` (arXiv:2112.03978) · Mastrogiuseppe & Ostojic 2018
`10.1016/j.neuron.2018.07.003` (arXiv:1711.09672) · Dubreuil et al. 2022 `10.1038/s41593-022-01088-4`
· Yang et al. 2019 `10.1038/s41593-018-0310-2` · Stroud, Porter, Hennequin & Vogels 2018
`10.1038/s41593-018-0276-0`.
*Reservoir:* Jaeger 2001 ESN tech report (GMD 148; ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf) ·
Bertschinger & Natschläger 2004 `10.1162/089976604323057443` · Legenstein & Maass 2007
`10.1016/j.neunet.2007.04.017` · Verstraeten et al. 2010 `10.1109/IJCNN.2010.5596492` · Dambre et al.
2012 `10.1038/srep00514` · Inubushi & Yoshimura 2017 `10.1038/s41598-017-10257-6` · Tanaka et al.
2022 `10.1103/PhysRevResearch.4.L032014` (arXiv:2108.09446) · Carroll 2020 `10.1063/5.0038163`
(arXiv:2012.01409).
*Liquid / CT-RNN:* Beer 1995 `10.1177/105971239500300405` · Beer 2006 `10.1162/neco.2006.18.12.3009`
· Hasani et al. 2021 (LTC) AAAI 35(9):7657 (arXiv:2006.04439) · Hasani et al. 2022 (CfC)
`10.1038/s42256-022-00556-7` (arXiv:2106.13898) · Lechner et al. 2020 (NCP)
`10.1038/s42256-020-00237-3` · Rusch & Mishra 2021 (coRNN) arXiv:2010.00951 · Rusch & Mishra 2021
(UnICORNN) arXiv:2103.05487 · Erichson et al. 2021 (Lipschitz RNN) arXiv:2006.12070.
*ML-arch synthesis:* Pascanu, Mikolov & Bengio 2013 arXiv:1211.5063 · Merrill, Petty & Sabharwal 2024
arXiv:2404.08819 · Grazzi et al. 2024 arXiv:2411.12537 · Siems et al. (DeltaProduct) 2025
arXiv:2502.10297 · Cirone et al. 2024 arXiv:2402.19047 · Movahedi et al. 2025 arXiv:2503.10799 ·
Ebrahimi & Memisevic 2025 arXiv:2505.21749 · Tiezzi et al. 2024 arXiv:2406.09062 · MoM (Du, Qin, Sun
et al.) 2025 arXiv:2502.13685 · Linear-MoE (Sun et al.) 2025 arXiv:2503.05447 · Siems, Grazzi et al.
2026 arXiv:2602.14814.

*Works seen but deliberately NOT asserted (honest gaps):* Maass, Natschläger & Markram 2002 (LSM,
"Real-time computing without stable states," *Neural Comp.* 14(11):2531) — title/venue surfaced in
search but the DOI was not independently fetched, so it is mentioned, not cited with an id;
Gallicchio & Micheli deep-ESN multiple-timescale papers (real, relevant to (c)) — exact DOIs not each
fetched this session, so not cited with ids; Funahashi & Nakamura 1993 (CTRNN universal approximation)
— verified real but omitted because it speaks to none of the four regime-taxonomy facets (and a
universal-approximation result cuts *against* a hard regime-exclusivity reading). No id above was
asserted without a matching web hit this session.

---

## 8. Validation against task checklist

- [x] **All four domains searched** — dynamical-systems/neuro attractor-landscape (§1, parts of
  §2–4), reservoir edge-of-chaos/spectral-radius (§2–4), liquid/CT-RNN (§1,§4, appendix),
  ML-architecture synthesis (§1–4). **Real, web-verifiable citations; zero fabrication** — every id
  in §7 confirmed via web this session (sub-agents + my own verification of all load-bearing works,
  including the flagged arXiv:2602.14814); honest non-asserted gaps listed.
- [x] **Each of (a)–(d) positioned** with closest prior framing, domain, citations, and **precise
  residual gap** (§1–§4); **explicit verdict** that the whole synthesis is **not** laid out in one
  paper or tight cluster, with the two half-clusters and the single closest miniature identified (§5).
- [x] **Honest reframe** that the regime-taxonomy (a) is **classic dynamics/neuro/reservoir**, the
  tension (b) is the known **memory-nonlinearity law**, and resolution-by-mixture (c) has a **direct
  reservoir precedent** (Inubushi 2017); **what is genuinely new vs re-instantiation** spelled out
  (§6) — new = the four-corner matrix-state cell, the *learned* heterogeneous-update-rule mixture,
  the *trainable* placement-not-emergence result (which runs against the neuro emergence view), and
  the LM-scale eigenvalue-causal demo.
- [x] **Prioritized reading list** for Erik (§7), 5 must-reads + facet-completers, one line each.
- [x] **`DEEP_NOVELTY_SYNTHESIS.md` committed; `main.typ` untouched; not pushed.**
