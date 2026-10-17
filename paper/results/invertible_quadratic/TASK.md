# Stage 1: invertible two-coordinate quadratic tracking

Status: **CPU task validation only; no training or GPU work authorized here.**
The operator's 2026-10-16 ruling supersedes the old one-coordinate mod-256
benchmark and the proposed odd-prime-only repair. This document governs this
new package; historical claims in the old task/handoff are not its framing.

## Task, target, and input flow

For an odd prime p, state s=(x,y) in F_p² and iid uniform drivers c in F_p,

    F_c(x,y) = (y, x + y² + c) mod p.
    F_c^{-1}(u,v) = (v - u² - c, u) mod p.

Thus every step is a permutation of ALL p² states for every fixed c. Distinct
full states cannot merge under any shared driver sequence. Primality alone did
not fix the old x'=x²+c map: with full-field drivers that map loses information.

`tasks/invertible_quadratic.py` implements the task and explicit inverse.
The generic `generate_batch(B,T,rng)` interface counts **total input tokens**:

| Position | Input | Target | Loss |
|---|---|---|---|
| 0 | p+x0 | placeholder 0 | masked |
| 1 | 2p+y0 | placeholder 0 | masked |
| t+1, t>=1 | c_t in [0,p) | p*x_t+y_t | active |

Initial coordinates are sampled independently and uniformly and supplied only
once. No intermediate target state or previous model prediction is an input.
The two typed initial-coordinate ranges are disjoint from driver tokens. The
joint class `p*x+y` is a lossless encoding of **both** coordinates; a correct
coordinate or marginal prediction alone is not full-state success. Existing
single-head cross-entropy/argmax metrics therefore score the exact full state
without altering trainers or creating an autoregressive copying path. Token
IDs may overlap between output classes and inputs; this is not feedback.
Vocabulary size is max(p²,3p). The two masked prefix positions do not contribute
to training loss or accuracy.

Registry names:
- `invertible_quadratic` (recommended production p=61)
- `invertible_quadratic_p13`, `invertible_quadratic_p31` (pilots)
- each also has a `_one_step` variant (`invertible_quadratic_one_step` for p=61).

One-step variants require T=3: only (x0,y0),c1 -> (x1,y1). They are designed to
test whether the modular arithmetic/encoding is learnable without recurrent
depth. Stage 1 verifies their oracle labels, **not learned performance**. A model
that fails this diagnostic has failed the task family; its long-depth failure
cannot be attributed to temporal composition.

## Formal suffix-limit lemma

**Lemma.** Fix any initial state s0, any suffix length k>=0, and t>=k+2. Let
C1,...,Ct be independent uniform field elements. Conditional on s0 and any
observed values of the last k drivers, S_t is uniform on F_p². Consequently any
predictor with access only to that initial state and driver suffix (and
independent private randomness) has full-state accuracy **exactly 1/p²**. This
is the Bayes-optimal cap, not merely the accuracy of a particular heuristic.

**Proof.** From any (x,y), two fresh drivers a,b give

    u = x + y² + a,
    v = y + u² + b,
    S_after_two = (u,v).

For every prescribed (u,v) there is exactly one driver pair
`a=u-x-y²`, `b=v-y-u²`. Hence uniform (a,b) yields uniform state, regardless of
the preceding state. Each subsequent fixed-driver F_c is bijective and
preserves uniform measure. There are at least two hidden drivers before the
observed suffix; conditioning on the independent suffix leaves that hidden
state uniform. Applying its known permutations preserves uniformity. Every
conditional guess therefore succeeds with probability 1/p². QED.

**Boundary:** with only one hidden driver the support has p states, not p²;
with none the state is computable from the known initialization and suffix.
The lemma does not apply to the first k+1 driver positions of running accuracy,
to predictors remembering extra prefix summaries, or to trained recurrent
models which can retain the entire history. Report post-burn-in suffix accuracy
separately from any running-average task metric.

`SuffixBaselinePredictor` propagates canonical (0,0) through only the observed
suffix: a legitimate deterministic MAP tie-break of the uniform posterior. Its
API cannot inspect targets or omitted drivers. Exact small-field tests condition
on **every initial state**, enumerate both hidden drivers, and verify one correct
guess out of p² after suffixes of lengths 0, 1, and 7.

## Old parity-plus-suffix calibration

For old `z'=z²+c mod 256`, parity obeys `z' mod 2 = (z+c) mod 2`.
Two states with the same parity have an even difference. Each common squaring
step multiplies that difference by their even sum, raising its 2-adic valuation
by at least one (or making it zero). After seven such steps, the difference is
zero modulo 256. Thus a single parity summary of the omitted prefix plus the
last seven drivers exactly recovers every old state. `OldParitySuffixBaseline`
implements this, including short prefixes and the original z0=1 convention;
tests compare every position of 4,096 length-128 old trajectories and randomized
initializations with the direct recurrence.

On the new task, the explicitly named `predict_invertible` negative control
transfers that old one-bit summary to representative state (bit,0), then applies
the new map to the last seven drivers. It does **not** use actual hidden new
states or an oracle parity of them. Fixed-initial-state exhaustive two-driver
checks show p² distinct true states but at most two predictions from this old
summary: it cannot reproduce new trajectories. Its Monte Carlo accuracy below
is near 1/p². This is an **empirical calibration**, not an extension of the exact
suffix lemma to observers with an additional prefix bit: such an observer can
in general beat 1/p² (counting its two possible summary values only bounds
this one-bit case by 2/p²). We make no universal lower-bound claim about all prefix summaries.

## CPU pilot and empirical protocol

Run on a non-Frontier host (Python 3.12+, NumPy, pytest):

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m pytest -q tests/test_invertible_quadratic.py
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.verify_invertible_quadratic \
  --output experiments/expressivity_tasks/invertible_quadratic_cpu.json
```

On Frontier first source `scripts/frontier/activate_emender_frontier.sh`, then
replace `/usr/bin/python3` with `"$EMENDER_PYTHON"`. No Slurm submissions or GPU
inspection/use are part of this protocol. These task checks do not change
resilient DiLoCo behavior; its requirement matrix is not applicable here.

The retained JSON contains 2,000,000 independent trajectories per p, each with
fixed **known** initial state (1,1) and 15 iid full-field drivers. Each contributes
one terminal prediction, not many correlated running positions. Suffix widths
0/1/7 leave at least two hidden drivers (eight for width seven). Seeds are
20261016+p. Six-sigma binomial consistency checks are predeclared fail-closed
smoke checks, not a proof or a fitted threshold. Marginal driver/initial-state
uniformity is separately tested on 200,000 independent samples per field; small
fields receive exhaustive inverse/bijection tests and larger fields 10,000
random inverse round trips. Production collision tests sample distinct states
without replacement and share drivers, including length-128 sequences.

The same runner validates data/oracle pilots at recurrence depths 128/512/1024
for p=13,31,61 and 10,000 one-step diagnostic examples for each. These are CPU
data pilots only: no optimizer updates, model probes, or learnability findings.
The table below is generated from `invertible_quadratic_cpu.json`; all entries
are accuracy fractions, not percentages. `X1-report.md` at the repository root
summarizes the validation evidence.

| p | Exact 1/p² cap | Suffix 0 | Suffix 1 | Suffix 7 | Old parity transfer |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.0059171598 | 0.0059225 | 0.0058780 | 0.0060020 | 0.0059405 |
| 31 | 0.0010405827 | 0.0010730 | 0.0010340 | 0.0010190 | 0.0009750 |
| 61 | 0.0002687450 | 0.0002650 | 0.0002735 | 0.0002815 | 0.0002955 |

## Production recommendation and length schedule

Recommend **p=61**, subject to operator approval of Stage 2:

| Role | p | Full states | Full-state chance | Joint readout classes |
|---|---:|---:|---:|---:|
| arithmetic/data pilot | 13 | 169 | 1/169 | 169 |
| intermediate pilot | 31 | 961 | 1/961 | 961 |
| production recommendation | 61 | 3,721 | 1/3,721 | 3,721 |

This is already >14 times the old 256-state space, while a joint head at dim1024
has about 3.81M weights (excluding bias). p=127 would require 16,129 classes;
p=257 would require 66,049. Starting there would confound arithmetic learning
and output-head cost with depth. p=61 is a practical recommendation, **not** an
empirically established learnability threshold or a claim that its one-hot
representation is infeasible.

Mirror the old benchmark's **128-driver training depth**, with primary held-out
extrapolation T -> 4T -> 8T = 128 -> 512 -> 1024 drivers; optional secondary
panels at 256/2048/4096 drivers retain the old long-length grid. Add two prefix
tokens at every length: actual `seq_len` values are 130/514/1026 (optional
258/2050/4098). Never multiply the initialization overhead as recurrent depth.
One-step diagnostic uses one driver / three tokens. Full proposed training
protocol is in `INVERTIBLE_QUADRATIC_STAGE2_PLAN.md`; nothing there is launched.

## Interpretation constraints

This benchmark tests whether nonlinear recurrence **learns the computation more
efficiently at practical state sizes**. Every finite-state task admits a
sufficiently large one-hot linear-state representation: here each driver is a
p²-by-p² permutation matrix acting on the state one-hot. No impossibility,
universal expressivity separation, or inability-of-linear-state claims follow.
Invertibility rules out state merging in the task, not all potential shortcuts
or optimization confounds. No CMA-causal explanation of GDN2's previous failure
has been established; none is claimed. Best-checkpoint comparisons, equal-step
comparisons, and the planned same-geometry tanh-removal ablation must precede
any cautious mechanism or efficiency interpretation.
