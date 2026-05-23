# NDM Paper Notes

NDM means **Nonlinear Delta Memory**.

These notes collect the current paper spine, reviewer feedback, formal boundary,
and empirical story. They are a working document, not polished prose.

## Current Thesis

Pure nonlinear recurrent models can be trained at serious scale when the
recurrent state update is the right resource. The decisive ingredient is not
matrix state alone and not temporal nonlinearity alone. The evidence points to
the **delta-correcting matrix-state update**.

E88 is the current production implementation lineage of NDM. The paper should
present NDM as the model family and E88 as an implementation/configuration, not
as the core scientific name.

## Claim Register

Say:

- NDM/E88 and M2RNN are distinct one-step transition families.
- M2RNN's fixed-right raw-write update cannot implement NDM's mixed-key delta
  correction in one step without an extra read-then-delta path.
- With that extra path, M2RNN can embed one NDM step. This is a resource
  separation, not an absolute computability separation.
- Fixed-width, finite-precision recurrent recognizers are finite-state. The
  ceiling is regular languages, hence inside NC1.
- S5 permutation composition is the right non-solvable witness task; parity
  and modular counters are useful solvable-control probes.
- The S5 tracker and exact finite transition-memory realization are now
  machine-checked.
- The empirical S5 curve supports the delta-correction mechanism: E88 wins,
  M2RNN underperforms despite being nonlinear and matrix-state.

Do not say:

- NDM/E88 exceeds NC1 at fixed width and fixed precision.
- This is the first nonlinear matrix-state RNN.
- The Lean core proves a linear/scan lower bound for S5.
- Nonlinearity in general is enough. M2RNN is the counterexample in our own
  data.

## Formal Core

The trusted proof surface is enforced by:

```bash
scripts/check_paper_core.sh
scripts/check_trusted_no_placeholders.sh ElmanProofs.lean
```

Current state:

- `check_paper_core.sh` passes.
- The paper core rejects `sorry`, `admit`, explicit `axiom`, `opaque`, and
  `native_decide` in the trusted import closure.
- Old TC0/parity files are explicitly marked historical sketches.

Trusted content:

- NDM/M2RNN update-family resource separation.
- M2RNN read-then-delta embedding of one NDM step.
- NDM/E88 1.27B many-program production geometry.
- S5 witness facts: S3 has 6 states, S5 has 120 states, S5 is non-solvable.
- Explicit S5 tracker over adjacent transpositions.
- Proof that the Python tuple-swap update matches the Lean tracker transition.
- Exact finite transition-memory realization for every finite online
  recognizer and specifically for S5, with `120 * 4 = 480` transition keys.

Still external or unproven:

Theory gaps:

- Barrington / NC1-completeness.
- Linear/scan/solvable lower bound for S5.
- Trained real-valued NDM learns the exact table robustly.

Implementation/formal-hygiene gaps:

- Exact Python `itertools.permutations` lexicographic class-id numbering.
- Separate proof that adjacent transpositions generate all of S5.

## Reviewer-Calibrated Response

Use this register:

> The trusted formal core is green. The formal contribution is a checked
> NDM-vs-M2RNN update-rule resource separation plus an honest finite-state
> ceiling. Since the initial review, we also formalized the S5 tracker and an
> exact finite transition-memory realization. We do not claim a Lean lower
> bound for linear scan models. The S5 separation is currently empirical,
> backed by a checked task formalization.

This is the right tone: honest, aggressive, and hard to unwind.

## S5 Empirical Core

Current 8M matched S3/S5 run, three seeds, train length 128:

| Task | Model | Mean acc | Min | Max | Baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| S3 | E88/NDM | 1.0000 | 0.9999 | 1.0000 | 0.1667 |
| S3 | FLA-GDN | 0.7185 | 0.6122 | 0.8516 | 0.1667 |
| S3 | M2RNN | 0.3124 | 0.2742 | 0.3529 | 0.1667 |
| S3 | M2RNN-paper | 0.3773 | 0.3669 | 0.3855 | 0.1667 |
| S5 | E88/NDM | 0.7918 | 0.6232 | 0.8880 | 0.0083 |
| S5 | FLA-GDN | 0.3552 | 0.3148 | 0.3850 | 0.0083 |
| S5 | M2RNN | 0.2157 | 0.1856 | 0.2309 | 0.0083 |
| S5 | M2RNN-paper | 0.1698 | 0.1555 | 0.1844 | 0.0083 |

Length extrapolation on S5:

| Model | T=128 | T=256 | T=512 | T=1024 |
| --- | ---: | ---: | ---: | ---: |
| E88/NDM | 0.7900 | 0.4158 | 0.2150 | 0.1104 |
| FLA-GDN | 0.3544 | 0.1843 | 0.0974 | 0.0521 |
| M2RNN | 0.2142 | 0.1120 | 0.0593 | 0.0339 |
| M2RNN-paper | 0.1696 | 0.0884 | 0.0488 | 0.0283 |

Interpretation:

- E88/NDM separates at train length, not only under extrapolation.
- E88/NDM also remains ahead under length extrapolation.
- Length extrapolation still degrades for all models. Do not blur this with
  train-length performance.
- M2RNN being below FLA-GDN is not awkward if the mechanism is framed
  correctly. It supports the claim that delta correction, not nonlinear
  matrix state in general, is the decisive state-tracking resource.

## First Figure To Build

Figure 1 should be the S5/S3 mechanism figure:

- Panel A: train-length bars for S3 and S5, four models.
- Panel B: S5 length-extrapolation curves for T=128,256,512,1024.
- Panel C, optional: mechanism schematic:
  - NDM: `H_t = tanh(lambda H_{t-1} + k_t (v_t - H_{t-1}^T k_t)^T)`
  - M2RNN: `H_t = f H_{t-1} + (1-f) tanh(H_{t-1} W + k_t v_t^T)`
  - GDN: delta-style but linear temporal state.

The caption should say that S5 tests non-solvable finite-state tracking, not
language-model perplexity.

## Paper Outline

### 1. Introduction

- Modern sequence models are mostly linear-scan, attention, or hybrids.
- Matrix state is not the novelty. Nonlinear matrix-state RNNs now exist in
  related work.
- The key question is which update resource trains and tracks state.
- NDM proposes a pure nonlinear delta-correcting recurrent stack.
- Contributions:
  - formal NDM/M2RNN update-family resource separation;
  - checked finite-state/S5 task formalization;
  - empirical S5 separation identifying delta correction as the mechanism;
  - 1.27B pure recurrent LM training competitive with leading linear models;
  - optimized Triton implementation making this practical.

### 2. Background And Related Work

- Linear SSMs, Mamba2, Gated DeltaNet.
- M2RNN and nonlinear matrix-state RNNs.
- Barrington/S5 motivation, stated as cited background not Lean-proven.
- Computational expressivity tasks and state tracking.

### 3. NDM Architecture

- Define Nonlinear Delta Memory.
- E88 as current implementation lineage.
- Many-program recurrent geometry.
- Delta-correcting associative memory intuition.
- Contrast with:
  - raw-write M2RNN;
  - linear delta GDN;
  - Mamba2 diagonal/selective SSM.

### 4. Formal Results

- Trusted proof surface and no-sorry policy.
- Fixed finite-precision ceiling: finite-state, regular, inside NC1.
- NDM/M2RNN resource separation.
- M2RNN read-then-delta embedding, preventing overclaim.
- S5 tracker formalization and exact finite transition-memory realization.
- What is not proved: lower bound and Barrington formalization.

### 5. Synthetic Expressivity Experiments

- Task suite overview.
- S3/S5 construction.
- Matched 8M model settings.
- Figure 1: S3/S5 bars and S5 extrapolation curves.
- Main result: E88/NDM wins; M2RNN underperforms, isolating delta correction.

### 6. Language Modeling At Scale

- 1.27B E88/NDM pure recurrent training.
- Context curriculum / 2K racers / later 64K scaling.
- Compare E88/NDM, FLA-GDN, Mamba2, M2RNN.
- Bits per byte / loss trajectories.
- Claim carefully: competitive with best linear recurrent baselines, not yet
  claiming released frontier-quality text model.

### 7. Systems Contribution

- Triton E88 kernel.
- Fused L2 norm, warp tuning, sparse checkpointing, fewer copies, output gate.
- End-to-end throughput vs CUDA and memory parity.
- ROCm portability.

### 8. Discussion

- Pure recurrence can be practical.
- Hybridization is not automatically right.
- Delta correction appears to be the key state-tracking resource.
- Biological / multiprogramming intuition, if used, should be restrained and
  downstream of the data.

### 9. Limitations

- No formal lower bound yet for linear/scan models on S5.
- S5 length extrapolation is not solved; E88 starts higher and remains ahead,
  but all models degrade.
- Training runs are still research-scale, not released foundation models.
- M2RNN reproduction depends on available implementation details and our
  faithful implementation choices.

## Mechanical Cleanup Later

Do not do this before stabilizing the paper notes and figures, but track it:

- Rename presentation away from "Elman" toward NDM.
- Clarify that E88 is an implementation/configuration, not the family name.
- Update READMEs and docs:
  - NDM = Nonlinear Delta Memory.
  - Pure nonlinear recurrent stack.
  - Delta-correcting matrix-state update.
- Decide repo/project naming. Avoid losing history.
- Audit old docs for stale "TC0" and "first matrix-state" claims.

## Immediate Next Actions

1. Build Figure 1 from `s5_witness_8m_20260521`.
2. Generate a compact markdown report for the S3/S5 run.
3. Add the report and figure scripts to version control if clean.
4. Start paper outline as prose only after Figure 1 exists.
5. Consider a longer/curriculum S5 run to test whether E88 can improve length
   generalization, but do not block paper framing on that.
