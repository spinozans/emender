# Provenance

Emender is a curated paper-facing repository extracted from two historical
repositories. The Python package and many source paths still use `ndm` for
compatibility and historical continuity.

- Code and experiments: `git@github.com:ekg/elman.git`
  - commit: `6f0724feae9fc82bd235408ac5c3ae61f2b17c79`
  - short log:
    - `6f0724f Add reasoning eval panel builder`
    - `2316740 Add periodic racer eval harness`
    - `4758dd1 Add racer checkpoint eval tools`
    - `99332f7 Add NDM paper notes`
    - `66a5e84 Add S5 witness expressivity tasks`
- Formal proofs: `git@github.com:ekg/elman-proofs.git`
  - commit: `5082610c9cdabf0b31e11dd14ee078273d486333`
  - short log:
    - `5082610 Prove explicit S5 tracker realization`
    - `2564d29 Mark historical proof sketches`
    - `13970b8 Add S5 witness proof scaffold`
    - `bf147b2 Add NDM paper core formalism`
    - `84011f5 Define trusted Lean proof surface`

The historical names `ElmanProofs`, `NDM`, and `E88` are retained in some source
files to preserve reproducibility during the migration. Public-facing
documentation should use **Emender** for the repository and model family,
**Emender/E88** for the current optimized implementation, and
**nonlinear delta memory** for the recurrent mechanism when the mechanism needs
to be named.
