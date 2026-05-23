# Nonlinear Delta Memory

This is the paper-facing repository for **Nonlinear Delta Memory** (NDM), a
family of pure nonlinear recurrent language models with matrix-valued state and
delta-correcting updates.

The current production NDM instance is still named `E88` in some code paths.
That name is historical: it is the optimized NDM implementation developed in
the `ekg/elman` research repository. The purpose of this repository is to keep
the paper artifact focused on the claims that survived the experimental and
formal audit.

## Current Claim Surface

- Pure nonlinear recurrent stacks can be trained at 1.27B-parameter scale
  without attention or hybridization.
- Once the nonlinear recurrence is engineered and tuned, next-token loss tracks
  useful wallclock compute more closely than raw steps or raw tokens processed.
- NDM/E88 sits in the same LM-loss regime as strong linear recurrent baselines
  while exposing a different computational profile.
- Controlled S5/noncommutative state-tracking experiments separate NDM from
  linear recurrent baselines and from M2RNN-style raw-write matrix updates.
- Broad closed-book QA and natural-language reasoning benchmarks are tracked as
  auxiliary probes, not as the main NDM claim.

## Repository Layout

- `ndm/`: model code, including the E88/NDM implementation, Triton kernels, and
  comparison baselines used by the current experiments.
- `experiments/expressivity_tasks/`: controlled algorithmic and S5-style
  expressivity tasks.
- `scripts/`: checkpoint evaluation, knowledge/reasoning panel builders, and
  data-preparation helpers.
- `formal/lean/`: Lean formalization imported from `elman-proofs`. The Lean
  namespace is intentionally still `ElmanProofs` until the trusted proof chain
  is renamed deliberately.
- `docs/`: architecture, stability, M2RNN comparison, and Frontier training
  notes.
- `paper/`: working paper notes.
- `provenance/`: source-repository commit anchors for the migration.

## Provenance

This repository is curated from:

- `ekg/elman` at `6f0724feae9fc82bd235408ac5c3ae61f2b17c79`
- `ekg/elman-proofs` at `5082610c9cdabf0b31e11dd14ee078273d486333`

The historical repositories remain the paper trail for when the architecture,
kernel, experiment, and proof work happened. This repository is the focused
artifact for reproduction and presentation.

## Minimal Checks

Python import/syntax smoke:

```bash
python -m py_compile train.py scripts/*.py ndm/triton/*.py ndm/models/e88_fused.py
python - <<'PY'
from ndm.models.e88_fused import E88FusedLM
print(E88FusedLM.__name__)
PY
```

Lean trusted-core check:

```bash
cd formal/lean
scripts/check_paper_core.sh
scripts/check_trusted_no_placeholders.sh ElmanProofs.lean
```

The Lean target still contains historical sketch modules outside the trusted
surface. The trusted checks define the paper-facing boundary.
