# Emender repo rename/public-readiness dry run - 2026-05-27

Task: `release-v01-emender-repo-dry-run`

## Scope and guardrails

This dry run tested a non-destructive path for publishing the repository as
`poietic-pbc/emender`. It did not rename the active workgraph checkout, did not
publish or create any GitHub repository, did not change remote visibility, and
did not stage tokens, checkpoints, safetensors, Hugging Face caches, Docker
layers, generated PDFs, or large generated artifacts.

The active checkout remote was checked before and after the dry run and remains:

```text
origin  git@github.com:ekg/ndm.git (fetch)
origin  git@github.com:ekg/ndm.git (push)
```

## Parallel checkout

A standalone clone was created under the user's home directory:

```text
/home/erikg/emender-dry-run-agent-398-20260527
```

The clone was made with:

```bash
git clone --no-hardlinks /home/erikg/ndm/.wg-worktrees/agent-398 \
  /home/erikg/emender-dry-run-agent-398-20260527
```

Inside the standalone clone only, the local source remote was preserved and the
proposed public GitHub target was configured as `origin`:

```text
local-source  /home/erikg/ndm/.wg-worktrees/agent-398 (fetch)
local-source  /home/erikg/ndm/.wg-worktrees/agent-398 (push)
origin        git@github.com:poietic-pbc/emender.git (fetch)
origin        git@github.com:poietic-pbc/emender.git (push)
```

No fetch, push, repository creation, repository rename, or visibility change was
performed.

## Environment and package smoke

The standalone clone was tested in an isolated virtual environment using system
site packages to avoid re-downloading already installed heavyweight ML
dependencies:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .
```

Result: editable install succeeded as distribution `ndm==0.2.0`.

Installed dependency availability in the dry-run environment:

```text
Python 3.12.3
torch: OK 2.9.1+cu128
numpy: OK 1.26.4
einops: OK 0.8.1
tqdm: OK 4.67.1
triton: OK 3.5.1
pytest: OK 9.0.2
datasets: OK 4.4.2
tiktoken: OK 0.12.0
```

Current-package import/load smoke:

```bash
.venv/bin/python -m pytest tests/test_standalone_minimal.py tests/test_imports.py -m 'not gpu'
```

Result:

```text
6 passed, 1 skipped, 1 deselected in 8.07s
```

The skipped test was `test_ladderlm_cpu_forward`, because `mamba_ssm` is present
in the shared Python environment and its fused-norm path requires CUDA tensors
for that CPU flow. `E88FusedLM` CPU forward, package exports, and import tests
passed.

Import-path status:

```text
ndm: /home/erikg/emender-dry-run-agent-398-20260527/ndm/__init__.py
emender: MISSING
import emender failed: ModuleNotFoundError: No module named 'emender'
```

Conclusion: the current package builds and loads correctly as `ndm`. A public
GitHub repo rename to `poietic-pbc/emender` can be treated separately from a
Python import-path migration. A full `import emender` identity is not supported
yet and should be planned explicitly.

## Current public-layout blockers

The repository is publishable as a renamed GitHub repository only if it is
acceptable for v0.1 to keep the Python distribution and import package named
`ndm`. The following files still bind the public package/repository identity to
`ndm` or `ekg/ndm`:

| File | Lines | Public-readiness issue |
| --- | ---: | --- |
| `pyproject.toml` | 6, 8, 48, 54 | Distribution is `ndm`, description says Nonlinear Delta Memory, repository URL is `https://github.com/ekg/ndm`, package discovery includes `ndm*`. |
| `README.md` | 1, 3, 18, 58, 99-111, 129, 152 | Front door still presents Nonlinear Delta Memory / NDM as the public name, with `ndm/` source paths. |
| `docs/STANDALONE_USAGE.md` | 1, 6, 12-24, 38-44, 62, 72, 78, 88 | Install, clone, and import examples all use `ndm` and `https://github.com/ekg/ndm`. |
| `docs/MODEL_CARD_TEMPLATE.md` | 10, 15, 17, 28-30, 67-69, 83-85, 125, 197-198, 204, 212, 221-225 | Model tag/title, update naming, model load examples, repo links, and citation URL still point to `ndm` / `ekg/ndm`. |
| `docs/HUGGINGFACE_RELEASE.md` | 17, 24, 40, 58, 63, 71, 78, 112-119, 123, 128-129, 144-158 | HF release flow still targets `ekg/ndm-1.27b`; it also describes `ndm` as the public package name. Any HF repo should remain private until explicit approval. |
| `NEXT_STEPS.md` | 1, 22, 85, 129, 227, 246, 264 | Release checklist still names NDM and `ekg/ndm-1.27b`. |
| `paper/README.md` | 1, 73 | Paper support doc still says "NDM Paper" and describes NDM as the architecture variant name. |
| `paper/OUTLINE.md` | 1, 56, 102-118, 167-192, 244-316, 546-572, 799 | Legacy outline uses Nonlinear Delta Memory / NDM throughout, while `paper/main.typ` already uses Emender. |
| `paper/ndmpapernotes.md` | 1, 5, 19, 29, 54-88, 134-178, 235-246, 276-350 | Legacy design notes remain NDM-branded. Consider archiving or clearly marking historical. |
| `paper/notes_reconciliation.md` | 1, 27, 36-51, 61-64, 100-120 | Reconciliation table remains NDM-branded and points at `ndm/` source paths. |
| `formal/lean/ElmanProofs.lean` | 1, 9 | Trusted-root comment calls this the public NDM repository. |
| `formal/lean/ElmanProofs/PaperCore.lean` | 23-48 | Public theorem surface describes NDM as the model family; theorem identifiers may remain historical unless a larger formal rename is approved. |
| `formal/lean/TRUSTED_PROOF_SURFACE.md` | 32-39 | Trusted proof summary remains NDM/E88-branded. |
| `provenance/README.md` | 3, 25 | Says `ndm` is the curated paper-facing repository and public docs should use NDM/E88. |

Many source imports under `scripts/`, `train.py`, `tests/`, and
`experiments/` intentionally import `ndm`. These are not safe mechanical edits
unless the Python package migration is part of the same change.

## Safe edit decision

No source-code rename was made in the active checkout. The current package
already builds and loads as `ndm`, and changing imports/package names now would
be a broad migration rather than a necessary fix for the dry run. The only safe
active-repo edit made by this task is this dry-run report.

## Recommended public-readiness path

### Option A: repo rename now, keep Python import as `ndm` for v0.1

This is the lowest-risk release path.

1. Create or rename the GitHub repository only after explicit user approval,
   with target `poietic-pbc/emender`.
2. Update `pyproject.toml` repository URL to `https://github.com/poietic-pbc/emender`.
3. Keep `[project].name = "ndm"` and `include = ["ndm*"]` for v0.1.
4. Rewrite the top-level README and release docs to present "Emender" as the
   public project/model family, while documenting that the v0.1 Python import is
   still `ndm`.
5. Update model-card and HF instructions from `ekg/ndm-1.27b` to the approved
   private/public target, likely `poietic-pbc/emender-e88-1.27b`, only after HF
   approval. Until then, keep HF repos private.
6. Run:
   ```bash
   python -m pytest tests/test_standalone_minimal.py tests/test_imports.py -m 'not gpu'
   python -m py_compile train.py scripts/*.py ndm/triton/*.py ndm/models/e88_fused.py
   ```

### Option B: full Python import migration to `emender`

This should be a separate implementation task.

1. Add an `emender/` package and either move `ndm/` into it or provide a
   compatibility shim.
2. Decide compatibility direction:
   - preferred public import: `import emender`
   - temporary compatibility import: `import ndm`
3. Update `pyproject.toml` package discovery and distribution name.
4. Update every internal absolute import in `train.py`, `scripts/`, `tests/`,
   `experiments/`, `ndm/models/`, and `ndm/triton/`.
5. Add explicit tests for both:
   ```python
   import emender
   from emender.models.e88_fused import E88FusedLM
   ```
   and, if compatibility is retained:
   ```python
   import ndm
   from ndm.models.e88_fused import E88FusedLM
   ```
6. Re-run the editable install and CPU smoke tests in a clean venv that does
   not already have `ndm` installed.

### Option C: distribution `emender`, import package `ndm`

This is technically easy but potentially confusing. It would require changing
`[project].name` to `emender` while continuing to install package `ndm`. If used,
the README and standalone docs must explicitly say:

```text
pip install emender
import ndm
```

This option is not recommended unless the release needs PyPI/HF package identity
before source import migration is ready.

## Validation checklist

- Parallel checkout/clone created at
  `/home/erikg/emender-dry-run-agent-398-20260527`.
- Active checkout remote was not changed; it remains `git@github.com:ekg/ndm.git`.
- Proposed GitHub target tested in the standalone clone as
  `git@github.com:poietic-pbc/emender.git`.
- Python editable install succeeded in the standalone clone.
- Current package import/load smoke passed for `ndm`.
- `import emender` blocker logged precisely: no `emender` package exists.
- Internal NDM/public-identity references are listed above.
- No publishing, GitHub repo creation, remote visibility change, or destructive
  remote rename was performed.
- No tokens, checkpoints, safetensors, HF caches, Docker layers, generated PDFs,
  or other large generated artifacts were staged or committed.
