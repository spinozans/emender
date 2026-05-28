# v0.1 Merge/Sync and GitHub Readiness Audit

Audit time: 2026-05-28T00:15:03Z

## Result

No release worker commit needed to be merged or cherry-picked during this audit.
The completed v0.1 release pipeline worker commits are present on canonical
`main` as workgraph squash commits, and every audited worker patch has a stable
`git patch-id` match on `main`.

The GitHub publication identity is not fully ready. The intended canonical public
target is `https://github.com/poietic-pbc/emender`, but authenticated GitHub API,
SSH, HTTPS, and public HTTP checks all report that `poietic-pbc/emender` is not
an accessible repository at this time. The current `origin` remains the historical
public repository `git@github.com:ekg/ndm.git`. A non-destructive `poietic` remote
was added locally and points to the intended future target:

```text
origin   git@github.com:ekg/ndm.git (fetch)
origin   git@github.com:ekg/ndm.git (push)
poietic  git@github.com:poietic-pbc/emender.git (fetch)
poietic  git@github.com:poietic-pbc/emender.git (push)
```

No force-push, destructive remote rename, repository transfer, visibility change,
or GitHub repo creation was performed.

## Canonical Branch State

At audit start, `HEAD`, `main`, and `origin/main` all pointed to:

```text
582b0d8 feat: quality-pass-20260528-public-readiness (agent-420)
```

The worktree had no tracked or staged changes. The only pre-existing untracked
entry was the workgraph symlink `.wg`.

## Completed Release Commit Inventory

These completed release-related worker branches have intentional source, docs,
or scripts that need to be reproducible from the canonical checkout. Each row
lists the worker commit, the canonical `main` commit carrying the same patch,
and the release-relevant file summary.

| Task | Worker branch/head | Canonical `main` commit | Patch status | File summary |
| --- | --- | --- | --- | --- |
| `quality-pass-20260527-hf-release-v01` | `wg/agent-386/...` `c1daa98` | `7acbb2f` | stable patch-id match | `docs/RELEASE_V01_QUALITY_PASS_20260527.md` |
| `release-v01-preflight` | `wg/agent-395/...` `34abf17` | `3595547` | stable patch-id match | `docs/RELEASE_V01_PREFLIGHT_20260527.md` |
| `release-v01-emender-repo-dry-run` | `wg/agent-398/...` `f6c6252` | `2cb5f8c` | stable patch-id match | `docs/EMENDER_REPO_DRY_RUN_20260527.md` |
| `release-v01-racer-checkpoint-pin` | `wg/agent-399/...` `2faa764` | `0483e6f` | stable patch-id match | `docs/RELEASE_V01_RACER_CHECKPOINT_PIN_20260527.md`; `paper/main.typ`; `paper/results/figure_2/AS_OF.md`; `paper/results/figure_2/E88_NDM.csv`; `paper/results/figure_2/FLA_GDN.csv`; `paper/results/figure_2/M2RNN_CMA.csv`; `paper/results/figure_2/figure_2_draft.png` |
| `release-v01-local-three-model-smoke` | `wg/agent-404/...` `f1f2757` | `8eb3dc1` | stable patch-id match | `docs/RELEASE_V01_LOCAL_THREE_MODEL_SMOKE_20260527.md`; `scripts/smoke_local_checkpoint_generation.py` |
| `release-v01-docker-local-smoke` | `wg/agent-407/...` `4a0d559` | `80630b0` | stable patch-id match | `.dockerignore`; `docker/release-v01-local-smoke.Dockerfile`; `docs/RELEASE_V01_DOCKER_LOCAL_SMOKE_20260527.md`; `scripts/docker_local_checkpoint_smoke.sh` |
| `release-v01-private-hf-staging-upload` | `wg/agent-410/...` `4479815` | `6bb1800` | stable patch-id match | `docs/RELEASE_V01_PRIVATE_HF_STAGING_UPLOAD_20260527.md`; `scripts/hf_private_staging_upload.py` |
| `release-v01-docker-private-hf-smoke` | `wg/agent-413/...` `a174d3b` | `ca43716` | stable patch-id match | `docker/release-v01-private-hf-smoke.Dockerfile`; `docs/RELEASE_V01_DOCKER_PRIVATE_HF_SMOKE_20260527.md`; `scripts/docker_private_hf_smoke.sh`; `scripts/smoke_private_hf_generation.py` |
| `release-v01-tag-and-paper-sync` | `wg/agent-416/...` `ba84c81` | `076d0bd` | stable patch-id match | `NEXT_STEPS.md`; `docs/EMENDER_REPO_DRY_RUN_20260527.md`; `docs/HUGGINGFACE_RELEASE.md`; `docs/MODEL_CARD_TEMPLATE.md`; `docs/RELEASE_V01_PRIVATE_HF_STAGING_UPLOAD_20260527.md`; `docs/RELEASE_V01_TAG_AND_PAPER_SYNC_20260527.md` |
| `quality-pass-20260528-public-readiness` | `wg/agent-420/...` `5401b36` | `582b0d8` | stable patch-id match | `docs/RELEASE_V01_PUBLIC_READINESS_QUALITY_PASS_20260528.md` |

Patch-id evidence:

```text
c1daa98 -> 7acbb2f | match=yes
34abf17 -> 3595547 | match=yes
f6c6252 -> 2cb5f8c | match=yes
2faa764 -> 0483e6f | match=yes
f1f2757 -> 8eb3dc1 | match=yes
4a0d559 -> 80630b0 | match=yes
4479815 -> 6bb1800 | match=yes
a174d3b -> ca43716 | match=yes
ba84c81 -> 076d0bd | match=yes
5401b36 -> 582b0d8 | match=yes
```

## v22 Release-Related Paper Precursors

The 2026-05-27 v22 release wording/PDF precursor tasks are also accounted for:

| Task | Worker branch/head | Canonical `main` commit | Patch status | Disposition |
| --- | --- | --- | --- | --- |
| `v22-poietic-v01-paper-pdf-refresh` | `wg/agent-379/...` `cbc5fff` | `446a7c3` | stable patch-id match | Present on `main`; later superseded by namespace correction. |
| `v22-poietic-pbc-namespace-pdf-refresh` | `wg/agent-382/...` `50b5cd8` | `fa24d62` | stable patch-id match | Present on `main`; paper release URL uses `poietic-pbc`. |

The failed `v22-final-hf-release-polish` task has worker commit `c240e8f`
touching `paper/main.typ`, but that task failed because Hugging Face upload/load
validation was blocked by missing HF auth. Its useful paper edits were recovered
by `v22-poietic-v01-paper-pdf-refresh` and then corrected by
`v22-poietic-pbc-namespace-pdf-refresh`; no direct cherry-pick from the failed
branch is required.

## Open or Paused Release Tasks Excluded From Merge Sync

These tasks were not merged by this audit because they were not completed release
artifacts at audit time:

| Task | Status at audit | Reason excluded |
| --- | --- | --- |
| `release-v01-model-card-docs-polish` | in-progress, no commits ahead at audit start | Worker had started docs/HF-card work but had not produced a completed commit. |
| `release-v01-final-v01-docker-smoke` | open | Depends on docs polish and has no completed artifact yet. |
| `release-v01-public-visibility-flip` | paused/approval-gated | Requires explicit approval before public HF visibility changes. |
| `release-v01-public-readiness-report` | open downstream | Depends on this audit and final Docker smoke. |

## GitHub Target Verification

Commands and results:

```text
gh repo view poietic-pbc/emender --json nameWithOwner,url,visibility,isPrivate,...
GraphQL: Could not resolve to a Repository with the name 'poietic-pbc/emender'.

gh api repos/poietic-pbc/emender
HTTP 404 Not Found

git ls-remote git@github.com:poietic-pbc/emender.git HEAD
ERROR: Repository not found.

curl -sSI https://github.com/poietic-pbc/emender
HTTP/2 404
```

The authenticated GitHub account can see the organization:

```text
gh api orgs/poietic-pbc
{"login":"poietic-pbc", ...}
```

Listing accessible organization repositories did not include `emender`. Current
observable state is therefore:

- `https://github.com/poietic-pbc/emender` is not public.
- `poietic-pbc/emender` does not exist or is inaccessible to the current
  authenticated account.
- The current checkout's `origin` still points to the historical public repo
  `git@github.com:ekg/ndm.git`.
- A non-destructive `poietic` remote now points to
  `git@github.com:poietic-pbc/emender.git`.

## Manual Publication Steps

Because the target repository does not currently resolve and no explicit approval
was present to create, transfer, rename, or push a public GitHub repository, a
manual owner action remains required.

Recommended non-destructive path:

1. In the GitHub `poietic-pbc` organization, create a public repository named
   `emender`. Prefer an empty repository with no README/license/gitignore so the
   existing `main` branch can be pushed without history conflicts.
2. From a clean canonical checkout, verify remotes:
   `git remote -v`.
3. Keep `origin` intact unless the owner explicitly approves a remote migration.
   The intended remote is already configured here as:
   `poietic git@github.com:poietic-pbc/emender.git`.
4. Push without force:
   `git push poietic main`.
5. If release tags are desired, create/select them deliberately, then push
   explicit tag names only. Do not use force push.
6. Verify:
   `gh repo view poietic-pbc/emender --json url,visibility,isPrivate,defaultBranchRef`
   and `git ls-remote poietic HEAD`.
7. After the repo exists, update any remaining public metadata that still points
   to `ekg/ndm`, including known current references in `pyproject.toml`,
   `docs/STANDALONE_USAGE.md`, and `docs/MODEL_CARD_TEMPLATE.md`, unless the
   ongoing `release-v01-model-card-docs-polish` task handles those edits first.

Alternative path requiring explicit owner approval:

1. Transfer or rename the existing public `ekg/ndm` repository to
   `poietic-pbc/emender` through GitHub UI/API.
2. Confirm redirects, branch protections, and remote URLs.
3. Only after approval, update or rename `origin`. This audit did not do that.

## Artifact Guardrails

The audited completed release commits introduced docs, scripts, Dockerfiles,
paper source, paper figure CSVs, and a small figure PNG needed for reproducible
paper/source state. They did not introduce checkpoints, safetensors, HF caches,
Docker layers, tokens, or generated release PDFs.

Checks performed:

```text
find . -maxdepth 3 \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.pdf' ... \)
# no matching untracked/generated release artifacts found

git ls-files | rg '\.(safetensors|pt|pth|ckpt|pdf)$|(^|/)\.cache/|(^|/)hf_cache/|(^|/)\.huggingface/'
paper/results/cma_flop_rate/convergence.pdf
```

The tracked PDF above is pre-existing historical paper evidence outside the
audited v0.1 release commits. No generated PDF was added by the audited release
pipeline or by this audit.

## Validation Run In This Audit

Release script syntax and lightweight Python tests:

```text
python -m py_compile \
  scripts/smoke_local_checkpoint_generation.py \
  scripts/hf_private_staging_upload.py \
  scripts/smoke_private_hf_generation.py

bash -n \
  scripts/docker_local_checkpoint_smoke.sh \
  scripts/docker_private_hf_smoke.sh

PYTHONPATH=. python -m pytest tests/test_standalone_minimal.py tests/test_imports.py -m 'not gpu'
```

Result:

```text
6 passed, 1 skipped, 1 deselected
```

Paper source was not changed by this audit, so the paper build and Lean gate were
not rerun here. The release tasks that changed the paper logged successful
`paper/build.sh` and Lean trust-gate validation before their matched commits were
accepted onto `main`.

## Remaining Readiness Blockers

1. `poietic-pbc/emender` must be created, transferred, or renamed by an owner and
   made public before the canonical GitHub target is live.
2. The current `origin` still points to `git@github.com:ekg/ndm.git`; this is
   preserved intentionally until an owner approves migration.
3. Public metadata still contains some `ekg/ndm` references. Known examples at
   audit time include `pyproject.toml:48`, `docs/STANDALONE_USAGE.md:12`, and
   `docs/MODEL_CARD_TEMPLATE.md:131`. These should be updated once the canonical
   GitHub repo exists, or by the in-progress model-card/docs polish task if it
   completes first.
4. The final model-card/docs polish and final Docker smoke tasks were still
   incomplete at audit time.
