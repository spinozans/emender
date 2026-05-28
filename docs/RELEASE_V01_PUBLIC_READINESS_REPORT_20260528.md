# Final v0.1 Public-Release Readiness Report

Audit time: 2026-05-28T00:52:07Z
Task: `release-v01-public-readiness-report`
Checkout before this report: `9d050ef26c91814109dc6da1bda53d636df2a6ce`

> Superseded status note, 2026-05-28: the GitHub repository
> <https://github.com/poietic-pbc/emender> is now live, and the three Hugging
> Face repositories are now public under the `1.3b` slugs. This document below
> is retained as historical pre-public-readiness evidence from before the public
> flip and slug rename.

## Executive Status

The three Hugging Face model repositories are still private, and their current
`v0.1` revisions are the same docs-polish SHAs validated by the final Docker
smoke on CPU and CUDA GPU 4.

The model-card and local release-document surfaces are ready in content: they
describe the raw Pile-trained base models, 2,048-token input context, delimiter
behavior, current BPB metrics, limitations, Apache-2.0 metadata/citation, and
`revision="v0.1"` load examples.

The blocking item is GitHub publication identity. The intended canonical public
repository, `https://github.com/poietic-pbc/emender`, still does not resolve as a
public or authenticated-accessible repository. The source branch is synchronized
enough to push once the owner creates/transfers/renames the repository, but the
canonical public GitHub URL and paper PDF release asset are not live yet.

This task made no source-code, model, tag, or visibility changes. It adds this
readout only.

## Hugging Face Repositories

Authenticated live readback with `HfApi().repo_info(..., revision="v0.1",
token=True)` reported `private=True` for every repo. Unauthenticated HTTP HEAD
requests returned `401` for all three model URLs, consistent with private HF
visibility.

| Model | Private HF URL | Public HEAD | Authenticated visibility | Release revision | Resolved `v0.1` SHA | Main-card SHA | Docker CPU | Docker GPU |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| Emender/E88 | <https://huggingface.co/poietic-pbc/emender-e88-1.3b> | 401 | `private=True` | `v0.1` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | `718b3e15bb8ed7f065c5aa65a569e62af7a12a02` | PASS | PASS |
| GDN | <https://huggingface.co/poietic-pbc/gdn-1.3b> | 401 | `private=True` | `v0.1` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | `7b267ad249cf57594feaa38ef6b3aebd108722c4` | PASS | PASS |
| M2RNN-CMA | <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b> | 401 | `private=True` | `v0.1` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | `74091c1457d0e6a46872d72d38d12f6a10170d29` | PASS | PASS |

Evidence:

- Final Docker smoke scope and SHAs: `docs/RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md:21`.
- Docker smoke required exact `v0.1` resolution and `private=True` before load:
  `docs/RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md:29`.
- Docker runtime and GPU probe: `docs/RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md:33`.
- CPU/GPU result table: `docs/RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md:179`.
- Final smoke validation checklist: `docs/RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md:206`.
- Hub SHAs and main-card SHAs: `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md:28`.

## Docker Load/Generate Status

All three `v0.1` revisions load and generate from Docker on CPU and GPU.

The final smoke ran six generation jobs:

- Emender/E88 CPU: PASS, generated token IDs `[218, 218]`, decoded as
  `'\x1e\x1e'`.
- Emender/E88 CUDA GPU 4: PASS, generated token IDs `[218, 218]`, decoded as
  `'\x1e\x1e'`.
- GDN CPU: PASS, generated token IDs `[318, 318]`, decoded as `' is is'`.
- GDN CUDA GPU 4: PASS, generated token IDs `[318, 318]`, decoded as
  `' is is'`.
- M2RNN-CMA CPU: PASS, generated token IDs `[2109, 34059]`, decoded as
  `'........Officers'`.
- M2RNN-CMA CUDA GPU 4: PASS, generated token IDs `[2109, 34059]`, decoded as
  `'........Officers'`.

The smoke success condition included exact private `v0.1` revision resolution,
`private=True` metadata, nonempty generated text, and finite logits at every
step. No NaNs, infinities, tracebacks, or crashes were reported.

## Model-Card And Docs Checklist

Model-card coverage is ready. Remote README spot checks on all three private
`v0.1` revisions passed for these required strings and sections:

- raw/base identity and no instruction/chat/RLHF/safety tuning;
- The Pile training data;
- `p50k_base`, `chunk_size=2048`, and 2,048 input positions;
- delimiter behavior for `\x1e` / `0x1e`, including token ID `218`;
- v0.1 BPB metrics: Emender/E88 `0.979`, GDN `0.975`, M2RNN-CMA `0.984`;
- `revision="v0.1"` and `trust_remote_code=True` load examples;
- intended use and limitations, including Pile text risks and custom-code
  loading;
- Apache-2.0 frontmatter and the `garrison2026emender` citation;
- GitHub, paper PDF, release hub, and all three HF repository links.

Local evidence:

- Model-card polish coverage list: `docs/RELEASE_V01_MODEL_CARD_DOCS_POLISH_20260528.md:43`.
- Remote readback validation from the polish task:
  `docs/RELEASE_V01_MODEL_CARD_DOCS_POLISH_20260528.md:75`.
- Current model-card template links: `docs/MODEL_CARD_TEMPLATE.md:22`.
- Current model-card identity/context/delimiter/metrics/load/limits/citation
  sections: `docs/MODEL_CARD_TEMPLATE.md:40`,
  `docs/MODEL_CARD_TEMPLATE.md:67`, `docs/MODEL_CARD_TEMPLATE.md:88`,
  `docs/MODEL_CARD_TEMPLATE.md:104`, `docs/MODEL_CARD_TEMPLATE.md:156`, and
  `docs/MODEL_CARD_TEMPLATE.md:179`.
- Current central release hub model facts:
  `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md:34`.

## GitHub Merge/Sync And Publication State

The release source state is synchronized on the current canonical development
remote, but the intended canonical public repository is not live.

Observed in this report checkout:

- `HEAD`, `main`, and `origin/main` all point to
  `9d050ef26c91814109dc6da1bda53d636df2a6ce`
  (`feat: release-v01-final-v01-docker-smoke (agent-433)`).
- Recent release commits include:
  - `d16ff12` - merge/sync audit;
  - `27fef35` - model-card/docs polish;
  - `9d050ef` - final v0.1 Docker smoke.
- Remotes:
  - `origin` -> `git@github.com:ekg/ndm.git`;
  - `poietic` -> `git@github.com:poietic-pbc/emender.git`.

Live public/accessibility checks:

- `https://github.com/poietic-pbc/emender` returned HTTP `404`.
- `gh repo view poietic-pbc/emender --json ...` returned:
  `GraphQL: Could not resolve to a Repository with the name 'poietic-pbc/emender'.`
- `GIT_TERMINAL_PROMPT=0 git ls-remote
  https://github.com/poietic-pbc/emender.git HEAD` failed because the repository
  is not publicly fetchable without credentials.

Merge/sync audit evidence:

- Completed release-worker patches matched canonical `main` by stable patch-id
  in the merge/sync audit: `docs/RELEASE_V01_MERGE_SYNC_AUDIT_20260528.md:40`.
- The intended `poietic` remote was added non-destructively:
  `docs/RELEASE_V01_MERGE_SYNC_AUDIT_20260528.md:12`.
- The audit's GitHub verification found the target repo inaccessible:
  `docs/RELEASE_V01_MERGE_SYNC_AUDIT_20260528.md:103`.
- Manual publication steps are recorded at
  `docs/RELEASE_V01_MERGE_SYNC_AUDIT_20260528.md:139`.

Conclusion: `poietic-pbc/emender` is not public, not currently accessible, and
therefore not yet a reproducible public canonical repository. Once an owner
creates/transfers/renames the public repository and pushes `main` to the
`poietic` remote without force, the synchronized source state should be enough
for reproduction.

## Cross-Link Audit

The link surfaces are internally consistent and point to the intended public
release targets. The unresolved issue is that the GitHub target and paper PDF
asset location do not exist publicly until the GitHub repository/release is
created.

| Surface | Required links | Result | Evidence |
| --- | --- | --- | --- |
| Paper -> release docs/HF/PDF | Release hub, all three HF `tree/v0.1` targets, paper PDF target | PASS by content; target GitHub/PDF not live yet | `paper/main.typ:335`, `paper/main.typ:1735` |
| GitHub README/docs -> paper/HF | README links hub, GitHub target, paper PDF/source, all three HF repos | PASS by content; target GitHub/PDF not live yet | `README.md:23`, `docs/HUGGINGFACE_RELEASE.md:186` |
| HF model cards -> GitHub/paper/citation | GitHub target, release hub, paper PDF/source, all three HF repos, citation | PASS by remote README readback and template | `docs/MODEL_CARD_TEMPLATE.md:22`, `docs/MODEL_CARD_TEMPLATE.md:179` |
| Central release docs/checklist -> paper/GitHub/HF/tags/smoke | GitHub, paper PDF/source, HF URLs, `v0.1` SHAs, Docker smoke evidence, checklist | PASS by content; checklist still blocks on GitHub/PDF/approval | `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md:10`, `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md:22`, `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md:56`, `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md:84` |

Legacy public-facing namespace scan:

- No `ekg/ndm`, `poietic-pbc/ndm`, or non-target `huggingface.co/...`
  references were found in `README.md`, `docs/HUGGINGFACE_RELEASE.md`,
  `docs/MODEL_CARD_TEMPLATE.md`, `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md`,
  `docs/STANDALONE_USAGE.md`, `paper/main.typ`, or `pyproject.toml`.
- Historical development provenance still intentionally references
  `ekg/elman` and `ekg/elman-proofs` in `README.md` and `pyproject.toml`.

## Exact URLs And SHAs Ready For Release

HF URLs and SHAs:

- Emender/E88:
  - repo: <https://huggingface.co/poietic-pbc/emender-e88-1.3b>
  - release revision URL:
    <https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.1>
  - resolved `v0.1` SHA:
    `a2e56cb82eec5e01ae6eb501569359c5ff64af6b`
  - main-card SHA:
    `718b3e15bb8ed7f065c5aa65a569e62af7a12a02`
- GDN:
  - repo: <https://huggingface.co/poietic-pbc/gdn-1.3b>
  - release revision URL:
    <https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.1>
  - resolved `v0.1` SHA:
    `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9`
  - main-card SHA:
    `7b267ad249cf57594feaa38ef6b3aebd108722c4`
- M2RNN-CMA:
  - repo: <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b>
  - release revision URL:
    <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.1>
  - resolved `v0.1` SHA:
    `8181b77803e130ffd78e37c33aa4d58c27e719c2`
  - main-card SHA:
    `74091c1457d0e6a46872d72d38d12f6a10170d29`

GitHub/source URLs and SHAs:

- Intended public GitHub repository:
  <https://github.com/poietic-pbc/emender>
  - status: not live; HTTP `404` at audit time.
- Intended paper PDF release asset:
  <https://github.com/poietic-pbc/emender/releases/download/v0.1/Garrison_2026_Emender.pdf>
  - status: not live until the GitHub repository/release asset exists.
- Paper source target:
  <https://github.com/poietic-pbc/emender/blob/main/paper/main.typ>
  - status: content points correctly, but URL depends on the missing GitHub repo.
- Current synchronized source checkout before this report:
  `9d050ef26c91814109dc6da1bda53d636df2a6ce`.

## Remaining Blockers

1. Create, transfer, or rename the GitHub repository so
   `poietic-pbc/emender` exists publicly.
2. Push `main` to `git@github.com:poietic-pbc/emender.git` without force and
   verify default branch/public visibility.
3. Build the paper PDF from `paper/main.typ`, review it, and attach it to the
   GitHub `v0.1` release asset URL. Generated PDFs should remain uncommitted.
4. Log explicit human approval for HF public visibility.
5. Only after approval, flip the three HF model repos public, then re-read HF
   metadata and `v0.1` SHAs.

## Approval-Gated HF Public Flip Commands

Do not run these until the owner explicitly approves public HF visibility and
the approval is logged. The current Hugging Face Hub client and official docs
support `update_repo_settings(..., private=<bool>)`; the older
`update_repo_visibility(..., private=False)` method exists locally but is
deprecated in current official docs. Official reference:
<https://huggingface.co/docs/huggingface_hub/en/guides/repository#update-visibility>.

Exact Python API calls for approval-gated publication:

```python
from huggingface_hub import HfApi

api = HfApi()
api.update_repo_settings(
    repo_id="poietic-pbc/emender-e88-1.3b",
    repo_type="model",
    private=False,
    token=True,
)
api.update_repo_settings(
    repo_id="poietic-pbc/gdn-1.3b",
    repo_type="model",
    private=False,
    token=True,
)
api.update_repo_settings(
    repo_id="poietic-pbc/m2rnn-cma-1.3b",
    repo_type="model",
    private=False,
    token=True,
)
```

Equivalent one-shot shell wrapper:

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
for repo_id in [
    "poietic-pbc/emender-e88-1.3b",
    "poietic-pbc/gdn-1.3b",
    "poietic-pbc/m2rnn-cma-1.3b",
]:
    api.update_repo_settings(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        token=True,
    )
PY
```

Post-flip verification command:

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
for repo_id in [
    "poietic-pbc/emender-e88-1.3b",
    "poietic-pbc/gdn-1.3b",
    "poietic-pbc/m2rnn-cma-1.3b",
]:
    info = api.repo_info(
        repo_id=repo_id,
        repo_type="model",
        revision="v0.1",
        token=True,
    )
    print(repo_id, "private=", getattr(info, "private", None), "sha=", info.sha)
PY
```

Expected post-flip result: `private=False` for all three repos, with the same
`v0.1` resolved SHAs listed in this report.

## Validation Summary

- [x] Final private HF repo URLs, `v0.1` SHAs, visibility state, and smoke
      results are listed.
- [x] Merge/sync status and unresolved blockers are listed.
- [x] GitHub repo name/public/remote status for `poietic-pbc/emender` is listed.
- [x] Model-card/docs checklist result is listed.
- [x] Cross-link audit covers paper, GitHub/docs, HF model cards, and central
      release docs.
- [x] Explicit next step for public release and the HF API method are listed.
- [x] No source/model/visibility changes were made.
