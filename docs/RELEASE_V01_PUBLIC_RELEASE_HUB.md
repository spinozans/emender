# v0.1 Public Release Hub

Date: 2026-05-28 UTC
Task: `release-v01-model-card-docs-polish`

This is the central v0.1 release handoff for the Emender paper, GitHub
repository, and three Hugging Face model repositories. The HF repositories are
still private; no visibility-changing command is part of this document.

## Primary Links

- GitHub repository: <https://github.com/poietic-pbc/emender>
- Paper PDF target: <https://github.com/poietic-pbc/emender/releases/download/v0.1/Garrison_2026_Emender.pdf>
- Paper source: <https://github.com/poietic-pbc/emender/blob/main/paper/main.typ>
- v0.1 racer/checkpoint source: <https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V01_RACER_CHECKPOINT_PIN_20260527.md>

The paper PDF target is the intended public-release asset location. Until that
asset is attached, `paper/main.typ` is the source of record and `paper/build.sh`
builds `paper/Garrison_2026_Emender-<commit>.pdf`. Generated PDFs remain
gitignored and must not be committed.

## Hugging Face Models

All model cards on `main` and `v0.1` were polished on 2026-05-28. The `v0.1`
tags now point at docs-only commits descended from the previously smoke-tested
artifact commits.

| Model | HF repository | Release revision | Current `v0.1` resolved SHA | Main-card SHA |
| --- | --- | --- | --- | --- |
| Emender/E88 | <https://huggingface.co/poietic-pbc/emender-e88-1.27b> | `v0.1` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | `718b3e15bb8ed7f065c5aa65a569e62af7a12a02` |
| GDN | <https://huggingface.co/poietic-pbc/gdn-1.27b> | `v0.1` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | `7b267ad249cf57594feaa38ef6b3aebd108722c4` |
| M2RNN-CMA | <https://huggingface.co/poietic-pbc/m2rnn-cma-1.27b> | `v0.1` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | `74091c1457d0e6a46872d72d38d12f6a10170d29` |

## Model-Card Facts

- Identity: raw/base 1.27B-class recurrent language models; not
  instruction-tuned, chat-tuned, RLHF-tuned, or safety-tuned.
- Architectures: Emender/E88 nonlinear delta-memory, Gated DeltaNet baseline,
  and M2RNN-CMA raw-write nonlinear matrix-state baseline.
- Data: The Pile, pinned locally as `/home/erikg/elman/data/pile.txt` in the
  release checkpoint docs.
- Tokenizer/context: `p50k_base`, vocab size `50,281`, pinned training arg
  `chunk_size=2048`; the loader reads `chunk_size + 1` tokens so next-token
  training/eval uses 2,048 input positions.
- Delimiter: the corpus stream uses ASCII record separator `\x1e` / byte
  `0x1e` between documents. For the pinned `p50k_base` training path,
  `train.py` uses `ndm.data.tokenized_dataset.TokenizedStreamDataset`, which
  tokenizes raw byte windows with tiktoken and treats `\x1e` as an ordinary
  token. Under `p50k_base`, `"\x1e"` encodes as token ID `218`.
- Current racer metrics: Emender/E88 `0.979` BPB, GDN `0.975` BPB,
  M2RNN-CMA `0.984` BPB from the refreshed Figure 2 source recorded on
  2026-05-27 in `docs/RELEASE_V01_RACER_CHECKPOINT_PIN_20260527.md`.
- Loading: all three repositories use custom HF code and must be loaded with
  `revision="v0.1"` and `trust_remote_code=True`.

## Docker Smoke

Final private-HF Docker smoke passed on 2026-05-28 against the current
`v0.1` SHAs in this hub on CPU and on CUDA GPU 4. Evidence is recorded in
[`RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md`](RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md).

To rerun the private-HF smoke after any further docs/tag changes and before
public visibility:

```bash
SMOKE_OUTPUT_DIR=/tmp/release-v01-final-v01-docker-smoke-$(date -u +%Y%m%d%H%M%S) \
SMOKE_ALLOW_LOCAL_HF_TOKEN_FILE=1 \
scripts/docker_private_hf_smoke.sh
```

To pin a specific GPU:

```bash
SMOKE_OUTPUT_DIR=/tmp/release-v01-final-v01-docker-smoke-$(date -u +%Y%m%d%H%M%S) \
SMOKE_ALLOW_LOCAL_HF_TOKEN_FILE=1 \
SMOKE_GPU_DEVICE=<gpu-id> \
scripts/docker_private_hf_smoke.sh
```

The smoke harness now loads `revision="v0.1"` for all three private HF repos and
verifies the resolved SHAs listed above. It passes auth to containers only via
runtime `HF_TOKEN` and uses fresh Docker HF cache volumes.

## Public-Release Checklist

- [x] Final private-HF Docker smoke passes against the `v0.1` SHAs in this hub.
- [ ] Readiness report confirms no legacy namespace or old single-repo release
      target remains on public-facing release surfaces.
- [ ] Paper PDF is built from `paper/main.typ`, reviewed, and attached at the
      release asset URL above; generated PDFs are not committed.
- [ ] GitHub README, paper release paragraph, local release docs, and all three
      HF model cards point to this hub, the GitHub repository, the paper PDF
      target, and the three HF repositories.
- [ ] Explicit human approval for public HF visibility is present and logged.
- [ ] Only after approval, flip visibility for the three HF repositories; do not
      run visibility commands during docs, smoke, or readiness tasks.
- [ ] After visibility flip, re-read HF repo metadata and cards to confirm the
      repositories are public and the `v0.1` tags still resolve to the intended
      SHAs.

## Artifact Guardrails

Do not commit tokens, checkpoints, `.pt`/`.pth` files, safetensors, HF caches,
Docker layers, generated PDFs, token files, or other large generated artifacts.
Release evidence should be recorded as small Markdown reports and sanitized JSON
summaries only.
