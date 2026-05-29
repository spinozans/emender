# v0.1 Public Release Hub

Date: 2026-05-28 UTC
Task: `release-v01-model-card-docs-polish`

This is the central v0.1 release handoff for the Emender paper, GitHub
repository, and three public Hugging Face model repositories. The HF
repositories were renamed to the `1.3b` public slugs on 2026-05-28; the
immutable `v0.1` tags continue to resolve to the same release commits.
Metric-only racer references were refreshed on 2026-05-29 from the live
Figure 2 logs without moving checkpoint tags or uploading weights. The current
paper labels use 100K-step trailing endpoint averages; 10K/50K/100K values are
recorded side by side in the Figure 2 source.

## Primary Links

- GitHub repository: <https://github.com/poietic-pbc/emender>
- Paper PDF target: <https://github.com/poietic-pbc/emender/releases/download/v0.1/Garrison_2026_Emender.pdf>
- Paper source: <https://github.com/poietic-pbc/emender/blob/main/paper/main.typ>
- v0.1 racer/checkpoint source: <https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V01_RACER_CHECKPOINT_PIN_20260527.md>
- Current Figure 2 metric source: <https://github.com/poietic-pbc/emender/blob/main/paper/results/figure_2/AS_OF.md>

The paper PDF target is the intended public-release asset location. Until that
asset is attached, `paper/main.typ` is the source of record and `paper/build.sh`
builds `paper/Garrison_2026_Emender-<commit>.pdf`. Generated PDFs remain
gitignored and must not be committed.

## Hugging Face Models

The repositories are public and the `v0.1` tags point at docs-only commits
descended from the previously smoke-tested artifact commits.

| Model | HF repository | Release revision | Current `v0.1` resolved SHA | Main-card SHA |
| --- | --- | --- | --- | --- |
| Emender/E88 | <https://huggingface.co/poietic-pbc/emender-e88-1.3b> | `v0.1` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | `8e2659df0db5d4f9b555cfa71ea12637d8401268` |
| GDN | <https://huggingface.co/poietic-pbc/gdn-1.3b> | `v0.1` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | `ba59a45656f3353265469ce7e58dce7ee25e6baa` |
| M2RNN-CMA | <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b> | `v0.1` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | `572f62635f29661f599b18948b7e49516b44a574` |

## Model-Card Facts

- Identity: raw/base 1.3B-class recurrent language models; not
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
- Current racer metrics: Emender/E88 `0.977` BPB, GDN `0.970` BPB,
  M2RNN-CMA `0.983` BPB from the refreshed Figure 2 source recorded on
  2026-05-29 in `paper/results/figure_2/AS_OF.md`.
- Loading: all three repositories use custom HF code and must be loaded with
  `revision="v0.1"` and `trust_remote_code=True`.

## Docker Smoke

Final Docker smoke passed on 2026-05-28 against the current `v0.1` SHAs in
this hub on CPU and on CUDA GPU 4 before the slug-only rename. Evidence is
recorded in
[`RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md`](RELEASE_V01_FINAL_V01_DOCKER_SMOKE_20260528.md).

For this slug-only rename, use lightweight public resolver checks instead of
rerunning full generation smoke unless a resolver or load check fails:

```bash
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for repo in [
    "poietic-pbc/emender-e88-1.3b",
    "poietic-pbc/gdn-1.3b",
    "poietic-pbc/m2rnn-cma-1.3b",
]:
    info = api.model_info(repo, revision="v0.1", token=False)
    print(repo, info.private, info.sha)
PY
```

The historical smoke harness loads `revision="v0.1"` and verifies the resolved
SHAs listed above. It can run unauthenticated against the public repositories,
passes auth to containers only via runtime `HF_TOKEN` when one is provided, and
uses fresh Docker HF cache volumes.

## Public-Release Checklist

- [x] Final pre-public Docker smoke passes against the `v0.1` SHAs in this hub.
- [x] HF repositories are public under the `1.3b` slugs and `v0.1` resolves to
      the intended SHAs.
- [x] Current link-sync checks confirm no legacy namespace or old single-repo
      release target remains on public-facing release surfaces.
- [x] Paper PDF is built from `paper/main.typ`, reviewed, and attached at the
      release asset URL above; generated PDFs are not committed.
- [x] GitHub README, paper release paragraph, local release docs, and all three
      HF model cards point to this hub, the GitHub repository, the paper PDF
      target, and the three HF repositories.
- [x] Explicit human approval for public HF visibility is present and logged.
- [x] After visibility flip, re-read HF repo metadata and cards to confirm the
      repositories are public and the `v0.1` tags still resolve to the intended
      SHAs.

## Artifact Guardrails

Do not commit tokens, checkpoints, `.pt`/`.pth` files, safetensors, HF caches,
Docker layers, generated PDFs, token files, or other large generated artifacts.
Release evidence should be recorded as small Markdown reports and sanitized JSON
summaries only.
