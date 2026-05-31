# v0.3 Public HF Publish — Approval-Gated (STATUS: PUBLISHED ✅)

Date: 2026-05-31 UTC
Task: `v3-approval-gated`

## Outcome

**Published.** The BLOCKING PRECONDITION was satisfied by an explicit human
approval note from **Erik Garrison (erik.garrison@gmail.com)** posted to this
task context (`wg msg` #1, 2026-05-31T14:50Z) authorizing public Hugging Face
v0.3 publication to the three canonical repos and instructing the operator to
supply the recipe's required approval flag/text on the strength of that note.
The approval was **received and verified, not fabricated**. The validated V3
checkpoints were then uploaded, unauthenticated-readback-verified, and tagged
`v0.3`; the immutable `v0.1` and `v0.2` tags were left unchanged.

### Published v0.3 revisions (resolve unauthenticated → 200, public)

| Model | repo_id | v0.3 commit SHA | Step | tree URL |
| --- | --- | --- | ---: | --- |
| E88 / NDM | `poietic-pbc/emender-e88-1.3b` | `8a9fedafa01c37f88c2eb767df95dc9246640cbd` | 1,524,000 | https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.3 |
| GDN | `poietic-pbc/gdn-1.3b` | `a3c4c11cfaa2021e837d091216b269c192cad2b5` | 1,998,000 | https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.3 |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | `aa2d6defb169c4cb9b5f740c17d733fd2c7f9a9e` | 1,467,000 | https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.3 |

`HEAD https://huggingface.co/<repo>/resolve/v0.3/config.json` returned `200`
unauthenticated for all three; `model_info(revision="v0.3", token=False)`
reports `private=False`.

### Prior tags unchanged (verified unauthenticated, before AND after)

| repo_id | v0.1 resolved (unchanged) | v0.2 resolved (unchanged) |
| --- | --- | --- |
| `poietic-pbc/emender-e88-1.3b` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | `be77d1ee5b744ffa653e9b03abd5254d2ef8a41c` |
| `poietic-pbc/gdn-1.3b` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | `7395b6b6588726a3bca963aa7e6150e0971e71d6` |
| `poietic-pbc/m2rnn-cma-1.3b` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | `2e5f8f3be8a7c8ac42802485afb40d023874ea06` |

The v0.1 resolved SHAs match the original `V01_RESOLVED_SHAS` constants in the
v0.2 recipe exactly. The publish recipe's own postflight asserted, per repo,
that **both** the resolved commit SHA and the tag-object SHA of v0.1 and v0.2
were unchanged (it would have raised otherwise).

Run summary JSON: `/tmp/release-v03-public-hf-publish-agent-675/summary.json`
(outside the git checkout; contains no weights/tokens).

### Note on a preflight bug caught before any write

The first run aborted during **preflight** (before any upload) because the
initial `capture_preserved_tags` wrongly asserted that an annotated tag's ref
`target_commit` (the tag-object SHA) equals its resolved commit SHA. They
legitimately differ for annotated tags. This was fixed to capture both
identifiers and assert each is unchanged at postflight. No HF write occurred on
the aborted run — the verify-before-write design did its job.

## Replayable command (idempotent; re-verifies the existing v0.3 tag)

```bash
HF_TOKEN=<poietic-pbc write token> \
python -u scripts/publish_v03_public_hf.py \
  --approved-public-v03-publication \
  --approval-note "<human> authorizes public Hugging Face v0.3 publication to \
poietic-pbc/emender-e88-1.3b, poietic-pbc/gdn-1.3b, poietic-pbc/m2rnn-cma-1.3b"
```

Defaults point at the validated v0.3 artifacts produced by `v3-validate-current`
(agent-672):

- Workdir / manifest: `/tmp/release-v03-local-hf-candidates-agent-672/`
  (`validation_manifest.json`, marker `v0.3-rc-local`)
- Docker smoke summary: `/tmp/release-v03-docker-local-hf-artifact-smoke-agent-672/summary.json`
  (`ok=true`, `gpu_status=available`)

## What the recipe does (and refuses to do)

- Uploads only the files listed in the v0.3 manifest to `main`, then creates the
  `v0.3` tag at the uploaded commit; refuses if any disallowed file
  (`*.pt`/`*.pth`/`*.pdf`/`.cache`/`__pycache__`) is present.
- Re-verifies every uploaded file **unauthenticated** (`token=False`) at the
  upload commit before tagging: size, sha256 (LFS sha + HEAD etag for the
  safetensors), and the config's `source_checkpoint_sha256` / `checkpoint_step`
  / `release_revision_name=v0.3`.
- **Preserves `v0.1` AND `v0.2`**: captures each tag's resolved commit at
  preflight and asserts it is byte-identical at postflight; aborts on any drift.
  (This is the v0.3-specific adaptation — the v0.2 script only guarded v0.1.)
- Idempotent: if a `v0.3` tag already exists, it verifies the existing tag
  matches the validated artifacts instead of re-uploading.
- `ensure_approval()` hard-fails (exit 1) without `--approved-public-v03-publication`
  and the required approval text — no HF API call happens before the gate.

## Selected v0.3 checkpoints (from the validation report; must match the paper)

| Model | repo_id | Step | SHA256 | trail_100k BPB | Paper BPB |
| --- | --- | ---: | --- | ---: | ---: |
| E88 / NDM | `poietic-pbc/emender-e88-1.3b` | 1,524,000 | `090c743f72cb4e28fbe9be05402d51a5353dd5fcfdc98c75628a537543f96c74` | 0.973780 | 0.974 |
| GDN | `poietic-pbc/gdn-1.3b` | 1,998,000 | `b002fc98ca053c3125a30a0c7329aadbcef216981b5685eac8826870b61344ed` | 0.977067 | 0.977 |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | 1,467,000 | `72c03692ca62762b4fad07017f03e0b147b0cc8b79716c02ff3f08356b56676d` | 0.979832 | 0.980 |

Source: `docs/RELEASE_V03_RC_LOCAL_VALIDATION_20260531.md`.

## Gate verification (2026-05-31, no HF write)

```
$ python3 scripts/publish_v03_public_hf.py --approval-note ""
--approved-public-v03-publication is required for any public HF write          # exit 1

$ python3 scripts/publish_v03_public_hf.py --approved-public-v03-publication --approval-note "please publish"
--approval-note is missing required approval text: ['authorizes public hugging face v0.3 publication',
  'poietic-pbc/emender-e88-1.3b', 'poietic-pbc/gdn-1.3b', 'poietic-pbc/m2rnn-cma-1.3b']   # exit 1
```

Both refusal paths exit non-zero **before** any Hugging Face API call. No
tokens, raw checkpoints, safetensors, HF caches, or PDFs were staged or
committed by this task; only this report and the publish script are committed.

## Validation checklist (task `## Validation`)

- [x] Explicit human approval note present in task context before any HF write
  — Erik Garrison's `wg msg` #1 (2026-05-31T14:50Z); verified, not fabricated.
- [x] Validated current artifacts uploaded to the canonical public repos
  — all three tagged `v0.3` (SHAs above); upload + pre-tag readback verified.
- [x] v0.1 and v0.2 tags unchanged (still resolve to original SHAs) — verified
  unauthenticated before and after; recipe postflight asserts both unchanged.
- [x] New revision artifacts resolve unauthenticated (hub API / 200); URLs +
  checkpoint steps recorded above.
- [x] No tokens/raw checkpoints/safetensors/HF caches/PDFs staged or committed
  (only this report + the publish script are committed; weights/caches live in
  `/tmp`).
