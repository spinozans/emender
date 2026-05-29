# v0.2 Public HF Publish Approval Gate

Date: 2026-05-29 UTC
Task: `approval-gated-public`

This note records the approval-gate result for publishing the locally validated
v0.2 racer artifacts to the existing public Hugging Face repositories.

No public Hugging Face upload, tag creation, visibility change, or model-card
mutation was performed. The task context did not contain explicit human approval
for public v0.2 publication at the time of this run.

This document is evidence of the blocker. It is not approval to publish.

## Approval Check

Checked sources:

- `wg msg read approval-gated-public --agent "$WG_AGENT_ID"`:
  no unread messages.
- `wg show approval-gated-public`:
  task description requires explicit approval before any HF write operation, but
  does not itself contain an explicit approval statement.
- `wg context approval-gated-public`:
  dependency context lists only the local validation artifacts and helpers.

Guardrail decision: blocked. Without a human statement that explicitly approves
public v0.2 publication, HF write operations are not allowed.

No command using `HfApi.upload_*`, `HfApi.create_tag`,
`HfApi.update_repo_visibility`, `huggingface-cli upload`, git push to an HF
remote, or any equivalent HF write path was run.

## Local Candidate State

Local inputs from `validate-v0-2-racer` are still present:

- `/tmp/release-v02-local-hf-candidates-agent-500/validation_manifest.json`
- `/tmp/release-v02-local-hf-candidates-agent-500/{e88,gdn,m2rnn}`
- `/tmp/release-v02-docker-local-hf-artifact-smoke-agent-500/summary.json`

Read-only local verification confirmed the manifest, local safetensors metadata,
and Docker smoke rows agree on the selected source checkpoints.

| Model | Repo | Step | Source checkpoint SHA256 | Local safetensors size | Keys | CPU smoke | GPU smoke |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| E88 / NDM | `poietic-pbc/emender-e88-1.3b` | 1,395,000 | `da847dcefac2d4bb9c077565a6d5f595a9af5187cc19a2dbfa4377b81a2762dc` | 2,713,728,024 | 87 | PASS | PASS |
| GDN | `poietic-pbc/gdn-1.3b` | 1,845,000 | `31a9181f407006b1bef51d2aefa62be9aafd5197845b19154c1a039f564e2c36` | 2,975,047,820 | 297 | PASS | PASS |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | 1,332,000 | `a2a282344e02eb2c237340b4379756d394fda0c4d0c424ddfdba91273030f061` | 2,807,297,000 | 150 | PASS | PASS |

Every local artifact directory contained exactly the manifest-listed files:

- `README.md`
- `config.json`
- `configuration_ndm.py`
- `generation_config.json`
- `model.safetensors`
- `modeling_ndm.py`
- `requirements.txt`
- `special_tokens_map.json`
- `tiktoken/tokenizer.model`
- `tokenizer.json`
- `tokenizer_config.json`

## Public Read-Only State

Unauthenticated Hugging Face read checks used `token=False` and HTTP HEAD
requests only.

The existing public `v0.1` revisions still resolve to the expected SHAs from
`docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md`:

| Model | Repo | Public | `v0.1` resolved SHA | Matches expected | `v0.2` revision |
| --- | --- | --- | --- | --- | --- |
| E88 / NDM | `poietic-pbc/emender-e88-1.3b` | yes | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | yes | absent |
| GDN | `poietic-pbc/gdn-1.3b` | yes | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | yes | absent |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | yes | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | yes | absent |

Unauthenticated `v0.1` file resolution returned HTTP 200 for each repo for:

- `config.json`
- `configuration_ndm.py`
- `modeling_ndm.py`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `tiktoken/tokenizer.model`
- `model.safetensors`

No public `v0.2` files were expected to resolve because the approval gate
blocked upload and tag creation.

## Validation Checklist Result

- [ ] Explicit human approval for public v0.2 upload is present in task context
      before any HF write API/CLI command is run: BLOCKED, approval absent.
- [ ] Upload validated local v0.2 artifacts for E88, GDN, and M2RNN-CMA to the
      existing public 1.3B repos without moving or modifying v0.1 tags:
      not attempted because approval is absent.
- [ ] Create v0.2 tags only after upload readback confirms intended artifacts
      and source checkpoint SHA256 values:
      not attempted because approval is absent.
- [ ] Public v0.2 config/modeling/tokenizer/model.safetensors files resolve
      unauthenticated for all three repos:
      not applicable until upload and tag creation are approved.
- [x] v0.1 tags still resolve to their original public SHAs for all three repos.
- [ ] CPU and GPU Docker generation smoke pass from revision=v0.2 for all three
      public repos from fresh cache/workdir:
      not applicable until public v0.2 revisions exist.
- [x] No tokens, raw checkpoints, local safetensors, HF caches, Docker layers,
      or generated PDFs were copied into the repository for this task.

## Next Approved Run

The next run should proceed only after a human adds an explicit approval message
to the task context, for example:

```text
I approve public v0.2 publication for poietic-pbc/emender-e88-1.3b,
poietic-pbc/gdn-1.3b, and poietic-pbc/m2rnn-cma-1.3b from the validated local
artifacts under /tmp/release-v02-local-hf-candidates-agent-500.
```

After that approval is present, the publish run should:

1. Re-read the approval source and log it before any HF write operation.
2. Re-validate the local artifact manifest, safetensors metadata, and Docker
   smoke summary.
3. Upload the local v0.2 files to the three existing public repos without
   touching the `v0.1` tags.
4. Read back uploaded artifacts and confirm the intended source checkpoint
   SHA256 metadata for each repo.
5. Create `v0.2` tags only after readback passes.
6. Confirm unauthenticated public resolution for the v0.2 config, custom code,
   tokenizer files, and `model.safetensors`.
7. Confirm `v0.1` still resolves to the original public SHAs listed above.
8. Run fresh-cache CPU and GPU Docker generation smoke against
   `revision="v0.2"` for all three public repos.
