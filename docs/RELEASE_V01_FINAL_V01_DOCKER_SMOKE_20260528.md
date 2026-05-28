# Final v0.1 Docker Private-HF Smoke

Date: 2026-05-28 UTC
Task: `release-v01-final-v01-docker-smoke`

This note records the final release-candidate Docker smoke against the private
Hugging Face `v0.1` revisions for the three 1.27B-class model repositories
after model-card/docs polish.

No Hugging Face repository visibility was changed. No tag update was required:
the private `v0.1` tags already resolved to the intended docs-polish staging
commits recorded by `release-v01-model-card-docs-polish`, and the Docker smoke
loaded artifacts from those exact revisions successfully on CPU and GPU.

No token values, checkpoints, safetensors files, Hugging Face caches, Docker
layers, generated PDFs, or other large generated artifacts were copied into the
repository, staged, or committed.

## Scope

Repositories tested:

| Model | Hugging Face repository | Required revision | Resolved `v0.1` SHA |
| --- | --- | --- | --- |
| Emender/E88 | `poietic-pbc/emender-e88-1.27b` | `v0.1` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` |
| GDN | `poietic-pbc/gdn-1.27b` | `v0.1` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.27b` | `v0.1` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` |

Every row below was loaded with exactly `revision="v0.1"`. The smoke script
called `HfApi().repo_info(..., revision="v0.1", token=token)` before loading
and required `private=True` and the expected SHA match.

## Runtime

Host/container runtime:

- Docker server: `27.2.0`
- Docker NVIDIA runtime: present in `docker info --format '{{json .Runtimes}}'`
- GPU selected for container smoke: physical GPU `4`, passed as
  `--gpus device=4`
- GPU probe inside the image:

```json
{"cuda_available": true, "device_count": 1, "device_name": "NVIDIA RTX 6000 Ada Generation"}
```

Smoke image:

- Tag: `ndm-release-v01-private-hf-smoke:20260527`
- Image ID: `sha256:b8fc1eeb9292108377060dbb708e72eb07f44024c0fe523e5efcc83eeeb7649c`
- Local image size from `docker image inspect`: `8651997976` bytes
- Docker layers are local Docker daemon state only and are not tracked by git.

Evidence directory outside the repository:

```text
/tmp/release-v01-final-v01-docker-smoke-agent-433/
```

The transcript directory is `48K` total and contains only command logs, short
JSON smoke outputs, a GPU probe, and summaries:

```text
commands.txt 4K
e88_cpu.json 4K
e88_cuda.json 4K
gdn_cpu.json 4K
gdn_cuda.json 4K
gpu_probe.txt 4K
m2rnn_cpu.json 4K
m2rnn_cuda.json 4K
summary.json 8K
summary.txt 4K
```

Fresh Docker HF cache volumes used by the successful run:

- CPU: `ndm-private-hf-smoke-cpu-20260528003501-3534365`
- GPU: `ndm-private-hf-smoke-gpu-20260528003501-3534365`

Both cache volumes were created immediately before their suites and removed by
the harness exit trap. A post-run
`docker volume ls --format '{{.Name}}' | rg '^ndm-private-hf-smoke-'` returned
no matching volumes.

## Authentication And Guardrails

The executor environment did not have `HF_TOKEN` exported, so the harness was
invoked with `SMOKE_ALLOW_LOCAL_HF_TOKEN_FILE=1`; it read the local HF token
file into the process environment and passed authentication to containers only
as runtime `-e HF_TOKEN`. The token value was not printed in the command
transcript, not passed as a Docker build argument, and not written to JSON
evidence.

A post-run scan compared the current token string with every file in the
evidence directory and returned:

```text
token_string_in_evidence=absent
```

The Docker build step did not receive `HF_TOKEN`. The Docker run transcript
contains `-e HF_TOKEN`, not a token value.

The smoke code used read/load APIs only:

- `HfApi().repo_info(..., token=token)`
- `HfApi().list_repo_refs(..., token=token)`
- `AutoConfig.from_pretrained(..., token=token)`
- `AutoTokenizer.from_pretrained(..., token=token)`
- `AutoModelForCausalLM.from_pretrained(..., token=token)`

No `update_repo_visibility`, `update_repo_settings(private=False)`,
`huggingface-cli repo update`, or equivalent visibility-changing command was
run.

## Exact Commands

Top-level harness invocation:

```bash
SMOKE_OUTPUT_DIR=/tmp/release-v01-final-v01-docker-smoke-agent-433 SMOKE_GPU_DEVICE=4 SMOKE_ALLOW_LOCAL_HF_TOKEN_FILE=1 scripts/docker_private_hf_smoke.sh
```

Exact build command:

```bash
docker build -f /home/erikg/ndm/.wg-worktrees/agent-433/docker/release-v01-private-hf-smoke.Dockerfile -t ndm-release-v01-private-hf-smoke:20260527 /home/erikg/ndm/.wg-worktrees/agent-433
```

Exact CPU cache and run commands:

```bash
docker volume create ndm-private-hf-smoke-cpu-20260528003501-3534365
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-final-v01-docker-smoke-agent-433\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-cpu-20260528003501-3534365\,dst=/hf-cache ndm-release-v01-private-hf-smoke:20260527 --model e88 --device cpu --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/e88_cpu.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-final-v01-docker-smoke-agent-433\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-cpu-20260528003501-3534365\,dst=/hf-cache ndm-release-v01-private-hf-smoke:20260527 --model gdn --device cpu --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/gdn_cpu.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-final-v01-docker-smoke-agent-433\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-cpu-20260528003501-3534365\,dst=/hf-cache ndm-release-v01-private-hf-smoke:20260527 --model m2rnn --device cpu --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/m2rnn_cpu.json
```

Exact GPU probe, cache, and run commands:

```bash
docker run --rm --gpus device=4 --network bridge --entrypoint python ndm-release-v01-private-hf-smoke:20260527 -c import\ json\,\ torch\;\ print\(json.dumps\(\{\"cuda_available\":\ torch.cuda.is_available\(\)\,\ \"device_count\":\ torch.cuda.device_count\(\)\,\ \"device_name\":\ torch.cuda.get_device_name\(0\)\ if\ torch.cuda.is_available\(\)\ else\ None\}\)\)\;\ raise\ SystemExit\(0\ if\ torch.cuda.is_available\(\)\ else\ 1\)
docker volume create ndm-private-hf-smoke-gpu-20260528003501-3534365
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-final-v01-docker-smoke-agent-433\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-gpu-20260528003501-3534365\,dst=/hf-cache --gpus device=4 ndm-release-v01-private-hf-smoke:20260527 --model e88 --device cuda --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/e88_cuda.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-final-v01-docker-smoke-agent-433\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-gpu-20260528003501-3534365\,dst=/hf-cache --gpus device=4 ndm-release-v01-private-hf-smoke:20260527 --model gdn --device cuda --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/gdn_cuda.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-final-v01-docker-smoke-agent-433\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-gpu-20260528003501-3534365\,dst=/hf-cache --gpus device=4 ndm-release-v01-private-hf-smoke:20260527 --model m2rnn --device cuda --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/m2rnn_cuda.json
```

The Docker run commands mounted only the small output directory and the fresh
HF cache volume. They did not mount local checkpoint directories, local
safetensors, or host Hugging Face caches.

## Results

All smokes used:

- Prompt: `The theorem states`
- Prompt token IDs: `[464, 44728, 2585]`
- Generation mode: greedy, full-context forward loop, `max_new_tokens=2`
- Success condition: exact private `v0.1` revision resolves, repo readback is
  `private=True`, generated text is nonempty, and logits are finite at every
  generation step.

Summary output:

```text
ok=True gpu_status=available
cpu_cache_volume=ndm-private-hf-smoke-cpu-20260528003501-3534365
gpu_cache_volume=ndm-private-hf-smoke-gpu-20260528003501-3534365
e88_cpu.json True e88 cpu v0.1 a2e56cb82eec5e01ae6eb501569359c5ff64af6b NdmForCausalLM ndm.models.ladder_lm.LadderLM [218, 218] '\x1e\x1e' True
e88_cuda.json True e88 cuda v0.1 a2e56cb82eec5e01ae6eb501569359c5ff64af6b NdmForCausalLM ndm.models.ladder_lm.LadderLM [218, 218] '\x1e\x1e' True
gdn_cpu.json True gdn cpu v0.1 556df7f00969c6a8dbeb381e3c8b51cf0c0385f9 NdmForCausalLM ndm.models.ladder_lm.LadderLM [318, 318] ' is is' True
gdn_cuda.json True gdn cuda v0.1 556df7f00969c6a8dbeb381e3c8b51cf0c0385f9 NdmForCausalLM ndm.models.ladder_lm.LadderLM [318, 318] ' is is' True
m2rnn_cpu.json True m2rnn cpu v0.1 8181b77803e130ffd78e37c33aa4d58c27e719c2 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [2109, 34059] '........Officers' True
m2rnn_cuda.json True m2rnn cuda v0.1 8181b77803e130ffd78e37c33aa4d58c27e719c2 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [2109, 34059] '........Officers' True
```

| Model | Repo | Revision | Device | Status | Model/core class | Param count | New token IDs | Decoded new text | Finite logits | Private |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| Emender/E88 | `poietic-pbc/emender-e88-1.27b` | `v0.1` / `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | CPU | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,273,191,856 | `[218, 218]` | `'\x1e\x1e'` | true | true |
| Emender/E88 | `poietic-pbc/emender-e88-1.27b` | `v0.1` / `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | CUDA GPU 4 | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,273,191,856 | `[218, 218]` | `'\x1e\x1e'` | true | true |
| GDN | `poietic-pbc/gdn-1.27b` | `v0.1` / `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | CPU | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,352,352,498 | `[318, 318]` | `' is is'` | true | true |
| GDN | `poietic-pbc/gdn-1.27b` | `v0.1` / `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | CUDA GPU 4 | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,352,352,498 | `[318, 318]` | `' is is'` | true | true |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.27b` | `v0.1` / `8181b77803e130ffd78e37c33aa4d58c27e719c2` | CPU | PASS | `NdmForCausalLM` / `ndm.models.m2rnn_baseline.M2RNNLM` | 1,307,101,140 | `[2109, 34059]` | `'........Officers'` | true | true |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.27b` | `v0.1` / `8181b77803e130ffd78e37c33aa4d58c27e719c2` | CUDA GPU 4 | PASS | `NdmForCausalLM` / `ndm.models.m2rnn_baseline.M2RNNLM` | 1,307,101,140 | `[2109, 34059]` | `'........Officers'` | true | true |

Remote metadata readback for every row reported:

- `private: true`
- `resolved_sha` exactly equal to the expected docs-polish `v0.1` SHA
- tags containing `v0.1`
- config class `NdmConfig`
- tokenizer class `PreTrainedTokenizerFast`

CPU-specific fallbacks:

- Emender/E88 CPU disabled `LadderLM.fused_add_norm`.
- GDN CPU disabled `LadderLM.fused_add_norm` and installed the Python CPU
  FLA-GDN forward on `21` layers.
- M2RNN-CMA CPU required no fallback.

GPU runs used `torch.bfloat16`; CPU runs used `torch.float32`. No NaNs,
infinities, tracebacks, or crashes occurred in the final six smoke runs.

## Validation Checklist

- [x] Emender/E88 loads from private HF with `revision="v0.1"` in Docker and
      generates on CPU.
- [x] GDN loads from private HF with `revision="v0.1"` in Docker and generates
      on CPU.
- [x] M2RNN-CMA loads from private HF with `revision="v0.1"` in Docker and
      generates on CPU.
- [x] GPU Docker smokes passed for all three models on Docker GPU device `4`.
- [x] `v0.1` tags resolve to the intended docs-polish commits listed in the
      release hub.
- [x] Repositories remain `private=True`; no visibility-change command was run.
- [x] Exact commands, sanitized logs, tested SHAs, and sample outputs are
      documented above.
- [x] No tokens, checkpoints, safetensors, HF caches, Docker layers, generated
      PDFs, or large artifacts are committed.
