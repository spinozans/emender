# v0.1 Docker Local-Checkpoint Smoke

Date: 2026-05-27
Task: `release-v01-docker-local-smoke`

This note records a Docker/container smoke harness for the pinned E88, GDN, and
M2RNN-CMA checkpoints validated by `release-v01-local-three-model-smoke`.
The harness builds an isolated image from the release checkout, mounts local
checkpoint directories read-only, and runs the existing local checkpoint
generation smoke script inside the container.

No HuggingFace or GitHub upload was attempted, no repository visibility was
changed, and no tokens, checkpoints, safetensors, HF caches, Docker layers,
generated PDFs, or other large generated artifacts were copied into the repo,
staged, or committed.

## Harness

Added files:

- `.dockerignore`: excludes VCS metadata, Python/build caches, checkpoints,
  `*.pt`, `*.pth`, `*.safetensors`, logs, generated PDFs, and local runtime
  cache directories from the Docker build context.
- `docker/release-v01-local-smoke.Dockerfile`: builds an isolated PyTorch
  CUDA-runtime image, installs the repo and runtime dependencies inside the
  image, preloads the small `p50k_base` tiktoken encoding for network-disabled
  runs, and uses `scripts/smoke_local_checkpoint_generation.py` as the
  entrypoint.
- `scripts/docker_local_checkpoint_smoke.sh`: builds the image, validates that
  the three checkpoint files and adjacent `args.json` files are readable, runs
  CPU smokes for all three models, probes Docker GPU availability, and runs GPU
  smokes for all three models when CUDA is available.

The final image includes `build-essential` because Triton JIT compilation needs
a C compiler at runtime for the CUDA E88 path.

## Runtime Evidence

Host/container runtime:

- Docker server: `27.2.0`
- Docker NVIDIA runtime: present in `docker info --format '{{json .Runtimes}}'`
- GPU selected for container smoke: physical GPU `4`, passed as
  `--gpus device=4`
- GPU probe inside the image:

```json
{"cuda_available": true, "device_count": 1, "device_name": "NVIDIA RTX 6000 Ada Generation"}
```

Image:

- Tag: `ndm-release-v01-local-smoke:20260527`
- Image ID: `sha256:1dacde8f8d15cafe270e454b34bd165342cdd054fa6ecc13caff0b2e2f6e3b6e`
- Local image size from `docker image inspect`: `8641981302` bytes
- Docker layers are local Docker daemon state only and are not tracked by git.

Evidence directory outside the repo:

```text
/tmp/release-v01-docker-local-smoke-agent-407/
```

The transcript directory is `48K` total and contains only command logs, short
JSON smoke outputs, a GPU probe, and summaries:

```text
commands.txt 5736
e88_cpu.json 1487
e88_cuda.json 1451
gdn_cpu.json 1548
gdn_cuda.json 1452
gpu_probe.txt 93
m2rnn_cpu.json 1474
m2rnn_cuda.json 1475
summary.json 2247
summary.txt 379
```

## Exact Commands

Top-level harness invocation:

```bash
SMOKE_OUTPUT_DIR=/tmp/release-v01-docker-local-smoke-agent-407 SMOKE_GPU_DEVICE=4 scripts/docker_local_checkpoint_smoke.sh
```

The harness wrote the full command transcript to:

```text
/tmp/release-v01-docker-local-smoke-agent-407/commands.txt
```

Exact build command:

```bash
docker build -f /home/erikg/ndm/.wg-worktrees/agent-407/docker/release-v01-local-smoke.Dockerfile -t ndm-release-v01-local-smoke:20260527 /home/erikg/ndm/.wg-worktrees/agent-407
```

Exact CPU run commands:

```bash
docker run --rm --network none --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832\,dst=/checkpoints/e88\,readonly --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832\,dst=/checkpoints/gdn\,readonly --mount type=bind\,src=/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023\,dst=/checkpoints/m2rnn\,readonly --mount type=bind\,src=/tmp/release-v01-docker-local-smoke-agent-407\,dst=/outputs -e HF_HOME=/tmp/hf-disabled -e TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers ndm-release-v01-local-smoke:20260527 --model e88 --device cpu --checkpoint /checkpoints/e88/checkpoint_step_1281000_loss_2.6850.pt --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/e88_cpu.json
docker run --rm --network none --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832\,dst=/checkpoints/e88\,readonly --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832\,dst=/checkpoints/gdn\,readonly --mount type=bind\,src=/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023\,dst=/checkpoints/m2rnn\,readonly --mount type=bind\,src=/tmp/release-v01-docker-local-smoke-agent-407\,dst=/outputs -e HF_HOME=/tmp/hf-disabled -e TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers ndm-release-v01-local-smoke:20260527 --model gdn --device cpu --checkpoint /checkpoints/gdn/checkpoint_step_1686000_loss_2.6105.pt --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/gdn_cpu.json
docker run --rm --network none --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832\,dst=/checkpoints/e88\,readonly --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832\,dst=/checkpoints/gdn\,readonly --mount type=bind\,src=/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023\,dst=/checkpoints/m2rnn\,readonly --mount type=bind\,src=/tmp/release-v01-docker-local-smoke-agent-407\,dst=/outputs -e HF_HOME=/tmp/hf-disabled -e TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers ndm-release-v01-local-smoke:20260527 --model m2rnn --device cpu --checkpoint /checkpoints/m2rnn/checkpoint_step_1212000_loss_2.6870.pt --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/m2rnn_cpu.json
```

Exact GPU probe and run commands:

```bash
docker run --rm --gpus device=4 --network none --entrypoint python ndm-release-v01-local-smoke:20260527 -c import\ json\,\ torch\;\ print\(json.dumps\(\{\"cuda_available\":\ torch.cuda.is_available\(\)\,\ \"device_count\":\ torch.cuda.device_count\(\)\,\ \"device_name\":\ torch.cuda.get_device_name\(0\)\ if\ torch.cuda.is_available\(\)\ else\ None\}\)\)\;\ raise\ SystemExit\(0\ if\ torch.cuda.is_available\(\)\ else\ 1\)
docker run --rm --network none --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832\,dst=/checkpoints/e88\,readonly --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832\,dst=/checkpoints/gdn\,readonly --mount type=bind\,src=/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023\,dst=/checkpoints/m2rnn\,readonly --mount type=bind\,src=/tmp/release-v01-docker-local-smoke-agent-407\,dst=/outputs -e HF_HOME=/tmp/hf-disabled -e TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers --gpus device=4 ndm-release-v01-local-smoke:20260527 --model e88 --device cuda --checkpoint /checkpoints/e88/checkpoint_step_1281000_loss_2.6850.pt --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/e88_cuda.json
docker run --rm --network none --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832\,dst=/checkpoints/e88\,readonly --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832\,dst=/checkpoints/gdn\,readonly --mount type=bind\,src=/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023\,dst=/checkpoints/m2rnn\,readonly --mount type=bind\,src=/tmp/release-v01-docker-local-smoke-agent-407\,dst=/outputs -e HF_HOME=/tmp/hf-disabled -e TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers --gpus device=4 ndm-release-v01-local-smoke:20260527 --model gdn --device cuda --checkpoint /checkpoints/gdn/checkpoint_step_1686000_loss_2.6105.pt --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/gdn_cuda.json
docker run --rm --network none --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832\,dst=/checkpoints/e88\,readonly --mount type=bind\,src=/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832\,dst=/checkpoints/gdn\,readonly --mount type=bind\,src=/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023\,dst=/checkpoints/m2rnn\,readonly --mount type=bind\,src=/tmp/release-v01-docker-local-smoke-agent-407\,dst=/outputs -e HF_HOME=/tmp/hf-disabled -e TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers --gpus device=4 ndm-release-v01-local-smoke:20260527 --model m2rnn --device cuda --checkpoint /checkpoints/m2rnn/checkpoint_step_1212000_loss_2.6870.pt --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/m2rnn_cuda.json
```

All run commands used `--network none` after image build. Checkpoints were
mounted with Docker `readonly` bind mounts. The only writable host mount was the
small transcript output directory under `/tmp`.

## Results

All smokes used:

- Prompt: `The theorem states`
- Prompt token IDs under `p50k_base`: `[464, 44728, 2585]`
- Seed: `20260527`
- Generation mode: greedy, full-context, `max_new_tokens=2`
- Success condition: nonempty decoded new text and finite logits for every
  generation step

Summary output:

```text
ok=True gpu_status=available
e88_cpu.json True e88 cpu [218, 218] '\x1e\x1e' True
e88_cuda.json True e88 cuda [218, 218] '\x1e\x1e' True
gdn_cpu.json True gdn cpu [318, 318] ' is is' True
gdn_cuda.json True gdn cuda [318, 318] ' is is' True
m2rnn_cpu.json True m2rnn cpu [2109, 34059] '........Officers' True
m2rnn_cuda.json True m2rnn cuda [2109, 34059] '........Officers' True
```

| Model | Device | Status | New token IDs | Decoded new text | Finite logits |
| --- | --- | --- | --- | --- | --- |
| E88 | CPU | PASS | `[218, 218]` | `'\x1e\x1e'` | true |
| E88 | CUDA GPU 4 | PASS | `[218, 218]` | `'\x1e\x1e'` | true |
| GDN | CPU | PASS | `[318, 318]` | `' is is'` | true |
| GDN | CUDA GPU 4 | PASS | `[318, 318]` | `' is is'` | true |
| M2RNN-CMA | CPU | PASS | `[2109, 34059]` | `'........Officers'` | true |
| M2RNN-CMA | CUDA GPU 4 | PASS | `[2109, 34059]` | `'........Officers'` | true |

CPU-specific fallbacks in the container matched the prior local smoke:

- E88 CPU disabled `LadderLM.fused_add_norm`.
- GDN CPU disabled `LadderLM.fused_add_norm` and installed the Python CPU
  FLA-GDN forward on `21` layers.
- M2RNN-CMA CPU did not require a fallback.

No NaNs, infinities, tracebacks, or crashes occurred in the final six smoke
runs.

## Checkpoint Mounts

The local checkpoint inputs remained at their existing host paths:

- E88:
  `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1281000_loss_2.6850.pt`
- GDN:
  `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt`
- M2RNN-CMA:
  `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt`

They were exposed inside the container only as:

- `/checkpoints/e88`
- `/checkpoints/gdn`
- `/checkpoints/m2rnn`

Every checkpoint mount used `type=bind,...,readonly`. No checkpoint files,
checkpoint directories, safetensors, or HF cache directories were copied into
the Docker image or repository.

## Guardrail Result

Only the Dockerfile, `.dockerignore`, runner script, and this small Markdown
report are intended for git. The smoke transcripts remain under `/tmp`, the
image remains local Docker daemon state, and no secrets or large generated
artifacts are staged.
