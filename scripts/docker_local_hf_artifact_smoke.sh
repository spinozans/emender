#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${SMOKE_IMAGE:-ndm-release-v02-local-hf-artifact-smoke:20260529}"
DOCKERFILE="${SMOKE_DOCKERFILE:-$ROOT/docker/release-v02-local-hf-artifact-smoke.Dockerfile}"
ARTIFACT_BASE="${SMOKE_ARTIFACT_BASE:-/tmp/release-v02-local-hf-candidates-${USER:-agent}}"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/tmp/release-v02-docker-local-hf-artifact-smoke-${USER:-agent}}"
PROMPT="${SMOKE_PROMPT:-The theorem states}"
MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-2}"
RUN_GPU="${SMOKE_RUN_GPU:-auto}"
GPU_DEVICE="${SMOKE_GPU_DEVICE:-0}"

E88_ARTIFACT_DIR="${E88_ARTIFACT_DIR:-$ARTIFACT_BASE/e88}"
GDN_ARTIFACT_DIR="${GDN_ARTIFACT_DIR:-$ARTIFACT_BASE/gdn}"
M2RNN_ARTIFACT_DIR="${M2RNN_ARTIFACT_DIR:-$ARTIFACT_BASE/m2rnn}"

mkdir -p "$OUTPUT_DIR"
COMMAND_LOG="$OUTPUT_DIR/commands.txt"
SUMMARY_TXT="$OUTPUT_DIR/summary.txt"
SUMMARY_JSON="$OUTPUT_DIR/summary.json"
: >"$COMMAND_LOG"
: >"$SUMMARY_TXT"

quote_cmd() {
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf '%q' "$arg")")
  done
  printf '%s\n' "${quoted[*]}"
}

run_logged() {
  printf '$ %s\n' "$(quote_cmd "$@")" | tee -a "$COMMAND_LOG"
  "$@"
}

require_artifact_dir() {
  local label="$1"
  local dir="$2"
  if [[ ! -r "$dir/config.json" ]]; then
    printf 'Missing readable config for %s: %s/config.json\n' "$label" "$dir" >&2
    exit 2
  fi
  if [[ ! -r "$dir/model.safetensors" ]]; then
    printf 'Missing readable safetensors for %s: %s/model.safetensors\n' "$label" "$dir" >&2
    exit 2
  fi
  if [[ ! -r "$dir/tokenizer.json" ]]; then
    printf 'Missing readable tokenizer for %s: %s/tokenizer.json\n' "$label" "$dir" >&2
    exit 2
  fi
}

require_artifact_dir e88 "$E88_ARTIFACT_DIR"
require_artifact_dir gdn "$GDN_ARTIFACT_DIR"
require_artifact_dir m2rnn "$M2RNN_ARTIFACT_DIR"

run_logged docker build -f "$DOCKERFILE" -t "$IMAGE" "$ROOT"

base_run=(
  docker run --rm
  --network none
  --mount "type=bind,src=$E88_ARTIFACT_DIR,dst=/artifacts/e88,readonly"
  --mount "type=bind,src=$GDN_ARTIFACT_DIR,dst=/artifacts/gdn,readonly"
  --mount "type=bind,src=$M2RNN_ARTIFACT_DIR,dst=/artifacts/m2rnn,readonly"
  --mount "type=bind,src=$OUTPUT_DIR,dst=/outputs"
  -e HF_HOME=/hf-cache
  -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub
  -e TRANSFORMERS_CACHE=/hf-cache/transformers
  -e HF_HUB_DISABLE_TELEMETRY=1
)

cache_volumes=()
cleanup() {
  local vol
  for vol in "${cache_volumes[@]:-}"; do
    docker volume rm "$vol" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

create_cache_volume() {
  local name="$1"
  docker volume rm "$name" >/dev/null 2>&1 || true
  run_logged docker volume create "$name" >/dev/null
  cache_volumes+=("$name")
}

run_model() {
  local model="$1"
  local device="$2"
  local artifact_dir="/artifacts/$model"
  local cache_volume="$3"
  local output="/outputs/${model}_${device}.json"
  local args=(
    --model "$model"
    --artifact-dir "$artifact_dir"
    --device "$device"
    --prompt "$PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --json-out "$output"
  )
  local docker_cmd=(
    "${base_run[@]}"
    --mount "type=volume,src=$cache_volume,dst=/hf-cache"
  )
  if [[ "$device" == "cuda" ]]; then
    docker_cmd+=(--gpus "device=$GPU_DEVICE")
  fi
  docker_cmd+=("$IMAGE" "${args[@]}")
  run_logged "${docker_cmd[@]}"
}

run_suite() {
  local device="$1"
  local cache_volume="$2"
  create_cache_volume "$cache_volume"
  run_model e88 "$device" "$cache_volume"
  run_model gdn "$device" "$cache_volume"
  run_model m2rnn "$device" "$cache_volume"
}

suffix="$(date -u +%Y%m%d%H%M%S)-$$"
cpu_cache="ndm-v02-local-hf-smoke-cpu-$suffix"
gpu_cache="ndm-v02-local-hf-smoke-gpu-$suffix"

run_suite cpu "$cpu_cache"

gpu_status="not_requested"
if [[ "$RUN_GPU" != "0" && "$RUN_GPU" != "false" && "$RUN_GPU" != "no" ]]; then
  if docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
    gpu_probe=(
      docker run --rm
      --gpus "device=$GPU_DEVICE"
      --network none
      --entrypoint python
      "$IMAGE"
      -c 'import json, torch; print(json.dumps({"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})); raise SystemExit(0 if torch.cuda.is_available() else 1)'
    )
    printf '$ %s\n' "$(quote_cmd "${gpu_probe[@]}")" | tee -a "$COMMAND_LOG"
    if "${gpu_probe[@]}" | tee "$OUTPUT_DIR/gpu_probe.txt"; then
      gpu_status="available"
      run_suite cuda "$gpu_cache"
    else
      gpu_status="unavailable_cuda_probe_failed"
      printf 'GPU probe failed despite nvidia runtime; see %s/gpu_probe.txt\n' "$OUTPUT_DIR" | tee "$OUTPUT_DIR/gpu_unavailable.txt"
      if [[ "$RUN_GPU" == "1" || "$RUN_GPU" == "true" || "$RUN_GPU" == "yes" ]]; then
        exit 4
      fi
    fi
  else
    gpu_status="unavailable_no_nvidia_runtime"
    {
      printf 'GPU runtime unavailable: docker info runtimes did not include nvidia.\n'
      docker info --format '{{json .Runtimes}}'
      printf '\n'
    } | tee "$OUTPUT_DIR/gpu_unavailable.txt"
  fi
fi

python - "$OUTPUT_DIR" "$gpu_status" "$SUMMARY_JSON" "$SUMMARY_TXT" "$cpu_cache" "$gpu_cache" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
gpu_status = sys.argv[2]
summary_json = pathlib.Path(sys.argv[3])
summary_txt = pathlib.Path(sys.argv[4])
cpu_cache = sys.argv[5]
gpu_cache = sys.argv[6]

rows = []
ok = True
for path in sorted(output_dir.glob("*.json")):
    if path.name == "summary.json":
        continue
    data = json.loads(path.read_text())
    row = {
        "file": path.name,
        "ok": data.get("ok"),
        "model": data.get("model"),
        "artifact_dir": data.get("artifact_dir"),
        "device": data.get("device"),
        "model_class": data.get("model_class"),
        "core_model_class": data.get("core_model_class"),
        "model_identity": data.get("model_identity"),
        "release_candidate": data.get("release_candidate"),
        "source_checkpoint_sha256": data.get("source_checkpoint_sha256"),
        "checkpoint_step": data.get("checkpoint_step"),
        "checkpoint_loss": data.get("checkpoint_loss"),
        "param_count": data.get("param_count"),
        "dtype_sample": data.get("dtype_sample"),
        "generated_new_token_ids": data.get("generated_new_token_ids"),
        "generated_new_text_repr": data.get("generated_new_text_repr"),
        "all_logits_finite": data.get("all_logits_finite"),
        "missing_keys": data.get("missing_keys"),
        "unexpected_keys": data.get("unexpected_keys"),
        "mismatched_keys": data.get("mismatched_keys"),
        "cpu_fallbacks": data.get("cpu_fallbacks"),
    }
    rows.append(row)
    ok = ok and bool(row["ok"])
    ok = ok and not row["missing_keys"] and not row["unexpected_keys"] and not row["mismatched_keys"]

expected = {
    ("e88", "cpu"),
    ("gdn", "cpu"),
    ("m2rnn", "cpu"),
}
if gpu_status == "available":
    expected |= {
        ("e88", "cuda"),
        ("gdn", "cuda"),
        ("m2rnn", "cuda"),
    }
present = {(row["model"], row["device"]) for row in rows}
ok = ok and expected.issubset(present)

summary = {
    "ok": ok,
    "gpu_status": gpu_status,
    "cpu_cache_volume": cpu_cache,
    "gpu_cache_volume": gpu_cache if gpu_status == "available" else None,
    "rows": rows,
}
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with summary_txt.open("w") as fh:
    print(f"ok={ok} gpu_status={gpu_status}", file=fh)
    print(f"cpu_cache_volume={cpu_cache}", file=fh)
    if gpu_status == "available":
        print(f"gpu_cache_volume={gpu_cache}", file=fh)
    for row in rows:
        print(
            row["file"],
            row["ok"],
            row["model"],
            row["device"],
            row["checkpoint_step"],
            row["model_class"],
            row["core_model_class"],
            row["generated_new_token_ids"],
            row["generated_new_text_repr"],
            row["all_logits_finite"],
            file=fh,
        )
print(summary_txt.read_text(), end="")
raise SystemExit(0 if ok else 3)
PY

printf 'Commands: %s\n' "$COMMAND_LOG"
printf 'Summary: %s\n' "$SUMMARY_TXT"
printf 'JSON summary: %s\n' "$SUMMARY_JSON"
