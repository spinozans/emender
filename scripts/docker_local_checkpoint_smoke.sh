#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${SMOKE_IMAGE:-ndm-release-v01-local-smoke:20260527}"
DOCKERFILE="${SMOKE_DOCKERFILE:-$ROOT/docker/release-v01-local-smoke.Dockerfile}"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-/tmp/release-v01-docker-local-smoke-${USER:-agent}}"
PROMPT="${SMOKE_PROMPT:-The theorem states}"
MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-2}"
RUN_GPU="${SMOKE_RUN_GPU:-auto}"
GPU_DEVICE="${SMOKE_GPU_DEVICE:-4}"

E88_DIR="${E88_CHECKPOINT_DIR:-/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832}"
GDN_DIR="${GDN_CHECKPOINT_DIR:-/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832}"
M2RNN_DIR="${M2RNN_CHECKPOINT_DIR:-/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023}"

E88_CKPT="${E88_CHECKPOINT_FILE:-checkpoint_step_1281000_loss_2.6850.pt}"
GDN_CKPT="${GDN_CHECKPOINT_FILE:-checkpoint_step_1686000_loss_2.6105.pt}"
M2RNN_CKPT="${M2RNN_CHECKPOINT_FILE:-checkpoint_step_1212000_loss_2.6870.pt}"

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

require_checkpoint() {
  local label="$1"
  local dir="$2"
  local file="$3"
  if [[ ! -r "$dir/args.json" ]]; then
    printf 'Missing readable args.json for %s: %s/args.json\n' "$label" "$dir" >&2
    exit 2
  fi
  if [[ ! -r "$dir/$file" ]]; then
    printf 'Missing readable checkpoint for %s: %s/%s\n' "$label" "$dir" "$file" >&2
    exit 2
  fi
}

require_checkpoint e88 "$E88_DIR" "$E88_CKPT"
require_checkpoint gdn "$GDN_DIR" "$GDN_CKPT"
require_checkpoint m2rnn "$M2RNN_DIR" "$M2RNN_CKPT"

run_logged docker build -f "$DOCKERFILE" -t "$IMAGE" "$ROOT"

base_run=(
  docker run --rm
  --network none
  --mount "type=bind,src=$E88_DIR,dst=/checkpoints/e88,readonly"
  --mount "type=bind,src=$GDN_DIR,dst=/checkpoints/gdn,readonly"
  --mount "type=bind,src=$M2RNN_DIR,dst=/checkpoints/m2rnn,readonly"
  --mount "type=bind,src=$OUTPUT_DIR,dst=/outputs"
  -e HF_HOME=/tmp/hf-disabled
  -e TRANSFORMERS_CACHE=/tmp/hf-disabled/transformers
)

run_model() {
  local model="$1"
  local device="$2"
  local checkpoint="$3"
  local output="/outputs/${model}_${device}.json"
  local args=(
    --model "$model"
    --device "$device"
    --checkpoint "$checkpoint"
    --prompt "$PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --json-out "$output"
  )
  if [[ "$device" == "cuda" ]]; then
    run_logged "${base_run[@]}" --gpus "device=$GPU_DEVICE" "$IMAGE" "${args[@]}"
  else
    run_logged "${base_run[@]}" "$IMAGE" "${args[@]}"
  fi
}

run_model e88 cpu /checkpoints/e88/"$E88_CKPT"
run_model gdn cpu /checkpoints/gdn/"$GDN_CKPT"
run_model m2rnn cpu /checkpoints/m2rnn/"$M2RNN_CKPT"

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
      run_model e88 cuda /checkpoints/e88/"$E88_CKPT"
      run_model gdn cuda /checkpoints/gdn/"$GDN_CKPT"
      run_model m2rnn cuda /checkpoints/m2rnn/"$M2RNN_CKPT"
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

python - "$OUTPUT_DIR" "$gpu_status" "$SUMMARY_JSON" "$SUMMARY_TXT" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
gpu_status = sys.argv[2]
summary_json = pathlib.Path(sys.argv[3])
summary_txt = pathlib.Path(sys.argv[4])

rows = []
ok = True
for path in sorted(output_dir.glob("*.json")):
    if path.name == "summary.json":
        continue
    data = json.loads(path.read_text())
    row = {
        "file": path.name,
        "ok": data.get("ok"),
        "label": data.get("label"),
        "device": data.get("device"),
        "generated_new_token_ids": data.get("generated_new_token_ids"),
        "generated_new_text": data.get("generated_new_text"),
        "all_logits_finite": data.get("all_logits_finite"),
        "generated_nonempty": data.get("generated_nonempty"),
        "cuda_available": data.get("cuda_available"),
        "cuda_device_name": data.get("cuda_device_name"),
    }
    rows.append(row)
    ok = ok and bool(row["ok"])

summary = {"ok": ok, "gpu_status": gpu_status, "rows": rows}
summary_json.write_text(json.dumps(summary, indent=2) + "\n")
with summary_txt.open("w") as fh:
    print(f"ok={ok} gpu_status={gpu_status}", file=fh)
    for row in rows:
        print(
            row["file"],
            row["ok"],
            row["label"],
            row["device"],
            row["generated_new_token_ids"],
            repr(row["generated_new_text"]),
            row["all_logits_finite"],
            file=fh,
        )
print(summary_txt.read_text(), end="")
raise SystemExit(0 if ok else 3)
PY

printf 'Commands: %s\n' "$COMMAND_LOG"
printf 'Summary: %s\n' "$SUMMARY_TXT"
printf 'JSON summary: %s\n' "$SUMMARY_JSON"
