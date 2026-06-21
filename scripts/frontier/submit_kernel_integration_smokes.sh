#!/bin/bash
# Submit the three ROCm kernel integration smokes one at a time.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

DATA=${DATA:?Set DATA to a short Frontier-readable smoke training text file}
VAL_DATA=${VAL_DATA:-}
export DATA VAL_DATA

export TRAIN_MINUTES=${TRAIN_MINUTES:-20}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${MEMBERWORK:-$ROOT}/emender/frontier_runs/debug}
if [[ -z "${EMENDER_CONDA_ENV:-}" && -d "${MEMBERWORK:-}/emender/conda/emender-rocm711" ]]; then
  export EMENDER_CONDA_ENV="${MEMBERWORK}/emender/conda/emender-rocm711"
fi
if [[ -z "${GDN2_PATH:-}" && -d "${MEMBERWORK:-}/emender/src/GatedDeltaNet-2" ]]; then
  export GDN2_PATH="${MEMBERWORK}/emender/src/GatedDeltaNet-2"
fi

mkdir -p logs/frontier/debug

variants=(e97-MLP e97-linear-MLP gdn2-MLP)

for variant in "${variants[@]}"; do
  export SMOKE_VARIANT="$variant"
  echo "Submitting ${variant} with DATA=${DATA}"
  job_output=$(sbatch --parsable scripts/frontier/debug_smoke_one_node.slurm)
  job_id=${job_output%%;*}
  echo "Submitted ${variant}: job ${job_id}"

  while true; do
    state=$(squeue -h -j "$job_id" -o "%T" || true)
    if [[ -z "$state" ]]; then
      break
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${variant} ${job_id} ${state}"
    sleep "${POLL_SECONDS:-60}"
  done

  sacct -j "$job_id" \
    --format=JobID,JobName,Partition,QOS,Account,State,Elapsed,AllocNodes,NNodes,NodeList,ExitCode

  final_state=$(sacct -n -X -j "$job_id" --format=State -P | awk -F'|' 'NF {print $1; exit}')
  if [[ "$final_state" != COMPLETED* ]]; then
    echo "${variant} job ${job_id} finished with state ${final_state}; stopping sequence" >&2
    exit 1
  fi
done
