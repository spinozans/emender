#!/usr/bin/env bash
# Post-search GPU work under an exclusive lease from cards 2/3, detached:
#   usage: run_cmaes_followups_leased.sh STEP [args...]
# where STEP is:
#   diagnosis CONFIG LABEL OUT   - one instrumented probe run (cmaes_diagnosis)
#   confirm   CONFIG LABEL OUT   - one 2700 s confirmation run (cmaes_search confirm)
# The wrapper mirrors run_matched_arms_leased.sh: acquire inside the detached
# process, release on exit.
set -euo pipefail
cd /home/erikg/emender
STEP=${1:?step: diagnosis|confirm}; shift
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
echo "leased card: $GPU_LEASE_HELD"
status=0
case "$STEP" in
  diagnosis)
    /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.cmaes_diagnosis \
      run --config "$1" --label "$2" --out "$3" || status=$? ;;
  confirm)
    /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.cmaes_search \
      confirm --config "$1" --label "$2" --output "$3" || status=$? ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
exit "$status"
