#!/usr/bin/env bash
# Detached leased driver for the SEP-KV key-separation probe (cards 2/3, the
# free pool this lane was assigned; the arms/CMA-ES lanes' cards are never
# touched here).  Runs the three parts sequentially -- part 1 (the depth-1
# necessity sweep: both tasks x both arms x 4 LRs x 3 seeds), part 2 (the
# key-dim sweep on the separated arm at depth 1), part 3 (the depth-2
# depth-substitution corroboration on the fused arm) -- writing receipts to
# receipts/keysep_*.json as they land.  Run this DETACHED (setsid +
# redirected stdio, the long-session survival playbook): the sweep and the
# lease owner survive the session that launched them.  The attended session
# commits each part's receipts and RESULT section as they land.
#
#   usage: run_keysep_leased.sh
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
MOD=experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe

export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none}"

$PY -m $MOD --part 1 --task both --arm both
echo "=== PART 1 DONE ==="
$PY -m $MOD --part 2 --task both --arm separated
echo "=== PART 2 DONE ==="
$PY -m $MOD --part 3 --task both --arm fused
echo "=== PART 3 DONE ==="

GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
echo "lease released; all parts complete"
