#!/usr/bin/env bash
set -euo pipefail
repo=/home/erikg/emender
bound=/home/erikg/emender_c14_unbudgeted_bound
receipt=experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_v2_unbudgeted
cd "$repo"
/home/erikg/emender/.venv/bin/python - <<'CHECK'
from pathlib import Path
import json,hashlib
r=Path('experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_v2_unbudgeted')
s=json.loads((r/'RESUME_STATE.json').read_text())
for name,key in [('archive','archive_sha256'),('checkpoint','checkpoint_sha256')]:
 assert hashlib.sha256(Path(s[name]).read_bytes()).hexdigest()==s[key],name
CHECK
[[ ! -e "$bound" ]]
git worktree add --detach "$bound" c190956d3f970efe18e4377d6d80ac2fab83e5ba
tar -xzf /home/erikg/c14_dense_bytes/c14_b_v2_unbudgeted_source_receipts.tar.gz -C "$bound/experiments/expressivity_tasks/x43_fork_resolved"
cd "$bound"
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
bash experiments/expressivity_tasks/x43_fork_resolved/run_c14_b_controls_leased.sh unbudgeted eval --resume
# Exit 0 may denote a graceful pause. Require eval_resume.status=complete,
# exact denominators and completed verdicts before harvesting or claiming success.
