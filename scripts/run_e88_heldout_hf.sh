#!/usr/bin/env bash
# Launch e88-heldout-hf BPB measurement on GPU 0 only.
# The HF v0.3 trust_remote_code modeling_ndm.py does
#   importlib.import_module("ndm.models.ladder_lm")
# so the bundled "modeling" code is a thin wrapper that REQUIRES the private
# `ndm` training package to be importable. We therefore run with the emender
# training venv (which has ndm + training deps + CUDA torch), and also export
# PYTHONPATH=/home/erikg/emender as a fallback so `import ndm...` resolves.
set -uo pipefail
cd /home/erikg/ndm/.wg-worktrees/agent-744
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/erikg/emender${PYTHONPATH:+:$PYTHONPATH}"

LOG=/home/erikg/ndm/.wg-worktrees/agent-744/scripts/e88_heldout_hf.log
: > "$LOG"

# Discover a python with ndm + torch(CUDA) + transformers. Prefer the emender
# training venv where the v0.3 forward is known-good.
PYBIN=""
for cand in \
    /home/erikg/emender/.venv/bin/python \
    python3 python ; do
  if [ -n "$cand" ] && command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import ndm.models.ladder_lm, torch, transformers; assert torch.cuda.is_available()" >/dev/null 2>&1; then
      PYBIN="$cand"; break
    fi
  fi
done

{
  echo "launcher: PYBIN=${PYBIN:-NONE}"
  echo "launcher: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "launcher: PYTHONPATH=$PYTHONPATH"
  if [ -z "$PYBIN" ]; then
    echo "FATAL: no python with ndm+torch(CUDA)+transformers found"
    echo "--- diagnostics ---"
    /home/erikg/emender/.venv/bin/python -c "import ndm,torch,transformers;print('emender venv:',ndm.__file__,torch.__version__,torch.cuda.is_available())" 2>&1 || true
    python3 -c "import ndm" 2>&1 || true
    exit 3
  fi
  "$PYBIN" -c "import ndm, torch, transformers; print('ndm', ndm.__file__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
} >> "$LOG" 2>&1

if [ -z "$PYBIN" ]; then exit 3; fi

"$PYBIN" /home/erikg/ndm/.wg-worktrees/agent-744/scripts/e88_heldout_hf_bpb.py >> "$LOG" 2>&1
echo "launcher: exit_code=$?" >> "$LOG"
