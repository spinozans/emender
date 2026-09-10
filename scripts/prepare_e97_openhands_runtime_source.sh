#!/usr/bin/env bash
# Fetch an exact upstream runtime candidate. No installation or tool execution.
set -euo pipefail
umask 077
OUT=${1:?usage: PYTHON_BIN=/approved/python prepare_e97_openhands_runtime_source.sh /absolute/new/root}
PYTHON_BIN=${PYTHON_BIN:?set an explicitly selected Python 3.12 interpreter}
COMMIT=9ee704a25a331d0d2eb9a8e87a4dcff1d948855b
if [[ "$OUT" != /* || -e "$OUT" || -L "$OUT" ]]; then
  echo 'Require a new absolute output directory; failed attempts are never overwritten.' >&2
  exit 2
fi
mkdir -m 700 "$OUT"
git -c core.hooksPath=/dev/null init --quiet "$OUT/upstream"
git -C "$OUT/upstream" config core.hooksPath /dev/null
git -C "$OUT/upstream" remote add origin https://github.com/OpenHands/OpenHands.git
timeout --foreground --signal=TERM --kill-after=10s 180s git -C "$OUT/upstream" fetch --depth 1 --no-tags origin "$COMMIT"
GIT_LFS_SKIP_SMUDGE=1 timeout --foreground --signal=TERM --kill-after=10s 180s git -C "$OUT/upstream" -c core.autocrlf=false checkout --detach FETCH_HEAD
test "$(git -C "$OUT/upstream" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$OUT/upstream" status --porcelain)"
"$PYTHON_BIN" - "$OUT" "$COMMIT" <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);source=root/'upstream';commit=sys.argv[2]
paths=subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode().split('\0')
selected=[p for p in paths if p.endswith('.py') or p in ('LICENSE','pyproject.toml','poetry.lock','README.md')]
files={}
for name in sorted(selected):
    path=source/name
    if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):raise ValueError('nonlocal source entry')
    files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
for required in ('LICENSE','poetry.lock','openhands/runtime/utils/bash.py','openhands/runtime/action_execution_server.py','openhands/agenthub/codeact_agent/function_calling.py'):
    if required not in files:raise ValueError('missing runtime source')
result={'schema':'emender-openhands-source-candidate-v1','repository':'https://github.com/OpenHands/OpenHands',
        'commit':commit,'reference_tag':'0.53.0','files':files,'status':'source-acquired-only',
        'historical_dataset_runtime_identity_verified':False,'execution_qualified':False,
        'dependencies_installed':False,'automatic_retry':False,'source_commands_executed':False}
with (root/'source-manifest.json').open('x') as f:
    json.dump(result,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('OPENHANDS_SOURCE_ACQUIRED '+commit,flush=True)
PY
