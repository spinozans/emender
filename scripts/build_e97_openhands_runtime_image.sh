#!/usr/bin/env bash
# Build only. The output is NOT a runtime qualification or training admission.
set -euo pipefail
umask 077
SOURCE_ROOT=${1:?usage: build_e97_openhands_runtime_image.sh /source-root /new-output-root}
OUT=${2:?new absolute output directory required}
PYTHON_BIN=${PYTHON_BIN:?select an approved Python 3.12 interpreter}
UVX_BIN=${UVX_BIN:?select uvx explicitly}
DOCKERFILE=${DOCKERFILE:?select the frozen Dockerfile explicitly}
SOURCE_MANIFEST_SHA256=${SOURCE_MANIFEST_SHA256:?bind the source manifest}
IMAGE_TAG=${IMAGE_TAG:-emender-openhands-native-candidate:source-9ee704a25a33-v1}
[[ "$IMAGE_TAG" =~ ^emender-openhands-native-candidate:source-9ee704a25a33-v[1-9][0-9]*$ ]] || { echo 'Require a versioned candidate image tag' >&2; exit 2; }
DOCKER=(docker --host unix:///var/run/docker.sock)
if [[ "$OUT" != /* || -e "$OUT" || -L "$OUT" ]]; then echo 'Refuse existing/nonabsolute output' >&2; exit 2; fi
if "${DOCKER[@]}" image inspect "$IMAGE_TAG" >/dev/null 2>&1; then echo 'Refuse existing candidate image tag' >&2; exit 2; fi
mkdir -m 700 "$OUT"
verify_source() {
  "$PYTHON_BIN" - "$SOURCE_ROOT" "$SOURCE_MANIFEST_SHA256" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);p=root/'source-manifest.json'
if hashlib.sha256(p.read_bytes()).hexdigest()!=sys.argv[2]:raise ValueError('source_manifest_identity')
m=json.loads(p.read_text())
if m['commit']!='9ee704a25a331d0d2eb9a8e87a4dcff1d948855b':raise ValueError('wrong_source_commit')
for name,digest in m['files'].items():
    p=root/'upstream'/name
    if p.is_symlink() or not p.resolve().is_relative_to((root/'upstream').resolve()):raise ValueError('nonlocal_source')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('source_file_identity: '+name)
PY
}
verify_source
export UV_CACHE_DIR="$OUT/uv-cache" UV_TOOL_DIR="$OUT/uv-tools" UV_PYTHON_INSTALL_DIR="$OUT/uv-python"
export POETRY_CACHE_DIR="$OUT/poetry-cache" POETRY_VIRTUALENVS_PATH="$OUT/poetry-envs"
"$UVX_BIN" --python "$PYTHON_BIN" --from poetry==2.1.2 --with poetry-plugin-export==1.9.0 \
  poetry --directory "$SOURCE_ROOT/upstream" export --only main --format requirements.txt --output "$OUT/requirements.txt" \
  > "$OUT/dependency-export.log" 2>&1
verify_source
mkdir -m 700 "$OUT/context"
cp "$OUT/requirements.txt" "$OUT/context/requirements.txt"
cp "$DOCKERFILE" "$OUT/context/Dockerfile"
mkdir "$OUT/context/upstream"
for item in openhands third_party microagents pyproject.toml poetry.lock LICENSE README.md; do
  cp -a "$SOURCE_ROOT/upstream/$item" "$OUT/context/upstream/$item"
done
"${DOCKER[@]}" pull --platform linux/amd64 python:3.12-slim | tee "$OUT/base-pull.log"
BASE=$("${DOCKER[@]}" image inspect --format '{{index .RepoDigests 0}}' python:3.12-slim)
[[ "$BASE" == *@sha256:* ]] || { echo 'Missing immutable base image identity' >&2; exit 2; }
printf '%s\n' "$BASE" > "$OUT/base-image.txt"
"${DOCKER[@]}" build --platform linux/amd64 --progress plain --build-arg "BASE_IMAGE=$BASE" --iidfile "$OUT/image-id.txt" \
  --tag "$IMAGE_TAG" "$OUT/context" 2>&1 | tee "$OUT/build.log"
IMAGE=$(head -n 1 "$OUT/image-id.txt")
[[ "$IMAGE" == sha256:* ]] || { echo 'Missing immutable result image ID' >&2; exit 2; }
"${DOCKER[@]}" image inspect "$IMAGE" > "$OUT/image-inspect.json"
"${DOCKER[@]}" run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 128 --memory 4g --cpus 2 "$IMAGE" python -m pip freeze > "$OUT/pip-freeze.txt"
verify_source
"$PYTHON_BIN" - "$OUT" "$SOURCE_MANIFEST_SHA256" "$IMAGE_TAG" <<'PY'
from pathlib import Path
import hashlib,json,os,sys
root=Path(sys.argv[1])
files=['requirements.txt','context/Dockerfile','base-image.txt','image-id.txt','image-inspect.json','pip-freeze.txt']
m={'schema':'emender-openhands-runtime-image-candidate-v1','status':'built-not-execution-qualified',
   'source_manifest_sha256':sys.argv[2],'image_id':(root/'image-id.txt').read_text().strip(),'image_tag':sys.argv[3],
   'base_image':(root/'base-image.txt').read_text().strip(),'execution_qualified':False,'training_eligible':False,
   'historical_dataset_runtime_identity_verified':False,'automatic_retry':False,
   'docker_daemon':'unix:///var/run/docker.sock','files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
with (root/'manifest.json').open('x') as f:
    json.dump(m,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('OPENHANDS_RUNTIME_IMAGE_BUILT '+m['image_id'],flush=True)
PY
