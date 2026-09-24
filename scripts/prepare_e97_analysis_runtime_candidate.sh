#!/usr/bin/env bash
# Isolated evaluation-only source refresh; never modifies training or admission authorities.
set -euo pipefail
: "${OUTPUT_ROOT:?new absolute candidate directory required}"
SOURCE_TAR=/mnt/nvme2n1/erikg/e97_systematic_posttraining/training_source_identity_systematic_v1/source-tree.tar
SOURCE_SHA=57b2ee253da5120155e528210647fd476516c7610418e3d42774d3f217b41dc4
PYTHON=/home/erikg/emender/.venv/bin/python
[[ "$OUTPUT_ROOT" == /* && ! -e "$OUTPUT_ROOT" ]] || exit 64
[[ $(sha256sum "$SOURCE_TAR" | awk '{print $1}') == "$SOURCE_SHA" ]] || exit 65
mkdir -m 700 "$OUTPUT_ROOT"
mkdir -m 700 "$OUTPUT_ROOT/worktree"
tar -xf "$SOURCE_TAR" -C "$OUTPUT_ROOT/worktree"
cd "$OUTPUT_ROOT/worktree"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path
from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_first_party_source_archive import (
    EXPECTED_GENERATOR_COMPONENT_PATHS, build_source_archive, verify_source_archive,
)
root = Path.cwd()
manifest = root / 'configs/pi/e97-firstparty-generator-manifest-v1.json'
archive = root / 'configs/pi/e97-firstparty-source-v1.tar'
before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (manifest, archive)}
components = [{'path': name, 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()}
              for name in sorted(EXPECTED_GENERATOR_COMPONENT_PATHS)]
# Only the new isolated copy is changed. Canonical registries/allowlists stay untouched.
manifest.write_text(json.dumps({'schema': 'emender-e97-firstparty-generator-manifest-v3',
                                'components': components}, sort_keys=True, separators=(',', ':')) + '\n')
digest = build_source_archive(manifest, archive, checkout_root=root)
assert verify_source_archive(manifest, archive, checkout_root=root) == digest
from scripts.serve_e97_agent_openai import _archived_controller_source_sha256
closure = _archived_controller_source_sha256()
receipt = {
    'schema': 'emender-e97-analysis-runtime-candidate-v1',
    'status': 'source-binding-verified', 'training_eligible': False,
    'production_admission': False,
    'purpose': 'evaluation-only candidate; tests and CUDA qualification still required',
    'parent_source_archive_sha256': '57b2ee253da5120155e528210647fd476516c7610418e3d42774d3f217b41dc4',
    'previous_files': before,
    'replacement_files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (manifest, archive)},
    'verified_loaded_controller_closure_sha256': closure,
}
publish_bytes_no_replace(root.parent / 'source-binding.json',
                         (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode(), mode=0o600)
print(json.dumps(receipt, sort_keys=True))
PY
"$PYTHON" -m pytest -q \
 tests/test_serve_e97_agent_openai_identity.py \
 tests/test_e97_first_party_source_snapshots.py \
 tests/test_e97_agent_protocol.py \
 tests/test_e97_agent_server.py \
 tests/test_e97_moe_agent_server.py \
 tests/test_e97_acquisition_controller.py \
 tests/test_e97_onpolicy_records.py \
 tests/test_e97_open_swe_private_analysis_sft.py \
 --junitxml="$OUTPUT_ROOT/runtime-tests.xml" > "$OUTPUT_ROOT/runtime-tests.log" 2>&1
printf 'ANALYSIS_RUNTIME_CPU_TESTS_PASSED root=%s\n' "$OUTPUT_ROOT"
