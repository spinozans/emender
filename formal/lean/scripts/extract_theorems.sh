#!/usr/bin/env bash
# Extract trust-gate theorems to structured JSON for the math-transcription
# render pipeline.
#
# What it does:
#   1. Resolves the trust-gate source-file list via
#        scripts/local_import_closure.sh ElmanProofs/PaperCore.lean
#      (the same enumeration used by `check_paper_core.sh`).
#   2. Runs `scripts/extract_theorems.lean`, which imports `PaperCore`,
#      walks the elaborated environment, and emits a JSON array
#      describing every declaration in the trust-gate import closure.
#   3. Wraps that array in a top-level object that pins:
#        - the current git commit hash
#        - the trust-gate source-file list
#        - the extraction timestamp (UTC ISO-8601)
#      and writes the final object to
#        paper/data/lean-trust-gate-theorems.json
#
# Usage:
#   ./scripts/extract_theorems.sh                            # default output path
#   OUTPUT=paper/data/foo.json ./scripts/extract_theorems.sh # custom output
#
# Must be run from `formal/lean/` (the directory containing this `scripts/`
# folder and `ElmanProofs/`).

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lean_root="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$lean_root/../.." && pwd)"

cd "$lean_root"

OUTPUT="${OUTPUT:-$repo_root/paper/data/lean-trust-gate-theorems.json}"
INNER_TMP="$(mktemp -t lean-extract.XXXXXX.json)"
trap 'rm -f "$INNER_TMP"' EXIT

mkdir -p "$(dirname -- "$OUTPUT")"

# 1. Enumerate trust-gate source files (project-local only).
source_list="$("$script_dir/local_import_closure.sh" "ElmanProofs/PaperCore.lean")"
mapfile -t sources <<< "$source_list"
for i in "${!sources[@]}"; do sources[i]="${sources[i]#"$lean_root/"}"; done

if [[ ${#sources[@]} -eq 0 ]]; then
  echo "extract_theorems: ERROR — empty trust-gate source list" >&2
  exit 1
fi

# 2. Run the Lean elaboration script, capturing the inner JSON array.
EXTRACT_OUTPUT="$INNER_TMP" \
EXTRACT_SRC_ROOT="$lean_root" \
  lake env lean scripts/extract_theorems.lean

# 3. Compose the final pinned JSON: commit + file list + timestamp + theorems.
commit_hash="$(cd "$repo_root" && git rev-parse HEAD)"
extracted_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

"${PYTHON_BIN:?Set PYTHON_BIN to the activated project interpreter}" - "$INNER_TMP" "$OUTPUT" "$commit_hash" "$extracted_at" "${sources[@]}" <<'PY'
import json
import sys

inner_path, out_path, commit, extracted_at, *trust_files = sys.argv[1:]
with open(inner_path) as f:
    theorems = json.load(f)

doc = {
    "commit": commit,
    "extracted_at": extracted_at,
    "trust_gate_files": sorted(trust_files),
    "theorem_count": len(theorems),
    "user_written_count": sum(1 for t in theorems if not t.get("auto_generated", False)),
    "schema_version": 1,
    "theorems": theorems,
}

with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")

print(f"extract_theorems: wrote {len(theorems)} declarations "
      f"({doc['user_written_count']} user-written) "
      f"across {len(trust_files)} trust-gate files to {out_path}")
PY
