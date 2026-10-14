#!/usr/bin/env bash
# Enumerate the recursive project-local import closure, including the root.
# Lean --src-deps reports direct imports on the pinned toolchain; visit each
# local source explicitly. Compiler failures must abort, not shrink coverage.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lean_root="$(cd -- "$script_dir/.." && pwd)"
cd "$lean_root"
root="$(realpath "${1:-ElmanProofs/PaperCore.lean}")"
[[ -f "$root" && "$root" == "$lean_root/"* ]] || {
  echo "import closure: missing or non-local root: $root" >&2
  exit 2
}
declare -A visited=()
pending=("$root")
for ((i = 0; i < ${#pending[@]}; i++)); do
  source="${pending[i]}"
  [[ -z "${visited[$source]:-}" ]] || continue
  visited["$source"]=1
  deps="$(lake env lean --src-deps "$source")"
  while IFS= read -r dep; do
    # Lake dependencies live under .lake/ but are not project source files.
    [[ "$dep" == "$lean_root/"* && "$dep" != "$lean_root/.lake/"* ]] || continue
    [[ -f "$dep" ]] || { echo "import closure: missing source: $dep" >&2; exit 2; }
    pending+=("$(realpath "$dep")")
  done <<< "$deps"
done
printf '%s\n' "${!visited[@]}" | sort
