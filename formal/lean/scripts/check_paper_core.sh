#!/usr/bin/env bash
set -euo pipefail

root="${1:-ElmanProofs/PaperCore.lean}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

cd "$repo_root"

"$script_dir/check_trusted_no_placeholders.sh" "$root"

source_list="$("$script_dir/local_import_closure.sh" "$root")"
mapfile -t sources <<< "$source_list"

if rg -n '\bnative_decide\b' "${sources[@]}"; then
  echo "paper core check failed: native_decide found in trusted paper core" >&2
  exit 1
fi

echo "paper core check passed: ${#sources[@]} project source files, no native_decide"
