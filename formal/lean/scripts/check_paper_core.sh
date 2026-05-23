#!/usr/bin/env bash
set -euo pipefail

root="${1:-ElmanProofs/PaperCore.lean}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

cd "$repo_root"

"$script_dir/check_trusted_no_placeholders.sh" "$root"

root_abs="$(realpath "$root")"
mapfile -t sources < <(
  {
    printf '%s\n' "$root_abs"
    lake env lean --src-deps "$root"
  } |
    awk -v prefix="$repo_root/" 'index($0, prefix) == 1 { print }' |
    sort -u
)

if rg -n '\bnative_decide\b' "${sources[@]}"; then
  echo "paper core check failed: native_decide found in trusted paper core" >&2
  exit 1
fi

echo "paper core check passed: ${#sources[@]} project source files, no native_decide"
