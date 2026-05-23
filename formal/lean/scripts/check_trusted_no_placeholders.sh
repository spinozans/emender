#!/usr/bin/env bash
set -euo pipefail

root="${1:-ElmanProofs.lean}"

if [[ ! -f "$root" ]]; then
  echo "trusted check: root file not found: $root" >&2
  exit 2
fi

repo_root="$(pwd)"
root_abs="$(realpath "$root")"

mapfile -t sources < <(
  {
    printf '%s\n' "$root_abs"
    lake env lean --src-deps "$root"
  } |
    awk -v prefix="$repo_root/" 'index($0, prefix) == 1 { print }' |
    sort -u
)

if [[ "${#sources[@]}" -eq 0 ]]; then
  echo "trusted check: no project sources found for $root" >&2
  exit 2
fi

failed=0

if rg -n '\b(sorry|admit)\b' "${sources[@]}"; then
  echo "trusted check failed: placeholder proof terms found" >&2
  failed=1
fi

if rg -n '^\s*(unsafe\s+)?axiom\b|^\s*opaque\b' "${sources[@]}"; then
  echo "trusted check failed: explicit axiom/opaque declarations found" >&2
  failed=1
fi

if [[ "$failed" -ne 0 ]]; then
  exit "$failed"
fi

echo "trusted check passed: ${#sources[@]} project source files"
