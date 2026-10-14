#!/usr/bin/env bash
set -euo pipefail

root="${1:-ElmanProofs.lean}"

if [[ ! -f "$root" ]]; then
  echo "trusted check: root file not found: $root" >&2
  exit 2
fi

source_list="$("$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/local_import_closure.sh" "$root")"
mapfile -t sources <<< "$source_list"

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
