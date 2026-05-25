#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMMIT=$(git -C "$SCRIPT_DIR" rev-parse --short=8 HEAD)

if ! git -C "$SCRIPT_DIR" diff --quiet || ! git -C "$SCRIPT_DIR" diff --cached --quiet; then
  COMMIT="${COMMIT}-dirty"
fi

OUTPUT="Garrison_2026_PNR-${COMMIT}.pdf"

typst compile main.typ "$OUTPUT"

echo "$SCRIPT_DIR/$OUTPUT"
