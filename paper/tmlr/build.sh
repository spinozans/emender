#!/usr/bin/env bash
# Rebuild the standalone packaging tree, without touching canonical Typst files.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${LATEX_ENGINE:-}" == tectonic ]] || command -v "${TECTONIC_BIN:-tectonic}" >/dev/null 2>&1; then
  for variant in submission preprint; do
    "${TECTONIC_BIN:-tectonic}" --keep-intermediates --keep-logs "$variant.tex"
  done
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error submission.tex preprint.tex
else
  echo 'Install Tectonic or a TeX Live distribution with latexmk.' >&2
  exit 1
fi
for variant in submission preprint; do
  if grep -Ei 'Missing character|Undefined control sequence|undefined references|Citation .* undefined|Reference .* undefined|Overfull|Float too large|^!' "$variant.log"; then
    echo "Unclean $variant build; inspect $variant.log" >&2
    exit 1
  fi
  [[ -s "$variant.pdf" && -s "$variant.bbl" ]]
done
cmp refs.bib ../refs.bib
printf 'Both variants built; no missing glyphs, unresolved references, overflows, or errors.\n'
