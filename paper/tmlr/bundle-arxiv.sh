#!/usr/bin/env bash
# Exactly one documentclass/master in the arXiv bundle, explicitly identified.
set -euo pipefail
cd "$(dirname "$0")"
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT
printf '\\def\\EmenderPreprint{1}\n' > "$staging/main.tex"
cat main.tex >> "$staging/main.tex"
cp refs.bib tmlr.sty tmlr.bst fancyhdr.sty LICENSE STYLE-COMMIT "$staging/"
cp preprint.bbl "$staging/main.bbl"
cp -r assets "$staging/"
cat > "$staging/README.txt" <<'EOF'
Emending Nonlinear Recurrence: identified preprint packaging.
Compile main.tex with pdflatex + BibTeX (or Tectonic).
The preprint option is selected explicitly in main.tex.
Official TMLR style revision is recorded in STYLE-COMMIT; see LICENSE.
This archive does not contain the anonymous submission driver.
EOF
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf arxiv-source.tar.gz -C "$staging" .
echo 'Created arxiv-source.tar.gz (one identified main.tex master).'
