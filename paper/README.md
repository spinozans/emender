# NDM Paper

## Format

**This paper is written in [Typst](https://typst.app), NOT LaTeX.**

The source is `paper/main.typ`. Do not create `.tex` files.

## Building

```bash
bash paper/build.sh
```

This produces `paper/Garrison_2026_NDM-<commit8>.pdf` (or `...-dirty.pdf` if the
working tree has uncommitted changes). The script can be run from the repo root
or from inside `paper/` — both work. The PDF is gitignored (it is a build
artifact; do not commit it).

Typst must be installed:

```bash
# Install via cargo
cargo install --locked typst-cli

# Or check if already available
typst --version
```

## Source layout

```
paper/
  main.typ          # Master source (all sections)
  refs.bib          # BibTeX bibliography (Typst reads BibTeX natively)
  build.sh          # Build script: typst compile → Garrison_2026_NDM-<commit>.pdf
  figures/          # Figure files (SVG, PNG, PDF); placeholder SVGs now
  results/          # Numerical results staged for figures (created per-figure)
  OUTLINE.md        # Section-by-section paper outline (reference, not the draft)
  ndmpapernotes.md  # Original paper notes (reference, not the draft)
  notes_reconciliation.md  # Claim-by-claim evidence reconciliation (reference)
```

## Adding a new figure

1. Put the figure file (SVG or PNG preferred) in `paper/figures/`.
2. Reference it in `main.typ`:
   ```typst
   #figure(
     image("figures/my_figure.svg", width: 80%),
     caption: [Caption text.],
   ) <fig_label>
   ```
3. Cite it in the text with `@fig_label`.
4. Run `bash paper/build.sh` to verify it renders.

## Adding a new citation

1. Add a BibTeX entry to `paper/refs.bib` with a unique key.
2. Use `#cite(<key>)` in `main.typ` (angle-bracket syntax, not backslash).
3. Typst resolves all `#bibliography("refs.bib")` citations at compile time.

## Pending figure data

Figure 3 (1.27B language-model loss racers) needs artifacts from `~/elman/`.
Stage frozen checkpoint loss curves into `paper/results/figure_3/` and
replace `figures/figure_3_placeholder.svg` with the real plot.

See `paper/OUTLINE.md` §5.2 for the full list of pending experimental closures.

## Internal references

- E88 / E-series = internal codenames for NDM architecture variants (do not
  use these in paper prose — the paper targets outside readers).
- The `docs/` tree contains design dossier, expressivity results, and
  related-work survey used to draft each section.
- `formal/lean/` contains the trusted Lean 4 proof core.
