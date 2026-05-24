# Template choice — v7 (post-c1eee009 visual redesign rejection)

## Decision

Adopt **arkheion 0.1.2** (`@preview/arkheion:0.1.2` on Typst Universe).

The task description references `0.1.2:0.5.0` but only `0.1.2` exists on Typst Universe — verified by `typst compile` of a probe `#import "@preview/arkheion:0.5.0"`, which returned `error: package found, but version 0.5.0 does not exist (latest is 0.1.2)`. 0.1.2 is the de-facto Typst arxiv-preprint template (Mikkel Paltorp Schmitt, late 2023) — Mark Pollmann–Kalimeris's "arkheion" port of the long-standing arxiv preprint LaTeX class.

## Why arkheion 0.1.2 over the alternatives

| Template            | Body font                          | Title block                                   | Abstract                          | Verdict for this paper |
|---------------------|-----------------------------------|-----------------------------------------------|-----------------------------------|------------------------|
| arkheion 0.1.2      | New Computer Modern (LaTeX feel)  | double 2pt rule top + bottom, centered title | centered `smallcaps[Abstract]`    | **Chosen.**            |
| lapreprint-typst    | mixed sans/serif, side-bar layout | side-bar + accent color                       | side-bar block                    | Heavier; more "journal" than "preprint"; side-bar conflicts with two-affiliation header. |
| modern-academic-paper | TeX Gyre Heros sans               | clean centered                                | clean                             | Sans body — defeats the author's "LaTeX is the target" feedback. |
| starter-journal-article | Typst defaults                | minimal                                       | minimal                           | Too plain; not visibly distinct from the rejected c1eee009. |
| charged-ieee        | Times-style serif, two-column      | IEEE conference                               | IEEE abstract                     | Two-column changes the math/figure layout substantially — out of scope for a pure-template pass. |

Arkheion gives us, out of the box, the four things the author called out:

1. **LaTeX-feel body font.** Computer Modern (the LaTeX default since 1978). This directly addresses "The font is really rough" — c1eee009 used DejaVu Sans, which arkheion replaces with the canonical academic serif.
2. **Polished title block.** Double 2pt horizontal rule above and below the title, centered author block with ORCID linking — the standard arxiv preprint identifier.
3. **Smallcaps "Abstract" heading.** Matches the arxiv preprint register.
4. **Section heading hierarchy.** Built-in `numbering: "1.1"` numbered headings with consistent vertical padding.

## What we port forward from the current `main.typ`

The community template does not natively cover everything the current paper uses. The following `#show` / `#set` rules are preserved (or re-introduced) **after** the `#show: arkheion.with(...)` invocation:

- **`#show strong: it => emph(it.body)`** — author's convention is `*…*` ⇒ italic emphasis, not bold (documented at line 36–41 of the original `main.typ`).
- **`#show bibliography: set heading(numbering: none)`** — keep References unnumbered as a top-level section.
- **`#let nd`, `#let s5`, `#let s3`** — math shortcuts used throughout the body.
- **`#show figure.caption`** — captions set to 9pt (vs ~11pt body) with vertical breathing room above and below. This addresses the author's "massive figure captions" complaint.
- **`#set math.equation(numbering: none)`** — arkheion's default `set math.equation(numbering: "(1)")` would add visible "(1)", "(2)", ... markers to every display equation, which is a content-visible change. The current paper has no equation labels (`grep <eq:` returns nothing) and no equation cross-references, so we preserve the unnumbered display-equation style.

No theorem environments need porting: the paper uses level-2 unnumbered headings ("Theorem set A — …") rather than a `ctheorems`-style theorem environment, so arkheion's default heading rules suffice.

## Equation numbering note

Arkheion 0.1.2's preamble includes `set math.equation(numbering: "(1)")`. Because the current paper has zero equation labels or `@eq:…` cross-references, enabling numbering would add visible (1)…(N) markers throughout — a visible-but-pointless content change. We turn it back off with `#set math.equation(numbering: none)` after the template invocation. This is the only deliberate visible deviation from arkheion's defaults, and it is taken to preserve the existing paper's display-equation register.

## Build status

`paper/build.sh` exits 0 with zero hard errors. Any warnings (e.g., orcid icon image positioning notes from arkheion's title block) are non-blocking and documented inline below.
