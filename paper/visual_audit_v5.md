# Visual audit v5 — NDM paper

Audit of `paper/Garrison_2026_NDM-c5156240.pdf` (29 pages, ≈1.0 MB) prior to the
visual redesign that targets the NIH R01 grant's typography
(`/home/erikg/pgwas_R01/delivery/20260512/pgwas_r01_complete.pdf`).

## 1. Current PDF — issues page-by-page

### Globally
- **Body font:** DejaVu Sans 10.5 pt (set explicitly in `paper/main.typ:19,23`).
  The font register is already sans-serif but the family is the Bitstream Vera /
  DejaVu register, which is wider and a touch heavier than the Helvetica /
  Arial register the author asked for. DejaVu Sans is what is installed
  hermetically on the build host (`typst fonts` lists only `DejaVu Sans` and
  `DejaVu Sans Mono` in the sans family).
- **Margins:** 1.0 in x / 1.0 in y (`paper/main.typ:12`). Author wants slightly
  tighter; the grant uses 0.5 in but that's an NIH constraint, not a target.
- **Paragraph leading:** 0.62 em with 0.85 em paragraph spacing
  (`paper/main.typ:24`). Reasonable.

### Page 1 — title, abstract
- Title typography is fine (17 pt bold + tracking).
- Throughout the abstract: model names and defined terms written as `*…*`
  render as **bold**, not *italic*: "**Pure Nonlinear Recurrent Language
  Models** (PNR)", "**NDM**", "**M²RNN-CMA**", "**Mamba2**",
  "**Gated DeltaNet**", "**one-step representational**". This is the
  italic-as-bold complaint.

### Page 2 — body, defined terms
- Pervasive bold-where-italic-was-intended: "**nonlinear**-state counterpart",
  "**NDM**", "**delta-correcting**", "**M²RNN-CMA**", "**raw-write**",
  "**architecture class**", "**multi-programming**", "**Nonlinear Delta Memory**".
  Bold for inline term definitions reads heavily and is the dominant visual
  defect of the current PDF.

### Page 6 — inline section labels
- "**Matrix state**" and "**The $S_5$ state-tracking probe**" act as inline
  sub-headings. These ARE intended bold (label-style). They live in the source
  as headings (`#heading(level: 2, numbering: none)`), so they are governed by
  the Level-2 heading show rule, not by `*…*` markup — they will remain bold
  after the planned strong→emph override.

### Page 7 — figure 1
- Caption begins with bold `*NDM architecture and multi-programmed shape.*`
  which renders as bold (as intended for a figure caption label).
- Inside the figure, the panel subtitle `*A. Per-head update step*` renders
  bold (as intended).
- The Lean witness annotation `(Lean: ndm_1p27B_programs_per_batch_token_bs5)`
  is wrapped in `text(style: "italic")` and DOES render as italic — confirming
  that DejaVu Sans **has a working italic face** on this machine.

### Page 11 — body with bold + italic emphasis
- "**per-family CMA-ES winner shape**", "**delta-correcting**",
  "**M²RNN-paper**", "**diverged**", "**parameter-efficiency corollary**",
  "**strictly more state capacity**", "**strictly more fixed weights**",
  "**follows from**", "**not**" — all rendered bold via `*…*`.
- Most of these reads more naturally as italic emphasis (term definition,
  contrast emphasis); a few (e.g. "**not**") are genuinely bold-style stress.
- Mixed bold + italic intent inside the same paragraph is one cause of
  visual noise.

### Page 21 — theorem-set bulleted labels
- Bulleted labels `- *Mechanism separation.*`, `- *One-step resource separation.*`,
  `- *Positive embedding.*` are `*…*` lead-ins followed by a continuation
  sentence. The grant uses the same pattern (`*Our team.*` …) and renders bold
  there.
- After the strong→emph override, these become italic lead-ins. This is
  acceptable academic style and matches what the author wrote (italic
  emphasis) more faithfully than bold does.

### Page 26 — late-paper synthesis
- "**comparison relation**" rendered bold-where-italic-intended; same pattern
  as elsewhere.

### Page 28 — references
- Bibliography is generated; italic journal/proceedings names render as
  italic correctly via the citation style. No issue here.

## 2. Reference template (NIH R01)

Source: `/home/erikg/pgwas_R01/grant.typ` (imports
`template/nih-r01.typ`).

### Font
```
set text(font: ("Liberation Sans", "Nimbus Sans"), size: 11pt, lang: "en")
```
- Liberation Sans is the metric-compatible free Arial replacement (Red Hat,
  Linux default). Nimbus Sans is the metric-compatible Helvetica replacement.
- **Neither is installed on the current NDM build host** (`fc-list | grep -iE
  "liberation|nimbus"` returns nothing; `typst fonts` lists only DejaVu Sans
  and DejaVu Sans Mono in the sans family). The grant template comment is
  explicit: "Arial/Helvetica intentionally omitted to keep builds warning-free
  on systems without Microsoft fonts."
- Practical fallback chain for the NDM build: ask Typst for
  `("Liberation Sans", "Nimbus Sans", "Helvetica", "Arial", "Inter",
  "DejaVu Sans")` — Typst falls back left-to-right and uses whichever is
  installed. On this host that resolves to DejaVu Sans (no warning).

### Size and leading
```
set par(leading: 3.5pt, spacing: 4.5pt, justify: true, first-line-indent: 0pt)
```
- 11 pt body, 3.5 pt leading (≈ 14.5 pt baseline pitch at 11 pt body), 4.5 pt
  paragraph spacing, no first-line indent. Justified.
- NDM is currently 10.5 pt with 0.62 em leading (≈ 6.5 pt absolute) and
  0.85 em paragraph spacing (≈ 8.9 pt) — looser leading and more paragraph
  whitespace than the grant. Pulling NDM toward 11 pt body and tighter
  paragraph spacing brings it closer to the grant's denser feel.

### Margins
```
set page(margin: (top: 0.5in, bottom: 0.5in, left: 0.5in, right: 0.5in))
```
- 0.5 in on all four sides — an NIH constraint. The NDM paper is not subject
  to that constraint, so we slot in 0.75 in (slightly smaller than the
  current 1 in but not as tight as the grant), per the author's "slightly
  smaller" cue.

### Heading hierarchy
- Level 1: 11 pt bold (no v-space before), 4 pt v-space after.
- Level 2: 6 pt v-space before, 11 pt bold, 3 pt v-space after.
- Level 3: 4 pt v-space before, 11 pt **bold italic**, 2 pt v-space after.
- `set heading(numbering: none)` — grant headings are unnumbered.
- NDM currently has 12.5 pt Level 1 bold, 11 pt Level 2 bold, and uses
  `numbering: "1.1"`. We keep the numbering (academic paper convention) and
  the slight size step on Level 1 but tighten the v-spaces and add a Level 3
  bold-italic rule for consistency with the grant.

### Italic / bold / strong
- The grant cleanly distinguishes:
  - `*Term.*` markup → bold (label-style; e.g. `*Our team.*`)
  - `_term_` markup → italic (e.g. `_HLA_`, `_CYP2D6_`, `_A. thaliana_`)
- The grant relies on Typst's default `show strong` (bold) and `show emph`
  (italic) — no overrides.

### Code blocks, math, tables
- Math: Typst default (no explicit math font in the grant template).
- Code: no explicit raw font override in the grant; tables use bold via
  `*Header*` in `table.header`.

## 3. Root cause: italics rendering as bold

**Hypothesis in task description:** font lacks an italic face, Typst
falls back to bold.

**Confirmed root cause: this hypothesis is wrong.** DejaVu Sans (the current
body font) *has* italic and bold-italic faces:

```
$ typst fonts --variants | grep -A 10 "DejaVu Sans$"
DejaVu Sans
- Style: Italic, Weight: 400, …
- Style: Italic, Weight: 700, …
- Style: Normal, Weight: 400, …
- Style: Normal, Weight: 700, …
…
```

Where italics are written via `text(style: "italic")` (e.g. the Lean witness
annotation in figure 1 on page 7) or via `_…_` markup, they render as proper
italics in the current PDF.

The actual failure mode is **markup syntax, not the font**: `paper/main.typ`
uses `*…*` for inline emphasis throughout. In Typst markup, `*…*` is the
**strong** form (bold) — the analogue of Markdown's `**…**`. The italic
analogue is `_…_`. The author wrote `*…*` intending italic and got bold,
because that is what `*…*` means in Typst.

The author asked for a **template-level** fix and explicitly forbade prose
edits ("This is purely a `#set` / `#show` / font-family pass on
`paper/main.typ`"). The cleanest such fix is

```typst
#show strong: it => emph(it.body)
```

which reinterprets every `*…*` as italic at render time. Side effects:
- Figure-caption labels, table headers, bulleted lead-ins written as `*…*`
  also become italic. This is consistent with the author's intent ("they
  should all be italic"), valid academic style, and matches what the grant
  achieves with deliberate `_…_` markup. Bold visual hierarchy is preserved
  by the heading show rules and by `text(weight: "bold", …)` directives used
  for the title and the "ABSTRACT" label.
- Body content is unchanged; only the visual interpretation of `*…*` flips
  from bold to italic.

## 4. Plan

1. Adopt the grant font fallback chain
   `("Liberation Sans", "Nimbus Sans", "Helvetica", "Arial", "Inter", "DejaVu Sans")`
   — resolves to DejaVu Sans on this host (warning-free), but documents the
   intent and renders correctly on any host with Helvetica/Arial/Liberation
   Sans installed.
2. Margins: 0.75 in all sides (slightly smaller than current 1 in; not as
   tight as the grant's NIH-mandated 0.5 in).
3. Add `#show strong: it => emph(it.body)` — fixes the italic-as-bold
   complaint at the template level without touching any prose.
4. Tighten leading and paragraph spacing toward the grant's denser feel
   (`leading: 0.55em`, `spacing: 0.7em`); keep the body at 10.5 pt to avoid
   reflow of equations and figures, which were tuned for that size.
5. Heading hierarchy: keep numbering (paper convention) but shrink the
   Level-1 size step (12.5 pt → 12 pt) and tighten v-spaces to match the
   grant's denser headings.
6. Math: leave New Computer Modern Math (good pairing with sans body; the
   grant does not specify a math font).

No prose, no content, no figures, no tables touched.
