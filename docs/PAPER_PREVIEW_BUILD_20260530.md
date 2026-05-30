# Paper Preview Build - 2026-05-30

Task: `build-paper-preview`

## Preview Artifact

- Local build output: `paper/Garrison_2026_Emender-fcad67f2.pdf`
- Local preview copy: `paper/Garrison_2026_Emender-figure-label-gdn2-preview-20260530T162352Z.pdf`
- Hypervolume preview URL:
  `http://hypervolu.me/~erik/ndm/Garrison_2026_Emender-figure-label-gdn2-preview-20260530T162352Z.pdf`
- SHA-256:
  `2863f1785da7381e171b0b0d051d1e2924e366858706583156775c17cb927555`

## Validation

- `bash paper/build.sh` completed successfully and wrote
  `paper/Garrison_2026_Emender-fcad67f2.pdf`.
- The preview copy was uploaded to Hypervolume under a timestamped non-stable
  filename; HTTP HEAD on the preview URL returned `200 OK` with
  `Content-Type: application/pdf` and `Content-Length: 852189`.
- Remote inspection showed the stable public
  `~/www/ndm/Garrison_2026_Emender.pdf` remained a separate file from
  2026-05-29 and was not overwritten.
- Page 13 of the generated PDF was rendered with PyMuPDF for visual checking.
  The Figure 2 panel includes right-edge final-window BPB labels:
  `M2RNN-CMA: 0.983`, `Emender: 0.977`, and `GDN: 0.970`.
- PDF text extraction found the GDN-2 ongoing-work note on page 29. In source,
  it lands in `paper/main.typ` under "A partial order on PNR update rules",
  lines 1920-1923, as an ongoing-work caveat for GDN-2-style split erase/write
  variants.

No arXiv update, public source push, stable Hypervolume overwrite, or Hugging
Face update was performed.
