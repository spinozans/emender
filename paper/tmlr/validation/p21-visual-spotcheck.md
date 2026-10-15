# P21 built-PDF visual spot check

Rendered with `/tmp/p9-poppler/usr/bin/pdftoppm -f PAGE -singlefile
-scale-to 1400 -png PDF /tmp/p21-...` (Frontier page at scale 1600), with
`LD_LIBRARY_PATH=/tmp/p9-poppler/usr/lib/x86_64-linux-gnu`.

Inspected rendered images during implementation:

- `submission.pdf`, p.1: anonymous title block; current abstract; legible text.
- `submission.pdf`, p.13: current Frontier accepted-token table, 73-merge
  duration/cadence figure and both captions; readable values and source paths;
  complete layout within page margins.
- `preprint.pdf`, p.1: Erik Garrison identity, affiliation/email and current
  abstract, with identified style.
- `preprint.pdf`, p.28: one Appendix A with four source categories, correctly
  scoped current-pair prose and legible source paths.

Result: PASS on these four spot-check pages. Physical references-before-appendix
and all-page text/symbol/anonymity checks have separate automated receipts.
An exhaustive visual page audit remains part of independent acceptance review.
