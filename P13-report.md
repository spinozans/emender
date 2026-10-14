# P13 — language, notation/protocol audit, and durability inventory

## Part 1 — operator-directed language sweep

Four occurrences removed from the active manuscript; no method or quantitative
content changed. EXHIBIT/ANALOGY/OPEN already used neutral scope definitions
(main.typ:1987–1990 before Part 2); names retained. No `carefully` or `rigorously`
occurrences in the manuscript or its caption module. Historical review/outline
notes, data source quotations, and frozen PDFs are not manuscript variants
and are retained as provenance, not rewritten.

Every changed source line (old/new line numbers in diff headers):

```diff
diff --git a/paper/main.typ b/paper/main.typ
index 3d2956a1..4794e91b 100644
--- a/paper/main.typ
+++ b/paper/main.typ
@@ -1367 +1367 @@ the window-dependent sign of the $Delta$ in @tab_matched_loss is the
-honest summary of a tie at this precision, not evidence for either
+summary of a tie at this precision, not evidence for either
@@ -1385 +1385 @@ Within that band E97-MLP is ahead of GDN2-MLP on *both* primary metrics
-but the honest statement is descriptive closeness with the reported
+but the statement is descriptive closeness with the reported
@@ -1529 +1529 @@ surprising, and why the paper's positive claims live in #ref(<sec:expressivity>)
-#heading(level: 2, numbering: none)[Downstream capability panels: an honest null, on the predecessor cohort]
+#heading(level: 2, numbering: none)[Downstream capability panels: a null, on the predecessor cohort]
@@ -1534,2 +1534,2 @@ from the predecessor 1.3 B racer cohort (E88, FLA-GDN, M²RNN, Mamba2 —
-#ref(<sec:appendix_cohort>)), and it is reported here without spin as the honest
-expectation to beat: at $approx 1 M$-step training extents, on a
+#ref(<sec:appendix_cohort>)), and it provides the
+reference comparison: at $approx 1 M$-step training extents, on a
```

Supervisor approved moving the guard to the P13 source SHA-256, retaining
the eb2b9636 accepted source snapshot/history and frozen PDF. New source digest
and lineage: `paper/tmlr/canonical-baseline.json`. The rebuilt `paper/current.pdf`
is the verification authority; no old fragment mismatch was bypassed.
Part 1: current PDF 62 pages, both TMLR PDFs 53 pages; 20 samples × 3 PDFs,
2,499 source fragments, 643 math expressions, 417 attachments, three state
primes per PDF, 12/12 symbol regression tests passed. Bundle rebuilt.

