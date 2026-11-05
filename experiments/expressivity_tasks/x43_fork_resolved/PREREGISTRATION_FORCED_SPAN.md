# X43 B: forced-span verbatim law

Registered before training. PB-L1: self-indexed training-corpus verbatim recall
is >=99% on forced spans and decreases with corpus branch entropy. All outcomes,
including refutation, empty bins, and failed runs, are results. C11 motivates
this diagnostic: RESULT_C11.md and receipts/c11/teacher.json describe exact
self-containing train-prefix distributions; C11's shortcut/OOD failures are
not evidence that near-perfect forced recall has already been demonstrated.

Use first 16 nontruncated train documents from the attested raw Pile offsets.
Match selected full-document hashes to documents.jsonl receipts. Natural UTF-8
bytes, vocab 256, DocumentWindows(128), right-pad and mask tails, reset state
per row. No packing or state carry across windows/documents. Every corpus byte
remains inside one document; no synthetic text or token-ID corpus.

Exact self index: next-byte counts for contexts of up to 4 preceding bytes,
constructed within each document, never across seams. At scoring, context is
truncated at the window boundary. Entropy H=-sum p log p from these counts.
Bins: forced (one branch), low (multiple branches and H<=log 2), high (H>log 2).
This deliberately tests a finite-context version of the law, not maximal
suffix intervals. Report counts and observed entropy in each bin.

Tier-1 emender head (~2M parameters), seed 4301, full 256-way dense CE,
zero consultation inputs so recall measures neural approximation in weights.
AdamW lr=3e-4 wd=.01 clip=1, batch16 context128, BF16 autocast, 180 active
training seconds (CUDA synchronized), sample windows uniformly with fixed RNG.
One seed, no tuning or early stopping by accuracy. Train on all selected docs;
this is training-corpus recall, not held-out generalization. Partition first8
and last8 documents as calibration/final-test recall panels, disjoint by doc;
training includes both. No claims of unseen-document recall.

Evaluate every training-corpus window after freezing. Report top1 byte accuracy,
NLL, and exact-match accuracy for maximal forced runs of >=4 targets entirely
inside a window, in both document panels. PB-L1 passes only if final-test
forced-run exact accuracy >=.99 and byte accuracy forced>=low>=high with all
bins populated. Distinguish forced byte vs whole-span recall. Report actual
steps, targets, exposure epochs, wall/active seconds, source/index hashes,
precision, window/boundary verification. Save no .pt into git. Card 5 or 6 via
broker. Single-threaded execution; no service, existing arm, or source mutation.
