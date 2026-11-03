# X43 — Positioning: the exact decision structure as the spine of a language model

The related-work skeleton for Paper 2. Every claim here needs the full-claim
verification pass (the X32 comparables discipline: retrieve the actual paper,
check the claim against its text, quote the binding passages) before the
paper cites it. This document is the map, not the verified citations yet.
Status: sweep-level (Nov 2026); neighbors identified by search + abstract-level
reading only.

## The claim being positioned

The exact combinatorial structure of a text corpus — its forced positions and
forks, the decision space over sequences — has been used to count, to draft,
and to constrain; it has not been used as the deterministic spine of a
language model whose neural component is redefined, measured, and chosen for
its role at the decision points. This program does that, with receipts:

1. The fork/forced decomposition as the organizing principle of an LM
   (C1-C2: derived, cross-validated two independent ways, 1,115,394/1,115,394;
   loss concentration at forks: 57% of positions carry 66-82% of NLL).
2. The systematic integration lattice — the training-coupling laws, each cell
   receipted with a named mechanism: loss-masking starves representations
   (C6); learned gates route around the head (C7a); fixed residual blends
   distill only the memory-less head (C7b); self-covering inputs collapse
   into the copy (C11); consultation against a non-covering index is the one
   coupling that works, with the gain flowing live through the array and
   nothing absorbed into weights (C12, the feature-off decomposition).
3. The head-choice question with an asymmetry finding: a memory-architecture
   head consults without copying at every coverage tested; a weight-based
   head re-copies as coverage rises (C13).
4. The swappable-array property: skills in weights, corpus in the replaceable
   index (C12-C13: unplug the features and the gain vanishes — nothing
   memorized).
5. The stacked system — consultation plus fusion — as the tier-1 record
   (C13: emender 1.650, beating its own fusion baseline and the transformer's
   best), with generation-character receipts (C9: walk/fork alternation,
   suffix-backoff re-entry, the novelty tables).

## The neighbors, and what each one lacks

| Neighbor | What it does | What it lacks (vs. our claim) |
|---|---|---|
| **infini-gram** (Liu et al. 2024, arXiv:2401.17377) | Exact unbounded-n n-gram counts over 5T tokens via suffix arrays; studies interpolation with neural LMs | The fusion cell alone: no fork/forced decomposition, no restructured model role, no head-choice question, no training lattice. Validates the exact-index premise at scale. |
| **REST** (He et al., NAACL 2024) + retrieval-speculative line (DReSD) | Suffix-array datastore proposes draft tokens; LM verifies by rejection sampling | The walk's mechanism, used purely as a speed accelerator that preserves the LM's distribution exactly. The index never enters the modeling hypothesis. |
| **kNN-LM** (Khandelwal et al. 2020) | Interpolation with a learned-key (k,v) datastore | Fuzzy, learned keys bolted on; no combinatorial structure; never analyzed for copy dynamics, coverage scaling, or swappability. |
| **Memorizing Transformers** (Wu et al. 2022) | Runtime kNN attention into an external KV memory | The nearest "device interface" precedent: approximate external memory, no exact automaton, no swappability analysis, no training-coupling laws. |
| **RETRO / SUREALM** (Borgeaud et al. 2022; suffix-retrieval LMs) | Retrieved passages/suffixes as conditioning context | Fuzzy retrieval as context augmentation; not a decomposition of the modeling problem; no structural index. |
| **Constrained decoding** (Outlines; automata-based constraints 2407.08103; trie automata) | An external automaton masks the softmax and emits forced structure | The walk's mechanism for *compliance* (grammar conformance), not capability: nobody asks what forced spans mean for the model's job or its loss. |
| **Suffixient arrays** (Cenzato-Depuydt-Gagie-Kim-Manzini-Olivares-Prezza 2024, arXiv:2407.18753; linear-time follow-ups, CPM 2026) | The chi formalism: the smallest sufficient set of decision positions; two-sided bounds | Indexing theory. No connection to language modeling anywhere in the line: fork density, decision-space framing, chi as an LM's theoretical spine — this connection exists only in this program (and the lower bound gives the whole architecture an information-theoretic floor). |
| **Mamba/SSM hybrids** (Jamba; hybrid memory-recall line) | Attention+SSM layer mixing | Memory architectures without any exact index. The memory-architecture × exact-index intersection is unoccupied (the X32 empty-quadrant finding, extended). |

## The unoccupied intersection, stated plainly

Nobody (a) uses the fork/forced decomposition as the organizing principle of
a language model; (b) holds the systematic integration lattice with named
mechanisms per cell; (c) asks which architecture class should be the fork
head — with the memory-based-vs-weight-based asymmetry receipted; (d)
demonstrates the swappable-array property by unplugging the features and
showing the gain flows live. Each neighbor misses a different essential
piece.

## Why this was sitting in the intersection of our own work

The program built the tool that measures the decision structure (xsa / chi),
the architecture whose certified property — OOD generalization at the
boundary — is precisely the fork role's qualification (EmenderLM, Papers 1
and the certified X28E-X40 receipts), and the fusion receipt that connected
them (C3: the crossover). No single result in the neighbors' list had a
reason to exist without all three in hand.

## Standing caveat

Sweep-level identification only. Before the paper cites any neighbor: full
retrieval, claim verification against the actual text, and passage-level
citations per the X32 comparables discipline (see
x32_comparables/ and x41_unified_report/RELATED.md). The empty cell is real;
the exact boundaries of each neighbor's territory are not yet verified.
