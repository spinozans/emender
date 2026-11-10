# C14 in-region consultation: source-occurrence leakage (found 2026-11-10)

**Trigger:** `~/sxgc/NOTE_TO_EMENDER_CONTEXT_DEPTH.md` — *"if the document is in the
training set, that is target leakage delivered as a feature. Source-document exclusion
must happen BEFORE any maximality or support test, not after."* We checked our own
code against that rule. It does not pass.

## The finding

`c14_branch_serve.py::select()` builds each consultation query as
`lefts[i].rsplit(b'\x1e',1)[-1][-1024:]` — the last ≤1024 bytes of the evaluated
position's context — and then, for the censored maximal suffix of length `k`, reads the
container's `width` and branch histogram **as-is**:

```python
row = cache[s[-k:]] if k else dict(width=0, eof=0, branches=[])
```

There is **no source-occurrence exclusion anywhere in the path**. The index used for the
C14 array is `/mnt/nvme2n1/erikg/wf-finish-as-100gb-20261102T044713Z/corpus.sxi2`,
built over the same C14 corpus from which the evaluated documents are drawn
(`c14_feature_extract.py`, `c14_coverage.py`, `c14_byte_register.py` all name it).

Therefore the histogram returned for an in-region evaluation position includes **that
position's own occurrence**, whose next byte is the target. For a censored maximal
context — the common case, since support collapses with depth (sxgc measured only 1.4 %
of positions have a 1024-byte context occurring elsewhere) — `width` is often exactly 1
and the histogram is `{target: 1}`. The feature is the answer.

## What this affects

| Claim | Status |
|---|---|
| **C14 array-alone "supported" 0.663434** | **optimistic** — largely the source occurrence read back. The doc already called array-alone "answer-conditioned"; this identifies the mechanism. |
| **C14 consulted arms 1.509260 (emender) / 1.266064 (transformer)** | **optimistic in-region**; the supported-target strata are contaminated by the same effect. |
| **The marginal-gain story ("the spine's value is transfer")** | **survives, and is now better explained**: in-region, the model has already memorized those documents, so the array's *marginal* value is low even with leakage; out-of-index, the model has not, so the array's marginal value is high (E3 survival 1.34–1.99×). Leakage inflates the array's *absolute* reading, not the transfer conclusion. |
| **The flagship and 150 M boards** | **unaffected by geometry**: the spine is `[0, 100 GB)` and every training *and* evaluation document lies beyond it, so no evaluation occurrence exists in the index. Repeated substrings from *other* documents are legitimate transfer, not leakage. |
| **Deployment** | source inclusion is *desirable* there — retrieving the corpus's own continuation is the feature. But it must be labelled as retrieval, never as held-out NLL. |

## Required follow-ups

1. **Annotate** `RESULT_C14_FIRST_PASS.md` and the C14 rungs bound to it: in-region
   consulted and array-alone numbers are optimistically biased by source leakage. Do not
   rewrite history; record the bias.
2. **Add source exclusion to the consult path** (`--exclude-source`): subtract the
   position's own occurrence(s) from `width` and the branch counts *before* the support
   test, so any in-region measurement is honest. This is a build-task on the live path.
3. **Never report in-region consulted NLL without source exclusion.** Out-of-region
   (beyond-spine) numbers need no exclusion by geometry, and remain the paper's clean
   evidence.
4. **Depth**: sxgc's companion measurement also shows the fixed 1024-byte spine is ~100×
   deeper than the data (1.4 % support at depth 1024; information concentrated in the
   first ~16 bytes). The registered feature definition should move to a variable-length
   spine (8–16 bytes) with support-weighted relax — a registration amendment, to be made
   before the 150 M consulted arms fire.
