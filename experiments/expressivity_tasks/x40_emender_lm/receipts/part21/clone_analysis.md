# Backward clone population (retained sample)

Input: commit 4051a48f, part20/reduced_clone.txt.gz. `cluster.py` groups by
(source base, copy size, Python site), not by individual undefined subranges.
The input raw SHA256 is
802baafec500eacd8734b0001d402806e10e4029ddfd8ca716927ee97d6b7626.

The 100 reports contain four allocation groups:

| Copy site | Bytes | Reports | Attribution |
| --- | ---: | ---: | --- |
| X31 backward.py:237 | 512 | 14 | due, consistent with allocation/owner order |
| X31 backward.py:237 | 256 | 7 | terminal, unique scalar-float size among the undefined owners |
| X31 backward.py:237 | 512 | 14 | births, consistent with allocation/owner order |
| X31 backward.py:245 | 56832 | 65 | journal addresses, explicit copy site |

The two 512-byte labels are source-order inference; the old log does not
record owner names. Their shared class attribution is supported by the exact
three empty allocations in `cell_fused.py`, metadata indices 2/3/4 and replay
owner indices 8/9/10. This is a sample, not a complete sanitizer inventory.
The first bad owner offsets 64/32/64 correspond to row 8 after the eight-token
prefix; address-tail reports begin at byte 10144. No claims are made about
unprinted reports after the print cap.

## Candidate and rationale

`candidate.py` installs only process-local changes. Two forward mask sets are
identical to part20. X31, X36 and X40 due/terminal/birth allocations are
initialized to zero. This targets three explicitly undefined owner fields,
not all buffers. Whole-owner copies occur in X31 backward, in retained-state
freeze in all three replay implementations, and in X36 replay_setup.prepare
(the non-payload branch loads every owner element). Initialization covers each
of those sites without altering their copy arithmetic or layout.

Final-live masking of owner copies is insufficient: reverse undo must restore
historically live rows, and freeze precedes a consuming continuation. A dead
row at the final clock may have participated earlier. Zero initialization
retains every historical defined value; the old uninitialized bits were not
part of the public state contract. The differential backward fixture varies
only these allocation sentinels (zero, finite, NaN terminal), retaining forward
outputs through consuming continuation and differentiating through all chunks.
It compares defined state, trace, pressure, history and gradients as bytes.
Journal event counts themselves can change when the previously undefined old
scalar happens to equal its first store; they are private replay bookkeeping.

The address clone becomes a Triton counted copy. It loads only x<Counts[n],
writes the complete destination with zero elsewhere, then uses the unchanged
rebase and undo routines. Those routines already read only counted entries;
the tail is unconsumed. No suppressions, API-check disabling, barriers,
reductions, tolerances or race fixes are included.

## Additional sites and limits

The first candidate (6285bd43) left AllFieldJournal.snapshot's capacity-wide
address/before/width clones unchanged. The follow-up candidate uses the same
counted-copy helper for all three. Snapshot sanitization requires its own
terminal test evidence; the earlier reduced pass alone does not establish it.

The first counted copy used the existing (stores, capacity-blocks) grid.
The follow-up transposes those axes and tests 65537 blocks. At context>512,
the certification harness also validates and installs the existing part5 X31
rebase launch adapter. The original direct B128/T2048 X31 int32 reservation
guard remains authoritative. No large-shape waiver or promotion is licensed
by the reduced diagnostic.

The 18 surviving race groups are explicitly outside the uninitialized-access
promotion blocker set by operator instruction. They remain defects; there is
no benign-race finding, subtraction, or race-freedom claim. Race work and
payoff/depth claims remain pending full uninitialized-access promotion gates.
