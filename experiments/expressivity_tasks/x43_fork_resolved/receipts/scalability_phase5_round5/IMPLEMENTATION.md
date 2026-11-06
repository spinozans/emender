# Register tile implementation checkpoint

Design a8fb1ac5 committed/pushed before candidate implementation. Tile32x128
products occupies16 product registers per unique thread before overhead, not a
context-sized tensor. Reshape/permute + an explicit register layout expose16
pair tensors through split; each is converted to the original pair layout.
Append them to parity carries in ordinal order0..15. Initial pair REPLACES the
placeholder accumulator, so there is no added zero. No tile subtotal or softmax
rescaling. All pre/post/control/score/normalize arithmetic imported from R3/R4.

First compilation failed on Gluon SliceLayout vs BlockedLayout addition; retain
tile_attempt1.py, reduction_console.txt and exit1. Explicit layout conversion
fixes the type mismatch without arithmetic. reduction_fixed:24/24 independent
READ/previous pairs pass, including BS4096 cancellation/NaN/signed zeros.
Quick T32 monolithic read/state/trace and repeat match. Matched R4 T1024:
255.924206 ->191.888684ms/head, five samples each, complete fingerprints/repeat
exact; READ42 registers/0 compiler spills, post255/26. This is a diagnostic
speed gain, still hundreds of times slower than the necessary .502691ms bar.

Wrapper temporarily replaces the imported READ function only inside try/finally;
this opt-in diagnostic assumes serial attended execution. It is not thread safe
and must not be installed in production. Full ten-shape/compositional/regression
battery remains to run. No production selector or sparse variant added.
