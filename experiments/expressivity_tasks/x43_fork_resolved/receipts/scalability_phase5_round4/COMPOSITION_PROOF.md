# T2048 certification by composition, not unchanged monolithic compilation

This uses the operator's explicit alternative: T1024 monolithic anchors plus a
carry-composition argument. The unchanged T2048 reference is not repaired and
is not claimed to compile. Preserve the R2/R3 exit124 evidence.

## State and scheduling induction

A token runs pre-control -> score -> normalization -> ordered READ pages ->
post-control, all on the current PyTorch CUDA stream. Each launch finishes its
writes before the next launch can consume them. Initial state and dedicated
scratch are allocated afresh for each native call/document. R4 imports the
exact R3 JIT initialization, pre/post control, score, normalization and READ
functions. There is no duplicate arithmetic implementation to drift.

Relative to R3 the ONLY semantic change is the number and boundaries of READ
launches. READ writes carry and output only: it cannot mutate keys, values,
strength or metadata. Pre/post control's 128-row value-update loop retains R2's
bounded value tile and the original values+(input-values) expression, and
selected-row previous-value lookup. Score keeps its original row-local64-wide
reduction. Normalization and scalar metadata still use the entire BS. Their
source and launch dimensions do not depend on READ page size.

Assume native persistent state agrees at token t entry. Identical pre-control,
score and normalization produce identical state and FP32 products. The READ
argument below gives identical output. Identical post-control then gives
identical physical state and raw trace at t+1. Initialization establishes the
base case. Induction covers the full sequence and all physical final fields,
including capacity exhaustion, expiry, pop and array/stream transitions, so
long as the shared control implementation agrees with the monolithic anchor.
The ten full-state/raw-trace monolithic pairs, eight operation-rich short state
pairs, and four short replay/gradient pairs test that latter obligation;
this is not a universal formal verification of the compiler or control.

## READ chain and page boundary induction

The fixed row reduction layout has two row warps, with column lanes, giving
two ascending row-parity chains before the final fixed_sum. The independent
primitive computes the original full matrix expression using fixed_sum, not
READ pages. For each column define p_i = round32(round32(w_i*h_i)*v_i).
The first pair initializes A0=p0 and A1=p1, WITHOUT adding zero. For subsequent
pair r, each Aj becomes round32(Aj+p_(2r+j)). R3 read_page performs exactly this
chain. It never forms a page subtotal or changes parenthesization. Fusion is
disabled in both arms.

Before any page after the first, carry[0:2,0:128] contains exactly the two
accumulators after the preceding page's final pair. A global FP32 store and
load retain those bits and perform no addition/conversion. Thus the next
page's start=0 resumes at exactly the next pair. All tested page sizes are
even powers of two. The guard excludes pairs beyond BS; it does not add extra
zero pairs for a page larger than BS. Only the last page calls the same
fixed_sum over the two parity rows. Induction on pages proves partition
independence, including signed-zero behavior and the absence of an extra
initial zero addition. NaN payload behavior is exercised independently.

The24 original matrix READ/previous primitive cases span BS128..4096, with
random, severe cancellation, signed-zero and NaN inputs at each BS, for EVERY
proposed page size (72 cases total; four BS4096 cases per page size). These compare int32 views,
including NaN bytes. They test the reduction-layout premise independently of
the full T2048 monolithic time loop that cannot compile. No approximate
comparison replaces any native byte fingerprint.

## Masking, replay and scope

For t>=valid, control makes no token state transition, READ stores exact zero
output, and normalization/score skip work. The long fixture has seven masked
tail positions. Re-running with a distinct document identity creates empty
state again. Full native read/state/raw-trace fingerprints, replay output and
all replay gradients are compared at T512/T1024 against the original reference;
T2048 compares PAGE1024 against the anchored PAGE256 decomposition. Graph replay
also asserts native numerical agreement and masked zeros. Replay code is shared
between arms, so its equality does not independently validate all derivatives.

The T2048 receipt is explicitly labelled compositional. It establishes this
changed page composition at the required S4097/BS4096. It does not certify all
operation-rich maximum-live long inputs, full trained models, optimizer/RNG/
cursor/checkpoint identity, or actual150M/1.3B training performance.
