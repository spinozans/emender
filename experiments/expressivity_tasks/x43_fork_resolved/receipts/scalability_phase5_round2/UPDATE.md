# First streaming probe completed

All ten native read/final-state/trace pairs and repeats pass. Matched-input
T1024 reference 718.757878 ms vs streaming 638.255253 ms, spills1524 (round1
6154). Long class improves but 1.3B/24h remains far out of reach. T32..128
regresses; a shape dispatch using reference through T128 is warranted only
once additional byte gates pass. Full candidate validation is running serially
on broker card5. Supervisor progress transport is unavailable.

T2048 did not produce a result after >15min of CPU work. The first own test
was stopped with143 under an agent-selected diagnostic budget (not an operator
ceiling). A separate180s retry with faulthandler timed out124. Its stack locates
the blocker in the unchanged reference's Triton make_llir -> LLVM O3 optimizer,
before candidate execution, not in replay or a proven native mismatch. This is
a failed/incomplete mandatory gate. Primitive checks for BS128..4096 all pass
24 random/cancellation/signed-zero/NaN read+selected-value byte pairs. A second
candidate removes full-context key-store temporaries and retains reference for
BS<=256; it is being measured separately and cannot be promoted without T2048.
