# Abstract — variant 3

On one workstation-class GPU, a 1.273-billion-parameter attention-free recurrent language model reaches 0.974 bits per byte on The Pile after roughly 23 days of training — no cluster, no time-axis scan trick. That model is E88, the 1.3 B-class instance of the Emender, a class of pure-nonlinear-in-time recurrent layers. This regime was thought out of reach for such recurrence on competitive wallclock; the route is multi-programming, a width-axis parallelism running ~22,200 small recurrent programs per token while each program's time loop stays serial. We present a controlled study: three 1.3 B-class architectures (E88, M²RNN-CMA, Gated DeltaNet) trained under matched per-architecture CMA-ES, where E88 holds the same loss-vs-wallclock band as Gated DeltaNet — parity, not a win. A Lean 4 trusted core derives an efficiency ordering within the class — the delta-correcting update reaches a strictly larger one-step function class than raw-write at matched FLOP — and the ordering is confirmed on an 8 M-parameter S₅ probe (0.79 vs 0.22).

---

**Angle:** Lead with the felt impossibility (sub-1-bpb, one GPU, no scan trick), then pivot from single result to *study* — name the class, the controlled three-way race at honest parity, and the proved-then-confirmed efficiency ordering — so the existence-proof character lands through structure rather than assertion.
