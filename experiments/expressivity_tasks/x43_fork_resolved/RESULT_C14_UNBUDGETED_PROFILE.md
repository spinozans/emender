# C14-B unbudgeted occupancy profile: kernel boundary reached

**STOP at the production-kernel decision boundary.** A wrapper-only replay
optimization is byte-exact and reduces replay work, but total compute speed
did not improve in these samples. The unbudgeted configuration has not been
made optimal, and budget-pressure causality remains unresolved. No twin was
trained and no frozen result is replaced.

## Measured work, leased physical card 3

The real first document-local 16x128 batch uses the read-only trained unbudgeted
checkpoint, original cap 257 and all operations. Its per-head peak live counts
are 21/53/14/44; calling this a 110-live batch would be false. A separate
synthetic probe reaches exactly 128 live slots in each row, with zero capacity
merges. Zero stream values force adjacent residual zero and hence an append
each position under the existing law. It is an input intervention, not new
training or a claim about trained high-occupancy loss.

| Measurement | Original slices | Token unbind |
|---|---:|---:|
| Real-batch replay + differentiation, synchronized ms/step | 49.914 | 42.466 |
| High-occupancy replay median ms/head | 13.635 | 11.725 |
| High-occupancy native median ms/head | 37.964 | 38.025 |
| Real-batch compute steps/s, mean duration statistic | 2.685948 | 2.597539 |

The real-batch samples are 20 reference / 50 candidate repeated fixed-weight
steps. They include disposable optimizer updates with weights restored outside
timing; source/decode/assembly and checkpoint serialization are excluded.
They are not sustained training rates or an apples-to-apples variance estimate.
Total throughput regressed 3.29% by this statistic; no session speed win is claimed.
Historical training remains 2.644903 steps/s, not overwritten by these profiles.
The 128-live controlled probe uses 30 samples for each implementation.

The removable wrapper work is repeated full-window slice-scatter backward,
already fixed in the budgeted replay but still present in the separate control
replay. Unbinding its seven time-indexed inputs once preserves all transition
arithmetic, gradient accumulation ordering and operation/ownership semantics.
It removes about 7.45 ms of synchronized real-batch replay work and about 14%
of synthetic high-occupancy replay time. Slot transitions were already batched
across rows and configured slots; no unbatched per-live-slot loop was found.

The dominant measured cost is the unchanged native kernel: the real-batch GPU
profile records eight launches, **285.892 ms / 84.86% of self CUDA time**,
including four checkpoint recomputations. Candidate wrapper timing separately
measures about 148.103 ms initial native work and 144.344 ms native recomputation
per step. Synchronizing diagnostic categories perturb execution and must not
be added to uninstrumented throughput as exact additive attribution.

The launch allocates **S=257, BS=512**, not 110. Read-only compiled metadata
reports **n_regs=40, n_spills=4078, shared=16384 bytes** for both head policies.
The 32/64 budgeted launch tiles are 32/64 slots. Thus the physical native slot
axis expands 8x/16x when moving to cap 257, including padded lanes; the replay
axis expands to 257. The source retains full-axis reductions and K/V state
regardless of live masks. Register spilling and allocation padding are concrete
engineering evidence consistent with the cost, not a measured decomposition
of spill latency or a proven causal scaling exponent. Comparing live peaks
alone (roughly 2.3x) cannot label the observed 5x slowdown intrinsic.

Per-operation native read, CORRECT, APPEND, POP, DECAY, MERGE and ownership
costs remain fused. A truthful finer native breakdown or eliminating those
full-tile costs requires production-kernel work. It was not performed. Neither
shrinking the cap (which changes occupancy-dependent policy) nor changing
checkpoint recomputation policy is a semantic-preserving substitute here.

## Identity, failures and boundary

Real-batch complete fingerprints compare original to candidate exactly: scalar
loss, native reads/final physical K/V/strength/bookkeeping, operation traces,
replay outputs/gradients, model gradients, None-versus-zero and telemetry.
Both runs repeat exactly. The independent 128-live probe also repeats and
matches reference/candidate fingerprints exactly, including physical state and
gradients. Native-versus-replay tolerance remains 2e-5; that is a separate gate.
No production kernel or checkpoint was changed. Single-batch fingerprints do
not prove byte identity for every input, device or compiler version.

Failures retained: initial profile expected an uncompressed training.json,
while the harvest retained training.json.gz; it exited before checkpoint load.
The corrected profiler uses the immutable HARVEST checkpoint hash. Initial
battery collection failed because an explicitly named stacking test bypassed
--ignore and uses package-relative imports. The rerun uses importlib collection
and includes all C14 tests plus operator-round tests. No failure is counted as a
passing gate. The profiler source changed during the original baseline process
only to add metadata observation for the subsequent candidate and remove an
unused split-capture function. Consequently baseline composition.json's
profiler_sha256 observes the later on-disk source, not the originally loaded
profiler. baseline_profiler_source.py reconstructs that loaded diagnostic;
production/control source hashes and actual raw fingerprints are unaffected.
This provenance limitation is retained rather than rewriting original evidence.

X31 bill, **planning estimate, not certification**: at minimum run the 109-test
parity/gradient/state/journal battery, registered cost/append/storage/structural
and checkpoint gates, then the complete C14 battery and strict fingerprints.
Historical X31 large-store H16/H1 was failing; it must remain a failure until
fixed and remeasured. Prior estimate: 0.5–2 GPU-hours plus 1–3 engineering days
for re-certification and diagnosis, implementation additional and wall time
unmeasured. No intrinsic ceiling or fresh X31 certificate is claimed. The new
native-spill evidence motivates kernel investigation only beyond this boundary.

## Triangle and exact retained state

The conditional “speed lands” gate did not pass; the named matched-exposure
unbudgeted twin remains NOT RUN. Its target is exactly 71,677,042 real byte
targets / 35,419 steps on the identical stream, compared to frozen budgeted
B1-v2 with ops on and nonbinding cap 257. At the measured candidate compute
rate, a full run would take about 13636 compute seconds (5.05x 2700),
not the hoped-for roughly 2x standard budget. A continuation from 7,142 steps
would require about 10886 more compute seconds, plus pipeline and evaluation.
Do not quietly amend the existing arm's 2700-second stopping rule or use its
old hash-bound launch gates with changed source. A new matched-exposure runner
and gates are still required before launch. Full paired evaluation previously
cost about 52,111 seconds; no extrapolated NLL is substituted for that work.

Exact original resume state, independently hash-verified before profiling:
`/home/erikg/c14_dense_bytes/emender_c14_b1_v2_unbudgeted_resume.pt`,
SHA256 `056b876cec5f5aae0b9bcea24544e9e3efccc2791fe9bb91654ca192cfb3eb06`,
7,142 steps, 14,455,380 targets, next window 114,272, active seconds
2700.2882562745363. Optimizer and RNG remain in that untouched checkpoint.
The registration/runner is not falsely marked executable or completed.

Single-seed limits apply to all quality/data-efficiency interpretation. This
profile does not resolve budget pressure against capacity, exposure or
trajectory. The trained-heavy-compute interpretation remains operational time
per position, not measured useful serial reasoning depth.

Manual notes: contact_supervisor/intercom is absent from the available tool
catalog, so no supervisor message or approval is claimed. Work was attended,
single-threaded, using only the specified Python and broker card 3. No WG
service/dispatcher/worker, union service, frozen arm, card-1 search, sxgc or
/mnt/nvme2n1 was operated on; no .pt file was committed. No resilient DiLoCo
behavior changed; gap-matrix requirement IDs are inapplicable. Existing dirty
files outside this task were preserved. See the raw receipts and validation
manifest for exact evidence and retained failures.
