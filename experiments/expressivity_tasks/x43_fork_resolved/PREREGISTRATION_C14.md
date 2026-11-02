# C14 preregistration — Pile training, Wikipedia consultation

Registered after phase-1 coverage.json and validation.json, before training or
held-out inference. Train the attested pile24 GPT-2 BPE stream, consult only the
immutable Wikipedia index. This is no longer a guaranteed disjoint/no-copy task.
The coverage receipt shows 16/64/256/1024-byte support rates
16: 36.5900%, 64: 2.3600%, 256: 0.5900%, 1024: 0.0900%; 256-byte span overlap
0.5000%; maximal-context next-byte agreement 69.3333%.
These training-interior diagnostics calibrate predictions, not held-out outcomes.
All five centered-span matches are whitespace only; non-whitespace supported
contexts are 3264/9605, 118/9882, 23/9962 and 2/9990 at those lengths.
Rare substantive long overlap exists (including one 1024-byte statistics table),
but no measurable Wikipedia document-component fraction follows. Expect any
aggregate gain to be driven mainly by short-context lexical consultation, not
broad long-span copying; score non-whitespace strata separately.

P1 (directional, evaluated separately for each head): Wikipedia-conditioned
inputs reduce raw held-out full-vocabulary token NLL relative to the matched
dense control. No numeric gain extrapolated from C12: long-context support is
sparse and overlap/preprocessing differ. Predict the larger gain on supported,
next-byte-agreeing strata than zero-target strata; do not drop any targets.
P2: all-input feature dropout (independent Bernoulli keep=.5, no rescaling,
including length metadata) preserves device-off NLL within 0.02 nats/token of
its matched dense control. P3: calibration plus consultation improves final-test
NLL relative to both consultation alone and dense plus calibration. These are
falsifiable predictions, not accepted improvements. Report all signs and losses.

Use velocity 150M trainable matched-head candidates (Emender and transformer),
GPT-2 vocabulary 50,257, 2700 active training seconds per arm. Within each head,
run dense zero-input control and Wikipedia consultation with identical data order,
initialization seed 3401, batch/context, optimizer and trainable parameter count
(tolerance 0.1%, including input projection). Retain a matched zero-input projection
in the control. Keep dense cross-entropy over every token, no fork-only writes,
no target masks, hard routing, or self-containing Pile index. Use only preceding
text at actual training reset boundaries; no cross-document left context. Register
exact model configs, counts, data/cache hashes and effective steps/tokens before
launch. Report both active time and end-to-end feature cost and the off-mode loss.
No new hyperparameter search or held-out selection is authorized by this ledger.

Use a causal maximal-supported-Wikipedia context selector capped at 1024 bytes;
this deliberately replaces C12's transductive selector. Token-boundary summary
geometry is the intended bridge, not exact 50,257-token support. The phase-1
255-byte geometry is provisional: before training, freeze how separator/EOS,
empty contexts, reset boundaries and normalization are handled, check a complete
training/held-out sidecar, and receipt throughput/storage/transfer. The current
CLI lacks an efficient persistent Python API and produced two impossible UINT64_MAX
extension counts. Two feature rows were rejected; no distribution is fabricated.
Require upstream repair and independent count/branch regression evidence before training. Fail closed until this prerequisite
is met; no large training was run in phase 1.

Stacking (second half): freeze trained weights; split the held-out evaluation
stream into disjoint calibration and final-test halves before inference. Fit the
C13-style confidence/context/support gate only on calibration data; freeze it
before final test. Compare dense neural, consultation neural, dense+calibration,
and consultation+calibration, with enabled and all-input-zero consultation rows.
Exact-byte distributions are not full BPE distributions: forced next byte cannot
force a BPE token. Permit only validated normalized full-token continuation for
forced-token walks, otherwise report that C13 component unavailable and use a
separately frozen first-byte-feature calibration. Do not silently substitute a
byte probability for token probability. Report total token NLL and comparable
byte-normalized NLL, denominators, context-length/support/zero-target strata,
feature dependence, steps/tokens, extraction cost and failures. No causal
operator-only interpretation or universal unplugged robustness from one seed.
