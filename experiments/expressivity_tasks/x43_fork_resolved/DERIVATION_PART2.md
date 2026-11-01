# X43 part 2: frozen exact-count fusion

The operator's part-2 request authorizes follow-up A, superseding the original
registration's execution stop. Its scalar calibration protocol is retained;
the requested grid and feature gate are additional arms. Follow-up B remains
conditional on session budget. This binding precedes count/gate execution.

Use exactly the direct-SA maximal repeated left context and occurrence interval
from DERIVATION_PART1C.md; no shortened-context backoff. For each target i,
count c_b = |{j in O(C(i,L_i)): T[j]=b}|. The train arm restricts j<split
(and j>=L_i already holds by interval construction). The separate corpus arm
uses all occurrences including the target and future held-out bytes, and is
an explicitly transductive oracle diagnostic. Labels and query context lengths
also come from the full corpus in both arms, so even the train-count arm is
not a causal deployment evaluation.

Normalize q_b=c_b/sum(c). If sum(c)=0, abstain: use p_model exactly,
regardless of the gate. Nonempty support with zero target count is distinct
from empty support: q_target=0 and g=1 gives infinite loss. Do not smooth,
clip probabilities, drop zero-count targets, or silently back off.
At forks p_fused=(1-g)*p_model+g*q, where g is index weight. Forced positions
retain model probabilities in interpolation-only arms. In *_walk variants,
covered forced positions use q exactly (q_target=1), hence zero loss;
uncovered forced positions retain the model. walk_oracle_with_model_abstention
uses exact counts at all covered positions. This is infinite on train-count
zero-target cases, not a universal attainable ceiling. The corpus oracle has
zero forced loss by construction.

Split the original held-out target array contiguously at floor(n_targets/2).
Use only the first half for fitting; report the second half as final test.
Grid: g in {0,.01,...,1}, minimizing calibration fork NLL, ties choose smallest.
Registered scalar: g=sigmoid(a), a initialized zero, optimize calibration fork
NLL on nonempty support (empty support contributes a constant).
Feature gate: sigmoid(w dot x), w initialized zero, intercept plus model maximum
softmax probability, log(1+context length), log(1+support size). Standardize
nonintercept features using covered calibration forks only. L-BFGS-B bounds
[-20,20] prevent divergent coefficients; no penalty or final-test selection.
Probability sigmoid input is bounded to [-35,35] for numerical stability.
All NLL uses natural logs. Paired delta is final-test fused mean minus frozen
model mean on exactly the same positions. Full softmax inference uses leased
GPU 2 or 4; target probability and maximum confidence suffice for NLL fitting.
The mixture's argmax accuracy is not reconstructed from target-only receipts.

Tests compare exact counts to independently enumerated overlapping occurrences,
exercise empty support and mixture endpoints, and retain the 511-string SA
exhaustive checks. Frozen model replay must match all retained accuracies and
per-target NLL within 2e-4. Checkpoint and corpus SHA-256 are retained.
