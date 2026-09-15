# Representation bridge: candidate preparation, not a new trained model

Following the [grounded expansion result](e97-grounded-expansion-v1.md), the
operator asked to continue. This step freezes candidate examples/evaluation and
qualifies a small authored native-executor slice. **No optimizer updates, model
sampling, production admission or checkpoint promotion are authorized by this
preparation recipe.** All completed learning budgets remain closed.

## Observed failure and design response

The previous run learned fresh trained formats (3/16→16/16), but prior cases fell
7/16→6/16 and structural variants reached only1/16. Further private trace inspection
confirms that **all four structural-edit cases added exactly1**. Only one required
that increment; the other three did not use the observed change. This is a concrete
instruction/observation-binding shortcut, not a demonstrated numerical fault.
The wrong-answer sums were neither the returned list nor the first-two-element
sum; no more specific arithmetic cause is established. The lost old lookup's
expected value was present in observations, but its20-character final was wrong.

The proposed curriculum uses three training representations per operation:

|Operation|Bridges|
|---|---|
|Lookup|Selected mapping entry; keyed row from an array; direct scalar field|
|Sum|Named pair; all list elements; designated numeric fields across rows|
|Edit|Explicit literal increment; observed top-level increment; observed nested increment|
|Recovery|Direct pointer; selected mapping of paths; selected row of paths|

Vary field names, source/destination basenames, JSON formatting and instruction
framing. Counterfactual worlds share prompts and paths. Mapping/row lookups change
the selector while retaining the value table. List sums change a later operand.
Observed edits retain the starting state while changing the requested increment;
magnitudes vary from2 through100 with both signs, never+1. Literal increments
remain represented and explicitly ignore a decoy change field. Recovery changes
observed routing, not the task prompt.

New edit teachers execute a semantic assertion against the actual source and
requested increment **after reading back the output**. This is a real tool call
and observation, not fabricated verifier prose. Host-side expected output and
source-byte checks remain separate. Old teacher behavior is unchanged unless the
new optional recipe fields are present; previous immutable source snapshots are
untouched. Failed authored prefixes remain unsupervised correction context.

## Frozen preparation scope

-768 candidate training fixtures (192 per operation), not yet a training authority.
-32 new fresh-layout evaluation cases and16 nested/composed evaluation cases.
-Existing48 evaluation cases stay excluded from training and remain regression/
development diagnostics, not independent held-out claims for newly taught layouts.
-48 **separate-seed, training-only preflight** trajectories, balanced over families,
three layouts, two naming/system-prompt styles and compact/pretty JSON.
-One native preflight≤600s, two sequential owned networkless/nonroot sandboxes,
no model/GPU and no retries. Source inventories checked before/after.

Checks exclude exact answer overlap with the checked previous evaluation and
last authored-training cases, enforce disjoint new cohort paths, and exclude
preflight/evaluation overlap. These are bounded provenance checks, not a claim
of universal corpus-wide or repository-level independence.

The first candidate freeze used only+3/-7 for observed increments. It was retained
at `representation-bridge-v1-preparation` and superseded **before any native
execution or model evaluation**, because two fixed magnitudes would invite another
shortcut. No prior artifact or measured gate was changed. Revised candidate cases
are frozen in `representation-bridge-v1-preparation-r2`:

|Artifact|SHA256|
|---|---|
|Plan|`4fe5dfe085f207d078a382fb9c727f0bfe8ab6188dd7bdff4a36e7d258d60166`|
|Fresh evaluation|`f60763a157f051161cfc2078640e019adf1a754fc49a9ae93a4364e96dd452e3`|
|Composition evaluation|`0741f2e05393f54df27e34e9b10e4e10f33646dc9550aa5d297bb32ff44c3126`|

**71 CPU tests passed**: real private-tempdir file/subprocess teacher execution,
independent oracle/mask reconstruction, source preservation, semantic edit checks,
row-order-independent pointer selection, rejection of ambiguous pointers, and
magnitude diversity within each naming/layout combination. Native-container
qualification is the next preparation step; neither test success nor teacher
success counts as autonomous model improvement.

Implementation:
-`scripts/e97_representation_bridge.py`
-`scripts/prepare_e97_representation_bridge.py`
-Optional typed pointer/delta and semantic-verification extensions in
 `scripts/e97_grounded_curriculum.py`
-`configs/pi/e97-representation-bridge-preparation-v1.json`

## Proposed learning follow-up—not launched

If preparation passes, specify a separate bounded SFT plan from exact expansion
live-y (`17aa2672…`, explicit train weight mode), using the exercised BF16-SR trainer.
A proposed32-update budget is a proposal, not authorization in this preparation.
Include prior verified grounded **training** examples and unchanged admitted replay
so the new representations do not simply displace old behavior. Audit actual
unique/repeated exposures and choose explicit old/fresh/composition and retention
gates before any model evaluation or update. No numerical-policy integration,
precision sweep, outcome RL or reuse of evaluation trajectories is part of this work.
